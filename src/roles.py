"""
Role recovery: which signal reveals the persona a coordinated account plays?

Account-level features, in four families:
  Rhythm     Moscow-time hour-of-day and weekday shares, night share, burstiness
  Behaviour  retweet / link / mention / hashtag rates, tweet length, audience size
  GloVe      SIF-weighted pretrained GloVe-Twitter embedding of the account's tweets
  TF-IDF     word and bigram TF-IDF of the account's tweets

Experiments
  1. 5-fold x 2 stratified cross-validation, split by account (macro-F1).
  2. Drift: train on accounts' pre-election tweets (before 9 Nov 2016), test on
     held-out accounts' pre-election vs post-election tweets. A drop shows how far
     the operation's personas moved after the election.
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from common import load_english, ROLES, RES, ELECTION
from language import clean, glove_tokens, SIFEmbedder

MIN_TWEETS, MAX_TWEETS = 20, 400
R_COLS = [f"h{i}" for i in range(24)] + [f"wd{i}" for i in range(7)] + ["night", "burst", "log_gap"]
B_COLS = ["rt_rate", "url_rate", "mention_rate", "hashtag_rate", "len_mean",
          "log_followers", "log_following", "ff_ratio"]
G_COLS = [f"g{i}" for i in range(50)]


def sample_tweets(df, seed=0):
    df = df.sample(frac=1.0, random_state=seed)
    return df.groupby("author", sort=False).head(MAX_TWEETS)


def account_table(df, emb):
    """One row per account with all four feature families."""
    msk = df.utc.dt.tz_convert("Europe/Moscow")
    df = df.assign(h=msk.dt.hour, wd=msk.dt.dayofweek, ts=df.utc.astype("int64") // 10**9,
                   txt=df.content.fillna(""))
    rows = []
    for a, g in df.groupby("author", sort=False):
        n = len(g)
        h = np.bincount(g.h, minlength=24) / n
        wd = np.bincount(g.wd, minlength=7) / n
        gaps = np.diff(np.sort(g.ts.to_numpy())).astype(float)
        mu, sd = (gaps.mean(), gaps.std()) if len(gaps) > 1 else (0, 0)
        txt = g.txt
        r = dict(author=a, role=g.role.iloc[0], n=n,
                 **{f"h{i}": h[i] for i in range(24)}, **{f"wd{i}": wd[i] for i in range(7)},
                 night=h[2:7].sum(), burst=(sd - mu) / (sd + mu + 1e-9),
                 log_gap=np.log1p(np.median(gaps)) if len(gaps) else 0,
                 rt_rate=g.retweet.mean(), url_rate=txt.str.contains("http").mean(),
                 mention_rate=txt.str.count("@").mean(), hashtag_rate=txt.str.count("#").mean(),
                 len_mean=txt.str.len().mean(),
                 log_followers=np.log1p(g.followers.median()), log_following=np.log1p(g.following.median()),
                 ff_ratio=np.log1p(g.followers.median()) - np.log1p(g.following.median()),
                 doc=" ".join(clean(t) for t in txt))
        G = np.stack([emb.embed(glove_tokens(t)) for t in txt]).mean(0)
        r.update({f"g{i}": G[i] for i in range(50)})
        rows.append(r)
    return pd.DataFrame(rows)


def models():
    hgb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                 class_weight="balanced", random_state=0)
    tfidf = lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000,
                                    sublinear_tf=True, token_pattern=r"<\w+>|#?\w\w+")
    svc = lambda: LinearSVC(C=0.5, class_weight="balanced", max_iter=5000)
    sel = lambda cols: ColumnTransformer([("x", "passthrough", cols)])
    return {
        "Rhythm": make_pipeline(sel(R_COLS), hgb()),
        "Behaviour": make_pipeline(sel(B_COLS), hgb()),
        "Language: GloVe": make_pipeline(sel(G_COLS), hgb()),
        "Language: TF-IDF": make_pipeline(ColumnTransformer([("t", tfidf(), "doc")]), svc()),
        "All combined": make_pipeline(ColumnTransformer([("t", tfidf(), "doc"),
                                                          ("d", StandardScaler(), R_COLS + B_COLS + G_COLS)]),
                                      svc()),
    }


def cv_experiment(T):
    y = T.role.to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
    out, preds = {}, {}
    for name, m in models().items():
        f1s, yp = [], np.empty(len(y), dtype=object)
        for k, (tr, te) in enumerate(cv.split(T, y)):
            m.fit(T.iloc[tr], y[tr]); p = m.predict(T.iloc[te])
            f1s.append(f1_score(y[te], p, average="macro"))
            if k < 5:
                yp[te] = p                    # first repeat -> one prediction per account
        out[name] = dict(macro_f1=float(np.mean(f1s)), sd=float(np.std(f1s)),
                         per_class_f1=dict(zip(ROLES, f1_score(y, yp, average=None, labels=ROLES).round(3).tolist())))
        preds[name] = yp
        print(f"  {name:18s} macro-F1 {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    cm = confusion_matrix(y, preds["All combined"], labels=ROLES)
    return out, cm.tolist()


def top_terms(T, k=12):
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=100_000, sublinear_tf=True,
                         token_pattern=r"<\w+>|#?\w\w+")
    X = tf.fit_transform(T.doc)
    clf = LinearSVC(C=0.5, class_weight="balanced", max_iter=5000).fit(X, T.role)
    vocab = np.array(tf.get_feature_names_out())
    return {c: vocab[np.argsort(-clf.coef_[i])[:k]].tolist() for i, c in enumerate(clf.classes_)}


def drift_experiment(df, emb):
    cut = ELECTION.tz_localize("UTC") + pd.Timedelta(days=1)
    pre, post = df[df.utc < cut], df[df.utc >= cut]
    cnt_pre, cnt_post = pre.author.value_counts(), post.author.value_counts()
    keep = set(cnt_pre[cnt_pre >= MIN_TWEETS].index) & set(cnt_post[cnt_post >= MIN_TWEETS].index)
    Tpre = account_table(sample_tweets(pre[pre.author.isin(keep)], 1), emb).set_index("author")
    Tpost = account_table(sample_tweets(post[post.author.isin(keep)], 2), emb).set_index("author")
    Tpost = Tpost.loc[Tpre.index]
    counts = Tpre.role.value_counts()
    ok = counts[counts >= 10].index                 # roles with enough accounts in both periods
    Tpre, Tpost = Tpre[Tpre.role.isin(ok)], Tpost[Tpost.role.isin(ok)]
    y = Tpre.role.to_numpy()
    res = {"roles": sorted(ok.tolist()), "accounts": int(len(y)),
           "accounts_per_role": Tpre.role.value_counts().to_dict(), "families": {}}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for name, m in models().items():
        same, shifted = [], []
        for tr, te in cv.split(Tpre, y):
            m.fit(Tpre.iloc[tr], y[tr])
            same.append(f1_score(y[te], m.predict(Tpre.iloc[te]), average="macro"))
            shifted.append(f1_score(y[te], m.predict(Tpost.iloc[te]), average="macro"))
        res["families"][name] = dict(pre_to_pre=float(np.mean(same)), pre_to_post=float(np.mean(shifted)),
                                     drop=float(np.mean(same) - np.mean(shifted)))
        print(f"  {name:18s} pre→pre {np.mean(same):.3f}  pre→post {np.mean(shifted):.3f}")
    return res


def run():
    df = load_english(["content", "retweet", "followers", "following"])
    cnt = df.author.value_counts()
    df = df[df.author.isin(cnt[cnt >= MIN_TWEETS].index)]
    s = sample_tweets(df)
    print("embedding vocabulary ...")
    emb = SIFEmbedder([glove_tokens(t) for t in s.content.fillna("")])
    print(f"  GloVe token coverage {emb.coverage:.1%}")
    T = account_table(s, emb)
    print(f"{len(T)} accounts; role counts {T.role.value_counts().to_dict()}")
    print("cross-validation:")
    cv, cm = cv_experiment(T)
    print("drift across the election:")
    drift = drift_experiment(df, emb)
    res = dict(accounts=int(len(T)), accounts_per_role=T.role.value_counts().to_dict(),
               glove_coverage=emb.coverage, cv=cv, confusion_all=dict(labels=ROLES, matrix=cm),
               top_terms=top_terms(T), drift=drift)
    T.drop(columns=["doc"]).to_csv(os.path.join(RES, "account_features.csv"), index=False)
    json.dump(res, open(os.path.join(RES, "roles.json"), "w"), indent=2)
    return res


if __name__ == "__main__":
    r = run()
    print(json.dumps(r["top_terms"], indent=1))
