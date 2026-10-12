"""
Coordination network: who posts identical text at the same time?

Timestamps come from tweet-ID snowflakes (millisecond precision).
Two accounts co-post when they publish the same normalised text (>= 30
characters, URLs and mentions stripped, case folded) within WINDOW seconds.
Accounts are linked when they co-post at least MIN_CO times. This is the
co-tweet / co-retweet signal used in coordinated-inauthentic-behaviour research.

Outputs
  - share of each role's accounts inside the network
  - how often copies land within 60 s (synchrony, a sign of automation or one operator)
  - cross-role link matrix: are the personas siloed?
  - Louvain communities vs the Clemson role labels (NMI, ARI, purity)
"""
import json, os, re
from collections import Counter, defaultdict
from itertools import combinations
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from common import load_english, ROLES, RES

WINDOW = 3600          # seconds
MIN_CO = 3
MIN_LEN = 30
MAX_GROUP = 300        # skip boilerplate strings posted hundreds of times

URL = re.compile(r"https?://\S+")
USER = re.compile(r"@\w+")
NONW = re.compile(r"[^a-z0-9# ]+")
WS = re.compile(r"\s+")


def norm(t):
    t = USER.sub(" ", URL.sub(" ", str(t).lower()))
    return WS.sub(" ", NONW.sub(" ", t)).strip()


def co_posts(df):
    df = df.assign(key=df.content.map(norm), ts=df.utc.astype("int64") / 1e9)   # seconds, ms precision
    df = df[df.key.str.len() >= MIN_LEN]
    groups = df.groupby("key")
    pair_w, lags = Counter(), []
    for _, g in groups:
        if g.author.nunique() < 2 or len(g) > MAX_GROUP:
            continue
        g = g.sort_values("ts")
        a, t = g.author.to_numpy(), g.ts.to_numpy()
        seen = set()
        j0 = 0
        for i in range(len(g)):
            while t[i] - t[j0] > WINDOW:
                j0 += 1
            for j in range(j0, i):
                if a[i] != a[j]:
                    p = (a[i], a[j]) if a[i] < a[j] else (a[j], a[i])
                    if p not in seen:              # count each text once per pair
                        seen.add(p); pair_w[p] += 1; lags.append(t[i] - t[j])
    return pair_w, np.array(lags)


def run():
    df = load_english(["content", "retweet"])
    role = df.groupby("author").role.first()
    n_role = role.value_counts()
    pair_w, lags = co_posts(df)
    G = nx.Graph()
    for (a, b), w in pair_w.items():
        if w >= MIN_CO:
            G.add_edge(a, b, weight=w)
    nodes = list(G.nodes)
    in_net = pd.Series(nodes).map(role).value_counts()

    # cross-role link matrix (share of each role's link weight)
    M = pd.DataFrame(0.0, index=ROLES, columns=ROLES)
    for a, b, d in G.edges(data=True):
        M.loc[role[a], role[b]] += d["weight"]
        if role[a] != role[b]:
            M.loc[role[b], role[a]] += d["weight"]
    M_share = M.div(M.sum(axis=1).replace(0, np.nan), axis=0)
    within = float(np.trace(M.values) / M.values.sum()) if M.values.sum() else float("nan")

    comms = nx.community.louvain_communities(G, weight="weight", seed=0, resolution=1.0)
    lab = {n: i for i, c in enumerate(comms) for n in c}
    y_true = [role[n] for n in nodes]; y_comm = [lab[n] for n in nodes]
    purity = sum(Counter(role[n] for n in c).most_common(1)[0][1] for c in comms) / len(nodes)
    big = sorted(comms, key=len, reverse=True)[:8]
    comm_table = [dict(size=len(c), roles=dict(Counter(role[n] for n in c).most_common())) for c in big]

    # role-preserving null for within-role share: shuffle labels over network nodes
    rng = np.random.default_rng(0); null = []
    edges = [(a, b, d["weight"]) for a, b, d in G.edges(data=True)]
    lbl = np.array([role[n] for n in nodes]); idx = {n: i for i, n in enumerate(nodes)}
    for _ in range(1000):
        p = rng.permutation(lbl)
        same = sum(w for a, b, w in edges if p[idx[a]] == p[idx[b]])
        null.append(same / sum(w for _, _, w in edges))

    res = dict(window_s=WINDOW, min_co=MIN_CO, co_post_pairs=len(pair_w),
               network_nodes=G.number_of_nodes(), network_edges=G.number_of_edges(),
               share_of_role_in_network={r: float(in_net.get(r, 0) / n_role[r]) for r in ROLES},
               copies_within_60s=float((lags <= 60).mean()), copies_within_10s=float((lags <= 10).mean()),
               median_lag_s=float(np.median(lags)),
               within_role_link_share=within, within_role_null_mean=float(np.mean(null)),
               within_role_null_95=[float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
               cross_role_share=M_share.round(4).fillna(0).to_dict(orient="index"),
               communities=len(comms), nmi_vs_roles=float(normalized_mutual_info_score(y_true, y_comm)),
               ari_vs_roles=float(adjusted_rand_score(y_true, y_comm)), purity=float(purity),
               largest_communities=comm_table)
    np.save(os.path.join(RES, "co_post_lags.npy"), lags)
    nx.write_weighted_edgelist(G, os.path.join(RES, "coordination_edges.tsv"), delimiter="\t")
    json.dump(res, open(os.path.join(RES, "network.json"), "w"), indent=2)
    return res


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
