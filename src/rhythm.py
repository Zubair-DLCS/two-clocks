"""
Rhythm forensics: whose clock does the operation run on?

Two natural experiments identify the operators' clock without any
authentic-user control group:

1. Daylight saving time. Russia abolished DST in 2014; the US did not.
   Around each US clock change, an account run on a US clock should shift its
   daily rhythm by 60 min in UTC (earlier in March, later in November). An
   account run on a Moscow clock should not move. Each account is compared with
   itself, 4 weeks before vs 4 weeks after. A placebo version runs the same test
   at dates with no clock change.

2. Holidays. Active accounts on Russian public holidays that are ordinary US
   workdays, vs on US federal holidays that are ordinary Russian workdays,
   each relative to the same weekday 1-2 weeks earlier/later. Significance
   comes from a null of randomly drawn ordinary weekdays.
"""
import json, os
import numpy as np
import pandas as pd
from common import load_english, ROLES, RES

BIN = 10                     # minutes
NB = 1440 // BIN
SHIFTS = np.arange(-12, 13)  # candidate shifts, in bins (+-120 min)

US_CHANGES = {"2015-11-01": +60, "2016-03-13": -60, "2016-11-06": +60,
              "2017-03-12": -60, "2017-11-05": +60}   # UTC shift a US clock would show

RU_HOLIDAYS = ["2015-11-04", "2016-01-04", "2016-01-05", "2016-01-06", "2016-01-07", "2016-01-08",
               "2016-02-22", "2016-02-23", "2016-03-07", "2016-03-08", "2016-05-02", "2016-05-03",
               "2016-05-09", "2016-06-13", "2016-11-04", "2017-01-03", "2017-01-04", "2017-01-05",
               "2017-01-06", "2017-02-23", "2017-02-24", "2017-03-08", "2017-05-01", "2017-05-08",
               "2017-05-09", "2017-06-12", "2017-11-06"]
US_HOLIDAYS = ["2015-07-03", "2015-09-07", "2015-11-26", "2015-12-25", "2016-01-18", "2016-02-15",
               "2016-05-30", "2016-07-04", "2016-09-05", "2016-11-24", "2016-12-26", "2017-01-16",
               "2017-02-20", "2017-05-29", "2017-07-04", "2017-09-04", "2017-11-23", "2017-12-25"]


# ---------------------------------------------------------------- descriptive

def describe(df):
    msk = df.utc.dt.tz_convert("Europe/Moscow")
    ny = df.utc.dt.tz_convert("America/New_York")
    out = {}
    for role in ["ALL"] + ROLES:
        m = np.ones(len(df), bool) if role == "ALL" else (df.role == role).to_numpy()
        h, wd, hny = msk[m].dt.hour, msk[m].dt.dayofweek, ny[m].dt.hour
        out[role] = dict(
            tweets=int(m.sum()), accounts=int(df.author[m].nunique()),
            hour_profile_msk=(np.bincount(h, minlength=24) / m.sum()).round(5).tolist(),
            share_msk_office_10_19=float(((h >= 10) & (h < 19)).mean()),
            share_msk_night_02_07=float(((h >= 2) & (h < 7)).mean()),
            share_us_prime_time_19_24_et=float((hny >= 19).mean()),
            share_weekend_msk=float((wd >= 5).mean()),
        )
    return out


def weekly_volume(df):
    w = df.groupby([df.utc.dt.tz_convert(None).dt.to_period("W").dt.start_time, "role"]).size()
    return w.unstack(fill_value=0)


# ---------------------------------------------------------------- DST test

def _smooth_logp(counts, sig=3):
    x = np.arange(NB)
    k = np.exp(-0.5 * (np.minimum(x, NB - x) / sig) ** 2); k /= k.sum()
    p = np.clip(np.real(np.fft.ifft(np.fft.fft(counts) * np.fft.fft(k))), 0, None)
    p = 0.9 * p / p.sum() + 0.1 / NB            # floor: robust to empty bins
    return np.log(p)


def _shift_test(df, date, us_shift, min_tweets=30, weeks=4):
    d = pd.Timestamp(date)
    day = df["day"]
    pre = df[(day >= d - pd.Timedelta(weeks=weeks)) & (day < d - pd.Timedelta(days=1))]
    post = df[(day > d + pd.Timedelta(days=1)) & (day <= d + pd.Timedelta(weeks=weeks))]
    cp = pre.groupby("author")["b"].apply(lambda v: np.bincount(v, minlength=NB))
    cq = post.groupby("author")["b"].apply(lambda v: np.bincount(v, minlength=NB))
    role = df.groupby("author")["role"].first()
    sgn = np.sign(us_shift); s_us = us_shift // BIN
    rows = []
    for a in cp.index.intersection(cq.index):
        if cp[a].sum() < min_tweets or cq[a].sum() < min_tweets:
            continue
        lp, q = _smooth_logp(cp[a].astype(float)), cq[a]
        ll = np.array([(q * np.roll(lp, k)).sum() for k in SHIFTS])
        best = SHIFTS[np.argmax(ll)] * BIN * sgn           # aligned: +60 = US clock
        llr = ((q * np.roll(lp, s_us)).sum() - (q * lp).sum()) / q.sum()
        rows.append(dict(date=str(d.date()), author=a, role=role[a], n_post=int(q.sum()),
                         aligned_shift_min=int(best), llr_us_vs_moscow=float(llr)))
    return rows


def dst_experiment(df, boots=2000, seed=0):
    df = df.copy()
    df["b"] = ((df.utc.dt.hour * 60 + df.utc.dt.minute) // BIN).astype(int)
    df["day"] = df.utc.dt.tz_convert(None).dt.normalize()
    real, placebo = [], []
    for date, sh in US_CHANGES.items():
        real += _shift_test(df, date, sh)
        for off in (-8, 8):    # placebo dates: no clock change, same alignment sign
            placebo += [dict(r, placebo_of=date) for r in
                        _shift_test(df, pd.Timestamp(date) + pd.Timedelta(weeks=off), sh)]
    real, placebo = pd.DataFrame(real), pd.DataFrame(placebo)
    rng = np.random.default_rng(seed)

    def summarise(t):
        if len(t) == 0:
            return None
        acc = t.author.unique()
        sh, us = t.aligned_shift_min.to_numpy(), (t.llr_us_vs_moscow > 0).to_numpy()
        groups = [np.flatnonzero(t.author.to_numpy() == a) for a in acc]
        med, frac = [], []
        for _ in range(boots):        # cluster bootstrap over accounts
            ix = np.concatenate([groups[i] for i in rng.integers(0, len(acc), len(acc))])
            med.append(np.median(sh[ix])); frac.append(us[ix].mean())
        return dict(account_transitions=int(len(t)), accounts=int(len(acc)),
                    median_shift_min=float(t.aligned_shift_min.median()),
                    median_shift_ci95=[float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))],
                    share_us_clock=float((t.llr_us_vs_moscow > 0).mean()),
                    share_us_clock_ci95=[float(np.percentile(frac, 2.5)), float(np.percentile(frac, 97.5))])

    out = {"real": {}, "placebo": {}, "real_excluding_election_week": {}}
    for role in ["ALL"] + ROLES:
        sel = (lambda t: t) if role == "ALL" else (lambda t, r=role: t[t.role == r])
        out["real"][role] = summarise(sel(real))
        out["placebo"][role] = summarise(sel(placebo))
        out["real_excluding_election_week"][role] = summarise(sel(real[real.date != "2016-11-06"]))
    real.to_csv(os.path.join(RES, "dst_account_shifts.csv"), index=False)
    placebo.to_csv(os.path.join(RES, "dst_placebo_shifts.csv"), index=False)
    return out


# ---------------------------------------------------------------- holiday test

def holiday_experiment(df_padded, draws=20000, seed=0):
    df = df_padded
    day = df.utc.dt.tz_convert("Europe/Moscow").dt.tz_localize(None).dt.normalize()
    idx = pd.date_range(day.min(), day.max())
    ru, us = pd.to_datetime(RU_HOLIDAYS), pd.to_datetime(US_HOLIDAYS)
    special = set(ru) | set(us)
    lo, hi = pd.Timestamp("2015-07-01"), pd.Timestamp("2017-12-31")
    ordinary = [d for d in idx if d.dayofweek < 5 and d not in special and lo <= d <= hi
                and not any(abs((d - x).days) <= 2 for x in special)]
    rng = np.random.default_rng(seed)
    out = {}
    for role in ["ALL"] + ROLES:
        m = np.ones(len(df), bool) if role == "ALL" else (df.role == role).to_numpy()
        act = df[m].groupby(day[m])["author"].nunique().reindex(idx, fill_value=0)

        def lr(d):   # log ratio vs median of same weekday +-1, +-2 weeks
            base = np.median([act.get(d + pd.Timedelta(weeks=k), np.nan) for k in (-2, -1, 1, 2)])
            return np.log((act.get(d, 0) + 1) / (base + 1))
        pool = np.array([lr(d) for d in ordinary])
        res = {}
        for name, days in (("russian", ru), ("us", us)):
            v = np.mean([lr(d) for d in days])
            null = np.array([rng.choice(pool, len(days), replace=False).mean() for _ in range(draws)])
            res[name] = dict(n_days=len(days), change_pct=float(100 * (np.exp(v) - 1)),
                             p_one_sided_drop=float((null <= v).mean()),
                             null_95_pct=[float(100 * (np.exp(np.percentile(null, 2.5)) - 1)),
                                          float(100 * (np.exp(np.percentile(null, 97.5)) - 1))])
        out[role] = res
    return out


def run():
    df = load_english()
    res = dict(describe=describe(df), dst=dst_experiment(df))
    res["holidays"] = holiday_experiment(load_english(core=False))
    weekly_volume(df).to_csv(os.path.join(RES, "weekly_volume_by_role.csv"))
    json.dump(res, open(os.path.join(RES, "rhythm.json"), "w"), indent=2)
    return res


if __name__ == "__main__":
    r = run()
    for k in ("real", "placebo", "real_excluding_election_week"):
        print(k); [print(f"  {role:13s}", v) for role, v in r["dst"][k].items()]
    print("holidays"); [print(f"  {role:13s}", v) for role, v in r["holidays"].items()]
    print("describe"); [print(f"  {role:13s}", {kk: round(vv, 3) for kk, vv in v.items() if kk != 'hour_profile_msk'})
                        for role, v in r["describe"].items()]
