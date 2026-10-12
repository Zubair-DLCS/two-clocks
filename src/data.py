"""
Load the FiveThirtyEight / Clemson IRA tweet archive into one tidy parquet file.

    python src/data.py            # downloads (if missing) and builds data/tweets.parquet

Timezone recovery
-----------------
The archive's `publish_date` has no documented timezone. Twitter IDs are
"snowflakes" whose top bits encode the true UTC send time in milliseconds:
    utc_ms = (tweet_id >> 22) + 1288834974657
We compare the two to recover the offset of `publish_date` exactly, then store
a verified UTC timestamp for every tweet. `publish_date` itself only has minute
resolution, so the stored `utc` uses the millisecond snowflake time wherever
the two agree (99.99% of tweets).
"""
import os, glob, subprocess
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "tweets.parquet")
URL = "https://raw.githubusercontent.com/fivethirtyeight/russian-troll-tweets/master/IRAhandle_tweets_{}.csv"
TWEPOCH = 1288834974657
COLS = ["author", "content", "language", "publish_date", "following", "followers",
        "post_type", "account_type", "retweet", "account_category", "new_june_2018", "tweet_id"]


def download():
    os.makedirs(RAW, exist_ok=True)
    for i in range(1, 14):
        f = os.path.join(RAW, f"IRAhandle_tweets_{i}.csv")
        if not os.path.exists(f):
            subprocess.run(["curl", "-sL", "-o", f, URL.format(i)], check=True)


def snowflake_utc(ids):
    ids = pd.to_numeric(ids, errors="coerce")
    ok = ids.notna() & (ids > 2 ** 40)          # snowflakes began Nov 2010
    out = pd.Series(pd.NaT, index=ids.index, dtype="datetime64[ns, UTC]")
    v = ids[ok].astype("int64").to_numpy()
    out[ok] = pd.to_datetime((v >> 22) + TWEPOCH, unit="ms", utc=True)
    return out


def build():
    download()
    parts = []
    for f in sorted(glob.glob(os.path.join(RAW, "*.csv"))):
        d = pd.read_csv(f, usecols=COLS, dtype={"tweet_id": str, "content": str,
                                                 "author": str}, low_memory=False)
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["publish_date"] = pd.to_datetime(df["publish_date"], format="%m/%d/%Y %H:%M")
    snow = snowflake_utc(df["tweet_id"])

    # --- recover the timezone of publish_date from snowflake IDs ---
    m = snow.notna()
    offset_h = ((df.loc[m, "publish_date"] - snow[m].dt.tz_localize(None))
                .dt.total_seconds() / 3600)
    print(f"snowflake-verifiable tweets: {m.mean():.1%}; "
          f"|publish_date - snowflake UTC| < 2 min for {(offset_h.abs() < 2/60).mean():.2%}")
    offset_h = offset_h.round()                 # publish_date is truncated to the minute
    report = offset_h.value_counts(normalize=True).head(5)
    print("publish_date minus snowflake UTC (hours), share of tweets:\n", report)
    best = float(report.index[0])
    minute_utc = (df["publish_date"] - pd.Timedelta(hours=best)).dt.tz_localize("UTC")
    df["tz_check_ok"] = offset_h.reindex(df.index).eq(best)
    # publish_date has minute resolution; snowflakes carry milliseconds. Use the
    # snowflake time wherever it agrees with publish_date, else fall back.
    df["utc"] = snow.where(df["tz_check_ok"], minute_utc)
    df["utc_precise"] = df["tz_check_ok"]

    df["author"] = df["author"].str.upper()
    df["retweet"] = df["retweet"].fillna(0).astype(int)
    df = df.drop(columns=["publish_date"])
    df.to_parquet(OUT, index=False)
    print(f"{len(df):,} tweets, {df.author.nunique():,} handles -> {OUT}")
    return best, report


if __name__ == "__main__":
    build()
