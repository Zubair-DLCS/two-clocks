"""Shared loading and constants."""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
for d in (RES, FIG):
    os.makedirs(d, exist_ok=True)

# The five English-facing roles coded by Linvill & Warren (Clemson).
ROLES = ["RightTroll", "LeftTroll", "NewsFeed", "HashtagGamer", "Fearmonger"]
# Analysis window used by Linvill & Warren, padded so holiday baselines at
# the window edges are defined.
START, END = "2015-06-01", "2018-01-20"
CORE_START, CORE_END = "2015-06-19", "2018-01-01"
ELECTION = pd.Timestamp("2016-11-08")


def load_english(columns=None, core=True):
    cols = ["author", "utc", "language", "account_category"] + [c for c in (columns or [])
                                                                if c not in ("author", "utc", "language", "account_category")]
    df = pd.read_parquet(os.path.join(DATA, "tweets.parquet"), columns=cols)
    df = df[(df.language == "English") & df.account_category.isin(ROLES)]
    lo, hi = (CORE_START, CORE_END) if core else (START, END)
    df = df[(df.utc >= lo) & (df.utc < hi)].copy()
    df["role"] = df.pop("account_category")
    return df.reset_index(drop=True)
