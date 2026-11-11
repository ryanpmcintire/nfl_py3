from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nfl_ats.clv import HISTORICAL_CAPTURE_KIND, load_decision_quotes
from nfl_ats.pick_refresh import pick_deadline, sunday_pick_lock

ROOT = Path(__file__).resolve().parents[1]
HOURLY = ROOT / "artifacts/experiments/sharp_book_movement/quotes.parquet"
RAW = ROOT / "data/market/raw"
MIN_BOOKS = 5
QUOTE_COLUMNS = [
    "nflverse_game_id",
    "season",
    "week",
    "commence_time_utc",
    "bookmaker_key",
    "observed_at_utc",
    "home_spread_line",
]


def hourly_quotes():
    q = pd.read_parquet(HOURLY, columns=[*QUOTE_COLUMNS, "market"])
    return q.loc[q["market"].eq("spreads")].drop(columns="market")


def snapshot_quotes():
    q = load_decision_quotes(RAW, capture_kind=HISTORICAL_CAPTURE_KIND, seasons=[2020, 2021, 2022])
    q = q.loc[q["market"].eq("spreads")].copy()
    return q[[*QUOTE_COLUMNS, "decision_label"]]


def deadlines(q):
    games = (
        q.groupby("nflverse_game_id", as_index=False)["commence_time_utc"]
        .min()
        .assign(cycle=lambda f: f["nflverse_game_id"].str[:7])
    )
    out = {}
    for _, g in games.groupby("cycle"):
        lock = sunday_pick_lock(g["commence_time_utc"])
        for gid, k in zip(g["nflverse_game_id"], g["commence_time_utc"], strict=True):
            out[gid] = pick_deadline(pd.Timestamp(k), lock)
    return pd.Series(out)


def consensus(q, deadline):
    q = q.dropna(subset=["home_spread_line"]).copy()
    q["observed_at_utc"] = pd.to_datetime(q["observed_at_utc"], utc=True)
    q["deadline"] = q["nflverse_game_id"].map(deadline)
    n_all = q.groupby("nflverse_game_id")["bookmaker_key"].nunique()
    before = q.loc[q["observed_at_utc"] < q["deadline"]]
    last = (
        before.sort_values("observed_at_utc")
        .groupby(["nflverse_game_id", "bookmaker_key"], as_index=False)
        .last()
    )
    agg = last.groupby("nflverse_game_id").agg(
        consensus_line=("home_spread_line", "median"),
        consensus_books=("bookmaker_key", "nunique"),
        newest_quote=("observed_at_utc", "max"),
        oldest_quote=("observed_at_utc", "min"),
    )
    agg["deadline"] = agg.index.map(deadline)
    agg["minutes_before_deadline_newest"] = (
        agg["deadline"] - agg["newest_quote"]
    ).dt.total_seconds() / 60
    agg["minutes_before_deadline_oldest"] = (
        agg["deadline"] - agg["oldest_quote"]
    ).dt.total_seconds() / 60
    agg["books_any_time"] = n_all.reindex(agg.index)
    agg["admissible"] = agg["consensus_books"] >= MIN_BOOKS
    agg.loc[~agg["admissible"], "consensus_line"] = np.nan
    return agg


def build():
    hq = hourly_quotes()
    sq = snapshot_quotes()
    hq = hq.loc[hq["season"].between(2023, 2025)]
    parts = []
    for q in (sq, hq):
        parts.append(consensus(q, deadlines(q)))
    return pd.concat(parts)


if __name__ == "__main__":
    t = build()
    t["season"] = t.index.str[:4].astype(int)
    t.to_csv(ROOT / "tests/scratch/mod57/p2j/times.csv")
    print(
        t.groupby("season")
        .agg(
            games=("admissible", "size"),
            admissible=("admissible", "sum"),
            median_min_newest=("minutes_before_deadline_newest", "median"),
            min_min_newest=("minutes_before_deadline_newest", "min"),
            max_min_newest=("minutes_before_deadline_newest", "max"),
            median_books=("consensus_books", "median"),
        )
        .to_string()
    )
