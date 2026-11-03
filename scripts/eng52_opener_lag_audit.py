from __future__ import annotations

import argparse
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from nfl_ats.clv import HISTORICAL_CAPTURE_KIND, load_snapshot_manifest_index
from nfl_ats.nfl_week import week_cycle_sunday
from nfl_ats.odds_backfill import DECISION_TIMES

EASTERN = ZoneInfo("America/New_York")
SECONDS_PER_HOUR = 3600.0


def tuesday_target(commence: pd.Timestamp) -> pd.Timestamp:
    local = commence.tz_convert(EASTERN)
    sunday = week_cycle_sunday(local.date())
    offset_days, clock = next((d, t) for label, d, t, _ in DECISION_TIMES if label == "tue_open")
    day = pd.Timestamp(sunday) + pd.Timedelta(days=offset_days)
    assert isinstance(clock, time)
    stamp = pd.Timestamp.combine(day.date(), clock).tz_localize(EASTERN)
    return stamp.tz_convert("UTC")


def per_game_lag(root: Path, game_ids: set[str]) -> pd.DataFrame:
    index = load_snapshot_manifest_index(root)
    sel = index.loc[
        index["capture_kind"].eq(HISTORICAL_CAPTURE_KIND) & index["decision_label"].eq("tue_open")
    ]
    rows = []
    for r in sel.itertuples(index=False):
        q = pd.read_parquet(
            Path(str(r.dir)) / "quotes.parquet",
            columns=[
                "nflverse_game_id",
                "commence_time_utc",
                "market",
                "outcome_side",
                "market_last_update_utc",
                "observed_at_utc",
            ],
        )
        q = q.loc[q["market"].eq("spreads") & q["outcome_side"].eq("HOME")]
        q = q.loc[q["nflverse_game_id"].isin(game_ids)]
        for gid, g in q.groupby("nflverse_game_id"):
            commence = pd.Timestamp(g["commence_time_utc"].iloc[0])
            age = (
                pd.to_datetime(g["observed_at_utc"], utc=True)
                - pd.to_datetime(g["market_last_update_utc"], utc=True)
            ).dt.total_seconds() / SECONDS_PER_HOUR
            rows.append(
                {
                    "game_id": gid,
                    "snapshot_id": r.snapshot_id,
                    "snapshot_utc": r.snapshot_timestamp_utc,
                    "requested_utc": r.requested_at_utc,
                    "commence_utc": commence,
                    "target_utc": tuesday_target(commence),
                    "quote_age_h_median": float(age.median()),
                    "quote_age_h_max": float(age.max()),
                    "books": len(g),
                }
            )
    out = pd.DataFrame(rows)
    out["lag_h"] = (out["snapshot_utc"] - out["target_utc"]).dt.total_seconds() / SECONDS_PER_HOUR
    out["request_lag_h"] = (
        out["requested_utc"] - out["target_utc"]
    ).dt.total_seconds() / SECONDS_PER_HOUR
    out["flagged"] = out["snapshot_utc"] > out["target_utc"]
    return out


def boot(x: np.ndarray, blocks: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    uniq = np.unique(blocks)
    idx = {b: np.where(blocks == b)[0] for b in uniq}
    out = np.empty(n)
    for i in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx[b] for b in pick])
        out[i] = x[rows].mean()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/market/raw")
    ap.add_argument(
        "--eval", default="artifacts/opener_evaluation/20261103T195331Z/per_game.parquet"
    )
    ap.add_argument("--out", default="artifacts/eng52/opener_lag.csv")
    ap.add_argument("--samples", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260817)
    args = ap.parse_args()
    pg = pd.read_parquet(args.eval)
    lag = per_game_lag(Path(args.root), set(pg["game_id"]))
    lag = lag.sort_values("snapshot_utc").drop_duplicates("game_id", keep="last")
    df = pg.merge(lag, on="game_id", how="left")
    df.to_csv(args.out, index=False)
    print("games", len(df), "no_lag_row", int(df["lag_h"].isna().sum()))
    print(df.groupby("season")["lag_h"].describe().round(3).to_string())
    print(df.groupby("season")["flagged"].agg(["sum", "count"]).to_string())
    print(
        df.groupby("season")[["quote_age_h_median", "quote_age_h_max"]]
        .median()
        .round(2)
        .to_string()
    )
    print(df["lag_h"].quantile([0, 0.01, 0.5, 0.99, 1]).to_string())
    print("request_lag", df["request_lag_h"].describe().to_string())


if __name__ == "__main__":
    main()
