from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from nfl_ats.nfl_week import week_cycle_sunday

REPO_ROOT = Path(__file__).resolve().parents[1]
FEATURE_TABLE = REPO_ROOT / "data" / "processed" / "game_features_weak_stack.parquet"
PLAYER_SNAPSHOT = REPO_ROOT / "data" / "players" / "raw" / "20260910T205112Z"
DECISION_HOURS_BEFORE_KICKOFF = 24
SEASON_START = 2020
SEASON_END = 2025
_EASTERN = ZoneInfo("America/New_York")


def tuesday_noon_et(kickoff_utc: pd.Timestamp) -> pd.Timestamp:

    kickoff_eastern = kickoff_utc.tz_convert(_EASTERN)
    sunday = week_cycle_sunday(kickoff_eastern.date())
    tuesday = sunday - timedelta(days=5)
    tuesday_noon = datetime.combine(tuesday, time(12), tzinfo=_EASTERN)
    return pd.Timestamp(tuesday_noon).tz_convert("UTC")


def load_games() -> pd.DataFrame:

    games = pd.read_parquet(
        FEATURE_TABLE,
        columns=["game_id", "season", "week", "game_type", "kickoff", "home_team", "away_team"],
    )
    games["kickoff"] = pd.to_datetime(games["kickoff"], utc=True, errors="coerce")
    games = games.loc[
        (games["game_type"] == "REG")
        & (games["season"] >= SEASON_START)
        & (games["season"] <= SEASON_END)
        & games["kickoff"].notna()
    ].copy()
    games["tuesday_noon_et_utc"] = games["kickoff"].apply(tuesday_noon_et)
    games["decision_at"] = games["kickoff"] - pd.Timedelta(hours=DECISION_HOURS_BEFORE_KICKOFF)
    long = pd.concat(
        [
            games.rename(columns={"home_team": "team"})[
                [
                    "game_id",
                    "season",
                    "week",
                    "team",
                    "kickoff",
                    "tuesday_noon_et_utc",
                    "decision_at",
                ]
            ],
            games.rename(columns={"away_team": "team"})[
                [
                    "game_id",
                    "season",
                    "week",
                    "team",
                    "kickoff",
                    "tuesday_noon_et_utc",
                    "decision_at",
                ]
            ],
        ],
        ignore_index=True,
    )
    return long


def load_injuries() -> pd.DataFrame:

    injuries = pd.read_parquet(
        PLAYER_SNAPSHOT / "injuries.parquet",
        columns=["season", "week", "team", "gsis_id", "effective_observed_at", "observed_at_basis"],
    )
    injuries["effective_observed_at"] = pd.to_datetime(
        injuries["effective_observed_at"], utc=True, errors="coerce"
    )
    injuries = injuries.loc[
        (injuries["season"] >= SEASON_START) & (injuries["season"] <= SEASON_END)
    ].copy()
    return injuries


def per_season_basis_mix(injuries: pd.DataFrame) -> dict:

    out = {}
    for season, group in injuries.groupby("season"):
        counts = group["observed_at_basis"].value_counts(dropna=False).to_dict()
        total = len(group)
        out[int(season)] = {
            "n_rows": total,
            "basis_counts": {str(k): int(v) for k, v in counts.items()},
            "share_real_date_modified": float(
                counts.get("date_modified", 0) / total if total else float("nan")
            ),
        }
    return out


def tuesday_vs_decision_coverage(games_long: pd.DataFrame, injuries: pd.DataFrame) -> dict:

    real = injuries.loc[injuries["observed_at_basis"] == "date_modified"].copy()
    merged = games_long.merge(real, on=["season", "week", "team"], how="left")

    def summarize(frame: pd.DataFrame) -> dict:
        keys = ["game_id", "season", "week", "team"]
        team_games = frame.drop_duplicates(keys)[keys]
        n_team_games = len(team_games)
        visible_tuesday = frame.loc[frame["effective_observed_at"] <= frame["tuesday_noon_et_utc"]]
        visible_decision = frame.loc[frame["effective_observed_at"] <= frame["decision_at"]]
        n_tuesday = visible_tuesday.drop_duplicates(keys).shape[0]
        n_decision = visible_decision.drop_duplicates(keys).shape[0]
        return {
            "n_team_games": n_team_games,
            "team_games_with_real_revision_visible_at_tuesday_noon_et": n_tuesday,
            "share_visible_at_tuesday_noon_et": float(n_tuesday / n_team_games)
            if n_team_games
            else float("nan"),
            "team_games_with_real_revision_visible_at_decision_time": n_decision,
            "share_visible_at_decision_time": float(n_decision / n_team_games)
            if n_team_games
            else float("nan"),
        }

    by_season = {}
    for season in sorted(games_long["season"].unique()):
        by_season[int(season)] = summarize(merged.loc[merged["season"] == season])

    overall = summarize(merged)
    return {"overall": overall, "by_season": by_season}


def proxy_instant_degeneracy(games_long: pd.DataFrame, injuries: pd.DataFrame) -> dict:

    proxy = injuries.loc[injuries["observed_at_basis"] == "week_proxy"].copy()
    if proxy.empty:
        return {"n_proxy_rows": 0}
    merged = proxy.merge(
        games_long.drop_duplicates(["season", "week", "team"]),
        on=["season", "week", "team"],
        how="left",
    )
    merged["hours_before_kickoff"] = (
        merged["kickoff"] - merged["effective_observed_at"]
    ).dt.total_seconds() / 3600.0
    merged["equals_decision_at"] = (
        merged["effective_observed_at"] - merged["decision_at"]
    ).abs() < pd.Timedelta(seconds=1)
    distinct_instants_per_team_week = merged.groupby(["season", "week", "team"])[
        "effective_observed_at"
    ].nunique()
    return {
        "n_proxy_rows": len(merged),
        "n_team_weeks_with_proxy_rows": len(distinct_instants_per_team_week),
        "team_weeks_with_more_than_one_distinct_timestamp": int(
            (distinct_instants_per_team_week > 1).sum()
        ),
        "share_rows_exactly_at_decision_time": float(merged["equals_decision_at"].mean()),
        "hours_before_kickoff_min": float(merged["hours_before_kickoff"].min()),
        "hours_before_kickoff_max": float(merged["hours_before_kickoff"].max()),
        "hours_before_kickoff_mean": float(merged["hours_before_kickoff"].mean()),
        "hours_before_kickoff_std": float(merged["hours_before_kickoff"].std()),
    }


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    games_long = load_games()
    injuries = load_injuries()

    result = {
        "population": {
            "season_start": SEASON_START,
            "season_end": SEASON_END,
            "n_team_games": len(games_long),
            "n_games": int(games_long["game_id"].nunique()),
        },
        "per_season_basis_mix": per_season_basis_mix(injuries),
        "tuesday_vs_decision_coverage_real_timestamps_only": tuesday_vs_decision_coverage(
            games_long, injuries
        ),
        "proxy_instant_degeneracy_2025": proxy_instant_degeneracy(games_long, injuries),
    }

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else REPO_ROOT / "artifacts" / "players_on_field_rating_unit3" / ts
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))
    print(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
