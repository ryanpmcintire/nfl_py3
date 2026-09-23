from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nfl_ats.availability import (
    build_availability_outcomes,
    build_season_lagged_availability_rates,
    score_availability_rates,
    summarize_availability_scores,
)
from nfl_ats.players import (
    attach_snap_player_ids,
    canonicalize_injuries,
    canonicalize_rosters,
    canonicalize_snaps,
    latest_player_snapshot,
    load_player_snapshot,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
FEATURES_PATH = DATA_ROOT / "processed" / "game_features.parquet"
PLAYER_RAW_ROOT = DATA_ROOT / "players" / "raw"
OUTPUT_TABLE = DATA_ROOT / "processed" / "injury_play_outcomes.parquet"
ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "injury_outcomes_table"
DECISION_HOURS_BEFORE_KICKOFF = 24


def _snap_shares(snaps_with_ids: pd.DataFrame) -> pd.DataFrame:
    rows = snaps_with_ids.loc[snaps_with_ids["gsis_id"].notna()].copy()
    for column in ("offense_pct", "defense_pct", "st_pct"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce").fillna(0.0)
    rows["primary_snap_share"] = rows[["offense_pct", "defense_pct", "st_pct"]].max(axis=1)
    return (
        rows.groupby(["season", "week", "team", "gsis_id"], observed=True)
        .agg(
            offense_pct=("offense_pct", "max"),
            defense_pct=("defense_pct", "max"),
            st_pct=("st_pct", "max"),
            snap_share=("primary_snap_share", "max"),
        )
        .reset_index()
    )


def _latest_report(canonical_injury_rows: pd.DataFrame) -> pd.DataFrame:
    ordered = canonical_injury_rows.sort_values("effective_observed_at")
    latest = ordered.drop_duplicates(["season", "week", "team", "gsis_id"], keep="last")
    return latest[
        [
            "season",
            "week",
            "team",
            "gsis_id",
            "report_status",
            "practice_status",
            "effective_observed_at",
        ]
    ].rename(columns={"effective_observed_at": "decision_observed_at"})


def main() -> None:
    started = datetime.now(UTC)
    features = pd.read_parquet(
        FEATURES_PATH,
        columns=["game_id", "season", "week", "home_team", "away_team", "kickoff"],
    )
    snapshot = latest_player_snapshot(PLAYER_RAW_ROOT)
    injuries, rosters, snaps = load_player_snapshot(snapshot)

    canonical_injury_rows = canonicalize_injuries(injuries, timestamp_fallback="drop")
    canonical_roster_rows = canonicalize_rosters(rosters)
    snaps_with_ids = attach_snap_player_ids(canonicalize_snaps(snaps), canonical_roster_rows)

    outcomes = build_availability_outcomes(
        canonical_injury_rows,
        snaps_with_ids,
        features,
        decision_hours_before_kickoff=DECISION_HOURS_BEFORE_KICKOFF,
    )

    latest_report = _latest_report(canonical_injury_rows)
    outcomes = outcomes.merge(
        latest_report,
        on=["season", "week", "team", "gsis_id"],
        how="left",
        validate="one_to_one",
    )

    snap_shares = _snap_shares(snaps_with_ids)
    outcomes = outcomes.merge(
        snap_shares,
        on=["season", "week", "team", "gsis_id"],
        how="left",
        validate="one_to_one",
    )
    for column in ("offense_pct", "defense_pct", "st_pct", "snap_share"):
        outcomes[column] = outcomes[column].fillna(0.0)

    target_seasons = sorted(outcomes["season"].astype(int).unique().tolist())
    rates = build_season_lagged_availability_rates(outcomes, target_seasons=target_seasons)
    scored = score_availability_rates(outcomes, rates)
    calibration_summary = summarize_availability_scores(scored)

    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    outcomes.to_parquet(OUTPUT_TABLE, index=False)

    timestamp = started.strftime("%Y%m%dT%H%M%SZ")
    output_dir = ARTIFACTS_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_calibration = (
        scored.groupby(["report_category", "practice_category"], observed=True)
        .agg(
            observations=("played", "size"),
            actual_play_rate=("played", "mean"),
            fixed_predicted_play_rate=("fixed_unavailability", lambda s: float(1.0 - s.mean())),
            learned_predicted_play_rate=("learned_unavailability", lambda s: float(1.0 - s.mean())),
        )
        .reset_index()
        .sort_values(["report_category", "practice_category"])
    )
    cell_calibration_path = output_dir / "cell_calibration.csv"
    cell_calibration.to_csv(cell_calibration_path, index=False)

    summary = {
        "built_at_utc": started.isoformat(),
        "features_source": str(FEATURES_PATH.relative_to(REPO_ROOT)),
        "player_snapshot": snapshot.snapshot_id,
        "decision_hours_before_kickoff": DECISION_HOURS_BEFORE_KICKOFF,
        "rate_scheme": "season_lagged_earlier_seasons_only",
        "output_table": str(OUTPUT_TABLE.relative_to(REPO_ROOT)),
        "rows": len(outcomes),
        "scored_rows": len(scored),
        "dropped_first_season_rows": len(outcomes) - len(scored),
        "seasons": target_seasons,
        "games": int(outcomes["game_id"].nunique()),
        "players": int(outcomes["gsis_id"].nunique()),
        "calibration_fixed_vs_learned": calibration_summary.to_dict(orient="records"),
        "cell_calibration_path": str(cell_calibration_path.relative_to(REPO_ROOT)),
        "cell_calibration": cell_calibration.to_dict(orient="records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
