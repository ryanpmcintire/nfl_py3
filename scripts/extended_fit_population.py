from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.pick_probability import (
    FLAG_SUM_COLUMN,
    MOVE_AVAILABLE_COLUMN,
    MOVE_COLUMN,
    PROBABILITY_EPSILON,
    signed_composition_flags,
)
from nfl_ats.pick_probability_fit import (
    FIT_RIDGE,
    SCHEDULE_COLUMNS,
    _arrest_incidents,
    _design,
    _fit_logit,
    _forecast_temperatures,
    _predict,
    _protection_back_side,
    _standardisers,
    build_fit_population,
)
from nfl_ats.snapshots import latest_snapshot, load_snapshot

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = REPO / "artifacts"
DATA_ROOT = REPO / "data"
SBR_ARTIFACT = ARTIFACTS_ROOT / "sbr_era_opener_eval" / "20260819T233013Z" / "scored.parquet"
SBR_SEASON_END = 2019
OUT_ROOT = ARTIFACTS_ROOT / "extended_fit_population"
KEEP_COLUMNS = [
    "game_id",
    "season",
    "week",
    "home_covered",
    "model_logit",
    FLAG_SUM_COLUMN,
    MOVE_COLUMN,
    MOVE_AVAILABLE_COLUMN,
    "opener_source",
]


def _sbr_population(schedules: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    scored = pd.read_parquet(SBR_ARTIFACT)
    scored = scored.loc[scored["season"].le(SBR_SEASON_END)].copy()
    scored["game_id"] = scored["game_id"].astype(str)
    schedule_columns = [c for c in SCHEDULE_COLUMNS if c in schedules.columns]
    joined = scored.merge(
        schedules[schedule_columns].drop_duplicates("game_id"),
        on=["game_id", "season"],
        how="left",
        suffixes=("", "_schedule"),
    )
    if "week_schedule" in joined.columns:
        joined = joined.drop(columns=["week_schedule"])
    before = len(joined)
    joined = joined.loc[joined["home_team"].notna()].copy()
    dropped_no_schedule = before - len(joined)

    margin = pd.to_numeric(joined["margin_vs_open_proxy"], errors="coerce")
    graded = joined.loc[margin.notna() & margin.ne(0.0)].reset_index(drop=True)
    graded["home_covered"] = (
        pd.to_numeric(graded["margin_vs_open_proxy"], errors="coerce").gt(0.0).astype(float)
    )
    stated = pd.to_numeric(graded["home_cover_probability_at_open_proxy"], errors="coerce").clip(
        PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON
    )
    graded["model_logit"] = np.log(stated / (1.0 - stated))

    incidents = _arrest_incidents(DATA_ROOT)
    forecasts = _forecast_temperatures(DATA_ROOT)
    protection = _protection_back_side(schedules, DATA_ROOT)
    flags = signed_composition_flags(
        graded,
        schedules,
        incidents=incidents,
        forecasts_tuesday_noon=forecasts,
        protection_back_side=protection,
    )
    graded = graded.merge(
        flags[["game_id", FLAG_SUM_COLUMN]], on="game_id", how="left", validate="one_to_one"
    )
    graded[FLAG_SUM_COLUMN] = graded[FLAG_SUM_COLUMN].fillna(0.0)
    graded[MOVE_COLUMN] = 0.0
    graded[MOVE_AVAILABLE_COLUMN] = 0.0
    graded["opener_source"] = "sbr_proxy"
    return graded[KEEP_COLUMNS].reset_index(drop=True), dropped_no_schedule


def _true_population() -> tuple[pd.DataFrame, dict]:
    graded, provenance = build_fit_population(ARTIFACTS_ROOT, DATA_ROOT)
    graded = graded.copy()
    graded["game_id"] = graded["game_id"].astype(str)
    graded["opener_source"] = "tue_open"
    return graded[KEEP_COLUMNS].reset_index(drop=True), provenance


def _loso_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    seasons = sorted(frame["season"].unique())
    predictions: list[pd.DataFrame] = []
    for season in seasons:
        train = frame.loc[frame["season"].ne(season)]
        test = frame.loc[frame["season"].eq(season)]
        if train.empty or test.empty:
            continue
        means, stds = _standardisers(train)
        beta = _fit_logit(
            _design(train, means, stds),
            train["home_covered"].to_numpy(dtype=float),
            FIT_RIDGE,
        )
        probability = _predict(test, beta, means, stds)
        pick_home = probability >= 0.5
        out = test[["game_id", "season"]].copy()
        out["correct"] = pick_home.astype(float) == test["home_covered"].to_numpy(dtype=float)
        predictions.append(out)
    if not predictions:
        return pd.DataFrame(columns=["game_id", "season", "correct"])
    return pd.concat(predictions, ignore_index=True)


def main() -> None:
    schedules, _team_stats = load_snapshot(latest_snapshot(DATA_ROOT / "raw"))
    sbr_population, dropped_no_schedule = _sbr_population(schedules)
    true_population, provenance = _true_population()
    extended = (
        pd.concat([sbr_population, true_population], ignore_index=True)
        .sort_values(["season", "week", "game_id"])
        .reset_index(drop=True)
    )

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    extended.to_parquet(out_dir / "population.parquet", index=False)

    games_per_season = {int(k): int(v) for k, v in extended.groupby("season").size().items()}
    source_counts = {str(k): int(v) for k, v in extended.groupby("opener_source").size().items()}
    flag_coverage_nonzero = float(extended[FLAG_SUM_COLUMN].ne(0.0).mean())

    extended_predictions = _loso_predictions(extended)
    true_only_predictions = _loso_predictions(true_population)

    extended_on_2020_2025 = extended_predictions.loc[extended_predictions["season"].ge(2020)]

    summary = {
        "generated_at_utc": ts,
        "games_total": len(extended),
        "games_per_season": games_per_season,
        "opener_source_counts": source_counts,
        "sbr_rows_dropped_no_schedule_match": int(dropped_no_schedule),
        "flag_coverage_nonzero_fraction": flag_coverage_nonzero,
        "sbr_model_logit_source": str(SBR_ARTIFACT.relative_to(REPO)).replace("\\", "/"),
        "sbr_model_logit_is_smooth_not_discrete": True,
        "sbr_model_logit_note": (
            "pre-2020 model_logit reuses sbr_era_opener_eval/proxy_opener_replication's "
            "preserved walk-forward output (home_cover_probability_at_open_proxy), which "
            "predates BASE_PROBABILITY_POLICY's discrete mass-preserving lattice; it is a "
            "smooth margin-model probability, a labeled caveat per AGENTS.md, not the served "
            "discrete policy. 2020-2025 rows reuse the served discrete build_fit_population "
            "output unchanged."
        ),
        "true_population_provenance": provenance,
        "sanity_look": {
            "note": "reported this run only, not recorded via weak-signals",
            "extended_2011_2025_loso_games": len(extended_predictions),
            "extended_2011_2025_loso_overall_accuracy": float(
                extended_predictions["correct"].mean()
            )
            if len(extended_predictions)
            else None,
            "extended_2011_2025_loso_accuracy_on_2020_2025_subset": float(
                extended_on_2020_2025["correct"].mean()
            )
            if len(extended_on_2020_2025)
            else None,
            "extended_2011_2025_loso_games_2020_2025_subset": len(extended_on_2020_2025),
            "true_2020_2025_only_loso_accuracy": float(true_only_predictions["correct"].mean())
            if len(true_only_predictions)
            else None,
            "true_2020_2025_only_loso_games": len(true_only_predictions),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
