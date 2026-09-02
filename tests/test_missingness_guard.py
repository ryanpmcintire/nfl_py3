"""Prediction-safety guard for WP15 / MOD-13 (docs/missingness_audit.md).

The production margin pipeline imputes with
``SimpleImputer(strategy="median", add_indicator=True)`` (see
``src/nfl_ats/margin.py::make_margin_estimator``), so every training-missing
feature gets a learned binary "this was missing" coefficient. If the live
card presents a missing/present state for some column that essentially never
occurred in training, the imputer + ridge combination is extrapolating on
that row rather than interpolating. This test reads the real local feature
table (when present -- it is a generated artifact, absent in a fresh clone)
and checks that invariant for whatever the CURRENT locked-but-unplayed week
is, rather than a hardcoded season/week, so it keeps testing the actual
upcoming lock as the season advances.

Diagnostic-stage only (WP15): this test does not run or grade an experiment,
write to the weak-signals/rotation registries, or gate a promotion decision.
It is a cheap structural check on the feature table `scripts/missingness_audit.py`
already computes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
FEATURE_TABLE_PATH = REPO / "data" / "processed" / "game_features_weak_stack.parquet"


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_script("missingness_audit_test", "missingness_audit.py")


def _current_lock_target(frame: pd.DataFrame) -> tuple[int, int]:
    """The earliest (season, week) with a posted line but no graded result.

    That is the imminent lock the live card will actually be scored on --
    the same population ``nfl-ats margin-predict`` would target "as of now".
    """

    reg = audit.regular_season_rows(frame)
    candidates = reg.loc[reg["result"].isna() & reg["spread_line"].notna()].sort_values(
        ["gameday", "game_id"]
    )
    if candidates.empty:
        pytest.skip("No upcoming (unplayed, lined) week found in the local feature table")
    first = candidates.iloc[0]
    return int(first["season"]), int(first["week"])


@pytest.fixture()
def frame() -> pd.DataFrame:
    if not FEATURE_TABLE_PATH.is_file():
        pytest.skip("local feature table absent (generated artifact, not in a fresh clone)")
    return audit.load_frame(FEATURE_TABLE_PATH)


def test_current_lock_missingness_states_are_not_rare_relative_to_prior_season(
    frame: pd.DataFrame,
) -> None:
    season, week = _current_lock_target(frame)
    reference_season = season - 1
    reg = audit.regular_season_rows(frame)
    if reg.loc[reg["season"].eq(reference_season)].empty:
        pytest.skip(f"No {reference_season} regular-season rows to use as a reference")

    columns = audit.margin_feature_columns("market_residual", "weak_stack")
    risk = audit.week1_extrapolation_risk(
        frame,
        columns,
        season=season,
        week=week,
        reference_season=reference_season,
    )
    flagged = risk.loc[risk["rare_relative_to_reference"]]
    flagged_columns = ["column", "game_id", "value_missing", "reference_state_frac"]
    assert flagged.empty, (
        f"{len(flagged)} (column, game) pairs for {season} week {week} present a "
        f"missing/present state that occurred in < {audit.RARE_THRESHOLD:.0%} of "
        f"{reference_season} regular-season training rows -- an imputer + ridge "
        "extrapolation risk on the live card:\n"
        f"{flagged[flagged_columns].to_string(index=False)}"
    )


def test_no_production_column_is_always_missing_in_training(frame: pd.DataFrame) -> None:
    """A column missing in every 2009-2025 training row would be dead weight
    the median imputer silently fills with a constant on every row -- not
    itself unsafe for a NEW row, but a sign a feature never had real signal
    to learn from. Cheap to catch here rather than downstream.
    """

    columns = audit.margin_feature_columns("market_residual", "weak_stack")
    seasons = list(range(2009, 2026))
    missingness = audit.per_season_missingness(frame, columns, seasons)
    classification = audit.classify_columns(missingness, seasons)
    always_missing = classification.loc[classification["category"] == "always_missing"]
    assert always_missing.empty, (
        "Production columns missing in every 2009-2025 training row: "
        f"{always_missing['column'].tolist()}"
    )
