"""Release-blocking tests for the XLG-06 Stage-2 NFL screen (no network)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nfl_ats.data import DataContractError
from scripts.xlg06_stage2_nfl_screen import (
    blocked_bootstrap_correlation,
    build_stage2_population,
    rookie_epa_totals,
    split_half_reliability,
)


def _crosswalk() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gsis_id": ["A", "B", "C", "D", "E"],
            "position": ["WR", "RB", "TE", "WR", "QB"],
            "rating": [0.95, 0.85, 0.90, None, 0.99],
            "year": [2018, 2018, 2019, 2019, 2019],
        }
    )


def _stats() -> pd.DataFrame:
    rows = []
    # A: 2020 rookie REG (2 weeks) + POST row that must not count; 2021 follow-up.
    rows += [
        {
            "player_id": "A",
            "season": 2020,
            "season_type": "REG",
            "week": 1,
            "rushing_epa": 1.0,
            "receiving_epa": 2.0,
        },
        {
            "player_id": "A",
            "season": 2020,
            "season_type": "REG",
            "week": 2,
            "rushing_epa": 0.5,
            "receiving_epa": 0.5,
        },
        {
            "player_id": "A",
            "season": 2020,
            "season_type": "POST",
            "week": 19,
            "rushing_epa": 100.0,
            "receiving_epa": 100.0,
        },
        {
            "player_id": "A",
            "season": 2021,
            "season_type": "REG",
            "week": 1,
            "rushing_epa": 50.0,
            "receiving_epa": 50.0,
        },
        # B: 2020 rookie REG.
        {
            "player_id": "B",
            "season": 2020,
            "season_type": "REG",
            "week": 1,
            "rushing_epa": -1.0,
            "receiving_epa": 0.0,
        },
        # C: 2025 debut (incomplete rookie season) -> excluded.
        {
            "player_id": "C",
            "season": 2025,
            "season_type": "REG",
            "week": 1,
            "rushing_epa": 9.0,
            "receiving_epa": 9.0,
        },
    ]
    return pd.DataFrame(rows)


def test_eligibility_excludes_without_silence() -> None:
    frame, excluded = build_stage2_population(_crosswalk(), _stats())
    # D has a null rating, E is a QB (non-skill), C debuts in 2025.
    assert set(frame["gsis_id"]) == {"A", "B"}
    assert excluded["null_rating"] == 1
    assert excluded["incomplete_rookie_season_2025"] == 1
    assert excluded["no_production_rows"] == 0


def test_outcome_sums_reg_rookie_rows_only() -> None:
    frame, _ = build_stage2_population(_crosswalk(), _stats())
    stats = _stats()
    stats["gsis_id"] = stats["player_id"].astype(str)
    reg = stats.loc[stats["season_type"].eq("REG")].copy()
    reg = reg.merge(frame.loc[:, ["gsis_id", "rookie_season"]], on="gsis_id")
    reg = reg.loc[reg["season"].eq(reg["rookie_season"])].copy()
    reg["rushing_epa"] = pd.to_numeric(reg["rushing_epa"], errors="coerce").fillna(0.0)
    reg["receiving_epa"] = pd.to_numeric(reg["receiving_epa"], errors="coerce").fillna(0.0)
    reg["weekly_epa"] = reg["rushing_epa"] + reg["receiving_epa"]
    scored = rookie_epa_totals(frame, reg)
    assert scored.loc[scored["gsis_id"].eq("A"), "rookie_epa"].iloc[0] == pytest.approx(4.0)
    assert scored.loc[scored["gsis_id"].eq("B"), "rookie_epa"].iloc[0] == pytest.approx(-1.0)


def test_bootstrap_is_deterministic_for_a_seed() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=40)
    y = 0.5 * x + rng.normal(size=40)
    blocks = np.array([f"c{i // 10}" for i in range(40)])
    first = blocked_bootstrap_correlation(x, y, blocks, seed=99, samples=200)
    second = blocked_bootstrap_correlation(x, y, blocks, seed=99, samples=200)
    assert first["pearson_r"] == pytest.approx(second["pearson_r"])
    assert first["pearson_r_ci95"] == pytest.approx(second["pearson_r_ci95"])
    assert first["pearson_probability_positive"] == pytest.approx(
        second["pearson_probability_positive"]
    )


def test_chronology_violation_fails_closed() -> None:
    crosswalk = _crosswalk()
    crosswalk.loc[crosswalk["gsis_id"].eq("A"), "year"] = 2021
    with pytest.raises(DataContractError, match="strictly predate"):
        build_stage2_population(crosswalk, _stats())


def test_reliability_needs_enough_weeks() -> None:
    thin = pd.DataFrame(
        {
            "gsis_id": ["A", "A", "B", "B"],
            "week": [1, 2, 1, 2],
            "weekly_epa": [1.0, 2.0, 0.5, 0.5],
        }
    )
    result = split_half_reliability(thin, min_weeks=4)
    assert result["insufficient"] is True
