"""Release-blocking tests for the pairwise co-absence screen (no network)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nfl_ats.data import DataContractError
from scripts.absence_pairwise_screen import (
    bootstrap_excess,
    enumerate_pairs,
    pooled_excess,
)


def _frames(coupled: bool = False) -> pd.DataFrame:
    """One team, twelve weeks, four players.

    Base world: A/B sit weeks 5-8, C sits odd weeks, D even weeks. The
    pooled excess dilutes one coupled pair among five independent ones, so
    direction is asserted on the fully-coupled world (every pair sits weeks
    5-8 together: joint 4/12 vs implied (1/3)^2, excess exactly 3.0).
    """

    rows = []
    sits = (
        {"A": {5, 6, 7, 8}, "B": {5, 6, 7, 8}, "C": {5, 6, 7, 8}, "D": {5, 6, 7, 8}}
        if coupled
        else {
            "A": {5, 6, 7, 8},
            "B": {5, 6, 7, 8},
            "C": {1, 3, 5, 7, 9, 11},
            "D": {2, 4, 6, 8, 10, 12},
        }
    )
    for week in range(1, 13):
        for player in ("A", "B", "C", "D"):
            rows.append(
                {
                    "gsis_id": player,
                    "season": 2020,
                    "week": week,
                    "team": "H",
                    "unit": "OFF_SKILL",
                    "absent": week in sits[player],
                    "side_snaps": 0.0 if week in sits[player] else 50.0,
                }
            )
    return pd.DataFrame(rows)


def test_pair_enumeration_and_overlap_floor() -> None:
    pairs = enumerate_pairs(_frames(), "OFF_SKILL")
    assert set(pairs["pair"].unique()) == {
        ("A", "B"),
        ("A", "C"),
        ("A", "D"),
        ("B", "C"),
        ("B", "D"),
        ("C", "D"),
    }
    # A/B together in all 12 weeks (≥10 floor), jointly absent in 5-8.
    ab = pairs.loc[pairs["pair"].map(lambda pair: set(pair) == {"A", "B"})]
    assert len(ab) == 12
    assert int(ab["both_absent"].sum()) == 4


def test_fully_coupled_world_excess_is_three() -> None:
    pairs = enumerate_pairs(_frames(coupled=True), "OFF_SKILL")
    result = pooled_excess(pairs)
    assert result["pairs"] == 6
    assert result["excess_ratio"] == pytest.approx(3.0)


def test_bootstrap_is_deterministic() -> None:
    pairs = enumerate_pairs(_frames(coupled=True), "OFF_SKILL")
    first = bootstrap_excess(pairs, seed=11, samples=50)
    second = bootstrap_excess(pairs, seed=11, samples=50)
    assert first["excess_ratio"] == pytest.approx(second["excess_ratio"])
    assert first["excess_ratio_ci95"] == pytest.approx(second["excess_ratio_ci95"])


def test_empty_unit_fails_closed() -> None:
    with pytest.raises(DataContractError, match="no contributor games"):
        enumerate_pairs(_frames().iloc[:0].copy(), "OFF_SKILL")
