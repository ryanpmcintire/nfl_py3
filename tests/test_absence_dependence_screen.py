"""Release-blocking tests for the absence-dependence screen (no network)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nfl_ats.data import DataContractError
from scripts.absence_dependence_screen import (
    bootstrap_excess_ratio,
    build_contributor_games,
    unit_excess,
)


def _frames(independent: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic worlds with disjoint absence structure.

    Independent world: exactly one absence per team-game (no full sits, no
    multi-absence games). Coupled world: whole-unit sits in weeks 5-6 with
    full participation elsewhere (full-sit rate far above implied).
    """
    snaps_rows, roster_rows = [], []
    weeks = range(1, 5) if independent else range(1, 9)
    teams = ("H", "A") if independent else ("H",)
    for team in teams:
        for week in weeks:
            for player in range(4):
                absent = player == (week % 4) and team == "H" if independent else week in (5, 6)
                snaps_rows.append(
                    {
                        "gsis_id": f"{team}P{player}",
                        "season": 2020,
                        "week": week,
                        "team": team,
                        "game_type": "REG",
                        "offense_snaps": 0.0 if absent else 50.0,
                        "defense_snaps": 0.0,
                    }
                )
                roster_rows.append(
                    {
                        "gsis_id": f"{team}P{player}",
                        "season": 2020,
                        "week": week,
                        "team": team,
                        "unit": "OFF_SKILL",
                    }
                )
    snaps = pd.DataFrame(snaps_rows)
    rosters = pd.DataFrame(roster_rows)
    return snaps, rosters


def test_independent_world_has_no_full_sits() -> None:
    contributors = build_contributor_games(*_frames(independent=True))
    result = unit_excess(contributors, "OFF_SKILL")
    assert result["p2plus"] == pytest.approx(0.0)
    assert result["p_full_sit"] == pytest.approx(0.0)
    assert result["team_games"] == 6


def test_coupled_world_full_sit_excess_is_large() -> None:
    contributors = build_contributor_games(*_frames(independent=False))
    result = unit_excess(contributors, "OFF_SKILL")
    assert result["p_full_sit"] == pytest.approx(2.0 / 7.0)
    assert result["excess_ratio"] > 10.0


def test_bootstrap_is_deterministic() -> None:
    contributors = build_contributor_games(*_frames(independent=False))
    first = bootstrap_excess_ratio(contributors, "OFF_SKILL", seed=11, samples=50)
    second = bootstrap_excess_ratio(contributors, "OFF_SKILL", seed=11, samples=50)
    assert first["excess_ratio"] == pytest.approx(second["excess_ratio"])
    assert first["excess_ratio_ci95"] == pytest.approx(second["excess_ratio_ci95"])


def test_empty_unit_fails_closed() -> None:
    contributors = build_contributor_games(*_frames(independent=True))
    with pytest.raises(DataContractError, match="no contributor games"):
        unit_excess(contributors, "DEF_FRONT")


def test_bye_like_and_data_missing_games_never_count() -> None:
    snaps, rosters = _frames(independent=True)
    contributors = build_contributor_games(snaps, rosters)
    # A rostered team-week with no game (bye-like) must never enter the
    # frame: restrict to scheduled weeks, then apply the data-presence gate.
    schedule = pd.DataFrame(
        [
            {"season": 2020, "week": week, "team": team}
            for team in ("H", "A")
            for week in (2, 3)  # week 4 has no game for anyone here
        ]
    )
    contributors = contributors.merge(
        schedule.assign(on_schedule=True), on=["season", "week", "team"], how="inner"
    )
    team_snaps = (
        contributors.groupby(["season", "team", "week"], sort=False)["side_snaps"]
        .sum()
        .rename("team_side_snaps")
    )
    contributors = contributors.merge(team_snaps, on=["season", "team", "week"], how="left")
    contributors = contributors.loc[contributors["team_side_snaps"].ge(100)].copy()
    assert set(contributors["week"].unique()) <= {2, 3}
    result = unit_excess(contributors, "OFF_SKILL")
    assert result["team_games"] == 4
