from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


screen = _load_script("qb_age_curve_screen_test", "qb_age_curve_screen.py")


def _starters_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4", "g5"],
            "team": ["ARI"] * 5,
            "qb_id": ["q1"] * 5,
            "qb_name": ["Q One"] * 5,
            "season": [2009, 2009, 2010, 2010, 2011],
            "week": [1, 5, 1, 9, 1],
            "cpoe_mean": [0.1, -0.2, 0.3, 0.4, 0.0],
        }
    )


def test_career_starts_use_only_strictly_prior_seasons() -> None:
    axes = screen.compute_career_axes(_starters_frame()).sort_values("game_id")
    entering = axes["career_starts_entering"].tolist()
    first = axes["first_start_season"].tolist()
    assert entering == [0, 0, 2, 2, 4]
    assert first == [2009] * 5


def test_career_axes_immune_to_future_season_mutations() -> None:
    base = screen.compute_career_axes(_starters_frame())
    mutated = _starters_frame()
    mutated.loc[mutated["season"] >= 2010, "cpoe_mean"] = 99.0
    extra = pd.DataFrame(
        {
            "game_id": ["g6"],
            "team": ["ARI"],
            "qb_id": ["q1"],
            "qb_name": ["Q One"],
            "season": [2012],
            "week": [1],
            "cpoe_mean": [-99.0],
        }
    )
    mutated_with_future = screen.compute_career_axes(pd.concat([mutated, extra]))
    early_base = base.loc[base["season"] <= 2011].sort_values("game_id")
    early_new = (
        mutated_with_future.loc[mutated_with_future["season"] <= 2011]
        .sort_values("game_id")["career_starts_entering"]
        .tolist()
    )
    assert early_base["career_starts_entering"].tolist() == early_new


def _cell_flags(long_df: pd.DataFrame) -> dict[str, list[bool]]:
    return {
        "is_first_year": (
            long_df["career_starts_entering"] <= screen.FIRST_YEAR_MAX_STARTS
        ).tolist(),
        "is_veteran": (long_df["career_starts_entering"] >= screen.VETERAN_MIN_STARTS).tolist(),
        "is_second_year": (long_df["first_start_season"] == long_df["season"] - 1).tolist(),
    }


def test_cell_flags_do_not_read_outcome_columns() -> None:
    long_df = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "season": [2020, 2020, 2020],
            "week": [3, 10, 14],
            "week_block": [202003, 202010, 202014],
            "team": ["ARI", "BUF", "CHI"],
            "team_covered": [1.0, 0.0, 1.0],
            "career_starts_entering": [2, 210, 210],
            "first_start_season": [2020, 2005, 2005],
            "prior_pressure": [0.02, -0.01, 0.05],
        }
    )
    scrambled = long_df.copy()
    scrambled["team_covered"] = [0.0, 1.0, 0.0]
    assert _cell_flags(long_df) == _cell_flags(scrambled)
    first_year = _cell_flags(long_df)["is_first_year"]
    late_rookie = [
        f and (w >= screen.ROOKIE_LATE_WEEK_MIN)
        for f, w in zip(first_year, long_df["week"].tolist(), strict=True)
    ]
    assert late_rookie == [False, False, False]
    veteran = _cell_flags(long_df)["is_veteran"]
    veteran_late = [
        f and (w >= screen.VETERAN_LATE_WEEK_MIN)
        for f, w in zip(veteran, long_df["week"].tolist(), strict=True)
    ]
    assert veteran_late == [False, False, True]
    assert _cell_flags(long_df)["is_second_year"] == [False, False, False]


def test_pressure_panel_is_prior_season_join_only() -> None:
    plays = pd.DataFrame(
        {
            "season": [2015],
            "season_type": ["REG"],
            "posteam": ["ARI"],
            "defteam": ["BUF"],
            "qb_dropback": [4.0],
            "sack": [1.0],
            "qb_hit": [2.0],
        }
    )
    panel = screen._pressure_panel_from_plays(plays)
    assert len(panel) == 1
    row = panel.iloc[0]
    assert row["team"] == "BUF"
    assert row["dropbacks"] == 4
    assert row["pressure_rate_allowed"] == pytest.approx(3.0 / 4.0)
