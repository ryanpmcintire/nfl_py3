"""Tests for the minimal pool workbench (ROADMAP UI-09)."""

from __future__ import annotations

import pandas as pd

from nfl_ats.pool_workbench import (
    PLAYOFF_GAMES,
    REGULAR_SEASON_GAMES,
    OwnershipScenario,
    PoolRules,
    build_entry_list,
    build_pool_workbench_body,
    derive_confidence_ranks,
    placeholder_ownership_scenarios,
)
from nfl_ats.public_board import (
    DISCLAIMER_FULL,
    DISCLAIMER_SHORT,
    PICKS_PAGE,
    render_pool_workbench_page,
)


def _forecast_fixture() -> pd.DataFrame:
    """The active model's recommendations.csv forecast format (minimal)."""

    return pd.DataFrame(
        {
            "game_id": ["2026_01_ARI_LAC", "2026_01_SF_LA"],
            "gameday": ["2026-09-13", "2026-09-10"],
            "away_team": ["ARI", "SF"],
            "home_team": ["LAC", "LA"],
            "spread_line": [3.5, -3.5],
            "home_cover_probability": [0.38, 0.62],
        }
    )


def test_pool_rules_defaults_match_the_confirmed_format() -> None:
    rules = PoolRules.from_defaults()
    assert rules.regular_season_games == REGULAR_SEASON_GAMES
    assert rules.playoff_games == PLAYOFF_GAMES
    # 272 + 13 = 285 forced picks (measured, docs/pool_edge_plan.md).
    assert rules.total_games == REGULAR_SEASON_GAMES + PLAYOFF_GAMES == 285
    assert rules.best_pick_per_regular_season_week == 1
    assert rules.forced_picks is True
    assert rules.passes_allowed is False
    assert rules.line_locks_tuesday is True


def test_pool_rules_from_dict_accepts_partial_overrides() -> None:
    rules = PoolRules.from_dict({"best_pick_per_regular_season_week": 2, "passes_allowed": True})
    assert rules.best_pick_per_regular_season_week == 2
    assert rules.passes_allowed is True
    # Untouched fields keep their defaults.
    assert rules.total_games == 285
    # Unknown keys are ignored, not erroring.
    assert PoolRules.from_dict({"not_a_field": 99}).total_games == 285


def test_build_entry_list_ranks_by_confidence() -> None:
    card = build_entry_list(_forecast_fixture())
    assert len(card) == 2
    assert list(card["confidence_rank"]) == [1, 2]
    # Every game gets a forced side; probabilities are calibrated covers.
    assert set(card["pool_side"]) == {"HOME", "AWAY"}
    assert card["pick_probability"].between(0.5, 1.0).all()


def test_build_entry_list_degrades_on_missing_columns() -> None:
    assert build_entry_list(pd.DataFrame({"game_id": ["x"]})).empty
    assert build_entry_list(pd.DataFrame()).empty


def test_derive_confidence_ranks_is_the_ranking_view() -> None:
    ranks = derive_confidence_ranks(_forecast_fixture())
    assert not ranks.empty
    for column in (
        "confidence_rank",
        "gameday",
        "away_team",
        "home_team",
        "pool_pick",
        "pool_side",
        "pick_probability",
        "confidence",
        "game_id",
    ):
        assert column in ranks.columns
    assert list(ranks["confidence_rank"]) == [1, 2]


def test_derive_confidence_ranks_empty_without_forecast() -> None:
    assert derive_confidence_ranks(pd.DataFrame({"game_id": ["x"]})).empty


def test_placeholder_ownership_scenario_is_not_available() -> None:
    scenario = placeholder_ownership_scenarios(best_pick_game_id="2026_01_ARI_LAC")
    assert isinstance(scenario, OwnershipScenario)
    assert scenario.available is False
    assert scenario.best_pick_game_id == "2026_01_ARI_LAC"
    assert "placeholder" in scenario.note.lower()


def test_build_pool_workbench_body_contains_every_section() -> None:
    body = build_pool_workbench_body(
        _forecast_fixture(),
        season=2026,
        week=1,
        best_pick_game_id="2026_01_ARI_LAC",
    )
    assert "Pool workbench" in body
    assert "Pool rules input" in body
    assert "Entry list" in body
    assert "Confidence ranks" in body
    assert "Ownership scenarios" in body
    # The confirmed total of forced picks is shown.
    assert "285" in body
    # Best Pick is badged in the entry list.
    assert "&#9733;" in body


def test_build_pool_workbench_body_empty_state_without_forecast() -> None:
    body = build_pool_workbench_body(pd.DataFrame({"game_id": ["x"]}))
    assert "Pool workbench" in body
    assert "No pick card yet" in body
    assert "Ownership scenarios" in body


def test_render_pool_workbench_page_is_public_safe() -> None:
    page = render_pool_workbench_page(
        _forecast_fixture(),
        season=2026,
        week=1,
        model_id="model-123",
        best_pick_game_id="2026_01_ARI_LAC",
    )
    assert page.startswith("<!doctype html>")
    assert '<meta charset="utf-8">' in page
    assert '<div class="ats">' in page
    assert DISCLAIMER_SHORT in page
    assert DISCLAIMER_FULL in page
    assert 'aria-current="page"' in page
    assert f'href="{PICKS_PAGE}"' in page
    # The page marks itself current and therefore does not link to itself.
    # No leaked market-feed fields or book names.
    for forbidden in ("home_spread_odds", "total_line", "DraftKings", "-110"):
        assert forbidden not in page


def test_render_pool_workbench_page_empty_state_is_safe() -> None:
    page = render_pool_workbench_page(pd.DataFrame({"game_id": ["x"]}))
    assert "No pick card yet" in page
    assert DISCLAIMER_SHORT in page
    assert '<div class="ats">' in page
