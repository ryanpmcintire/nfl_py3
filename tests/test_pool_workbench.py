"""Tests for the minimal pool workbench (ROADMAP UI-09, POL-01)."""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.pick_refresh import pick_deadline, sunday_pick_lock
from nfl_ats.pool_workbench import (
    DEFAULT_OWNERSHIP_SCENARIOS,
    ENTRY_STORAGE_VERSION,
    PLAYOFF_GAMES,
    REGULAR_SEASON_GAMES,
    OwnershipScenario,
    PoolRules,
    build_entry_list,
    build_ownership_scenarios,
    build_pool_workbench_body,
    derive_confidence_ranks,
)
from nfl_ats.public_board import (
    DISCLAIMER_FULL,
    DISCLAIMER_SHORT,
    PICKS_PAGE,
    render_pool_workbench_page,
)

_WEEK_KICKOFFS_UTC = {
    "thursday": pd.Timestamp("2026-09-18T00:15:00+00:00"),
    "sunday_1pm": pd.Timestamp("2026-09-20T17:00:00+00:00"),
    "sunday_425pm": pd.Timestamp("2026-09-20T20:25:00+00:00"),
    "snf": pd.Timestamp("2026-09-21T00:20:00+00:00"),
    "mnf": pd.Timestamp("2026-09-22T00:15:00+00:00"),
}
_SUNDAY_LOCK_UTC = pd.Timestamp("2026-09-20T20:00:00+00:00")


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
    assert rules.total_games == REGULAR_SEASON_GAMES + PLAYOFF_GAMES == 285
    assert rules.best_pick_per_regular_season_week == 1
    assert rules.forced_picks is True
    assert rules.passes_allowed is False
    assert rules.line_locks_tuesday is True


def test_pool_rules_from_dict_accepts_partial_overrides() -> None:
    rules = PoolRules.from_dict(
        {
            "best_pick_per_regular_season_week": 2,
            "forced_picks": False,
            "passes_allowed": True,
        }
    )
    assert rules.best_pick_per_regular_season_week == 2
    assert rules.passes_allowed is True
    assert rules.total_games == 285
    assert PoolRules.from_dict({"not_a_field": 99}).total_games == 285


def test_pool_rules_composed_fields_match_cited_sources() -> None:
    """POL-01: the pool facts the workbench previously left uncomposed --
    forced-pick card count, grading line, tiebreak rule, and the per-game
    deadline -- are now typed fields with provenance in their docstrings,
    not re-derived or reimplemented."""

    rules = PoolRules.from_defaults()
    assert rules.cards_per_season == rules.total_games == 285
    assert rules.grading_line == "opener"
    assert rules.tiebreak == "final_score_last_game"
    assert PoolRules.deadline_rule is pick_deadline
    assert rules.deadline_rule is pick_deadline


def test_pool_rules_deadline_for_agrees_with_pick_refresh_on_every_slot() -> None:
    """PoolRules.deadline_for must never diverge from
    nfl_ats.pick_refresh.pick_deadline/sunday_pick_lock: it is a thin
    wrapper, not a second implementation of the owner's per-game deadline
    rule (owner, 2026-08-20, re-confirmed 2026-09-01)."""

    rules = PoolRules.from_defaults()
    all_kickoffs = list(_WEEK_KICKOFFS_UTC.values())

    reference_lock = sunday_pick_lock(pd.Series(all_kickoffs))
    assert reference_lock == _SUNDAY_LOCK_UTC

    for label, kickoff in _WEEK_KICKOFFS_UTC.items():
        got = rules.deadline_for(kickoff, all_kickoffs)
        expected = pick_deadline(kickoff, reference_lock)
        assert got == expected, label

    assert (
        rules.deadline_for(_WEEK_KICKOFFS_UTC["thursday"], all_kickoffs)
        == (_WEEK_KICKOFFS_UTC["thursday"])
    )
    assert (
        rules.deadline_for(_WEEK_KICKOFFS_UTC["sunday_1pm"], all_kickoffs)
        == (_WEEK_KICKOFFS_UTC["sunday_1pm"])
    )

    for label in ("sunday_425pm", "snf", "mnf"):
        assert rules.deadline_for(_WEEK_KICKOFFS_UTC[label], all_kickoffs) == _SUNDAY_LOCK_UTC


def test_pool_rules_describe_is_plain_english_and_cites_the_rules() -> None:
    lines = PoolRules.from_defaults().describe()
    assert isinstance(lines, list)
    assert lines and all(isinstance(line, str) for line in lines)
    joined = " ".join(lines)
    assert "285" in joined
    assert "Best Pick" in joined
    assert "opener" in joined
    assert "16:00 ET" in joined
    assert "final score last game" in joined.lower()


def test_build_entry_list_ranks_by_confidence() -> None:
    card = build_entry_list(_forecast_fixture())
    assert len(card) == 2
    assert list(card["confidence_rank"]) == [1, 2]
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


def test_ownership_scenarios_are_disclosed_assumptions_not_observations() -> None:
    assert [scenario.favorite_share for scenario in DEFAULT_OWNERSHIP_SCENARIOS] == [
        0.50,
        0.65,
        0.85,
    ]
    summaries = build_ownership_scenarios(pd.DataFrame({"pick_line": [-3.5, 3.5, 0.0]}))
    assert list(summaries["observed"]) == [False, False, False]
    assert list(summaries["entry_side_share"]) == pytest.approx([0.5, 0.5, 0.5])
    assert list(summaries["disagreements_per_100"]) == pytest.approx([50.0, 50.0, 50.0])


def test_ownership_scenario_validates_assumption_range() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        OwnershipScenario("bad", "Bad", 1.01, "invalid")
    with pytest.raises(ValueError, match="must not be empty"):
        OwnershipScenario("", "Bad", 0.5, "invalid")


def test_ownership_scenarios_degrade_without_an_entry() -> None:
    empty = build_ownership_scenarios(pd.DataFrame())
    assert empty.empty
    assert "observed" in empty.columns


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
    assert "Ownership scenarios" in body
    assert "285" in body
    assert "&#9733;" in body
    assert body.count("Forced picks, editable entry") == 1
    assert "Confidence ranks" not in body
    assert "Model cover probability" in body
    assert 'aria-label="cover:' in body
    assert "has not proven to rank pick quality" in body
    assert f'data-storage-key="nfl-ats:pool-entry:v{ENTRY_STORAGE_VERSION}:2026:1"' in body
    assert body.count('class="entry-pick"') == 4
    assert body.count('class="entry-best"') == 2
    assert "window.localStorage.setItem" in body
    assert "window.localStorage.getItem" in body
    assert "window.localStorage.removeItem" in body
    assert "does not change or publish the model forecast" in body
    assert body.count("data-ownership-scenario=") == 3
    assert "Sensitivity only — no ownership feed" in body
    assert "not measured popularity" in body
    assert "Contrarian leverage (placeholder)" not in body


def test_unscoped_entry_cannot_collide_in_browser_storage() -> None:
    body = build_pool_workbench_body(_forecast_fixture())
    assert "data-storage-key=" not in body
    assert '<button type="button" id="pool-entry-save" disabled>' in body
    assert "A season and week are required before an entry can be saved." in body


def test_persistence_script_rejects_stale_or_invalid_saved_values() -> None:
    body = build_pool_workbench_body(_forecast_fixture(), season=2026, week=1)
    assert f"state.version !== {ENTRY_STORAGE_VERSION}" in body
    assert 'state.picks[game] === "HOME" || state.picks[game] === "AWAY"' in body
    assert "state.bestPickGameId && known[state.bestPickGameId]" in body
    assert "Saved entry was incompatible and was ignored" in body
    assert "Browser storage is unavailable" in body


def test_build_pool_workbench_body_empty_state_without_forecast() -> None:
    body = build_pool_workbench_body(pd.DataFrame({"game_id": ["x"]}))
    assert "Pool workbench" in body
    assert "No pick card yet" in body
    assert "Ownership scenarios" in body
    assert "No entry to compare" in body
    assert "window.localStorage" not in body


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
    for forbidden in ("home_spread_odds", "total_line", "DraftKings", "-110"):
        assert forbidden not in page


def test_render_pool_workbench_page_empty_state_is_safe() -> None:
    page = render_pool_workbench_page(pd.DataFrame({"game_id": ["x"]}))
    assert "No pick card yet" in page
    assert DISCLAIMER_SHORT in page
    assert '<div class="ats">' in page
