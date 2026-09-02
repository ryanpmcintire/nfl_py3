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

# Same calendar week and kickoff times as tests/test_pick_refresh.py's
# TNF_KICKOFF/SUN_EARLY_KICKOFF/SNF_KICKOFF/MNF_KICKOFF/SUNDAY_LOCK
# (2026 week of Sep 17-21), duplicated as literals here rather than imported
# so this test file does not depend on another test module's private
# fixtures. A Sunday 4:25pm ET (late-afternoon "doubleheader window") game
# is added because it is NOT SNF/MNF but still kicks off after the pool's
# 4:00pm ET lock, so its deadline must also be capped early.
_WEEK_KICKOFFS_UTC = {
    "thursday": pd.Timestamp("2026-09-18T00:15:00+00:00"),  # Thu 8:15pm ET
    "sunday_1pm": pd.Timestamp("2026-09-20T17:00:00+00:00"),  # Sun 1:00pm ET
    "sunday_425pm": pd.Timestamp("2026-09-20T20:25:00+00:00"),  # Sun 4:25pm ET
    "snf": pd.Timestamp("2026-09-21T00:20:00+00:00"),  # Sun 8:20pm ET
    "mnf": pd.Timestamp("2026-09-22T00:15:00+00:00"),  # Mon 8:15pm ET
}
_SUNDAY_LOCK_UTC = pd.Timestamp("2026-09-20T20:00:00+00:00")  # Sun 4:00pm ET


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


def test_pool_rules_composed_fields_match_cited_sources() -> None:
    """POL-01: the pool facts the workbench previously left uncomposed --
    forced-pick card count, grading line, tiebreak rule, and the per-game
    deadline -- are now typed fields with provenance in their docstrings,
    not re-derived or reimplemented."""

    rules = PoolRules.from_defaults()
    # docs/pool_edge_plan.md:76-77 / AGENTS.md "285 cards must be submitted
    # either way" -- cards_per_season is a derived alias of total_games, not
    # a second hardcoded literal.
    assert rules.cards_per_season == rules.total_games == 285
    # docs/pool_edge_plan.md:5, "beat the OPENING line the user's Splash
    # Sports pool grades against".
    assert rules.grading_line == "opener"
    # src/nfl_ats/tiebreaker.py module docstring: "The pool breaks ties on
    # the final score of the week's LAST game".
    assert rules.tiebreak == "final_score_last_game"
    # deadline_rule is the SAME function object as
    # nfl_ats.pick_refresh.pick_deadline -- imported, never reimplemented.
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

    # Thursday and the Sunday 1:00pm ET game lock at their own kickoff --
    # nothing constrains them to 4:00pm ET.
    assert (
        rules.deadline_for(_WEEK_KICKOFFS_UTC["thursday"], all_kickoffs)
        == (_WEEK_KICKOFFS_UTC["thursday"])
    )
    assert (
        rules.deadline_for(_WEEK_KICKOFFS_UTC["sunday_1pm"], all_kickoffs)
        == (_WEEK_KICKOFFS_UTC["sunday_1pm"])
    )

    # The Sunday 4:25pm ET window, SNF, and MNF all lock EARLY at the
    # week's Sunday 16:00 ET cap, even though only SNF/MNF's own kickoff
    # falls on a later calendar day than the cap.
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


def test_ownership_scenarios_are_disclosed_assumptions_not_observations() -> None:
    assert [scenario.favorite_share for scenario in DEFAULT_OWNERSHIP_SCENARIOS] == [
        0.50,
        0.65,
        0.85,
    ]
    summaries = build_ownership_scenarios(pd.DataFrame({"pick_line": [-3.5, 3.5, 0.0]}))
    assert list(summaries["observed"]) == [False, False, False]
    # One favorite pick, one underdog pick, and one pick'em always average
    # to 50% overlap. No crowd-ownership observation is being invented.
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
    # The confirmed total of forced picks is shown.
    assert "285" in body
    # Best Pick is editable in the entry list.
    assert "&#9733;" in body
    # Entry list and confidence ranks were merged into ONE table (owner,
    # 2026-08-26: they showed "identical/duplicated data"): only one heading
    # survives, cover probability renders via the probability meter (taken
    # from the former confidence-ranks table), and its residual-magnitude
    # caveat is preserved in the merged footnote.
    assert body.count("Forced picks, editable entry") == 1
    assert "Confidence ranks" not in body
    assert "Model cover probability" in body
    assert 'aria-label="cover:' in body
    assert "has not proven to rank pick quality" in body
    # UI-09 entry persistence is explicitly browser-local and week-scoped.
    assert f'data-storage-key="nfl-ats:pool-entry:v{ENTRY_STORAGE_VERSION}:2026:1"' in body
    assert body.count('class="entry-pick"') == 4
    assert body.count('class="entry-best"') == 2
    assert "window.localStorage.setItem" in body
    assert "window.localStorage.getItem" in body
    assert "window.localStorage.removeItem" in body
    assert "does not change or publish the model forecast" in body
    # Ownership output is a live sensitivity table, never a fabricated feed.
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
    # The page marks itself current and therefore does not link to itself.
    # No leaked market-feed fields or book names.
    for forbidden in ("home_spread_odds", "total_line", "DraftKings", "-110"):
        assert forbidden not in page


def test_render_pool_workbench_page_empty_state_is_safe() -> None:
    page = render_pool_workbench_page(pd.DataFrame({"game_id": ["x"]}))
    assert "No pick card yet" in page
    assert DISCLAIMER_SHORT in page
    assert '<div class="ats">' in page
