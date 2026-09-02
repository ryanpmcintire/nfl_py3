from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.pool_workbench import PoolRules


def test_default_rules_remain_the_real_forced_pick_ats_format() -> None:
    rules = PoolRules.from_defaults()

    assert rules.pick_type == "ats"
    assert rules.pool_type == "standard"
    assert rules.scoring_method == "correct_picks"
    assert rules.entry_count == 1
    assert rules.submissions_per_season == rules.cards_per_season == 285
    assert rules.forced_picks
    assert not rules.passes_allowed
    assert rules.grading_line == "opener"
    assert rules.line_locks_tuesday


def test_straight_up_rules_change_target_without_reusing_spread_grading() -> None:
    rules = PoolRules.straight_up(entry_count=3)

    assert rules.pick_type == "straight_up"
    assert rules.pool_type == "standard"
    assert rules.grading_line == "result"
    assert not rules.line_locks_tuesday
    assert rules.entry_count == 3
    assert rules.submissions_per_season == 855
    assert "straight-up game result" in " ".join(rules.describe())


@pytest.mark.parametrize("pick_type", ["ats", "straight_up"])
def test_confidence_rules_assign_every_weekly_rank_once(pick_type: str) -> None:
    rules = PoolRules.confidence(pick_type=pick_type, entry_count=2)

    assert rules.pool_type == "confidence"
    assert rules.pick_type == pick_type
    assert rules.scoring_method == "confidence_points"
    assert rules.confidence_assignment == "unique_1_to_game_count"
    assert rules.best_pick_per_regular_season_week == 0
    assert rules.submissions_per_season == 570
    assert "1 through the week's game count once" in " ".join(rules.describe())


def test_survivor_rules_encode_one_team_use_and_lives() -> None:
    rules = PoolRules.survivor(entry_count=4, survivor_lives=2)

    assert rules.pick_type == "straight_up"
    assert rules.pool_type == "survivor"
    assert rules.scoring_method == "survival"
    assert rules.team_use_limit == 1
    assert rules.survivor_lives == 2
    assert rules.submissions_per_season is None
    assert "each team usable 1 time" in " ".join(rules.describe())


def test_all_variants_keep_the_real_per_game_deadline_function() -> None:
    kickoff = pd.Timestamp("2026-09-21T00:20:00+00:00")
    week = [
        pd.Timestamp("2026-09-18T00:15:00+00:00"),
        pd.Timestamp("2026-09-20T17:00:00+00:00"),
        kickoff,
    ]
    expected = PoolRules.from_defaults().deadline_for(kickoff, week)

    for rules in (
        PoolRules.straight_up(),
        PoolRules.confidence(),
        PoolRules.confidence(pick_type="straight_up"),
        PoolRules.survivor(),
    ):
        assert rules.deadline_for(kickoff, week) == expected


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pick_type": "totals"}, "pick_type"),
        ({"pool_type": "bracket"}, "pool_type"),
        ({"entry_count": 0}, "positive integer"),
        ({"entry_count": 1.5}, "positive integer"),
        ({"forced_picks": True, "passes_allowed": True}, "cannot both"),
        ({"grading_line": "halftime"}, "grading_line"),
        ({"pick_type": "straight_up"}, "grading_line='result'"),
        ({"grading_line": "result"}, "opener or close"),
        ({"survivor_lives": 2}, "only configurable"),
        ({"pool_type": "confidence"}, "scoring_method"),
        ({"confidence_assignment": "unique_1_to_game_count"}, "only valid"),
        ({"team_use_limit": 1}, "only valid"),
    ],
)
def test_invalid_generic_rule_combinations_fail_closed(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PoolRules(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("rules", "message"),
    [
        (
            lambda: PoolRules.survivor(pick_type="ats"),
            "opener or close",
        ),
        (
            lambda: PoolRules.survivor(team_use_limit=2),
            "team_use_limit=1",
        ),
        (
            lambda: PoolRules.survivor(best_pick_per_regular_season_week=1),
            "Best Pick",
        ),
        (
            lambda: PoolRules.confidence(confidence_assignment="none"),
            "unique_1_to_game_count",
        ),
    ],
)
def test_invalid_variant_rules_fail_closed(rules: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        rules()  # type: ignore[operator]


def test_from_dict_infers_coherent_variant_defaults_and_ignores_unknown_keys() -> None:
    confidence = PoolRules.from_dict(
        {"pool_type": "confidence", "pick_type": "straight_up", "entry_count": 2}
    )
    survivor = PoolRules.from_dict({"pool_type": "survivor", "not_a_rule": True})

    assert confidence == PoolRules.confidence(pick_type="straight_up", entry_count=2)
    assert survivor == PoolRules.survivor()
