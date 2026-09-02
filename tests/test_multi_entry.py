from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.multi_entry import build_multi_entry_plan, multi_entry_plan_markdown


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2026] * 4,
            "week": [1] * 4,
            "game_id": ["g1", "g2", "g3", "g4"],
            "gameday": ["2026-09-13"] * 4,
            "away_team": ["A1", "A2", "A3", "A4"],
            "home_team": ["H1", "H2", "H3", "H4"],
            "spread_line": [3.5, -2.5, 1.5, -7.5],
            "home_cover_probability": [0.70, 0.40, 0.55, 0.90],
        }
    )


def test_multi_entry_plan_protects_primary_and_controls_every_pairwise_overlap() -> None:
    plan = build_multi_entry_plan(
        _predictions(),
        entry_count=3,
        max_pairwise_overlap=0.5,
    )

    primary = plan.entries.loc[plan.entries["entry_id"].eq(1)]
    assert primary["pool_side"].tolist() == ["HOME", "AWAY", "HOME", "HOME"]
    assert not primary["flipped_from_primary"].any()
    assert plan.overlap["within_limit"].all()
    assert plan.overlap["overlap_rate"].max() <= 0.5
    assert plan.metrics["paper_only"] is True
    assert plan.metrics["minimum_pairwise_disagreements"] == 2


def test_additional_entries_flip_lowest_expected_cost_subject_to_overlap() -> None:
    plan = build_multi_entry_plan(
        _predictions(),
        entry_count=3,
        max_pairwise_overlap=0.5,
    )

    entry_two = plan.entries.loc[plan.entries["entry_id"].eq(2)]
    entry_three = plan.entries.loc[plan.entries["entry_id"].eq(3)]
    assert entry_two.loc[entry_two["flipped_from_primary"], "game_id"].tolist() == ["g2", "g3"]
    assert entry_two["entry_expected_loss_vs_primary"].iloc[0] == pytest.approx(0.30)
    assert entry_three.loc[entry_three["flipped_from_primary"], "game_id"].tolist() == ["g1", "g3"]
    assert entry_three["entry_expected_loss_vs_primary"].iloc[0] == pytest.approx(0.50)


def test_plan_is_deterministic_under_input_order() -> None:
    first = build_multi_entry_plan(_predictions(), entry_count=4, max_pairwise_overlap=0.75)
    second = build_multi_entry_plan(
        _predictions().sample(frac=1.0, random_state=7),
        entry_count=4,
        max_pairwise_overlap=0.75,
    )

    assert first.entries.equals(second.entries)
    assert first.overlap.equals(second.overlap)
    assert first.metrics == second.metrics


def test_flip_and_expected_loss_caps_fail_closed_when_diversification_is_impossible() -> None:
    with pytest.raises(ValueError, match="No multi-entry allocation"):
        build_multi_entry_plan(
            _predictions(),
            entry_count=2,
            max_pairwise_overlap=0.5,
            max_flips_from_primary=1,
        )
    with pytest.raises(ValueError, match="No multi-entry allocation"):
        build_multi_entry_plan(
            _predictions(),
            entry_count=2,
            max_pairwise_overlap=0.5,
            max_expected_correct_loss=0.2,
        )


def test_single_entry_has_empty_overlap_audit_and_exact_expected_score() -> None:
    plan = build_multi_entry_plan(_predictions(), entry_count=1)

    assert plan.overlap.empty
    assert plan.metrics["observed_max_pairwise_overlap"] == 1.0
    assert plan.metrics["baseline_expected_correct"] == pytest.approx(2.75)
    assert plan.metrics["total_expected_correct"] == pytest.approx(2.75)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"entry_count": 0}, "positive integer"),
        ({"entry_count": 2.5}, "positive integer"),
        ({"entry_count": 2, "max_pairwise_overlap": 1.0}, "below 1"),
        ({"entry_count": 2, "max_pairwise_overlap": -0.1}, "must lie"),
        ({"entry_count": 2, "max_flips_from_primary": 5}, "between 0"),
        ({"entry_count": 2, "max_expected_correct_loss": -1.0}, "non-negative"),
    ],
)
def test_rule_contract_rejects_invalid_controls(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_multi_entry_plan(_predictions(), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.drop(columns="spread_line"), "missing multi-entry columns"),
        (lambda frame: frame.assign(game_id=["g1", "g1", "g3", "g4"]), "must be unique"),
        (lambda frame: frame.assign(home_cover_probability=1.1), "must lie"),
        (lambda frame: frame.assign(home_cover_probability=float("nan")), "finite numbers"),
        (lambda frame: frame.assign(spread_line=float("nan")), "spread lines must be finite"),
        (lambda frame: frame.assign(week=[1, 1, 2, 2]), "exactly one week"),
        (lambda frame: frame.assign(home_team=["H1", "H1", "H3", "H4"]), "more than once"),
    ],
)
def test_prediction_contract_rejects_ambiguous_cards(mutate: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_multi_entry_plan(mutate(_predictions()), entry_count=1)  # type: ignore[operator]


def test_weekly_scope_is_explicitly_bounded() -> None:
    predictions = pd.concat([_predictions()] * 5, ignore_index=True)
    predictions["game_id"] = [f"g{index}" for index in range(len(predictions))]
    predictions["home_team"] = [f"H{index}" for index in range(len(predictions))]
    predictions["away_team"] = [f"A{index}" for index in range(len(predictions))]

    with pytest.raises(ValueError, match="at most 18 games"):
        build_multi_entry_plan(predictions, entry_count=2)


def test_markdown_retains_expected_score_and_overlap_audits() -> None:
    plan = build_multi_entry_plan(_predictions(), entry_count=2, max_pairwise_overlap=0.5)
    markdown = multi_entry_plan_markdown(plan)

    assert "Paper pool cards only" in markdown
    assert "expected_correct" in markdown
    assert "Pairwise overlap" in markdown
    assert "within_limit" in markdown
