"""Tests for the PBP-08 protection-mismatch flags and tilt overlay."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.pbp08_matchup_flags import (
    MIN_QUANTILE_POOL,
    QUARTILE_BOTTOM,
    QUARTILE_TOP,
    QUARTILE_UNASSIGNED,
    expanding_quartile_flags,
    flag_summary,
)
from nfl_ats.pbp08_protection_mismatch_tilt_overlay import (
    apply_pbp08_protection_mismatch_tilt,
    overlay_disclosure_note,
)


def _flags(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "season": 2026,
                "week": 1,
                "home_team": "HOM",
                "away_team": "AWY",
                "home_press_allow_w": 0.2,
                "away_press_allow_w": 0.2,
                "home_press_gen_w": 0.2,
                "away_press_gen_w": 0.2,
                "home_offense_flagged": back == "AWAY",
                "away_offense_flagged": back == "HOME",
                "back_side": back,
            }
            for game_id, back in rows
        ]
    )


def _card(probabilities: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": list(probabilities),
            "home_team": ["HOM"] * len(probabilities),
            "away_team": ["AWY"] * len(probabilities),
            "home_cover_probability": list(probabilities.values()),
        }
    )


class TestExpandingQuartiles:
    def test_the_first_block_is_unassigned_because_no_prior_pool_exists(self) -> None:
        values = pd.Series(np.linspace(0.0, 1.0, 50))
        blocks = pd.Series([202601] * 50)
        assert set(expanding_quartile_flags(values, blocks)) == {QUARTILE_UNASSIGNED}

    def test_thresholds_come_only_from_strictly_earlier_blocks(self) -> None:
        prior = np.linspace(0.0, 1.0, MIN_QUANTILE_POOL)
        values = pd.Series(np.concatenate([prior, [0.01, 0.5, 0.99]]))
        blocks = pd.Series([202601] * MIN_QUANTILE_POOL + [202602] * 3)
        codes = expanding_quartile_flags(values, blocks)
        # The prior block itself is unassigned; the later block is scored
        # against the prior block's own quartiles.
        assert set(codes[:MIN_QUANTILE_POOL]) == {QUARTILE_UNASSIGNED}
        assert list(codes[MIN_QUANTILE_POOL:]) == [QUARTILE_BOTTOM, 1, QUARTILE_TOP]

    def test_a_pool_below_the_minimum_leaves_the_next_block_unassigned(self) -> None:
        small = MIN_QUANTILE_POOL - 1
        values = pd.Series(np.concatenate([np.linspace(0, 1, small), [0.5]]))
        blocks = pd.Series([202601] * small + [202602])
        assert expanding_quartile_flags(values, blocks)[-1] == QUARTILE_UNASSIGNED

    def test_a_missing_window_is_never_folded_into_the_complement(self) -> None:
        prior = np.linspace(0.0, 1.0, MIN_QUANTILE_POOL)
        values = pd.Series(np.concatenate([prior, [np.nan]]))
        blocks = pd.Series([202601] * MIN_QUANTILE_POOL + [202602])
        assert expanding_quartile_flags(values, blocks)[-1] == QUARTILE_UNASSIGNED


class TestTilt:
    def test_a_pick_on_the_flagged_offense_flips_to_the_defense(self) -> None:
        card = _card({"g1": 0.62})  # model holds HOME
        result = apply_pbp08_protection_mismatch_tilt(card, _flags([("g1", "AWAY")]))
        assert result.flip_count == 1
        assert result.flips[0].flipped_to_team == "AWY"
        assert result.overlaid_predictions.loc[0, "home_cover_probability"] == pytest.approx(0.38)

    def test_a_pick_already_on_the_defense_is_left_alone(self) -> None:
        card = _card({"g1": 0.62})  # model holds HOME, and HOME is the lean
        result = apply_pbp08_protection_mismatch_tilt(card, _flags([("g1", "HOME")]))
        assert result.flip_count == 0
        assert result.overlaid_predictions.loc[0, "home_cover_probability"] == pytest.approx(0.62)

    def test_a_game_with_no_lean_is_untouched(self) -> None:
        card = _card({"g1": 0.62})
        result = apply_pbp08_protection_mismatch_tilt(card, _flags([("g1", "")]))
        assert result.flip_count == 0
        assert result.overlaid_predictions.loc[0, "home_cover_probability"] == pytest.approx(0.62)

    def test_both_sides_flagged_is_a_mutual_mismatch_and_never_flips(self) -> None:
        flags = _flags([("g1", "")])
        flags.loc[0, ["home_offense_flagged", "away_offense_flagged"]] = [True, True]
        result = apply_pbp08_protection_mismatch_tilt(_card({"g1": 0.62}), flags)
        assert result.flip_count == 0

    def test_the_overlay_is_asymmetric_and_never_moves_a_pick_onto_a_flagged_offense(self) -> None:
        # HOME's offense is the flagged one, so the lean backs AWAY. A model
        # already on AWAY must not be flipped ONTO the flagged offense.
        card = _card({"g1": 0.30})  # model holds AWAY
        result = apply_pbp08_protection_mismatch_tilt(card, _flags([("g1", "AWAY")]))
        assert result.flip_count == 0
        assert result.overlaid_predictions.loc[0, "home_cover_probability"] == pytest.approx(0.30)

    def test_an_empty_flag_table_is_a_documented_no_op_not_a_crash(self) -> None:
        result = apply_pbp08_protection_mismatch_tilt(_card({"g1": 0.62}), pd.DataFrame())
        assert result.flip_count == 0
        assert result.overlaid_predictions.equals(_card({"g1": 0.62}))

    def test_disabled_returns_the_card_unchanged(self) -> None:
        card = _card({"g1": 0.62})
        result = apply_pbp08_protection_mismatch_tilt(card, _flags([("g1", "AWAY")]), enabled=False)
        assert result.flip_count == 0
        assert not result.enabled

    def test_a_card_missing_a_required_column_fails_closed(self) -> None:
        card = _card({"g1": 0.62}).drop(columns=["home_cover_probability"])
        with pytest.raises(DataContractError, match="missing overlay columns"):
            apply_pbp08_protection_mismatch_tilt(card, _flags([("g1", "AWAY")]))

    def test_only_the_probability_column_changes_on_a_flip(self) -> None:
        card = _card({"g1": 0.62})
        card["spread_line"] = -3.5
        card["season"] = 2026
        result = apply_pbp08_protection_mismatch_tilt(card, _flags([("g1", "AWAY")]))
        other = [c for c in card.columns if c != "home_cover_probability"]
        pd.testing.assert_frame_equal(result.overlaid_predictions[other], card[other])

    def test_the_disclosure_note_is_empty_when_nothing_moved(self) -> None:
        result = apply_pbp08_protection_mismatch_tilt(_card({"g1": 0.62}), _flags([("g1", "")]))
        assert overlay_disclosure_note(result) == ""

    def test_the_disclosure_note_names_the_flip_and_says_it_is_not_played(self) -> None:
        result = apply_pbp08_protection_mismatch_tilt(_card({"g1": 0.62}), _flags([("g1", "AWAY")]))
        note = overlay_disclosure_note(result)
        assert "1 pick flipped" in note
        assert "not applied to the published card" in note


class TestFlagSummary:
    def test_counts_partition_the_slate(self) -> None:
        flags = _flags([("g1", "HOME"), ("g2", "AWAY"), ("g3", "")])
        summary = flag_summary(flags)
        assert summary["games"] == 3
        assert summary["backs_home"] + summary["backs_away"] + summary["no_lean"] == 3
