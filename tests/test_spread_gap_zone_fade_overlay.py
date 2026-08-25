"""Spread-gap-zone fade overlay (docs/spread_gap_zone_fade_overlay.md).

Three things are load-bearing here, mirroring
``tests/test_division_revenge_tilt_overlay.py``'s structure (adapted: this
overlay has no separate schedule-derived flag function, since the zone is a
pure function of the card's own ``spread_line``):

1. :func:`apply_spread_gap_zone_fade_overlay` flips EVERY forced pick whose
   market line sits in the frozen ``[7.5, 10.0]`` zone, regardless of which
   side was originally picked, respects the REG-only gate, and is
   parameter-free beyond the two frozen bounds.
2. :func:`overlay_disclosure_note` states the flip count and matchups.
3. :func:`record_spread_gap_zone_fade_challenger_decisions` writes the
   overlay's own picks to the prospective challenger ledger, dual-tracked
   and at no rotation-registry window cost.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from _overlay_test_kit import write_active_model_and_card, write_challenger_registry

from nfl_ats.data import DataContractError
from nfl_ats.prospective_scoring import (
    CHALLENGER_DECISION_COLUMNS,
    artifact_model_config,
    config_fingerprint,
    load_challenger_decisions,
)
from nfl_ats.spread_gap_zone_fade_overlay import (
    CHALLENGER_ID,
    SPREAD_GAP_LOWER_BOUND,
    SPREAD_GAP_UPPER_BOUND,
    apply_spread_gap_zone_fade_overlay,
    overlay_disclosure_note,
    record_spread_gap_zone_fade_challenger_decisions,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
#
# G-lower: |spread_line| == 7.5 exactly (the lower bound, inclusive), model
#   picks HOME -- should flip to AWAY.
# G-upper: |spread_line| == 10.0 exactly (the upper bound, inclusive), model
#   picks AWAY -- should flip to HOME.
# G-below: |spread_line| == 7.4, just outside the lower bound -- no flip.
# G-above: |spread_line| == 10.1, just outside the upper bound -- no flip.
# G-post: same in-zone shape as G-lower, but POST season -- REG-only gate.
# G-missing-spread: spread_line is NaN -- treated as no signal (not numeric).


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [
                "2026_05_LOWER_HOME",
                "2026_05_UPPER_AWAY",
                "2026_05_BELOW",
                "2026_05_ABOVE",
                "2026_20_POST_INZONE",
                "2026_05_MISSING_SPREAD",
            ],
            "season": [2026] * 6,
            "week": [5, 5, 5, 5, 20, 5],
            "game_type": ["REG", "REG", "REG", "REG", "POST", "REG"],
            "home_team": ["HOMEL", "HOMEU", "HOMEB", "HOMEA", "HOMEP", "HOMEM"],
            "away_team": ["AWAYL", "AWAYU", "AWAYB", "AWAYA", "AWAYP", "AWAYM"],
            "kickoff": ["2026-10-08T17:00:00+00:00"] * 6,
            "spread_line": [-7.5, 10.0, -7.4, 10.1, -8.0, math.nan],
            "home_cover_probability": [0.60, 0.30, 0.55, 0.45, 0.65, 0.55],
        }
    )


# ---------------------------------------------------------------------------
# 1. apply_spread_gap_zone_fade_overlay: the pick-level transform
# ---------------------------------------------------------------------------


def test_frozen_bounds_are_the_predeclared_seven_point_five_and_ten() -> None:
    assert SPREAD_GAP_LOWER_BOUND == 7.5
    assert SPREAD_GAP_UPPER_BOUND == 10.0


def test_overlay_flips_at_the_lower_bound_inclusive_from_a_home_pick() -> None:
    result = apply_spread_gap_zone_fade_overlay(_predictions())

    flipped_ids = {flip.game_id for flip in result.flips}
    assert "2026_05_LOWER_HOME" in flipped_ids
    flip = next(f for f in result.flips if f.game_id == "2026_05_LOWER_HOME")
    assert flip.original_pick_team == "HOMEL"
    assert flip.flipped_to_team == "AWAYL"

    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_05_LOWER_HOME", "home_cover_probability"] == pytest.approx(0.40)


def test_overlay_flips_at_the_upper_bound_inclusive_from_an_away_pick() -> None:
    result = apply_spread_gap_zone_fade_overlay(_predictions())

    flipped_ids = {flip.game_id for flip in result.flips}
    assert "2026_05_UPPER_AWAY" in flipped_ids
    flip = next(f for f in result.flips if f.game_id == "2026_05_UPPER_AWAY")
    assert flip.original_pick_team == "AWAYU"
    assert flip.flipped_to_team == "HOMEU"

    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_05_UPPER_AWAY", "home_cover_probability"] == pytest.approx(0.70)


def test_overlay_does_not_flip_just_below_the_lower_bound() -> None:
    result = apply_spread_gap_zone_fade_overlay(_predictions())
    assert all(flip.game_id != "2026_05_BELOW" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_05_BELOW", "home_cover_probability"] == pytest.approx(0.55)


def test_overlay_does_not_flip_just_above_the_upper_bound() -> None:
    result = apply_spread_gap_zone_fade_overlay(_predictions())
    assert all(flip.game_id != "2026_05_ABOVE" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_05_ABOVE", "home_cover_probability"] == pytest.approx(0.45)


def test_overlay_leaves_postseason_games_untouched() -> None:
    """Same in-zone shape as G-lower, but POST season -- the REG-only gate
    blocks it."""

    result = apply_spread_gap_zone_fade_overlay(_predictions())
    assert all(flip.game_id != "2026_20_POST_INZONE" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_20_POST_INZONE", "home_cover_probability"] == pytest.approx(0.65)


def test_overlay_treats_a_missing_spread_line_as_no_signal() -> None:
    result = apply_spread_gap_zone_fade_overlay(_predictions())
    assert all(flip.game_id != "2026_05_MISSING_SPREAD" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_05_MISSING_SPREAD", "home_cover_probability"] == pytest.approx(0.55)


def test_overlay_disabled_is_a_no_op() -> None:
    predictions = _predictions()
    result = apply_spread_gap_zone_fade_overlay(predictions, enabled=False)

    assert result.flip_count == 0
    pd.testing.assert_frame_equal(
        result.overlaid_predictions.reset_index(drop=True),
        predictions.reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_changes_only_home_cover_probability_on_flipped_rows() -> None:
    """Additivity: every other column, and every untouched row, stays
    byte-identical -- the pick-level design's whole point."""

    predictions = _predictions()
    result = apply_spread_gap_zone_fade_overlay(predictions)
    overlaid = result.overlaid_predictions

    assert list(overlaid.columns) == list(predictions.columns)
    other_columns = [c for c in predictions.columns if c != "home_cover_probability"]
    pd.testing.assert_frame_equal(
        overlaid[other_columns].reset_index(drop=True),
        predictions[other_columns].reset_index(drop=True),
        check_exact=True,
    )
    untouched = predictions["game_id"].isin(
        [
            "2026_05_BELOW",
            "2026_05_ABOVE",
            "2026_20_POST_INZONE",
            "2026_05_MISSING_SPREAD",
        ]
    )
    pd.testing.assert_series_equal(
        overlaid.loc[untouched, "home_cover_probability"].reset_index(drop=True),
        predictions.loc[untouched, "home_cover_probability"].reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_requires_its_prediction_columns() -> None:
    with pytest.raises(DataContractError, match="overlay columns"):
        apply_spread_gap_zone_fade_overlay(pd.DataFrame({"game_id": ["G1"]}))


# ---------------------------------------------------------------------------
# 2. overlay_disclosure_note: the plain-English provenance sentence
# ---------------------------------------------------------------------------


def test_disclosure_note_is_empty_when_nothing_flipped() -> None:
    out_of_zone_only = _predictions().loc[lambda frame: frame["game_id"].eq("2026_05_BELOW")]
    result = apply_spread_gap_zone_fade_overlay(out_of_zone_only)
    assert overlay_disclosure_note(result) == ""

    disabled = apply_spread_gap_zone_fade_overlay(_predictions(), enabled=False)
    assert overlay_disclosure_note(disabled) == ""


def test_disclosure_note_states_the_flip_count_and_does_not_claim_production() -> None:
    result = apply_spread_gap_zone_fade_overlay(_predictions())
    note = overlay_disclosure_note(result)

    assert "Tilt applied: 2 picks flipped" in note
    assert "HOMEL -> AWAYL" in note
    assert "AWAYU -> HOMEU" in note
    assert "not applied to the published card" in note


# ---------------------------------------------------------------------------
# 3. record_spread_gap_zone_fade_challenger_decisions: dual-tracked, no window
# ---------------------------------------------------------------------------

_MODEL_CONFIG = {
    "method": "market_residual",
    "target": "market_residual",
    "regressor": "ridge",
    "ridge_alpha": 10.0,
    "calibration_method": "none",
    "feature_profile": "weak_stack",
    "min_edge": 0.02,
    "min_train_games": 500,
    "feature_table": "data/processed/game_features_weak_stack.parquet",
}


def _recorder_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2026_05_LOWER_HOME", "2026_05_BELOW"],
            "season": [2026, 2026],
            "week": [5, 5],
            "game_type": ["REG", "REG"],
            "home_team": ["HOMEL", "HOMEB"],
            "away_team": ["AWAYL", "AWAYB"],
            "kickoff": ["2026-10-08T17:00:00+00:00", "2026-10-08T17:00:00+00:00"],
            "spread_line": [-7.5, -7.4],
            "home_cover_probability": [0.60, 0.55],
        }
    )


def _write_registry(artifacts: Path, *, status: str = "ACTIVE_PROSPECTIVE") -> None:
    write_challenger_registry(
        artifacts, challenger_id=CHALLENGER_ID, model_config=_MODEL_CONFIG, status=status
    )


_SEASON, _WEEK, _CREATED_AT_UTC = 2026, 5, "2026-10-01T15:00:00+00:00"


def _write_active_model_and_card(artifacts: Path, *, ridge_alpha: float = 10.0) -> None:
    write_active_model_and_card(
        artifacts,
        season=_SEASON,
        week=_WEEK,
        created_at_utc=_CREATED_AT_UTC,
        ridge_alpha=ridge_alpha,
        recommendations=_recorder_predictions(),
    )


def test_record_fade_challenger_decisions_records_the_fade_arm(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    # data_root is accepted for call-signature parity but never read -- see
    # the module docstring -- so an unused, non-existent path is sufficient.
    data_root = tmp_path / "data"
    now = datetime(2026, 10, 4, 16, 0, tzinfo=UTC)

    result = record_spread_gap_zone_fade_challenger_decisions(artifacts, data_root, now=now)

    assert result["recorded"] == 2
    assert result["flip_count"] == 1
    assert result["flipped_game_ids"] == ["2026_05_LOWER_HOME"]

    ledger = load_challenger_decisions(artifacts).set_index("game_id")
    assert list(load_challenger_decisions(artifacts).columns) == list(CHALLENGER_DECISION_COLUMNS)
    assert (ledger["bet_side"] == "PASS").all()
    assert ledger["edge"].isna().all()

    # The fade's own arm diverges from the active model's raw pick (0.60 ->
    # HOME): the fade flips it to AWAY, since the game sits in the zone.
    assert ledger.loc["2026_05_LOWER_HOME", "pick_side"] == "AWAY"
    # The out-of-zone game keeps the model's own HOME pick untouched.
    assert ledger.loc["2026_05_BELOW", "pick_side"] == "HOME"

    # Re-running is a no-op: append-only, never rewrites.
    again = record_spread_gap_zone_fade_challenger_decisions(artifacts, data_root, now=now)
    assert again["recorded"] == 0
    assert again["already_recorded"] == 2


def test_record_fade_challenger_refuses_outside_recording_lock_window(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = tmp_path / "data"

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_spread_gap_zone_fade_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 8, 18, 1, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_fade_challenger_refuses_a_fingerprint_mismatch(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    # The active model's OWN configuration moved (a promotion) since this
    # challenger was pinned -- recording must refuse, not silently switch
    # base models under the same challenger id.
    _write_active_model_and_card(artifacts, ridge_alpha=1.0)
    data_root = tmp_path / "data"

    with pytest.raises(DataContractError, match="configuration fingerprint"):
        record_spread_gap_zone_fade_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 10, 4, 16, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_fade_challenger_refuses_an_inactive_registration(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts, status="CLOSED_BEFORE_ACTIVATION")
    _write_active_model_and_card(artifacts)
    data_root = tmp_path / "data"

    with pytest.raises(ValueError, match="only ACTIVE_PROSPECTIVE"):
        record_spread_gap_zone_fade_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 10, 4, 16, 0, tzinfo=UTC)
        )


def test_fade_fingerprint_helper_agrees_with_the_registered_model_block() -> None:
    """Sanity check that the fixture's config really matches CONFIG_FINGERPRINT_KEYS."""

    metadata = {
        "ats_method": "market_residual",
        "regressor": "ridge",
        "ridge_alpha": 10.0,
        "calibration_method": "none",
        "feature_profile": "weak_stack",
        "min_edge": 0.02,
        "min_train_games": 500,
        "provenance": {
            "feature_table": {"path": "data/processed/game_features_weak_stack.parquet"}
        },
    }
    assert config_fingerprint(artifact_model_config(metadata)) == config_fingerprint(_MODEL_CONFIG)
