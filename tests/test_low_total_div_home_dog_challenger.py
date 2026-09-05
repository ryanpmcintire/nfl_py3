"""Low-total divisional home-dog challenger (LEAD-42,
docs/schedule_flag_battery.md Wave 2 section 6).

Mirrors ``tests/test_spread_gap_zone_fade_overlay.py``'s structure (the
closest precedent: no separate schedule-derived flag function, since
eligibility is a pure function of the card's own ``div_game``/``total_line``/
``spread_line`` columns).

Three things are load-bearing here:

1. :func:`apply_low_total_div_home_dog_overlay` flips ONLY the clean case
   (away pick, divisional, decision total <= 42, home is the underdog),
   respects the REG-only gate, and is deliberately ASYMMETRIC (never flips a
   HOME pick to AWAY).
2. :func:`overlay_disclosure_note` states the flip count and matchups.
3. :func:`record_low_total_div_home_dog_challenger_decisions` writes the
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
from nfl_ats.low_total_div_home_dog_challenger import (
    CHALLENGER_ID,
    LOW_TOTAL_MAX,
    apply_low_total_div_home_dog_overlay,
    overlay_disclosure_note,
    record_low_total_div_home_dog_challenger_decisions,
)
from nfl_ats.prospective_scoring import (
    CHALLENGER_DECISION_COLUMNS,
    artifact_model_config,
    config_fingerprint,
    load_challenger_decisions,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
#
# G-clean: divisional, total 40 (<=42), home dog by 3, model picks AWAY ->
#   flips to HOME.
# G-nondiv: same total/spread shape but NOT divisional -> no flip.
# G-hightotal: divisional, home dog, but total 43 (>42) -> no flip.
# G-homefav: divisional, low total, but home is the FAVORITE (spread +3),
#   not the dog -> no flip.
# G-alreadyhome: divisional, low total, home dog, but model already picks
#   HOME -> already correct, untouched.
# G-post: same in-zone shape as G-clean but POST season -> REG-only gate.
# G-missingtotal: total_line is NaN -> treated as not eligible.


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [
                "2026_05_CLEAN",
                "2026_05_NONDIV",
                "2026_05_HIGHTOTAL",
                "2026_05_HOMEFAV",
                "2026_05_ALREADYHOME",
                "2026_20_POST",
                "2026_05_MISSINGTOTAL",
            ],
            "season": [2026] * 7,
            "week": [5, 5, 5, 5, 5, 20, 5],
            "game_type": ["REG", "REG", "REG", "REG", "REG", "POST", "REG"],
            "home_team": ["HOMEC", "HOMEN", "HOMEH", "HOMEF", "HOMEA", "HOMEP", "HOMEM"],
            "away_team": ["AWAYC", "AWAYN", "AWAYH", "AWAYF", "AWAYA", "AWAYP", "AWAYM"],
            "kickoff": ["2026-10-08T17:00:00+00:00"] * 7,
            "div_game": [1, 0, 1, 1, 1, 1, 1],
            "total_line": [40.0, 40.0, 43.0, 40.0, 40.0, 40.0, math.nan],
            "spread_line": [-3.0, -3.0, -3.0, 3.0, -3.0, -3.0, -3.0],
            "home_cover_probability": [0.35, 0.35, 0.35, 0.35, 0.60, 0.35, 0.35],
        }
    )


# ---------------------------------------------------------------------------
# 1. apply_low_total_div_home_dog_overlay: the pick-level transform
# ---------------------------------------------------------------------------


def test_frozen_total_max_is_forty_two() -> None:
    assert LOW_TOTAL_MAX == 42.0


def test_overlay_flips_away_to_home_on_the_clean_case() -> None:
    result = apply_low_total_div_home_dog_overlay(_predictions())

    flipped_ids = {flip.game_id for flip in result.flips}
    assert "2026_05_CLEAN" in flipped_ids
    flip = next(f for f in result.flips if f.game_id == "2026_05_CLEAN")
    assert flip.total_line == pytest.approx(40.0)
    assert flip.spread_line == pytest.approx(-3.0)

    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_05_CLEAN", "home_cover_probability"] == pytest.approx(0.65)


def test_overlay_does_not_flip_a_non_divisional_game() -> None:
    result = apply_low_total_div_home_dog_overlay(_predictions())
    assert all(flip.game_id != "2026_05_NONDIV" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_05_NONDIV", "home_cover_probability"] == pytest.approx(0.35)


def test_overlay_does_not_flip_above_the_total_threshold() -> None:
    result = apply_low_total_div_home_dog_overlay(_predictions())
    assert all(flip.game_id != "2026_05_HIGHTOTAL" for flip in result.flips)


def test_overlay_does_not_flip_when_home_is_the_favorite() -> None:
    result = apply_low_total_div_home_dog_overlay(_predictions())
    assert all(flip.game_id != "2026_05_HOMEFAV" for flip in result.flips)


def test_overlay_leaves_an_already_home_pick_untouched() -> None:
    result = apply_low_total_div_home_dog_overlay(_predictions())
    assert all(flip.game_id != "2026_05_ALREADYHOME" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_05_ALREADYHOME", "home_cover_probability"] == pytest.approx(0.60)


def test_overlay_leaves_postseason_games_untouched() -> None:
    result = apply_low_total_div_home_dog_overlay(_predictions())
    assert all(flip.game_id != "2026_20_POST" for flip in result.flips)


def test_overlay_treats_a_missing_total_line_as_not_eligible() -> None:
    result = apply_low_total_div_home_dog_overlay(_predictions())
    assert all(flip.game_id != "2026_05_MISSINGTOTAL" for flip in result.flips)


def test_overlay_never_flips_a_home_pick_to_away() -> None:
    """Deliberately asymmetric: eligibility only ever moves the pick TOWARD
    the home dog, never off it."""

    predictions = _predictions()
    predictions.loc[predictions["game_id"].eq("2026_05_CLEAN"), "home_cover_probability"] = 0.55
    result = apply_low_total_div_home_dog_overlay(predictions)
    assert all(flip.game_id != "2026_05_CLEAN" for flip in result.flips)


def test_overlay_disabled_is_a_no_op() -> None:
    predictions = _predictions()
    result = apply_low_total_div_home_dog_overlay(predictions, enabled=False)
    assert result.flip_count == 0
    pd.testing.assert_frame_equal(
        result.overlaid_predictions.reset_index(drop=True),
        predictions.reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_changes_only_home_cover_probability_on_flipped_rows() -> None:
    predictions = _predictions()
    result = apply_low_total_div_home_dog_overlay(predictions)
    overlaid = result.overlaid_predictions
    assert list(overlaid.columns) == list(predictions.columns)
    other_columns = [c for c in predictions.columns if c != "home_cover_probability"]
    pd.testing.assert_frame_equal(
        overlaid[other_columns].reset_index(drop=True),
        predictions[other_columns].reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_requires_its_prediction_columns() -> None:
    with pytest.raises(DataContractError, match="overlay columns"):
        apply_low_total_div_home_dog_overlay(pd.DataFrame({"game_id": ["G1"]}))


# ---------------------------------------------------------------------------
# 2. overlay_disclosure_note
# ---------------------------------------------------------------------------


def test_disclosure_note_is_empty_when_nothing_flipped() -> None:
    only_nondiv = _predictions().loc[lambda frame: frame["game_id"].eq("2026_05_NONDIV")]
    result = apply_low_total_div_home_dog_overlay(only_nondiv)
    assert overlay_disclosure_note(result) == ""

    disabled = apply_low_total_div_home_dog_overlay(_predictions(), enabled=False)
    assert overlay_disclosure_note(disabled) == ""


def test_disclosure_note_states_the_flip_count_and_does_not_claim_production() -> None:
    result = apply_low_total_div_home_dog_overlay(_predictions())
    note = overlay_disclosure_note(result)
    assert "Tilt applied: 1 pick flipped" in note
    assert "AWAY -> HOME" in note
    assert "not applied to the published card" in note


# ---------------------------------------------------------------------------
# 3. record_low_total_div_home_dog_challenger_decisions
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
            "game_id": ["2026_05_CLEAN", "2026_05_NONDIV"],
            "season": [2026, 2026],
            "week": [5, 5],
            "game_type": ["REG", "REG"],
            "home_team": ["HOMEC", "HOMEN"],
            "away_team": ["AWAYC", "AWAYN"],
            "kickoff": ["2026-10-08T17:00:00+00:00", "2026-10-08T17:00:00+00:00"],
            "div_game": [1, 0],
            "total_line": [40.0, 40.0],
            "spread_line": [-3.0, -3.0],
            "home_cover_probability": [0.35, 0.35],
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


def test_record_challenger_decisions_records_the_overlay_arm(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    # data_root is accepted for call-signature parity but never read -- see
    # the module docstring -- so an unused, non-existent path is sufficient.
    data_root = tmp_path / "data"
    now = datetime(2026, 10, 4, 16, 0, tzinfo=UTC)

    result = record_low_total_div_home_dog_challenger_decisions(artifacts, data_root, now=now)

    assert result["recorded"] == 2
    assert result["flip_count"] == 1
    assert result["flipped_game_ids"] == ["2026_05_CLEAN"]

    ledger = load_challenger_decisions(artifacts).set_index("game_id")
    assert list(load_challenger_decisions(artifacts).columns) == list(CHALLENGER_DECISION_COLUMNS)
    assert (ledger["bet_side"] == "PASS").all()
    assert ledger["edge"].isna().all()

    assert ledger.loc["2026_05_CLEAN", "pick_side"] == "HOME"
    assert ledger.loc["2026_05_NONDIV", "pick_side"] == "AWAY"
    paired_path = artifacts / "prospective" / f"{CHALLENGER_ID}_paired_decisions.parquet"
    paired = pd.read_parquet(paired_path).set_index("game_id")
    assert paired["baseline_pick_side"].tolist() == ["AWAY", "AWAY"]
    assert paired["pick_side"].tolist() == ["HOME", "AWAY"]
    paired_bytes = paired_path.read_bytes()

    # Re-running is a no-op: append-only, never rewrites.
    again = record_low_total_div_home_dog_challenger_decisions(artifacts, data_root, now=now)
    assert again["recorded"] == 0
    assert again["already_recorded"] == 2
    assert paired_path.read_bytes() == paired_bytes


def test_missing_decision_source_skips_without_a_ledger(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    card_path = next((artifacts / "margin_predictions").glob("*/recommendations.csv"))
    card = pd.read_csv(card_path).drop(columns=["total_line"])
    card.to_csv(card_path, index=False)
    before = card_path.read_bytes()
    result = record_low_total_div_home_dog_challenger_decisions(
        artifacts, tmp_path / "data", now=datetime(2026, 10, 4, 16, tzinfo=UTC)
    )
    assert result["recorded"] == 0
    assert result["skipped"] is True
    assert load_challenger_decisions(artifacts).empty
    assert card_path.read_bytes() == before


def test_record_challenger_refuses_outside_recording_lock_window(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = tmp_path / "data"

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_low_total_div_home_dog_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 8, 18, 1, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_challenger_refuses_a_fingerprint_mismatch(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts, ridge_alpha=1.0)
    data_root = tmp_path / "data"

    with pytest.raises(DataContractError, match="configuration fingerprint"):
        record_low_total_div_home_dog_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 10, 4, 16, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_challenger_refuses_an_inactive_registration(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts, status="CLOSED_BEFORE_ACTIVATION")
    _write_active_model_and_card(artifacts)
    data_root = tmp_path / "data"

    with pytest.raises(ValueError, match="only ACTIVE_PROSPECTIVE"):
        record_low_total_div_home_dog_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 10, 4, 16, 0, tzinfo=UTC)
        )


def test_fingerprint_helper_agrees_with_the_registered_model_block() -> None:
    """Sanity check that the fixture's config really matches the recorder's
    own fingerprint computation."""

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


# ---------------------------------------------------------------------------
# 4. Registration self-consistency (the TRACKED registry entry)
# ---------------------------------------------------------------------------


def test_real_registry_entry_fingerprint_is_internally_consistent() -> None:
    import json

    registry_path = (
        Path(__file__).resolve().parents[1] / "artifacts" / "prospective" / "challengers.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = [
        entry for entry in registry["challengers"] if entry.get("challenger_id") == CHALLENGER_ID
    ]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["status"] == "ACTIVE_PROSPECTIVE"
    assert entry["config_fingerprint"] == config_fingerprint(entry["model"])
    assert "nfl-ats publish-predictions --record-decisions" in entry["weekly_recording_command"]
