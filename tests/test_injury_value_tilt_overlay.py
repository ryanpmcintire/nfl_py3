"""Injury value-lost tilt overlay (docs/injury_value_lost_tilt_overlay.md).

Three things are load-bearing here, mirroring
``tests/test_coach_fade_overlay.py``'s structure and AGENTS.md's "add a
leakage regression test for every new feature family" spirit:

1. :func:`raw_value_lost_diff` reads exactly the two columns
   ``docs/injury_value_lost.md`` section 4 isolated, and only those.
2. :func:`apply_injury_value_tilt_overlay` flips ONLY the clean-disagreement
   case (the model's own pick sits on the side that lost STRICTLY MORE
   injury value), touches no other column, respects the REG-only gate, and
   is parameter-free -- there is no threshold to pin.
3. :func:`record_injury_value_tilt_challenger_decisions` writes the tilt's
   own picks (which can diverge from the active model's raw picks on a
   flipped game) to the prospective challenger ledger, dual-tracked and at
   no rotation-registry window cost.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.active_model import ACTIVE_ATS_MODEL_VERSION
from nfl_ats.data import DataContractError
from nfl_ats.injury_value_tilt_overlay import (
    CHALLENGER_ID,
    apply_injury_value_tilt_overlay,
    overlay_disclosure_note,
    raw_value_lost_diff,
    record_injury_value_tilt_challenger_decisions,
)
from nfl_ats.prospective_scoring import (
    CHALLENGER_DECISION_COLUMNS,
    artifact_model_config,
    config_fingerprint,
    load_challenger_decisions,
)
from nfl_ats.surgical_gating import VALUE_LOST_DIFF_COLUMNS

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
#
# G1: home lost strictly more value, model picks home -> clean disagreement,
#     flips to away.
# G2: away lost strictly more value, model picks away -> clean disagreement,
#     flips to home.
# G3: tie (zero differential), model picks home -> no signal, no flip.
# G4: home lost strictly more value, model ALREADY picks away -> agrees,
#     no flip.
# G5: same shape as G1 but POST season -> the REG-only gate blocks it.
# G6: no feature row at all (missing from the merge) -> treated as a tie.


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["G1", "G2", "G3", "G4", "G5"],
            "diff_injury_skill_epa_value_lost": [3.0, -2.0, 0.0, 5.0, 10.0],
            "diff_injury_defense_disruption_value_lost": [1.0, -1.5, 0.0, 0.0, 0.0],
        }
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["G1", "G2", "G3", "G4", "G5", "G6"],
            "season": [2026, 2026, 2026, 2026, 2026, 2026],
            "week": [1, 1, 1, 1, 20, 1],
            "game_type": ["REG", "REG", "REG", "REG", "POST", "REG"],
            "home_team": ["HM1", "HM2", "HM3", "HM4", "HM5", "HM6"],
            "away_team": ["AW1", "AW2", "AW3", "AW4", "AW5", "AW6"],
            "kickoff": ["2026-09-10T17:00:00+00:00"] * 6,
            "spread_line": [-3.5, 2.5, -1.0, 4.0, -6.0, 1.0],
            "home_cover_probability": [0.7, 0.2, 0.65, 0.3, 0.9, 0.55],
        }
    )


# ---------------------------------------------------------------------------
# 1. raw_value_lost_diff: derived, pregame-safe
# ---------------------------------------------------------------------------


def test_raw_value_lost_diff_sums_the_two_diff_columns() -> None:
    diff = raw_value_lost_diff(_features()).set_index("game_id")
    assert diff.loc["G1", "value_lost_diff"] == pytest.approx(4.0)
    assert diff.loc["G2", "value_lost_diff"] == pytest.approx(-3.5)
    assert diff.loc["G3", "value_lost_diff"] == pytest.approx(0.0)


def test_raw_value_lost_diff_uses_the_shared_surgical_gating_columns() -> None:
    """Reads exactly ``surgical_gating.VALUE_LOST_DIFF_COLUMNS`` -- never a
    re-typed, potentially-drifted column list."""

    features = _features()[["game_id", *VALUE_LOST_DIFF_COLUMNS]]
    diff = raw_value_lost_diff(features)
    assert len(diff) == len(features)


def test_raw_value_lost_diff_requires_its_columns() -> None:
    with pytest.raises(DataContractError, match="value-lost columns"):
        raw_value_lost_diff(pd.DataFrame({"game_id": ["G1"]}))


# ---------------------------------------------------------------------------
# 2. apply_injury_value_tilt_overlay: the pick-level transform
# ---------------------------------------------------------------------------


def test_overlay_flips_when_home_lost_strictly_more_and_model_picked_home() -> None:
    result = apply_injury_value_tilt_overlay(_predictions(), _features())

    flipped_ids = {flip.game_id for flip in result.flips}
    assert "G1" in flipped_ids
    flip = next(f for f in result.flips if f.game_id == "G1")
    assert flip.hurt_team == "HM1"
    assert flip.healthier_team == "AW1"

    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["G1", "home_cover_probability"] == pytest.approx(0.3)


def test_overlay_flips_when_away_lost_strictly_more_and_model_picked_away() -> None:
    result = apply_injury_value_tilt_overlay(_predictions(), _features())

    flipped_ids = {flip.game_id for flip in result.flips}
    assert "G2" in flipped_ids
    flip = next(f for f in result.flips if f.game_id == "G2")
    assert flip.hurt_team == "AW2"
    assert flip.healthier_team == "HM2"

    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["G2", "home_cover_probability"] == pytest.approx(0.8)


def test_overlay_does_not_flip_a_tie() -> None:
    result = apply_injury_value_tilt_overlay(_predictions(), _features())
    assert all(flip.game_id != "G3" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["G3", "home_cover_probability"] == pytest.approx(0.65)


def test_overlay_does_not_flip_when_the_pick_already_agrees_with_the_signal() -> None:
    result = apply_injury_value_tilt_overlay(_predictions(), _features())
    assert all(flip.game_id != "G4" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["G4", "home_cover_probability"] == pytest.approx(0.3)


def test_overlay_leaves_postseason_games_untouched() -> None:
    """Same shape as G1 (would flip if REG), but the REG-only gate blocks it
    -- the measured mechanism is a regular-season-only result."""

    result = apply_injury_value_tilt_overlay(_predictions(), _features())
    assert all(flip.game_id != "G5" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["G5", "home_cover_probability"] == pytest.approx(0.9)


def test_overlay_treats_a_missing_feature_row_as_a_tie() -> None:
    result = apply_injury_value_tilt_overlay(_predictions(), _features())
    assert all(flip.game_id != "G6" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["G6", "home_cover_probability"] == pytest.approx(0.55)


def test_overlay_disabled_is_a_no_op() -> None:
    predictions = _predictions()
    result = apply_injury_value_tilt_overlay(predictions, _features(), enabled=False)

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
    result = apply_injury_value_tilt_overlay(predictions, _features())
    overlaid = result.overlaid_predictions

    assert list(overlaid.columns) == list(predictions.columns)
    other_columns = [c for c in predictions.columns if c != "home_cover_probability"]
    pd.testing.assert_frame_equal(
        overlaid[other_columns].reset_index(drop=True),
        predictions[other_columns].reset_index(drop=True),
        check_exact=True,
    )
    untouched = predictions["game_id"].isin(["G3", "G4", "G5", "G6"])
    pd.testing.assert_series_equal(
        overlaid.loc[untouched, "home_cover_probability"].reset_index(drop=True),
        predictions.loc[untouched, "home_cover_probability"].reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_requires_its_prediction_columns() -> None:
    with pytest.raises(DataContractError, match="overlay columns"):
        apply_injury_value_tilt_overlay(pd.DataFrame({"game_id": ["G1"]}), _features())


# ---------------------------------------------------------------------------
# 3. overlay_disclosure_note: the plain-English provenance sentence
# ---------------------------------------------------------------------------


def test_disclosure_note_is_empty_when_nothing_flipped() -> None:
    ties_only = _predictions().loc[lambda frame: frame["game_id"].eq("G3")]
    result = apply_injury_value_tilt_overlay(ties_only, _features())
    assert overlay_disclosure_note(result) == ""

    disabled = apply_injury_value_tilt_overlay(_predictions(), _features(), enabled=False)
    assert overlay_disclosure_note(disabled) == ""


def test_disclosure_note_states_the_flip_count_and_does_not_claim_production() -> None:
    result = apply_injury_value_tilt_overlay(_predictions(), _features())
    note = overlay_disclosure_note(result)

    assert "Tilt applied: 2 picks flipped" in note
    assert "HM1 -> AW1" in note
    assert "AW2 -> HM2" in note
    assert "not applied to the published card" in note


# ---------------------------------------------------------------------------
# 4. record_injury_value_tilt_challenger_decisions: dual-tracked, no window
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
            "game_id": ["G1", "G3"],
            "season": [2026, 2026],
            "week": [1, 1],
            "game_type": ["REG", "REG"],
            "home_team": ["HM1", "HM3"],
            "away_team": ["AW1", "AW3"],
            "kickoff": ["2026-09-10T17:00:00+00:00", "2026-09-10T17:00:00+00:00"],
            "spread_line": [-3.5, -1.0],
            "home_cover_probability": [0.7, 0.65],
        }
    )


def _write_tilt_registry(artifacts: Path, *, status: str = "ACTIVE_PROSPECTIVE") -> None:
    payload = {
        "ledger": "prospective_challengers",
        "schema_version": 1,
        "challengers": [
            {
                "challenger_id": CHALLENGER_ID,
                "status": status,
                "model": dict(_MODEL_CONFIG),
            }
        ],
    }
    path = artifacts / "prospective" / "challengers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_active_model_and_card(artifacts: Path, *, ridge_alpha: float = 10.0) -> None:
    forecast = artifacts / "margin_predictions" / "2026-week-01-forecast"
    forecast.mkdir(parents=True, exist_ok=True)
    metadata = {
        "active_model_id": "model-xyz",
        "synchronization_status": "SYNCHRONIZED",
        "season": 2026,
        "week": 1,
        "created_at_utc": "2026-09-08T15:00:00+00:00",
        "ats_method": "market_residual",
        "regressor": "ridge",
        "ridge_alpha": ridge_alpha,
        "calibration_method": "none",
        "feature_profile": "weak_stack",
        "min_edge": 0.02,
        "min_train_games": 500,
        "provenance": {
            "feature_table": {
                "path": "data/processed/game_features_weak_stack.parquet",
                "sha256": "abc123",
            }
        },
    }
    (forecast / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    _recorder_predictions().to_csv(forecast / "recommendations.csv", index=False)

    active = {
        "version": ACTIVE_ATS_MODEL_VERSION,
        "status": "SYNCHRONIZED",
        "model_id": "model-xyz",
        "method": "market_residual",
        "feature_profile": "weak_stack",
        "historical_evaluation": {"accuracy": 0.52, "correct": 1, "games": 1, "intervals": {}},
        "weekly_forecast": {
            "artifact": "margin_predictions/2026-week-01-forecast",
            "season": 2026,
            "week": 1,
        },
    }
    (artifacts / "active_ats_model.json").write_text(json.dumps(active), encoding="utf-8")


def _write_data_root(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    processed = data_root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    _features().to_parquet(processed / "game_features_player.parquet")
    return data_root


def test_record_tilt_challenger_decisions_records_the_tilt_arm(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_tilt_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)
    now = datetime(2026, 9, 8, 16, 0, tzinfo=UTC)

    result = record_injury_value_tilt_challenger_decisions(artifacts, data_root, now=now)

    assert result["recorded"] == 2
    assert result["flip_count"] == 1
    assert result["flipped_game_ids"] == ["G1"]

    ledger = load_challenger_decisions(artifacts).set_index("game_id")
    assert list(load_challenger_decisions(artifacts).columns) == list(CHALLENGER_DECISION_COLUMNS)
    assert (ledger["bet_side"] == "PASS").all()
    assert ledger["edge"].isna().all()

    # G1's tilt arm diverges from the active model's own raw pick (0.7 ->
    # HOME): the tilt flips it to AWAY, since home lost strictly more value.
    assert ledger.loc["G1", "pick_side"] == "AWAY"
    # G3 (a tie) keeps the model's own HOME pick.
    assert ledger.loc["G3", "pick_side"] == "HOME"

    # Re-running is a no-op: append-only, never rewrites.
    again = record_injury_value_tilt_challenger_decisions(artifacts, data_root, now=now)
    assert again["recorded"] == 0
    assert again["already_recorded"] == 2


def test_record_tilt_challenger_refuses_outside_recording_lock_window(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_tilt_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_injury_value_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 8, 18, 1, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_tilt_challenger_refuses_a_fingerprint_mismatch(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_tilt_registry(artifacts)
    # The active model's OWN configuration moved (a promotion) since this
    # challenger was pinned -- recording must refuse, not silently switch
    # base models under the same challenger id.
    _write_active_model_and_card(artifacts, ridge_alpha=1.0)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(DataContractError, match="configuration fingerprint"):
        record_injury_value_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 9, 8, 16, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_tilt_challenger_refuses_an_inactive_registration(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_tilt_registry(artifacts, status="CLOSED_BEFORE_ACTIVATION")
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(ValueError, match="only ACTIVE_PROSPECTIVE"):
        record_injury_value_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 9, 8, 16, 0, tzinfo=UTC)
        )


def test_record_tilt_challenger_refuses_a_missing_feature_table(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_tilt_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = tmp_path / "data"
    (data_root / "processed").mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="not built yet"):
        record_injury_value_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 9, 8, 16, 0, tzinfo=UTC)
        )


def test_tilt_fingerprint_helper_agrees_with_the_registered_model_block() -> None:
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
