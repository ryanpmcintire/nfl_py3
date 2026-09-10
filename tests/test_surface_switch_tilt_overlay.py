from __future__ import annotations

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
from nfl_ats.snapshots import write_snapshot
from nfl_ats.surface_switch_tilt_overlay import (
    CHALLENGER_ID,
    apply_surface_switch_tilt_overlay,
    overlay_disclosure_note,
    record_surface_switch_tilt_challenger_decisions,
    surface_switch_flag_by_game,
)


def _surface_schedule() -> pd.DataFrame:
    rows = [
        ("2026_01_GRASSAWAY_OPP1", 2026, "REG", 1, "GRASSAWAY", "OPP1", "grass"),
        ("2026_02_GRASSAWAY_OPP2", 2026, "REG", 2, "GRASSAWAY", "OPP2", "grass"),
        ("2026_03_TURFHOST_GRASSAWAY", 2026, "REG", 3, "TURFHOST", "GRASSAWAY", "fieldturf"),
        ("2026_04_GRASSHOST_GRASSAWAY", 2026, "REG", 4, "GRASSHOST", "GRASSAWAY", "grass"),
        ("2026_06_MISSHOST_GRASSAWAY", 2026, "REG", 6, "MISSHOST", "GRASSAWAY", ""),
        ("2026_01_TURFAWAY_OPP3", 2026, "REG", 1, "TURFAWAY", "OPP3", "fieldturf"),
        ("2026_02_TURFAWAY_OPP4", 2026, "REG", 2, "TURFAWAY", "OPP4", "astroturf"),
        ("2026_03_TURFHOST2_TURFAWAY", 2026, "REG", 3, "TURFHOST2", "TURFAWAY", "sportturf"),
        ("2026_01_NOSURF_OPPX", 2026, "REG", 1, "NOSURF", "OPPX", ""),
        ("2026_05_TURFHOST3_NOSURF", 2026, "REG", 5, "TURFHOST3", "NOSURF", "fieldturf"),
        ("2026_01_GRASSAWAY2_OPP5", 2026, "REG", 1, "GRASSAWAY2", "OPP5", "grass"),
        ("2026_02_GRASSAWAY2_OPP6", 2026, "REG", 2, "GRASSAWAY2", "OPP6", "grass"),
        ("2026_07_TURFHOST4_GRASSAWAY2", 2026, "REG", 7, "TURFHOST4", "GRASSAWAY2", "astroturf"),
        ("2026_20_POSTHOST_GRASSAWAY", 2026, "POST", 20, "POSTHOST", "GRASSAWAY", "fieldturf"),
    ]
    return pd.DataFrame(
        rows,
        columns=["game_id", "season", "game_type", "week", "home_team", "away_team", "surface"],
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [
                "2026_03_TURFHOST_GRASSAWAY",
                "2026_03_TURFHOST2_TURFAWAY",
                "2026_04_GRASSHOST_GRASSAWAY",
                "2026_05_TURFHOST3_NOSURF",
                "2026_06_MISSHOST_GRASSAWAY",
                "2026_07_TURFHOST4_GRASSAWAY2",
                "2026_20_POSTHOST_GRASSAWAY",
                "2026_MISSING_GAME",
            ],
            "season": [2026, 2026, 2026, 2026, 2026, 2026, 2026, 2026],
            "week": [3, 3, 4, 5, 6, 7, 20, 3],
            "game_type": ["REG", "REG", "REG", "REG", "REG", "REG", "POST", "REG"],
            "home_team": [
                "TURFHOST",
                "TURFHOST2",
                "GRASSHOST",
                "TURFHOST3",
                "MISSHOST",
                "TURFHOST4",
                "POSTHOST",
                "MISS_H",
            ],
            "away_team": [
                "GRASSAWAY",
                "TURFAWAY",
                "GRASSAWAY",
                "NOSURF",
                "GRASSAWAY",
                "GRASSAWAY2",
                "GRASSAWAY",
                "MISS_A",
            ],
            "kickoff": ["2026-09-24T17:00:00+00:00"] * 8,
            "spread_line": [-3.0, 2.0, -1.5, 1.0, -2.5, -4.0, -3.0, 1.0],
            "home_cover_probability": [0.35, 0.40, 0.45, 0.42, 0.38, 0.60, 0.30, 0.50],
        }
    )


def test_surface_switch_flag_fires_on_a_grass_modal_visitor_onto_turf() -> None:
    flags = surface_switch_flag_by_game(_surface_schedule()).set_index("game_id")
    assert bool(flags.loc["2026_03_TURFHOST_GRASSAWAY", "surface_switch_flag"]) is True


def test_surface_switch_flag_is_false_when_surfaces_match() -> None:
    flags = surface_switch_flag_by_game(_surface_schedule()).set_index("game_id")
    assert bool(flags.loc["2026_03_TURFHOST2_TURFAWAY", "surface_switch_flag"]) is False


def test_surface_switch_flag_is_false_without_a_turf_venue() -> None:

    flags = surface_switch_flag_by_game(_surface_schedule()).set_index("game_id")
    assert bool(flags.loc["2026_04_GRASSHOST_GRASSAWAY", "surface_switch_flag"]) is False


def test_surface_switch_flag_is_false_with_missing_visitor_surface_history() -> None:
    flags = surface_switch_flag_by_game(_surface_schedule()).set_index("game_id")
    assert bool(flags.loc["2026_05_TURFHOST3_NOSURF", "surface_switch_flag"]) is False


def test_surface_switch_flag_is_false_with_a_missing_game_surface() -> None:
    flags = surface_switch_flag_by_game(_surface_schedule()).set_index("game_id")
    assert bool(flags.loc["2026_06_MISSHOST_GRASSAWAY", "surface_switch_flag"]) is False


def test_surface_switch_flag_requires_its_schedule_columns() -> None:
    with pytest.raises(DataContractError, match="surface-switch"):
        surface_switch_flag_by_game(pd.DataFrame({"game_id": ["G1"]}))


def test_surface_switch_flag_never_reads_outcome_columns() -> None:

    schedule = _surface_schedule()
    schedule["result"] = 0.0
    schedule["spread_line"] = -3.0
    baseline = surface_switch_flag_by_game(schedule).set_index("game_id")

    mutated = schedule.copy()
    mutated.loc[mutated["game_id"].eq("2026_03_TURFHOST_GRASSAWAY"), "result"] = 99.0
    mutated.loc[mutated["game_id"].eq("2026_03_TURFHOST_GRASSAWAY"), "spread_line"] = 14.0
    changed = surface_switch_flag_by_game(mutated).set_index("game_id")

    pd.testing.assert_frame_equal(changed, baseline, check_exact=True)


def test_surface_switch_flag_is_leak_safe_across_the_season_boundary() -> None:

    schedule = _surface_schedule()
    baseline = surface_switch_flag_by_game(schedule)

    future = pd.DataFrame(
        [
            ("2027_01_GRASSAWAY_OPP1", 2027, "REG", 1, "GRASSAWAY", "OPP1", "fieldturf"),
            ("2027_03_TURFHOST_GRASSAWAY", 2027, "REG", 3, "TURFHOST", "GRASSAWAY", "fieldturf"),
        ],
        columns=schedule.columns,
    )
    changed = surface_switch_flag_by_game(pd.concat([schedule, future], ignore_index=True))

    pd.testing.assert_frame_equal(
        changed.loc[changed["season"].le(2026)].reset_index(drop=True),
        baseline.loc[baseline["season"].le(2026)].reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_flips_an_away_pick_on_the_flagged_side() -> None:
    result = apply_surface_switch_tilt_overlay(_predictions(), _surface_schedule())

    flipped_ids = {flip.game_id for flip in result.flips}
    assert "2026_03_TURFHOST_GRASSAWAY" in flipped_ids
    flip = next(f for f in result.flips if f.game_id == "2026_03_TURFHOST_GRASSAWAY")
    assert flip.grass_modal_visitor == "GRASSAWAY"
    assert flip.turf_venue_home == "TURFHOST"

    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_03_TURFHOST_GRASSAWAY", "home_cover_probability"] == pytest.approx(
        0.65
    )


def test_overlay_does_not_flip_a_home_pick_even_when_flagged() -> None:

    result = apply_surface_switch_tilt_overlay(_predictions(), _surface_schedule())
    assert all(flip.game_id != "2026_07_TURFHOST4_GRASSAWAY2" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_07_TURFHOST4_GRASSAWAY2", "home_cover_probability"] == pytest.approx(
        0.60
    )


def test_overlay_does_not_flip_when_surfaces_match() -> None:
    result = apply_surface_switch_tilt_overlay(_predictions(), _surface_schedule())
    assert all(flip.game_id != "2026_03_TURFHOST2_TURFAWAY" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_03_TURFHOST2_TURFAWAY", "home_cover_probability"] == pytest.approx(
        0.40
    )


def test_overlay_does_not_flip_without_a_turf_venue() -> None:
    result = apply_surface_switch_tilt_overlay(_predictions(), _surface_schedule())
    assert all(flip.game_id != "2026_04_GRASSHOST_GRASSAWAY" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_04_GRASSHOST_GRASSAWAY", "home_cover_probability"] == pytest.approx(
        0.45
    )


def test_overlay_leaves_postseason_games_untouched() -> None:

    result = apply_surface_switch_tilt_overlay(_predictions(), _surface_schedule())
    assert all(flip.game_id != "2026_20_POSTHOST_GRASSAWAY" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_20_POSTHOST_GRASSAWAY", "home_cover_probability"] == pytest.approx(
        0.30
    )


def test_overlay_treats_a_missing_schedule_row_as_no_signal() -> None:
    result = apply_surface_switch_tilt_overlay(_predictions(), _surface_schedule())
    assert all(flip.game_id != "2026_MISSING_GAME" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_MISSING_GAME", "home_cover_probability"] == pytest.approx(0.50)


def test_overlay_disabled_is_a_no_op() -> None:
    predictions = _predictions()
    result = apply_surface_switch_tilt_overlay(predictions, _surface_schedule(), enabled=False)

    assert result.flip_count == 0
    pd.testing.assert_frame_equal(
        result.overlaid_predictions.reset_index(drop=True),
        predictions.reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_changes_only_home_cover_probability_on_flipped_rows() -> None:

    predictions = _predictions()
    result = apply_surface_switch_tilt_overlay(predictions, _surface_schedule())
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
            "2026_03_TURFHOST2_TURFAWAY",
            "2026_04_GRASSHOST_GRASSAWAY",
            "2026_05_TURFHOST3_NOSURF",
            "2026_06_MISSHOST_GRASSAWAY",
            "2026_07_TURFHOST4_GRASSAWAY2",
            "2026_20_POSTHOST_GRASSAWAY",
            "2026_MISSING_GAME",
        ]
    )
    pd.testing.assert_series_equal(
        overlaid.loc[untouched, "home_cover_probability"].reset_index(drop=True),
        predictions.loc[untouched, "home_cover_probability"].reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_requires_its_prediction_columns() -> None:
    with pytest.raises(DataContractError, match="overlay columns"):
        apply_surface_switch_tilt_overlay(pd.DataFrame({"game_id": ["G1"]}), _surface_schedule())


def test_overlay_survives_predictions_that_already_carry_the_flag_column() -> None:

    predictions = _predictions()
    predictions["surface_switch_flag"] = False

    result = apply_surface_switch_tilt_overlay(predictions, _surface_schedule())

    flipped_ids = {flip.game_id for flip in result.flips}
    assert flipped_ids == {"2026_03_TURFHOST_GRASSAWAY"}
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_03_TURFHOST_GRASSAWAY", "home_cover_probability"] == pytest.approx(
        0.65
    )
    assert list(result.overlaid_predictions.columns) == list(predictions.columns)
    assert (result.overlaid_predictions["surface_switch_flag"] == False).all()  # noqa: E712


def test_overlay_ignores_a_misleading_preexisting_flag_column() -> None:

    predictions = _predictions()
    predictions["surface_switch_flag"] = [
        False,
        True,
        *([False] * 6),
    ]

    result = apply_surface_switch_tilt_overlay(predictions, _surface_schedule())

    assert {flip.game_id for flip in result.flips} == {"2026_03_TURFHOST_GRASSAWAY"}
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_03_TURFHOST2_TURFAWAY", "home_cover_probability"] == pytest.approx(
        0.40
    )


def test_record_surface_switch_challenger_survives_a_card_carrying_the_flag_column(
    tmp_path: Path,
) -> None:

    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    card_path = artifacts / "margin_predictions" / "2026-week-03-forecast" / "recommendations.csv"
    card = pd.read_csv(card_path)
    card["surface_switch_flag"] = [False, True]
    card.to_csv(card_path, index=False)
    data_root = _write_data_root(tmp_path)
    now = datetime(2026, 9, 20, 16, 0, tzinfo=UTC)

    result = record_surface_switch_tilt_challenger_decisions(artifacts, data_root, now=now)

    assert result["recorded"] == 2
    assert result["flip_count"] == 1
    assert result["flipped_game_ids"] == ["2026_03_TURFHOST_GRASSAWAY"]


def test_disclosure_note_is_empty_when_nothing_flipped() -> None:
    matched_only = _predictions().loc[
        lambda frame: frame["game_id"].eq("2026_03_TURFHOST2_TURFAWAY")
    ]
    result = apply_surface_switch_tilt_overlay(matched_only, _surface_schedule())
    assert overlay_disclosure_note(result) == ""

    disabled = apply_surface_switch_tilt_overlay(_predictions(), _surface_schedule(), enabled=False)
    assert overlay_disclosure_note(disabled) == ""


def test_disclosure_note_states_the_flip_count_and_does_not_claim_production() -> None:
    result = apply_surface_switch_tilt_overlay(_predictions(), _surface_schedule())
    note = overlay_disclosure_note(result)

    assert "Tilt applied: 1 pick flipped" in note
    assert "GRASSAWAY -> TURFHOST" in note
    assert "not applied to the published card" in note


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
            "game_id": ["2026_03_TURFHOST_GRASSAWAY", "2026_03_TURFHOST2_TURFAWAY"],
            "season": [2026, 2026],
            "week": [3, 3],
            "game_type": ["REG", "REG"],
            "home_team": ["TURFHOST", "TURFHOST2"],
            "away_team": ["GRASSAWAY", "TURFAWAY"],
            "kickoff": ["2026-09-24T17:00:00+00:00", "2026-09-24T17:00:00+00:00"],
            "spread_line": [-3.0, 2.0],
            "home_cover_probability": [0.35, 0.40],
        }
    )


def _write_registry(artifacts: Path, *, status: str = "ACTIVE_PROSPECTIVE") -> None:
    write_challenger_registry(
        artifacts, challenger_id=CHALLENGER_ID, model_config=_MODEL_CONFIG, status=status
    )


_SEASON, _WEEK, _CREATED_AT_UTC = 2026, 3, "2026-09-17T15:00:00+00:00"


def _write_active_model_and_card(artifacts: Path, *, ridge_alpha: float = 10.0) -> None:
    write_active_model_and_card(
        artifacts,
        season=_SEASON,
        week=_WEEK,
        created_at_utc=_CREATED_AT_UTC,
        ridge_alpha=ridge_alpha,
        recommendations=_recorder_predictions(),
    )


def _write_data_root(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    write_snapshot(
        _surface_schedule(),
        pd.DataFrame({"game_id": [], "team": []}),
        seasons=[2026],
        raw_root=data_root / "raw",
    )
    return data_root


def test_record_surface_switch_challenger_decisions_records_the_tilt_arm(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)
    now = datetime(2026, 9, 20, 16, 0, tzinfo=UTC)

    result = record_surface_switch_tilt_challenger_decisions(artifacts, data_root, now=now)

    assert result["recorded"] == 2
    assert result["flip_count"] == 1
    assert result["flipped_game_ids"] == ["2026_03_TURFHOST_GRASSAWAY"]

    ledger = load_challenger_decisions(artifacts).set_index("game_id")
    assert list(load_challenger_decisions(artifacts).columns) == list(CHALLENGER_DECISION_COLUMNS)
    assert (ledger["bet_side"] == "PASS").all()
    assert ledger["edge"].isna().all()

    assert ledger.loc["2026_03_TURFHOST_GRASSAWAY", "pick_side"] == "HOME"
    assert ledger.loc["2026_03_TURFHOST2_TURFAWAY", "pick_side"] == "AWAY"

    again = record_surface_switch_tilt_challenger_decisions(artifacts, data_root, now=now)
    assert again["recorded"] == 0
    assert again["already_recorded"] == 2


def test_record_surface_switch_challenger_refuses_outside_recording_lock_window(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_surface_switch_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 8, 18, 1, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_surface_switch_challenger_refuses_a_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts, ridge_alpha=1.0)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(DataContractError, match="configuration fingerprint"):
        record_surface_switch_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 9, 20, 16, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_surface_switch_challenger_refuses_an_inactive_registration(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts, status="CLOSED_BEFORE_ACTIVATION")
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(ValueError, match="only ACTIVE_PROSPECTIVE"):
        record_surface_switch_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 9, 20, 16, 0, tzinfo=UTC)
        )


def test_surface_switch_fingerprint_helper_agrees_with_the_registered_model_block() -> None:

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
