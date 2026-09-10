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
from nfl_ats.turnover_luck_rebound_tilt_overlay import (
    CHALLENGER_ID,
    TURNOVER_UNDER_Q25_THRESHOLD,
    apply_turnover_luck_rebound_tilt_overlay,
    overlay_disclosure_note,
    record_turnover_luck_rebound_tilt_challenger_decisions,
    turnover_under_flag_by_game,
)


def _pbp_row(
    game_id: str, posteam: str, *, interception: float = 0.0, fumble_lost: float = 0.0
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "season_type": "REG",
        "posteam": posteam,
        "interception": interception,
        "fumble_lost": fumble_lost,
    }


def _pbp_2025(*, extra_2026_rows: list[dict[str, object]] | None = None) -> pd.DataFrame:

    rows = [
        *[_pbp_row("2025_01_TEAMA_OPPX", "TEAMA", interception=1.0) for _ in range(3)],
        *[_pbp_row("2025_01_TEAMA_OPPX", "TEAMA", fumble_lost=1.0) for _ in range(2)],
        *[_pbp_row("2025_02_OPPY_TEAMA", "TEAMA", interception=1.0) for _ in range(3)],
        *[_pbp_row("2025_02_OPPY_TEAMA", "TEAMA", fumble_lost=1.0) for _ in range(2)],
        *[_pbp_row("2025_03_TEAME_OPPX", "TEAME", interception=1.0) for _ in range(4)],
        *[_pbp_row("2025_03_TEAME_OPPX", "TEAME", fumble_lost=1.0) for _ in range(2)],
        *[_pbp_row("2025_04_OPPY_TEAME", "TEAME", interception=1.0) for _ in range(4)],
        *[_pbp_row("2025_04_OPPY_TEAME", "TEAME", fumble_lost=1.0) for _ in range(2)],
        _pbp_row("2025_05_TEAMB_TEAMC", "TEAMB"),
        _pbp_row("2025_05_TEAMB_TEAMC", "TEAMC"),
    ]
    if extra_2026_rows:
        rows.extend(extra_2026_rows)
    return pd.DataFrame(rows)


def _schedule() -> pd.DataFrame:
    rows_2025 = [
        ("2025_01_TEAMA_OPPX", 2025, 1, "REG", "TEAMA", "OPPX"),
        ("2025_02_OPPY_TEAMA", 2025, 2, "REG", "OPPY", "TEAMA"),
        ("2025_03_TEAME_OPPX", 2025, 3, "REG", "TEAME", "OPPX"),
        ("2025_04_OPPY_TEAME", 2025, 4, "REG", "OPPY", "TEAME"),
        ("2025_05_TEAMB_TEAMC", 2025, 5, "REG", "TEAMB", "TEAMC"),
    ]
    rows_2026 = [
        ("2026_01_TEAMD_TEAMA", 2026, 1, "REG", "TEAMD", "TEAMA"),
        ("2026_01_TEAMB_TEAMC", 2026, 1, "REG", "TEAMB", "TEAMC"),
        ("2026_01_TEAMA_TEAME", 2026, 1, "REG", "TEAMA", "TEAME"),
        ("2026_01_OPPX_TEAMA", 2026, 1, "REG", "OPPX", "TEAMA"),
    ]
    columns = ["game_id", "season", "week", "game_type", "home_team", "away_team"]
    return pd.DataFrame(rows_2025 + rows_2026, columns=columns)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [
                "2026_01_TEAMD_TEAMA",
                "2026_01_TEAMB_TEAMC",
                "2026_01_TEAMA_TEAME",
                "2026_01_OPPX_TEAMA",
            ],
            "season": [2026, 2026, 2026, 2026],
            "week": [1, 1, 1, 1],
            "game_type": ["REG", "REG", "REG", "REG"],
            "home_team": ["TEAMD", "TEAMB", "TEAMA", "OPPX"],
            "away_team": ["TEAMA", "TEAMC", "TEAME", "TEAMA"],
            "kickoff": ["2026-09-13T17:00:00+00:00"] * 4,
            "spread_line": [-2.5, 1.0, -3.0, -6.0],
            "home_cover_probability": [0.58, 0.55, 0.65, 0.30],
        }
    )


def test_the_frozen_threshold_matches_the_screens_measured_value() -> None:
    assert pytest.approx(-0.4026832217261905) == TURNOVER_UNDER_Q25_THRESHOLD


def test_flag_fires_on_the_bottom_quartile_prior_season_teams() -> None:
    flags = turnover_under_flag_by_game(_schedule(), _pbp_2025())
    row = flags.set_index("game_id").loc["2026_01_TEAMD_TEAMA"]
    assert bool(row["home_turnover_under_flag"]) is False
    assert bool(row["away_turnover_under_flag"]) is True


def test_flag_reproduces_the_screens_quartile_cut_on_this_fixture() -> None:

    flags = turnover_under_flag_by_game(_schedule(), _pbp_2025())
    both = flags.set_index("game_id").loc["2026_01_TEAMA_TEAME"]
    assert bool(both["home_turnover_under_flag"]) is True
    assert bool(both["away_turnover_under_flag"]) is True

    neutral = flags.set_index("game_id").loc["2026_01_TEAMB_TEAMC"]
    assert bool(neutral["home_turnover_under_flag"]) is False
    assert bool(neutral["away_turnover_under_flag"]) is False


def test_flag_is_false_when_the_team_has_no_prior_season_data() -> None:

    flags = turnover_under_flag_by_game(_schedule(), _pbp_2025())
    row = flags.set_index("game_id").loc["2026_01_TEAMD_TEAMA"]
    assert bool(row["home_turnover_under_flag"]) is False


def test_flag_requires_its_schedule_columns() -> None:
    with pytest.raises(DataContractError, match="schedules is missing columns"):
        turnover_under_flag_by_game(pd.DataFrame({"game_id": ["G1"]}), _pbp_2025())


def test_flag_requires_its_play_by_play_columns() -> None:
    with pytest.raises(DataContractError, match="play-by-play is missing columns"):
        turnover_under_flag_by_game(_schedule(), pd.DataFrame({"game_id": ["G1"]}))


def test_flag_is_unchanged_by_the_current_seasons_own_turnover_events() -> None:

    baseline = turnover_under_flag_by_game(_schedule(), _pbp_2025())

    leaky_extra = [
        _pbp_row("2026_01_TEAMD_TEAMA", "TEAMA", interception=1.0) for _ in range(20)
    ] + [_pbp_row("2026_01_TEAMD_TEAMA", "TEAMD", fumble_lost=1.0) for _ in range(20)]
    with_current_season = turnover_under_flag_by_game(
        _schedule(), _pbp_2025(extra_2026_rows=leaky_extra)
    )

    pd.testing.assert_frame_equal(
        baseline.sort_values("game_id").reset_index(drop=True),
        with_current_season.sort_values("game_id").reset_index(drop=True),
    )


def test_flag_is_unchanged_by_a_flipped_outcome_in_the_current_seasons_other_games() -> None:

    baseline = turnover_under_flag_by_game(_schedule(), _pbp_2025())

    leaky_extra = [_pbp_row("2026_01_TEAMB_TEAMC", "TEAMB", interception=1.0) for _ in range(10)]
    mutated = turnover_under_flag_by_game(_schedule(), _pbp_2025(extra_2026_rows=leaky_extra))

    pd.testing.assert_frame_equal(
        baseline.sort_values("game_id").reset_index(drop=True),
        mutated.sort_values("game_id").reset_index(drop=True),
    )


def test_overlay_flips_onto_the_flagged_team_when_not_already_picked() -> None:
    schedule = _schedule()
    result = apply_turnover_luck_rebound_tilt_overlay(_predictions(), schedule, _pbp_2025())

    flipped_ids = {flip.game_id for flip in result.flips}
    assert "2026_01_TEAMD_TEAMA" in flipped_ids
    flip = next(f for f in result.flips if f.game_id == "2026_01_TEAMD_TEAMA")
    assert flip.flagged_team == "TEAMA"
    assert flip.original_pick_team == "TEAMD"

    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_01_TEAMD_TEAMA", "home_cover_probability"] == pytest.approx(0.42)


def test_overlay_leaves_a_neutral_game_untouched() -> None:
    schedule = _schedule()
    result = apply_turnover_luck_rebound_tilt_overlay(_predictions(), schedule, _pbp_2025())
    assert all(flip.game_id != "2026_01_TEAMB_TEAMC" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_01_TEAMB_TEAMC", "home_cover_probability"] == pytest.approx(0.55)


def test_overlay_never_flips_a_both_flagged_game() -> None:
    schedule = _schedule()
    result = apply_turnover_luck_rebound_tilt_overlay(_predictions(), schedule, _pbp_2025())
    assert all(flip.game_id != "2026_01_TEAMA_TEAME" for flip in result.flips)
    assert "2026_01_TEAMA_TEAME" in result.both_flagged_games
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_01_TEAMA_TEAME", "home_cover_probability"] == pytest.approx(0.65)


def test_overlay_leaves_a_pick_already_on_the_flagged_team_untouched() -> None:
    schedule = _schedule()
    result = apply_turnover_luck_rebound_tilt_overlay(_predictions(), schedule, _pbp_2025())
    assert all(flip.game_id != "2026_01_OPPX_TEAMA" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_01_OPPX_TEAMA", "home_cover_probability"] == pytest.approx(0.30)


def test_overlay_disabled_is_a_no_op() -> None:
    schedule = _schedule()
    predictions = _predictions()
    result = apply_turnover_luck_rebound_tilt_overlay(
        predictions, schedule, _pbp_2025(), enabled=False
    )
    assert result.flip_count == 0
    pd.testing.assert_frame_equal(
        result.overlaid_predictions.reset_index(drop=True),
        predictions.reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_changes_only_home_cover_probability_on_flipped_rows() -> None:
    schedule = _schedule()
    predictions = _predictions()
    result = apply_turnover_luck_rebound_tilt_overlay(predictions, schedule, _pbp_2025())
    overlaid = result.overlaid_predictions

    assert list(overlaid.columns) == list(predictions.columns)
    other_columns = [c for c in predictions.columns if c != "home_cover_probability"]
    pd.testing.assert_frame_equal(
        overlaid[other_columns].reset_index(drop=True),
        predictions[other_columns].reset_index(drop=True),
        check_exact=True,
    )
    untouched = predictions["game_id"].isin(
        ["2026_01_TEAMB_TEAMC", "2026_01_TEAMA_TEAME", "2026_01_OPPX_TEAMA"]
    )
    pd.testing.assert_series_equal(
        overlaid.loc[untouched, "home_cover_probability"].reset_index(drop=True),
        predictions.loc[untouched, "home_cover_probability"].reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_leaves_a_flagged_game_untouched_when_marked_postseason() -> None:
    schedule = _schedule()
    predictions = pd.DataFrame(
        {
            "game_id": ["2026_01_TEAMD_TEAMA"],
            "season": [2026],
            "week": [1],
            "game_type": ["POST"],
            "home_team": ["TEAMD"],
            "away_team": ["TEAMA"],
            "kickoff": ["2026-09-13T17:00:00+00:00"],
            "spread_line": [-2.5],
            "home_cover_probability": [0.58],
        }
    )
    result = apply_turnover_luck_rebound_tilt_overlay(predictions, schedule, _pbp_2025())
    assert result.flip_count == 0
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2026_01_TEAMD_TEAMA", "home_cover_probability"] == pytest.approx(0.58)


def test_overlay_requires_its_prediction_columns() -> None:
    with pytest.raises(DataContractError, match="overlay columns"):
        apply_turnover_luck_rebound_tilt_overlay(
            pd.DataFrame({"game_id": ["G1"]}), _schedule(), _pbp_2025()
        )


def test_disclosure_note_is_empty_when_nothing_flipped() -> None:
    schedule = _schedule()
    matched_only = _predictions().loc[lambda frame: frame["game_id"].eq("2026_01_TEAMB_TEAMC")]
    result = apply_turnover_luck_rebound_tilt_overlay(matched_only, schedule, _pbp_2025())
    assert overlay_disclosure_note(result) == ""

    disabled = apply_turnover_luck_rebound_tilt_overlay(
        _predictions(), schedule, _pbp_2025(), enabled=False
    )
    assert overlay_disclosure_note(disabled) == ""


def test_disclosure_note_states_the_flip_count_and_does_not_claim_production() -> None:
    schedule = _schedule()
    result = apply_turnover_luck_rebound_tilt_overlay(_predictions(), schedule, _pbp_2025())
    note = overlay_disclosure_note(result)

    assert "Tilt applied: 1 pick flipped" in note
    assert "TEAMD -> TEAMA" in note
    assert "not applied to the published card" in note


_MODEL_CONFIG = {
    "method": "market_residual",
    "target": "market_residual",
    "regressor": "ridge",
    "ridge_alpha": 10.0,
    "calibration_method": "none",
    "feature_profile": "weak_stack",
    "feature_set": "full_weak_stack",
    "min_edge": 0.02,
    "min_train_games": 500,
    "feature_table": "data/processed/game_features_weak_stack.parquet",
}


def _recorder_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2026_01_TEAMD_TEAMA", "2026_01_TEAMB_TEAMC"],
            "season": [2026, 2026],
            "week": [1, 1],
            "game_type": ["REG", "REG"],
            "home_team": ["TEAMD", "TEAMB"],
            "away_team": ["TEAMA", "TEAMC"],
            "kickoff": ["2026-09-13T17:00:00+00:00", "2026-09-13T17:00:00+00:00"],
            "spread_line": [-2.5, 1.0],
            "home_cover_probability": [0.58, 0.55],
        }
    )


def _write_registry(artifacts: Path, *, status: str = "ACTIVE_PROSPECTIVE") -> None:
    write_challenger_registry(
        artifacts, challenger_id=CHALLENGER_ID, model_config=_MODEL_CONFIG, status=status
    )


_SEASON, _WEEK, _CREATED_AT_UTC = 2026, 1, "2026-09-08T15:00:00+00:00"


def _write_active_model_and_card(artifacts: Path, *, ridge_alpha: float = 10.0) -> None:
    write_active_model_and_card(
        artifacts,
        season=_SEASON,
        week=_WEEK,
        created_at_utc=_CREATED_AT_UTC,
        ridge_alpha=ridge_alpha,
        recommendations=_recorder_predictions(),
    )


def _full_pbp_2025() -> pd.DataFrame:

    schedule_2025 = _schedule().loc[lambda f: f["season"].eq(2025)].set_index("game_id")
    rows: list[dict[str, object]] = []
    for play_id, record in enumerate(_pbp_2025().to_dict("records"), start=1):
        game_id = str(record["game_id"])
        home_team = schedule_2025.loc[game_id, "home_team"]
        away_team = schedule_2025.loc[game_id, "away_team"]
        posteam = record["posteam"]
        defteam = away_team if posteam == home_team else home_team
        rows.append(
            {
                "play_id": play_id,
                "game_id": game_id,
                "season": 2025,
                "season_type": "REG",
                "week": int(schedule_2025.loc[game_id, "week"]),
                "home_team": home_team,
                "away_team": away_team,
                "posteam": posteam,
                "defteam": defteam,
                "fixed_drive": 1,
                "down": 1,
                "play_type": "pass",
                "yards_gained": 0,
                "pass_attempt": 1,
                "rush_attempt": 0,
                "sack": 0,
                "qb_hit": 0,
                "epa": 0.0,
                "success": 0,
                "wp": 0.5,
                "interception": record["interception"],
                "fumble_lost": record["fumble_lost"],
            }
        )
    return pd.DataFrame(rows)


def _write_data_root(tmp_path: Path) -> Path:

    from nfl_ats.pbp import write_pbp_snapshot
    from nfl_ats.snapshots import write_snapshot

    data_root = tmp_path / "data"
    write_snapshot(
        _schedule(),
        pd.DataFrame({"game_id": [], "team": []}),
        seasons=[2025, 2026],
        raw_root=data_root / "raw",
    )
    write_pbp_snapshot({2025: _full_pbp_2025()}, data_root / "pbp" / "raw")
    return data_root


def test_record_turnover_luck_rebound_challenger_decisions_records_the_tilt_arm(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)
    now = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)

    result = record_turnover_luck_rebound_tilt_challenger_decisions(artifacts, data_root, now=now)

    assert result["recorded"] == 2
    assert result["flip_count"] == 1
    assert result["flipped_game_ids"] == ["2026_01_TEAMD_TEAMA"]

    ledger = load_challenger_decisions(artifacts).set_index("game_id")
    assert list(load_challenger_decisions(artifacts).columns) == list(CHALLENGER_DECISION_COLUMNS)
    assert (ledger["bet_side"] == "PASS").all()
    assert ledger["edge"].isna().all()

    assert ledger.loc["2026_01_TEAMD_TEAMA", "pick_side"] == "AWAY"
    assert ledger.loc["2026_01_TEAMB_TEAMC", "pick_side"] == "HOME"

    again = record_turnover_luck_rebound_tilt_challenger_decisions(artifacts, data_root, now=now)
    assert again["recorded"] == 0
    assert again["already_recorded"] == 2


def test_record_turnover_luck_rebound_challenger_refuses_outside_recording_lock_window(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_turnover_luck_rebound_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 8, 1, 1, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_turnover_luck_rebound_challenger_refuses_a_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts, ridge_alpha=1.0)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(DataContractError, match="configuration fingerprint"):
        record_turnover_luck_rebound_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_turnover_luck_rebound_challenger_refuses_an_inactive_registration(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts, status="CLOSED_BEFORE_ACTIVATION")
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)

    with pytest.raises(ValueError, match="only ACTIVE_PROSPECTIVE"):
        record_turnover_luck_rebound_tilt_challenger_decisions(
            artifacts, data_root, now=datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
        )


def test_turnover_luck_rebound_fingerprint_helper_agrees_with_the_registered_model_block() -> None:

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
