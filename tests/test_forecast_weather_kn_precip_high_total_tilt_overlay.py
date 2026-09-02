"""Forecast (kickoff-nearest) precip-high-total tilt overlay
(docs/forecast_weather_screen.md, "Wiring recommendations" #3).

Mirrors ``tests/test_forecast_weather_kn_warm_team_cold_late_tilt_overlay.py``'s
structure. Load-bearing here:

1. :func:`precip_high_total_flag_by_game` -- the frozen flag definition
   (outdoor AND kickoff-nearest forecast precip prob>=60% AND this game's
   own total_line>=47), missing-data-safe.
2. :func:`apply_precip_high_total_tilt_overlay` -- flips ONLY the clean case
   (away pick, flag fires), REG-only, parameter-free, asymmetric, reads
   ``total_line`` directly off the card.
3. This module's own fetch is a thin import of its sibling
   (``forecast_weather_kn_warm_team_cold_late_tilt_overlay``)'s
   kickoff_nearest fetch machinery -- no real network call is made in any
   test here, and a total fetch failure folds into zero flags, never an
   exception.
4. :func:`record_forecast_weather_kn_precip_high_total_tilt_challenger_decisions`
   writes the overlay's own picks to the prospective challenger ledger,
   dual-tracked and at no rotation-registry window cost, and honors a
   pre-fetched ``forecasts=`` override (the "one fetch, several consumers"
   path, shared with the warm-team-cold-late sibling) without making its
   own network call.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from _overlay_test_kit import (
    write_active_model_and_card,
    write_challenger_registry,
    write_registry_root,
)

from nfl_ats.data import DataContractError
from nfl_ats.forecast_weather_kn_precip_high_total_tilt_overlay import (
    CHALLENGER_ID,
    PrecipHighTotalFlip,
    PrecipHighTotalResult,
    apply_precip_high_total_tilt_overlay,
    overlay_disclosure_note,
    precip_high_total_flag_by_game,
    record_forecast_weather_kn_precip_high_total_tilt_challenger_decisions,
)
from nfl_ats.prospective_scoring import CHALLENGER_DECISION_COLUMNS, load_challenger_decisions
from nfl_ats.snapshots import write_snapshot

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
#
# WET at RAIN, outdoor, precip 70%, total_line 48 -> flagged. Model's pick is
#   AWAY (WET) -> flips HOME.
# WET at DOME, roof=closed, precip 70%, total_line 48: flag would fire on
#   precip/total alone, but the game itself is indoors -> no flip.
# WET at RAIN2, outdoor, precip 40% (below 60), total_line 51 -> not flagged.
# WET at RAIN3, outdoor, precip 80%, total_line 44 (below 47) -> not flagged.
# WET at RAIN4, outdoor, NO forecast row at all -> no signal -> no flip.
# WET at RAINP mirrors the clean flagged shape but game_type=POST.


def _schedule() -> pd.DataFrame:
    rows = [
        # game_id, season, game_type, week, home, away, roof
        ("2025_10_RAIN_WET", 2025, "REG", 10, "RAIN", "WET", "outdoors"),
        ("2025_10_DOME_WET", 2025, "REG", 10, "DOME", "WET", "closed"),
        ("2025_10_RAIN2_WET", 2025, "REG", 10, "RAIN2", "WET", "outdoors"),
        ("2025_10_RAIN3_WET", 2025, "REG", 10, "RAIN3", "WET", "outdoors"),
        ("2025_10_RAIN4_WET", 2025, "REG", 10, "RAIN4", "WET", "outdoors"),
        ("2025_20_RAINP_WET", 2025, "POST", 20, "RAINP", "WET", "outdoors"),
    ]
    return pd.DataFrame(
        rows, columns=["game_id", "season", "game_type", "week", "home_team", "away_team", "roof"]
    )


def _forecasts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [
                "2025_10_RAIN_WET",
                "2025_10_DOME_WET",
                "2025_10_RAIN2_WET",
                "2025_10_RAIN3_WET",
                # 2025_10_RAIN4_WET deliberately absent -- missing forecast row
                "2025_20_RAINP_WET",
            ],
            "forecast_precip_prob_pct": [70.0, 70.0, 40.0, 80.0, 70.0],
            "forecast_temp_f": [55.0, 55.0, 55.0, 55.0, 55.0],
            "fetch_status": ["ok"] * 5,
        }
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [
                "2025_10_RAIN_WET",
                "2025_10_DOME_WET",
                "2025_10_RAIN2_WET",
                "2025_10_RAIN3_WET",
                "2025_10_RAIN4_WET",
                "2025_20_RAINP_WET",
            ],
            "season": [2025] * 6,
            "week": [10] * 5 + [20],
            "game_type": ["REG", "REG", "REG", "REG", "REG", "POST"],
            "home_team": ["RAIN", "DOME", "RAIN2", "RAIN3", "RAIN4", "RAINP"],
            "away_team": ["WET"] * 6,
            "kickoff": ["2025-11-09T18:00:00+00:00"] * 6,
            "spread_line": [-3.0, -2.0, -1.5, -1.0, -2.5, -3.0],
            "total_line": [48.0, 48.0, 51.0, 44.0, 48.0, 48.0],
            # G-clean: away pick, flagged (precip 70>=60, total 48>=47) -> flips.
            # G-dome: away pick, flagged on precip/total but indoors -> no flip.
            # G-lowprecip: away pick, precip 40<60 -> no flip.
            # G-lowtotal: away pick, total 44<47 -> no flip.
            # G-nowx: away pick, no forecast row -> no flip.
            # G-post: same shape as G-clean but POST season -> no flip.
            "home_cover_probability": [0.35, 0.35, 0.35, 0.35, 0.35, 0.35],
        }
    )


# ---------------------------------------------------------------------------
# 1. precip_high_total_flag_by_game
# ---------------------------------------------------------------------------


def _total_lines() -> pd.DataFrame:
    return _predictions()[["game_id", "total_line"]]


def test_flag_fires_on_high_precip_and_high_total() -> None:
    flags = precip_high_total_flag_by_game(_schedule(), _forecasts(), _total_lines()).set_index(
        "game_id"
    )
    assert bool(flags.loc["2025_10_RAIN_WET", "precip_high_total_flag"]) is True


def test_flag_does_not_fire_indoors() -> None:
    flags = precip_high_total_flag_by_game(_schedule(), _forecasts(), _total_lines()).set_index(
        "game_id"
    )
    assert bool(flags.loc["2025_10_DOME_WET", "precip_high_total_flag"]) is False


def test_flag_does_not_fire_below_precip_threshold() -> None:
    flags = precip_high_total_flag_by_game(_schedule(), _forecasts(), _total_lines()).set_index(
        "game_id"
    )
    assert bool(flags.loc["2025_10_RAIN2_WET", "precip_high_total_flag"]) is False


def test_flag_does_not_fire_below_total_line_threshold() -> None:
    flags = precip_high_total_flag_by_game(_schedule(), _forecasts(), _total_lines()).set_index(
        "game_id"
    )
    assert bool(flags.loc["2025_10_RAIN3_WET", "precip_high_total_flag"]) is False


def test_flag_does_not_fire_without_a_forecast_row() -> None:
    flags = precip_high_total_flag_by_game(_schedule(), _forecasts(), _total_lines()).set_index(
        "game_id"
    )
    assert bool(flags.loc["2025_10_RAIN4_WET", "precip_high_total_flag"]) is False


def test_flag_by_game_excludes_postseason_rows_entirely() -> None:
    flags = precip_high_total_flag_by_game(_schedule(), _forecasts(), _total_lines())
    assert "2025_20_RAINP_WET" not in set(flags["game_id"])


def test_flag_requires_forecast_columns() -> None:
    with pytest.raises(DataContractError, match="forecasts is missing"):
        precip_high_total_flag_by_game(
            _schedule(), pd.DataFrame({"game_id": ["G1"]}), _total_lines()
        )


def test_flag_requires_schedule_columns() -> None:
    with pytest.raises(DataContractError, match="precip high-total"):
        precip_high_total_flag_by_game(
            pd.DataFrame({"game_id": ["G1"]}), _forecasts(), _total_lines()
        )


def test_flag_requires_total_line_columns() -> None:
    with pytest.raises(DataContractError, match="total_lines is missing"):
        precip_high_total_flag_by_game(_schedule(), _forecasts(), pd.DataFrame({"game_id": ["G1"]}))


# ---------------------------------------------------------------------------
# 2. apply_precip_high_total_tilt_overlay: the pick-level transform
# ---------------------------------------------------------------------------


def test_overlay_flips_away_to_home_on_the_clean_case() -> None:
    result = apply_precip_high_total_tilt_overlay(_predictions(), _schedule(), _forecasts())

    flipped_ids = {flip.game_id for flip in result.flips}
    assert "2025_10_RAIN_WET" in flipped_ids
    flip = next(f for f in result.flips if f.game_id == "2025_10_RAIN_WET")
    assert flip.away_team == "WET"
    assert flip.home_team == "RAIN"

    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2025_10_RAIN_WET", "home_cover_probability"] == pytest.approx(0.65)


def test_overlay_does_not_flip_indoors() -> None:
    result = apply_precip_high_total_tilt_overlay(_predictions(), _schedule(), _forecasts())
    assert all(flip.game_id != "2025_10_DOME_WET" for flip in result.flips)


def test_overlay_does_not_flip_below_precip_threshold() -> None:
    result = apply_precip_high_total_tilt_overlay(_predictions(), _schedule(), _forecasts())
    assert all(flip.game_id != "2025_10_RAIN2_WET" for flip in result.flips)


def test_overlay_does_not_flip_below_total_line_threshold() -> None:
    result = apply_precip_high_total_tilt_overlay(_predictions(), _schedule(), _forecasts())
    assert all(flip.game_id != "2025_10_RAIN3_WET" for flip in result.flips)


def test_overlay_does_not_flip_without_a_forecast() -> None:
    result = apply_precip_high_total_tilt_overlay(_predictions(), _schedule(), _forecasts())
    assert all(flip.game_id != "2025_10_RAIN4_WET" for flip in result.flips)


def test_overlay_leaves_postseason_games_untouched() -> None:
    result = apply_precip_high_total_tilt_overlay(_predictions(), _schedule(), _forecasts())
    assert all(flip.game_id != "2025_20_RAINP_WET" for flip in result.flips)


def test_overlay_never_flips_a_home_pick() -> None:
    """Deliberately asymmetric: a HOME pick on a flagged game is untouched."""

    predictions = _predictions()
    predictions.loc[predictions["game_id"].eq("2025_10_RAIN_WET"), "home_cover_probability"] = 0.60
    result = apply_precip_high_total_tilt_overlay(predictions, _schedule(), _forecasts())
    assert all(flip.game_id != "2025_10_RAIN_WET" for flip in result.flips)


def test_overlay_disabled_is_a_no_op() -> None:
    predictions = _predictions()
    result = apply_precip_high_total_tilt_overlay(
        predictions, _schedule(), _forecasts(), enabled=False
    )
    assert result.flip_count == 0
    pd.testing.assert_frame_equal(
        result.overlaid_predictions.reset_index(drop=True),
        predictions.reset_index(drop=True),
        check_exact=True,
    )


def test_overlay_changes_only_home_cover_probability_on_flipped_rows() -> None:
    predictions = _predictions()
    result = apply_precip_high_total_tilt_overlay(predictions, _schedule(), _forecasts())
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
        apply_precip_high_total_tilt_overlay(
            pd.DataFrame({"game_id": ["G1"]}), _schedule(), _forecasts()
        )


def test_overlay_requires_total_line_on_predictions() -> None:
    predictions = _predictions().drop(columns=["total_line"])
    with pytest.raises(DataContractError, match="total_line"):
        apply_precip_high_total_tilt_overlay(predictions, _schedule(), _forecasts())


# ---------------------------------------------------------------------------
# 3. overlay_disclosure_note
# ---------------------------------------------------------------------------


def test_disclosure_note_is_empty_when_nothing_flipped() -> None:
    only_dome = _predictions().loc[lambda frame: frame["game_id"].eq("2025_10_DOME_WET")]
    result = apply_precip_high_total_tilt_overlay(only_dome, _schedule(), _forecasts())
    assert overlay_disclosure_note(result) == ""


def test_disclosure_note_states_the_flip_count() -> None:
    result = apply_precip_high_total_tilt_overlay(_predictions(), _schedule(), _forecasts())
    note = overlay_disclosure_note(result)
    assert "Tilt applied: 1 pick flipped" in note
    assert "WET at RAIN" in note
    assert "pool-decision" in note
    assert "not applied to the published card" in note


def test_disclosure_note_formats_a_hand_built_flip() -> None:
    result = PrecipHighTotalResult(
        overlaid_predictions=pd.DataFrame({"game_id": ["G1"]}),
        flips=(
            PrecipHighTotalFlip(
                game_id="G1",
                matchup="AW1 at HM1",
                away_team="AW1",
                home_team="HM1",
                forecast_precip_prob_pct=70.0,
                total_line=48.0,
            ),
        ),
        enabled=True,
    )
    note = overlay_disclosure_note(result)
    assert "AWAY -> HOME" in note
    assert "precip 70%" in note
    assert "total 48.0" in note


# ---------------------------------------------------------------------------
# 4. record_forecast_weather_kn_precip_high_total_tilt_challenger_decisions
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
    predictions = _predictions()[
        [
            "game_id",
            "season",
            "week",
            "game_type",
            "home_team",
            "away_team",
            "kickoff",
            "spread_line",
            "total_line",
            "home_cover_probability",
        ]
    ].copy()
    predictions["kickoff"] = ["2025-11-09T18:00:00+00:00"] * 5 + ["2026-01-11T18:00:00+00:00"]
    return predictions


def _write_registry(artifacts: Path, *, status: str = "ACTIVE_PROSPECTIVE") -> None:
    write_challenger_registry(
        artifacts, challenger_id=CHALLENGER_ID, model_config=_MODEL_CONFIG, status=status
    )


_SEASON, _WEEK, _CREATED_AT_UTC = 2025, 10, "2025-11-04T15:00:00+00:00"


def _write_active_model_and_card(artifacts: Path, *, ridge_alpha: float = 10.0) -> None:
    write_active_model_and_card(
        artifacts,
        season=_SEASON,
        week=_WEEK,
        created_at_utc=_CREATED_AT_UTC,
        ridge_alpha=ridge_alpha,
        recommendations=_recorder_predictions(),
    )


def _schedule_with_stadiums() -> pd.DataFrame:
    schedule = _schedule().copy()
    stadium_by_team = {
        "RAIN": "Rain Stadium",
        "DOME": "Dome Arena",
        "RAIN2": "Rain 2 Field",
        "RAIN3": "Rain 3 Field",
        "RAIN4": "Rain 4 Field",
        "RAINP": "Rain Playoff Field",
    }
    schedule["stadium"] = schedule["home_team"].map(stadium_by_team)
    return schedule


def _write_data_root(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    write_snapshot(
        _schedule_with_stadiums(),
        pd.DataFrame({"game_id": [], "team": []}),
        seasons=[2025],
        raw_root=data_root / "raw",
    )
    return data_root


_STADIUM_MAP_CSV = (
    "stadium,icao_station,mappable\n"
    "Rain Stadium,KRN1,True\n"
    "Dome Arena,KDOM,True\n"
    "Rain 2 Field,KRN2,True\n"
    "Rain 3 Field,KRN3,True\n"
    "Rain 4 Field,KRN4,True\n"
    "Rain Playoff Field,KRNP,True\n"
)


def _write_registry_root(tmp_path: Path) -> Path:
    return write_registry_root(tmp_path, stadium_station_map_csv=_STADIUM_MAP_CSV)


def _no_network_stub(station: str, runtime_utc, *, model: str) -> list[dict]:
    return [
        {
            "tmp": 55.0,
            "p06": 70.0,
            "ftime_utc": "2025-11-09T18:00:00+00:00",
            "runtime_utc": "2025-11-04T12:00:00+00:00",
        }
    ]


def test_record_challenger_decisions_records_the_tilt_arm(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)
    registry_root = _write_registry_root(tmp_path)
    now = datetime(2025, 11, 4, 16, 0, tzinfo=UTC)

    result = record_forecast_weather_kn_precip_high_total_tilt_challenger_decisions(
        artifacts, data_root, registry_root, now=now, fetch_bulletin=_no_network_stub
    )

    assert result["recorded"] == 6
    ledger = load_challenger_decisions(artifacts).set_index("game_id")
    assert list(load_challenger_decisions(artifacts).columns) == list(CHALLENGER_DECISION_COLUMNS)
    assert (ledger["bet_side"] == "PASS").all()
    assert ledger["edge"].isna().all()

    assert ledger.loc["2025_10_RAIN_WET", "pick_side"] == "HOME"
    assert ledger.loc["2025_10_DOME_WET", "pick_side"] == "AWAY"

    again = record_forecast_weather_kn_precip_high_total_tilt_challenger_decisions(
        artifacts, data_root, registry_root, now=now, fetch_bulletin=_no_network_stub
    )
    assert again["recorded"] == 0
    assert again["already_recorded"] == 6


def test_record_challenger_decisions_is_fail_open_on_a_missing_station_map(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)
    registry_root = tmp_path / "registry_without_station_map"
    now = datetime(2025, 11, 4, 16, 0, tzinfo=UTC)

    with pytest.warns(RuntimeWarning, match="forecast fetch failed"):
        result = record_forecast_weather_kn_precip_high_total_tilt_challenger_decisions(
            artifacts, data_root, registry_root, now=now
        )

    assert result["recorded"] == 6
    assert result["flip_count"] == 0
    assert result["forecast_cutoff_mode"] == "pool_decision"
    assert result["forecast_fetch_status_counts"] == {"fetch_failed": 6}
    ledger = load_challenger_decisions(artifacts).set_index("game_id")
    assert ledger.loc["2025_10_RAIN_WET", "pick_side"] == "AWAY"


def test_record_challenger_uses_a_supplied_forecasts_frame_without_fetching(tmp_path: Path) -> None:
    """The "one fetch, several consumers" path: when ``forecasts`` is
    supplied (e.g. the frame the warm-team-cold-late sibling already
    fetched), the recorder must not call ``fetch_bulletin`` at all."""

    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)
    registry_root = _write_registry_root(tmp_path)
    now = datetime(2025, 11, 4, 16, 0, tzinfo=UTC)

    def exploding_bulletin(station: str, runtime_utc, *, model: str) -> list[dict]:
        raise AssertionError("fetch_bulletin must not be called when forecasts= is supplied")

    shared_forecasts = pd.DataFrame(
        {
            "game_id": [
                "2025_10_RAIN_WET",
                "2025_10_DOME_WET",
                "2025_10_RAIN2_WET",
                "2025_10_RAIN3_WET",
                "2025_10_RAIN4_WET",
                "2025_20_RAINP_WET",
            ],
            "forecast_temp_f": [55.0, 55.0, 55.0, 55.0, None, 55.0],
            "forecast_precip_prob_pct": [70.0, 70.0, 40.0, 80.0, None, 70.0],
            "fetch_status": ["ok"] * 6,
            "cutoff_mode": ["pool_decision"] * 6,
            "decision_cutoff_utc": ["2025-11-09T18:00:00+00:00"] * 5
            + ["2026-01-11T18:00:00+00:00"],
            "issuance_runtime_utc": ["2025-11-04T12:00:00+00:00"] * 6,
        }
    )

    result = record_forecast_weather_kn_precip_high_total_tilt_challenger_decisions(
        artifacts,
        data_root,
        registry_root,
        now=now,
        fetch_bulletin=exploding_bulletin,
        forecasts=shared_forecasts,
    )
    assert result["recorded"] == 6
    assert result["flip_count"] == 1
    assert result["forecast_cutoff_mode"] == "pool_decision"


def test_record_challenger_refuses_outside_recording_lock_window(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)
    registry_root = _write_registry_root(tmp_path)

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_forecast_weather_kn_precip_high_total_tilt_challenger_decisions(
            artifacts,
            data_root,
            registry_root,
            now=datetime(2025, 8, 1, 1, 0, tzinfo=UTC),
            fetch_bulletin=_no_network_stub,
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_challenger_refuses_a_fingerprint_mismatch(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts, ridge_alpha=1.0)
    data_root = _write_data_root(tmp_path)
    registry_root = _write_registry_root(tmp_path)

    with pytest.raises(DataContractError, match="configuration fingerprint"):
        record_forecast_weather_kn_precip_high_total_tilt_challenger_decisions(
            artifacts,
            data_root,
            registry_root,
            now=datetime(2025, 11, 4, 16, 0, tzinfo=UTC),
            fetch_bulletin=_no_network_stub,
        )
    assert load_challenger_decisions(artifacts).empty


def test_record_challenger_refuses_an_inactive_registration(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts, status="CLOSED_BEFORE_ACTIVATION")
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)
    registry_root = _write_registry_root(tmp_path)

    with pytest.raises(ValueError, match="only ACTIVE_PROSPECTIVE"):
        record_forecast_weather_kn_precip_high_total_tilt_challenger_decisions(
            artifacts,
            data_root,
            registry_root,
            now=datetime(2025, 11, 4, 16, 0, tzinfo=UTC),
            fetch_bulletin=_no_network_stub,
        )
