"""Rain-on-grass underdog challenger (LEAD-37, docs/weather_venue_leads.md).

Mirrors ``tests/test_forecast_weather_kn_precip_high_total_tilt_overlay.py``'s
structure -- this module is a THIRD consumer of the same shared live
kickoff-nearest GFS-MOS fetch, so no real network call is made in any test
here. Load-bearing here:

1. :func:`rain_on_grass_flag_by_game` -- grass surface AND live
   kickoff-nearest forecast precip prob>=60%, missing-data-safe, REG-only.
2. :func:`apply_rain_on_grass_dog_tilt_overlay` -- flips ONLY the clean case
   (a real underdog exists, the flag fires, the model's pick is not already
   on the dog), REG-only, parameter-free, SYMMETRIC (both directions,
   whichever side the market names as the underdog).
3. :func:`record_rain_on_grass_dog_challenger_decisions` writes the overlay's
   own picks to the prospective challenger ledger, dual-tracked and at no
   rotation-registry window cost, and honors a pre-fetched ``forecasts=``
   override (the "one fetch, several consumers" path) without making its own
   network call.
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
from nfl_ats.prospective_scoring import CHALLENGER_DECISION_COLUMNS, load_challenger_decisions
from nfl_ats.rain_on_grass_dog_challenger import (
    CHALLENGER_ID,
    PRECIP_PROB_THRESHOLD_PCT,
    RainOnGrassFlip,
    RainOnGrassResult,
    apply_rain_on_grass_dog_tilt_overlay,
    overlay_disclosure_note,
    rain_on_grass_flag_by_game,
    record_rain_on_grass_dog_challenger_decisions,
)
from nfl_ats.snapshots import write_snapshot

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
#
# HDOG at FAV, grass, precip 70%, home is the underdog (spread -3), model
#   picks AWAY (favorite) -> flips HOME.
# FAV2 at ADOG, grass, precip 70%, AWAY is the underdog (spread +3), model
#   picks HOME (favorite) -> flips AWAY.
# TURF at WET2, turf (not grass), precip 70%, home dog -- no flip.
# GRASS3 at WET3, grass, precip 40% (below 60) -- no flip.
# PICKEM at WET4, grass, precip 70%, spread_line == 0 (no underdog) -- no flip.
# GRASS5 at WET5, grass, precip 70%, home dog, model ALREADY on HOME -- no flip.
# GRASS6 at WET6, grass, NO forecast row at all -- no flip.
# POSTG at POSTW mirrors the clean flagged shape but game_type=POST.


def _schedule() -> pd.DataFrame:
    rows = [
        # game_id, season, game_type, week, home, away, surface
        ("2025_10_HDOG_FAV", 2025, "REG", 10, "HDOG", "FAV", "grass"),
        ("2025_10_FAV2_ADOG", 2025, "REG", 10, "FAV2", "ADOG", "grass"),
        ("2025_10_TURF_WET2", 2025, "REG", 10, "TURF", "WET2", "fieldturf"),
        ("2025_10_GRASS3_WET3", 2025, "REG", 10, "GRASS3", "WET3", "grass"),
        ("2025_10_PICKEM_WET4", 2025, "REG", 10, "PICKEM", "WET4", "grass"),
        ("2025_10_GRASS5_WET5", 2025, "REG", 10, "GRASS5", "WET5", "grass"),
        ("2025_10_GRASS6_WET6", 2025, "REG", 10, "GRASS6", "WET6", "grass"),
        ("2025_20_POSTG_POSTW", 2025, "POST", 20, "POSTG", "POSTW", "grass"),
    ]
    return pd.DataFrame(
        rows,
        columns=["game_id", "season", "game_type", "week", "home_team", "away_team", "surface"],
    )


def _forecasts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [
                "2025_10_HDOG_FAV",
                "2025_10_FAV2_ADOG",
                "2025_10_TURF_WET2",
                "2025_10_GRASS3_WET3",
                "2025_10_PICKEM_WET4",
                "2025_10_GRASS5_WET5",
                # 2025_10_GRASS6_WET6 deliberately absent -- missing forecast row
                "2025_20_POSTG_POSTW",
            ],
            "forecast_precip_prob_pct": [70.0, 70.0, 70.0, 40.0, 70.0, 70.0, 70.0],
            "fetch_status": ["ok"] * 7,
        }
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [
                "2025_10_HDOG_FAV",
                "2025_10_FAV2_ADOG",
                "2025_10_TURF_WET2",
                "2025_10_GRASS3_WET3",
                "2025_10_PICKEM_WET4",
                "2025_10_GRASS5_WET5",
                "2025_10_GRASS6_WET6",
                "2025_20_POSTG_POSTW",
            ],
            "season": [2025] * 8,
            "week": [10] * 7 + [20],
            "game_type": ["REG"] * 7 + ["POST"],
            "home_team": ["HDOG", "FAV2", "TURF", "GRASS3", "PICKEM", "GRASS5", "GRASS6", "POSTG"],
            "away_team": ["FAV", "ADOG", "WET2", "WET3", "WET4", "WET5", "WET6", "POSTW"],
            "kickoff": ["2025-11-09T18:00:00+00:00"] * 8,
            "spread_line": [-3.0, 3.0, -3.0, -3.0, 0.0, -3.0, -3.0, -3.0],
            "home_cover_probability": [0.35, 0.65, 0.35, 0.35, 0.35, 0.60, 0.35, 0.35],
        }
    )


# ---------------------------------------------------------------------------
# 1. rain_on_grass_flag_by_game
# ---------------------------------------------------------------------------


def test_flag_fires_on_grass_and_high_precip() -> None:
    flags = rain_on_grass_flag_by_game(_schedule(), _forecasts()).set_index("game_id")
    assert bool(flags.loc["2025_10_HDOG_FAV", "rain_on_grass_flag"]) is True


def test_flag_does_not_fire_on_turf() -> None:
    flags = rain_on_grass_flag_by_game(_schedule(), _forecasts()).set_index("game_id")
    assert bool(flags.loc["2025_10_TURF_WET2", "rain_on_grass_flag"]) is False


def test_flag_does_not_fire_below_precip_threshold() -> None:
    flags = rain_on_grass_flag_by_game(_schedule(), _forecasts()).set_index("game_id")
    assert bool(flags.loc["2025_10_GRASS3_WET3", "rain_on_grass_flag"]) is False


def test_flag_does_not_fire_without_a_forecast_row() -> None:
    flags = rain_on_grass_flag_by_game(_schedule(), _forecasts()).set_index("game_id")
    assert bool(flags.loc["2025_10_GRASS6_WET6", "rain_on_grass_flag"]) is False


def test_flag_by_game_excludes_postseason_rows_entirely() -> None:
    flags = rain_on_grass_flag_by_game(_schedule(), _forecasts())
    assert "2025_20_POSTG_POSTW" not in set(flags["game_id"])


def test_flag_requires_forecast_columns() -> None:
    with pytest.raises(DataContractError, match="forecasts is missing"):
        rain_on_grass_flag_by_game(_schedule(), pd.DataFrame({"game_id": ["G1"]}))


def test_flag_requires_schedule_columns() -> None:
    with pytest.raises(DataContractError, match="rain-on-grass"):
        rain_on_grass_flag_by_game(pd.DataFrame({"game_id": ["G1"]}), _forecasts())


def test_threshold_is_the_frozen_sixty_percent() -> None:
    assert PRECIP_PROB_THRESHOLD_PCT == 60.0


# ---------------------------------------------------------------------------
# 2. apply_rain_on_grass_dog_tilt_overlay: the pick-level transform
# ---------------------------------------------------------------------------


def test_overlay_flips_to_the_home_underdog() -> None:
    result = apply_rain_on_grass_dog_tilt_overlay(_predictions(), _schedule(), _forecasts())

    flipped_ids = {flip.game_id for flip in result.flips}
    assert "2025_10_HDOG_FAV" in flipped_ids
    flip = next(f for f in result.flips if f.game_id == "2025_10_HDOG_FAV")
    assert flip.underdog_team == "HDOG"

    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2025_10_HDOG_FAV", "home_cover_probability"] == pytest.approx(0.65)


def test_overlay_flips_to_the_away_underdog() -> None:
    """Symmetric direction: the away side is just as flippable as home."""

    result = apply_rain_on_grass_dog_tilt_overlay(_predictions(), _schedule(), _forecasts())

    flipped_ids = {flip.game_id for flip in result.flips}
    assert "2025_10_FAV2_ADOG" in flipped_ids
    flip = next(f for f in result.flips if f.game_id == "2025_10_FAV2_ADOG")
    assert flip.underdog_team == "ADOG"

    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2025_10_FAV2_ADOG", "home_cover_probability"] == pytest.approx(0.35)


def test_overlay_does_not_flip_on_turf() -> None:
    result = apply_rain_on_grass_dog_tilt_overlay(_predictions(), _schedule(), _forecasts())
    assert all(flip.game_id != "2025_10_TURF_WET2" for flip in result.flips)


def test_overlay_does_not_flip_below_precip_threshold() -> None:
    result = apply_rain_on_grass_dog_tilt_overlay(_predictions(), _schedule(), _forecasts())
    assert all(flip.game_id != "2025_10_GRASS3_WET3" for flip in result.flips)


def test_overlay_does_not_flip_a_pickem_with_no_underdog() -> None:
    result = apply_rain_on_grass_dog_tilt_overlay(_predictions(), _schedule(), _forecasts())
    assert all(flip.game_id != "2025_10_PICKEM_WET4" for flip in result.flips)


def test_overlay_leaves_an_already_correct_pick_untouched() -> None:
    result = apply_rain_on_grass_dog_tilt_overlay(_predictions(), _schedule(), _forecasts())
    assert all(flip.game_id != "2025_10_GRASS5_WET5" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2025_10_GRASS5_WET5", "home_cover_probability"] == pytest.approx(0.60)


def test_overlay_does_not_flip_without_a_forecast() -> None:
    result = apply_rain_on_grass_dog_tilt_overlay(_predictions(), _schedule(), _forecasts())
    assert all(flip.game_id != "2025_10_GRASS6_WET6" for flip in result.flips)


def test_overlay_leaves_postseason_games_untouched() -> None:
    result = apply_rain_on_grass_dog_tilt_overlay(_predictions(), _schedule(), _forecasts())
    assert all(flip.game_id != "2025_20_POSTG_POSTW" for flip in result.flips)


def test_overlay_disabled_is_a_no_op() -> None:
    predictions = _predictions()
    result = apply_rain_on_grass_dog_tilt_overlay(
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
    result = apply_rain_on_grass_dog_tilt_overlay(predictions, _schedule(), _forecasts())
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
        apply_rain_on_grass_dog_tilt_overlay(
            pd.DataFrame({"game_id": ["G1"]}), _schedule(), _forecasts()
        )


# ---------------------------------------------------------------------------
# 3. overlay_disclosure_note
# ---------------------------------------------------------------------------


def test_disclosure_note_is_empty_when_nothing_flipped() -> None:
    only_turf = _predictions().loc[lambda frame: frame["game_id"].eq("2025_10_TURF_WET2")]
    result = apply_rain_on_grass_dog_tilt_overlay(only_turf, _schedule(), _forecasts())
    assert overlay_disclosure_note(result) == ""


def test_disclosure_note_states_the_flip_count() -> None:
    result = apply_rain_on_grass_dog_tilt_overlay(_predictions(), _schedule(), _forecasts())
    note = overlay_disclosure_note(result)
    assert "Tilt applied: 2 picks flipped" in note
    assert "pool-decision" in note
    assert "not applied to the published card" in note


def test_disclosure_note_formats_a_hand_built_flip() -> None:
    result = RainOnGrassResult(
        overlaid_predictions=pd.DataFrame({"game_id": ["G1"]}),
        flips=(
            RainOnGrassFlip(
                game_id="G1",
                matchup="AW1 at HM1",
                underdog_team="HM1",
                forecast_precip_prob_pct=70.0,
                spread_line=-3.0,
            ),
        ),
        enabled=True,
    )
    note = overlay_disclosure_note(result)
    assert "-> HM1" in note
    assert "precip 70%" in note


# ---------------------------------------------------------------------------
# 4. record_rain_on_grass_dog_challenger_decisions
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
            "home_cover_probability",
        ]
    ].copy()
    predictions["kickoff"] = ["2025-11-09T18:00:00+00:00"] * 7 + ["2026-01-11T18:00:00+00:00"]
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


def _write_data_root(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    schedule = _schedule().copy()
    stadium_by_team = {
        "HDOG": "Hdog Stadium",
        "FAV2": "Fav2 Stadium",
        "TURF": "Turf Stadium",
        "GRASS3": "Grass3 Field",
        "PICKEM": "Pickem Field",
        "GRASS5": "Grass5 Field",
        "GRASS6": "Grass6 Field",
        "POSTG": "Postg Field",
    }
    schedule["stadium"] = schedule["home_team"].map(stadium_by_team)
    write_snapshot(
        schedule,
        pd.DataFrame({"game_id": [], "team": []}),
        seasons=[2025],
        raw_root=data_root / "raw",
    )
    return data_root


_STADIUM_MAP_CSV = (
    "stadium,icao_station,mappable\n"
    "Hdog Stadium,KHD1,True\n"
    "Fav2 Stadium,KFV2,True\n"
    "Turf Stadium,KTRF,True\n"
    "Grass3 Field,KGR3,True\n"
    "Pickem Field,KPCK,True\n"
    "Grass5 Field,KGR5,True\n"
    "Grass6 Field,KGR6,True\n"
    "Postg Field,KPST,True\n"
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


@pytest.mark.full  # ENG-11: dominates --durations
def test_record_challenger_decisions_records_the_tilt_arm(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)
    registry_root = _write_registry_root(tmp_path)
    now = datetime(2025, 11, 4, 16, 0, tzinfo=UTC)

    result = record_rain_on_grass_dog_challenger_decisions(
        artifacts, data_root, registry_root, now=now, fetch_bulletin=_no_network_stub
    )

    assert result["recorded"] == 8
    ledger = load_challenger_decisions(artifacts).set_index("game_id")
    assert list(load_challenger_decisions(artifacts).columns) == list(CHALLENGER_DECISION_COLUMNS)
    assert (ledger["bet_side"] == "PASS").all()
    assert ledger["edge"].isna().all()

    assert ledger.loc["2025_10_HDOG_FAV", "pick_side"] == "HOME"
    assert ledger.loc["2025_10_TURF_WET2", "pick_side"] == "AWAY"
    paired_path = artifacts / "prospective" / f"{CHALLENGER_ID}_paired_decisions.parquet"
    paired = pd.read_parquet(paired_path).set_index("game_id")
    assert paired.loc["2025_10_HDOG_FAV", "baseline_pick_side"] == "AWAY"
    assert paired.loc["2025_10_HDOG_FAV", "pick_side"] == "HOME"
    paired_bytes = paired_path.read_bytes()

    again = record_rain_on_grass_dog_challenger_decisions(
        artifacts, data_root, registry_root, now=now, fetch_bulletin=_no_network_stub
    )
    assert again["recorded"] == 0
    assert again["already_recorded"] == 8
    assert paired_path.read_bytes() == paired_bytes


def test_record_challenger_decisions_is_fail_open_on_a_missing_station_map(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)
    registry_root = tmp_path / "registry_without_station_map"
    now = datetime(2025, 11, 4, 16, 0, tzinfo=UTC)

    with pytest.warns(RuntimeWarning, match="forecast fetch failed"):
        result = record_rain_on_grass_dog_challenger_decisions(
            artifacts, data_root, registry_root, now=now
        )

    assert result["recorded"] == 0
    assert result["skipped"] is True
    assert result["flip_count"] == 0
    assert result["forecast_cutoff_mode"] == "pool_decision"
    assert load_challenger_decisions(artifacts).empty


def test_recorder_rejects_forecasts_issued_after_recording(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)

    def future_bulletin(station: str, runtime_utc, *, model: str) -> list[dict]:
        rows = _no_network_stub(station, runtime_utc, model=model)
        rows[0]["runtime_utc"] = "2025-11-05T12:00:00+00:00"
        return rows

    with pytest.raises(DataContractError, match="post-recording issuance"):
        record_rain_on_grass_dog_challenger_decisions(
            artifacts,
            _write_data_root(tmp_path),
            _write_registry_root(tmp_path),
            now=datetime(2025, 11, 4, 16, tzinfo=UTC),
            fetch_bulletin=future_bulletin,
        )
    assert load_challenger_decisions(artifacts).empty


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

    game_ids = [
        "2025_10_HDOG_FAV",
        "2025_10_FAV2_ADOG",
        "2025_10_TURF_WET2",
        "2025_10_GRASS3_WET3",
        "2025_10_PICKEM_WET4",
        "2025_10_GRASS5_WET5",
        "2025_10_GRASS6_WET6",
        "2025_20_POSTG_POSTW",
    ]
    shared_forecasts = pd.DataFrame(
        {
            "game_id": game_ids,
            "forecast_temp_f": [55.0] * 8,
            "forecast_precip_prob_pct": [70.0, 70.0, 70.0, 40.0, 70.0, 70.0, None, 70.0],
            "fetch_status": ["ok"] * 8,
            "cutoff_mode": ["pool_decision"] * 8,
            "decision_cutoff_utc": ["2025-11-09T18:00:00+00:00"] * 7
            + ["2026-01-11T18:00:00+00:00"],
            "issuance_runtime_utc": ["2025-11-04T12:00:00+00:00"] * 8,
        }
    )

    result = record_rain_on_grass_dog_challenger_decisions(
        artifacts,
        data_root,
        registry_root,
        now=now,
        fetch_bulletin=exploding_bulletin,
        forecasts=shared_forecasts,
    )
    assert result["recorded"] == 8
    assert result["flip_count"] == 2
    assert result["forecast_cutoff_mode"] == "pool_decision"


@pytest.mark.full  # ENG-11: dominates --durations
def test_record_challenger_refuses_outside_recording_lock_window(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)
    registry_root = _write_registry_root(tmp_path)

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_rain_on_grass_dog_challenger_decisions(
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
        record_rain_on_grass_dog_challenger_decisions(
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
        record_rain_on_grass_dog_challenger_decisions(
            artifacts,
            data_root,
            registry_root,
            now=datetime(2025, 11, 4, 16, 0, tzinfo=UTC),
            fetch_bulletin=_no_network_stub,
        )


# ---------------------------------------------------------------------------
# 5. Registration self-consistency (the TRACKED registry entry)
# ---------------------------------------------------------------------------


def test_real_registry_entry_fingerprint_is_internally_consistent() -> None:
    import json

    from nfl_ats.prospective_scoring import config_fingerprint

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
