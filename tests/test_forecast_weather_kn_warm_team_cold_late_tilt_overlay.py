"""Forecast (kickoff-nearest) warm-team-cold-late tilt overlay
(docs/forecast_weather_screen.md, "Wiring recommendations" #1).

Mirrors ``tests/test_forecast_cold_visitor_tilt_overlay.py``'s structure.
Load-bearing here:

1. :func:`warm_team_cold_late_flag_by_game` -- the frozen flag definition
   (warm-metro away team AND outdoor AND kickoff-nearest forecast temp<=35F
   AND week>=13), missing-data-safe.
2. :func:`apply_warm_team_cold_late_tilt_overlay` -- flips ONLY the clean
   case (away pick, flag fires), REG-only, parameter-free, asymmetric.
3. The live-fetch layer is FAIL-OPEN and captures BOTH temperature and
   precipitation from the same bulletin (shared with the precip-high-total
   sibling challenger): no real network call is made in any test here
   (every ``fetch_bulletin`` is a local stub), and a total fetch failure
   (missing station map) is proven to fold into zero flags with a logged
   warning, never an exception.
4. :func:`record_forecast_weather_kn_warm_team_cold_late_tilt_challenger_decisions`
   writes the overlay's own picks to the prospective challenger ledger,
   dual-tracked and at no rotation-registry window cost, and honors a
   pre-fetched ``forecasts=`` override (the "one fetch, several consumers"
   path) without making its own network call.
"""

from __future__ import annotations

import warnings
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
from nfl_ats.forecast_cold_visitor_tilt_overlay import MosFetchError
from nfl_ats.forecast_weather_kn_warm_team_cold_late_tilt_overlay import (
    CHALLENGER_ID,
    WarmTeamColdLateFlip,
    WarmTeamColdLateResult,
    apply_warm_team_cold_late_tilt_overlay,
    fetch_kickoff_nearest_forecasts_fail_open,
    fetch_one_game_kickoff_nearest,
    overlay_disclosure_note,
    record_forecast_weather_kn_warm_team_cold_late_tilt_challenger_decisions,
    warm_team_cold_late_flag_by_game,
)
from nfl_ats.prospective_scoring import CHALLENGER_DECISION_COLUMNS, load_challenger_decisions
from nfl_ats.snapshots import write_snapshot

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
#
# MIA (away, a real code in WARM_METRO_TEAM_CODES) plays at FROST week 15,
#   outdoor, forecast 30F -> flagged. Model's pick is AWAY (MIA) -> flips HOME.
# MIA plays at DOME week 15 (roof=closed, forecast 20F): flag would fire on
#   temp alone, but the game itself is indoors -> no flip.
# MIA plays at FROST2 week 15 (outdoor, NO forecast row at all): missing
#   forecast -> no signal -> no flip.
# MIA plays at FROST3 week 8 (outdoor, forecast 20F): week<13 -> no flip.
# COLD (a fictional code NOT in WARM_METRO_TEAM_CODES) plays at FROST4 week
#   15 (outdoor, forecast 20F): team not on the static list -> no flip.
# MIA at FROSTP mirrors the week-15 flagged shape but game_type=POST.


def _schedule() -> pd.DataFrame:
    rows = [
        # game_id, season, game_type, week, home, away, roof
        ("2025_15_FROST_WARM", 2025, "REG", 15, "FROST", "MIA", "outdoors"),
        ("2025_15_DOME_WARM", 2025, "REG", 15, "DOME", "MIA", "closed"),
        ("2025_15_FROST2_WARM", 2025, "REG", 15, "FROST2", "MIA", "outdoors"),
        ("2025_08_FROST3_WARM", 2025, "REG", 8, "FROST3", "MIA", "outdoors"),
        ("2025_15_FROST4_COLD", 2025, "REG", 15, "FROST4", "COLD", "outdoors"),
        ("2025_20_FROSTP_WARM", 2025, "POST", 20, "FROSTP", "MIA", "outdoors"),
    ]
    return pd.DataFrame(
        rows, columns=["game_id", "season", "game_type", "week", "home_team", "away_team", "roof"]
    )


def _forecasts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [
                "2025_15_FROST_WARM",
                "2025_15_DOME_WARM",
                # 2025_15_FROST2_WARM deliberately absent -- missing forecast row
                "2025_08_FROST3_WARM",
                "2025_15_FROST4_COLD",
                "2025_20_FROSTP_WARM",
            ],
            "forecast_temp_f": [30.0, 20.0, 20.0, 20.0, 25.0],
            "fetch_status": ["ok"] * 5,
        }
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [
                "2025_15_FROST_WARM",
                "2025_15_DOME_WARM",
                "2025_15_FROST2_WARM",
                "2025_08_FROST3_WARM",
                "2025_15_FROST4_COLD",
                "2025_20_FROSTP_WARM",
            ],
            "season": [2025] * 6,
            "week": [15, 15, 15, 8, 15, 20],
            "game_type": ["REG", "REG", "REG", "REG", "REG", "POST"],
            "home_team": ["FROST", "DOME", "FROST2", "FROST3", "FROST4", "FROSTP"],
            "away_team": ["MIA", "MIA", "MIA", "MIA", "COLD", "MIA"],
            "kickoff": ["2025-12-14T18:00:00+00:00"] * 6,
            "spread_line": [-3.0, -2.0, -1.5, -1.0, -2.5, -3.0],
            # G-clean: away pick (MIA), flagged -> flips to HOME.
            # G-dome: away pick (MIA), indoors -> no flip.
            # G-nowx: away pick (MIA), no forecast -> no flip.
            # G-early: away pick (MIA), week<13 -> no flip.
            # G-notlisted: away pick (COLD, not on the warm-metro list) -> no flip.
            # G-post: same shape as G-clean but POST season -> no flip.
            "home_cover_probability": [0.35, 0.35, 0.35, 0.35, 0.35, 0.35],
        }
    )


# ---------------------------------------------------------------------------
# 1. warm_team_cold_late_flag_by_game
# ---------------------------------------------------------------------------


def test_flag_fires_on_warm_metro_visitor_cold_late_forecast() -> None:
    flags = warm_team_cold_late_flag_by_game(_schedule(), _forecasts()).set_index("game_id")
    assert bool(flags.loc["2025_15_FROST_WARM", "warm_team_cold_late_flag"]) is True


def test_flag_does_not_fire_indoors() -> None:
    flags = warm_team_cold_late_flag_by_game(_schedule(), _forecasts()).set_index("game_id")
    assert bool(flags.loc["2025_15_DOME_WARM", "warm_team_cold_late_flag"]) is False


def test_flag_does_not_fire_without_a_forecast_row() -> None:
    flags = warm_team_cold_late_flag_by_game(_schedule(), _forecasts()).set_index("game_id")
    assert bool(flags.loc["2025_15_FROST2_WARM", "warm_team_cold_late_flag"]) is False


def test_flag_does_not_fire_before_week_13() -> None:
    flags = warm_team_cold_late_flag_by_game(_schedule(), _forecasts()).set_index("game_id")
    assert bool(flags.loc["2025_08_FROST3_WARM", "warm_team_cold_late_flag"]) is False


def test_flag_does_not_fire_for_a_team_off_the_warm_metro_list() -> None:
    flags = warm_team_cold_late_flag_by_game(_schedule(), _forecasts()).set_index("game_id")
    assert bool(flags.loc["2025_15_FROST4_COLD", "warm_team_cold_late_flag"]) is False


def test_flag_by_game_excludes_postseason_rows_entirely() -> None:
    flags = warm_team_cold_late_flag_by_game(_schedule(), _forecasts())
    assert "2025_20_FROSTP_WARM" not in set(flags["game_id"])


def test_flag_requires_forecast_columns() -> None:
    with pytest.raises(DataContractError, match="forecasts is missing"):
        warm_team_cold_late_flag_by_game(_schedule(), pd.DataFrame({"game_id": ["G1"]}))


def test_flag_requires_schedule_columns() -> None:
    with pytest.raises(DataContractError, match="warm-team cold-late"):
        warm_team_cold_late_flag_by_game(pd.DataFrame({"game_id": ["G1"]}), _forecasts())


# ---------------------------------------------------------------------------
# 2. apply_warm_team_cold_late_tilt_overlay: the pick-level transform
# ---------------------------------------------------------------------------


def test_overlay_flips_away_to_home_on_the_clean_case() -> None:
    result = apply_warm_team_cold_late_tilt_overlay(_predictions(), _schedule(), _forecasts())

    flipped_ids = {flip.game_id for flip in result.flips}
    assert "2025_15_FROST_WARM" in flipped_ids
    flip = next(f for f in result.flips if f.game_id == "2025_15_FROST_WARM")
    assert flip.away_team == "MIA"
    assert flip.home_team == "FROST"

    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2025_15_FROST_WARM", "home_cover_probability"] == pytest.approx(0.65)


def test_overlay_does_not_flip_indoors() -> None:
    result = apply_warm_team_cold_late_tilt_overlay(_predictions(), _schedule(), _forecasts())
    assert all(flip.game_id != "2025_15_DOME_WARM" for flip in result.flips)


def test_overlay_does_not_flip_without_a_forecast() -> None:
    result = apply_warm_team_cold_late_tilt_overlay(_predictions(), _schedule(), _forecasts())
    assert all(flip.game_id != "2025_15_FROST2_WARM" for flip in result.flips)


def test_overlay_does_not_flip_before_week_13() -> None:
    result = apply_warm_team_cold_late_tilt_overlay(_predictions(), _schedule(), _forecasts())
    assert all(flip.game_id != "2025_08_FROST3_WARM" for flip in result.flips)


def test_overlay_leaves_postseason_games_untouched() -> None:
    result = apply_warm_team_cold_late_tilt_overlay(_predictions(), _schedule(), _forecasts())
    assert all(flip.game_id != "2025_20_FROSTP_WARM" for flip in result.flips)


def test_overlay_never_flips_a_home_pick() -> None:
    """Deliberately asymmetric: a HOME pick on a flagged game is untouched."""

    predictions = _predictions()
    predictions.loc[predictions["game_id"].eq("2025_15_FROST_WARM"), "home_cover_probability"] = (
        0.60
    )
    result = apply_warm_team_cold_late_tilt_overlay(predictions, _schedule(), _forecasts())
    assert all(flip.game_id != "2025_15_FROST_WARM" for flip in result.flips)


def test_overlay_disabled_is_a_no_op() -> None:
    predictions = _predictions()
    result = apply_warm_team_cold_late_tilt_overlay(
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
    result = apply_warm_team_cold_late_tilt_overlay(predictions, _schedule(), _forecasts())
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
        apply_warm_team_cold_late_tilt_overlay(
            pd.DataFrame({"game_id": ["G1"]}), _schedule(), _forecasts()
        )


# ---------------------------------------------------------------------------
# 3. overlay_disclosure_note
# ---------------------------------------------------------------------------


def test_disclosure_note_is_empty_when_nothing_flipped() -> None:
    only_dome = _predictions().loc[lambda frame: frame["game_id"].eq("2025_15_DOME_WARM")]
    result = apply_warm_team_cold_late_tilt_overlay(only_dome, _schedule(), _forecasts())
    assert overlay_disclosure_note(result) == ""


def test_disclosure_note_states_the_flip_count() -> None:
    result = apply_warm_team_cold_late_tilt_overlay(_predictions(), _schedule(), _forecasts())
    note = overlay_disclosure_note(result)
    assert "Tilt applied: 1 pick flipped" in note
    assert "MIA at FROST" in note
    assert "pool-decision" in note
    assert "not applied to the published card" in note


def test_disclosure_note_formats_a_hand_built_flip() -> None:
    result = WarmTeamColdLateResult(
        overlaid_predictions=pd.DataFrame({"game_id": ["G1"]}),
        flips=(
            WarmTeamColdLateFlip(
                game_id="G1",
                matchup="AW1 at HM1",
                away_team="AW1",
                home_team="HM1",
                forecast_temp_f=30.0,
            ),
        ),
        enabled=True,
    )
    note = overlay_disclosure_note(result)
    assert "AWAY -> HOME" in note
    assert "forecast 30F" in note


# ---------------------------------------------------------------------------
# 4. Live fetch layer -- FAIL-OPEN, kickoff_nearest/GFS, no real network call
# ---------------------------------------------------------------------------


def _stub_bulletin_rows(
    temp_f: float,
    precip_pct: float | None = None,
    ftime_utc: str = "2025-12-14T18:00:00+00:00",
) -> list[dict]:
    row = {"tmp": temp_f, "ftime_utc": ftime_utc, "runtime_utc": "2025-12-09T12:00:00+00:00"}
    if precip_pct is not None:
        row["p06"] = precip_pct
    return [row]


def test_fetch_one_game_kickoff_nearest_returns_ok_with_temp_and_precip() -> None:
    def stub(station: str, runtime_utc, *, model: str) -> list[dict]:
        assert model == "GFS"
        return _stub_bulletin_rows(30.0, 40.0)

    kickoff = pd.Timestamp("2025-12-14T18:00:00+00:00")
    result = fetch_one_game_kickoff_nearest("KXYZ", kickoff, fetch_bulletin=stub, delay_seconds=0.0)
    assert result["forecast_temp_f"] == 30.0
    assert result["forecast_precip_prob_pct"] == 40.0
    assert result["fetch_status"] == "ok"
    assert result["cutoff_mode"] == "pool_decision"
    assert result["decision_cutoff_utc"] == "2025-12-14T18:00:00+00:00"
    assert result["issuance_runtime_utc"] == "2025-12-09T12:00:00+00:00"


def test_fetch_one_game_kickoff_nearest_handles_missing_precip_field() -> None:
    def stub(station: str, runtime_utc, *, model: str) -> list[dict]:
        return _stub_bulletin_rows(30.0, None)

    kickoff = pd.Timestamp("2025-12-14T18:00:00+00:00")
    result = fetch_one_game_kickoff_nearest("KXYZ", kickoff, fetch_bulletin=stub, delay_seconds=0.0)
    assert result["forecast_temp_f"] == pytest.approx(30.0)
    assert result["forecast_precip_prob_pct"] is None
    assert result["fetch_status"] == "ok"


def test_fetch_one_game_kickoff_nearest_exhausts_lookback_without_a_bulletin() -> None:
    def stub(station: str, runtime_utc, *, model: str) -> list[dict]:
        return []

    kickoff = pd.Timestamp("2025-12-14T18:00:00+00:00")
    result = fetch_one_game_kickoff_nearest(
        "KXYZ", kickoff, fetch_bulletin=stub, max_lookback_steps=2, delay_seconds=0.0
    )
    assert result["forecast_temp_f"] is None
    assert result["forecast_precip_prob_pct"] is None
    assert result["fetch_status"] == "no_bulletin_within_lookback"
    assert result["cutoff_mode"] == "pool_decision"


def test_fetch_one_game_kickoff_nearest_folds_a_transport_error_into_a_status() -> None:
    def stub(station: str, runtime_utc, *, model: str) -> list[dict]:
        raise MosFetchError("simulated network down")

    kickoff = pd.Timestamp("2025-12-14T18:00:00+00:00")
    result = fetch_one_game_kickoff_nearest(
        "KXYZ", kickoff, fetch_bulletin=stub, max_lookback_steps=1, delay_seconds=0.0
    )
    assert result["forecast_temp_f"] is None
    assert result["forecast_precip_prob_pct"] is None
    assert result["fetch_status"] == "transport_error"
    assert result["cutoff_mode"] == "pool_decision"


def test_live_mnf_fetch_never_requests_a_post_lock_bulletin() -> None:
    requested = []

    def stub(station: str, runtime_utc, *, model: str) -> list[dict]:
        requested.append(runtime_utc)
        return []

    kickoff = pd.Timestamp("2025-09-09T00:15:00+00:00")
    fetch_one_game_kickoff_nearest(
        "KXYZ",
        kickoff,
        fetch_bulletin=stub,
        max_lookback_steps=2,
        delay_seconds=0.0,
    )

    sunday_lock = pd.Timestamp("2025-09-07T20:00:00+00:00")
    assert requested
    assert all(pd.Timestamp(runtime) <= sunday_lock for runtime in requested)
    assert pd.Timestamp(requested[0]) == pd.Timestamp("2025-09-07T12:00:00+00:00")


def test_live_snf_fetch_rejects_a_bulletin_labeled_after_the_lock() -> None:
    def stub(station: str, runtime_utc, *, model: str) -> list[dict]:
        return _stub_bulletin_rows(
            30.0,
            40.0,
            ftime_utc="2025-09-08T00:00:00+00:00",
        )

    # Simulate a malformed provider response: its row claims a runtime later
    # than the requested Sunday cycle.
    def malformed(station: str, runtime_utc, *, model: str) -> list[dict]:
        rows = stub(station, runtime_utc, model=model)
        rows[0]["runtime_utc"] = "2025-09-08T00:00:00+00:00"
        return rows

    result = fetch_one_game_kickoff_nearest(
        "KXYZ",
        pd.Timestamp("2025-09-08T00:20:00+00:00"),
        fetch_bulletin=malformed,
        delay_seconds=0.0,
    )
    assert result["fetch_status"] == "invalid_issuance_timestamp"
    assert result["forecast_temp_f"] is None


def test_fetch_fail_open_returns_zero_flags_on_a_missing_station_map(tmp_path: Path) -> None:
    """THE no-network unit test: a total fetch failure must fold into zero
    flags with a logged warning, never an exception, and the resulting
    forecasts frame must make the overlay a complete no-op."""

    games = pd.DataFrame(
        {
            "game_id": ["2025_15_FROST_WARM"],
            "stadium": ["Frost Stadium"],
            "kickoff": ["2025-12-14T18:00:00+00:00"],
        }
    )
    missing_station_map = tmp_path / "does_not_exist.csv"

    with pytest.warns(RuntimeWarning, match="forecast fetch failed"):
        forecasts = fetch_kickoff_nearest_forecasts_fail_open(games, missing_station_map)

    assert forecasts["fetch_status"].eq("fetch_failed").all()
    assert forecasts["forecast_temp_f"].isna().all()
    assert forecasts["forecast_precip_prob_pct"].isna().all()
    assert forecasts["cutoff_mode"].eq("pool_decision").all()

    result = apply_warm_team_cold_late_tilt_overlay(_predictions(), _schedule(), forecasts)
    assert result.flip_count == 0


def test_fetch_fail_open_works_end_to_end_with_a_stub_bulletin(tmp_path: Path) -> None:
    station_map = tmp_path / "stadium_station_map.csv"
    station_map.write_text(
        "stadium,icao_station,mappable\nFrost Stadium,KFRS,True\n", encoding="utf-8"
    )
    games = pd.DataFrame(
        {
            "game_id": ["2025_15_FROST_WARM"],
            "stadium": ["Frost Stadium"],
            "kickoff": ["2025-12-14T18:00:00+00:00"],
        }
    )

    def stub(station: str, runtime_utc, *, model: str) -> list[dict]:
        assert station == "KFRS"
        assert model == "GFS"
        return _stub_bulletin_rows(30.0, 40.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a working fetch must not warn at all
        forecasts = fetch_kickoff_nearest_forecasts_fail_open(
            games, station_map, fetch_bulletin=stub, delay_seconds=0.0
        )
    row = forecasts.set_index("game_id").loc["2025_15_FROST_WARM"]
    assert row["forecast_temp_f"] == 30.0
    assert row["forecast_precip_prob_pct"] == 40.0
    assert row["fetch_status"] == "ok"


# ---------------------------------------------------------------------------
# 5. record_forecast_weather_kn_warm_team_cold_late_tilt_challenger_decisions
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
    predictions["kickoff"] = [
        "2025-12-14T18:00:00+00:00",
        "2025-12-14T18:00:00+00:00",
        "2025-12-14T18:00:00+00:00",
        # The week-8 game's real gameday is earlier in the season than the
        # other rows' week-15 kickoff, which would put it in the past
        # relative to this fixture's recording instant (now=2025-12-09) --
        # kept a future timestamp here instead so the recorder-level test
        # can assert on ALL six rows being recorded pre-kickoff, matching
        # how a real multi-week card's kickoffs would all still be in the
        # future (mirrors forecast_cold_visitor_tilt_overlay's own test
        # fixture, which makes the identical adjustment for its ROOKIE row).
        "2025-12-14T18:00:00+00:00",
        "2025-12-14T18:00:00+00:00",
        "2026-01-11T18:00:00+00:00",
    ]
    return predictions


def _write_registry(artifacts: Path, *, status: str = "ACTIVE_PROSPECTIVE") -> None:
    write_challenger_registry(
        artifacts, challenger_id=CHALLENGER_ID, model_config=_MODEL_CONFIG, status=status
    )


_SEASON, _WEEK, _CREATED_AT_UTC = 2025, 15, "2025-12-09T15:00:00+00:00"


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
        "FROST": "Frost Stadium",
        "DOME": "Dome Arena",
        "FROST2": "Frost 2 Field",
        "FROST3": "Frost 3 Field",
        "FROST4": "Frost 4 Field",
        "FROSTP": "Frost Playoff Field",
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
    "Frost Stadium,KFRS,True\n"
    "Dome Arena,KDOM,True\n"
    "Frost 2 Field,KFR2,True\n"
    "Frost 3 Field,KFR3,True\n"
    "Frost 4 Field,KFR4,True\n"
    "Frost Playoff Field,KFRP,True\n"
)


def _write_registry_root(tmp_path: Path) -> Path:
    return write_registry_root(tmp_path, stadium_station_map_csv=_STADIUM_MAP_CSV)


def _no_network_stub(station: str, runtime_utc, *, model: str) -> list[dict]:
    return _stub_bulletin_rows(30.0, 40.0, ftime_utc="2025-12-14T18:00:00+00:00")


@pytest.mark.full  # ENG-11: dominates --durations
def test_record_challenger_decisions_records_the_tilt_arm(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)
    registry_root = _write_registry_root(tmp_path)
    now = datetime(2025, 12, 9, 16, 0, tzinfo=UTC)

    result = record_forecast_weather_kn_warm_team_cold_late_tilt_challenger_decisions(
        artifacts, data_root, registry_root, now=now, fetch_bulletin=_no_network_stub
    )

    assert result["recorded"] == 6
    ledger = load_challenger_decisions(artifacts).set_index("game_id")
    assert list(load_challenger_decisions(artifacts).columns) == list(CHALLENGER_DECISION_COLUMNS)
    assert (ledger["bet_side"] == "PASS").all()
    assert ledger["edge"].isna().all()

    assert ledger.loc["2025_15_FROST_WARM", "pick_side"] == "HOME"
    assert ledger.loc["2025_15_DOME_WARM", "pick_side"] == "AWAY"

    again = record_forecast_weather_kn_warm_team_cold_late_tilt_challenger_decisions(
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
    now = datetime(2025, 12, 9, 16, 0, tzinfo=UTC)

    with pytest.warns(RuntimeWarning, match="forecast fetch failed"):
        result = record_forecast_weather_kn_warm_team_cold_late_tilt_challenger_decisions(
            artifacts, data_root, registry_root, now=now
        )

    assert result["recorded"] == 6
    assert result["flip_count"] == 0
    assert result["forecast_cutoff_mode"] == "pool_decision"
    assert result["forecast_fetch_status_counts"] == {"fetch_failed": 6}
    ledger = load_challenger_decisions(artifacts).set_index("game_id")
    assert ledger.loc["2025_15_FROST_WARM", "pick_side"] == "AWAY"


def test_record_challenger_uses_a_supplied_forecasts_frame_without_fetching(tmp_path: Path) -> None:
    """The "one fetch, several consumers" path: when ``forecasts`` is
    supplied, the recorder must not call ``fetch_bulletin`` at all."""

    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)
    registry_root = _write_registry_root(tmp_path)
    now = datetime(2025, 12, 9, 16, 0, tzinfo=UTC)

    def exploding_bulletin(station: str, runtime_utc, *, model: str) -> list[dict]:
        raise AssertionError("fetch_bulletin must not be called when forecasts= is supplied")

    shared_forecasts = pd.DataFrame(
        {
            "game_id": [
                "2025_15_FROST_WARM",
                "2025_15_DOME_WARM",
                "2025_15_FROST2_WARM",
                "2025_08_FROST3_WARM",
                "2025_15_FROST4_COLD",
                "2025_20_FROSTP_WARM",
            ],
            "forecast_temp_f": [30.0, 20.0, None, 20.0, 20.0, 25.0],
            "forecast_precip_prob_pct": [10.0, 10.0, None, 10.0, 10.0, 10.0],
            "fetch_status": ["ok"] * 6,
            "cutoff_mode": ["pool_decision"] * 6,
            "decision_cutoff_utc": ["2025-12-14T18:00:00+00:00"] * 5
            + ["2026-01-11T18:00:00+00:00"],
            "issuance_runtime_utc": ["2025-12-09T12:00:00+00:00"] * 6,
        }
    )

    with pytest.raises(DataContractError, match="lack pool-decision provenance"):
        record_forecast_weather_kn_warm_team_cold_late_tilt_challenger_decisions(
            artifacts,
            data_root,
            registry_root,
            now=now,
            fetch_bulletin=exploding_bulletin,
            forecasts=shared_forecasts.drop(columns="cutoff_mode"),
        )

    result = record_forecast_weather_kn_warm_team_cold_late_tilt_challenger_decisions(
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


@pytest.mark.full  # ENG-11: dominates --durations
def test_record_challenger_refuses_outside_recording_lock_window(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)
    registry_root = _write_registry_root(tmp_path)

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_forecast_weather_kn_warm_team_cold_late_tilt_challenger_decisions(
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
        record_forecast_weather_kn_warm_team_cold_late_tilt_challenger_decisions(
            artifacts,
            data_root,
            registry_root,
            now=datetime(2025, 12, 9, 16, 0, tzinfo=UTC),
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
        record_forecast_weather_kn_warm_team_cold_late_tilt_challenger_decisions(
            artifacts,
            data_root,
            registry_root,
            now=datetime(2025, 12, 9, 16, 0, tzinfo=UTC),
            fetch_bulletin=_no_network_stub,
        )
