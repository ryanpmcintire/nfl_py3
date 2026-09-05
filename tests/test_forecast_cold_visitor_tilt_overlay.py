"""Forecast cold-visitor tilt overlay (docs/forecast_weather_screen.md, ENV-01).

Mirrors ``tests/test_surface_switch_tilt_overlay.py``/
``tests/test_backup_qb_fade_overlay.py``'s structure. Five things are
load-bearing here:

1. :func:`team_climate_temp_by_away_game` -- pregame-safe climatology (a
   team's own climate baseline is derived only from its STRICTLY EARLIER
   outdoor home games, across any season, never a future or same-day game).
2. :func:`forecast_cold_visitor_flag_by_game` -- the frozen flag definition
   (outdoor AND climate minus forecast >= 25F), missing-data-safe.
3. :func:`apply_forecast_cold_visitor_tilt_overlay` -- flips ONLY the clean
   case (away pick, flag fires), REG-only, parameter-free, asymmetric.
4. The live-fetch layer is FAIL-OPEN: no real network call is made in any
   test here (every ``fetch_bulletin`` is a local stub), and a total fetch
   failure (missing station map) is proven to fold into zero flags with a
   logged warning, never an exception -- the no-network unit test this
   overlay's build task explicitly requires.
5. :func:`record_forecast_cold_visitor_tilt_challenger_decisions` writes the
   overlay's own picks to the prospective challenger ledger, dual-tracked
   and at no rotation-registry window cost.
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from _overlay_test_kit import (
    write_active_model_and_card,
    write_challenger_registry,
    write_registry_root,
)

from nfl_ats.data import DataContractError
from nfl_ats.forecast_cold_visitor_tilt_overlay import (
    CHALLENGER_ID,
    ForecastColdVisitorFlip,
    ForecastColdVisitorResult,
    MosFetchError,
    apply_forecast_cold_visitor_tilt_overlay,
    fetch_one_game_temp,
    fetch_tuesday_noon_forecast_temps_fail_open,
    forecast_cold_visitor_flag_by_game,
    overlay_disclosure_note,
    record_forecast_cold_visitor_tilt_challenger_decisions,
    team_climate_temp_by_away_game,
)
from nfl_ats.prospective_scoring import CHALLENGER_DECISION_COLUMNS, load_challenger_decisions
from nfl_ats.snapshots import write_snapshot

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
#
# WARM's own outdoor home-game temps this season/prior seasons: 82F (2024
#   week 1), 74F (2024 week 5) -- climate_temp after both = 78.0.
# WARM then plays AWAY at FROST (2025 week 3, outdoor, forecast 30F): gap
#   78-30=48 >= 25 -> flagged. Model's own pick is AWAY (WARM) -> flips HOME.
# WARM plays AWAY at DOME (2025 week 4, roof=closed, forecast 20F): flag
#   would fire on temp gap alone, but the game itself is indoors -> no flip.
# WARM plays AWAY at FROST2 (2025 week 5, outdoor, NO forecast row at all):
#   missing forecast -> no signal -> no flip.
# WARM plays HOME again at OPP3 (2025 week 8, outdoor, 95F) -- a LATER game,
#   used only for the leak-safety check (must not move week 3's flag).
# ROOKIE has no prior home game at all before playing away at FROST3 (2024
#   week 1) -- climate_temp NaN -> no signal -> no flip.
# POSTWARM at FROST mirrors the week-3 flagged shape but game_type=POST.


def _schedule() -> pd.DataFrame:
    rows = [
        # game_id, season, game_type, week, gameday, home, away, roof, temp
        ("2024_01_WARM_OPP1", 2024, "REG", 1, "2024-09-08", "WARM", "OPP1", "outdoors", 82.0),
        ("2024_05_WARM_OPP2", 2024, "REG", 5, "2024-10-06", "WARM", "OPP2", "outdoors", 74.0),
        ("2025_03_FROST_WARM", 2025, "REG", 3, "2025-09-21", "FROST", "WARM", "outdoors", None),
        ("2025_04_DOME_WARM", 2025, "REG", 4, "2025-09-28", "DOME", "WARM", "closed", None),
        ("2025_05_FROST2_WARM", 2025, "REG", 5, "2025-10-05", "FROST2", "WARM", "outdoors", None),
        ("2025_08_WARM_OPP3", 2025, "REG", 8, "2025-10-26", "WARM", "OPP3", "outdoors", 95.0),
        (
            "2024_01_FROST3_ROOKIE",
            2024,
            "REG",
            1,
            "2024-09-08",
            "FROST3",
            "ROOKIE",
            "outdoors",
            None,
        ),
        ("2025_20_FROSTP_WARM", 2025, "POST", 20, "2026-01-11", "FROSTP", "WARM", "outdoors", None),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "game_id",
            "season",
            "game_type",
            "week",
            "gameday",
            "home_team",
            "away_team",
            "roof",
            "temp",
        ],
    )


def _forecasts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [
                "2025_03_FROST_WARM",
                "2025_04_DOME_WARM",
                # 2025_05_FROST2_WARM deliberately absent -- missing forecast row
                "2025_08_WARM_OPP3",
                "2024_01_FROST3_ROOKIE",
                "2025_20_FROSTP_WARM",
            ],
            "forecast_temp_f": [30.0, 20.0, 60.0, 40.0, 25.0],
            "fetch_status": ["ok", "ok", "ok", "ok", "ok"],
        }
    )


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [
                "2025_03_FROST_WARM",
                "2025_04_DOME_WARM",
                "2025_05_FROST2_WARM",
                "2024_01_FROST3_ROOKIE",
                "2025_20_FROSTP_WARM",
            ],
            "season": [2025, 2025, 2025, 2024, 2025],
            "week": [3, 4, 5, 1, 20],
            "game_type": ["REG", "REG", "REG", "REG", "POST"],
            "home_team": ["FROST", "DOME", "FROST2", "FROST3", "FROSTP"],
            "away_team": ["WARM", "WARM", "WARM", "ROOKIE", "WARM"],
            "kickoff": ["2025-09-21T17:00:00+00:00"] * 5,
            "spread_line": [-3.0, -2.0, -1.5, 1.0, -3.0],
            # G-clean: model picks AWAY (WARM, flagged, outdoor, gap 48) --
            #   should flip to HOME.
            # G-dome: model picks AWAY (WARM) but the game is indoors -- no flip.
            # G-nowx: model picks AWAY (WARM) but no forecast row -- no flip.
            # G-rookie: model picks AWAY (ROOKIE, no climate history) -- no flip.
            # G-post: same shape as G-clean but POST season -- no flip.
            "home_cover_probability": [0.35, 0.35, 0.35, 0.40, 0.35],
        }
    )


# ---------------------------------------------------------------------------
# 1. team_climate_temp_by_away_game: pregame-safe climatology
# ---------------------------------------------------------------------------


def test_climate_temp_averages_strictly_prior_outdoor_home_games() -> None:
    climate = team_climate_temp_by_away_game(_schedule()).set_index("game_id")
    assert climate.loc["2025_03_FROST_WARM", "climate_temp"] == pytest.approx((82.0 + 74.0) / 2.0)


def test_climate_temp_is_nan_before_any_qualifying_prior_game() -> None:
    climate = team_climate_temp_by_away_game(_schedule()).set_index("game_id")
    assert pd.isna(climate.loc["2024_01_FROST3_ROOKIE", "climate_temp"])


def test_climate_temp_is_leak_safe_against_a_later_home_game_mutation() -> None:
    """AGENTS.md: a leakage regression test for every new feature family.

    Week 3's climate_temp depends only on WARM's two 2024 home games; mutating
    a LATER (week 8, 2025) home game's temp must not move it.
    """

    baseline = team_climate_temp_by_away_game(_schedule()).set_index("game_id")

    mutated = _schedule()
    mutated.loc[mutated["game_id"].eq("2025_08_WARM_OPP3"), "temp"] = 999.0
    changed = team_climate_temp_by_away_game(mutated).set_index("game_id")

    assert changed.loc["2025_03_FROST_WARM", "climate_temp"] == pytest.approx(
        baseline.loc["2025_03_FROST_WARM", "climate_temp"]
    )


def test_climate_temp_is_leak_safe_across_a_future_season() -> None:
    """A future season's home-game temp for the same team must never change
    an earlier game's already-computed climate_temp."""

    schedule = _schedule()
    baseline = team_climate_temp_by_away_game(schedule)

    future = pd.DataFrame(
        [
            (
                "2026_01_WARM_OPP4",
                2026,
                "REG",
                1,
                "2026-09-07",
                "WARM",
                "OPP4",
                "outdoors",
                55.0,
            ),
        ],
        columns=schedule.columns,
    )
    changed = team_climate_temp_by_away_game(pd.concat([schedule, future], ignore_index=True))

    pd.testing.assert_frame_equal(
        changed.loc[changed["game_id"].isin(baseline["game_id"])].reset_index(drop=True),
        baseline.reset_index(drop=True),
        check_exact=True,
    )


def test_climate_temp_ignores_non_outdoor_home_games() -> None:
    """A dome home game contributes no temp to its own team's climatology."""

    schedule = _schedule()
    dome_home = pd.DataFrame(
        [("2024_02_WARM_OPPX", 2024, "REG", 2, "2024-09-15", "WARM", "OPPX", "dome", 68.0)],
        columns=schedule.columns,
    )
    with_dome = pd.concat([schedule, dome_home], ignore_index=True)
    climate = team_climate_temp_by_away_game(with_dome).set_index("game_id")
    # WARM's dome game (68F) must not move the week-3 climate away from the
    # two outdoor games' own average.
    assert climate.loc["2025_03_FROST_WARM", "climate_temp"] == pytest.approx((82.0 + 74.0) / 2.0)


def test_climate_temp_requires_its_schedule_columns() -> None:
    with pytest.raises(DataContractError, match="forecast cold-visitor"):
        team_climate_temp_by_away_game(pd.DataFrame({"game_id": ["G1"]}))


# ---------------------------------------------------------------------------
# 2. forecast_cold_visitor_flag_by_game
# ---------------------------------------------------------------------------


def test_flag_fires_on_a_large_cold_gap() -> None:
    flags = forecast_cold_visitor_flag_by_game(_schedule(), _forecasts()).set_index("game_id")
    assert bool(flags.loc["2025_03_FROST_WARM", "forecast_cold_visitor_flag"]) is True


def test_flag_does_not_fire_indoors_even_with_a_qualifying_gap() -> None:
    flags = forecast_cold_visitor_flag_by_game(_schedule(), _forecasts()).set_index("game_id")
    assert bool(flags.loc["2025_04_DOME_WARM", "forecast_cold_visitor_flag"]) is False


def test_flag_does_not_fire_without_a_forecast_row() -> None:
    flags = forecast_cold_visitor_flag_by_game(_schedule(), _forecasts()).set_index("game_id")
    assert bool(flags.loc["2025_05_FROST2_WARM", "forecast_cold_visitor_flag"]) is False


def test_flag_does_not_fire_without_climate_history() -> None:
    flags = forecast_cold_visitor_flag_by_game(_schedule(), _forecasts()).set_index("game_id")
    assert bool(flags.loc["2024_01_FROST3_ROOKIE", "forecast_cold_visitor_flag"]) is False


def test_flag_by_game_excludes_postseason_rows_entirely() -> None:
    flags = forecast_cold_visitor_flag_by_game(_schedule(), _forecasts())
    assert "2025_20_FROSTP_WARM" not in set(flags["game_id"])


def test_flag_requires_forecast_columns() -> None:
    with pytest.raises(DataContractError, match="forecasts is missing"):
        forecast_cold_visitor_flag_by_game(_schedule(), pd.DataFrame({"game_id": ["G1"]}))


# ---------------------------------------------------------------------------
# 3. apply_forecast_cold_visitor_tilt_overlay: the pick-level transform
# ---------------------------------------------------------------------------


def test_overlay_flips_away_to_home_on_the_clean_cold_visitor_case() -> None:
    result = apply_forecast_cold_visitor_tilt_overlay(_predictions(), _schedule(), _forecasts())

    flipped_ids = {flip.game_id for flip in result.flips}
    assert "2025_03_FROST_WARM" in flipped_ids
    flip = next(f for f in result.flips if f.game_id == "2025_03_FROST_WARM")
    assert flip.away_team == "WARM"
    assert flip.home_team == "FROST"
    assert flip.temp_gap_f == pytest.approx(48.0)

    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2025_03_FROST_WARM", "home_cover_probability"] == pytest.approx(0.65)


def test_overlay_does_not_flip_indoors() -> None:
    result = apply_forecast_cold_visitor_tilt_overlay(_predictions(), _schedule(), _forecasts())
    assert all(flip.game_id != "2025_04_DOME_WARM" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2025_04_DOME_WARM", "home_cover_probability"] == pytest.approx(0.35)


def test_overlay_does_not_flip_without_a_forecast() -> None:
    result = apply_forecast_cold_visitor_tilt_overlay(_predictions(), _schedule(), _forecasts())
    assert all(flip.game_id != "2025_05_FROST2_WARM" for flip in result.flips)


def test_overlay_does_not_flip_without_climate_history() -> None:
    result = apply_forecast_cold_visitor_tilt_overlay(_predictions(), _schedule(), _forecasts())
    assert all(flip.game_id != "2024_01_FROST3_ROOKIE" for flip in result.flips)


def test_overlay_leaves_postseason_games_untouched() -> None:
    result = apply_forecast_cold_visitor_tilt_overlay(_predictions(), _schedule(), _forecasts())
    assert all(flip.game_id != "2025_20_FROSTP_WARM" for flip in result.flips)
    overlaid = result.overlaid_predictions.set_index("game_id")
    assert overlaid.loc["2025_20_FROSTP_WARM", "home_cover_probability"] == pytest.approx(0.35)


def test_overlay_never_flips_a_home_pick() -> None:
    """Deliberately asymmetric: a HOME pick on a flagged game is untouched."""

    predictions = _predictions()
    predictions.loc[predictions["game_id"].eq("2025_03_FROST_WARM"), "home_cover_probability"] = (
        0.60
    )
    result = apply_forecast_cold_visitor_tilt_overlay(predictions, _schedule(), _forecasts())
    assert all(flip.game_id != "2025_03_FROST_WARM" for flip in result.flips)


def test_overlay_disabled_is_a_no_op() -> None:
    predictions = _predictions()
    result = apply_forecast_cold_visitor_tilt_overlay(
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
    result = apply_forecast_cold_visitor_tilt_overlay(predictions, _schedule(), _forecasts())
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
        apply_forecast_cold_visitor_tilt_overlay(
            pd.DataFrame({"game_id": ["G1"]}), _schedule(), _forecasts()
        )


# ---------------------------------------------------------------------------
# 4. overlay_disclosure_note
# ---------------------------------------------------------------------------


def test_disclosure_note_is_empty_when_nothing_flipped() -> None:
    only_dome = _predictions().loc[lambda frame: frame["game_id"].eq("2025_04_DOME_WARM")]
    result = apply_forecast_cold_visitor_tilt_overlay(only_dome, _schedule(), _forecasts())
    assert overlay_disclosure_note(result) == ""


def test_disclosure_note_states_the_flip_count() -> None:
    result = apply_forecast_cold_visitor_tilt_overlay(_predictions(), _schedule(), _forecasts())
    note = overlay_disclosure_note(result)
    assert "Tilt applied: 1 pick flipped" in note
    assert "WARM at FROST" in note
    assert "not applied to the published card" in note


def test_disclosure_note_formats_a_hand_built_flip() -> None:
    result = ForecastColdVisitorResult(
        overlaid_predictions=pd.DataFrame({"game_id": ["G1"]}),
        flips=(
            ForecastColdVisitorFlip(
                game_id="G1",
                matchup="AW1 at HM1",
                away_team="AW1",
                home_team="HM1",
                climate_temp=78.0,
                forecast_temp_f=30.0,
                temp_gap_f=48.0,
            ),
        ),
        enabled=True,
    )
    note = overlay_disclosure_note(result)
    assert "AWAY -> HOME" in note
    assert "gap 48F" in note


# ---------------------------------------------------------------------------
# 5. Live fetch layer -- FAIL-OPEN, no real network call anywhere below
# ---------------------------------------------------------------------------


def _stub_bulletin_rows(temp_f: float, ftime_utc: str = "2025-09-21T18:00:00+00:00") -> list[dict]:
    return [{"tmp": temp_f, "ftime_utc": ftime_utc, "runtime_utc": "2025-09-16T12:00:00+00:00"}]


def test_fetch_one_game_temp_returns_ok_on_a_successful_bulletin() -> None:
    def stub(station: str, runtime_utc, *, model: str) -> list[dict]:
        return _stub_bulletin_rows(30.0)

    kickoff = pd.Timestamp("2025-09-21T18:00:00+00:00")
    cutoff = pd.Timestamp("2025-09-16T16:00:00+00:00")
    result = fetch_one_game_temp("KXYZ", kickoff, cutoff, fetch_bulletin=stub)
    assert result == {"forecast_temp_f": 30.0, "fetch_status": "ok"}


def test_fetch_one_game_temp_walks_backward_past_empty_bulletins() -> None:
    calls: list[Any] = []

    def stub(station: str, runtime_utc, *, model: str) -> list[dict]:
        calls.append(runtime_utc)
        if len(calls) < 3:
            return []
        return _stub_bulletin_rows(25.0)

    kickoff = pd.Timestamp("2025-09-21T18:00:00+00:00")
    cutoff = pd.Timestamp("2025-09-16T16:00:00+00:00")
    result = fetch_one_game_temp("KXYZ", kickoff, cutoff, fetch_bulletin=stub, delay_seconds=0.0)
    assert result["fetch_status"] == "ok"
    assert result["forecast_temp_f"] == pytest.approx(25.0)
    assert len(calls) == 3


def test_fetch_one_game_temp_exhausts_lookback_without_a_bulletin() -> None:
    def stub(station: str, runtime_utc, *, model: str) -> list[dict]:
        return []

    kickoff = pd.Timestamp("2025-09-21T18:00:00+00:00")
    cutoff = pd.Timestamp("2025-09-16T16:00:00+00:00")
    result = fetch_one_game_temp(
        "KXYZ", kickoff, cutoff, fetch_bulletin=stub, max_lookback_steps=2, delay_seconds=0.0
    )
    assert result == {"forecast_temp_f": None, "fetch_status": "no_bulletin_within_lookback"}


def test_fetch_one_game_temp_folds_a_transport_error_into_a_status() -> None:
    def stub(station: str, runtime_utc, *, model: str) -> list[dict]:
        raise MosFetchError("simulated network down")

    kickoff = pd.Timestamp("2025-09-21T18:00:00+00:00")
    cutoff = pd.Timestamp("2025-09-16T16:00:00+00:00")
    result = fetch_one_game_temp(
        "KXYZ", kickoff, cutoff, fetch_bulletin=stub, max_lookback_steps=1, delay_seconds=0.0
    )
    assert result == {"forecast_temp_f": None, "fetch_status": "transport_error"}


def test_fetch_fail_open_returns_zero_flags_on_a_missing_station_map(tmp_path: Path) -> None:
    """THE no-network unit test: a total fetch failure (here, a station map
    that does not exist -- the same failure mode as a network being down
    long enough that the batch orchestration itself errors) must fold into
    zero flags with a logged warning, never an exception, and the resulting
    forecasts frame must make the overlay a complete no-op."""

    games = pd.DataFrame(
        {
            "game_id": ["2025_03_FROST_WARM"],
            "stadium": ["Frost Stadium"],
            "kickoff": ["2025-09-21T18:00:00+00:00"],
        }
    )
    missing_station_map = tmp_path / "does_not_exist.csv"

    with pytest.warns(RuntimeWarning, match="forecast fetch failed"):
        forecasts = fetch_tuesday_noon_forecast_temps_fail_open(games, missing_station_map)

    assert forecasts["fetch_status"].eq("fetch_failed").all()
    assert forecasts["forecast_temp_f"].isna().all()

    # Feeding this into the overlay end-to-end must be a genuine no-op: the
    # flagged case from the earlier tests must NOT flip when the forecast
    # data never arrived.
    result = apply_forecast_cold_visitor_tilt_overlay(_predictions(), _schedule(), forecasts)
    assert result.flip_count == 0


def test_fetch_fail_open_never_raises_even_on_an_unexpected_exception(tmp_path: Path) -> None:
    def exploding_bulletin(station: str, runtime_utc, *, model: str) -> list[dict]:
        raise RuntimeError("something totally unexpected")

    station_map = tmp_path / "stadium_station_map.csv"
    station_map.write_text(
        "stadium,icao_station,mappable\nFrost Stadium,KXYZ,True\n", encoding="utf-8"
    )
    games = pd.DataFrame(
        {
            "game_id": ["2025_03_FROST_WARM"],
            "stadium": ["Frost Stadium"],
            "kickoff": ["2025-09-21T18:00:00+00:00"],
        }
    )

    with pytest.warns(RuntimeWarning, match="forecast fetch failed"):
        forecasts = fetch_tuesday_noon_forecast_temps_fail_open(
            games, station_map, fetch_bulletin=exploding_bulletin
        )
    assert forecasts["fetch_status"].eq("fetch_failed").all()


def test_fetch_fail_open_works_end_to_end_with_a_stub_bulletin(tmp_path: Path) -> None:
    station_map = tmp_path / "stadium_station_map.csv"
    station_map.write_text(
        "stadium,icao_station,mappable\nFrost Stadium,KXYZ,True\n", encoding="utf-8"
    )
    games = pd.DataFrame(
        {
            "game_id": ["2025_03_FROST_WARM"],
            "stadium": ["Frost Stadium"],
            "kickoff": ["2025-09-21T18:00:00+00:00"],
        }
    )

    def stub(station: str, runtime_utc, *, model: str) -> list[dict]:
        assert station == "KXYZ"
        return _stub_bulletin_rows(30.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a working fetch must not warn at all
        forecasts = fetch_tuesday_noon_forecast_temps_fail_open(
            games, station_map, fetch_bulletin=stub, delay_seconds=0.0
        )
    assert forecasts.set_index("game_id").loc["2025_03_FROST_WARM", "forecast_temp_f"] == 30.0
    assert forecasts.set_index("game_id").loc["2025_03_FROST_WARM", "fetch_status"] == "ok"


def test_fetch_fail_open_raises_on_an_unmapped_stadium_and_is_caught(tmp_path: Path) -> None:
    station_map = tmp_path / "stadium_station_map.csv"
    station_map.write_text(
        "stadium,icao_station,mappable\nSome Other Stadium,KABC,True\n", encoding="utf-8"
    )
    games = pd.DataFrame(
        {
            "game_id": ["2025_03_FROST_WARM"],
            "stadium": ["Frost Stadium"],  # not in the station map at all
            "kickoff": ["2025-09-21T18:00:00+00:00"],
        }
    )
    with pytest.warns(RuntimeWarning, match="forecast fetch failed"):
        forecasts = fetch_tuesday_noon_forecast_temps_fail_open(games, station_map)
    assert forecasts["fetch_status"].eq("fetch_failed").all()


# ---------------------------------------------------------------------------
# 6. record_forecast_cold_visitor_tilt_challenger_decisions
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
        "2025-09-21T17:00:00+00:00",
        "2025-09-28T17:00:00+00:00",
        "2025-10-05T17:00:00+00:00",
        # ROOKIE's real gameday (2024-09-08) is in the past relative to this
        # fixture's recording instant (now=2025-09-16) -- kept a future
        # timestamp here instead so the recorder-level test can assert on
        # ALL five rows being recorded pre-kickoff, matching how a real
        # multi-week card's kickoffs would all still be in the future.
        "2025-10-12T17:00:00+00:00",
        "2026-01-11T17:00:00+00:00",
    ]
    return predictions


def _write_registry(artifacts: Path, *, status: str = "ACTIVE_PROSPECTIVE") -> None:
    write_challenger_registry(
        artifacts, challenger_id=CHALLENGER_ID, model_config=_MODEL_CONFIG, status=status
    )


_SEASON, _WEEK, _CREATED_AT_UTC = 2025, 3, "2025-09-16T15:00:00+00:00"


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
        "FROSTP": "Frost Playoff Field",
        "WARM": "Warm Field",
    }
    schedule["stadium"] = schedule["home_team"].map(stadium_by_team)
    return schedule


def _write_data_root(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    write_snapshot(
        _schedule_with_stadiums(),
        pd.DataFrame({"game_id": [], "team": []}),
        seasons=[2024, 2025],
        raw_root=data_root / "raw",
    )
    return data_root


_STADIUM_MAP_CSV = (
    "stadium,icao_station,mappable\n"
    "Frost Stadium,KFRS,True\n"
    "Dome Arena,KDOM,True\n"
    "Frost 2 Field,KFR2,True\n"
    "Frost 3 Field,KFR3,True\n"
    "Frost Playoff Field,KFRP,True\n"
    "Warm Field,KWRM,True\n"
)


def _write_registry_root(tmp_path: Path) -> Path:
    return write_registry_root(tmp_path, stadium_station_map_csv=_STADIUM_MAP_CSV)


def _no_network_stub(station: str, runtime_utc, *, model: str) -> list[dict]:
    return _stub_bulletin_rows(30.0, ftime_utc="2025-09-21T17:00:00+00:00")


@pytest.mark.full  # ENG-11: dominates --durations
def test_record_challenger_decisions_records_the_tilt_arm(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)
    registry_root = _write_registry_root(tmp_path)
    now = datetime(2025, 9, 16, 16, 0, tzinfo=UTC)

    result = record_forecast_cold_visitor_tilt_challenger_decisions(
        artifacts, data_root, registry_root, now=now, fetch_bulletin=_no_network_stub
    )

    assert result["recorded"] == 5
    ledger = load_challenger_decisions(artifacts).set_index("game_id")
    assert list(load_challenger_decisions(artifacts).columns) == list(CHALLENGER_DECISION_COLUMNS)
    assert (ledger["bet_side"] == "PASS").all()
    assert ledger["edge"].isna().all()

    # The flagged, clean-case game flips from AWAY to HOME.
    assert ledger.loc["2025_03_FROST_WARM", "pick_side"] == "HOME"
    # The dome game keeps the model's own AWAY pick (no flip indoors).
    assert ledger.loc["2025_04_DOME_WARM", "pick_side"] == "AWAY"

    # Re-running is a no-op: append-only, never rewrites.
    again = record_forecast_cold_visitor_tilt_challenger_decisions(
        artifacts, data_root, registry_root, now=now, fetch_bulletin=_no_network_stub
    )
    assert again["recorded"] == 0
    assert again["already_recorded"] == 5


def test_record_challenger_decisions_is_fail_open_on_a_missing_station_map(tmp_path: Path) -> None:
    """The publish-relevant fail-open guarantee, exercised through the full
    record path: a missing station map must not raise out of the recorder,
    it must simply record the active model's own un-tilted picks."""

    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)
    registry_root = tmp_path / "registry_without_station_map"
    now = datetime(2025, 9, 16, 16, 0, tzinfo=UTC)

    with pytest.warns(RuntimeWarning, match="forecast fetch failed"):
        result = record_forecast_cold_visitor_tilt_challenger_decisions(
            artifacts, data_root, registry_root, now=now
        )

    assert result["recorded"] == 5
    assert result["flip_count"] == 0
    ledger = load_challenger_decisions(artifacts).set_index("game_id")
    # Without any forecast data, even the geometrically-flaggable game keeps
    # the model's own raw AWAY pick.
    assert ledger.loc["2025_03_FROST_WARM", "pick_side"] == "AWAY"


@pytest.mark.full  # ENG-11: dominates --durations
def test_record_challenger_refuses_outside_recording_lock_window(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    _write_registry(artifacts)
    _write_active_model_and_card(artifacts)
    data_root = _write_data_root(tmp_path)
    registry_root = _write_registry_root(tmp_path)

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_forecast_cold_visitor_tilt_challenger_decisions(
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
        record_forecast_cold_visitor_tilt_challenger_decisions(
            artifacts,
            data_root,
            registry_root,
            now=datetime(2025, 9, 16, 16, 0, tzinfo=UTC),
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
        record_forecast_cold_visitor_tilt_challenger_decisions(
            artifacts,
            data_root,
            registry_root,
            now=datetime(2025, 9, 16, 16, 0, tzinfo=UTC),
            fetch_bulletin=_no_network_stub,
        )
