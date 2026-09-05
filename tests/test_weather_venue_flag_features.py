"""Construction, sign-convention, venue-list and leakage contracts for the
two ROADMAP LEAD-36/LEAD-37 weather/venue flags, plus the on-production
confirmation wrapper's duck-typed reuse of
``scripts/on_production_opener_confirmation.py``.

Predeclared in ``docs/weather_venue_leads.md``. Every fixture is built in
memory: these tests must pass in a fresh clone with no local data snapshot
ever read.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import weather_venue_flags_on_production as wvop  # noqa: E402

from nfl_ats.data import DataContractError  # noqa: E402
from nfl_ats.margin import margin_feature_columns  # noqa: E402
from nfl_ats.weather_venue_flag_features import (  # noqa: E402
    OPEN_CORNER_STADIUMS,
    OPEN_CORNER_WIND_DOG_COLUMN,
    RAIN_ON_GRASS_DOG_COLUMN,
    attach_open_corner_wind_dog_features,
    attach_rain_on_grass_dog_features,
    derive_open_corner_wind_dog_features,
    derive_rain_on_grass_dog_features,
    open_corner_wind_population_diagnostic,
    rain_on_grass_population_diagnostic,
)


def _opener_lines(rows: list[tuple[str, float | None]]) -> pd.DataFrame:
    """(game_id, tue_open_home_spread) -- positive means HOME favored."""

    return pd.DataFrame(rows, columns=["game_id", "tue_open_home_spread"])


# ---------------------------------------------------------------------------
# LEAD-36: open-corner stadium wind
# ---------------------------------------------------------------------------


def _wind_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # BUF00 open-corner venue, outdoor, wind 20 -- qualifies. Home
            # (BUF) is the underdog (positive away-favorite spread convention:
            # tue_open_home_spread < 0 means home is the dog) -> +1.
            {
                "game_id": "g_home_dog",
                "home_team": "BUF",
                "away_team": "MIA",
                "stadium_id": "BUF00",
                "roof": "outdoors",
                "wind": 20.0,
            },
            # Same venue/wind, but the AWAY team is the underdog -> -1.
            {
                "game_id": "g_away_dog",
                "home_team": "BUF",
                "away_team": "NYJ",
                "stadium_id": "BUF00",
                "roof": "outdoors",
                "wind": 22.0,
            },
            # Qualifies on venue/wind/roof but opener spread is an exact
            # pick'em -> 0 (no side to back).
            {
                "game_id": "g_pickem",
                "home_team": "BUF",
                "away_team": "NE",
                "stadium_id": "BUF00",
                "roof": "outdoors",
                "wind": 18.0,
            },
            # Qualifies on venue/roof but wind is only 10 mph -- below
            # threshold -> 0 regardless of the opener spread.
            {
                "game_id": "g_low_wind",
                "home_team": "BUF",
                "away_team": "MIA",
                "stadium_id": "BUF00",
                "roof": "outdoors",
                "wind": 10.0,
            },
            # Open-corner venue, high wind, but roof is a DOME for this game
            # (should not happen in practice for BUF00, but must still gate
            # correctly if it did) -> 0.
            {
                "game_id": "g_dome",
                "home_team": "BUF",
                "away_team": "MIA",
                "stadium_id": "BUF00",
                "roof": "dome",
                "wind": 25.0,
            },
            # NOT a frozen open-corner venue (e.g. a dome team) -> 0 even
            # though wind happens to be recorded high (data artifact).
            {
                "game_id": "g_not_open_corner",
                "home_team": "DAL",
                "away_team": "NYG",
                "stadium_id": "DAL00",
                "roof": "outdoors",
                "wind": 30.0,
            },
            # DEN home game at a ONE-OFF away stadium code (international
            # relocation) -- must NOT qualify even though home_team == DEN
            # and wind/roof otherwise satisfy the gate.
            {
                "game_id": "g_den_one_off",
                "home_team": "DEN",
                "away_team": "KC",
                "stadium_id": "SFO01",
                "roof": "outdoors",
                "wind": 25.0,
            },
            # NYJ/NYG frozen ONLY at NYC01 (MetLife) -- a home game at the
            # old NYC00 code must not qualify.
            {
                "game_id": "g_old_meadowlands",
                "home_team": "NYJ",
                "away_team": "BUF",
                "stadium_id": "NYC00",
                "roof": "outdoors",
                "wind": 25.0,
            },
            # NYG at NYC01 (MetLife) qualifies like every other frozen venue.
            {
                "game_id": "g_metlife",
                "home_team": "NYG",
                "away_team": "PHI",
                "stadium_id": "NYC01",
                "roof": "outdoors",
                "wind": 16.0,
            },
            # Missing wind value entirely -> never guessed, treated as
            # not qualifying.
            {
                "game_id": "g_missing_wind",
                "home_team": "BUF",
                "away_team": "MIA",
                "stadium_id": "BUF00",
                "roof": "outdoors",
                "wind": None,
            },
        ]
    ).assign(season=2020)


def _wind_opener_lines() -> pd.DataFrame:
    return _opener_lines(
        [
            ("g_home_dog", -3.0),  # home (BUF) underdog
            ("g_away_dog", 3.0),  # away (NYJ) underdog
            ("g_pickem", 0.0),  # exact pick'em
            ("g_low_wind", -3.0),
            ("g_dome", -3.0),
            ("g_not_open_corner", 3.0),
            ("g_den_one_off", -3.0),
            ("g_old_meadowlands", -3.0),
            ("g_metlife", 3.0),  # away (PHI) underdog
            ("g_missing_wind", -3.0),
        ]
    )


def test_open_corner_stadiums_matches_the_frozen_task_anchor_list() -> None:
    assert set(OPEN_CORNER_STADIUMS.values()) == {
        "BUF",
        "CHI",
        "NE",
        "CLE",
        "GB",
        "PIT",
        "KC",
        "DEN",
        "NYJ/NYG",
        "PHI",
    }
    # NYJ/NYG intentionally share exactly one stadium_id (MetLife).
    assert list(OPEN_CORNER_STADIUMS).count("NYC01") == 1


def test_open_corner_wind_sign_home_underdog_is_positive() -> None:
    result = derive_open_corner_wind_dog_features(_wind_schedule(), _wind_opener_lines())
    result = result.set_index("game_id")
    assert result.loc["g_home_dog", OPEN_CORNER_WIND_DOG_COLUMN] == 1.0


def test_open_corner_wind_sign_away_underdog_is_negative() -> None:
    result = derive_open_corner_wind_dog_features(_wind_schedule(), _wind_opener_lines())
    result = result.set_index("game_id")
    assert result.loc["g_away_dog", OPEN_CORNER_WIND_DOG_COLUMN] == -1.0


@pytest.mark.parametrize(
    "game_id",
    [
        "g_pickem",
        "g_low_wind",
        "g_dome",
        "g_not_open_corner",
        "g_den_one_off",
        "g_old_meadowlands",
        "g_missing_wind",
    ],
)
def test_open_corner_wind_non_qualifying_games_are_zero(game_id: str) -> None:
    result = derive_open_corner_wind_dog_features(_wind_schedule(), _wind_opener_lines())
    result = result.set_index("game_id")
    assert result.loc[game_id, OPEN_CORNER_WIND_DOG_COLUMN] == 0.0


def test_open_corner_wind_metlife_qualifies_at_nyc01_only() -> None:
    result = derive_open_corner_wind_dog_features(_wind_schedule(), _wind_opener_lines())
    result = result.set_index("game_id")
    # g_metlife: away (PHI) is the underdog -> -1.
    assert result.loc["g_metlife", OPEN_CORNER_WIND_DOG_COLUMN] == -1.0


def test_open_corner_wind_leakage_ignores_unrelated_outcome_columns() -> None:
    """Mutating an unrelated outcome-shaped column must never change the
    flag -- neither the derivation nor its required-column set references
    any such column."""

    schedule = _wind_schedule()
    schedule["result"] = 3.0
    schedule["home_score"] = 20
    schedule["away_score"] = 17
    schedule["spread_line"] = -3.0
    baseline = derive_open_corner_wind_dog_features(schedule, _wind_opener_lines()).set_index(
        "game_id"
    )

    mutated = schedule.copy()
    mutated["result"] = -14.0
    mutated["home_score"] = 3
    mutated["away_score"] = 41
    mutated["spread_line"] = 9.0
    after = derive_open_corner_wind_dog_features(mutated, _wind_opener_lines()).set_index("game_id")
    pd.testing.assert_series_equal(
        baseline[OPEN_CORNER_WIND_DOG_COLUMN], after[OPEN_CORNER_WIND_DOG_COLUMN]
    )


def test_open_corner_wind_attach_is_purely_additive() -> None:
    schedule = _wind_schedule()
    features = pd.DataFrame({"game_id": schedule["game_id"], "some_existing_feature": 1.0})
    widened = attach_open_corner_wind_dog_features(
        features, schedule=schedule, opener_lines=_wind_opener_lines()
    )
    assert sorted(set(widened.columns) - set(features.columns)) == [OPEN_CORNER_WIND_DOG_COLUMN]
    pd.testing.assert_frame_equal(features, widened[features.columns], check_exact=True)
    assert list(widened.index) == list(features.index)


def test_open_corner_wind_attach_requires_the_join_key() -> None:
    schedule = _wind_schedule()
    features = pd.DataFrame({"not_game_id": schedule["game_id"]})
    with pytest.raises(DataContractError, match="game_id"):
        attach_open_corner_wind_dog_features(
            features, schedule=schedule, opener_lines=_wind_opener_lines()
        )


def test_open_corner_wind_attach_refuses_to_overwrite_an_existing_column() -> None:
    schedule = _wind_schedule()
    features = pd.DataFrame({"game_id": schedule["game_id"], OPEN_CORNER_WIND_DOG_COLUMN: 0.0})
    with pytest.raises(DataContractError, match=OPEN_CORNER_WIND_DOG_COLUMN):
        attach_open_corner_wind_dog_features(
            features, schedule=schedule, opener_lines=_wind_opener_lines()
        )


def test_open_corner_wind_derive_requires_every_schedule_column() -> None:
    schedule = _wind_schedule().drop(columns=["wind"])
    with pytest.raises(DataContractError, match="wind"):
        derive_open_corner_wind_dog_features(schedule, _wind_opener_lines())


def test_open_corner_wind_population_diagnostic_counts() -> None:
    diagnostic = open_corner_wind_population_diagnostic(_wind_schedule(), _wind_opener_lines())
    # Open-corner venue games: every row except g_not_open_corner (DAL00),
    # g_den_one_off (SFO01), and g_old_meadowlands (NYC00) = 10 - 3 = 7.
    assert diagnostic["n_open_corner_venue_games"] == 7
    assert diagnostic["n_open_corner_outdoor_games"] == 6  # excludes g_dome
    # Eligible (venue + outdoor + wind>=15): g_home_dog, g_away_dog,
    # g_pickem, g_metlife = 4 (g_low_wind is 10mph, g_missing_wind is None).
    assert diagnostic["n_eligible_high_wind_games"] == 4
    assert diagnostic["eligible_missing_opener_spread"] == 0


# ---------------------------------------------------------------------------
# LEAD-37: rain-on-grass fumble chaos
# ---------------------------------------------------------------------------


def _rain_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"game_id": "r_home_dog", "home_team": "GB", "away_team": "CHI", "surface": "grass"},
            {"game_id": "r_away_dog", "home_team": "GB", "away_team": "MIN", "surface": "grass"},
            # Trailing-space raw value, must still normalize to grass.
            {
                "game_id": "r_trailing_space",
                "home_team": "GB",
                "away_team": "DET",
                "surface": "grass ",
            },
            # Turf surface -- never qualifies regardless of precip.
            {"game_id": "r_turf", "home_team": "DAL", "away_team": "NYG", "surface": "fieldturf"},
            # Grass but precip probability below threshold.
            {"game_id": "r_low_precip", "home_team": "GB", "away_team": "CHI", "surface": "grass"},
            # Grass, high precip, but opener spread is an exact pick'em.
            {"game_id": "r_pickem", "home_team": "GB", "away_team": "CHI", "surface": "grass"},
            # Grass with a missing forecast row entirely -> never guessed.
            {
                "game_id": "r_missing_forecast",
                "home_team": "GB",
                "away_team": "CHI",
                "surface": "grass",
            },
        ]
    ).assign(season=2020)


def _rain_opener_lines() -> pd.DataFrame:
    return _opener_lines(
        [
            ("r_home_dog", -3.0),
            ("r_away_dog", 3.0),
            ("r_trailing_space", -3.0),
            ("r_turf", -3.0),
            ("r_low_precip", -3.0),
            ("r_pickem", 0.0),
            ("r_missing_forecast", -3.0),
        ]
    )


def _rain_forecast() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"game_id": "r_home_dog", "forecast_precip_prob_pct": 75.0},
            {"game_id": "r_away_dog", "forecast_precip_prob_pct": 80.0},
            {"game_id": "r_trailing_space", "forecast_precip_prob_pct": 60.0},
            {"game_id": "r_turf", "forecast_precip_prob_pct": 90.0},
            {"game_id": "r_low_precip", "forecast_precip_prob_pct": 20.0},
            {"game_id": "r_pickem", "forecast_precip_prob_pct": 65.0},
            # r_missing_forecast intentionally absent.
        ]
    )


def test_rain_on_grass_sign_home_underdog_is_positive() -> None:
    result = derive_rain_on_grass_dog_features(
        _rain_schedule(), _rain_opener_lines(), _rain_forecast()
    ).set_index("game_id")
    assert result.loc["r_home_dog", RAIN_ON_GRASS_DOG_COLUMN] == 1.0


def test_rain_on_grass_sign_away_underdog_is_negative() -> None:
    result = derive_rain_on_grass_dog_features(
        _rain_schedule(), _rain_opener_lines(), _rain_forecast()
    ).set_index("game_id")
    assert result.loc["r_away_dog", RAIN_ON_GRASS_DOG_COLUMN] == -1.0


def test_rain_on_grass_trailing_space_surface_normalizes_to_grass() -> None:
    result = derive_rain_on_grass_dog_features(
        _rain_schedule(), _rain_opener_lines(), _rain_forecast()
    ).set_index("game_id")
    assert result.loc["r_trailing_space", RAIN_ON_GRASS_DOG_COLUMN] == 1.0


@pytest.mark.parametrize("game_id", ["r_turf", "r_low_precip", "r_pickem", "r_missing_forecast"])
def test_rain_on_grass_non_qualifying_games_are_zero(game_id: str) -> None:
    result = derive_rain_on_grass_dog_features(
        _rain_schedule(), _rain_opener_lines(), _rain_forecast()
    ).set_index("game_id")
    assert result.loc[game_id, RAIN_ON_GRASS_DOG_COLUMN] == 0.0


def test_rain_on_grass_leakage_ignores_unrelated_outcome_columns() -> None:
    schedule = _rain_schedule()
    schedule["result"] = 3.0
    schedule["home_score"] = 20
    schedule["away_score"] = 17
    schedule["spread_line"] = -3.0
    baseline = derive_rain_on_grass_dog_features(
        schedule, _rain_opener_lines(), _rain_forecast()
    ).set_index("game_id")

    mutated = schedule.copy()
    mutated["result"] = -14.0
    mutated["home_score"] = 3
    mutated["away_score"] = 41
    mutated["spread_line"] = 9.0
    after = derive_rain_on_grass_dog_features(
        mutated, _rain_opener_lines(), _rain_forecast()
    ).set_index("game_id")
    pd.testing.assert_series_equal(
        baseline[RAIN_ON_GRASS_DOG_COLUMN], after[RAIN_ON_GRASS_DOG_COLUMN]
    )


def test_rain_on_grass_attach_is_purely_additive() -> None:
    schedule = _rain_schedule()
    features = pd.DataFrame({"game_id": schedule["game_id"], "some_existing_feature": 1.0})
    widened = attach_rain_on_grass_dog_features(
        features,
        schedule=schedule,
        opener_lines=_rain_opener_lines(),
        forecast=_rain_forecast(),
    )
    assert sorted(set(widened.columns) - set(features.columns)) == [RAIN_ON_GRASS_DOG_COLUMN]
    pd.testing.assert_frame_equal(features, widened[features.columns], check_exact=True)
    assert list(widened.index) == list(features.index)


def test_rain_on_grass_attach_requires_the_join_key() -> None:
    schedule = _rain_schedule()
    features = pd.DataFrame({"not_game_id": schedule["game_id"]})
    with pytest.raises(DataContractError, match="game_id"):
        attach_rain_on_grass_dog_features(
            features,
            schedule=schedule,
            opener_lines=_rain_opener_lines(),
            forecast=_rain_forecast(),
        )


def test_rain_on_grass_attach_refuses_to_overwrite_an_existing_column() -> None:
    schedule = _rain_schedule()
    features = pd.DataFrame({"game_id": schedule["game_id"], RAIN_ON_GRASS_DOG_COLUMN: 0.0})
    with pytest.raises(DataContractError, match=RAIN_ON_GRASS_DOG_COLUMN):
        attach_rain_on_grass_dog_features(
            features,
            schedule=schedule,
            opener_lines=_rain_opener_lines(),
            forecast=_rain_forecast(),
        )


def test_rain_on_grass_derive_requires_every_schedule_column() -> None:
    schedule = _rain_schedule().drop(columns=["surface"])
    with pytest.raises(DataContractError, match="surface"):
        derive_rain_on_grass_dog_features(schedule, _rain_opener_lines(), _rain_forecast())


def test_rain_on_grass_derive_requires_forecast_columns() -> None:
    forecast = _rain_forecast().drop(columns=["forecast_precip_prob_pct"])
    with pytest.raises(DataContractError, match="forecast_precip_prob_pct"):
        derive_rain_on_grass_dog_features(_rain_schedule(), _rain_opener_lines(), forecast)


def test_rain_on_grass_population_diagnostic_counts() -> None:
    diagnostic = rain_on_grass_population_diagnostic(
        _rain_schedule(), _rain_opener_lines(), _rain_forecast()
    )
    assert diagnostic["n_grass_games"] == 6  # excludes r_turf
    assert diagnostic["n_grass_games_with_forecast"] == 5  # excludes r_missing_forecast
    # Eligible (grass + precip>=60): r_home_dog, r_away_dog,
    # r_trailing_space, r_pickem = 4.
    assert diagnostic["n_eligible_high_precip_games"] == 4
    assert diagnostic["eligible_missing_opener_spread"] == 0


# ---------------------------------------------------------------------------
# On-production wrapper contracts (both candidates)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(wvop.CANDIDATES))
def test_registered_profile_is_production_plus_the_declared_one_column(key: str) -> None:
    candidate = wvop.CANDIDATES[key]
    baseline = set(margin_feature_columns("market_residual", wvop.BASELINE_PROFILE))
    treatment = set(margin_feature_columns("market_residual", candidate.profile))
    assert treatment - baseline == {candidate.column}
    assert baseline - treatment == set()


@pytest.mark.parametrize("key", sorted(wvop.CANDIDATES))
def test_candidate_duck_types_with_the_template_profile_identity(key: str) -> None:
    """``on_production_opener_confirmation.profile_identity`` is reused
    unmodified: our ``WeatherVenueCandidate`` need only carry the same
    ``profile``/``column`` attribute names."""

    candidate = wvop.CANDIDATES[key]
    columns = margin_feature_columns("market_residual", candidate.profile)
    frame = pd.DataFrame({column: [0.0] for column in columns})
    observed = wvop.confirmation.profile_identity(candidate, frame)
    assert observed["only_added_column"] == candidate.column
