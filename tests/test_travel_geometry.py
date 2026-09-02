"""ENV-03 deterministic travel-geometry and leakage contracts."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from nfl_ats.constants import FEATURE_FAMILIES, FEATURE_SETS, TRAVEL_GEOMETRY_FEATURE_COLUMNS
from nfl_ats.data import DataContractError
from nfl_ats.travel_geometry import (
    add_travel_geometry_features,
    build_travel_geometry_features,
    load_stadium_coordinate_registry,
    validate_stadium_coordinate_registry,
)


def _registry():
    return validate_stadium_coordinate_registry(
        {
            "New York": {
                "lat": 40.8135,
                "lon": -74.0745,
                "tz": "America/New_York",
                "city": "East Rutherford, NJ",
            },
            "Los Angeles": {
                "lat": 33.9535,
                "lon": -118.3392,
                "tz": "America/Los_Angeles",
                "city": "Inglewood, CA",
            },
            "London": {
                "lat": 51.6043,
                "lon": -0.0664,
                "tz": "Europe/London",
                "city": "London, UK",
            },
        },
        source="unit-test-registry",
    )


def _schedules() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("g1", 2021, "2021-09-05", "X", "Y", "New York", "Home"),
            ("g2", 2021, "2021-09-12", "Y", "X", "Los Angeles", "Home"),
            ("g3", 2021, "2021-09-19", "X", "Y", "New York", "Home"),
            ("g4", 2021, "2021-10-03", "X", "Y", "London", "Neutral"),
        ],
        columns=[
            "game_id",
            "season",
            "gameday",
            "home_team",
            "away_team",
            "stadium",
            "location",
        ],
    ).assign(
        result=[7.0, -3.0, 10.0, math.nan],
        spread_line=[3.0, -1.0, 2.0, math.nan],
        temp=[72.0, 80.0, 60.0, math.nan],
    )


def _by_game(features: pd.DataFrame, game_id: str) -> pd.Series:
    return features.loc[features["game_id"].eq(game_id)].iloc[0]


def test_family_is_registered_but_not_admitted_to_any_model_profile() -> None:
    assert FEATURE_FAMILIES["travel_geometry"] == TRAVEL_GEOMETRY_FEATURE_COLUMNS
    for profile, columns in FEATURE_SETS.items():
        assert set(columns).isdisjoint(TRAVEL_GEOMETRY_FEATURE_COLUMNS), profile


def test_checked_in_stadium_registry_passes_the_runtime_contract() -> None:
    registry = load_stadium_coordinate_registry()
    assert len(registry.venues) >= 82
    assert len(registry.sha256) == 64
    assert registry.source.endswith(
        "registry\\stadium_coordinates.json"
    ) or registry.source.endswith("registry/stadium_coordinates.json")


def test_distance_timezone_direction_and_international_neutral_flags() -> None:
    features = build_travel_geometry_features(_schedules(), _registry())

    westbound = _by_game(features, "g2")
    assert westbound["travel_home_distance_mi"] == pytest.approx(0.0, abs=1e-9)
    assert westbound["travel_away_distance_mi"] == pytest.approx(2450.0, rel=0.02)
    assert westbound["travel_away_tz_change_hours"] == pytest.approx(-3.0)
    assert westbound["travel_away_body_clock_direction"] == -1.0
    assert westbound["travel_international_game"] == 0.0
    assert westbound["travel_neutral_site"] == 0.0

    eastbound = _by_game(features, "g3")
    assert eastbound["travel_away_tz_change_hours"] == pytest.approx(3.0)
    assert eastbound["travel_away_body_clock_direction"] == 1.0

    london = _by_game(features, "g4")
    assert london["travel_international_game"] == 1.0
    assert london["travel_neutral_site"] == 1.0
    assert london["travel_home_tz_change_hours"] == pytest.approx(5.0)
    assert london["travel_away_tz_change_hours"] == pytest.approx(8.0)


def test_origin_and_prior_trip_remain_missing_until_chronologically_available() -> None:
    features = build_travel_geometry_features(_schedules(), _registry())

    opener = _by_game(features, "g1")
    assert opener["travel_home_distance_mi"] == pytest.approx(0.0, abs=1e-9)
    assert pd.isna(opener["travel_away_distance_mi"])
    assert pd.isna(opener["travel_home_prior_game_distance_mi"])
    assert pd.isna(opener["travel_away_prior_game_distance_mi"])

    second = _by_game(features, "g2")
    assert pd.isna(second["travel_home_prior_game_distance_mi"])
    assert second["travel_away_prior_game_distance_mi"] == pytest.approx(0.0, abs=1e-9)

    third = _by_game(features, "g3")
    assert third["travel_home_prior_game_distance_mi"] == pytest.approx(2450.0, rel=0.02)
    assert third["travel_away_prior_game_distance_mi"] == pytest.approx(0.0, abs=1e-9)


def test_postgame_and_post_cutoff_mutations_cannot_change_decision_row() -> None:
    """Required leakage regression for the ENV-03 feature family."""

    schedules = _schedules()
    baseline = _by_game(build_travel_geometry_features(schedules, _registry()), "g3")

    changed = schedules.copy()
    changed.loc[changed["game_id"].eq("g3"), ["result", "spread_line", "temp"]] = [
        -1000.0,
        1000.0,
        -1000.0,
    ]
    future = changed["game_id"].eq("g4")
    changed.loc[future, ["result", "spread_line", "temp"]] = [1000.0, -1000.0, 1000.0]
    changed.loc[future, "stadium"] = "Los Angeles"
    changed.loc[future, "location"] = "Home"
    mutated = _by_game(build_travel_geometry_features(changed, _registry()), "g3")

    pd.testing.assert_series_equal(
        mutated[[*TRAVEL_GEOMETRY_FEATURE_COLUMNS]],
        baseline[[*TRAVEL_GEOMETRY_FEATURE_COLUMNS]],
        check_names=False,
        check_exact=True,
    )


def test_input_order_cannot_change_features() -> None:
    schedules = _schedules()
    baseline = build_travel_geometry_features(schedules, _registry())
    shuffled = build_travel_geometry_features(
        schedules.sample(frac=1.0, random_state=7).reset_index(drop=True), _registry()
    )
    pd.testing.assert_frame_equal(shuffled, baseline)


def test_unknown_game_venue_fails_closed_or_is_explicitly_missing() -> None:
    schedules = _schedules()
    schedules.loc[schedules["game_id"].eq("g4"), "stadium"] = "Unknown"
    with pytest.raises(DataContractError, match=r"unresolved stadium names:.*Unknown"):
        build_travel_geometry_features(schedules, _registry())

    permissive = build_travel_geometry_features(schedules, _registry(), strict_venues=False)
    row = _by_game(permissive, "g4")
    assert pd.isna(row["travel_home_distance_mi"])
    assert pd.isna(row["travel_international_game"])
    assert row["travel_neutral_site"] == 1.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("lat", 91.0, "lat must be between"),
        ("lon", float("inf"), "non-finite lon"),
        ("tz", "Mars/Olympus", "unknown IANA timezone"),
        ("city", "", "city must be a non-empty string"),
    ],
)
def test_registry_validation_rejects_invalid_metadata(
    field: str, value: object, message: str
) -> None:
    payload: dict[str, object] = {
        "lat": 40.0,
        "lon": -74.0,
        "tz": "America/New_York",
        "city": "New York, NY",
    }
    payload[field] = value
    with pytest.raises(DataContractError, match=message):
        validate_stadium_coordinate_registry({"Broken": payload})


def test_attach_preserves_rows_and_records_registry_provenance() -> None:
    games = pd.DataFrame({"game_id": ["g3", "g1"], "existing": [3, 1]})
    result = add_travel_geometry_features(games, _schedules(), _registry())

    assert result["game_id"].tolist() == ["g3", "g1"]
    assert result["existing"].tolist() == [3, 1]
    assert result.attrs["travel_geometry_provenance"]["registry_source"] == ("unit-test-registry")
    assert result.attrs["travel_geometry_provenance"]["outcome_columns_read"] == []
    assert len(result.attrs["travel_geometry_provenance"]["registry_sha256"]) == 64


def test_schedule_contract_rejects_ambiguous_or_invalid_rows() -> None:
    duplicate_team_date = pd.concat([_schedules(), _schedules().iloc[[0]]], ignore_index=True)
    duplicate_team_date.loc[len(duplicate_team_date) - 1, "game_id"] = "other"
    with pytest.raises(DataContractError, match="multiple games for one team"):
        build_travel_geometry_features(duplicate_team_date, _registry())

    bad_location = _schedules()
    bad_location.loc[0, "location"] = "Somewhere"
    with pytest.raises(DataContractError, match="invalid location values"):
        build_travel_geometry_features(bad_location, _registry())
