from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from nfl_ats.experiment_runner import _flag_drought_severe_grass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_environmental_exposure_join import (
    asof_merge_drought,
    drought_release_timestamps,
)


def _drought_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "county_fips": ["42101", "42101"],
            "valid_start": pd.to_datetime(["2020-08-25", "2020-09-01"]),
            "valid_end": pd.to_datetime(["2020-08-31", "2020-09-07"]),
            "none": [80.0, 20.0],
            "d0": [20.0, 80.0],
            "d1": [10.0, 70.0],
            "d2": [0.0, 60.0],
            "d3": [0.0, 10.0],
            "d4": [0.0, 0.0],
        }
    )


def test_usdm_release_timestamp_is_thursday_0830_eastern_and_dst_aware() -> None:
    releases = drought_release_timestamps(pd.Series(pd.to_datetime(["2020-09-01", "2020-12-01"])))

    assert releases.iloc[0] == pd.Timestamp("2020-09-03T12:30:00Z")
    assert releases.iloc[1] == pd.Timestamp("2020-12-03T13:30:00Z")


def test_drought_join_refuses_current_map_until_official_release_timestamp() -> None:
    games = pd.DataFrame(
        {
            "game_id": ["before", "at_release"],
            "county_fips": ["42101", "42101"],
            "tuesday_date": pd.to_datetime(["2020-09-01", "2020-09-01"]),
            "decision_at_utc": pd.to_datetime(
                ["2020-09-03T12:29:59Z", "2020-09-03T12:30:00Z"], utc=True
            ),
        }
    )

    joined = asof_merge_drought(games, _drought_rows()).set_index("game_id")

    assert joined.loc["before", "drought_valid_start"] == pd.Timestamp("2020-08-25")
    assert joined.loc["before", "drought_d2"] == 0.0
    assert joined.loc["at_release", "drought_valid_start"] == pd.Timestamp("2020-09-01")
    assert joined.loc["at_release", "drought_d2"] == 60.0


def test_future_drought_values_cannot_change_an_earlier_checkpoint() -> None:
    games = pd.DataFrame(
        {
            "game_id": ["g1"],
            "county_fips": ["42101"],
            "tuesday_date": pd.to_datetime(["2020-09-01"]),
            "decision_at_utc": pd.to_datetime(["2020-09-01T16:00:00Z"], utc=True),
        }
    )
    drought = _drought_rows()
    baseline = asof_merge_drought(games, drought).set_index("game_id")
    mutated = drought.copy()
    mutated.loc[mutated["valid_start"] >= pd.Timestamp("2020-09-01"), "d2"] = 99.0
    changed = asof_merge_drought(games, mutated).set_index("game_id")

    assert baseline.loc["g1", "drought_valid_start"] == pd.Timestamp("2020-08-25")
    assert changed.loc["g1", "drought_valid_start"] == pd.Timestamp("2020-08-25")
    assert changed.loc["g1", "drought_d2"] == baseline.loc["g1", "drought_d2"]


def test_drought_builder_is_low_dimensional_and_uses_only_fresh_released_rows(
    tmp_path: Path,
) -> None:
    join_path = tmp_path / "data/processed/environmental_exposures/game_join.parquet"
    join_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "game_id": ["flag", "turf", "stale", "future"],
            "surface": ["grass", "fieldturf", "grass", "grass"],
            "is_outdoor_exposed": [True, True, True, True],
            "drought_d2": [60.0, 60.0, 60.0, 60.0],
            "drought_is_stale_carryforward": [False, False, True, False],
            "drought_available_at_utc": pd.to_datetime(
                [
                    "2020-09-03T12:30:00Z",
                    "2020-09-03T12:30:00Z",
                    "2020-09-03T12:30:00Z",
                    "2020-09-10T12:30:00Z",
                ],
                utc=True,
            ),
            "decision_at_utc": pd.to_datetime(["2020-09-08T16:00:00Z"] * 4, utc=True),
        }
    ).to_parquet(join_path, index=False)
    features = pd.DataFrame(
        {
            "game_id": ["flag", "turf", "stale", "future"],
            "season": [2020] * 4,
            "week": [1] * 4,
            "game_type": ["REG"] * 4,
            "home_cover": [1.0, 0.0, 1.0, 0.0],
            "surface": ["grass", "fieldturf", "grass", "grass"],
        }
    )

    construct = _flag_drought_severe_grass(features, (2020, 2020), {}, tmp_path)

    assert construct.table["game_id"].tolist() == ["flag", "turf"]
    assert construct.flag.tolist() == [True, False]
    assert construct.sign == 1
    assert construct.eligible is None


def test_stadium_county_reference_has_provenance_for_every_domestic_row() -> None:
    path = Path(__file__).resolve().parents[1] / "registry/reference/stadium_county_fips.csv"
    stadiums = pd.read_csv(path, dtype={"county_fips": str, "state_fips": str})
    domestic = stadiums.loc[stadiums["in_scope"]].copy()

    assert domestic["county_fips"].str.fullmatch(r"\d{5}").all()
    assert domestic["county_name"].notna().all()
    assert domestic["state_code"].notna().all()
    assert domestic["fcc_status"].eq("OK").all()
