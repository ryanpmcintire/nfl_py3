from __future__ import annotations

from pathlib import Path

import pandas as pd

from nfl_ats.experiment_runner import _flag_drought_severe_grass


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
