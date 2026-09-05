"""Construction, sign-convention and leakage contracts for the four Wave 2
LEAD-39/41/42/35 venue/market-context flags.

Predeclared in ``docs/schedule_flag_battery.md`` ("Wave 2" section). Every
fixture is built in memory: these tests must pass in a fresh clone with no
local data snapshots (no ``schedules.parquet`` snapshot and no
``data/market/raw`` market store is ever read).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import schedule_flag_on_production as sfop  # noqa: E402

from nfl_ats.data import DataContractError  # noqa: E402
from nfl_ats.margin import margin_feature_columns  # noqa: E402
from nfl_ats.schedule_flag_features import (  # noqa: E402
    DOME_SHOOTOUT_COLUMN,
    LOW_TOTAL_DIV_DOG_COLUMN,
    NEW_STADIUM_COLUMN,
    NEW_STADIUM_HONEYMOON_SEASONS,
    SEPT_HEAT_COLUMN,
    attach_dome_shootout_favorite_features,
    attach_low_total_div_home_dog_features,
    attach_new_stadium_home_features,
    attach_sept_heat_home_features,
    derive_dome_shootout_favorite_features,
    derive_low_total_div_home_dog_features,
    derive_new_stadium_home_features,
    derive_sept_heat_home_features,
)


def _schedule(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LEAD-39: new-stadium honeymoon
# ---------------------------------------------------------------------------


def _new_stadium_schedule() -> pd.DataFrame:
    return _schedule(
        [
            {"game_id": "n1", "season": 2010, "stadium_id": "NYC01"},  # honeymoon yr 1
            {"game_id": "n2", "season": 2011, "stadium_id": "NYC01"},  # honeymoon yr 2
            {"game_id": "n3", "season": 2012, "stadium_id": "NYC01"},  # 3rd season, no longer new
            {"game_id": "n4", "season": 2009, "stadium_id": "NYC01"},  # before first use
            {"game_id": "n5", "season": 2020, "stadium_id": "LAX01"},  # different venue, honeymoon
            {"game_id": "n6", "season": 2020, "stadium_id": "PHI00"},  # not a frozen venue at all
        ]
    )


def test_new_stadium_flags_only_first_two_seasons_of_a_frozen_venue() -> None:
    derived = derive_new_stadium_home_features(_new_stadium_schedule()).set_index("game_id")
    assert derived.loc["n1", NEW_STADIUM_COLUMN] == 1.0
    assert derived.loc["n2", NEW_STADIUM_COLUMN] == 1.0
    assert derived.loc["n3", NEW_STADIUM_COLUMN] == 0.0
    assert derived.loc["n4", NEW_STADIUM_COLUMN] == 0.0
    assert derived.loc["n5", NEW_STADIUM_COLUMN] == 1.0
    assert derived.loc["n6", NEW_STADIUM_COLUMN] == 0.0


def test_new_stadium_frozen_list_has_exactly_six_venues() -> None:
    assert set(NEW_STADIUM_HONEYMOON_SEASONS) == {
        "NYC01",
        "SFO01",
        "MIN01",
        "ATL97",
        "LAX01",
        "VEG00",
    }
    assert NEW_STADIUM_HONEYMOON_SEASONS["NYC01"] == (2010, 2011)
    assert NEW_STADIUM_HONEYMOON_SEASONS["VEG00"] == (2020, 2021)


def test_new_stadium_never_reads_an_outcome_column() -> None:
    schedule = _new_stadium_schedule()
    schedule["result"] = 0.0
    schedule["home_score"] = 10.0
    before = derive_new_stadium_home_features(schedule).set_index("game_id")
    mutated = schedule.copy()
    mutated["result"] = 999.0
    mutated["home_score"] = -999.0
    after = derive_new_stadium_home_features(mutated).set_index("game_id")
    pd.testing.assert_series_equal(before[NEW_STADIUM_COLUMN], after[NEW_STADIUM_COLUMN])


def test_new_stadium_requires_schedule_columns() -> None:
    schedule = _new_stadium_schedule().drop(columns=["stadium_id"])
    with pytest.raises(DataContractError, match="stadium_id"):
        derive_new_stadium_home_features(schedule)


def test_new_stadium_attach_is_purely_additive() -> None:
    schedule = _new_stadium_schedule()
    features = pd.DataFrame({"game_id": schedule["game_id"], "some_existing_feature": 1.0})
    widened = attach_new_stadium_home_features(features, schedule=schedule)
    assert sorted(set(widened.columns) - set(features.columns)) == [NEW_STADIUM_COLUMN]
    pd.testing.assert_frame_equal(features, widened[features.columns], check_exact=True)


# ---------------------------------------------------------------------------
# LEAD-41: dome-shootout favorite archetype
# ---------------------------------------------------------------------------


def _dome_schedule() -> pd.DataFrame:
    return _schedule(
        [
            {"game_id": "d1", "roof": "dome"},  # home favorite archetype
            {"game_id": "d2", "roof": "closed"},  # away favorite archetype
            {"game_id": "d3", "roof": "outdoors"},  # otherwise-qualifying but not dome/closed
            {"game_id": "d4", "roof": "dome"},  # total too low
            {"game_id": "d5", "roof": "dome"},  # spread too wide
            {"game_id": "d6", "roof": "dome"},  # exact pick'em, no favorite
            {"game_id": "d7", "roof": "dome"},  # missing opener total
            {"game_id": "d8", "roof": "dome"},  # missing opener spread
        ]
    )


def _dome_opener_lines() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["d1", "d2", "d3", "d4", "d5", "d6", "d8"],
            "tue_open_home_spread": [2.5, -1.0, 1.0, 1.0, 6.0, 0.0, float("nan")],
            "tue_open_total_line": [51.0, 49.0, 50.0, 44.0, 52.0, 49.5, 55.0],
        }
    )
    # d7 deliberately absent entirely -> both fields NaN after the left merge


def test_dome_shootout_sign_convention_and_thresholds() -> None:
    derived = derive_dome_shootout_favorite_features(
        _dome_schedule(), _dome_opener_lines()
    ).set_index("game_id")
    assert derived.loc["d1", DOME_SHOOTOUT_COLUMN] == 1.0  # home favorite, archetype
    assert derived.loc["d2", DOME_SHOOTOUT_COLUMN] == -1.0  # away favorite, archetype
    assert derived.loc["d3", DOME_SHOOTOUT_COLUMN] == 0.0  # roof not dome/closed
    assert derived.loc["d4", DOME_SHOOTOUT_COLUMN] == 0.0  # total < 49
    assert derived.loc["d5", DOME_SHOOTOUT_COLUMN] == 0.0  # |spread| > 3
    assert derived.loc["d6", DOME_SHOOTOUT_COLUMN] == 0.0  # pick'em, no favorite
    assert derived.loc["d7", DOME_SHOOTOUT_COLUMN] == 0.0  # missing opener total/spread entirely
    assert derived.loc["d8", DOME_SHOOTOUT_COLUMN] == 0.0  # missing opener spread only


def test_dome_shootout_missing_total_never_silently_satisfies_threshold() -> None:
    """A missing opener total must NEVER be treated as satisfying >= 49."""

    schedule = _schedule([{"game_id": "d9", "roof": "dome"}])
    opener_lines = pd.DataFrame(
        {
            "game_id": ["d9"],
            "tue_open_home_spread": [1.0],
            "tue_open_total_line": [float("nan")],
        }
    )
    derived = derive_dome_shootout_favorite_features(schedule, opener_lines).set_index("game_id")
    assert derived.loc["d9", DOME_SHOOTOUT_COLUMN] == 0.0


def test_dome_shootout_never_reads_an_outcome_column() -> None:
    schedule = _dome_schedule()
    schedule["result"] = 0.0
    opener_lines = _dome_opener_lines()
    before = derive_dome_shootout_favorite_features(schedule, opener_lines).set_index("game_id")
    mutated = schedule.copy()
    mutated["result"] = 999.0
    after = derive_dome_shootout_favorite_features(mutated, opener_lines).set_index("game_id")
    pd.testing.assert_series_equal(before[DOME_SHOOTOUT_COLUMN], after[DOME_SHOOTOUT_COLUMN])


def test_dome_shootout_requires_schedule_columns() -> None:
    schedule = _dome_schedule().drop(columns=["roof"])
    with pytest.raises(DataContractError, match="roof"):
        derive_dome_shootout_favorite_features(schedule, _dome_opener_lines())


def test_dome_shootout_requires_opener_lines_join_key() -> None:
    schedule = _dome_schedule()
    bad_lines = _dome_opener_lines().rename(columns={"game_id": "not_game_id"})
    with pytest.raises(DataContractError, match="game_id"):
        derive_dome_shootout_favorite_features(schedule, bad_lines)


def test_dome_shootout_attach_accepts_supplied_opener_lines_without_touching_the_store() -> None:
    """Passing ``opener_lines`` directly must never call ``default_opener_lines``
    (i.e. never touch ``data/market/raw``), which is the contract every test
    in this module relies on to run without local data."""

    schedule = _dome_schedule()
    features = pd.DataFrame({"game_id": schedule["game_id"]})
    widened = attach_dome_shootout_favorite_features(
        features, schedule=schedule, opener_lines=_dome_opener_lines()
    )
    assert sorted(set(widened.columns) - set(features.columns)) == [DOME_SHOOTOUT_COLUMN]
    assert widened.set_index("game_id").loc["d1", DOME_SHOOTOUT_COLUMN] == 1.0


# ---------------------------------------------------------------------------
# LEAD-42: low-total divisional home dog
# ---------------------------------------------------------------------------


def _low_total_schedule() -> pd.DataFrame:
    return _schedule(
        [
            {"game_id": "l1", "div_game": 1},  # qualifies: divisional, low total, home dog
            {"game_id": "l2", "div_game": 0},  # not divisional
            {"game_id": "l3", "div_game": 1},  # total too high
            {"game_id": "l4", "div_game": 1},  # home is favorite, not dog
            {"game_id": "l5", "div_game": 1},  # pick'em, home not a dog
            {"game_id": "l6", "div_game": 1},  # missing opener total
        ]
    )


def _low_total_opener_lines() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["l1", "l2", "l3", "l4", "l5"],
            "tue_open_home_spread": [-2.5, -2.5, -2.5, 3.0, 0.0],
            "tue_open_total_line": [40.0, 40.0, 45.0, 40.0, 40.0],
        }
    )
    # l6 deliberately absent -> NaN after the left merge


def test_low_total_div_dog_sign_convention_and_thresholds() -> None:
    derived = derive_low_total_div_home_dog_features(
        _low_total_schedule(), _low_total_opener_lines()
    ).set_index("game_id")
    assert derived.loc["l1", LOW_TOTAL_DIV_DOG_COLUMN] == 1.0
    assert derived.loc["l2", LOW_TOTAL_DIV_DOG_COLUMN] == 0.0  # not divisional
    assert derived.loc["l3", LOW_TOTAL_DIV_DOG_COLUMN] == 0.0  # total > 42
    assert derived.loc["l4", LOW_TOTAL_DIV_DOG_COLUMN] == 0.0  # home favored, not a dog
    assert derived.loc["l5", LOW_TOTAL_DIV_DOG_COLUMN] == 0.0  # pick'em, not a dog
    assert derived.loc["l6", LOW_TOTAL_DIV_DOG_COLUMN] == 0.0  # missing opener total


def test_low_total_div_dog_missing_total_never_silently_satisfies_threshold() -> None:
    schedule = _schedule([{"game_id": "l7", "div_game": 1}])
    opener_lines = pd.DataFrame(
        {
            "game_id": ["l7"],
            "tue_open_home_spread": [-5.0],
            "tue_open_total_line": [float("nan")],
        }
    )
    derived = derive_low_total_div_home_dog_features(schedule, opener_lines).set_index("game_id")
    assert derived.loc["l7", LOW_TOTAL_DIV_DOG_COLUMN] == 0.0


def test_low_total_div_dog_never_reads_an_outcome_column() -> None:
    schedule = _low_total_schedule()
    schedule["home_score"] = 10.0
    opener_lines = _low_total_opener_lines()
    before = derive_low_total_div_home_dog_features(schedule, opener_lines).set_index("game_id")
    mutated = schedule.copy()
    mutated["home_score"] = -999.0
    after = derive_low_total_div_home_dog_features(mutated, opener_lines).set_index("game_id")
    pd.testing.assert_series_equal(
        before[LOW_TOTAL_DIV_DOG_COLUMN], after[LOW_TOTAL_DIV_DOG_COLUMN]
    )


def test_low_total_div_dog_attach_accepts_supplied_opener_lines() -> None:
    schedule = _low_total_schedule()
    features = pd.DataFrame({"game_id": schedule["game_id"]})
    widened = attach_low_total_div_home_dog_features(
        features, schedule=schedule, opener_lines=_low_total_opener_lines()
    )
    assert sorted(set(widened.columns) - set(features.columns)) == [LOW_TOTAL_DIV_DOG_COLUMN]
    assert widened.set_index("game_id").loc["l1", LOW_TOTAL_DIV_DOG_COLUMN] == 1.0


# ---------------------------------------------------------------------------
# LEAD-35: September heat-humidity home edge
# ---------------------------------------------------------------------------


def _heat_game(
    game_id: str,
    home: str,
    away: str,
    *,
    week: int = 1,
    roof: str = "outdoors",
    gametime: str = "13:00",
    game_type: str = "REG",
) -> dict:
    return {
        "game_id": game_id,
        "season": 2024,
        "week": week,
        "game_type": game_type,
        "home_team": home,
        "away_team": away,
        "roof": roof,
        "gametime": gametime,
    }


def test_sept_heat_unconditional_home_team_qualifies_at_1pm_local() -> None:
    schedule = _schedule([_heat_game("h1", "MIA", "BUF")])
    derived = derive_sept_heat_home_features(schedule).set_index("game_id")
    assert derived.loc["h1", SEPT_HEAT_COLUMN] == 1.0


def test_sept_heat_atl_requires_open_air_roof() -> None:
    schedule = _schedule(
        [
            _heat_game("h2", "ATL", "GB", roof="open"),
            _heat_game("h3", "ATL", "GB", roof="dome"),
            _heat_game("h4", "ATL", "GB", roof="closed"),
        ]
    )
    derived = derive_sept_heat_home_features(schedule).set_index("game_id")
    assert derived.loc["h2", SEPT_HEAT_COLUMN] == 1.0
    assert derived.loc["h3", SEPT_HEAT_COLUMN] == 0.0
    assert derived.loc["h4", SEPT_HEAT_COLUMN] == 0.0


def test_sept_heat_hou_local_time_conversion() -> None:
    """HOU is Central (1h behind ET): a 14:00 ET kickoff IS 1 PM local and
    qualifies; HOU's normal early-window 13:00 ET kickoff is NOON local and
    does NOT -- the measured consequence stated in the predeclaration doc."""

    schedule = _schedule(
        [
            _heat_game("h5", "HOU", "BUF", roof="open", gametime="14:00"),
            _heat_game("h6", "HOU", "BUF", roof="open", gametime="13:00"),
        ]
    )
    derived = derive_sept_heat_home_features(schedule).set_index("game_id")
    assert derived.loc["h5", SEPT_HEAT_COLUMN] == 1.0
    assert derived.loc["h6", SEPT_HEAT_COLUMN] == 0.0


def test_sept_heat_requires_cold_visitor() -> None:
    schedule = _schedule([_heat_game("h7", "MIA", "TB")])  # TB is not on the cold list
    derived = derive_sept_heat_home_features(schedule).set_index("game_id")
    assert derived.loc["h7", SEPT_HEAT_COLUMN] == 0.0


def test_sept_heat_requires_week_le_3() -> None:
    schedule = _schedule([_heat_game("h8", "MIA", "BUF", week=4)])
    derived = derive_sept_heat_home_features(schedule).set_index("game_id")
    assert derived.loc["h8", SEPT_HEAT_COLUMN] == 0.0


def test_sept_heat_requires_reg_season() -> None:
    schedule = _schedule([_heat_game("h9", "MIA", "BUF", week=1, game_type="WC")])
    derived = derive_sept_heat_home_features(schedule).set_index("game_id")
    assert derived.loc["h9", SEPT_HEAT_COLUMN] == 0.0


def test_sept_heat_requires_1pm_local_hour_bucket() -> None:
    schedule = _schedule(
        [
            _heat_game("h10", "MIA", "BUF", gametime="12:59"),
            _heat_game("h11", "MIA", "BUF", gametime="13:59"),
            _heat_game("h12", "MIA", "BUF", gametime="14:00"),
        ]
    )
    derived = derive_sept_heat_home_features(schedule).set_index("game_id")
    assert derived.loc["h10", SEPT_HEAT_COLUMN] == 0.0
    assert derived.loc["h11", SEPT_HEAT_COLUMN] == 1.0
    assert derived.loc["h12", SEPT_HEAT_COLUMN] == 0.0


def test_sept_heat_never_reads_an_outcome_column() -> None:
    schedule = _schedule([_heat_game("h13", "MIA", "BUF")])
    schedule["result"] = 0.0
    schedule["home_score"] = 10.0
    before = derive_sept_heat_home_features(schedule).set_index("game_id")
    mutated = schedule.copy()
    mutated["result"] = 999.0
    mutated["home_score"] = -999.0
    after = derive_sept_heat_home_features(mutated).set_index("game_id")
    pd.testing.assert_series_equal(before[SEPT_HEAT_COLUMN], after[SEPT_HEAT_COLUMN])


def test_sept_heat_requires_schedule_columns() -> None:
    schedule = _schedule([_heat_game("h14", "MIA", "BUF")]).drop(columns=["gametime"])
    with pytest.raises(DataContractError, match="gametime"):
        derive_sept_heat_home_features(schedule)


def test_sept_heat_attach_is_purely_additive() -> None:
    schedule = _schedule([_heat_game("h15", "MIA", "BUF")])
    features = pd.DataFrame({"game_id": schedule["game_id"], "some_existing_feature": 1.0})
    widened = attach_sept_heat_home_features(features, schedule=schedule)
    assert sorted(set(widened.columns) - set(features.columns)) == [SEPT_HEAT_COLUMN]
    pd.testing.assert_frame_equal(features, widened[features.columns], check_exact=True)


# ---------------------------------------------------------------------------
# Registered candidate profiles: production plus exactly the one column
# ---------------------------------------------------------------------------

WAVE_2_CANDIDATE_KEYS = ("new_stadium", "dome_shootout", "low_total_div_dog", "sept_heat")


@pytest.mark.parametrize("key", WAVE_2_CANDIDATE_KEYS)
def test_wave_2_registered_profile_is_production_plus_the_declared_one_column(key: str) -> None:
    candidate = sfop.CANDIDATES[key]
    baseline = set(margin_feature_columns("market_residual", sfop.BASELINE_PROFILE))
    treatment = set(margin_feature_columns("market_residual", candidate.profile))
    assert treatment - baseline == {candidate.column}
    assert baseline - treatment == set()


@pytest.mark.parametrize("key", WAVE_2_CANDIDATE_KEYS)
def test_wave_2_candidate_duck_types_with_the_template_profile_identity(key: str) -> None:
    candidate = sfop.CANDIDATES[key]
    columns = margin_feature_columns("market_residual", candidate.profile)
    frame = pd.DataFrame({column: [0.0] for column in columns})
    observed = sfop.confirmation.profile_identity(candidate, frame)
    assert observed["only_added_column"] == candidate.column
