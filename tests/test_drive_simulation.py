from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.drive_simulation import (
    DriveSimulatorModel,
    _candidate_pool,
    _outcome,
    fit_drive_simulator,
    simulate_drive_distribution,
)
from nfl_ats.pbp import PBP_SNAPSHOT_COLUMNS


def _schedules() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["past", "future-history"],
            "gameday": ["2025-09-01", "2026-09-01"],
            "home_team": ["A", "A"],
            "away_team": ["B", "B"],
        }
    )


def _pbp() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    drive_specs = (
        ("past", "A", "B", 1, 3_600, 3_450, 0, "Punt", 0, 75),
        ("past", "B", "A", 2, 3_440, 3_220, 0, "Touchdown", 7, 65),
        ("past", "A", "B", 3, 800, 780, -7, "Touchdown", 7, 45),
        ("past", "B", "A", 4, 770, 470, 7, "Punt", 0, 80),
        ("future-history", "A", "B", 1, 3_600, 3_599, 0, "Touchdown", 7, 1),
    )
    for game_id, offense, defense, drive, start, end, diff, result, points, yardline in drive_specs:
        for play_offset, seconds in enumerate((start, end), start=1):
            row = dict.fromkeys(PBP_SNAPSHOT_COLUMNS, np.nan)
            row.update(
                {
                    "play_id": drive * 10 + play_offset,
                    "game_id": game_id,
                    "season": 2025 if game_id == "past" else 2026,
                    "season_type": "REG",
                    "week": 1,
                    "home_team": "A",
                    "away_team": "B",
                    "posteam": offense,
                    "defteam": defense,
                    "fixed_drive": drive,
                    "down": 1,
                    "play_type": "run",
                    "yards_gained": 5,
                    "pass_attempt": 0,
                    "rush_attempt": 1,
                    "sack": 0,
                    "qb_hit": 0,
                    "epa": 0.1,
                    "success": 1,
                    "wp": 0.5,
                    "qb_kneel": 0,
                    "qb_spike": 0,
                    "aborted_play": 0,
                    "yardline_100": yardline,
                    "game_seconds_remaining": seconds,
                    "interception": 0,
                    "fumble_lost": 0,
                    "score_differential": diff,
                    "fixed_drive_result": result,
                    "posteam_score": 0,
                    "posteam_score_post": points if play_offset == 2 else 0,
                    "play": 1,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _future_games() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["target"],
            "gameday": ["2026-09-10"],
            "home_team": ["A"],
            "away_team": ["B"],
        }
    )


def test_fit_is_invariant_to_post_cutoff_play_by_play() -> None:
    pbp = _pbp()
    model = fit_drive_simulator(pbp, _schedules(), training_max_gameday="2025-12-31")
    changed = pbp.copy()
    future = changed["game_id"].eq("future-history")
    changed.loc[future, "posteam_score_post"] = 0
    changed.loc[future, "yardline_100"] = 99
    changed_model = fit_drive_simulator(changed, _schedules(), training_max_gameday="2025-12-31")

    pd.testing.assert_frame_equal(model.observations, changed_model.observations)
    assert model.training_games == 1
    assert model.training_drives == 4
    assert set(model.observations["game_state"]) == {
        "neutral",
        "late_trailing",
        "late_leading",
    }


def test_drive_identity_includes_offense_when_fixed_drive_numbers_repeat() -> None:
    pbp = _pbp()
    repeated = pbp.loc[pbp["game_id"].eq("past") & pbp["fixed_drive"].eq(1)].copy()
    repeated["posteam"] = "B"
    repeated["defteam"] = "A"
    repeated["play_id"] = repeated["play_id"] + 1_000
    pbp = pd.concat([pbp, repeated], ignore_index=True)

    model = fit_drive_simulator(pbp, _schedules(), training_max_gameday="2025-12-31")

    assert model.training_drives == 5


def test_simulation_is_deterministic_and_retains_auditable_possessions() -> None:
    model = fit_drive_simulator(_pbp(), _schedules(), training_max_gameday="2025-12-31")
    first = simulate_drive_distribution(model, _future_games(), samples=40, seed=20260902)
    second = simulate_drive_distribution(model, _future_games(), samples=40, seed=20260902)

    pd.testing.assert_frame_equal(first.games, second.games)
    pd.testing.assert_frame_equal(first.drives, second.drives)
    assert first.games["simulation_id"].nunique() == 40
    assert {
        "offense",
        "start_yardline_100",
        "outcome",
        "sampled_drive_seconds",
        "drive_seconds",
        "game_state",
        "profile_source",
    }.issubset(first.drives.columns)
    assert (
        first.drives.groupby("simulation_id")["offense"]
        .apply(lambda teams: teams.ne(teams.shift()).iloc[1:].all())
        .all()
    )
    scores = (
        first.drives.groupby(["simulation_id", "offense"])["points"].sum().unstack(fill_value=0)
    )
    assert np.array_equal(first.games["home_score"], scores["A"])
    assert np.array_equal(first.games["away_score"], scores["B"])


def test_late_game_state_selects_distinct_empirical_behavior() -> None:
    model = fit_drive_simulator(_pbp(), _schedules(), training_max_gameday="2025-12-31")
    result = simulate_drive_distribution(model, _future_games(), samples=200, seed=8)
    late = result.drives.loc[result.drives["game_state"].ne("neutral")]

    assert not late.empty
    leading = late.loc[late["game_state"].eq("late_leading")]
    trailing = late.loc[late["game_state"].eq("late_trailing")]
    assert not leading.empty
    assert not trailing.empty
    assert set(leading["sampled_drive_seconds"]) == {300}
    assert set(trailing["sampled_drive_seconds"]) == {20}
    assert leading["drive_seconds"].le(leading["sampled_drive_seconds"]).all()
    assert trailing["drive_seconds"].le(trailing["sampled_drive_seconds"]).all()
    assert set(leading["outcome"]) == {"punt"}
    assert set(trailing["outcome"]) == {"touchdown"}


def test_target_games_must_be_strictly_after_training_cutoff() -> None:
    model = fit_drive_simulator(_pbp(), _schedules(), training_max_gameday="2025-12-31")
    target = _future_games()
    target.loc[0, "gameday"] = "2025-12-31"
    with pytest.raises(DataContractError, match="strictly after"):
        simulate_drive_distribution(model, target, samples=1)


@pytest.mark.parametrize("samples", [0, -1, True, 1.5])
def test_invalid_sample_counts_are_rejected(samples: object) -> None:
    model = fit_drive_simulator(_pbp(), _schedules(), training_max_gameday="2025-12-31")
    with pytest.raises(ValueError, match="positive integer"):
        simulate_drive_distribution(model, _future_games(), samples=samples)  # type: ignore[arg-type]


def test_schedule_and_training_data_contracts_fail_closed() -> None:
    duplicate = _schedules().copy()
    duplicate.loc[1, "game_id"] = "past"
    with pytest.raises(DataContractError, match="unique"):
        fit_drive_simulator(_pbp(), duplicate, training_max_gameday="2025-12-31")

    no_training = _schedules()
    with pytest.raises(DataContractError, match="no training games"):
        fit_drive_simulator(_pbp(), no_training, training_max_gameday="2024-01-01")


@pytest.mark.parametrize(
    ("schedule", "message"),
    [
        (pd.DataFrame(), "missing columns"),
        (_schedules().iloc[0:0], "at least one"),
        (_schedules().assign(home_team=np.nan), "both teams"),
        (_schedules().assign(home_team="B"), "against itself"),
        (_schedules().assign(gameday=pd.NaT), "valid gamedays"),
    ],
)
def test_schedule_validation_rejects_malformed_inputs(schedule: pd.DataFrame, message: str) -> None:
    error = (ValueError, DataContractError) if schedule.empty else DataContractError
    with pytest.raises(error, match=message):
        fit_drive_simulator(_pbp(), schedule, training_max_gameday="2025-12-31")


def test_fit_rejects_invalid_cutoff_and_unusable_drives() -> None:
    with pytest.raises(ValueError, match="valid date"):
        fit_drive_simulator(_pbp(), _schedules(), training_max_gameday="not-a-date")

    no_plays = _pbp().copy()
    no_plays["qb_kneel"] = 1
    with pytest.raises(DataContractError, match="no eligible training drives"):
        fit_drive_simulator(no_plays, _schedules(), training_max_gameday="2025-12-31")

    incomplete = _pbp().copy()
    incomplete["yardline_100"] = np.nan
    with pytest.raises(DataContractError, match="no complete training drives"):
        fit_drive_simulator(incomplete, _schedules(), training_max_gameday="2025-12-31")


def test_outcome_normalization_covers_scoreless_drive_endings() -> None:
    assert _outcome(pd.Series({"points": 3, "result": "Field Goal", "drive_turnover": 0})) == (
        "field_goal"
    )
    assert _outcome(pd.Series({"points": 0, "result": "Interception", "drive_turnover": 1})) == (
        "turnover"
    )
    assert _outcome(
        pd.Series({"points": 0, "result": "Missed Field Goal", "drive_turnover": 0})
    ) == ("missed_field_goal")
    assert _outcome(pd.Series({"points": 0, "result": "End of Half", "drive_turnover": 0})) == (
        "scoreless"
    )


def test_profile_fallbacks_are_named_and_exhaustive() -> None:
    observations = pd.DataFrame(
        {
            "posteam": ["A", "C", "E"],
            "defteam": ["B", "D", "F"],
            "game_state": ["neutral", "neutral", "late_leading"],
        }
    )
    rng = np.random.default_rng(12)
    paired_sources = {
        _candidate_pool(observations, offense="A", defense="B", game_state="neutral", rng=rng)[1]
        for _ in range(20)
    }
    assert paired_sources == {"offense_state", "defense_state"}
    assert (
        _candidate_pool(observations, offense="A", defense="Z", game_state="neutral", rng=rng)[1]
        == "offense_state"
    )
    assert (
        _candidate_pool(observations, offense="Z", defense="B", game_state="neutral", rng=rng)[1]
        == "defense_state"
    )
    assert (
        _candidate_pool(observations, offense="Z", defense="Z", game_state="late_leading", rng=rng)[
            1
        ]
        == "league_state"
    )

    neutral_sources = {
        _candidate_pool(
            observations, offense="A", defense="B", game_state="late_trailing", rng=rng
        )[1]
        for _ in range(20)
    }
    assert neutral_sources == {"offense_neutral", "defense_neutral"}
    assert (
        _candidate_pool(
            observations, offense="A", defense="Z", game_state="late_trailing", rng=rng
        )[1]
        == "offense_neutral"
    )
    assert (
        _candidate_pool(
            observations, offense="Z", defense="B", game_state="late_trailing", rng=rng
        )[1]
        == "defense_neutral"
    )
    assert (
        _candidate_pool(
            observations, offense="Z", defense="Z", game_state="late_trailing", rng=rng
        )[1]
        == "league_all"
    )


def test_simulation_rejects_invalid_drive_limit_and_model_observations() -> None:
    model = fit_drive_simulator(_pbp(), _schedules(), training_max_gameday="2025-12-31")
    with pytest.raises(ValueError, match="max_drives_per_game"):
        simulate_drive_distribution(model, _future_games(), samples=1, max_drives_per_game=0)

    empty_model = DriveSimulatorModel(pd.DataFrame(), "2025-12-31", 0, 0)
    with pytest.raises(DataContractError, match="no usable observations"):
        simulate_drive_distribution(empty_model, _future_games(), samples=1)

    missing_model = DriveSimulatorModel(pd.DataFrame({"posteam": ["A"]}), "2025-12-31", 1, 1)
    with pytest.raises(DataContractError, match="defteam"):
        simulate_drive_distribution(missing_model, _future_games(), samples=1)
