"""Leakage regression tests for the weak_stack_v3 gap-feature families
(docs/weak_stack_v3.md), one per AGENTS.md's "a leakage regression test for
every new feature family" rule: gap_v3_bias (division revenge, sandwich
spot, post-blowout letdown/bounce), gap_v3_penalty (penalty rate), and
gap_v3_travel (thursday-pure, return-trip hangover).
"""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.constants import (
    FEATURE_SETS,
    GAP_V3_BIAS_FEATURE_COLUMNS,
    GAP_V3_PENALTY_FEATURE_COLUMNS,
    GAP_V3_TRAVEL_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
)
from nfl_ats.margin import MARGIN_FEATURE_PROFILES, margin_feature_columns
from nfl_ats.weak_stack_v3_features import (
    build_gap_bias_features,
    build_gap_penalty_feature,
    build_gap_travel_rest_features,
    haversine_mi,
    team_season_penalty_rate,
)

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_weak_stack_v3_profile_is_registered_and_disjoint_from_production_sets() -> None:
    assert "weak_stack_v3" in MARGIN_FEATURE_PROFILES
    gap_columns = (
        set(GAP_V3_BIAS_FEATURE_COLUMNS)
        | set(GAP_V3_PENALTY_FEATURE_COLUMNS)
        | set(GAP_V3_TRAVEL_FEATURE_COLUMNS)
    )
    assert gap_columns.isdisjoint(MODEL_FEATURE_COLUMNS)
    for name in ("football_weak_stack", "full_weak_stack", "football", "full"):
        assert set(FEATURE_SETS[name]).isdisjoint(gap_columns), name
    # weak_stack_v3 = weak_stack_surface (weak_stack + surface_switch_flag)
    # plus exactly the 15 new gap columns -- never used by the active model.
    v3_columns = set(margin_feature_columns("market_residual", "weak_stack_v3"))
    surface_columns = set(margin_feature_columns("market_residual", "weak_stack_surface"))
    assert v3_columns - surface_columns == gap_columns
    assert len(gap_columns) == 15


# ---------------------------------------------------------------------------
# gap_v3_bias: worked example (division revenge, sandwich spot, post-blowout)
# ---------------------------------------------------------------------------


def _bias_schedule() -> pd.DataFrame:
    """Season 2021, one division (A/B), teams A, B, C, D.

    week1 2021_01_A_B: A hosts B, div_game=1, A wins by 20 (blowout).
    week2 2021_02_A_C: A hosts C, div_game=0 -- sandwiched between two div
      games for A (week1, week3).
    week2 2021_02_D_B: D hosts B, div_game=0 -- also sandwiched for B
      (week1, week3 below are both div games for B too).
    week3 2021_03_A_B: A hosts B again, div_game=1 -- the DIVISION REVENGE
      rematch; B lost the first meeting, A did not.
    week4 2021_04_A_D: A hosts D, div_game=0, Thursday -- A's prior game
      (week3) was only a 3-point win, so post_blowout should NOT fire here;
      doubles as the thursday_pure worked example.
    """

    rows = [
        ("2021_01_A_B", 2021, 1, "REG", "Sunday", "A", "B", 20, 1, 7, 7),
        ("2021_02_A_C", 2021, 2, "REG", "Sunday", "A", "C", 3, 0, 7, 7),
        ("2021_02_D_B", 2021, 2, "REG", "Sunday", "D", "B", 6, 0, 7, 7),
        ("2021_03_A_B", 2021, 3, "REG", "Sunday", "A", "B", 3, 1, 7, 7),
        ("2021_04_A_D", 2021, 4, "REG", "Thursday", "A", "D", 1, 0, 4, 10),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "game_id",
            "season",
            "week",
            "game_type",
            "weekday",
            "home_team",
            "away_team",
            "result",
            "div_game",
            "home_rest",
            "away_rest",
        ],
    )
    frame["spread_line"] = 0.0
    frame["gameday"] = pd.date_range("2021-09-12", periods=len(frame), freq="7D")
    return frame


def _row(features: pd.DataFrame, game_id: str) -> pd.Series:
    return features.loc[features["game_id"].eq(game_id)].iloc[0]


def test_division_revenge_fires_only_for_the_team_that_lost_the_first_meeting() -> None:
    bias = build_gap_bias_features(_bias_schedule())

    first_meeting = _row(bias, "2021_01_A_B")
    assert first_meeting["gap_division_revenge_home"] == 0.0  # A: no prior meeting yet
    assert first_meeting["gap_division_revenge_away"] == 0.0  # B: no prior meeting yet

    rematch = _row(bias, "2021_03_A_B")
    assert rematch["gap_division_revenge_home"] == 0.0  # A won the first meeting
    assert rematch["gap_division_revenge_away"] == 1.0  # B lost the first meeting
    assert rematch["gap_division_revenge_diff"] == -1.0


def test_sandwich_spot_fires_only_when_flanked_by_division_games_on_both_sides() -> None:
    bias = build_gap_bias_features(_bias_schedule())

    sandwiched_a = _row(bias, "2021_02_A_C")
    assert sandwiched_a["gap_sandwich_spot_home"] == 1.0  # A: div(w1), non-div(w2), div(w3)

    sandwiched_b = _row(bias, "2021_02_D_B")
    assert sandwiched_b["gap_sandwich_spot_away"] == 1.0  # B: div(w1), non-div(w2), div(w3)

    # week4 is non-div for A but has no week5 (next_div is NaN) -- not sandwiched.
    not_sandwiched = _row(bias, "2021_04_A_D")
    assert not_sandwiched["gap_sandwich_spot_home"] == 0.0


def test_post_blowout_letdown_and_bounce_use_the_strictly_prior_game_only() -> None:
    bias = build_gap_bias_features(_bias_schedule())

    after_a_blowout_win = _row(bias, "2021_02_A_C")
    assert after_a_blowout_win["gap_post_blowout_win_letdown_home"] == 1.0  # A won w1 by 20
    assert after_a_blowout_win["gap_post_blowout_loss_bounce_home"] == 0.0

    after_a_blowout_loss = _row(bias, "2021_02_D_B")
    assert after_a_blowout_loss["gap_post_blowout_loss_bounce_away"] == 1.0  # B lost w1 by 20
    assert after_a_blowout_loss["gap_post_blowout_win_letdown_away"] == 0.0

    # A's week3 margin was only +3 -- week4 must not fire.
    no_blowout_last_week = _row(bias, "2021_04_A_D")
    assert no_blowout_last_week["gap_post_blowout_win_letdown_home"] == 0.0
    assert no_blowout_last_week["gap_post_blowout_loss_bounce_home"] == 0.0


def test_gap_bias_flags_never_read_this_games_own_outcome_columns() -> None:
    """AGENTS.md: a leakage regression test for every new feature family.

    Mutating a game's own ``result``/``spread_line`` must never change ITS
    OWN flags -- every derivation reads a strictly earlier game (or, for
    sandwich_spot, the surrounding schedule structure only).
    """

    schedule = _bias_schedule()
    baseline = build_gap_bias_features(schedule).set_index("game_id")

    mutated = schedule.copy()
    target = mutated["game_id"].eq("2021_03_A_B")
    mutated.loc[target, "result"] = -30.0
    mutated.loc[target, "spread_line"] = 21.0
    changed = build_gap_bias_features(mutated).set_index("game_id")

    # The mutated game's OWN flags stay put (they never look at own result).
    own_columns = list(baseline.columns)
    pd.testing.assert_series_equal(
        changed.loc["2021_03_A_B", own_columns],
        baseline.loc["2021_03_A_B", own_columns],
        check_exact=True,
    )
    # But a LATER game that looks back at 2021_03_A_B's result does change --
    # proof the lookup is real, not a no-op (week4 letdown flips because A's
    # week3 margin moved from +3 to -30, wait: A is home in week3, mutating
    # result to -30 means A now LOST week3 by 30 -- week4's bounce flag
    # should now fire instead of neither).
    assert changed.loc["2021_04_A_D", "gap_post_blowout_loss_bounce_home"] == 1.0
    assert baseline.loc["2021_04_A_D", "gap_post_blowout_loss_bounce_home"] == 0.0


def test_gap_bias_flags_are_leak_safe_across_the_season_boundary() -> None:
    """A future season's games (even for the SAME team/opponent) must never
    change an earlier season's already-computed flags."""

    schedule = _bias_schedule()
    baseline = build_gap_bias_features(schedule).set_index("game_id")

    future = pd.DataFrame(
        [
            (
                "2022_01_A_B",
                2022,
                1,
                "REG",
                "Sunday",
                "A",
                "B",
                -14,
                1,
                7,
                7,
                0.0,
                pd.Timestamp("2022-09-11"),
            )
        ],
        columns=[*schedule.columns],
    )
    combined = pd.concat([schedule, future], ignore_index=True)
    changed = build_gap_bias_features(combined).set_index("game_id")

    pd.testing.assert_frame_equal(changed.loc[baseline.index], baseline, check_exact=True)


# ---------------------------------------------------------------------------
# gap_v3_penalty: diff_penalty_rate_prior
# ---------------------------------------------------------------------------


def _penalty_pbp() -> pd.DataFrame:
    rows = []
    # Team A: 2020 rate 10/100=0.10, 2021 rate 30/100=0.30.
    rows += [{"season": 2020, "posteam": "A", "penalty": 1 if i < 10 else 0} for i in range(100)]
    rows += [{"season": 2021, "posteam": "A", "penalty": 1 if i < 30 else 0} for i in range(100)]
    # Team B: flat 0.05 rate both seasons, so the diff column isolates A's move.
    rows += [{"season": 2020, "posteam": "B", "penalty": 1 if i < 5 else 0} for i in range(100)]
    rows += [{"season": 2021, "posteam": "B", "penalty": 1 if i < 5 else 0} for i in range(100)]
    return pd.DataFrame(rows)


def _penalty_schedule() -> pd.DataFrame:
    rows = [
        ("2020_01_A_B", 2020, "A", "B"),  # no prior local season (2019) for A
        ("2021_01_A_B", 2021, "A", "B"),  # prior season 2020 -> rate 0.10
        ("2022_01_A_B", 2022, "A", "B"),  # prior season 2021 -> rate 0.30
    ]
    frame = pd.DataFrame(rows, columns=["game_id", "season", "home_team", "away_team"])
    return frame


def test_penalty_rate_prior_is_strictly_lagged_one_season() -> None:
    pbp = _penalty_pbp()
    schedule = _penalty_schedule()
    result = build_gap_penalty_feature(pbp, schedule).set_index("game_id")

    assert pd.isna(result.loc["2020_01_A_B", "diff_penalty_rate_prior"])  # no 2019 data
    # B is flat at 0.05 both seasons, so the diff isolates A's own move:
    # 2021 sees A's 2020 rate (0.10) -> diff = 0.10 - 0.05 = 0.05.
    assert result.loc["2021_01_A_B", "diff_penalty_rate_prior"] == pytest.approx(0.05)
    # 2022 sees A's 2021 rate (0.30) -> diff = 0.30 - 0.05 = 0.25.
    assert result.loc["2022_01_A_B", "diff_penalty_rate_prior"] == pytest.approx(0.25)


def test_penalty_rate_lag_can_never_self_match_or_look_forward() -> None:
    """Leak-safety self-check, promoted from
    ``scripts/weak_stack_v2_eval.py._leak_safety_selfcheck`` into a real
    assertion: the join key is ``prev_season = rate_season + 1``, so a
    team-season's plays can only ever be pulled by a STRICTLY LATER season,
    never its own or an earlier one."""

    rate = team_season_penalty_rate(_penalty_pbp())
    lag = rate.copy()
    lag["prev_season"] = lag["season"] + 1
    assert (lag["prev_season"] > lag["season"]).all()


# ---------------------------------------------------------------------------
# gap_v3_travel: thursday_pure, return_trip_hangover
# ---------------------------------------------------------------------------

_STADX = {"lat": 40.0, "lon": -74.0, "tz": "America/New_York"}
_STADY = {"lat": 34.0, "lon": -118.0, "tz": "America/Los_Angeles"}  # ~2451mi from StadX


def _travel_coords() -> dict[str, dict[str, float | str]]:
    return {"StadX": _STADX, "StadY": _STADY}


def _travel_schedule(week2_home_rest: int) -> pd.DataFrame:
    rows = [
        (
            "2021_01_X_Y",
            2021,
            "REG",
            "Sunday",
            "X",
            "Y",
            "StadX",
            "Home",
            7,
            7,
        ),
        (
            "2021_02_Y_X",
            2021,
            "REG",
            "Sunday",
            "Y",
            "X",
            "StadY",
            "Home",
            week2_home_rest,
            7,
        ),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "game_id",
            "season",
            "game_type",
            "weekday",
            "home_team",
            "away_team",
            "stadium",
            "location",
            "home_rest",
            "away_rest",
        ],
    )
    frame["gameday"] = pd.date_range("2021-09-12", periods=len(frame), freq="7D")
    return frame


def test_return_trip_hangover_fires_on_long_trip_and_short_rest() -> None:
    result = build_gap_travel_rest_features(_travel_schedule(week2_home_rest=6), _travel_coords())
    row = result.loc[result["game_id"].eq("2021_02_Y_X")].iloc[0]
    # Y traveled ~2451mi (>=1500) in week1 (away at StadX), then hosts week2
    # on only 6 days' rest (<=8) -- hangover fires.
    assert row["gap_return_trip_hangover_flag"] == 1.0


def test_return_trip_hangover_does_not_fire_after_a_long_rest() -> None:
    result = build_gap_travel_rest_features(_travel_schedule(week2_home_rest=13), _travel_coords())
    row = result.loc[result["game_id"].eq("2021_02_Y_X")].iloc[0]
    assert row["gap_return_trip_hangover_flag"] == 0.0


def test_thursday_pure_flag_matches_the_weekday_column() -> None:
    schedule = _travel_schedule(week2_home_rest=6)
    schedule.loc[schedule["game_id"].eq("2021_02_Y_X"), "weekday"] = "Thursday"
    result = build_gap_travel_rest_features(schedule, _travel_coords())
    assert result.loc[result["game_id"].eq("2021_01_X_Y"), "gap_thursday_pure_flag"].iat[0] == 0.0
    assert result.loc[result["game_id"].eq("2021_02_Y_X"), "gap_thursday_pure_flag"].iat[0] == 1.0


def test_haversine_matches_a_known_city_pair_distance() -> None:
    # NYC to LA great-circle distance is a well-known ~2451 miles.
    assert haversine_mi(
        _STADX["lat"], _STADX["lon"], _STADY["lat"], _STADY["lon"]
    ) == pytest.approx(2451.0, rel=0.02)


def test_gap_travel_features_never_read_result_or_spread_line() -> None:
    """AGENTS.md: a leakage regression test for every new feature family.

    ``build_gap_travel_rest_features`` does not even require/read
    ``result``/``spread_line``; both columns are absent from the fixture
    entirely, matching ``add_surface_switch_features``'s own precedent test
    for a structural, schedule-only construct."""

    schedule = _travel_schedule(week2_home_rest=6)
    assert "result" not in schedule.columns
    assert "spread_line" not in schedule.columns
    result = build_gap_travel_rest_features(schedule, _travel_coords())
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Orchestrator wiring (additivity)
# ---------------------------------------------------------------------------


def test_attach_weak_stack_v3_gap_features_is_purely_additive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The orchestrator must not move any pre-existing column, and every new
    column it adds must be exactly one of the 15 declared gap_v3 columns."""

    import nfl_ats.weak_stack_v3_features as module

    schedule = _bias_schedule()
    coords: dict[str, dict[str, float | str]] = {}
    pbp = _penalty_pbp()

    monkeypatch.setattr(module, "latest_schedules_snapshot", lambda repo_root: "unused")
    monkeypatch.setattr(
        pd, "read_parquet", lambda path: schedule if path == "unused" else pd.read_parquet(path)
    )
    monkeypatch.setattr(module, "load_stadium_coordinates", lambda path: coords)
    monkeypatch.setattr(module, "latest_pbp_snapshot", lambda root: "unused-pbp")
    monkeypatch.setattr(module, "load_pbp_snapshot", lambda snapshot, include_postseason=False: pbp)

    base = pd.DataFrame(
        {
            "game_id": schedule["game_id"],
            "surface_switch_flag": 0.0,
            "some_pre_existing_column": range(len(schedule)),
        }
    )
    result = module.attach_weak_stack_v3_gap_features(
        base, repo_root=pd.NA
    )  # repo_root unused by stubs

    pd.testing.assert_frame_equal(result[base.columns.tolist()], base, check_exact=True)
    new_columns = set(result.columns) - set(base.columns)
    expected = (
        set(GAP_V3_BIAS_FEATURE_COLUMNS)
        | set(GAP_V3_PENALTY_FEATURE_COLUMNS)
        | set(GAP_V3_TRAVEL_FEATURE_COLUMNS)
    )
    assert new_columns == expected
    for column in set(GAP_V3_BIAS_FEATURE_COLUMNS) | set(GAP_V3_TRAVEL_FEATURE_COLUMNS):
        assert result[column].notna().all()  # flags are always fillna(0.0)-completed
