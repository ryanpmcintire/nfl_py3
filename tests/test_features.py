from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from nfl_ats.constants import (
    BIAS_FEATURE_COLUMNS,
    FEATURE_FAMILIES,
    FEATURE_SETS,
    GRAPH_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    OUTCOME_COLUMNS,
    SURFACE_SWITCH_FEATURE_COLUMNS,
)
from nfl_ats.data import DataContractError
from nfl_ats.features import (
    DecisionLineOverride,
    add_ats_outcomes,
    add_surface_switch_features,
    apply_decision_lines,
    build_game_features,
    build_team_game_metrics,
)


def test_ats_target_sign_and_push() -> None:
    frame = pd.DataFrame({"result": [7, -1, 3], "spread_line": [3, 2, 3]})
    result = add_ats_outcomes(frame)
    assert result["ats_margin"].tolist() == [4.0, -3.0, 0.0]
    assert result["home_cover"].iloc[:2].tolist() == [1.0, 0.0]
    assert pd.isna(result["home_cover"].iloc[2])


def test_feature_table_has_one_row_per_game_and_no_label_features(
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    schedules, stats = schedules_and_stats
    schedules = schedules.assign(gametime="20:20", weekday="Sunday")
    features = build_game_features(schedules, stats, span=3, min_periods=1)
    assert features["game_id"].is_unique
    assert len(features) == len(schedules)
    assert set(MODEL_FEATURE_COLUMNS).issubset(features.columns)
    assert set(GRAPH_FEATURE_COLUMNS).issubset(features.columns)
    assert set(MODEL_FEATURE_COLUMNS).isdisjoint(OUTCOME_COLUMNS)
    assert features.loc[0, "elo_diff"] == pytest.approx(55.0)
    assert features.loc[0, "kickoff"] == pd.Timestamp("2022-09-12 00:20:00+00:00")


def test_current_game_stats_cannot_change_current_pregame_features(
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    schedules, stats = schedules_and_stats
    baseline = build_game_features(schedules, stats, span=3, min_periods=1)
    changed_stats = stats.copy()
    second_game = schedules.loc[1, "game_id"]
    mask = changed_stats["game_id"].eq(second_game) & changed_stats["team"].eq("A")
    changed_stats.loc[mask, "passing_epa"] = 1_000.0
    changed = build_game_features(schedules, changed_stats, span=3, min_periods=1)

    column = "home_off_pass_epa_per_play"
    assert changed.loc[1, column] == pytest.approx(baseline.loc[1, column])
    assert changed.loc[2, column] != pytest.approx(baseline.loc[2, column])


def test_team_metric_builder_requires_two_teams(
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    schedules, stats = schedules_and_stats
    malformed = stats.loc[
        ~((stats["game_id"] == schedules.loc[0, "game_id"]) & (stats["team"] == "B"))
    ]
    with pytest.raises(DataContractError, match="Expected two"):
        build_team_game_metrics(schedules, malformed)


def test_bias_family_is_registered_but_outside_every_frozen_feature_set() -> None:
    assert FEATURE_FAMILIES["bias"] == BIAS_FEATURE_COLUMNS
    assert BIAS_FEATURE_COLUMNS == (
        "bias_playoff_holdover_home",
        "bias_playoff_holdover_away",
        "bias_playoff_holdover_diff",
        "bias_prior_week_ats_home",
        "bias_prior_week_ats_away",
        "bias_prior_week_ats_diff",
        "bias_week2_anchor_home",
        "bias_week2_anchor_away",
        "bias_week2_anchor_diff",
    )
    assert set(BIAS_FEATURE_COLUMNS).isdisjoint(MODEL_FEATURE_COLUMNS)
    for name in ("full", "full_player", "full_player_value", "football", "football_player"):
        assert set(FEATURE_SETS[name]).isdisjoint(BIAS_FEATURE_COLUMNS), name


def test_bias_features_cannot_see_the_result_of_their_own_game(
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    schedules, stats = schedules_and_stats
    baseline = build_game_features(schedules, stats, span=3, min_periods=1)
    changed_schedules = schedules.copy()
    changed_schedules.loc[2, "result"] = 40.0
    changed = build_game_features(changed_schedules, stats, span=3, min_periods=1)

    assert set(BIAS_FEATURE_COLUMNS).issubset(baseline.columns)
    for column in ("bias_prior_week_ats_home", "bias_prior_week_ats_away"):
        assert changed.loc[2, column] == pytest.approx(baseline.loc[2, column])
        assert changed.loc[3, column] != pytest.approx(baseline.loc[3, column])
    assert pd.isna(baseline.loc[0, "bias_prior_week_ats_home"])
    assert baseline["bias_playoff_holdover_home"].eq(0.0).all()


def _surface_switch_schedule() -> pd.DataFrame:

    rows = [
        ("2026_01_GRASSAWAY_OPP1", 2026, "REG", "GRASSAWAY", "OPP1", "grass"),
        ("2026_02_GRASSAWAY_OPP2", 2026, "REG", "GRASSAWAY", "OPP2", "grass"),
        ("2026_03_TURFHOST_GRASSAWAY", 2026, "REG", "TURFHOST", "GRASSAWAY", "fieldturf"),
        ("2026_04_GRASSHOST_GRASSAWAY", 2026, "REG", "GRASSHOST", "GRASSAWAY", "grass"),
        ("2026_01_TURFAWAY_OPP3", 2026, "REG", "TURFAWAY", "OPP3", "fieldturf"),
        ("2026_02_TURFAWAY_OPP4", 2026, "REG", "TURFAWAY", "OPP4", "astroturf"),
        ("2026_03_TURFHOST2_TURFAWAY", 2026, "REG", "TURFHOST2", "TURFAWAY", "sportturf"),
        ("2026_01_NOSURF_OPPX", 2026, "REG", "NOSURF", "OPPX", ""),
        ("2026_05_TURFHOST3_NOSURF", 2026, "REG", "TURFHOST3", "NOSURF", "fieldturf"),
        ("2026_20_POSTHOST_GRASSAWAY", 2026, "POST", "POSTHOST", "GRASSAWAY", "fieldturf"),
    ]
    frame = pd.DataFrame(
        rows, columns=["game_id", "season", "game_type", "home_team", "away_team", "surface"]
    )
    frame["week"] = [1, 2, 3, 4, 1, 2, 3, 1, 5, 20]
    frame["gameday"] = pd.date_range("2026-09-10", periods=len(frame), freq="7D")
    return frame


def _surface_row(features: pd.DataFrame, game_id: str) -> pd.Series:
    return features.loc[features["game_id"].eq(game_id)].iloc[0]


def test_surface_switch_family_is_registered_but_outside_every_frozen_feature_set() -> None:
    assert FEATURE_FAMILIES["surface_switch"] == SURFACE_SWITCH_FEATURE_COLUMNS
    assert SURFACE_SWITCH_FEATURE_COLUMNS == ("surface_switch_flag",)
    assert set(SURFACE_SWITCH_FEATURE_COLUMNS).isdisjoint(MODEL_FEATURE_COLUMNS)
    admitting = {
        name
        for name, columns in FEATURE_SETS.items()
        if set(columns) & set(SURFACE_SWITCH_FEATURE_COLUMNS)
    }
    assert admitting == {
        "football_weak_stack_surface",
        "full_weak_stack_surface",
        "football_weak_stack_v3",
        "full_weak_stack_v3",
    }
    for name in (
        "full",
        "full_player",
        "full_player_value",
        "football",
        "football_player",
        "football_weak_stack",
        "full_weak_stack",
    ):
        assert set(FEATURE_SETS[name]).isdisjoint(SURFACE_SWITCH_FEATURE_COLUMNS), name


def test_surface_switch_flag_fires_on_grass_modal_visitor_onto_turf() -> None:
    schedule = _surface_switch_schedule()
    flagged = add_surface_switch_features(schedule, schedule)

    assert _surface_row(flagged, "2026_03_TURFHOST_GRASSAWAY")["surface_switch_flag"] == 1.0
    assert _surface_row(flagged, "2026_04_GRASSHOST_GRASSAWAY")["surface_switch_flag"] == 0.0
    assert _surface_row(flagged, "2026_03_TURFHOST2_TURFAWAY")["surface_switch_flag"] == 0.0
    assert _surface_row(flagged, "2026_05_TURFHOST3_NOSURF")["surface_switch_flag"] == 0.0
    assert _surface_row(flagged, "2026_20_POSTHOST_GRASSAWAY")["surface_switch_flag"] == 0.0


def test_surface_switch_flag_never_reads_outcome_columns() -> None:

    schedule = _surface_switch_schedule()
    schedule["result"] = 0.0
    schedule["spread_line"] = -3.0
    baseline = add_surface_switch_features(schedule, schedule).set_index("game_id")[
        "surface_switch_flag"
    ]

    mutated = schedule.copy()
    mutated.loc[mutated["game_id"].eq("2026_03_TURFHOST_GRASSAWAY"), "result"] = 99.0
    mutated.loc[mutated["game_id"].eq("2026_03_TURFHOST_GRASSAWAY"), "spread_line"] = 14.0
    changed = add_surface_switch_features(mutated, mutated).set_index("game_id")[
        "surface_switch_flag"
    ]

    pd.testing.assert_series_equal(changed, baseline, check_exact=True)


def test_surface_switch_flag_is_leak_safe_across_the_season_boundary() -> None:

    schedule = _surface_switch_schedule()
    baseline = add_surface_switch_features(schedule, schedule).set_index("game_id")[
        "surface_switch_flag"
    ]

    future = pd.DataFrame(
        [
            (
                "2027_01_GRASSAWAY_OPP1",
                2027,
                "REG",
                "GRASSAWAY",
                "OPP1",
                "fieldturf",
                1,
                pd.Timestamp("2027-09-09"),
            ),
            (
                "2027_03_TURFHOST_GRASSAWAY",
                2027,
                "REG",
                "TURFHOST",
                "GRASSAWAY",
                "fieldturf",
                3,
                pd.Timestamp("2027-09-23"),
            ),
        ],
        columns=schedule.columns,
    )
    combined = pd.concat([schedule, future], ignore_index=True)
    changed = add_surface_switch_features(combined, combined).set_index("game_id")[
        "surface_switch_flag"
    ]

    pd.testing.assert_series_equal(changed.loc[baseline.index], baseline, check_exact=True)


def test_surface_switch_features_land_in_build_game_features_and_leave_other_columns_untouched(
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    from nfl_ats import features as features_module

    schedules, stats = schedules_and_stats
    with_surface = build_game_features(schedules, stats, span=3, min_periods=1)
    assert SURFACE_SWITCH_FEATURE_COLUMNS[0] in with_surface.columns
    assert with_surface[SURFACE_SWITCH_FEATURE_COLUMNS[0]].eq(0.0).all()

    def _stub(games: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
        return games.assign(**dict.fromkeys(SURFACE_SWITCH_FEATURE_COLUMNS, 0.0))

    monkeypatch.setattr(features_module, "add_surface_switch_features", _stub)
    without_surface = build_game_features(schedules, stats, span=3, min_periods=1)

    pre_existing = [
        column for column in with_surface.columns if column not in SURFACE_SWITCH_FEATURE_COLUMNS
    ]
    assert list(without_surface.columns) == list(with_surface.columns)
    pd.testing.assert_frame_equal(
        with_surface[pre_existing],
        without_surface[pre_existing],
        check_exact=True,
    )


_EASTERN = ZoneInfo("America/New_York")

_POOL_CAPTURE = DecisionLineOverride(
    season=2022,
    week=6,
    lines={"2022_06_B_A": -1.5},
    source="splashsports.com",
    capture_id="2022_week06_20221011_noon",
    captured_at_utc="2022-10-11T12:45:00-04:00",
    captured_at=datetime(2022, 10, 11, 12, 45, tzinfo=_EASTERN),
)


def _with_upcoming_week(schedules: pd.DataFrame) -> pd.DataFrame:

    upcoming = schedules.iloc[[-1]].copy()
    upcoming["game_id"] = "2022_06_B_A"
    upcoming["week"] = 6
    upcoming["gameday"] = pd.Timestamp("2022-10-16")
    upcoming["away_score"] = float("nan")
    upcoming["home_score"] = float("nan")
    upcoming["result"] = float("nan")
    upcoming["spread_line"] = 2.0
    return pd.concat([schedules, upcoming], ignore_index=True)


def _with_played_week(schedules: pd.DataFrame) -> pd.DataFrame:

    played = _with_upcoming_week(schedules)
    played["gametime"] = "20:20"
    row = played["game_id"].eq("2022_06_B_A")
    played.loc[row, "away_score"] = 17.0
    played.loc[row, "home_score"] = 20.0
    played.loc[row, "result"] = 3.0
    return played


def test_pool_capture_becomes_the_decision_line_for_the_week_it_covers(
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
) -> None:

    schedules, stats = schedules_and_stats
    schedules = _with_upcoming_week(schedules)

    overridden, applied = apply_decision_lines(schedules, (_POOL_CAPTURE,))
    features = build_game_features(overridden, stats, span=3, min_periods=1)

    row = features.loc[features["game_id"].eq("2022_06_B_A")].iloc[0]
    assert row["spread_line"] == pytest.approx(-1.5)
    assert len(applied) == 1
    assert applied[0].game_ids == ("2022_06_B_A",)
    assert applied[0].changed_game_ids == ("2022_06_B_A",)
    assert applied[0].override.capture_id == "2022_week06_20221011_noon"


def test_pool_capture_leaves_every_uncaptured_row_bit_identical(
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
) -> None:

    schedules, stats = schedules_and_stats
    schedules = _with_upcoming_week(schedules)

    baseline = build_game_features(schedules, stats, span=3, min_periods=1)
    overridden, _ = apply_decision_lines(schedules, (_POOL_CAPTURE,))
    rebuilt = build_game_features(overridden, stats, span=3, min_periods=1)

    uncaptured = baseline["game_id"].ne("2022_06_B_A")
    assert int(uncaptured.sum()) == 5
    pd.testing.assert_frame_equal(
        baseline.loc[uncaptured].reset_index(drop=True),
        rebuilt.loc[uncaptured.to_numpy()].reset_index(drop=True),
        check_exact=True,
    )


def test_pool_capture_applies_to_a_played_game_when_the_board_predates_kickoff(
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
) -> None:

    schedules, _ = schedules_and_stats
    played = _with_played_week(schedules)

    overridden, applied = apply_decision_lines(played, (_POOL_CAPTURE,))

    row = overridden.loc[overridden["game_id"].eq("2022_06_B_A")].iloc[0]
    assert row["spread_line"] == pytest.approx(-1.5)
    assert row["result"] == pytest.approx(3.0)
    assert len(applied) == 1
    assert applied[0].changed_game_ids == ("2022_06_B_A",)


def test_pool_capture_refuses_a_played_game_when_the_board_postdates_kickoff(
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
) -> None:

    schedules, _ = schedules_and_stats
    played = _with_played_week(schedules)
    afterwards = DecisionLineOverride(
        season=2022,
        week=6,
        lines={"2022_06_B_A": -1.5},
        source="splashsports.com",
        capture_id="2022_week06_20221017_morning",
        captured_at_utc="2022-10-17T09:00:00-04:00",
        captured_at=datetime(2022, 10, 17, 9, 0, tzinfo=_EASTERN),
    )

    with pytest.raises(DataContractError, match="RETROACTIVE") as error:
        apply_decision_lines(played, (afterwards,))
    assert "2022_06_B_A" in str(error.value)


def test_pool_capture_refuses_a_played_game_when_the_capture_has_no_instant(
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
) -> None:

    schedules, _ = schedules_and_stats
    played = _with_played_week(schedules)
    unstamped = DecisionLineOverride(
        season=2022,
        week=6,
        lines={"2022_06_B_A": -1.5},
        source="splashsports.com",
        capture_id="2022_week06_unstamped",
    )

    with pytest.raises(DataContractError, match="missing timestamp is not permission"):
        apply_decision_lines(played, (unstamped,))


def test_pool_capture_refuses_a_week_it_only_half_covers(
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
) -> None:

    schedules, _ = schedules_and_stats
    schedules = _with_upcoming_week(schedules)
    second = schedules.iloc[[-1]].copy()
    second["game_id"] = "2022_06_D_C"
    second["home_team"] = "C"
    second["away_team"] = "D"
    schedules = pd.concat([schedules, second], ignore_index=True)

    with pytest.raises(DataContractError, match="PARTIAL") as error:
        apply_decision_lines(schedules, (_POOL_CAPTURE,))
    assert "2022_06_D_C" in str(error.value)


def test_pool_capture_refuses_a_line_for_a_game_that_is_not_scheduled(
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
) -> None:

    schedules, _ = schedules_and_stats
    schedules = _with_upcoming_week(schedules)
    misread = DecisionLineOverride(
        season=2022,
        week=6,
        lines={"2022_06_B_A": -1.5, "2022_06_D_C": 3.5},
        source="splashsports.com",
        capture_id="2022_week06_20221011_noon",
    )

    with pytest.raises(DataContractError, match="not on the 2022 week 6 schedule"):
        apply_decision_lines(schedules, (misread,))
