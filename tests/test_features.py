from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.constants import (
    BIAS_FEATURE_COLUMNS,
    FEATURE_FAMILIES,
    FEATURE_SETS,
    GRAPH_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    OUTCOME_COLUMNS,
    STATE_METRICS,
)
from nfl_ats.data import DataContractError
from nfl_ats.features import (
    add_ats_outcomes,
    add_bias_features,
    attach_team_states,
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


def test_historical_franchise_abbreviations_share_current_team_state(
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    schedules, stats = schedules_and_stats
    historical = schedules.copy()
    historical.loc[:, "home_team"] = "OAK"
    historical.loc[:, "away_team"] = "STL"
    current_stats = stats.copy()
    current_stats.loc[current_stats["team"].eq("A"), "team"] = "LV"
    current_stats.loc[current_stats["team"].eq("B"), "team"] = "LA"

    features = build_game_features(
        historical,
        current_stats,
        span=3,
        min_periods=1,
        graph_min_games=2,
    )
    assert features["home_team"].eq("LV").all()
    assert features["away_team"].eq("LA").all()
    assert features.loc[1:, "home_off_epa_per_play"].notna().all()
    assert features.loc[1:, "away_off_epa_per_play"].notna().all()


def test_offseason_state_regresses_and_season_game_count_resets() -> None:
    rows = []
    for team, state_value in (("A", 10.0), ("B", 0.0)):
        row: dict[str, object] = {
            "game_id": f"2022_18_{team}",
            "season": 2022,
            "gameday": pd.Timestamp("2023-01-08"),
            "team": team,
            "team_games": 17,
        }
        for metric in STATE_METRICS:
            row[f"state_{metric}"] = state_value
            row[f"league_mean_{metric}"] = 5.0
        rows.append(row)
    states = pd.DataFrame(rows)
    game = pd.DataFrame(
        {
            "game_id": ["2023_01_B_A"],
            "season": [2023],
            "gameday": [pd.Timestamp("2023-09-10")],
            "home_team": ["A"],
            "away_team": ["B"],
        }
    )
    attached = attach_team_states(game, states, offseason_retention=0.5)
    assert attached.loc[0, "home_team_games"] == 0
    assert attached.loc[0, "away_team_games"] == 0
    assert attached.loc[0, "home_off_epa_per_play"] == pytest.approx(7.5)
    assert attached.loc[0, "away_off_epa_per_play"] == pytest.approx(2.5)
    assert attached.loc[0, "diff_off_epa_per_play"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Opener-bias family (MOD-07)
# ---------------------------------------------------------------------------


def _bracket_schedules() -> pd.DataFrame:
    """Two seasons around a synthetic four-team postseason bracket.

    2022: A, B, C, D play two regular-season weeks; A, B and C reach the
    postseason (A beats C in the wild card, A beats B in the Super Bowl) and D
    does not. 2023 opens with A hosting D and B hosting C.
    """

    rows = [
        ("2022_01_B_A", 2022, "REG", 1, "2022-09-11", "B", "A", 3.0, 1.0),
        ("2022_01_D_C", 2022, "REG", 1, "2022-09-11", "D", "C", -7.0, -3.0),
        ("2022_02_C_A", 2022, "REG", 2, "2022-09-18", "C", "A", 10.0, 4.0),
        ("2022_02_D_B", 2022, "REG", 2, "2022-09-18", "D", "B", 1.0, 3.0),
        ("2022_19_C_A", 2022, "WC", 19, "2023-01-14", "C", "A", 5.0, 2.0),
        ("2022_22_B_A", 2022, "SB", 22, "2023-02-12", "B", "A", 3.0, 1.0),
        ("2023_01_D_A", 2023, "REG", 1, "2023-09-10", "D", "A", 6.0, 3.0),
        ("2023_01_C_B", 2023, "REG", 1, "2023-09-10", "C", "B", -1.0, 2.0),
        ("2023_02_B_A", 2023, "REG", 2, "2023-09-17", "B", "A", 4.0, 2.0),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "game_id",
            "season",
            "game_type",
            "week",
            "gameday",
            "away_team",
            "home_team",
            "result",
            "spread_line",
        ],
    )
    frame["gameday"] = pd.to_datetime(frame["gameday"])
    return frame


def _bias_row(features: pd.DataFrame, game_id: str) -> pd.Series:
    return features.loc[features["game_id"].eq(game_id)].iloc[0]


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
    # The MOD-07 candidate profile (SPEC-4 step 2) is the ONE declared consumer
    # of this family. Every other set -- above all the frozen ones the active
    # model is fitted on -- must stay clear of it, so this pins the exception
    # rather than merely permitting one.
    admitting = {
        name for name, columns in FEATURE_SETS.items() if set(columns) & set(BIAS_FEATURE_COLUMNS)
    }
    assert admitting == {"football_weak_stack", "full_weak_stack"}
    for name in ("full", "full_player", "full_player_value", "football", "football_player"):
        assert set(FEATURE_SETS[name]).isdisjoint(BIAS_FEATURE_COLUMNS), name


def test_playoff_holdover_flags_week_one_teams_from_the_previous_bracket() -> None:
    schedules = _bracket_schedules()
    bias = add_bias_features(schedules, schedules)

    opener = _bias_row(bias, "2023_01_D_A")
    assert opener["bias_playoff_holdover_home"] == 1.0  # A reached the Super Bowl
    assert opener["bias_playoff_holdover_away"] == 0.0  # D missed the bracket
    assert opener["bias_playoff_holdover_diff"] == 1.0

    both = _bias_row(bias, "2023_01_C_B")
    assert both["bias_playoff_holdover_home"] == 1.0
    assert both["bias_playoff_holdover_away"] == 1.0
    assert both["bias_playoff_holdover_diff"] == 0.0

    # Week 1 of the bracket season itself has no prior postseason in the frame,
    # and later weeks are never flagged even for holdover teams.
    assert _bias_row(bias, "2022_01_B_A")["bias_playoff_holdover_home"] == 0.0
    assert _bias_row(bias, "2023_02_B_A")["bias_playoff_holdover_home"] == 0.0


def test_prior_week_ats_uses_the_single_previous_game_team_signed() -> None:
    schedules = _bracket_schedules()
    bias = add_bias_features(schedules, schedules)

    opener = _bias_row(bias, "2023_01_D_A")
    assert pd.isna(opener["bias_prior_week_ats_home"])
    assert pd.isna(opener["bias_prior_week_ats_away"])
    assert opener["bias_week2_anchor_home"] == 0.0
    assert opener["bias_week2_anchor_away"] == 0.0

    # A won by 6 laying 3 in week 1; B lost by 1 while getting 2 as the host.
    week_two = _bias_row(bias, "2023_02_B_A")
    assert week_two["bias_prior_week_ats_home"] == pytest.approx(3.0)
    assert week_two["bias_prior_week_ats_away"] == pytest.approx(-3.0)
    assert week_two["bias_prior_week_ats_diff"] == pytest.approx(6.0)
    assert week_two["bias_week2_anchor_home"] == pytest.approx(3.0)
    assert week_two["bias_week2_anchor_away"] == pytest.approx(-3.0)
    assert week_two["bias_week2_anchor_diff"] == pytest.approx(6.0)

    # The season resets: the 2023 opener does not see the 2022 Super Bowl.
    assert pd.isna(_bias_row(bias, "2023_01_C_B")["bias_prior_week_ats_home"])

    # Away-team sign convention, and postseason rows read the earlier round.
    assert _bias_row(bias, "2022_02_C_A")["bias_prior_week_ats_away"] == pytest.approx(-4.0)
    assert _bias_row(bias, "2022_19_C_A")["bias_prior_week_ats_home"] == pytest.approx(6.0)
    assert _bias_row(bias, "2022_22_B_A")["bias_prior_week_ats_home"] == pytest.approx(3.0)


def test_bias_features_cannot_see_the_result_of_their_own_game(
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    schedules, stats = schedules_and_stats
    baseline = build_game_features(schedules, stats, span=3, min_periods=1)
    changed_schedules = schedules.copy()
    changed_schedules.loc[2, "result"] = 40.0  # week 3, originally +7
    changed = build_game_features(changed_schedules, stats, span=3, min_periods=1)

    assert set(BIAS_FEATURE_COLUMNS).issubset(baseline.columns)
    for column in ("bias_prior_week_ats_home", "bias_prior_week_ats_away"):
        # Week 3's own result stays invisible to week 3.
        assert changed.loc[2, column] == pytest.approx(baseline.loc[2, column])
        # It reaches week 4, which is what makes the lookup non-trivial.
        assert changed.loc[3, column] != pytest.approx(baseline.loc[3, column])
    assert pd.isna(baseline.loc[0, "bias_prior_week_ats_home"])
    assert baseline["bias_playoff_holdover_home"].eq(0.0).all()


def test_bias_family_leaves_every_pre_existing_column_bit_identical(
    schedules_and_stats: tuple[pd.DataFrame, pd.DataFrame],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Additivity: the frozen columns must not move when the family is added."""

    from nfl_ats import features as features_module

    schedules, stats = schedules_and_stats
    with_bias = build_game_features(schedules, stats, span=3, min_periods=1)

    def _stub(games: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
        return games.assign(**dict.fromkeys(BIAS_FEATURE_COLUMNS, 0.0))

    monkeypatch.setattr(features_module, "add_bias_features", _stub)
    without_bias = build_game_features(schedules, stats, span=3, min_periods=1)

    pre_existing = [column for column in with_bias.columns if column not in BIAS_FEATURE_COLUMNS]
    assert list(without_bias.columns) == list(with_bias.columns)
    pd.testing.assert_frame_equal(
        with_bias[pre_existing],
        without_bias[pre_existing],
        check_exact=True,
    )
