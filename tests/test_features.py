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
    SURFACE_SWITCH_FEATURE_COLUMNS,
)
from nfl_ats.data import DataContractError
from nfl_ats.features import (
    add_ats_outcomes,
    add_bias_features,
    add_surface_switch_features,
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
    # weak_stack_surface (MOD-08) is declared as weak_stack + the
    # surface-switch family, so it inherits the bias columns too -- not a
    # second independent consumer, the same one extended. weak_stack_js_prior
    # (MOD-06, docs/mod06_position_prior_shrinkage.md) is weak_stack with the
    # player_values family swapped for player_values_js_prior -- same reason.
    # weak_stack_v3 (docs/weak_stack_v3.md) is weak_stack_surface plus the new
    # gap_v3_* families -- same inheritance, not a fourth independent consumer.
    # weak_stack_v4 (docs/weak_stack_v4.md) is weak_stack plus the
    # forecast_weather family -- same inheritance again, for the same reason.
    inherited_suffixes = {
        "weak_stack",
        "weak_stack_surface",
        "weak_stack_js_prior",
        "weak_stack_v3",
        "weak_stack_v4",
        # weak_stack_oracle_weather (docs/weak_stack_v4.md, "wind oracle") is
        # weak_stack plus OBSERVED weather -- a POSITIVE CONTROL that is
        # deliberately leaky and never promotable. Same inheritance again.
        "weak_stack_oracle_weather",
        # weak_stack_graph_sack (docs/graph_team_stat_on_production.md) is
        # weak_stack plus the one graph-propagated off_sack_rate column --
        # same inheritance again, not a fifth independent consumer.
        "weak_stack_graph_sack",
        # weak_stack_graph_def_ypp (docs/graph_team_stat_def_ypp_on_production.md)
        # is weak_stack plus the one graph-propagated def_yards_per_play column --
        # same inheritance again, not a sixth independent consumer.
        "weak_stack_graph_def_ypp",
        # weak_stack_graph_off_rush_epa
        # (docs/graph_team_stat_off_rush_epa_on_production.md) is weak_stack plus
        # the one graph-propagated off_rush_epa_per_play column -- same
        # inheritance again, not a seventh independent consumer.
        "weak_stack_graph_off_rush_epa",
        # weak_stack_fluview_home/_away (docs/fluview_on_production.md) are
        # weak_stack plus exactly one FluView elevated-illness column each --
        # same inheritance again, not a sixth/seventh independent consumer.
        "weak_stack_fluview_home",
        "weak_stack_fluview_away",
        # 2026-09-01/02 on-production confirmation profiles all extend the
        # same declared weak stack with exactly one measured candidate column.
        "weak_stack_durability",
        "weak_stack_illness_home",
        "weak_stack_illness_away",
        "weak_stack_reddit_ratio_home",
        "weak_stack_reddit_spike_away",
        "weak_stack_redzone_third_down",
        "weak_stack_source_availability",
        "weak_stack_team_style_pace",
        # 2026-09-05 overnight lead batteries (docs/schedule_flag_battery.md,
        # docs/market_lead_battery.md): each is weak_stack plus exactly one
        # deterministic pregame flag column, the same inheritance again.
        "weak_stack_post_ot",
        "weak_stack_mnf_road",
        "weak_stack_home_thursday",
        "weak_stack_opener_softness",
        "weak_stack_ml_divergence",
        # Wave 2 venue/market-context leads (docs/schedule_flag_battery.md
        # "Wave 2"): each is weak_stack plus exactly one more deterministic
        # pregame flag column, the same inheritance again.
        "weak_stack_new_stadium",
        "weak_stack_dome_shootout",
        "weak_stack_low_total_div_dog",
        "weak_stack_sept_heat",
        # Wave 3 public-claim leads on production (docs/schedule_flag_battery.md
        # "Wave 3", LEAD-57 leads): each is weak_stack plus exactly one more
        # deterministic pregame flag column, the same inheritance again.
        "weak_stack_road_fav_fade",
        "weak_stack_division_dog",
        "weak_stack_week1_dog",
        "weak_stack_ats_streak_regress",
        # Wave 4 PBP coaching-trait leads on production
        # (docs/schedule_flag_battery.md "Wave 4", LEAD-26/27/30): each is
        # weak_stack plus exactly one more deterministic pregame trait
        # column, the same inheritance again.
        "weak_stack_opening_drive_epa",
        "weak_stack_q3_point_diff",
        "weak_stack_fourth_down_interaction",
        # Wave 5 quarterback-identity leads on production
        # (docs/schedule_flag_battery.md "Wave 5", LEAD-20/LEAD-25): each is
        # weak_stack plus exactly one more deterministic pregame flag column,
        # the same inheritance again.
        "weak_stack_rookie_qb_debut_fade",
        "weak_stack_qb_revenge",
        # 2026-09-05 transaction-wire lead battery (docs/schedule_flag_battery.md
        # "Wave 6", LEAD-12/LEAD-23/LEAD-14): each is weak_stack plus exactly one
        # more deterministic pregame flag column, the same inheritance again.
        "weak_stack_holdout_slow_start",
        "weak_stack_deadline_drag",
        "weak_stack_suspension_rust",
        # 2026-09-05 officiating-crew lead battery (docs/officials_crew_leads.md,
        # LEAD-34/LEAD-31): each is weak_stack plus exactly one more
        # deterministic pregame flag column, the same inheritance again.
        "weak_stack_crew_second_meeting_favorite",
        "weak_stack_rookie_crew_underdog",
        # 2026-09-05 weather/venue lead battery (docs/weather_venue_leads.md,
        # ROADMAP LEAD-36/LEAD-37): each is weak_stack plus exactly one more
        # deterministic pregame flag column, the same inheritance again.
        "weak_stack_open_corner_wind_dog",
        "weak_stack_rain_on_grass_dog",
        # Lane T promotion evaluation (docs/promotion_eval_20260905.md) is
        # weak_stack plus BOTH the qb_revenge and deadline_drag columns at
        # once (composition test) -- same inheritance again, not a new
        # independent consumer.
        "weak_stack_qb_revenge_deadline_drag",
        # Wave 7 roster-availability lead battery
        # (docs/schedule_flag_battery.md "Wave 7", LEAD-13/LEAD-17): each is
        # weak_stack plus exactly one more deterministic pregame flag column,
        # the same inheritance again.
        "weak_stack_ir_return_reinforcement",
        "weak_stack_specialist_absence_fade",
    }
    assert admitting == {
        f"{scope}_{suffix}" for scope in ("football", "full") for suffix in inherited_suffixes
    }
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


# ---------------------------------------------------------------------------
# Surface-switch tilt candidate feature (docs/surface_switch_feature_arm.md,
# MOD-08). Mirrors tests/test_surface_switch_tilt_overlay.py's own fixture
# and leakage-test shapes exactly, since add_surface_switch_features is a
# verbatim port of that module's surface_switch_flag_by_game construct.
# ---------------------------------------------------------------------------


def _surface_switch_schedule() -> pd.DataFrame:
    """GRASSAWAY hosts two 2026 games on grass -- its 2026 modal home surface
    is grass. It plays three road games: at TURFHOST (fieldturf, flagged --
    grass-modal visitor on turf), at GRASSHOST (grass, not flagged -- no
    switch), and a POST-season game at POSTHOST (fieldturf, same flagged
    shape as the week-3 game -- excluded by the REG-only gate). TURFAWAY
    hosts two 2026 games on turf -- modal turf; its road game at TURFHOST2
    (turf) is not flagged (surfaces match, no switch). NOSURF hosts one game
    with an unresolved (empty-string) surface -- modal None; its road game
    at TURFHOST3 (turf) is not flagged (visitor's own modal surface is
    unresolved)."""

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
    # weak_stack_surface (MOD-08) is the ONE direct consumer, mirroring
    # BIAS_FEATURE_COLUMNS' own exception-pinning test above; weak_stack_v3
    # (docs/weak_stack_v3.md) inherits it by being declared as
    # weak_stack_surface plus the new gap_v3_* families, not a second
    # independent consumer.
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
    # Same visitor, surfaces match (grass at grass) -- no switch.
    assert _surface_row(flagged, "2026_04_GRASSHOST_GRASSAWAY")["surface_switch_flag"] == 0.0
    # Turf-modal visitor playing at another turf venue -- no switch.
    assert _surface_row(flagged, "2026_03_TURFHOST2_TURFAWAY")["surface_switch_flag"] == 0.0
    # Visitor's own modal home surface is unresolved -- no signal.
    assert _surface_row(flagged, "2026_05_TURFHOST3_NOSURF")["surface_switch_flag"] == 0.0
    # POST-season game with the same flagged shape as week 3 -- excluded by
    # the REG-only gate.
    assert _surface_row(flagged, "2026_20_POSTHOST_GRASSAWAY")["surface_switch_flag"] == 0.0


def test_surface_switch_flag_is_missing_surface_column_safe() -> None:
    """Older/synthetic schedules without a ``surface`` column at all (the
    ``schedules_and_stats`` fixture, e.g.) must degrade to 0.0, not raise --
    this is a schedule-shaped enrichment, not a hard data contract."""

    schedule = _surface_switch_schedule().drop(columns=["surface"])
    flagged = add_surface_switch_features(schedule, schedule)
    assert flagged["surface_switch_flag"].eq(0.0).all()


def test_surface_switch_flag_never_reads_outcome_columns() -> None:
    """AGENTS.md: a leakage regression test for every new feature family.

    ``add_surface_switch_features`` does not even require/read
    ``result``/``spread_line`` -- adding them (with arbitrary values) and
    mutating them must never change the already-computed flags, proving the
    derivation is purely structural (surface/team/season), never outcome-
    based.
    """

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
    """A future season's surface data (even for the SAME team) must never
    change an earlier season's already-computed flags -- mirrors
    tests/test_surface_switch_tilt_overlay.py's identical test for
    surface_switch_flag_by_game."""

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
    """Wiring + additivity, mirroring the bias family's own equivalent test
    above: the column is present after a real ``build_game_features`` run
    (through the missing-column fallback path, since this fixture carries no
    ``surface`` column) and every pre-existing column stays bit-identical
    whether or not the family runs."""

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
