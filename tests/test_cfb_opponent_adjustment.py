"""Leakage and contract tests for the CFB opponent-adjustment substitution.

The leakage regression is release-blocking: a leaked opponent adjustment looks
spectacular and is worthless, and this feature family exists only if the
adjusted columns provably cannot see the week they are scoring.
"""

from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS, build_cfb_team_game_metrics
from nfl_ats.cfb_opponent_adjustment import (
    CFB_OPPONENT_ADJUSTED_FEATURE_COLUMNS,
    CFB_OPPONENT_ADJUSTED_MODEL_FEATURE_COLUMNS,
    CFB_OPPONENT_SOURCE_METRIC,
    CFB_TIME_DECAYED_FEATURE_COLUMNS,
    CFB_TIME_DECAYED_MODEL_FEATURE_COLUMNS,
    OPPONENT_BENCHMARK_BASELINE_METHOD,
    OPPONENT_BENCHMARK_CANDIDATE_METHOD,
    add_cfb_opponent_adjusted_features,
    build_cfb_opponent_history,
    cfb_opponent_adjustment_benchmark,
    opponent_adjusted_substitution,
    paired_margin_error_comparison,
    substitute_opponent_adjusted_columns,
)
from nfl_ats.data import DataContractError
from nfl_ats.opponent_adjustment import fit_opponent_effects

# The fixture league has eight teams, so a realistic 64-team-game warm-up would
# never fit. Everything else is the frozen configuration.
FIXTURE_MIN_TEAM_GAMES = 8


@pytest.fixture
def cfb_team_games(cfb_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    _, _, pbp = cfb_inputs
    team_games, _ = build_cfb_team_game_metrics(pbp)
    return team_games


def _adjusted(games: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    frame = add_cfb_opponent_adjusted_features(
        games, team_games, min_team_games=FIXTURE_MIN_TEAM_GAMES
    )
    return frame.sort_values("game_id").reset_index(drop=True)


def _perturb(team_games: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
    changed = team_games.copy()
    changed.loc[mask, CFB_OPPONENT_SOURCE_METRIC] = (
        changed.loc[mask, CFB_OPPONENT_SOURCE_METRIC] + 5.0
    )
    changed.loc[mask, "def_epa_per_play"] = changed.loc[mask, "def_epa_per_play"] + 5.0
    return changed


def test_adjusted_columns_cannot_see_their_own_week_or_the_future(
    cfb_features_frame: pd.DataFrame, cfb_team_games: pd.DataFrame
) -> None:
    baseline = _adjusted(cfb_features_frame, cfb_team_games)
    columns = ["game_id", *CFB_OPPONENT_ADJUSTED_FEATURE_COLUMNS]
    assert baseline[list(CFB_OPPONENT_ADJUSTED_FEATURE_COLUMNS)].notna().any().all()

    # Rewrite the current week and every later week beyond recognition.
    boundary_season, boundary_week = 2014, 8
    future = (cfb_team_games["season"].gt(boundary_season)) | (
        cfb_team_games["season"].eq(boundary_season) & cfb_team_games["week"].ge(boundary_week)
    )
    rescored = _adjusted(cfb_features_frame, _perturb(cfb_team_games, future))

    unchanged = (baseline["season"].lt(boundary_season)) | (
        baseline["season"].eq(boundary_season) & baseline["week"].le(boundary_week)
    )
    pd.testing.assert_frame_equal(
        baseline.loc[unchanged, columns].reset_index(drop=True),
        rescored.loc[unchanged, columns].reset_index(drop=True),
    )
    # The test is not vacuous: later weeks must move when their history moves.
    later = ~unchanged
    assert later.any()
    assert not baseline.loc[later, columns].equals(rescored.loc[later, columns])


def test_adjusted_columns_depend_on_strictly_earlier_history(
    cfb_features_frame: pd.DataFrame, cfb_team_games: pd.DataFrame
) -> None:
    baseline = _adjusted(cfb_features_frame, cfb_team_games)
    early = cfb_team_games["season"].eq(2014) & cfb_team_games["week"].eq(3)
    rescored = _adjusted(cfb_features_frame, _perturb(cfb_team_games, early))

    columns = list(CFB_OPPONENT_ADJUSTED_FEATURE_COLUMNS)
    before = baseline["season"].eq(2014) & baseline["week"].le(3)
    after = baseline["season"].eq(2014) & baseline["week"].ge(4)
    pd.testing.assert_frame_equal(
        baseline.loc[before, columns].reset_index(drop=True),
        rescored.loc[before, columns].reset_index(drop=True),
    )
    assert not baseline.loc[after, columns].equals(rescored.loc[after, columns])


def test_adjusted_columns_are_invariant_to_input_order(
    cfb_features_frame: pd.DataFrame, cfb_team_games: pd.DataFrame
) -> None:
    expected = _adjusted(cfb_features_frame, cfb_team_games)
    shuffled = _adjusted(
        cfb_features_frame.sample(frac=1.0, random_state=7),
        cfb_team_games.sample(frac=1.0, random_state=11),
    )
    columns = ["game_id", *CFB_OPPONENT_ADJUSTED_FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(expected[columns], shuffled[columns])


def test_one_weekly_fit_serves_offense_and_defense(
    cfb_features_frame: pd.DataFrame, cfb_team_games: pd.DataFrame
) -> None:
    """The defensive decomposition is the offensive fit with blocks swapped."""

    history = build_cfb_opponent_history(cfb_team_games, cfb_features_frame)
    cutoff = pd.Timestamp("2014-10-01")
    eligible = history.loc[history["gameday"].lt(cutoff)]
    teams = tuple(sorted(set(eligible["team"]) | set(eligible["opponent"])))
    shared = {
        "teams": teams,
        "cutoff": cutoff,
        "half_life_weeks": 16.0,
        "ridge_alpha": 10.0,
        "min_team_games": FIXTURE_MIN_TEAM_GAMES,
    }
    offensive_fit = fit_opponent_effects(eligible, metric=CFB_OPPONENT_SOURCE_METRIC, **shared)
    defensive_fit = fit_opponent_effects(eligible, metric="def_epa_per_play", **shared)
    assert offensive_fit is not None
    assert defensive_fit is not None
    assert defensive_fit.intercept == pytest.approx(offensive_fit.intercept)
    for team in teams:
        assert defensive_fit.offense[team] == pytest.approx(offensive_fit.defense[team])
        assert defensive_fit.defense[team] == pytest.approx(offensive_fit.offense[team])


def test_time_decay_control_drops_only_the_opponent_block(
    cfb_features_frame: pd.DataFrame, cfb_team_games: pd.DataFrame
) -> None:
    """The post-hoc control keeps the decay and the width, loses the opponent."""

    control = add_cfb_opponent_adjusted_features(
        cfb_features_frame,
        cfb_team_games,
        min_team_games=FIXTURE_MIN_TEAM_GAMES,
        include_opponent=False,
    )
    assert set(CFB_TIME_DECAYED_FEATURE_COLUMNS).issubset(control.columns)
    assert not set(CFB_OPPONENT_ADJUSTED_FEATURE_COLUMNS).intersection(control.columns)
    assert control[list(CFB_TIME_DECAYED_FEATURE_COLUMNS)].notna().any().all()
    assert len(CFB_TIME_DECAYED_MODEL_FEATURE_COLUMNS) == len(CFB_MODEL_FEATURE_COLUMNS)

    history = build_cfb_opponent_history(cfb_team_games, cfb_features_frame)
    cutoff = pd.Timestamp("2014-10-01")
    eligible = history.loc[history["gameday"].lt(cutoff)]
    teams = tuple(sorted(set(eligible["team"]) | set(eligible["opponent"])))
    team_only = fit_opponent_effects(
        eligible,
        metric=CFB_OPPONENT_SOURCE_METRIC,
        teams=teams,
        cutoff=cutoff,
        half_life_weeks=16.0,
        ridge_alpha=10.0,
        min_team_games=FIXTURE_MIN_TEAM_GAMES,
        include_opponent=False,
    )
    assert team_only is not None
    assert set(team_only.defense.values()) == {0.0}
    assert any(value != 0.0 for value in team_only.offense.values())


def test_substitution_is_dimension_neutral() -> None:
    substituted = substitute_opponent_adjusted_columns()
    mapping = opponent_adjusted_substitution()

    assert substituted == CFB_OPPONENT_ADJUSTED_MODEL_FEATURE_COLUMNS
    assert len(substituted) == len(CFB_MODEL_FEATURE_COLUMNS)
    assert set(CFB_MODEL_FEATURE_COLUMNS).difference(substituted) == set(mapping)
    assert set(substituted).difference(CFB_MODEL_FEATURE_COLUMNS) == set(
        CFB_OPPONENT_ADJUSTED_FEATURE_COLUMNS
    )
    for position, column in enumerate(CFB_MODEL_FEATURE_COLUMNS):
        assert substituted[position] == mapping.get(column, column)

    with pytest.raises(DataContractError, match="missing substitutable"):
        substitute_opponent_adjusted_columns(("spread_line", "total_line"))


@pytest.mark.full  # ENG-11: dominates --durations; full CFB benchmark fit
def test_benchmark_scores_both_arms_on_identical_games(
    cfb_features_frame: pd.DataFrame, cfb_team_games: pd.DataFrame
) -> None:
    features = add_cfb_opponent_adjusted_features(
        cfb_features_frame, cfb_team_games, min_team_games=FIXTURE_MIN_TEAM_GAMES
    )
    result = cfb_opponent_adjustment_benchmark(
        features,
        start_season=2014,
        end_season=2014,
        min_train_games=50,
        bootstrap_samples=50,
    )
    methods = set(result.predictions["method"])
    assert methods == {
        "market",
        OPPONENT_BENCHMARK_BASELINE_METHOD,
        OPPONENT_BENCHMARK_CANDIDATE_METHOD,
    }
    per_arm = {
        method: set(group["game_id"])
        for method, group in result.predictions.groupby("method", sort=False)
    }
    assert (
        per_arm[OPPONENT_BENCHMARK_BASELINE_METHOD] == per_arm[OPPONENT_BENCHMARK_CANDIDATE_METHOD]
    )
    assert (
        result.diagnostics["baseline_input_columns"]
        == result.diagnostics["candidate_input_columns"]
        == len(CFB_MODEL_FEATURE_COLUMNS)
    )
    assert set(result.paired_margin["metric"]) == {
        "margin_mae_improvement",
        "margin_rmse_improvement",
    }
    assert set(result.paired_margin["block"]) == {"week", "season"}
    assert result.paired_margin["probability_positive"].between(0.0, 1.0).all()
    assert (result.paired_margin["lower"] <= result.paired_margin["estimate"]).all()
    assert (result.paired_margin["estimate"] <= result.paired_margin["upper"]).all()
    assert set(result.paired_probability["metric"]) == {
        "accuracy_improvement",
        "brier_improvement",
        "log_loss_improvement",
    }


@pytest.mark.full  # ENG-11: dominates --durations; full CFB benchmark fit
def test_paired_margin_comparison_contracts(
    cfb_features_frame: pd.DataFrame, cfb_team_games: pd.DataFrame
) -> None:
    features = add_cfb_opponent_adjusted_features(
        cfb_features_frame, cfb_team_games, min_team_games=FIXTURE_MIN_TEAM_GAMES
    )
    result = cfb_opponent_adjustment_benchmark(
        features,
        start_season=2014,
        end_season=2014,
        min_train_games=50,
        bootstrap_samples=50,
    )
    predictions = result.predictions
    with pytest.raises(ValueError, match="No predictions for method"):
        paired_margin_error_comparison(
            predictions,
            baseline_method="missing",
            candidate_method=OPPONENT_BENCHMARK_CANDIDATE_METHOD,
            samples=50,
        )
    with pytest.raises(ValueError, match="samples"):
        paired_margin_error_comparison(
            predictions,
            baseline_method=OPPONENT_BENCHMARK_BASELINE_METHOD,
            candidate_method=OPPONENT_BENCHMARK_CANDIDATE_METHOD,
            samples=2,
        )
    identical = paired_margin_error_comparison(
        predictions,
        baseline_method=OPPONENT_BENCHMARK_BASELINE_METHOD,
        candidate_method=OPPONENT_BENCHMARK_BASELINE_METHOD,
        samples=50,
    )
    assert identical["estimate"].abs().max() == pytest.approx(0.0)
    assert identical["probability_positive"].eq(0.0).all()


def test_adjusted_feature_contracts(
    cfb_features_frame: pd.DataFrame, cfb_team_games: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="half_life_weeks"):
        add_cfb_opponent_adjusted_features(cfb_features_frame, cfb_team_games, half_life_weeks=0.0)
    with pytest.raises(ValueError, match="ridge_alpha"):
        add_cfb_opponent_adjusted_features(cfb_features_frame, cfb_team_games, ridge_alpha=0.0)
    with pytest.raises(ValueError, match="min_team_games"):
        add_cfb_opponent_adjusted_features(cfb_features_frame, cfb_team_games, min_team_games=1)
    with pytest.raises(DataContractError, match="missing required columns"):
        add_cfb_opponent_adjusted_features(
            cfb_features_frame.drop(columns=["home_id"]), cfb_team_games
        )
    duplicated = pd.concat([cfb_team_games, cfb_team_games.head(1)], ignore_index=True)
    with pytest.raises(DataContractError, match="duplicate"):
        add_cfb_opponent_adjusted_features(
            cfb_features_frame, duplicated, min_team_games=FIXTURE_MIN_TEAM_GAMES
        )
