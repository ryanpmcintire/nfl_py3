"""Correctness, degenerate-limit, and leakage tests for cross-league transfer.

The leakage regression is release-blocking (AGENTS.md): a walk-forward that
can see its own week or the future looks spectacular and is worthless. The
degenerate-limit tests pin the prior-mean and hierarchical arms to closed-form
answers that do not depend on any real CFB/NFL data being present, so this
suite never touches ``data/processed`` and runs entirely on synthetic frames.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS
from nfl_ats.constants import FEATURE_SETS
from nfl_ats.cross_league_transfer import (
    ALIGNED_TRANSFER_FEATURE_COLUMNS,
    CROSS_LEAGUE_RIDGE_ALPHA,
    _augmented_design,
    _fit_theta,
    cross_league_transfer_benchmark,
    derive_shrinkage_weights,
    fit_hierarchical_shrinkage_model,
    fit_joint_league_model,
    fit_pooled_preprocessor,
    fit_prior_mean_ridge_model,
    fit_target_only_model,
    measure_league_mismatch,
)
from nfl_ats.data import DataContractError
from nfl_ats.margin import MarginModel


def _coefficients(model: MarginModel) -> np.ndarray:
    """Pull the fitted coefficient vector out of a transfer arm's ``MarginModel``.

    A thin, explicitly-typed accessor so every test below does not need its
    own ``# type: ignore`` for reaching into the private ``_FixedLinearRegressor``.
    """

    assert model.estimator is not None
    return model.estimator.coefficients  # type: ignore[attr-defined,no-any-return]


def _make_league_frame(
    *,
    league: str,
    seasons: range,
    weeks_per_season: int,
    games_per_week: int,
    seed: int,
    coefficient: float = 2.0,
    noise_scale: float = 10.0,
) -> pd.DataFrame:
    """A small synthetic league with the aligned transfer contract populated.

    ``ats_margin`` is generated from a known linear signal in the EPA-diff
    columns plus noise, so tests can check fitted coefficients point the
    right direction without depending on any real data file.
    """

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    base = pd.Timestamp("2012-08-01")
    for season in seasons:
        for week in range(1, weeks_per_season + 1):
            gameday = base + pd.Timedelta(days=(season - min(seasons)) * 360 + week * 7)
            for game_index in range(games_per_week):
                off_home = rng.normal(0.0, 0.12)
                off_away = rng.normal(0.0, 0.12)
                def_home = rng.normal(0.0, 0.12)
                def_away = rng.normal(0.0, 0.12)
                diff_off = off_home - off_away
                diff_def = def_home - def_away
                spread_line = float(rng.normal(0.0, 7.0))
                total_line = float(rng.normal(45.0, 4.0))
                rest_diff = int(rng.integers(-3, 4))
                neutral_site = int(rng.integers(0, 2))
                week_angle = 2.0 * math.pi * week / max(weeks_per_season, 1)
                home_games = int(rng.integers(1, 60))
                away_games = int(rng.integers(1, 60))
                signal = coefficient * diff_off - coefficient * diff_def
                noise = float(rng.normal(0.0, noise_scale))
                ats_margin = signal + noise
                result = spread_line + ats_margin
                rows.append(
                    {
                        "game_id": f"{league}_{season}_{week}_{game_index}",
                        "season": season,
                        "week": week,
                        "gameday": gameday,
                        "spread_line": spread_line,
                        "total_line": total_line,
                        "rest_diff": rest_diff,
                        "neutral_site": neutral_site,
                        "week_sin": math.sin(week_angle),
                        "week_cos": math.cos(week_angle),
                        "home_team_games": home_games,
                        "away_team_games": away_games,
                        "home_off_epa_per_play": off_home,
                        "away_off_epa_per_play": off_away,
                        "diff_off_epa_per_play": diff_off,
                        "home_def_epa_per_play": def_home,
                        "away_def_epa_per_play": def_away,
                        "diff_def_epa_per_play": diff_def,
                        "home_spread_odds": -110.0,
                        "away_spread_odds": -110.0,
                        "result": result,
                        "ats_margin": ats_margin,
                        "home_cover": float(ats_margin > 0.0),
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def target_frame() -> pd.DataFrame:
    return _make_league_frame(
        league="target", seasons=range(2012, 2020), weeks_per_season=10, games_per_week=4, seed=7
    )


@pytest.fixture
def auxiliary_frame() -> pd.DataFrame:
    return _make_league_frame(
        league="aux",
        seasons=range(2012, 2020),
        weeks_per_season=10,
        games_per_week=10,
        seed=11,
        coefficient=2.4,
    )


# ---------------------------------------------------------------------------
# The aligned contract is real, not a private assumption
# ---------------------------------------------------------------------------


def test_aligned_columns_are_a_true_subset_of_both_leagues_contracts() -> None:
    cfb_columns = set(CFB_MODEL_FEATURE_COLUMNS)
    nfl_columns = set(FEATURE_SETS["full"])
    missing_from_cfb = set(ALIGNED_TRANSFER_FEATURE_COLUMNS) - cfb_columns
    missing_from_nfl = set(ALIGNED_TRANSFER_FEATURE_COLUMNS) - nfl_columns
    assert not missing_from_cfb, missing_from_cfb
    assert not missing_from_nfl, missing_from_nfl


# ---------------------------------------------------------------------------
# Prior-mean ridge: closed-form identity and degenerate limits
# ---------------------------------------------------------------------------


def test_prior_mean_closed_form_matches_direct_linear_algebra() -> None:
    rng = np.random.default_rng(3)
    design = rng.normal(size=(60, 5))
    target = design @ np.array([1.0, -2.0, 0.5, 0.0, 3.0]) + rng.normal(scale=0.5, size=60)
    theta0 = np.array([0.2, -0.1, 0.3, 0.0, -0.4])
    alpha = 7.0

    delta = _fit_theta(design, target - design @ theta0, alpha)
    closed_form = theta0 + delta

    # Direct minimizer of ||y - X theta||^2 + alpha ||theta - theta0||^2:
    # theta = (X'X + alpha I)^-1 (X'y + alpha theta0).
    gram = design.T @ design + alpha * np.eye(design.shape[1])
    rhs = design.T @ target + alpha * theta0
    manual = np.linalg.solve(gram, rhs)
    np.testing.assert_allclose(closed_form, manual, rtol=1e-8, atol=1e-8)


def test_prior_mean_ridge_converges_to_auxiliary_as_alpha_grows(
    target_frame: pd.DataFrame, auxiliary_frame: pd.DataFrame
) -> None:
    imputer, scaler = fit_pooled_preprocessor(target_frame, auxiliary_frame)
    model = fit_prior_mean_ridge_model(
        target_frame, auxiliary_frame, imputer, scaler, ridge_alpha=1e8
    )
    aux_only_design = _augmented_design(
        auxiliary_frame, ALIGNED_TRANSFER_FEATURE_COLUMNS, imputer, scaler
    )
    theta_aux = _fit_theta(
        aux_only_design, auxiliary_frame["ats_margin"].to_numpy(dtype=float), 1e8
    )
    np.testing.assert_allclose(_coefficients(model), theta_aux, rtol=1e-4, atol=1e-4)


def test_prior_mean_ridge_converges_to_target_only_as_alpha_shrinks(
    target_frame: pd.DataFrame, auxiliary_frame: pd.DataFrame
) -> None:
    imputer, scaler = fit_pooled_preprocessor(target_frame, auxiliary_frame)
    tiny_alpha = 1e-6
    prior = fit_prior_mean_ridge_model(
        target_frame, auxiliary_frame, imputer, scaler, ridge_alpha=tiny_alpha
    )
    target_only = fit_target_only_model(target_frame, imputer, scaler, ridge_alpha=tiny_alpha)
    np.testing.assert_allclose(
        _coefficients(prior),
        _coefficients(target_only),
        rtol=1e-3,
        atol=1e-3,
    )


# ---------------------------------------------------------------------------
# Hierarchical shrinkage: derived weights are bounded and blend correctly
# ---------------------------------------------------------------------------


def test_shrinkage_weights_are_bounded_and_derived(
    target_frame: pd.DataFrame, auxiliary_frame: pd.DataFrame
) -> None:
    imputer, scaler = fit_pooled_preprocessor(target_frame, auxiliary_frame)
    derivation = derive_shrinkage_weights(
        target_frame, auxiliary_frame, imputer, scaler, samples=40, seed=1
    )
    assert derivation.weights.shape == (len(ALIGNED_TRANSFER_FEATURE_COLUMNS) + 1,)
    assert np.all(derivation.weights >= 0.0)
    assert np.all(derivation.weights <= 1.0)
    assert derivation.tau_squared >= 0.0
    # Recompute the DerSimonian-Laird tau^2 independently from the reported
    # pieces to pin the formula itself, not just its output shape.
    diff_sq = np.square(derivation.theta_target - derivation.theta_aux)
    q_stat = float(np.sum(diff_sq / derivation.target_variance))
    k = len(derivation.theta_target)
    expected_tau_sq = max(0.0, (q_stat - (k - 1)) / float(np.sum(1.0 / derivation.target_variance)))
    assert derivation.tau_squared == pytest.approx(expected_tau_sq, rel=1e-9)
    expected_weights = expected_tau_sq / (expected_tau_sq + derivation.target_variance)
    np.testing.assert_allclose(derivation.weights, expected_weights, rtol=1e-9)


def test_hierarchical_blend_lies_between_the_two_anchors(
    target_frame: pd.DataFrame, auxiliary_frame: pd.DataFrame
) -> None:
    imputer, scaler = fit_pooled_preprocessor(target_frame, auxiliary_frame)
    derivation = derive_shrinkage_weights(
        target_frame, auxiliary_frame, imputer, scaler, samples=40, seed=2
    )
    hierarchical = fit_hierarchical_shrinkage_model(
        target_frame,
        auxiliary_frame,
        imputer,
        scaler,
        ALIGNED_TRANSFER_FEATURE_COLUMNS,
        CROSS_LEAGUE_RIDGE_ALPHA,
        derivation.weights,
    )
    target_only = fit_target_only_model(target_frame, imputer, scaler)
    lower = np.minimum(derivation.theta_target, derivation.theta_aux)
    upper = np.maximum(derivation.theta_target, derivation.theta_aux)
    blended = _coefficients(hierarchical)
    assert np.all(blended >= lower - 1e-9)
    assert np.all(blended <= upper + 1e-9)
    # A weight of exactly 1 everywhere would make hierarchical == target_only;
    # since the two leagues' generating coefficients differ here, it should not.
    assert not np.allclose(blended, _coefficients(target_only))


# ---------------------------------------------------------------------------
# Joint fitting and the mismatch report
# ---------------------------------------------------------------------------


def test_joint_model_runs_and_differs_from_target_only(
    target_frame: pd.DataFrame, auxiliary_frame: pd.DataFrame
) -> None:
    imputer, scaler = fit_pooled_preprocessor(target_frame, auxiliary_frame)
    joint = fit_joint_league_model(target_frame, auxiliary_frame, imputer, scaler)
    target_only = fit_target_only_model(target_frame, imputer, scaler)
    joint_predictions = joint.predict(target_frame.head(20))
    solo_predictions = target_only.predict(target_frame.head(20))
    assert joint_predictions["predicted_margin"].notna().all()
    assert not np.allclose(
        joint_predictions["predicted_margin"].to_numpy(),
        solo_predictions["predicted_margin"].to_numpy(),
    )


def test_joint_model_matches_plain_ridge_on_pooled_indicator_design(
    target_frame: pd.DataFrame, auxiliary_frame: pd.DataFrame
) -> None:
    imputer, scaler = fit_pooled_preprocessor(target_frame, auxiliary_frame)
    joint = fit_joint_league_model(target_frame, auxiliary_frame, imputer, scaler)

    aux_design = _augmented_design(
        auxiliary_frame, ALIGNED_TRANSFER_FEATURE_COLUMNS, imputer, scaler, indicator_value=0.0
    )
    target_design = _augmented_design(
        target_frame, ALIGNED_TRANSFER_FEATURE_COLUMNS, imputer, scaler, indicator_value=1.0
    )
    pooled_design = np.vstack([aux_design, target_design])
    pooled_target = np.concatenate(
        [
            auxiliary_frame["ats_margin"].to_numpy(dtype=float),
            target_frame["ats_margin"].to_numpy(dtype=float),
        ]
    )
    manual = Ridge(alpha=CROSS_LEAGUE_RIDGE_ALPHA, fit_intercept=False).fit(
        pooled_design, pooled_target
    )
    np.testing.assert_allclose(_coefficients(joint), manual.coef_, rtol=1e-8, atol=1e-8)


def test_measure_league_mismatch_reports_sane_bounds(
    target_frame: pd.DataFrame, auxiliary_frame: pd.DataFrame
) -> None:
    report = measure_league_mismatch(
        target_frame, auxiliary_frame, label_a="target", label_b="auxiliary"
    )
    assert -1.0 - 1e-9 <= report.cosine_similarity <= 1.0 + 1e-9
    assert report.residual_std_ratio > 0.0
    assert set(report.per_feature["feature"]) == set(ALIGNED_TRANSFER_FEATURE_COLUMNS)
    # Both leagues share the same generating mechanism (only the signal's
    # magnitude differs), so the two coefficient vectors should point in
    # roughly the same overall direction. Per-component sign agreement is NOT
    # asserted: home/away/diff triples are collinear by construction (``diff
    # = home - away``), exactly as in both leagues' real feature contracts,
    # so ridge can trade weight between them noisily at low signal-to-noise --
    # the whole-vector cosine similarity is the robust summary.
    assert report.cosine_similarity > 0.0


# ---------------------------------------------------------------------------
# Leakage regression (release-blocking)
# ---------------------------------------------------------------------------


def test_benchmark_predictions_do_not_depend_on_future_rows(
    target_frame: pd.DataFrame, auxiliary_frame: pd.DataFrame
) -> None:
    result = cross_league_transfer_benchmark(
        target_frame,
        auxiliary_frame,
        start_season=2018,
        end_season=2018,
        min_train_games=50,
        shrinkage_samples=25,
    )
    baseline = result.predictions.sort_values(["method", "game_id"]).reset_index(drop=True)

    perturbed_target = target_frame.copy()
    future_target = perturbed_target["season"].gt(2018) | (
        perturbed_target["season"].eq(2018) & perturbed_target["week"].gt(1)
    )
    rng = np.random.default_rng(99)
    perturbed_target.loc[future_target, "ats_margin"] = rng.normal(
        0.0, 50.0, size=int(future_target.sum())
    )
    perturbed_target.loc[future_target, "result"] = (
        perturbed_target.loc[future_target, "spread_line"]
        + perturbed_target.loc[future_target, "ats_margin"]
    )
    perturbed_target.loc[future_target, "home_off_epa_per_play"] = rng.normal(
        0.0, 5.0, size=int(future_target.sum())
    )

    perturbed_aux = auxiliary_frame.copy()
    future_aux = perturbed_aux["season"].gt(2018) | (
        perturbed_aux["season"].eq(2018) & perturbed_aux["week"].gt(1)
    )
    perturbed_aux.loc[future_aux, "ats_margin"] = rng.normal(0.0, 50.0, size=int(future_aux.sum()))
    perturbed_aux.loc[future_aux, "home_def_epa_per_play"] = rng.normal(
        0.0, 5.0, size=int(future_aux.sum())
    )

    result_perturbed = cross_league_transfer_benchmark(
        perturbed_target,
        perturbed_aux,
        start_season=2018,
        end_season=2018,
        min_train_games=50,
        shrinkage_samples=25,
    )
    perturbed = result_perturbed.predictions.sort_values(["method", "game_id"]).reset_index(
        drop=True
    )

    week1 = baseline["week"].eq(1)
    week1_perturbed = perturbed["week"].eq(1)
    pd.testing.assert_series_equal(
        baseline.loc[week1, "predicted_margin"].reset_index(drop=True),
        perturbed.loc[week1_perturbed, "predicted_margin"].reset_index(drop=True),
        check_names=False,
    )


def test_benchmark_training_never_reaches_the_scored_week(
    target_frame: pd.DataFrame, auxiliary_frame: pd.DataFrame
) -> None:
    result = cross_league_transfer_benchmark(
        target_frame,
        auxiliary_frame,
        start_season=2019,
        end_season=2019,
        min_train_games=50,
        shrinkage_samples=25,
    )
    cutoffs = pd.to_datetime(result.predictions["train_max_gameday"])
    assert (cutoffs < result.predictions["gameday"]).all()


def test_benchmark_rejects_missing_columns(auxiliary_frame: pd.DataFrame) -> None:
    broken = auxiliary_frame.drop(columns=["home_off_epa_per_play"])
    with pytest.raises(DataContractError):
        cross_league_transfer_benchmark(
            broken, auxiliary_frame, start_season=2018, end_season=2018, min_train_games=50
        )
