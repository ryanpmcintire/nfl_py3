"""Degenerate-limit and leakage tests for the XLG-05 CFB→NFL coefficient prior.

Predeclared in ``docs/xlg05_transfer_prior.md`` section 3. Two families of test
live here and they answer different questions.

**Degenerate limits.** Arm (d)'s whole claim is that it is an interpolation
containing its two neighbours exactly -- ``kappa = 0`` IS the NFL-only arm and
``kappa = 1`` IS the CFB-prior arm -- rather than a third mechanism whose
relationship to them has to be taken on trust. These tests pin those identities
to closed-form answers on synthetic frames, so they never touch
``data/processed`` and never depend on any real league's data being present.

**Leakage.** Release-blocking, per AGENTS.md. A walk-forward that can see its
own week looks spectacular and is worthless, and this design has three distinct
ways to leak that a single test would not catch: the NFL training frame, the
CFB auxiliary frame (a second league whose rows are dated independently), and
the leave-one-season-out fold structure that selects the prior strength. Each
gets its own test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.xlg05_transfer_screen as screen
from nfl_ats.cross_league_transfer import (
    CROSS_LEAGUE_RIDGE_ALPHA,
    _augmented_design,
    _fit_theta,
    fit_pooled_preprocessor,
    fit_prior_mean_ridge_model,
    fit_target_only_model,
)
from nfl_ats.xlg05_transfer import (
    XLG05_FEATURE_COLUMNS,
    XLG05_PRIOR_STRENGTH_GRID,
    XLG05_TEAM_QUALITY_COLUMNS,
    auxiliary_prior_theta,
    fit_partially_pooled_model,
    prior_scaled_theta,
    prior_strength_path,
    prior_vector_stability,
    select_prior_strength,
    team_quality_mask,
)

ALPHA = CROSS_LEAGUE_RIDGE_ALPHA
SCORED_SEASON = 2013


def _league_frame(
    *,
    seasons: range,
    weeks: int,
    games_per_week: int,
    seed: int,
    signal: float = 2.0,
    noise: float = 10.0,
) -> pd.DataFrame:
    """A synthetic league carrying the full aligned transfer contract.

    ``ats_margin`` is a known linear function of the EPA-diff columns plus
    noise, so a fitted coefficient vector points somewhere meaningful and the
    degenerate-limit comparisons are not comparing two piles of zeros.
    """

    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2009-09-01")
    rows: list[dict[str, object]] = []
    for season in seasons:
        for week in range(1, weeks + 1):
            gameday = base + pd.Timedelta(days=(season - min(seasons)) * 365 + week * 7)
            for index in range(games_per_week):
                off_home, off_away = rng.normal(0.0, 0.12, size=2)
                def_home, def_away = rng.normal(0.0, 0.12, size=2)
                spread_line = float(rng.normal(0.0, 6.0))
                ats_margin = float(
                    signal * ((off_home - off_away) - (def_home - def_away))
                    + rng.normal(0.0, noise)
                )
                rows.append(
                    {
                        "game_id": f"{season}_{week:02d}_{index:02d}_{seed}",
                        "season": season,
                        "week": week,
                        "gameday": gameday,
                        "spread_line": spread_line,
                        "total_line": float(rng.normal(45.0, 4.0)),
                        "rest_diff": float(rng.integers(-4, 5)),
                        "neutral_site": 0,
                        "week_sin": float(np.sin(2 * np.pi * week / weeks)),
                        "week_cos": float(np.cos(2 * np.pi * week / weeks)),
                        "home_team_games": float(week),
                        "away_team_games": float(week),
                        "home_off_epa_per_play": off_home,
                        "away_off_epa_per_play": off_away,
                        "diff_off_epa_per_play": off_home - off_away,
                        "home_def_epa_per_play": def_home,
                        "away_def_epa_per_play": def_away,
                        "diff_def_epa_per_play": def_home - def_away,
                        "result": spread_line + ats_margin,
                        "ats_margin": ats_margin,
                        "home_cover": float(ats_margin > 0),
                        "season_type": "REG",
                        "game_type": "REG",
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def target_frame() -> pd.DataFrame:
    return _league_frame(seasons=range(2010, 2014), weeks=8, games_per_week=8, seed=11)


@pytest.fixture(scope="module")
def auxiliary_frame() -> pd.DataFrame:
    return _league_frame(
        seasons=range(2009, 2014), weeks=10, games_per_week=12, seed=29, signal=1.2
    )


@pytest.fixture(scope="module")
def preprocessor(
    target_frame: pd.DataFrame, auxiliary_frame: pd.DataFrame
) -> tuple[object, object]:
    return fit_pooled_preprocessor(target_frame, auxiliary_frame, XLG05_FEATURE_COLUMNS)


def _coefficients(model: object) -> np.ndarray:
    estimator = model.estimator  # type: ignore[attr-defined]
    assert estimator is not None
    return np.asarray(estimator.coefficients, dtype=float)


# ---------------------------------------------------------------------------
# Degenerate limits: the prior-mean ridge is the ridge it claims to be
# ---------------------------------------------------------------------------


def test_prior_mean_ridge_reduces_to_plain_ridge_when_the_prior_is_zero(
    target_frame: pd.DataFrame, preprocessor: tuple[object, object]
) -> None:
    """beta_cfb = 0 must give back the project's standing shrink-toward-zero ridge.

    This is the identity that makes arm (a) a special case of arm (d) rather
    than a separate estimator, so any drift here silently changes what the
    baseline means.
    """

    imputer, scaler = preprocessor
    design = _augmented_design(target_frame, XLG05_FEATURE_COLUMNS, imputer, scaler)
    target = pd.to_numeric(target_frame["ats_margin"]).to_numpy(dtype=float)
    plain = _fit_theta(design, target, ALPHA)
    zero_prior = np.zeros(len(XLG05_FEATURE_COLUMNS) + 1, dtype=float)
    for kappa in (0.0, 0.5, 1.0, 3.0):
        np.testing.assert_allclose(
            prior_scaled_theta(design, target, zero_prior, kappa, ALPHA), plain, atol=1e-9
        )


def test_kappa_zero_is_the_nfl_only_arm_and_kappa_one_is_the_cfb_prior_arm(
    target_frame: pd.DataFrame,
    auxiliary_frame: pd.DataFrame,
    preprocessor: tuple[object, object],
) -> None:
    imputer, scaler = preprocessor
    nfl_only = fit_target_only_model(target_frame, imputer, scaler, XLG05_FEATURE_COLUMNS, ALPHA)
    cfb_prior = fit_prior_mean_ridge_model(
        target_frame, auxiliary_frame, imputer, scaler, XLG05_FEATURE_COLUMNS, ALPHA
    )
    at_zero, _ = fit_partially_pooled_model(
        target_frame,
        auxiliary_frame,
        imputer,
        scaler,
        XLG05_FEATURE_COLUMNS,
        ALPHA,
        grid=(0.0,),
    )
    at_one, _ = fit_partially_pooled_model(
        target_frame,
        auxiliary_frame,
        imputer,
        scaler,
        XLG05_FEATURE_COLUMNS,
        ALPHA,
        grid=(1.0,),
    )
    np.testing.assert_allclose(_coefficients(at_zero), _coefficients(nfl_only), atol=1e-9)
    np.testing.assert_allclose(_coefficients(at_one), _coefficients(cfb_prior), atol=1e-9)
    # ... and the two endpoints are genuinely different, so the identities above
    # are not both true merely because the prior happens to be inert.
    assert not np.allclose(_coefficients(at_zero), _coefficients(at_one), atol=1e-6)


def test_prior_strength_path_is_exactly_linear_in_kappa(
    target_frame: pd.DataFrame,
    auxiliary_frame: pd.DataFrame,
    preprocessor: tuple[object, object],
) -> None:
    """The grid is priced by two fits; that shortcut must be exact, not close."""

    imputer, scaler = preprocessor
    design = _augmented_design(target_frame, XLG05_FEATURE_COLUMNS, imputer, scaler)
    target = pd.to_numeric(target_frame["ats_margin"]).to_numpy(dtype=float)
    theta_prior = auxiliary_prior_theta(
        auxiliary_frame, imputer, scaler, XLG05_FEATURE_COLUMNS, ALPHA
    )
    theta_zero, slope = prior_strength_path(design, target, theta_prior, ALPHA)
    for kappa in XLG05_PRIOR_STRENGTH_GRID:
        np.testing.assert_allclose(
            theta_zero + kappa * slope,
            prior_scaled_theta(design, target, theta_prior, kappa, ALPHA),
            atol=1e-9,
        )


def test_masking_every_component_collapses_the_partial_arm_onto_the_baseline(
    target_frame: pd.DataFrame,
    auxiliary_frame: pd.DataFrame,
    preprocessor: tuple[object, object],
) -> None:
    """A prior forbidden from informing anything must equal shrink-toward-zero.

    The bound-check diagnostic masks only the six team-quality components; this
    pins the limiting case so the mask is known to do what it says.
    """

    imputer, scaler = preprocessor
    full_mask = np.ones(len(XLG05_FEATURE_COLUMNS) + 1, dtype=bool)
    masked, _ = fit_partially_pooled_model(
        target_frame,
        auxiliary_frame,
        imputer,
        scaler,
        XLG05_FEATURE_COLUMNS,
        ALPHA,
        prior_mask=full_mask,
    )
    nfl_only = fit_target_only_model(target_frame, imputer, scaler, XLG05_FEATURE_COLUMNS, ALPHA)
    np.testing.assert_allclose(_coefficients(masked), _coefficients(nfl_only), atol=1e-9)


def test_team_quality_mask_marks_the_epa_block_and_never_the_intercept() -> None:
    mask = team_quality_mask(XLG05_FEATURE_COLUMNS)
    assert mask.shape == (len(XLG05_FEATURE_COLUMNS) + 1,)
    assert not bool(mask[-1])
    marked = {column for column, flag in zip(XLG05_FEATURE_COLUMNS, mask[:-1], strict=True) if flag}
    assert marked == set(XLG05_TEAM_QUALITY_COLUMNS)


def test_selection_breaks_ties_toward_the_smallest_kappa(
    target_frame: pd.DataFrame, preprocessor: tuple[object, object]
) -> None:
    """With a zero prior every grid point scores identically; take the status quo.

    The predeclaration freezes "ties break toward the smallest kappa", i.e.
    toward the NFL-only baseline. A tie is exactly what a zero prior produces.
    """

    imputer, scaler = preprocessor
    zero_prior = np.zeros(len(XLG05_FEATURE_COLUMNS) + 1, dtype=float)
    selection = select_prior_strength(
        target_frame, zero_prior, imputer, scaler, XLG05_FEATURE_COLUMNS, ALPHA
    )
    assert selection.used_fallback is False
    assert selection.kappa == min(XLG05_PRIOR_STRENGTH_GRID)
    assert len(set(np.round(selection.mean_squared_error, 12))) == 1


def test_selection_falls_back_when_leave_one_season_out_cannot_run(
    target_frame: pd.DataFrame,
    auxiliary_frame: pd.DataFrame,
    preprocessor: tuple[object, object],
) -> None:
    imputer, scaler = preprocessor
    single_season = target_frame.loc[target_frame["season"].eq(2010)]
    theta_prior = auxiliary_prior_theta(
        auxiliary_frame, imputer, scaler, XLG05_FEATURE_COLUMNS, ALPHA
    )
    selection = select_prior_strength(
        single_season, theta_prior, imputer, scaler, XLG05_FEATURE_COLUMNS, ALPHA
    )
    assert selection.used_fallback is True
    assert selection.kappa == pytest.approx(1.0)


def test_prior_vector_stability_is_bounded_and_uses_both_halves(
    auxiliary_frame: pd.DataFrame, preprocessor: tuple[object, object]
) -> None:
    imputer, scaler = preprocessor
    stability = prior_vector_stability(
        auxiliary_frame, imputer, scaler, XLG05_FEATURE_COLUMNS, ALPHA
    )
    assert -1.0 <= stability.pearson_correlation <= 1.0
    assert -1.0 <= stability.cosine_similarity <= 1.0
    assert stability.odd_rows > 0 and stability.even_rows > 0
    assert stability.odd_rows + stability.even_rows == len(auxiliary_frame)


# ---------------------------------------------------------------------------
# Leakage: three independent ways this design could see its own week
# ---------------------------------------------------------------------------

TEST_ARMS = ("nfl_only", "naive_pooled", "cfb_prior", "partial_pooled", "prior_market_only")


def _scored(target: pd.DataFrame, auxiliary: pd.DataFrame) -> pd.DataFrame:
    frame, _ = screen.run_window(target, auxiliary, (SCORED_SEASON,), arms=TEST_ARMS)
    return frame


def test_the_loso_folds_never_hold_out_the_scored_season(
    target_frame: pd.DataFrame,
    auxiliary_frame: pd.DataFrame,
    preprocessor: tuple[object, object],
) -> None:
    """The prior strength is selected on training seasons only, by construction.

    Checks the fold LABELS, not just the outcome: a fold that held out the
    scored season would still produce a number, and only the label reveals it.
    """

    imputer, scaler = preprocessor
    cutoff = target_frame.loc[target_frame["season"].eq(SCORED_SEASON), "gameday"].min()
    training = target_frame.loc[target_frame["gameday"].lt(cutoff)]
    theta_prior = auxiliary_prior_theta(
        auxiliary_frame.loc[auxiliary_frame["gameday"].lt(cutoff)],
        imputer,
        scaler,
        XLG05_FEATURE_COLUMNS,
        ALPHA,
    )
    selection = select_prior_strength(
        training, theta_prior, imputer, scaler, XLG05_FEATURE_COLUMNS, ALPHA
    )
    assert selection.used_fallback is False
    assert len(selection.fold_seasons) >= 2
    assert SCORED_SEASON not in selection.fold_seasons
    assert max(selection.fold_seasons) < SCORED_SEASON
    assert selection.fold_rows == len(training)


def test_walk_forward_predictions_do_not_depend_on_future_nfl_rows(
    target_frame: pd.DataFrame, auxiliary_frame: pd.DataFrame
) -> None:
    baseline = _scored(target_frame, auxiliary_frame)
    split_week = int(baseline["week"].median())
    corrupted = target_frame.copy()
    future = corrupted["season"].gt(SCORED_SEASON) | (
        corrupted["season"].eq(SCORED_SEASON) & corrupted["week"].gt(split_week)
    )
    corrupted.loc[future, "ats_margin"] = corrupted.loc[future, "ats_margin"] + 500.0
    corrupted.loc[future, "result"] = corrupted.loc[future, "result"] + 500.0
    perturbed = _scored(corrupted, auxiliary_frame)

    early = baseline["week"].le(split_week)
    assert early.any()
    for arm in TEST_ARMS:
        np.testing.assert_allclose(
            baseline.loc[early, f"{arm}_probability"].to_numpy(dtype=float),
            perturbed.loc[early.to_numpy(), f"{arm}_probability"].to_numpy(dtype=float),
            atol=1e-12,
        )


def test_cfb_rows_at_or_after_an_nfl_week_never_enter_that_week(
    target_frame: pd.DataFrame, auxiliary_frame: pd.DataFrame
) -> None:
    """The auxiliary league is dated independently, so it needs its own guard.

    The NFL training frame could be perfectly point-in-time while the CFB frame
    quietly hands the same week its own future -- the two frames are filtered by
    separate expressions, and only this test would notice if one of them stopped
    being filtered.
    """

    baseline = _scored(target_frame, auxiliary_frame)
    last_cutoff = target_frame.loc[
        target_frame["season"].eq(SCORED_SEASON)
        & target_frame["week"].eq(int(target_frame["week"].max())),
        "gameday",
    ].min()
    corrupted = auxiliary_frame.copy()
    future = corrupted["gameday"].ge(last_cutoff)
    assert bool(future.any()), "the fixture must contain CFB games at or after the last NFL week"
    corrupted.loc[future, "ats_margin"] = corrupted.loc[future, "ats_margin"] + 500.0
    corrupted.loc[future, "result"] = corrupted.loc[future, "result"] + 500.0
    perturbed = _scored(target_frame, corrupted)

    for arm in TEST_ARMS:
        np.testing.assert_allclose(
            baseline[f"{arm}_probability"].to_numpy(dtype=float),
            perturbed[f"{arm}_probability"].to_numpy(dtype=float),
            atol=1e-12,
        )


def test_every_arm_trains_strictly_before_the_week_it_scores(
    target_frame: pd.DataFrame, auxiliary_frame: pd.DataFrame
) -> None:
    """A direct assertion on the training cutoff, not an indirect one on outputs."""

    completed = screen._completed(target_frame, regular_only=False)
    window = completed.loc[completed["season"].eq(SCORED_SEASON)]
    for (_season, _week), group in window.groupby(["season", "week"], sort=True):
        cutoff = group["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        assert training["gameday"].max() < cutoff
        assert not training["game_id"].isin(set(group["game_id"])).any()


def test_the_window_is_read_from_the_ledger_not_hand_picked() -> None:
    """An undeclared family must fail loudly rather than fall back to a default."""

    with pytest.raises(SystemExit):
        screen.assigned_seasons("a_family_that_was_never_declared")
