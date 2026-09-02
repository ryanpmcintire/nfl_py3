"""XLG-05: a CFB-anchored PRIOR on the NFL residual model's coefficients.

Predeclared in ``docs/xlg05_transfer_prior.md`` before any NFL outcome number
existed. Read that document first: it declares the shared feature subset S,
the four arms, the rotation family and window, the null, the positive control,
and the recording rules.

What this module adds on top of ``nfl_ats.cross_league_transfer`` -- which
already ships arms (a) ``fit_target_only_model``, (b) ``fit_joint_league_model``
and (c) ``fit_prior_mean_ridge_model`` -- is the *partially pooled* arm (d) and
the machinery to interrogate it:

* :func:`prior_scaled_theta` -- generalized ridge toward ``kappa * theta_prior``.
  ``kappa`` is the PRIOR STRENGTH; ``ridge_alpha`` stays frozen at the project's
  10.0 and is never tuned here.
* :func:`select_prior_strength` -- leave-one-season-out selection of ``kappa``
  over a frozen grid, on TRAINING seasons only.
* :func:`fit_partially_pooled_model` -- arm (d), and (via ``prior_mask``) the
  ``prior_market_only`` bound-check diagnostic that forbids the CFB prior from
  touching any team-quality coefficient.
* :func:`prior_vector_stability` -- odd/even-season split-half agreement of the
  transferred coefficient vector itself, the honest analogue of trait
  reliability for a family whose arms differ in the ESTIMATOR rather than in a
  feature column.

**This is a MODEL change, not a team-quality feature.** Every arm is fit on the
identical 14-column design (:data:`XLG05_FEATURE_COLUMNS`); no arm sees a column
another does not. The only thing that varies is what the ridge shrinks toward.
That is enforced by construction, so a difference between arms cannot be a
team-quality-measurement difference -- which is exactly why the owner's standing
"team quality is already priced" bound does not automatically apply, and why
:data:`XLG05_TEAM_QUALITY_COLUMNS` is named here so section 7 of the
predeclaration can check the point rather than assert it.

The private helpers ``_augmented_design`` / ``_fit_theta`` / ``_target_values``
/ ``_build_transfer_margin_model`` are imported from
``nfl_ats.cross_league_transfer`` deliberately rather than reimplemented: the
whole experiment rests on all four arms' coefficient vectors living in the
LITERAL same augmented, pooled-standardized space, and a second copy of that
construction could silently drift away from the one arms (a)-(c) use.
``tests/test_cross_league_transfer.py`` already imports the same two helpers,
so the pattern is the established one here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from nfl_ats.constants import MIN_FITTABLE_TRAIN_GAMES
from nfl_ats.cross_league_transfer import (
    ALIGNED_TRANSFER_FEATURE_COLUMNS,
    CROSS_LEAGUE_RIDGE_ALPHA,
    _augmented_design,
    _build_transfer_margin_model,
    _fit_theta,
    _target_values,
)
from nfl_ats.margin import MarginModel

#: The shared feature subset S: the measured name-and-semantics intersection of
#: the CFB benchmark contract and the NFL production feature table. Identical to
#: ``ALIGNED_TRANSFER_FEATURE_COLUMNS``, aliased here so this module's own
#: contract is explicit at its call sites.
XLG05_FEATURE_COLUMNS: tuple[str, ...] = ALIGNED_TRANSFER_FEATURE_COLUMNS

#: The team-quality block Q inside S -- the only columns in S that estimate how
#: good the teams are. Named BEFORE the look (predeclaration section 2) so the
#: "team quality is already priced" bound check partitions on a frozen set.
XLG05_TEAM_QUALITY_COLUMNS: tuple[str, ...] = (
    "home_off_epa_per_play",
    "away_off_epa_per_play",
    "diff_off_epa_per_play",
    "home_def_epa_per_play",
    "away_def_epa_per_play",
    "diff_def_epa_per_play",
)

#: Frozen prior-strength grid for arm (d). ``0.0`` reproduces arm (a) exactly
#: and ``1.0`` reproduces arm (c) exactly, so the arm is an interpolation
#: containing both of its neighbours rather than a third mechanism.
XLG05_PRIOR_STRENGTH_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

#: Prior strength used when leave-one-season-out cannot run (fewer than this
#: many usable folds). Falls back onto arm (c)'s setting, not arm (a)'s, so an
#: unselectable week collapses (d) onto the fixed-prior arm.
XLG05_MIN_LOSO_SEASONS = 2
XLG05_FALLBACK_PRIOR_STRENGTH = 1.0


def team_quality_mask(
    feature_columns: tuple[str, ...] = XLG05_FEATURE_COLUMNS,
) -> npt.NDArray[np.bool_]:
    """Boolean mask over the AUGMENTED coefficient vector marking the Q block.

    Length ``len(feature_columns) + 1``; the trailing intercept component is
    always ``False`` (an intercept is not a team-quality coefficient).
    """

    mask = np.array(
        [column in XLG05_TEAM_QUALITY_COLUMNS for column in feature_columns], dtype=bool
    )
    return np.concatenate([mask, np.array([False], dtype=bool)])


# ---------------------------------------------------------------------------
# The prior-strength ridge
# ---------------------------------------------------------------------------


def prior_scaled_theta(
    design: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    theta_prior: npt.NDArray[np.float64],
    kappa: float,
    ridge_alpha: float = CROSS_LEAGUE_RIDGE_ALPHA,
) -> npt.NDArray[np.float64]:
    r"""Minimise ``||y - X t||^2 + alpha * ||t - kappa * theta_prior||^2``.

    Solved by residualizing against the scaled prior's own predictions and
    running an ordinary ``Ridge(fit_intercept=False)`` on the residual -- the
    same substitution ``nfl_ats.cross_league_transfer.fit_prior_mean_ridge_model``
    uses (let ``z = t - kappa*theta_prior``; the objective becomes
    ``||(y - kappa*X*theta_prior) - X z||^2 + alpha ||z||^2``).

    Two exact identities follow, and both are pinned by tests:

    * ``kappa = 0`` (or ``theta_prior = 0``) gives plain ridge -- arm (a).
    * ``kappa = 1`` gives the CFB-prior-mean ridge -- arm (c).

    The solution is also exactly LINEAR in ``kappa``: the normal equations give
    ``(X'X + alpha I) t = X'y + alpha kappa theta_prior``. That is what makes a
    whole grid cost two fits rather than one fit per grid point, with no
    approximation -- see :func:`prior_strength_path`.
    """

    if not np.isfinite(kappa):
        raise ValueError("kappa must be finite")
    shifted = target - kappa * (design @ theta_prior)
    return kappa * theta_prior + _fit_theta(design, shifted, ridge_alpha)


def prior_strength_path(
    design: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    theta_prior: npt.NDArray[np.float64],
    ridge_alpha: float = CROSS_LEAGUE_RIDGE_ALPHA,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return ``(theta_at_zero, slope)`` so ``theta(kappa) = theta0 + kappa*slope``.

    Exact, not an approximation (see :func:`prior_scaled_theta`). Two ridge fits
    price every point on the grid.
    """

    theta_zero = prior_scaled_theta(design, target, theta_prior, 0.0, ridge_alpha)
    theta_one = prior_scaled_theta(design, target, theta_prior, 1.0, ridge_alpha)
    return theta_zero, theta_one - theta_zero


# ---------------------------------------------------------------------------
# Leave-one-season-out selection of the prior strength
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriorStrengthSelection:
    """The chosen prior strength and the evidence behind it.

    ``fold_seasons`` names exactly which seasons were held out, so a reader can
    verify none of them is the scored week's own season.
    """

    kappa: float
    grid: tuple[float, ...]
    mean_squared_error: tuple[float, ...]
    fold_seasons: tuple[int, ...]
    fold_rows: int
    used_fallback: bool


def select_prior_strength(
    target_training: pd.DataFrame,
    theta_prior: npt.NDArray[np.float64],
    imputer: SimpleImputer,
    scaler: StandardScaler,
    feature_columns: tuple[str, ...] = XLG05_FEATURE_COLUMNS,
    ridge_alpha: float = CROSS_LEAGUE_RIDGE_ALPHA,
    *,
    grid: tuple[float, ...] = XLG05_PRIOR_STRENGTH_GRID,
    min_seasons: int = XLG05_MIN_LOSO_SEASONS,
    min_fold_rows: int = MIN_FITTABLE_TRAIN_GAMES,
) -> PriorStrengthSelection:
    """Choose ``kappa`` by leave-one-season-out on ``target_training`` ONLY.

    ``target_training`` is, by the walk-forward contract, every completed target
    -league game strictly before the scored week's earliest kickoff, so no fold
    can contain the scored week. The selection criterion is the pooled
    out-of-fold MEAN SQUARED ERROR of the predicted market residual, frozen in
    the predeclaration: an NFL season is ~256 games, so a held-out-season
    forced-pick accuracy carries roughly +/-3 points of noise and would mostly
    select noise, whereas MSE is the loss the ridge itself minimises.

    Ties break toward the SMALLEST ``kappa`` (toward the NFL-only status quo),
    which ``np.argmin`` on an ascending grid does for free.

    ``theta_prior`` is fixed across folds on purpose: it is fit on auxiliary
    -league games already played before the same cutoff, which are legitimately
    available at prediction time. The fold structure is a target-league device
    and does not restrict them.
    """

    if not grid:
        raise ValueError("The prior-strength grid must not be empty")
    if any(not np.isfinite(value) for value in grid):
        raise ValueError("Every prior-strength grid value must be finite")
    grid_values = np.asarray(grid, dtype=np.float64)

    seasons = sorted(int(value) for value in pd.unique(target_training["season"]))
    squared_error = np.zeros(len(grid_values), dtype=np.float64)
    used_seasons: list[int] = []
    rows = 0
    for season in seasons:
        held_out = target_training.loc[target_training["season"].astype(int).eq(season)]
        fit_part = target_training.loc[~target_training["season"].astype(int).eq(season)]
        if held_out.empty or len(fit_part) < min_fold_rows:
            continue
        fit_design = _augmented_design(fit_part, feature_columns, imputer, scaler)
        out_design = _augmented_design(held_out, feature_columns, imputer, scaler)
        out_target = _target_values(held_out)
        theta_zero, slope = prior_strength_path(
            fit_design, _target_values(fit_part), theta_prior, ridge_alpha
        )
        base_prediction = out_design @ theta_zero
        slope_prediction = out_design @ slope
        for index, kappa in enumerate(grid_values):
            residual = out_target - (base_prediction + kappa * slope_prediction)
            squared_error[index] += float(np.sum(np.square(residual)))
        used_seasons.append(season)
        rows += len(held_out)

    if len(used_seasons) < min_seasons or rows == 0:
        return PriorStrengthSelection(
            kappa=XLG05_FALLBACK_PRIOR_STRENGTH,
            grid=tuple(float(value) for value in grid_values),
            mean_squared_error=tuple(float("nan") for _ in grid_values),
            fold_seasons=tuple(used_seasons),
            fold_rows=rows,
            used_fallback=True,
        )
    mse = squared_error / float(rows)
    return PriorStrengthSelection(
        kappa=float(grid_values[int(np.argmin(mse))]),
        grid=tuple(float(value) for value in grid_values),
        mean_squared_error=tuple(float(value) for value in mse),
        fold_seasons=tuple(used_seasons),
        fold_rows=rows,
        used_fallback=False,
    )


# ---------------------------------------------------------------------------
# Arm (d), and the bound-check diagnostic that shares its machinery
# ---------------------------------------------------------------------------


def auxiliary_prior_theta(
    auxiliary_training: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    feature_columns: tuple[str, ...] = XLG05_FEATURE_COLUMNS,
    ridge_alpha: float = CROSS_LEAGUE_RIDGE_ALPHA,
) -> npt.NDArray[np.float64]:
    """``theta_cfb``: the auxiliary league's own ridge fit, the transferred object."""

    design = _augmented_design(auxiliary_training, feature_columns, imputer, scaler)
    return _fit_theta(design, _target_values(auxiliary_training), ridge_alpha)


def fit_partially_pooled_model(
    target_training: pd.DataFrame,
    auxiliary_training: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    feature_columns: tuple[str, ...] = XLG05_FEATURE_COLUMNS,
    ridge_alpha: float = CROSS_LEAGUE_RIDGE_ALPHA,
    *,
    grid: tuple[float, ...] = XLG05_PRIOR_STRENGTH_GRID,
    prior_mask: npt.NDArray[np.bool_] | None = None,
    model_name: str = "partial_pooled",
) -> tuple[MarginModel, PriorStrengthSelection]:
    """Arm (d): generalized ridge toward ``kappa * theta_cfb``, ``kappa`` by LOSO.

    ``prior_mask``, when given, is a boolean mask over the augmented coefficient
    vector marking components the prior is FORBIDDEN to inform (their prior mean
    is forced to zero, i.e. the project's standing convention). Passing
    :func:`team_quality_mask` produces the ``prior_market_only`` bound-check
    diagnostic of predeclaration section 3: CFB may inform the market, context,
    experience and intercept coefficients but not a single team-quality one.

    ``kappa`` is selected ONCE per call, on the full ``target_training`` frame,
    and then held fixed for both of ``_build_transfer_margin_model``'s coefficient
    fits (the 80% residual-distribution fit and the 100% scoring fit). It is a
    hyperparameter of the week's model chosen on training data, so re-selecting
    it inside the 80% split would only add noise, not leak-safety.
    """

    theta_prior = auxiliary_prior_theta(
        auxiliary_training, imputer, scaler, feature_columns, ridge_alpha
    )
    if prior_mask is not None:
        if prior_mask.shape != theta_prior.shape:
            raise ValueError("prior_mask must match the augmented coefficient vector's length")
        theta_prior = np.where(prior_mask, 0.0, theta_prior)

    selection = select_prior_strength(
        target_training,
        theta_prior,
        imputer,
        scaler,
        feature_columns,
        ridge_alpha,
        grid=grid,
    )
    kappa = selection.kappa

    def coefficient_fn(subset: pd.DataFrame) -> npt.NDArray[np.float64]:
        design = _augmented_design(subset, feature_columns, imputer, scaler)
        return prior_scaled_theta(design, _target_values(subset), theta_prior, kappa, ridge_alpha)

    model = _build_transfer_margin_model(
        target_training,
        coefficient_fn,
        imputer,
        scaler,
        feature_columns,
        ridge_alpha,
        model_name,
    )
    return model, selection


# ---------------------------------------------------------------------------
# Stability of the transferred object itself
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriorStability:
    """Odd/even-season split-half agreement of ``theta_cfb``.

    NOT trait reliability: this family's arms differ in the ESTIMATOR, not in a
    feature column, so there is no NFL-side trait to split. This measures
    whether the transferred coefficient vector itself replicates across halves
    of the auxiliary corpus. Per ``docs/scaling_and_transfer.md``'s caveat,
    several columns are exactly collinear by construction (``diff = home -
    away``), so the WHOLE-VECTOR agreement is the robust read and individual
    component signs are not.
    """

    pearson_correlation: float
    spearman_brown: float
    cosine_similarity: float
    odd_rows: int
    even_rows: int


def prior_vector_stability(
    auxiliary: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    feature_columns: tuple[str, ...] = XLG05_FEATURE_COLUMNS,
    ridge_alpha: float = CROSS_LEAGUE_RIDGE_ALPHA,
) -> PriorStability:
    """Fit ``theta_cfb`` on odd and on even seasons and compare the two vectors.

    Auxiliary-league only, so this costs no target-league rotation window. The
    comparison uses the FEATURE coefficients (the intercept is on a different
    scale and would dominate a raw correlation).
    """

    season = auxiliary["season"].astype(int)
    odd = auxiliary.loc[season.mod(2).eq(1)]
    even = auxiliary.loc[season.mod(2).eq(0)]
    if len(odd) < MIN_FITTABLE_TRAIN_GAMES or len(even) < MIN_FITTABLE_TRAIN_GAMES:
        raise ValueError("Both season halves need enough rows to fit a prior vector")
    theta_odd = auxiliary_prior_theta(odd, imputer, scaler, feature_columns, ridge_alpha)[:-1]
    theta_even = auxiliary_prior_theta(even, imputer, scaler, feature_columns, ridge_alpha)[:-1]
    correlation = float(np.corrcoef(theta_odd, theta_even)[0, 1])
    norms = float(np.linalg.norm(theta_odd)) * float(np.linalg.norm(theta_even))
    cosine = float(np.dot(theta_odd, theta_even) / norms) if norms > 0 else float("nan")
    denominator = 1.0 + correlation
    spearman_brown = (
        float(2.0 * correlation / denominator) if abs(denominator) > 1e-12 else float("nan")
    )
    return PriorStability(
        pearson_correlation=correlation,
        spearman_brown=spearman_brown,
        cosine_similarity=cosine,
        odd_rows=len(odd),
        even_rows=len(even),
    )
