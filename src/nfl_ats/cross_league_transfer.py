"""Cross-league transfer: does CFB's larger corpus help the NFL fit?

CFB currently exists in this project only as a *screen* -- a hypothesis is
tested there and, if it clears, licenses spending an NFL confirmation
window. Its fitted parameters are discarded once the screen is scored. This
module builds the machinery to reuse them instead, mirroring how a
pretraining corpus is normally used: not just for evals, but as a source of
transferable structure.

Three mechanisms, all built on one **aligned feature contract**
(``ALIGNED_TRANSFER_FEATURE_COLUMNS``) -- the subset of columns that name the
literal same football quantity under the literal same name in both the CFB
benchmark contract (``nfl_ats.cfb_features.CFB_MODEL_FEATURE_COLUMNS``) and
the NFL ``football``/``full`` feature family (``nfl_ats.constants``). The two
contracts otherwise diverge (CFB lacks pass/rush-split EPA, CPOE, Elo, ...;
NFL's base profile lacks success/explosive rate at this contract level), so
only the 14-column intersection is a well-posed space to compare or blend
coefficients in:

- ``spread_line``, ``total_line`` (market)
- ``rest_diff``, ``neutral_site``, ``week_sin``, ``week_cos`` (context)
- ``home_team_games``, ``away_team_games`` (experience)
- ``home_off_epa_per_play``, ``away_off_epa_per_play``, ``diff_off_epa_per_play``
- ``home_def_epa_per_play``, ``away_def_epa_per_play``, ``diff_def_epa_per_play``

Mechanisms:

(a) ``fit_joint_league_model`` -- pools both leagues' rows plus a binary
    league indicator into one ridge fit, so the 14 shared coefficients are
    estimated on the combined sample and only the indicator's coefficient
    (an additive league-level offset) is league-specific.
(b) ``fit_hierarchical_shrinkage_model`` / ``derive_shrinkage_weights`` --
    partial pooling: the target league's own ridge estimate is shrunk toward
    the auxiliary league's estimate, per coefficient, by an empirical-Bayes
    weight derived from data (``docs/scaling_and_transfer.md`` derives the
    formula) rather than assumed.
(c) ``fit_prior_mean_ridge_model`` -- generalized ridge with the auxiliary
    league's fitted coefficients as an explicit, non-zero prior mean:
    minimises ``||y - X*theta||^2 + alpha*||theta - theta_0||^2``.

``measure_league_mismatch`` fits both leagues on a **pooled** preprocessing
pipeline (one imputer, one scaler, fit on the union of both leagues' rows) so
coefficients live in a genuinely common, comparable space, and reports
cosine similarity, correlation, and the residual-scale ratio between the two
fits. This is a diagnostic (whole-history, both fits static) -- it is never
scored against held-out outcomes and touches no NFL rotation-registry window.

**Every walk-forward entry point in this module (``cross_league_transfer_benchmark``)
is league-agnostic**: it takes a ``target`` frame and an ``auxiliary`` frame
and knows nothing about which league is which. That is deliberate. This
session runs it once as a free, unlimited, fully-CFB-internal validity check
(Power-Five as the large "pretraining" auxiliary league, Group-of-Five as the
smaller "target" league -- a real, measured distribution shift in talent and
market depth, entirely inside rule 8's free CFB budget) to ask whether the
mechanism can ever help before any NFL confirmation is proposed. Running it
with real CFB and NFL frames is a **future, separately predeclared** step
that spends an NFL rotation-registry window this session does not spend --
see ``docs/scaling_and_transfer.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from nfl_ats.cfb_common import blocked_bootstrap_positions, week_block_indices
from nfl_ats.data import require_columns
from nfl_ats.margin import MarginModel, fit_market_baseline

# ---------------------------------------------------------------------------
# The aligned feature contract
# ---------------------------------------------------------------------------

ALIGNED_TRANSFER_FEATURE_COLUMNS: tuple[str, ...] = (
    "spread_line",
    "total_line",
    "rest_diff",
    "neutral_site",
    "week_sin",
    "week_cos",
    "home_team_games",
    "away_team_games",
    "home_off_epa_per_play",
    "away_off_epa_per_play",
    "diff_off_epa_per_play",
    "home_def_epa_per_play",
    "away_def_epa_per_play",
    "diff_def_epa_per_play",
)

# Frozen, taken verbatim from the project's standing ridge convention
# (CFB_BENCHMARK_RIDGE_ALPHA / the NFL market_residual model). Never tuned
# in this module -- a swept alpha would be a new, separate question.
CROSS_LEAGUE_RIDGE_ALPHA = 10.0
CROSS_LEAGUE_DISTRIBUTION_FRACTION = 0.20
CROSS_LEAGUE_MIN_DISTRIBUTION_ROWS = 10
CROSS_LEAGUE_MIN_TRAIN_GAMES = 50
CROSS_LEAGUE_SHRINKAGE_BOOTSTRAP_SAMPLES = 200
CROSS_LEAGUE_SHRINKAGE_BOOTSTRAP_SEED = 20260818
CROSS_LEAGUE_BOOTSTRAP_SAMPLES = 2_000
CROSS_LEAGUE_BOOTSTRAP_SEED = 20260818

_REQUIRED_BASE_COLUMNS = (
    "game_id",
    "season",
    "week",
    "gameday",
    "spread_line",
    "result",
    "ats_margin",
)


def _require_transfer_columns(
    frame: pd.DataFrame, dataset: str, feature_columns: tuple[str, ...]
) -> None:
    require_columns(frame, (*_REQUIRED_BASE_COLUMNS, *feature_columns), dataset)


# ---------------------------------------------------------------------------
# Preprocessing shared by every arm
# ---------------------------------------------------------------------------


def fit_pooled_preprocessor(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    feature_columns: tuple[str, ...] = ALIGNED_TRANSFER_FEATURE_COLUMNS,
) -> tuple[SimpleImputer, StandardScaler]:
    """Fit one median-imputer and one scaler on the UNION of both leagues' rows.

    Coefficients are only comparable, and a prior mean / blended estimate is
    only meaningful, if both leagues' designs live in the same standardized
    space. A per-league scaler would silently confound "the leagues differ"
    with "the leagues have different feature scales".
    """

    pooled = pd.concat(
        [frame_a.loc[:, list(feature_columns)], frame_b.loc[:, list(feature_columns)]],
        ignore_index=True,
    )
    if pooled.empty:
        raise ValueError("Cannot fit a pooled preprocessor on zero rows")
    imputer = SimpleImputer(strategy="median").fit(pooled)
    scaler = StandardScaler().fit(imputer.transform(pooled))
    return imputer, scaler


def _augmented_design(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    imputer: SimpleImputer,
    scaler: StandardScaler,
    *,
    indicator_value: float | None = None,
) -> npt.NDArray[np.float64]:
    """``[standardized features, (league indicator), constant 1]``.

    The trailing constant column stands in for the intercept: every arm below
    fits ``Ridge(fit_intercept=False)`` on this design so the intercept is
    just one more component of the coefficient vector. That is deliberate --
    it lets the hierarchical and prior-mean arms blend/shrink the intercept
    with exactly the same machinery as every slope, rather than special-casing
    it. The one side effect (the intercept picks up a light ridge penalty
    too) is immaterial here: both leagues' ``ats_margin`` is a market
    residual, already centered near zero by construction.
    """

    scaled = scaler.transform(imputer.transform(frame.loc[:, list(feature_columns)]))
    blocks = [np.asarray(scaled, dtype=np.float64)]
    if indicator_value is not None:
        blocks.append(np.full((len(frame), 1), float(indicator_value), dtype=np.float64))
    blocks.append(np.ones((len(frame), 1), dtype=np.float64))
    return np.hstack(blocks)


def _target_values(frame: pd.DataFrame) -> npt.NDArray[np.float64]:
    values = pd.to_numeric(frame["ats_margin"], errors="raise").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("ats_margin must be finite for every row used to fit a transfer arm")
    return values


def _fit_theta(
    design: npt.NDArray[np.float64],
    target: npt.NDArray[np.float64],
    ridge_alpha: float,
) -> npt.NDArray[np.float64]:
    model = Ridge(alpha=ridge_alpha, fit_intercept=False)
    model.fit(design, target)
    return np.asarray(model.coef_, dtype=np.float64)


# ---------------------------------------------------------------------------
# Single-league ridge fit + the mismatch measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeagueRidgeFit:
    """A single league's ridge fit on the pooled-preprocessor design.

    ``coefficients`` has length ``len(feature_columns) + 1``: the standardized
    feature coefficients followed by the intercept (see ``_augmented_design``).
    """

    league: str
    feature_columns: tuple[str, ...]
    coefficients: npt.NDArray[np.float64]
    residual_std: float
    training_rows: int

    @property
    def feature_coefficients(self) -> npt.NDArray[np.float64]:
        return self.coefficients[:-1]

    @property
    def intercept(self) -> float:
        return float(self.coefficients[-1])


def fit_league_ridge(
    frame: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    feature_columns: tuple[str, ...] = ALIGNED_TRANSFER_FEATURE_COLUMNS,
    ridge_alpha: float = CROSS_LEAGUE_RIDGE_ALPHA,
    *,
    league: str = "league",
) -> LeagueRidgeFit:
    _require_transfer_columns(frame, f"{league} training", feature_columns)
    if len(frame) < CROSS_LEAGUE_MIN_TRAIN_GAMES:
        raise ValueError(
            f"At least {CROSS_LEAGUE_MIN_TRAIN_GAMES} games are required to fit a league ridge"
        )
    design = _augmented_design(frame, feature_columns, imputer, scaler)
    target = _target_values(frame)
    coefficients = _fit_theta(design, target, ridge_alpha)
    residual_std = float(np.std(target - design @ coefficients, ddof=1))
    return LeagueRidgeFit(
        league=league,
        feature_columns=feature_columns,
        coefficients=coefficients,
        residual_std=residual_std,
        training_rows=len(frame),
    )


@dataclass(frozen=True)
class LeagueMismatchReport:
    """Measured mismatch between two leagues' fitted market-residual models.

    Diagnostic only: both fits use every completed row each frame contains, a
    single static comparison, never scored against held-out outcomes. It
    exists to give the shrinkage arms an empirical basis instead of a
    hopeful one, per the task framing.
    """

    label_a: str
    label_b: str
    feature_columns: tuple[str, ...]
    fit_a: LeagueRidgeFit
    fit_b: LeagueRidgeFit
    cosine_similarity: float
    pearson_correlation: float
    residual_std_ratio: float
    per_feature: pd.DataFrame


def measure_league_mismatch(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    feature_columns: tuple[str, ...] = ALIGNED_TRANSFER_FEATURE_COLUMNS,
    ridge_alpha: float = CROSS_LEAGUE_RIDGE_ALPHA,
    *,
    label_a: str = "a",
    label_b: str = "b",
) -> LeagueMismatchReport:
    imputer, scaler = fit_pooled_preprocessor(frame_a, frame_b, feature_columns)
    fit_a = fit_league_ridge(frame_a, imputer, scaler, feature_columns, ridge_alpha, league=label_a)
    fit_b = fit_league_ridge(frame_b, imputer, scaler, feature_columns, ridge_alpha, league=label_b)
    coef_a, coef_b = fit_a.feature_coefficients, fit_b.feature_coefficients
    norm_a, norm_b = float(np.linalg.norm(coef_a)), float(np.linalg.norm(coef_b))
    cosine = (
        float(np.dot(coef_a, coef_b) / (norm_a * norm_b))
        if norm_a > 0 and norm_b > 0
        else float("nan")
    )
    correlation = float(np.corrcoef(coef_a, coef_b)[0, 1]) if len(coef_a) > 1 else float("nan")
    per_feature = pd.DataFrame(
        {
            "feature": list(feature_columns),
            f"coefficient_{label_a}": coef_a,
            f"coefficient_{label_b}": coef_b,
            "abs_difference": np.abs(coef_a - coef_b),
            "same_sign": np.sign(coef_a) == np.sign(coef_b),
        }
    )
    return LeagueMismatchReport(
        label_a=label_a,
        label_b=label_b,
        feature_columns=feature_columns,
        fit_a=fit_a,
        fit_b=fit_b,
        cosine_similarity=cosine,
        pearson_correlation=correlation,
        residual_std_ratio=fit_b.residual_std / fit_a.residual_std,
        per_feature=per_feature,
    )


# ---------------------------------------------------------------------------
# Derived shrinkage strength (empirical-Bayes / random-effects partial pooling)
# ---------------------------------------------------------------------------


def _bootstrap_theta_variance(
    frame: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    feature_columns: tuple[str, ...],
    ridge_alpha: float,
    *,
    samples: int,
    seed: int,
) -> npt.NDArray[np.float64]:
    """Week-blocked bootstrap variance of the target league's own ridge fit.

    Blocks whole weeks together (not individual games) to preserve the same
    within-week schedule dependence every other paired bootstrap in this
    project respects.
    """

    blocks = week_block_indices(frame)
    if len(blocks) < 2:
        raise ValueError("At least two week-blocks are required to bootstrap coefficient variance")
    draws = np.empty((samples, len(feature_columns) + 1), dtype=np.float64)
    for sample_index, positions in enumerate(
        blocked_bootstrap_positions(blocks, samples=samples, seed=seed)
    ):
        resample = frame.iloc[positions]
        design = _augmented_design(resample, feature_columns, imputer, scaler)
        target = _target_values(resample)
        draws[sample_index] = _fit_theta(design, target, ridge_alpha)
    variance = np.var(draws, axis=0, ddof=1)
    # A component that never moves across resamples (e.g. a constant column
    # after imputation) would otherwise divide by zero downstream; floor it
    # at a tiny epsilon rather than special-casing.
    return np.maximum(variance, 1e-12)


@dataclass(frozen=True)
class ShrinkageDerivation:
    """Per-component empirical-Bayes shrinkage weight toward the auxiliary fit.

    ``weights[j]`` is the posterior weight on the TARGET league's own
    estimate for coefficient ``j`` (the complement goes to the auxiliary
    fit): ``theta_hier[j] = weights[j] * theta_target[j] + (1 - weights[j]) *
    theta_aux[j]``. Derived, not assumed -- see
    ``docs/scaling_and_transfer.md`` for the formula's provenance
    (DerSimonian-Laird random-effects variance with known per-coefficient
    prior means).
    """

    feature_columns: tuple[str, ...]
    theta_target: npt.NDArray[np.float64]
    theta_aux: npt.NDArray[np.float64]
    target_variance: npt.NDArray[np.float64]
    tau_squared: float
    weights: npt.NDArray[np.float64]


def derive_shrinkage_weights(
    target_training: pd.DataFrame,
    auxiliary_training: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    feature_columns: tuple[str, ...] = ALIGNED_TRANSFER_FEATURE_COLUMNS,
    ridge_alpha: float = CROSS_LEAGUE_RIDGE_ALPHA,
    *,
    samples: int = CROSS_LEAGUE_SHRINKAGE_BOOTSTRAP_SAMPLES,
    seed: int = CROSS_LEAGUE_SHRINKAGE_BOOTSTRAP_SEED,
) -> ShrinkageDerivation:
    r"""Derive a per-coefficient shrinkage weight from data.

    Treats each of the ``K + 1`` augmented coefficients as one "study" in a
    random-effects meta-analysis with a KNOWN mean (the auxiliary league's
    fitted value acts as the prior mean, not a free parameter):

    ``theta_target[j] ~ N(theta_aux[j], tau^2 + sigma_j^2)``

    ``sigma_j^2`` (the target league's own sampling variance for coefficient
    ``j``) is estimated by a week-blocked bootstrap of the target-only ridge
    fit. ``tau^2`` (how much, in general, a target-league coefficient tends
    to drift from its auxiliary anchor beyond that sampling noise) is the
    DerSimonian-Laird closed-form moment estimator with known per-study
    means:

    ``Q = sum_j (theta_target[j] - theta_aux[j])^2 / sigma_j^2``
    ``tau^2 = max(0, (Q - (K - 1)) / sum_j (1 / sigma_j^2))``

    and the posterior weight on the target's own estimate is
    ``w_j = tau^2 / (tau^2 + sigma_j^2)``: heavy shrinkage toward the
    auxiliary anchor when the target's own estimate is noisy relative to how
    much leagues typically differ, light shrinkage when the target has ample
    data of its own.
    """

    target_fit = fit_league_ridge(
        target_training, imputer, scaler, feature_columns, ridge_alpha, league="target"
    )
    aux_fit = fit_league_ridge(
        auxiliary_training, imputer, scaler, feature_columns, ridge_alpha, league="auxiliary"
    )
    variance = _bootstrap_theta_variance(
        target_training, imputer, scaler, feature_columns, ridge_alpha, samples=samples, seed=seed
    )
    theta_target = target_fit.coefficients
    theta_aux = aux_fit.coefficients
    diff_squared = np.square(theta_target - theta_aux)
    q_statistic = float(np.sum(diff_squared / variance))
    k = len(theta_target)
    inverse_variance_sum = float(np.sum(1.0 / variance))
    tau_squared = max(0.0, (q_statistic - (k - 1)) / inverse_variance_sum)
    weights = tau_squared / (tau_squared + variance)
    return ShrinkageDerivation(
        feature_columns=feature_columns,
        theta_target=theta_target,
        theta_aux=theta_aux,
        target_variance=variance,
        tau_squared=tau_squared,
        weights=weights,
    )


# ---------------------------------------------------------------------------
# A fitted coefficient vector, wrapped so MarginModel's machinery applies
# unchanged (cover probabilities, line sweep, the outcome bootstrap, ...)
# ---------------------------------------------------------------------------


class _FixedLinearRegressor(BaseEstimator):
    """A frozen ``predict(X) = augmented_design(X) @ coefficients``.

    Lets any of this module's coefficient vectors (target-only, joint,
    hierarchical, prior-mean) plug into ``nfl_ats.margin.MarginModel``
    unchanged, so cover probabilities, the three-way push split, the line
    sweep, and ``nfl_ats.outcomes`` all come for free instead of being
    re-implemented.
    """

    def __init__(
        self,
        imputer: SimpleImputer,
        scaler: StandardScaler,
        feature_columns: tuple[str, ...],
        coefficients: npt.NDArray[np.float64],
        indicator_value: float | None,
    ) -> None:
        self.imputer = imputer
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.coefficients = coefficients
        self.indicator_value = indicator_value

    def predict(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        design = _augmented_design(
            frame,
            self.feature_columns,
            self.imputer,
            self.scaler,
            indicator_value=self.indicator_value,
        )
        return np.asarray(design @ self.coefficients, dtype=np.float64)


def _out_of_time_split(
    frame: pd.DataFrame,
    *,
    distribution_fraction: float = CROSS_LEAGUE_DISTRIBUTION_FRACTION,
    min_distribution_rows: int = CROSS_LEAGUE_MIN_DISTRIBUTION_ROWS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    distribution_rows = int(len(ordered) * distribution_fraction)
    if distribution_rows < min_distribution_rows or len(ordered) - distribution_rows < 40:
        raise ValueError("Not enough rows for an out-of-time residual distribution")
    split = len(ordered) - distribution_rows
    return ordered.iloc[:split], ordered.iloc[split:]


def _build_transfer_margin_model(
    target_training: pd.DataFrame,
    coefficient_fn: Callable[[pd.DataFrame], npt.NDArray[np.float64]],
    imputer: SimpleImputer,
    scaler: StandardScaler,
    feature_columns: tuple[str, ...],
    ridge_alpha: float,
    model_name: str,
    *,
    indicator_value: float | None = None,
) -> MarginModel:
    """Fit + out-of-time residual distribution, mirroring ``fit_cfb_residual_model``.

    ``coefficient_fn`` receives whatever slice of ``target_training`` should
    be fit (the 80% fit-part, then the full 100%) and returns an augmented
    coefficient vector; it closes over the auxiliary league's data so each
    arm can decide how to use it (pool it, anchor a prior mean, blend by a
    derived weight, or ignore it entirely for the target-only baseline).
    """

    fit_part, distribution_part = _out_of_time_split(target_training)
    fit_coefficients = coefficient_fn(fit_part)
    temporary = _FixedLinearRegressor(
        imputer, scaler, feature_columns, fit_coefficients, indicator_value
    )
    calibration_prediction = temporary.predict(distribution_part)
    residuals = _target_values(distribution_part) - calibration_prediction
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < CROSS_LEAGUE_MIN_DISTRIBUTION_ROWS:
        raise ValueError("Out-of-time residual distribution has too few finite values")

    full_coefficients = coefficient_fn(target_training)
    estimator = _FixedLinearRegressor(
        imputer, scaler, feature_columns, full_coefficients, indicator_value
    )
    ordered = target_training.sort_values(["gameday", "game_id"])
    return MarginModel(
        estimator=estimator,
        residuals=residuals,
        model_name=model_name,
        ridge_alpha=ridge_alpha,
        target="market_residual",
        feature_columns=feature_columns,
        training_rows=len(target_training),
        distribution_rows=len(residuals),
        training_max_gameday=ordered["gameday"].max().date().isoformat(),
    )


# ---------------------------------------------------------------------------
# The three transfer arms
# ---------------------------------------------------------------------------

TransferArm = Literal["target_only", "joint", "hierarchical", "prior_mean"]
TRANSFER_ARMS: tuple[TransferArm, ...] = ("target_only", "joint", "hierarchical", "prior_mean")


def fit_target_only_model(
    target_training: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    feature_columns: tuple[str, ...] = ALIGNED_TRANSFER_FEATURE_COLUMNS,
    ridge_alpha: float = CROSS_LEAGUE_RIDGE_ALPHA,
) -> MarginModel:
    """The current NFL-only-style fit: the target league, alone.

    The baseline every arm below is measured against.
    """

    def coefficient_fn(subset: pd.DataFrame) -> npt.NDArray[np.float64]:
        design = _augmented_design(subset, feature_columns, imputer, scaler)
        return _fit_theta(design, _target_values(subset), ridge_alpha)

    return _build_transfer_margin_model(
        target_training,
        coefficient_fn,
        imputer,
        scaler,
        feature_columns,
        ridge_alpha,
        "target_only",
    )


def fit_joint_league_model(
    target_training: pd.DataFrame,
    auxiliary_training: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    feature_columns: tuple[str, ...] = ALIGNED_TRANSFER_FEATURE_COLUMNS,
    ridge_alpha: float = CROSS_LEAGUE_RIDGE_ALPHA,
) -> MarginModel:
    """Pool both leagues' rows plus a binary league indicator into one ridge fit.

    The 14 shared coefficients are estimated jointly on the pooled sample;
    only the indicator's own coefficient is league-specific (an additive
    offset). Scored predictions fix the indicator at 1 (target league).
    """

    aux_design = _augmented_design(
        auxiliary_training, feature_columns, imputer, scaler, indicator_value=0.0
    )
    aux_target = _target_values(auxiliary_training)

    def coefficient_fn(subset: pd.DataFrame) -> npt.NDArray[np.float64]:
        target_design = _augmented_design(
            subset, feature_columns, imputer, scaler, indicator_value=1.0
        )
        pooled_design = np.vstack([aux_design, target_design])
        pooled_target = np.concatenate([aux_target, _target_values(subset)])
        return _fit_theta(pooled_design, pooled_target, ridge_alpha)

    return _build_transfer_margin_model(
        target_training,
        coefficient_fn,
        imputer,
        scaler,
        feature_columns,
        ridge_alpha,
        "joint",
        indicator_value=1.0,
    )


def fit_hierarchical_shrinkage_model(
    target_training: pd.DataFrame,
    auxiliary_training: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    feature_columns: tuple[str, ...],
    ridge_alpha: float,
    weights: npt.NDArray[np.float64],
) -> MarginModel:
    """Partial pooling: blend the target-only and auxiliary-only estimates by ``weights``.

    ``weights`` is precomputed once by ``derive_shrinkage_weights`` (see that
    function for how the weight is derived) and held fixed while the two
    anchor estimates continue to update as training data expands walking
    forward -- separating "how much to trust the target's own data" from
    "what the target's own data currently says".
    """

    def coefficient_fn(subset: pd.DataFrame) -> npt.NDArray[np.float64]:
        target_design = _augmented_design(subset, feature_columns, imputer, scaler)
        theta_target = _fit_theta(target_design, _target_values(subset), ridge_alpha)
        aux_design = _augmented_design(auxiliary_training, feature_columns, imputer, scaler)
        theta_aux = _fit_theta(aux_design, _target_values(auxiliary_training), ridge_alpha)
        return weights * theta_target + (1.0 - weights) * theta_aux

    return _build_transfer_margin_model(
        target_training,
        coefficient_fn,
        imputer,
        scaler,
        feature_columns,
        ridge_alpha,
        "hierarchical",
    )


def fit_prior_mean_ridge_model(
    target_training: pd.DataFrame,
    auxiliary_training: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    feature_columns: tuple[str, ...] = ALIGNED_TRANSFER_FEATURE_COLUMNS,
    ridge_alpha: float = CROSS_LEAGUE_RIDGE_ALPHA,
) -> MarginModel:
    r"""Generalized ridge with the auxiliary league's coefficients as the prior mean.

    Minimises ``||y - X theta||^2 + alpha ||theta - theta_0||^2`` where
    ``theta_0`` is the auxiliary-only fit. Solved by residualizing the target
    against the prior's own predictions and running an ordinary
    ``Ridge(fit_intercept=False)`` on the residual -- algebraically identical
    to the generalized-ridge closed form (let ``z = theta - theta_0``; the
    objective becomes ``||(y - X theta_0) - X z||^2 + alpha ||z||^2``, an
    ordinary ridge regression of ``y - X theta_0`` on ``X``), and reuses
    ``sklearn.Ridge`` instead of a second hand-rolled linear-algebra path.
    """

    def coefficient_fn(subset: pd.DataFrame) -> npt.NDArray[np.float64]:
        aux_design = _augmented_design(auxiliary_training, feature_columns, imputer, scaler)
        theta_aux = _fit_theta(aux_design, _target_values(auxiliary_training), ridge_alpha)
        design = _augmented_design(subset, feature_columns, imputer, scaler)
        target = _target_values(subset)
        residual_target = target - design @ theta_aux
        delta = _fit_theta(design, residual_target, ridge_alpha)
        return theta_aux + delta

    return _build_transfer_margin_model(
        target_training, coefficient_fn, imputer, scaler, feature_columns, ridge_alpha, "prior_mean"
    )


# ---------------------------------------------------------------------------
# Walk-forward driver
# ---------------------------------------------------------------------------

_PREDICTION_PASSTHROUGH = (
    "game_id",
    "season",
    "week",
    "gameday",
    "spread_line",
    "home_spread_odds",
    "away_spread_odds",
    "result",
    "ats_margin",
    "home_cover",
)


def _score_week(weekly_games: pd.DataFrame, models: dict[str, MarginModel]) -> list[pd.DataFrame]:
    available = [column for column in _PREDICTION_PASSTHROUGH if column in weekly_games.columns]
    batches: list[pd.DataFrame] = []
    for method, model in models.items():
        batch = weekly_games.loc[:, available].copy()
        forecasts = model.predict(weekly_games)
        for column in forecasts:
            batch[column] = forecasts[column]
        batch["method"] = method
        batch["model_name"] = model.model_name
        batch["train_rows"] = model.training_rows
        batch["distribution_rows"] = model.distribution_rows
        batch["train_max_gameday"] = model.training_max_gameday
        batch["bet_side"] = "PASS"
        batch["bet_odds"] = np.nan
        batches.append(batch)
    return batches


@dataclass(frozen=True)
class CrossLeagueTransferResult:
    predictions: pd.DataFrame
    shrinkage: ShrinkageDerivation
    mismatch: LeagueMismatchReport
    diagnostics: dict[str, Any] = field(default_factory=dict)


def cross_league_transfer_benchmark(
    target: pd.DataFrame,
    auxiliary: pd.DataFrame,
    *,
    start_season: int,
    end_season: int,
    shrinkage_cutoff: pd.Timestamp | None = None,
    min_train_games: int = CROSS_LEAGUE_MIN_TRAIN_GAMES,
    feature_columns: tuple[str, ...] = ALIGNED_TRANSFER_FEATURE_COLUMNS,
    ridge_alpha: float = CROSS_LEAGUE_RIDGE_ALPHA,
    include_market: bool = True,
    shrinkage_samples: int = CROSS_LEAGUE_SHRINKAGE_BOOTSTRAP_SAMPLES,
    shrinkage_seed: int = CROSS_LEAGUE_SHRINKAGE_BOOTSTRAP_SEED,
) -> CrossLeagueTransferResult:
    """Walk ``target`` forward, scoring target-only, joint, hierarchical, and prior-mean arms.

    League-agnostic by construction: ``target``/``auxiliary`` may be any two
    leak-safely-dated frames sharing ``feature_columns``. Point-in-time
    contract, identical to every other walk-forward in this project: each
    week's training (both leagues) is every completed game strictly before
    that week's earliest kickoff.

    The hierarchical arm's shrinkage WEIGHTS are derived once from data
    strictly before ``shrinkage_cutoff`` (default: the test window's own
    start) and then held fixed through the walk; the two anchor estimates
    being blended keep updating every week as training data expands. This
    separates "how much to trust the target league's own data" (derived
    once) from "what the target league's own data currently says" (updated
    walking forward), and keeps the derivation itself a single one-time cost
    instead of a bootstrap repeated every week.
    """

    if end_season < start_season:
        raise ValueError("end_season cannot be earlier than start_season")
    _require_transfer_columns(target, "target", feature_columns)
    _require_transfer_columns(auxiliary, "auxiliary", feature_columns)

    target = target.copy()
    auxiliary = auxiliary.copy()
    target["gameday"] = pd.to_datetime(target["gameday"], errors="raise")
    auxiliary["gameday"] = pd.to_datetime(auxiliary["gameday"], errors="raise")
    target_completed = (
        target.loc[
            pd.to_numeric(target["result"], errors="coerce").notna()
            & pd.to_numeric(target["ats_margin"], errors="coerce").notna()
        ]
        .sort_values(["gameday", "game_id"])
        .reset_index(drop=True)
    )
    auxiliary_completed = (
        auxiliary.loc[
            pd.to_numeric(auxiliary["result"], errors="coerce").notna()
            & pd.to_numeric(auxiliary["ats_margin"], errors="coerce").notna()
        ]
        .sort_values(["gameday", "game_id"])
        .reset_index(drop=True)
    )

    test = target_completed.loc[target_completed["season"].between(start_season, end_season)]
    if test.empty:
        raise ValueError(f"No completed target games found from {start_season} to {end_season}")

    if shrinkage_cutoff is None:
        shrinkage_cutoff = test["gameday"].min()
    shrinkage_target_pool = target_completed.loc[target_completed["gameday"].lt(shrinkage_cutoff)]
    shrinkage_aux_pool = auxiliary_completed.loc[
        auxiliary_completed["gameday"].lt(shrinkage_cutoff)
    ]
    if len(shrinkage_target_pool) < min_train_games or len(shrinkage_aux_pool) < min_train_games:
        raise ValueError("Not enough pre-window history to derive shrinkage weights")
    imputer, scaler = fit_pooled_preprocessor(
        shrinkage_target_pool, shrinkage_aux_pool, feature_columns
    )
    mismatch = measure_league_mismatch(
        shrinkage_target_pool,
        shrinkage_aux_pool,
        feature_columns,
        ridge_alpha,
        label_a="target",
        label_b="auxiliary",
    )
    shrinkage = derive_shrinkage_weights(
        shrinkage_target_pool,
        shrinkage_aux_pool,
        imputer,
        scaler,
        feature_columns,
        ridge_alpha,
        samples=shrinkage_samples,
        seed=shrinkage_seed,
    )

    batches: list[pd.DataFrame] = []
    for (_, _), weekly_games in test.groupby(["season", "week"], sort=True):
        cutoff = weekly_games["gameday"].min()
        target_training = target_completed.loc[target_completed["gameday"].lt(cutoff)]
        auxiliary_training = auxiliary_completed.loc[auxiliary_completed["gameday"].lt(cutoff)]
        if len(target_training) < min_train_games or len(auxiliary_training) < min_train_games:
            continue
        models: dict[str, MarginModel] = {
            "target_only": fit_target_only_model(
                target_training, imputer, scaler, feature_columns, ridge_alpha
            ),
            "joint": fit_joint_league_model(
                target_training, auxiliary_training, imputer, scaler, feature_columns, ridge_alpha
            ),
            "hierarchical": fit_hierarchical_shrinkage_model(
                target_training,
                auxiliary_training,
                imputer,
                scaler,
                feature_columns,
                ridge_alpha,
                shrinkage.weights,
            ),
            "prior_mean": fit_prior_mean_ridge_model(
                target_training, auxiliary_training, imputer, scaler, feature_columns, ridge_alpha
            ),
        }
        if include_market:
            models["market"] = fit_market_baseline(target_training)
        batches.extend(_score_week(weekly_games, models))
    if not batches:
        raise ValueError("No target week had enough prior training games in both leagues")

    predictions = pd.concat(batches, ignore_index=True).sort_values(
        ["gameday", "game_id", "method"]
    )
    diagnostics = {
        "start_season": start_season,
        "end_season": end_season,
        "min_train_games": min_train_games,
        "ridge_alpha": ridge_alpha,
        "feature_columns": list(feature_columns),
        "shrinkage_cutoff": pd.Timestamp(shrinkage_cutoff).date().isoformat(),
        "shrinkage_target_pool_rows": len(shrinkage_target_pool),
        "shrinkage_aux_pool_rows": len(shrinkage_aux_pool),
        "tau_squared": shrinkage.tau_squared,
        "shrinkage_weights_mean": float(np.mean(shrinkage.weights)),
        "shrinkage_weights_min": float(np.min(shrinkage.weights)),
        "shrinkage_weights_max": float(np.max(shrinkage.weights)),
    }
    return CrossLeagueTransferResult(
        predictions=predictions.reset_index(drop=True),
        shrinkage=shrinkage,
        mismatch=mismatch,
        diagnostics=diagnostics,
    )
