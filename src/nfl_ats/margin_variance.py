"""MOD-16 conditional margin variance — the CFB benchmark screen.

The active margin models center ONE pooled out-of-time residual sample on
every game's predicted margin, so every game gets the same distribution
*shape*. MOD-16 asks whether pregame context (mismatch size, expected
scoring, pace, early-season sample thinness) predicts per-game residual
*scale*, and whether conditioning on it produces better-calibrated
cover/push/loss probabilities than the pooled distribution.

Per the roadmap this is screened on the CFB benchmark first (12,500 games
resolve calibration differences the NFL sample cannot), with a frozen
predeclaration in ``docs/margin_variance.md``. The mean model — and
therefore every forced pick — is byte-identical to the frozen XLG-03
``market_residual`` arm; ONLY the residual distribution's per-game scale
changes. Acceptance is about probability calibration, not accuracy.

Recipe (frozen)
---------------
1. Fit the frozen pooled arm exactly as XLG-03 does. Reconstruct its own
   chronological trailing-20% holdout, and on those held-out rows fit a
   Ridge (alpha 10, same standardize+impute pipeline) predicting
   ``log(|residual| + 1)`` from the frozen variance feature list.
2. At prediction time, the per-game scale ratio is
   ``r = (exp(f(x)) - 1) / s_bar`` — where ``s_bar`` is the holdout's
   baseline expected absolute residual on the same log scale — clipped to
   the frozen band [2/3, 3/2]. The predictive sample is
   ``center + pooled_residuals * r``; ``r = 1`` recovers the pooled arm.
3. Walk-forward three arms on identical weeks (market, pooled
   market_residual, heteroskedastic market_residual_variance) and report
   paired per-game Brier/log-loss improvements of the variance arm over
   the pooled arm with week- and season-blocked intervals on the clean
   core (``paired_feature_comparisons``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.pipeline import Pipeline

from nfl_ats.cfb_benchmark import (
    _PREDICTION_PASSTHROUGH,
    CFB_BENCHMARK_END_SEASON,
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_RIDGE_ALPHA,
    CFB_BENCHMARK_START_SEASON,
    cfb_evaluation_window,
    fit_cfb_residual_model,
    summarize_cfb_benchmark,
)
from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS
from nfl_ats.data import DataContractError, require_columns
from nfl_ats.experiments import paired_feature_comparisons
from nfl_ats.margin import (
    MarginModel,
    _smoothed_probability,
    _three_way_probabilities,
    fit_market_baseline,
    make_margin_estimator,
)

# ---------------------------------------------------------------------------
# Frozen configuration (see docs/margin_variance.md; fixed before the run)
# ---------------------------------------------------------------------------

# Pregame context believed to move margin dispersion: mismatch size,
# expected scoring, pace, early-season sample thinness, and seasonal phase.
# All are existing canonical CFB columns except abs_spread_line, which is
# derived (|spread_line|) so the variance model sees mismatch magnitude
# rather than home/away sign.
CFB_VARIANCE_FEATURE_COLUMNS: tuple[str, ...] = (
    "abs_spread_line",
    "total_line",
    "home_off_plays_per_game",
    "away_off_plays_per_game",
    "home_team_games",
    "away_team_games",
    "week_sin",
    "week_cos",
)

# The per-game scale ratio is clipped to this band (symmetric in log space);
# the pooled arm is the special case ratio = 1.
VARIANCE_RATIO_FLOOR: float = 2.0 / 3.0
VARIANCE_RATIO_CEILING: float = 1.5

VARIANCE_BENCHMARK_BASELINE_ARM = "cfb_benchmark_v1"
VARIANCE_BENCHMARK_CANDIDATE_ARM = "cfb_benchmark_v1_variance"


def add_variance_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the variance-model inputs that are not already canonical columns."""

    require_columns(frame, ("spread_line",), "variance features")
    result = frame.copy()
    result["abs_spread_line"] = pd.to_numeric(result["spread_line"], errors="coerce").abs()
    return result


@dataclass(frozen=True)
class VarianceModel:
    """A fitted per-game residual-scale model plus its pooled baseline scale."""

    estimator: Pipeline
    s_bar: float
    holdout_rows: int

    def scale_ratio(self, frame: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Clipped per-game scale ratios (1.0 = the pooled distribution)."""

        missing = sorted(set(CFB_VARIANCE_FEATURE_COLUMNS).difference(frame.columns))
        if missing:
            raise DataContractError(f"Variance scoring is missing features: {', '.join(missing)}")
        predicted = np.asarray(
            self.estimator.predict(frame.loc[:, list(CFB_VARIANCE_FEATURE_COLUMNS)]),
            dtype=float,
        )
        expected_abs = np.exp(predicted) - 1.0
        ratio = expected_abs / self.s_bar
        return np.clip(ratio, VARIANCE_RATIO_FLOOR, VARIANCE_RATIO_CEILING)


def fit_cfb_variance_model(
    training: pd.DataFrame,
    *,
    ridge_alpha: float = CFB_BENCHMARK_RIDGE_ALPHA,
    distribution_fraction: float = 0.20,
    random_state: int = 42,
) -> VarianceModel:
    """Fit the residual-scale model on the frozen recipe's own holdout.

    Mirrors ``fit_cfb_residual_model``'s chronological split exactly (same
    sort keys, same trailing fraction, same estimator recipe and seed): the
    first 80% fits a temporary mean model, the trailing 20% supplies
    out-of-time residuals, and the variance Ridge is fit on those held-out
    rows with target ``log(|residual| + 1)``. ``s_bar`` is the baseline
    expected absolute residual on the same log scale, so a game whose
    features look exactly average gets ratio ~= 1.
    """

    required = {"game_id", "gameday", "ats_margin", *CFB_MODEL_FEATURE_COLUMNS}
    missing = sorted(required.difference(training.columns))
    if missing:
        raise DataContractError(f"Variance training is missing columns: {', '.join(missing)}")

    frame = training.loc[pd.to_numeric(training["ats_margin"], errors="coerce").notna()].copy()
    frame = add_variance_features(frame)
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    frame = frame.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    if len(frame) < 50:
        raise ValueError("At least 50 completed games are required for a variance model")
    distribution_rows = int(len(frame) * distribution_fraction)
    if distribution_rows < 10 or len(frame) - distribution_rows < 40:
        raise ValueError("Not enough rows for an out-of-time variance holdout")

    split = len(frame) - distribution_rows
    target = pd.to_numeric(frame["ats_margin"], errors="raise").to_numpy(dtype=float)
    temporary = make_margin_estimator("ridge", random_state, ridge_alpha=ridge_alpha)
    temporary.fit(frame.iloc[:split].loc[:, list(CFB_MODEL_FEATURE_COLUMNS)], target[:split])
    holdout = frame.iloc[split:]
    predicted = np.asarray(
        temporary.predict(holdout.loc[:, list(CFB_MODEL_FEATURE_COLUMNS)]), dtype=float
    )
    residuals = target[split:] - predicted
    finite = np.isfinite(residuals)
    holdout = holdout.loc[finite]
    residuals = residuals[finite]
    if len(residuals) < 10:
        raise ValueError("Variance holdout has too few finite residuals")

    log_abs = np.log(np.abs(residuals) + 1.0)
    estimator = make_margin_estimator("ridge", random_state, ridge_alpha=ridge_alpha)
    estimator.fit(holdout.loc[:, list(CFB_VARIANCE_FEATURE_COLUMNS)], log_abs)
    s_bar = float(np.exp(np.mean(log_abs)) - 1.0)
    if not np.isfinite(s_bar) or s_bar <= 0.0:
        raise ValueError("Variance baseline scale is degenerate")
    return VarianceModel(estimator=estimator, s_bar=s_bar, holdout_rows=len(residuals))


@dataclass(frozen=True)
class HeteroskedasticMarginModel:
    """The pooled model's center with a per-game scaled residual sample."""

    pooled: MarginModel
    variance: VarianceModel

    @property
    def model_name(self) -> str:
        return f"{self.pooled.model_name}_variance"

    @property
    def training_rows(self) -> int:
        return self.pooled.training_rows

    @property
    def distribution_rows(self) -> int:
        return self.pooled.distribution_rows

    @property
    def training_max_gameday(self) -> str:
        return self.pooled.training_max_gameday

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The pooled prediction schema with per-game scaled distributions.

        ``predicted_margin``/``fair_spread``/``predicted_market_residual``
        are copied from the pooled model unchanged (same center, same
        forced pick); every distribution-derived quantity is recomputed
        from ``center + pooled_residuals * ratio``.
        """

        scored = add_variance_features(frame)
        pooled_frame = self.pooled.predict(frame)
        ratios = self.variance.scale_ratio(scored)
        spread = pd.to_numeric(frame["spread_line"], errors="coerce").to_numpy(dtype=float)
        centers = pooled_frame["predicted_margin"].to_numpy(dtype=float)

        result = pooled_frame.copy()
        result["variance_scale_ratio"] = ratios
        win: list[float] = []
        cover: list[float] = []
        cover_excluding_push: list[float] = []
        push: list[float] = []
        loss: list[float] = []
        lower_50: list[float] = []
        upper_50: list[float] = []
        lower_80: list[float] = []
        upper_80: list[float] = []
        for center, line, ratio in zip(centers, spread, ratios, strict=True):
            distribution = np.asarray(center + self.pooled.residuals * ratio, dtype=np.float64)
            win.append(_smoothed_probability(distribution, 0.0))
            cover.append(_smoothed_probability(distribution, float(line)))
            three_way = _three_way_probabilities(distribution, float(line))
            cover_excluding_push.append(three_way[0])
            push.append(three_way[1])
            loss.append(three_way[2])
            quantiles = np.quantile(distribution, [0.10, 0.25, 0.75, 0.90])
            lower_80.append(float(quantiles[0]))
            lower_50.append(float(quantiles[1]))
            upper_50.append(float(quantiles[2]))
            upper_80.append(float(quantiles[3]))
        result["home_win_probability"] = win
        result["home_cover_probability"] = cover
        result["home_cover_probability_excluding_push"] = cover_excluding_push
        result["push_probability"] = push
        result["home_loss_probability"] = loss
        result["margin_lower_50"] = lower_50
        result["margin_upper_50"] = upper_50
        result["margin_lower_80"] = lower_80
        result["margin_upper_80"] = upper_80
        return result


@dataclass(frozen=True)
class CfbVarianceBenchmarkResult:
    predictions: pd.DataFrame
    summary: pd.DataFrame
    season_summary: pd.DataFrame
    paired: pd.DataFrame
    scale_ratio_summary: dict[str, float]


def cfb_variance_benchmark(
    features: pd.DataFrame,
    *,
    start_season: int = CFB_BENCHMARK_START_SEASON,
    end_season: int = CFB_BENCHMARK_END_SEASON,
    min_train_games: int = CFB_BENCHMARK_MIN_TRAIN_GAMES,
    ridge_alpha: float = CFB_BENCHMARK_RIDGE_ALPHA,
    bootstrap_samples: int = 2_000,
    bootstrap_seed: int = 20260817,
) -> CfbVarianceBenchmarkResult:
    """Walk-forward the pooled and heteroskedastic arms on identical weeks."""

    required = {
        *_PREDICTION_PASSTHROUGH,
        *CFB_MODEL_FEATURE_COLUMNS,
        "total_line",
    }
    missing = sorted(required.difference(features.columns))
    if missing:
        raise DataContractError(f"CFB variance features are missing columns: {', '.join(missing)}")

    frame = features.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    completed = frame.loc[
        pd.to_numeric(frame["result"], errors="coerce").notna()
        & pd.to_numeric(frame["ats_margin"], errors="coerce").notna()
    ].copy()
    completed = completed.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    test = completed.loc[completed["season"].between(start_season, end_season)]
    if test.empty:
        raise ValueError(f"No completed CFB games found from {start_season} to {end_season}")

    batches: list[pd.DataFrame] = []
    ratio_samples: list[npt.NDArray[np.float64]] = []
    for (_, _), weekly_games in test.groupby(["season", "week"], sort=True):
        cutoff = weekly_games["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < min_train_games:
            continue
        pooled = fit_cfb_residual_model(training, ridge_alpha=ridge_alpha)
        variance = fit_cfb_variance_model(training, ridge_alpha=ridge_alpha)
        models: dict[str, MarginModel | HeteroskedasticMarginModel] = {
            "market": fit_market_baseline(training),
            "market_residual": pooled,
            "market_residual_variance": HeteroskedasticMarginModel(
                pooled=pooled, variance=variance
            ),
        }
        for method, model in models.items():
            batch = weekly_games.loc[:, list(_PREDICTION_PASSTHROUGH)].copy()
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
            if method == "market_residual_variance":
                ratio_samples.append(forecasts["variance_scale_ratio"].to_numpy(dtype=float))
    if not batches:
        raise ValueError("No CFB week had enough prior training games")

    predictions = pd.concat(batches, ignore_index=True)
    predictions["evaluation_window"] = predictions["season"].map(
        lambda season: cfb_evaluation_window(int(season))
    )
    predictions = predictions.sort_values(["gameday", "game_id", "method"]).reset_index(drop=True)
    summary, season_summary = summarize_cfb_benchmark(predictions)

    clean = predictions.loc[
        predictions["evaluation_window"].eq("clean_core")
        & predictions["method"].isin(("market_residual", "market_residual_variance"))
    ].copy()
    if clean.empty:
        raise ValueError("No clean-core predictions available for the paired comparison")
    clean["feature_set"] = clean["method"].map(
        {
            "market_residual": VARIANCE_BENCHMARK_BASELINE_ARM,
            "market_residual_variance": VARIANCE_BENCHMARK_CANDIDATE_ARM,
        }
    )
    paired = pd.concat(
        [
            paired_feature_comparisons(
                clean,
                baseline_feature_set=VARIANCE_BENCHMARK_BASELINE_ARM,
                samples=bootstrap_samples,
                block="week",
                seed=bootstrap_seed,
            ),
            paired_feature_comparisons(
                clean,
                baseline_feature_set=VARIANCE_BENCHMARK_BASELINE_ARM,
                samples=bootstrap_samples,
                block="season",
                seed=bootstrap_seed,
            ),
        ],
        ignore_index=True,
    )
    ratios = np.concatenate(ratio_samples) if ratio_samples else np.empty(0, dtype=np.float64)
    scale_ratio_summary = {
        "n": float(len(ratios)),
        "mean": float(np.mean(ratios)) if len(ratios) else float("nan"),
        "p10": float(np.quantile(ratios, 0.10)) if len(ratios) else float("nan"),
        "p90": float(np.quantile(ratios, 0.90)) if len(ratios) else float("nan"),
        "at_floor": float(np.mean(ratios <= VARIANCE_RATIO_FLOOR + 1e-12))
        if len(ratios)
        else float("nan"),
        "at_ceiling": float(np.mean(ratios >= VARIANCE_RATIO_CEILING - 1e-12))
        if len(ratios)
        else float("nan"),
    }
    return CfbVarianceBenchmarkResult(
        predictions=predictions,
        summary=summary,
        season_summary=season_summary,
        paired=paired,
        scale_ratio_summary=scale_ratio_summary,
    )
