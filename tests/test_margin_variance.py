from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.cfb_benchmark import fit_cfb_residual_model
from nfl_ats.data import DataContractError
from nfl_ats.margin_variance import (
    VARIANCE_BENCHMARK_BASELINE_ARM,
    VARIANCE_BENCHMARK_CANDIDATE_ARM,
    VARIANCE_RATIO_CEILING,
    VARIANCE_RATIO_FLOOR,
    HeteroskedasticMarginModel,
    VarianceModel,
    add_variance_features,
    cfb_variance_benchmark,
    fit_cfb_variance_model,
)

# ---------------------------------------------------------------------------
# 1. Feature derivation and the variance fit
# ---------------------------------------------------------------------------


def test_add_variance_features_derives_absolute_spread() -> None:
    frame = pd.DataFrame({"spread_line": [-7.5, 0.0, 3.0]})
    result = add_variance_features(frame)
    assert result["abs_spread_line"].tolist() == pytest.approx([7.5, 0.0, 3.0])


def test_fit_cfb_variance_model_produces_bounded_ratios(
    cfb_features_frame: pd.DataFrame,
) -> None:
    variance = fit_cfb_variance_model(cfb_features_frame)
    assert variance.s_bar > 0.0
    assert variance.holdout_rows >= 10

    scored = add_variance_features(cfb_features_frame)
    ratios = variance.scale_ratio(scored)
    assert np.isfinite(ratios).all()
    assert (ratios >= VARIANCE_RATIO_FLOOR - 1e-12).all()
    assert (ratios <= VARIANCE_RATIO_CEILING + 1e-12).all()
    # The fixture has real feature variation, so the ratios must not be one
    # constant -- that is the pooled model, not a conditional one.
    assert len(np.unique(np.round(ratios, 6))) > 1


def test_variance_model_requires_features(cfb_features_frame: pd.DataFrame) -> None:
    variance = fit_cfb_variance_model(cfb_features_frame)
    with pytest.raises(DataContractError, match="missing features"):
        variance.scale_ratio(cfb_features_frame.drop(columns=["total_line"]))


# ---------------------------------------------------------------------------
# 2. The heteroskedastic wrapper: same center, scaled distribution
# ---------------------------------------------------------------------------


class _ConstantEstimator:
    """Predicts a constant log-scale, forcing every ratio to exactly 1."""

    def __init__(self, constant: float) -> None:
        self._constant = constant

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self._constant, dtype=float)


def test_hetero_model_keeps_center_and_recovers_pooled_at_ratio_one(
    cfb_features_frame: pd.DataFrame,
) -> None:
    pooled = fit_cfb_residual_model(cfb_features_frame)
    variance = fit_cfb_variance_model(cfb_features_frame)
    hetero = HeteroskedasticMarginModel(pooled=pooled, variance=variance)

    scored = cfb_features_frame.tail(40)
    pooled_predictions = pooled.predict(scored)
    hetero_predictions = hetero.predict(scored)

    # The center (and therefore the forced pick direction of the mean) is
    # byte-identical; only distribution-derived quantities may move.
    for column in ("predicted_margin", "fair_spread", "predicted_market_residual"):
        assert hetero_predictions[column].tolist() == pooled_predictions[column].tolist()
    assert "variance_scale_ratio" in hetero_predictions.columns

    # With the estimator pinned to the baseline scale, every ratio is 1 and
    # the pooled probabilities are recovered exactly.
    neutral = VarianceModel(
        estimator=_ConstantEstimator(float(np.log(variance.s_bar + 1.0))),  # type: ignore[arg-type]
        s_bar=variance.s_bar,
        holdout_rows=variance.holdout_rows,
    )
    recovered = HeteroskedasticMarginModel(pooled=pooled, variance=neutral).predict(scored)
    assert recovered["variance_scale_ratio"].tolist() == pytest.approx([1.0] * len(scored))
    assert recovered["home_cover_probability"].tolist() == pytest.approx(
        pooled_predictions["home_cover_probability"].tolist()
    )
    assert recovered["margin_lower_80"].tolist() == pytest.approx(
        pooled_predictions["margin_lower_80"].tolist()
    )


# ---------------------------------------------------------------------------
# 3. The two-arm benchmark run
# ---------------------------------------------------------------------------


def test_cfb_variance_benchmark_matched_arms(cfb_features_frame: pd.DataFrame) -> None:
    result = cfb_variance_benchmark(
        cfb_features_frame,
        start_season=2014,
        end_season=2014,
        min_train_games=50,
        bootstrap_samples=25,
    )
    methods = set(result.predictions["method"])
    assert methods == {"market", "market_residual", "market_residual_variance"}

    per_method = result.predictions.groupby("method")["game_id"].apply(set)
    assert per_method["market_residual_variance"] == per_method["market_residual"]

    # Identical centers game-for-game: the variance arm changes only the
    # distribution, never the forced pick's mean.
    pooled_rows = result.predictions.loc[
        result.predictions["method"].eq("market_residual")
    ].set_index("game_id")
    variance_rows = result.predictions.loc[
        result.predictions["method"].eq("market_residual_variance")
    ].set_index("game_id")
    assert variance_rows["predicted_margin"].to_dict() == pytest.approx(
        pooled_rows["predicted_margin"].to_dict()
    )

    paired = result.paired
    assert set(paired["block"]) == {"week", "season"}
    assert set(paired["baseline_feature_set"]) == {VARIANCE_BENCHMARK_BASELINE_ARM}
    assert set(paired["candidate_feature_set"]) == {VARIANCE_BENCHMARK_CANDIDATE_ARM}
    assert {"brier_improvement", "log_loss_improvement"}.issubset(set(paired["metric"]))

    ratios = result.scale_ratio_summary
    assert ratios["n"] > 0
    assert VARIANCE_RATIO_FLOOR <= ratios["mean"] <= VARIANCE_RATIO_CEILING


def test_cfb_variance_benchmark_requires_columns(cfb_features_frame: pd.DataFrame) -> None:
    with pytest.raises(DataContractError, match="missing columns"):
        cfb_variance_benchmark(cfb_features_frame.drop(columns=["total_line"]))
