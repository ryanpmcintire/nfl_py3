"""MOD-18 lane U: the derived shrinkage and its reduction to the incumbent.

Pins docs/residual_slope_shrunk.md's load-bearing claims on synthetic frames:
the least-squares standard error, the fixed-target DerSimonian-Laird moment
estimator of ``tau^2``, the empirical-Bayes weight
``1 + (beta_hat - 1) * tau^2 / (tau^2 + se^2)``, that ``tau^2 = 0`` gives
``beta = 1`` in every bucket and therefore reproduces S3 exactly (not
approximately), and that R2b moves only the ``10.5+`` bucket.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from residual_slope_opener_eval import INCUMBENT_SLOPE, center_shift, shrink
from residual_slope_shrunk_opener_eval import (
    MIN_SLOPE_ROWS,
    RESTRICTED_BUCKET,
    empirical_bayes,
    fit_shrunk,
    shrinkage_fraction,
    slope_and_standard_error,
    tau_squared,
)
from test_residual_slope import prior_frame, synthetic_margin_frame

from nfl_ats.margin import fit_margin_model
from nfl_ats.spread_regime import BUCKETS


def test_standard_error_matches_the_textbook_least_squares_formula():
    rng = np.random.default_rng(20260817)
    x = rng.normal(size=40)
    y = 2.0 + 0.75 * x + rng.normal(scale=3.0, size=40)
    slope, standard_error = slope_and_standard_error(x, y)
    design = np.column_stack([np.ones_like(x), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    expected = math.sqrt((residual @ residual) / (x.size - 2) / float(((x - x.mean()) ** 2).sum()))
    assert slope == pytest.approx(coefficients[1])
    assert standard_error == pytest.approx(expected)
    # More rows around the same line pin the slope down harder.
    wide = rng.normal(size=400)
    _, tighter = slope_and_standard_error(wide, 2.0 + 0.75 * wide + rng.normal(scale=3.0, size=400))
    assert tighter < standard_error


def test_a_bucket_that_cannot_support_a_standard_error_serves_the_incumbent():
    # Fewer than four rows, no variation in x, and a perfect (zero-residual)
    # fit all mark the bucket not estimable: nan, so beta stays 1.
    for x, y in (
        (np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0, 5.0])),
        (np.full(20, 2.0), np.arange(20.0)),
        (np.array([1.0, 2.0, 3.0, 4.0]), np.array([2.0, 4.0, 6.0, 8.0])),
    ):
        slope, standard_error = slope_and_standard_error(x, y)
        assert math.isnan(standard_error)
        assert shrinkage_fraction(standard_error, 5.0) == 0.0
        assert empirical_bayes(slope, standard_error, 5.0) == INCUMBENT_SLOPE
    assert MIN_SLOPE_ROWS == 4


def test_tau_squared_is_the_fixed_target_moment_estimator():
    estimates = {"a": 0.2, "b": 1.4, "c": -0.6}
    errors = {"a": 0.3, "b": 0.5, "c": 0.4}
    weights = {k: 1.0 / errors[k] ** 2 for k in estimates}
    q = sum(weights[k] * (estimates[k] - 1.0) ** 2 for k in estimates)
    assert tau_squared(estimates, errors) == pytest.approx((q - 3) / sum(weights.values()))
    # Estimates sitting on the incumbent leave no between-bucket variance, and
    # the estimator is floored at zero rather than going negative.
    assert tau_squared(dict.fromkeys(estimates, 1.0), errors) == 0.0
    # A bucket with no estimable slope drops out of both Q and the count.
    partial = tau_squared(estimates, {**errors, "c": float("nan")})
    pair = {"a": estimates["a"], "b": estimates["b"]}
    assert partial == pytest.approx(tau_squared(pair, errors))
    assert tau_squared(estimates, dict.fromkeys(estimates, float("nan"))) == 0.0


def test_empirical_bayes_keeps_noisy_buckets_on_the_incumbent():
    # Same departure from 1.0, ten times the standard error: the noisy bucket
    # is pulled much closer to the incumbent.
    precise = empirical_bayes(-1.0, 0.10, 0.25)
    noisy = empirical_bayes(-1.0, 1.00, 0.25)
    assert precise == pytest.approx(1.0 + (-2.0) * 0.25 / (0.25 + 0.01))
    assert noisy == pytest.approx(1.0 + (-2.0) * 0.25 / (0.25 + 1.0))
    assert abs(precise - 1.0) > abs(noisy - 1.0)
    # tau^2 = 0 is the incumbent everywhere, for any estimate or error.
    for slope in (-3.0, 0.0, 0.5, 4.0):
        for standard_error in (0.01, 0.5, 9.0):
            assert empirical_bayes(slope, standard_error, 0.0) == INCUMBENT_SLOPE
    # Perfect precision serves the raw estimate.
    assert empirical_bayes(0.4, 1e-12, 0.25) == pytest.approx(0.4, abs=1e-9)


def eb_rows() -> list[tuple[float, float, float]]:
    """Two buckets whose slopes genuinely differ, so ``tau^2`` is positive."""

    rng = np.random.default_rng(20260817)
    small = [(1.0, float(x), float(0.3 * x + rng.normal(scale=1.0))) for x in rng.normal(size=60)]
    big = [(12.0, float(x), float(-1.2 * x + rng.normal(scale=1.0))) for x in rng.normal(size=60)]
    return small + big


def test_r2_fits_every_bucket_from_its_own_precision():
    fit = fit_shrunk(prior_frame(eb_rows()), restricted=False)
    assert fit.tau_squared > 0.0
    assert fit.games["0-3"] == 60
    assert fit.games[RESTRICTED_BUCKET] == 60
    for bucket in ("0-3", RESTRICTED_BUCKET):
        expected = 1.0 + (fit.raw[bucket] - 1.0) * fit.tau_squared / (
            fit.tau_squared + fit.standard_error[bucket] ** 2
        )
        assert fit.betas[bucket] == pytest.approx(expected)
        assert fit.fraction[bucket] == pytest.approx(
            fit.tau_squared / (fit.tau_squared + fit.standard_error[bucket] ** 2)
        )
    # An empty bucket has no slope to shrink and serves the incumbent exactly.
    for bucket in ("7", "7.5-10"):
        assert fit.games[bucket] == 0
        assert fit.betas[bucket] == INCUMBENT_SLOPE
        assert math.isnan(fit.standard_error[bucket])


def test_tau_squared_zero_reproduces_beta_one_in_every_bucket():
    # Every bucket's slope planted exactly on the incumbent, with real scatter
    # about it (the noise is orthogonal to x, so the fit is not degenerate and
    # every standard error is finite): Q collapses, the moment estimator floors
    # at zero and the arm IS S3.
    x = np.arange(-9.0, 10.0)
    noise = 0.4 * (x**2 - (x**2).mean())
    rows = [
        (line, float(xi), float(xi + ei))
        for line in (1.0, 5.0, 7.0, 9.0, 13.0)
        for xi, ei in zip(x, noise, strict=True)
    ]
    fit = fit_shrunk(prior_frame(rows), restricted=False)
    assert fit.tau_squared == 0.0
    for bucket in BUCKETS:
        assert fit.raw[bucket] == pytest.approx(1.0)
        assert math.isfinite(fit.standard_error[bucket])
        assert fit.standard_error[bucket] > 0.0
        assert fit.betas[bucket] == INCUMBENT_SLOPE
        assert fit.fraction[bucket] == 0.0
    lines = pd.Series([1.0, 5.0, 7.0, 9.0, 13.0])
    residual = np.array([2.0, -1.0, 0.5, -3.0, -6.0])
    offset = np.array([0.0, 0.0, -0.17, 0.78, 1.92])
    np.testing.assert_array_equal(center_shift(fit.betas, lines, residual, offset), offset)


def test_r2b_moves_only_the_biggest_bucket_and_keeps_lane_r_shrinkage():
    fit = fit_shrunk(prior_frame(eb_rows()), restricted=True)
    assert math.isnan(fit.tau_squared)
    assert fit.betas[RESTRICTED_BUCKET] == pytest.approx(
        shrink(fit.raw[RESTRICTED_BUCKET], fit.games[RESTRICTED_BUCKET])
    )
    assert fit.fraction[RESTRICTED_BUCKET] == pytest.approx(60 / 160)
    for bucket in BUCKETS:
        if bucket == RESTRICTED_BUCKET:
            continue
        assert fit.betas[bucket] == INCUMBENT_SLOPE
        assert fit.fraction[bucket] == 0.0
    # Only 10.5+ rows can reach the served weight; the small bucket's slope is
    # estimated and reported but never served.
    assert fit.raw["0-3"] != INCUMBENT_SLOPE


def test_future_and_out_of_window_rows_cannot_reach_either_arm():
    stream = pd.DataFrame(
        {
            "season": [2019, 2020, 2020, 2020, 2025, 2025, 2026],
            "week": [1, 1, 2, 3, 1, 2, 1],
            "spread_line": [12.0] * 7,
            "point_incumbent": [12.0 + x for x in (5.0, -4.0, 2.0, 6.0, 4.0, 5.0, 5.0)],
            "result": [12.0 + y for y in (500.0, 8.0, -3.0, 1.0, -8.0, 500.0, 500.0)],
        }
    )
    from nfl_ats.home_side_location import prior_rows_before

    prior = prior_rows_before(stream, 2025, 2)
    for restricted in (False, True):
        fit = fit_shrunk(prior, restricted=restricted)
        assert fit.games[RESTRICTED_BUCKET] == 4  # 2019 and the 2025 wk2/2026 rows excluded
        baseline = fit.betas[RESTRICTED_BUCKET]
        poisoned = stream.copy()
        poisoned.loc[[0, 5, 6], "result"] = -1e6
        assert (
            fit_shrunk(prior_rows_before(poisoned, 2025, 2), restricted=restricted).betas[
                RESTRICTED_BUCKET
            ]
            == baseline
        )


def test_predict_serves_line_plus_shrunk_beta_residual_plus_offset():
    frame = synthetic_margin_frame()
    model = fit_margin_model(
        frame, target="market_residual", model_name="ridge", feature_profile="base"
    )
    scoring = frame.head(5).reset_index(drop=True)
    scoring["spread_line"] = [1.0, 5.0, 7.0, 9.0, 13.0]
    raw = model.predict(scoring, probability_method="gaussian_median")
    residual = raw.predicted_market_residual.to_numpy()
    offset = np.array([0.0, 0.0, -0.17, 0.78, 1.92])
    estimates = {"0-3": 0.05, "3.5-6.5": 0.57, "7": 0.44, "7.5-10": 0.91, "10.5+": -1.17}
    errors = {"0-3": 0.28, "3.5-6.5": 0.33, "7": 0.90, "7.5-10": 0.45, "10.5+": 0.52}
    tau2 = tau_squared(estimates, errors)
    betas = {b: empirical_bayes(estimates[b], errors[b], tau2) for b in estimates}
    beta = np.array([betas[b] for b in ("0-3", "3.5-6.5", "7", "7.5-10", "10.5+")])
    shift = center_shift(betas, scoring.spread_line, residual, offset)
    served = model.predict(scoring, probability_method="gaussian_median", center_offset=shift)
    np.testing.assert_allclose(
        served.predicted_margin.to_numpy(),
        scoring.spread_line.to_numpy() + beta * residual + offset,
        atol=1e-12,
    )
    # tau^2 = 0 IS the served S3 read, not an approximation of it.
    incumbent = model.predict(
        scoring,
        probability_method="gaussian_median",
        center_offset=center_shift(
            {b: empirical_bayes(estimates[b], errors[b], 0.0) for b in estimates},
            scoring.spread_line,
            residual,
            offset,
        ),
    )
    expected = model.predict(scoring, probability_method="gaussian_median", center_offset=offset)
    pd.testing.assert_frame_equal(incumbent, expected)
