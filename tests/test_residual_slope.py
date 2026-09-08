"""MOD-18 lane R: the predeclared slope estimator and its centre-shift algebra.

Pins docs/residual_slope.md's four load-bearing claims on synthetic frames:
the least-squares slope, the 100-game shrinkage toward the incumbent, the
``(beta - 1) * residual + offset`` centre shift that serves
``line + beta * residual + offset`` through ``MarginModel.predict``, and that
every ``beta = 1`` reproduces the served S3 read bit-for-bit.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from residual_slope_opener_eval import (
    INCUMBENT_SLOPE,
    center_shift,
    fit_slopes,
    ols_slope,
    shrink,
)

from nfl_ats.home_side_location import PRIOR_WEIGHT_GAMES, prior_rows_before
from nfl_ats.margin import fit_margin_model, margin_feature_columns


def prior_frame(rows: list[tuple[float, float, float]], *, season: int = 2025) -> pd.DataFrame:
    """``(line, residual, y)`` triples as the archive stream the lane fits on."""

    lines = np.array([row[0] for row in rows], dtype=float)
    residual = np.array([row[1] for row in rows], dtype=float)
    y = np.array([row[2] for row in rows], dtype=float)
    return pd.DataFrame(
        {
            "season": season,
            "week": np.arange(1, len(rows) + 1) % 18 + 1,
            "spread_line": lines,
            "point_incumbent": lines + residual,
            "result": lines + y,
        }
    )


def test_least_squares_slope_recovers_a_planted_slope_with_an_intercept():
    x = np.array([-3.0, -1.5, 0.0, 2.0, 4.5, 6.0])
    assert ols_slope(x, 4.0 + 2.5 * x) == pytest.approx(2.5)
    assert ols_slope(x, -7.0 - 0.75 * x) == pytest.approx(-0.75)
    # Nothing to estimate from serves the incumbent, never a fabricated slope.
    assert ols_slope(np.array([1.0, 2.0]), np.array([5.0, 9.0])) == INCUMBENT_SLOPE
    assert ols_slope(np.full(20, 2.0), np.arange(20.0)) == INCUMBENT_SLOPE


def test_shrinkage_is_the_declared_hundred_game_prior_toward_one():
    assert PRIOR_WEIGHT_GAMES == 100.0
    assert shrink(0.0, 100) == pytest.approx(0.5)
    assert shrink(-1.0, 100) == pytest.approx(0.0)
    assert shrink(3.0, 0) == pytest.approx(INCUMBENT_SLOPE)
    for slope, games in ((0.53, 468), (-1.17, 113), (0.91, 160)):
        assert shrink(slope, games) == pytest.approx((games * slope + 100.0) / (games + 100.0))
    # Monotone in the sample: more prior games move further off the incumbent.
    assert shrink(0.2, 400) < shrink(0.2, 100) < shrink(0.2, 10) < INCUMBENT_SLOPE


def test_slopes_are_fitted_per_bucket_and_shrunk():
    rows = [(1.0, x, 3.0 + 0.25 * x) for x in (-4.0, -2.0, 0.0, 2.0, 4.0)]
    rows += [(12.0, x, -2.0 - 1.5 * x) for x in (-6.0, -3.0, 0.0, 3.0, 6.0, 9.0)]
    fit = fit_slopes(prior_frame(rows), pooled=False)
    assert fit.games["0-3"] == 5
    assert fit.games["10.5+"] == 6
    assert fit.raw["0-3"] == pytest.approx(0.25)
    assert fit.raw["10.5+"] == pytest.approx(-1.5)
    assert fit.betas["0-3"] == pytest.approx((5 * 0.25 + 100.0) / 105.0)
    assert fit.betas["10.5+"] == pytest.approx((6 * -1.5 + 100.0) / 106.0)
    # An empty bucket keeps the incumbent exactly, so S3 is served there.
    assert fit.raw["7"] == INCUMBENT_SLOPE
    assert fit.betas["7"] == INCUMBENT_SLOPE
    assert fit.games["7"] == 0


def test_r1b_pools_one_slope_across_the_three_big_buckets():
    rows = [(1.0, x, 3.0 + 0.25 * x) for x in (-4.0, -2.0, 0.0, 2.0, 4.0)]
    big = [(7.0, -3.0, 6.0), (9.0, 1.0, -2.0), (12.0, 4.0, -8.0), (14.0, -2.0, 4.0)]
    rows += big
    fit = fit_slopes(prior_frame(rows), pooled=True)
    pooled = ols_slope(np.array([row[1] for row in big]), np.array([row[2] for row in big]))
    for bucket in ("7", "7.5-10", "10.5+"):
        assert fit.raw[bucket] == pytest.approx(pooled)
        assert fit.betas[bucket] == pytest.approx(shrink(pooled, len(big)))
        assert fit.games[bucket] == len(big)
    # The small buckets keep their own R1 slope; only 7+ is pooled.
    assert fit.raw["0-3"] == pytest.approx(0.25)
    assert fit.games["0-3"] == 5


def test_future_and_out_of_window_rows_cannot_reach_the_slope():
    stream = pd.DataFrame(
        {
            "season": [2019, 2020, 2025, 2025, 2026],
            "week": [1, 1, 1, 2, 1],
            "spread_line": [12.0] * 5,
            "point_incumbent": [12.0 + x for x in (5.0, -4.0, 4.0, 5.0, 5.0)],
            "result": [12.0 + y for y in (500.0, 8.0, -8.0, 500.0, 500.0)],
        }
    )
    fitted = fit_slopes(prior_rows_before(stream, 2025, 2), pooled=False)
    # Only the 2020 and 2025 week-1 rows are eligible; two rows cannot be
    # fitted (n < 3), so the incumbent is served rather than a leaked slope.
    assert fitted.games["10.5+"] == 2
    assert fitted.betas["10.5+"] == INCUMBENT_SLOPE
    poisoned = stream.copy()
    poisoned.loc[[0, 3, 4], "result"] = -1e6
    assert fit_slopes(prior_rows_before(poisoned, 2025, 2), pooled=False).games["10.5+"] == 2


def test_center_shift_algebra_and_beta_one_reproduces_s3():
    lines = pd.Series([1.0, 5.0, 7.0, 9.0, 13.0])
    residual = np.array([2.0, -1.0, 0.5, -3.0, -6.0])
    offset = np.array([0.0, 0.0, -0.17, 0.78, 1.92])
    betas = {"0-3": 0.2, "3.5-6.5": 0.64, "7": 0.78, "7.5-10": 0.95, "10.5+": -0.15}
    shift = center_shift(betas, lines, residual, offset)
    beta = np.array([betas[b] for b in ("0-3", "3.5-6.5", "7", "7.5-10", "10.5+")])
    np.testing.assert_allclose(shift, (beta - 1.0) * residual + offset)
    # The served point is line + beta * residual + offset.
    np.testing.assert_allclose(
        lines.to_numpy() + residual + shift, lines.to_numpy() + beta * residual + offset
    )
    incumbent = center_shift(dict.fromkeys(betas, INCUMBENT_SLOPE), lines, residual, offset)
    np.testing.assert_array_equal(incumbent, offset)
    # A line with no resolvable bucket keeps the incumbent weight.
    missing = center_shift(betas, pd.Series([np.nan]), np.array([4.0]), np.array([0.5]))
    np.testing.assert_allclose(missing, [0.5])


def synthetic_margin_frame(rows: int = 140) -> pd.DataFrame:
    columns = margin_feature_columns("market_residual", "base")
    rng = np.random.default_rng(20260817)
    frame = pd.DataFrame({column: rng.normal(size=rows) for column in columns})
    frame["spread_line"] = rng.choice([-13.0, -6.0, -1.0, 2.5, 7.0, 9.5, 12.0], size=rows)
    frame["neutral_site"] = 0
    frame["div_game"] = rng.integers(0, 2, size=rows)
    frame["game_id"] = [f"g{index:03d}" for index in range(rows)]
    frame["game_type"] = "REG"
    frame["gameday"] = pd.date_range("2020-09-10", periods=rows, freq="D")
    frame["ats_margin"] = 3.0 * frame["elo_diff"] + rng.normal(scale=10.0, size=rows)
    frame["result"] = frame["ats_margin"] + frame["spread_line"]
    return frame


def test_predict_serves_line_plus_beta_residual_plus_offset():
    frame = synthetic_margin_frame()
    model = fit_margin_model(
        frame, target="market_residual", model_name="ridge", feature_profile="base"
    )
    scoring = frame.head(5).reset_index(drop=True)
    scoring["spread_line"] = [1.0, 5.0, 7.0, 9.0, 13.0]
    raw = model.predict(scoring, probability_method="gaussian_median")
    residual = raw.predicted_market_residual.to_numpy()
    offset = np.array([0.0, 0.0, -0.17, 0.78, 1.92])
    betas = {"0-3": 0.2, "3.5-6.5": 0.64, "7": 0.78, "7.5-10": 0.95, "10.5+": -0.15}
    beta = np.array([betas[b] for b in ("0-3", "3.5-6.5", "7", "7.5-10", "10.5+")])
    shift = center_shift(betas, scoring.spread_line, residual, offset)
    served = model.predict(scoring, probability_method="gaussian_median", center_offset=shift)
    np.testing.assert_allclose(
        served.predicted_market_residual.to_numpy(), beta * residual + offset, atol=1e-12
    )
    np.testing.assert_allclose(
        served.predicted_margin.to_numpy(),
        scoring.spread_line.to_numpy() + beta * residual + offset,
        atol=1e-12,
    )
    # beta = 1 everywhere IS the served S3 read, not an approximation of it.
    incumbent = model.predict(
        scoring,
        probability_method="gaussian_median",
        center_offset=center_shift(
            dict.fromkeys(betas, INCUMBENT_SLOPE), scoring.spread_line, residual, offset
        ),
    )
    expected = model.predict(scoring, probability_method="gaussian_median", center_offset=offset)
    pd.testing.assert_frame_equal(incumbent, expected)
