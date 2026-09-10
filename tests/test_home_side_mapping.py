from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nfl_ats.calibration import smoothed_home_cover_probability
from nfl_ats.conditional_margin import fit_conditional_margin
from nfl_ats.home_side_mapping import (
    HOME_SIDE_MAPPING_METHODS,
    LATTICE_SIDE_SUPPORT,
    SHIFT_PRIOR_GAMES,
    fit_home_side_shift,
    fit_lattice_home_side,
    home_side,
    predict_home_side_mapping,
    shift_cell,
    shifted_probability,
)


def history(n: int = 1200) -> pd.DataFrame:
    line = np.tile([8.0, -8.0, 3.0], n // 3)
    center = np.tile([2.0, -2.0, 1.0], n // 3)
    result = center + np.tile([-2.0, 6.0, 0.0], n // 3)
    return pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(n)],
            "season": np.repeat([2020, 2021, 2022, 2023], n // 4),
            "week": np.tile(np.repeat(np.arange(1, 16), n // 60), 4),
            "gameday": pd.date_range("2020-01-01", periods=n),
            "spread_line": line,
            "result": result,
            "predicted_margin": center,
            "p_smooth": 0.45,
            "smooth_scale": 13.0,
        }
    )


def test_home_side_labels_and_cells() -> None:
    np.testing.assert_equal(
        home_side([3.5, -7.0, 0.0]), ["home_favourite", "home_underdog", "pickem"]
    )
    cells = shift_cell(pd.Series([8.0, -8.0, 0.0, -3.5]))
    assert list(cells) == [
        "home_favourite/7.5-10",
        "home_underdog/7.5-10",
        "pickem/0-3",
        "home_underdog/3.5-6.5",
    ]


def test_shift_is_shrunken_cell_mean_toward_zero() -> None:
    h = history()
    fitted = fit_home_side_shift(h)
    n = 400
    assert fitted.shifts["home_underdog/7.5-10"] == pytest.approx(6 * n / (n + SHIFT_PRIOR_GAMES))
    assert fitted.shifts["home_favourite/7.5-10"] == pytest.approx(-2 * n / (n + SHIFT_PRIOR_GAMES))
    assert fitted.shifts["home_favourite/0-3"] == pytest.approx(0.0)
    assert fitted.shift([-8.0, 8.0, -10.5, 0.0]).tolist() == pytest.approx(
        [6 * n / (n + 100), -2 * n / (n + 100), 0.0, 0.0]
    )
    assert fitted.rows([-8.0, 0.0]).tolist() == [n, 0]
    small = fit_home_side_shift(h.loc[h.spread_line.eq(-8.0)].head(10))
    assert small.shifts["home_underdog/7.5-10"] == pytest.approx(60 / 110)


def test_shifted_probability_matches_gaussian_at_shifted_centre() -> None:
    centre, line, scale = np.array([2.0]), np.array([-8.0]), 13.0
    p = stats.norm.sf(line - centre, scale=scale)
    shifted = shifted_probability(p, np.array([5.0]), scale)
    np.testing.assert_allclose(shifted, stats.norm.sf(line - (centre + 5.0), scale=scale))
    assert shifted[0] > p[0]
    np.testing.assert_allclose(shifted_probability(p, 0.0, scale), p)


def test_lattice_uses_same_side_when_supported_and_falls_back_otherwise() -> None:
    h = history()
    supported = fit_lattice_home_side(h, -2.0, -8.0)
    assert supported.used_side
    assert supported.side_support >= LATTICE_SIDE_SUPPORT
    same_only = fit_conditional_margin(h.loc[h.spread_line.lt(0)], -2.0, -8.0)
    np.testing.assert_allclose(supported.lattice.mass, same_only.mass)
    assert supported.decision_probability(-8.0) == pytest.approx(
        same_only.decision_probability(-8.0)
    )
    assert set(supported.lattice.margins) == {4.0}
    thin = h.loc[h.spread_line.ge(0) | h.game_id.isin(h.loc[h.spread_line.lt(0)].game_id[:50])]
    fallback = fit_lattice_home_side(thin, -2.0, -8.0)
    assert not fallback.used_side
    np.testing.assert_allclose(fallback.lattice.mass, fit_conditional_margin(thin, -2.0, -8.0).mass)
    pickem = fit_lattice_home_side(h, 1.0, 0.0)
    assert not pickem.used_side and pickem.side_support == 0


@pytest.mark.parametrize("method", HOME_SIDE_MAPPING_METHODS)
def test_future_and_same_week_cannot_move_earlier_shift_or_probability(method: str) -> None:
    h = history()
    target = h.tail(20).copy()
    expected = predict_home_side_mapping(h, target, method=method)
    altered = h.copy()
    altered.loc[altered.game_id.isin(target.game_id), "result"] = -70.0
    same_week = target.head(5).assign(game_id=lambda x: "same" + x.game_id, result=-70.0)
    future = h.head(30).assign(
        game_id=lambda x: "future" + x.game_id,
        season=2027,
        gameday=pd.Timestamp("2027-10-01"),
        result=-70.0,
    )
    got = predict_home_side_mapping(
        pd.concat([altered, same_week, future], ignore_index=True), target, method=method
    )
    columns = ["home_cover_probability", "home_side_history_rows"]
    if method == "smooth_home_side_shift":
        columns += ["home_side_shift", "home_side_shift_rows"]
    else:
        columns += ["push_probability", "lattice_side_used", "lattice_side_support"]
    pd.testing.assert_frame_equal(expected[columns], got[columns])
    moved = h.copy()
    moved.loc[moved.season.eq(2020) & moved.spread_line.eq(-8.0), "result"] = -30.0
    changed = predict_home_side_mapping(moved, target, method=method)
    assert not np.allclose(changed.home_cover_probability, expected.home_cover_probability)


def test_predict_shift_reads_incumbent_probability_and_scale() -> None:
    h = history()
    target = h.tail(3).copy()
    got = predict_home_side_mapping(h, target)
    eligible = h.loc[
        (h.gameday + pd.Timedelta(days=1)).lt(target.gameday.min())
        & ~(h.season.eq(2023) & h.week.eq(int(target.week.iloc[0])))
    ]
    fitted = fit_home_side_shift(eligible)
    expected = shifted_probability(
        target.p_smooth, fitted.shift(target.spread_line.to_numpy()), target.smooth_scale
    )
    np.testing.assert_allclose(got.home_cover_probability, expected)
    assert got.home_side_history_rows.eq(len(eligible)).all()


@pytest.mark.parametrize("method", HOME_SIDE_MAPPING_METHODS)
def test_central_mapper_requires_history_and_matches_module(method: str) -> None:
    residuals = np.linspace(-20, 20, 101)
    with pytest.raises(ValueError, match="history"):
        smoothed_home_cover_probability(residuals, np.array([2.0]), np.array([-8.0]), method=method)
    h = history()
    value = smoothed_home_cover_probability(
        residuals, np.array([2.0]), np.array([-8.0]), method=method, conditional_history=h
    )
    if method == "smooth_home_side_shift":
        shift = fit_home_side_shift(h).shift([-8.0])
        expected = smoothed_home_cover_probability(
            residuals, np.array([2.0]) + shift, np.array([-8.0]), method="gaussian_median"
        )
        assert (
            value[0]
            > smoothed_home_cover_probability(
                residuals, np.array([2.0]), np.array([-8.0]), method="gaussian_median"
            )[0]
        )
    else:
        expected = np.array([fit_lattice_home_side(h, 2.0, -8.0).decision_probability(-8.0)])
    np.testing.assert_allclose(value, expected)


def test_unknown_method_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown home-side"):
        predict_home_side_mapping(history(), history().tail(2), method="bogus")


@pytest.mark.parametrize("command", ["margin-predict", "margin-backtest", "opener-evaluation"])
@pytest.mark.parametrize("method", HOME_SIDE_MAPPING_METHODS)
def test_cli_accepts_additive_method(command: str, method: str) -> None:
    from nfl_ats.cli import build_parser

    args = [command, "--probability-method", method]
    if command == "margin-predict":
        args += ["--season", "2026", "--week", "1"]
    assert build_parser().parse_args(args).probability_method == method
