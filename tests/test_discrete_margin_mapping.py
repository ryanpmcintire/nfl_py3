from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from nfl_ats import weak_signals
from nfl_ats.calibration import (
    RESIDUAL_SMOOTHING_METHODS,
    fit_residual_smoother,
    smoothed_home_cover_probability,
)
from nfl_ats.discrete_margin_mapping import (
    ACCURACY_POINT_CEILING,
    ARMS,
    DISCRETE_MARGIN_METHODS,
    KEY_NUMBERS,
    SERVED_ATOMS,
    apply_arm,
    discrete_conditional_cover_probability,
    discrete_side_read,
    distance_to_nearest_atom,
    floor_degenerate_cell,
    key_neighbourhood_mask,
    plausible_standard_error_floor,
    walk_forward_side_reads,
)
from nfl_ats.key_line_pick_read import key_line_decision_probability, key_line_mask
from nfl_ats.mass_preserving_lattice import band_read, prior_pool_for_week

LINE_GRID = np.array(
    [
        -14.0,
        -10.5,
        -7.5,
        -7.0,
        -6.5,
        -3.5,
        -3.0,
        -2.5,
        -1.0,
        0.0,
        1.5,
        2.5,
        3.0,
        3.5,
        6.5,
        7.0,
        7.5,
        9.5,
        10.0,
        10.5,
        13.5,
        14.0,
    ],
    dtype=float,
)


def _pool(seed: int = 11, rows: int = 900) -> tuple[np.ndarray, np.ndarray]:

    rng = np.random.default_rng(seed)
    lines = rng.choice(np.arange(-14.0, 14.5, 0.5), size=rows)
    margins = np.rint(rng.normal(loc=lines, scale=13.0))
    lumps = rng.random(rows) < 0.22
    margins[lumps] = rng.choice([-7.0, -3.0, 3.0, 7.0], size=int(lumps.sum()))
    return lines, margins


def test_distance_and_neighbourhood_use_the_absolute_line() -> None:
    distance = distance_to_nearest_atom([-3.5, 3.5, -7.0, 5.0], SERVED_ATOMS)
    np.testing.assert_allclose(distance, [0.5, 0.5, 0.0, 2.0])
    mask = key_neighbourhood_mask([-3.5, 3.5, -7.0, 5.0])
    np.testing.assert_array_equal(mask, [True, True, True, False])


def test_half_point_only_is_disjoint_from_the_served_exact_atom_read() -> None:
    adjacent = key_neighbourhood_mask(LINE_GRID, SERVED_ATOMS, half_point_only=True)
    served = key_line_mask(pd.Series(LINE_GRID))
    assert not bool((adjacent & served).any())
    np.testing.assert_array_equal(
        adjacent | served, key_neighbourhood_mask(LINE_GRID, SERVED_ATOMS)
    )
    assert set(np.abs(LINE_GRID[adjacent])) == {2.5, 3.5, 6.5, 7.5}


def test_the_four_arms_are_nested_as_declared() -> None:
    masks = {name: ARMS[name].mask(LINE_GRID) for name in ARMS}
    assert bool((masks["G4"] <= masks["G2"]).all())
    assert bool((masks["G2"] <= masks["G3"]).all())
    assert bool((masks["G3"] <= masks["G1"]).all())
    assert bool(masks["G1"].all())
    assert set(np.abs(LINE_GRID[masks["G3"] & ~masks["G2"]])) == {9.5, 10.0, 10.5, 13.5, 14.0}


def test_key_neighbourhood_covers_every_declared_key_number() -> None:
    mask = key_neighbourhood_mask(KEY_NUMBERS, KEY_NUMBERS)
    assert bool(mask.all())


def test_apply_arm_leaves_untouched_games_bit_identical() -> None:
    rng = np.random.default_rng(3)
    baseline = rng.random(len(LINE_GRID))
    candidate = rng.random(len(LINE_GRID))
    applied = apply_arm(
        LINE_GRID, baseline, np.zeros_like(baseline), candidate, np.ones_like(candidate), "G2"
    )
    untouched = ~applied["touched"]
    np.testing.assert_array_equal(applied["probability"][untouched], baseline[untouched])
    np.testing.assert_array_equal(
        applied["probability"][applied["touched"]], candidate[applied["touched"]]
    )
    np.testing.assert_array_equal(applied["push"][untouched], np.zeros(int(untouched.sum())))


def test_apply_arm_rejects_a_ragged_input() -> None:
    with pytest.raises(ValueError, match="one value per quoted line"):
        apply_arm([3.0, 3.5], np.zeros(2), np.zeros(2), np.zeros(3), np.zeros(3), "G1")


def test_push_falls_out_of_the_same_object_and_is_empty_at_a_half_point() -> None:
    lines, margins = _pool()
    integer = discrete_side_read(lines, margins, 3.0, 1.0)
    half = discrete_side_read(lines, margins, 3.5, 1.0)
    assert integer.push > 0.0
    assert half.push == 0.0
    for read in (integer, half):
        assert read.cover + read.push + read.loss == pytest.approx(1.0, abs=1e-12)
        assert read.home_cover_probability == pytest.approx(read.cover + 0.5 * read.push)


def test_the_read_keeps_key_number_mass_a_gaussian_would_smooth_away() -> None:

    lines, margins = _pool()
    read = discrete_side_read(lines, margins, 3.0, 1.0)
    scale = float(np.std(margins, ddof=1))
    gaussian = sum(
        stats.norm.cdf(key + 0.5, loc=1.0, scale=scale)
        - stats.norm.cdf(key - 0.5, loc=1.0, scale=scale)
        for key in (-3.0, 3.0)
    )
    assert read.key_mass_3 > gaussian


def test_discrete_side_read_is_lane_ks_core_and_lane_ts_decision_number() -> None:
    lines, margins = _pool()
    mine = discrete_side_read(lines, margins, 7.0, 2.5)
    theirs = band_read(lines, margins, 7.0, 2.5)
    assert mine == theirs
    assert key_line_decision_probability(mine) == mine.home_cover_probability


def _target_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["t1", "t2"],
            "season": [2024, 2024],
            "week": [5, 5],
            "gameday": pd.to_datetime(["2024-10-06", "2024-10-06"]),
            "line": [3.0, 3.5],
            "point": [1.0, 1.0],
        }
    )


def _leakage_pool(future_margin: float) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    past = pd.DataFrame(
        {
            "game_id": [f"p{i}" for i in range(400)],
            "season": 2023,
            "week": rng.integers(1, 18, 400),
            "gameday": pd.to_datetime("2023-10-01")
            + pd.to_timedelta(rng.integers(0, 60, 400), "D"),
            "line": rng.choice(np.arange(-10.0, 10.5, 0.5), 400),
            "result": np.rint(rng.normal(0.0, 12.0, 400)),
        }
    )
    future = pd.DataFrame(
        {
            "game_id": ["t1", "t2", *[f"f{i}" for i in range(300)]],
            "season": 2024,
            "week": [5, 5, *([5] * 150), *([6] * 150)],
            "gameday": pd.to_datetime(
                ["2024-10-06", "2024-10-06", *(["2024-10-06"] * 150), *(["2024-10-13"] * 150)]
            ),
            "line": 3.0,
            "result": future_margin,
        }
    )
    return pd.concat([past, future], ignore_index=True)


def test_the_target_week_and_everything_after_it_cannot_move_the_read() -> None:
    honest = walk_forward_side_reads(_leakage_pool(3.0), _target_frame())
    poisoned = walk_forward_side_reads(_leakage_pool(60.0), _target_frame())
    pd.testing.assert_frame_equal(honest, poisoned)


def test_the_window_helper_excludes_the_target_week_and_later_games() -> None:
    pool = _leakage_pool(60.0)
    eligible = prior_pool_for_week(
        pool,
        season=2024,
        week=5,
        cutoff=pd.Timestamp("2024-10-06"),
        exclude_game_ids=("t1", "t2"),
    )
    assert eligible.season.eq(2023).all()
    assert not eligible.game_id.isin({"t1", "t2"}).any()
    assert len(eligible) == 400


def _history() -> pd.DataFrame:
    lines, margins = _pool()
    return pd.DataFrame({"spread_line": lines, "result": margins, "predicted_margin": lines})


def test_the_two_new_methods_are_registered_and_are_not_residual_smoothers() -> None:
    for method in DISCRETE_MARGIN_METHODS:
        assert method in RESIDUAL_SMOOTHING_METHODS
        with pytest.raises(ValueError, match="prior predicted/actual margin pairs"):
            fit_residual_smoother(np.arange(-20.0, 20.0), method)


def test_the_lattice_method_reads_the_served_point_through_band_read() -> None:
    residuals = np.arange(-30.0, 31.0, 1.0)
    history = _history()
    centers = np.array([1.0, -2.0])
    lines = np.array([3.0, 3.5])
    actual = smoothed_home_cover_probability(
        residuals,
        centers,
        lines,
        method="discrete_conditional_lattice",
        conditional_history=history,
    )
    location = float(np.median(residuals))
    expected = [
        band_read(
            history.spread_line.to_numpy(), history.result.to_numpy(), line, center + location
        ).home_cover_probability
        for center, line in zip(centers, lines, strict=True)
    ]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_the_neighbourhood_method_keeps_the_smooth_read_off_the_key_numbers() -> None:
    residuals = np.arange(-30.0, 31.0, 1.0)
    history = _history()
    centers = np.zeros(len(LINE_GRID))
    smooth = smoothed_home_cover_probability(
        residuals, centers, LINE_GRID, method="gaussian_median"
    )
    actual = smoothed_home_cover_probability(
        residuals,
        centers,
        LINE_GRID,
        method="discrete_conditional_key_neighbourhood",
        conditional_history=history,
    )
    touched = key_neighbourhood_mask(LINE_GRID)
    np.testing.assert_array_equal(actual[~touched], smooth[~touched])
    assert bool((actual[touched] != smooth[touched]).any())


def test_the_neighbourhood_method_needs_the_incumbent_read() -> None:
    with pytest.raises(ValueError, match="needs the incumbent smooth probabilities"):
        discrete_conditional_cover_probability(
            np.arange(-30.0, 31.0, 1.0),
            np.zeros(2),
            np.array([3.0, 9.0]),
            _history(),
            method="discrete_conditional_key_neighbourhood",
        )


def test_a_non_integer_prior_margin_is_refused() -> None:
    history = _history()
    history.loc[0, "result"] = 3.5
    with pytest.raises(ValueError, match="Actual margins must be integers"):
        discrete_conditional_cover_probability(
            np.arange(-30.0, 31.0, 1.0), np.zeros(1), np.array([3.0]), history
        )


def test_the_incumbent_smooth_read_is_untouched_by_this_family() -> None:
    residuals = np.array([-9.0, -4.0, -1.0, 0.0, 2.0, 3.0, 5.0, 8.0, 11.0, 15.0])
    centers = np.array([-1.5, 0.0, 4.25])
    lines = np.array([-3.0, 3.5, 7.0])
    actual = smoothed_home_cover_probability(residuals, centers, lines, method="gaussian_median")
    expected = stats.norm.sf(
        lines - centers, loc=float(np.median(residuals)), scale=float(np.std(residuals, ddof=1))
    )
    np.testing.assert_array_equal(actual, np.clip(expected, 1e-9, 1 - 1e-9))


def _reference_family() -> list[tuple[float, int]]:

    sigma = 28.0
    return [(sigma / np.sqrt(n), n) for n in (1503, 758, 745, 342, 266, 179, 156, 70)]


def test_the_floor_tracks_the_familys_own_sigma_over_root_n_curve() -> None:
    floor = plausible_standard_error_floor(_reference_family(), 15)
    assert floor is not None
    assert floor == pytest.approx(28.0 / np.sqrt(15), rel=1e-9)
    bigger = plausible_standard_error_floor(_reference_family(), 1503)
    assert bigger is not None and bigger < floor


def test_the_floor_declines_to_invent_a_band_from_too_thin_a_family() -> None:
    assert plausible_standard_error_floor([(1.0, 100), (2.0, 50)], 15) is None
    assert plausible_standard_error_floor(_reference_family(), 0) is None


def test_an_admissible_cell_is_returned_untouched() -> None:
    metrics = {"delta": -0.333, "lower": -1.3, "upper": 0.6, "standard_error": 0.48, "n": 758}
    floored, note = floor_degenerate_cell(metrics, _reference_family())
    assert note is None
    assert floored == metrics


def test_a_degenerate_cell_keeps_its_point_estimate_and_gains_a_floored_band() -> None:

    metrics = {
        "delta": 100.0,
        "lower": 100.0,
        "upper": 100.0,
        "standard_error": 0.0,
        "n": 15,
        "weeks": 15,
        "probability_positive": 1.0,
    }
    floored, note = floor_degenerate_cell(metrics, _reference_family())
    assert note is not None and "FLOORED" in note
    assert floored["delta"] == 100.0
    assert floored["probability_positive"] == 1.0
    assert floored["n"] == 15 and floored["weeks"] == 15
    assert floored["standard_error"] > 0.0
    assert floored["upper"] == ACCURACY_POINT_CEILING
    assert floored["lower"] == pytest.approx(100.0 - 1.959963984540054 * floored["standard_error"])
    assert floored["lower"] < floored["upper"]


def test_a_floored_cell_satisfies_the_registry_contract(tmp_path: Path) -> None:

    metrics = {"delta": 100.0, "lower": 100.0, "upper": 100.0, "standard_error": 0.0, "n": 15}
    floored, _ = floor_degenerate_cell(metrics, _reference_family())
    path = tmp_path / "weak_signals.json"
    registry = weak_signals.Registry(version=1, notes=(), signals={})
    signal = weak_signals.WeakSignal(
        name="lane_c2_degenerate_fixture",
        recorded_at="2026-09-08",
        description="A degenerate positive-control slice, floored",
        source="tests",
        effect=floored["delta"],
        effect_units="accuracy_points",
        classification="unresolved_below_power",
        league="nfl",
        seasons=(2020, 2025),
        standard_error=floored["standard_error"],
        interval=(floored["lower"], floored["upper"]),
        probability_positive=1.0,
        sample_games=15,
        classification_evidence="Positive control; no closing ground established.",
    )
    weak_signals.save_registry(weak_signals.record_signal(registry, signal), path)
    reloaded = weak_signals.load_registry(path)
    stored = reloaded.signals["lane_c2_degenerate_fixture"]
    assert stored.standard_error is not None and stored.standard_error > 0.0
    assert stored.effect == 100.0
    assert stored.classification == "unresolved_below_power"
