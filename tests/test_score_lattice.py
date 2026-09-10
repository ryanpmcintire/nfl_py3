from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from nfl_ats import score_lattice as lattice_module
from nfl_ats.score_lattice import (
    PSEUDO_OBSERVATIONS,
    ScoreLattice,
    build_lattice,
    feasible_team_scores,
    mode_list_probability,
    pick_consistent_top_score,
    pick_cover_probability,
    ranked_modes,
    score_lattice,
)
from nfl_ats.tiebreaker import _neighborhood, weighted_score_counts


def _finals(rows: list[tuple[float, float, int, int]]) -> pd.DataFrame:

    return pd.DataFrame(
        {
            "spread_line": [row[0] for row in rows],
            "total_line": [row[1] for row in rows],
            "home_score": [row[2] for row in rows],
            "away_score": [row[3] for row in rows],
        }
    )


def test_feasible_scores_come_from_the_data_and_exclude_what_never_happened() -> None:
    finals = _finals([(3.0, 43.0, 24, 21), (-2.5, 47.0, 20, 17), (0.0, 41.0, 3, 0)])
    assert feasible_team_scores(finals).tolist() == [0, 3, 17, 20, 21, 24]


def test_impossible_nfl_scores_are_absent_from_the_real_history() -> None:
    finals = _finals([(3.0, 43.0, 1, 4), (0.0, 40.0, 24, 20)])
    assert feasible_team_scores(finals).tolist() == [1, 4, 20, 24]
    without = _finals([(0.0, 40.0, 24, 20), (0.0, 40.0, 21, 17)])
    support = feasible_team_scores(without).tolist()
    assert 1 not in support and 4 not in support


def test_feasible_scores_reject_an_empty_history() -> None:
    with pytest.raises(ValueError, match="feasible score set"):
        feasible_team_scores(_finals([]))


def test_a_single_game_at_the_target_market_lands_entirely_on_its_own_final() -> None:

    finals = _finals([(3.0, 43.0, 24, 21)])
    built = score_lattice(finals, 3.0, 43.0)
    assert built.probability(24, 21) == pytest.approx(1.0)
    assert built.top_scores(1) == ((24, 21, pytest.approx(1.0)),)


def test_a_half_point_offset_splits_mass_over_exactly_four_cells() -> None:

    finals = _finals([(3.0, 43.0, 24, 21), (0.0, 40.0, 20, 20), (0.0, 40.0, 25, 25)])
    neighborhood = _neighborhood(finals, 4.0, 43.0)
    support = np.array([20, 21, 24, 25], dtype=np.int64)
    weights = np.array([1.0, 0.0, 0.0])
    built = build_lattice(neighborhood.frame, weights, 4.0, 43.0, support, recentre=True)
    for home, away in ((24, 20), (24, 21), (25, 20), (25, 21)):
        assert built.probability(home, away) == pytest.approx(0.25)
    assert built.probabilities.sum() == pytest.approx(1.0)


def test_probabilities_sum_to_one_on_real_shaped_input() -> None:
    rng = np.random.default_rng(20260901)
    finals = _finals(
        [
            (
                float(rng.integers(-14, 15)),
                float(rng.integers(36, 55)),
                int(rng.integers(0, 45)),
                int(rng.integers(0, 45)),
            )
            for _ in range(400)
        ]
    )
    built = score_lattice(finals, 2.5, 43.0)
    assert built.probabilities.sum() == pytest.approx(1.0)
    assert built.total_distribution().sum() == pytest.approx(1.0)
    assert built.margin_distribution().sum() == pytest.approx(1.0)


def test_interpolation_preserves_mass_when_every_cell_is_feasible() -> None:

    support = np.arange(0, 61, dtype=np.int64)
    grid = lattice_module._lattice_weights(
        np.array([24.3, 10.9, 40.0]),
        np.array([20.7, 31.25, 3.0]),
        np.array([2.0, 3.0, 5.0]),
        support,
    )
    assert grid.sum() == pytest.approx(10.0)


def test_push_probability_is_exactly_zero_at_a_half_point_line() -> None:
    finals = _finals([(3.0, 43.0, 24, 21), (3.0, 43.0, 20, 17), (3.0, 43.0, 27, 20)])
    built = score_lattice(finals, 3.0, 43.0)
    assert built.push_probability(2.5) == 0.0
    assert built.push_probability(3.0) == pytest.approx(2.0 / 3.0)


def test_median_and_modal_total_are_reported_separately() -> None:
    finals = _finals([(0.0, 40.0, 20, 20), (0.0, 40.0, 20, 20), (0.0, 40.0, 35, 30)])
    built = score_lattice(finals, 0.0, 40.0)
    assert built.modal_total() == 40
    assert built.median_total() == pytest.approx(40.0)


def test_smoothed_probability_is_finite_off_support_and_sums_to_one_on_it() -> None:
    finals = _finals([(0.0, 40.0, 20, 20), (0.0, 40.0, 24, 17)])
    built = score_lattice(finals, 0.0, 40.0)
    total = sum(
        built.smoothed_probability(int(home), int(away))
        for home in built.scores
        for away in built.scores
    )
    assert total == pytest.approx(1.0)
    off_support = built.smoothed_probability(99, 99)
    assert off_support > 0.0
    assert math.isfinite(-math.log(off_support))
    assert off_support == pytest.approx(
        PSEUDO_OBSERVATIONS / built.support_size / (built.weight_total + PSEUDO_OBSERVATIONS)
    )


def test_conditioning_on_a_total_keeps_only_that_total() -> None:
    finals = _finals([(0.0, 40.0, 20, 20), (0.0, 40.0, 24, 17), (0.0, 40.0, 21, 20)])
    built = score_lattice(finals, 0.0, 40.0).condition_on_total(41)
    assert built.probabilities.sum() == pytest.approx(1.0)
    assert built.probability(21, 20) == pytest.approx(0.5)
    assert built.probability(24, 17) == pytest.approx(0.5)
    assert built.probability(20, 20) == 0.0


def test_the_lattice_reuses_the_shipped_tiebreaker_neighborhood(monkeypatch) -> None:  # type: ignore[no-untyped-def]

    finals = _finals([(0.0, 40.0, 20, 20), (0.0, 40.0, 24, 17)])
    calls: list[tuple[float, float]] = []
    real = lattice_module._neighborhood

    def spy(frame: pd.DataFrame, margin: float, total: float):  # type: ignore[no-untyped-def]
        calls.append((margin, total))
        return real(frame, margin, total)

    monkeypatch.setattr(lattice_module, "_neighborhood", spy)
    score_lattice(finals, 0.0, 40.0)
    assert calls == [(0.0, 40.0)]


def test_without_recentring_the_lattice_is_exactly_the_shipped_mode_list() -> None:

    rng = np.random.default_rng(4)
    finals = _finals(
        [
            (
                float(rng.integers(-10, 11)),
                float(rng.integers(38, 52)),
                int(rng.integers(0, 40)),
                int(rng.integers(0, 40)),
            )
            for _ in range(120)
        ]
    )
    neighborhood = _neighborhood(finals, 2.0, 44.0)
    support = feasible_team_scores(finals)
    counts = weighted_score_counts(neighborhood.frame, neighborhood.weights)
    built = build_lattice(
        neighborhood.frame, neighborhood.weights, 2.0, 44.0, support, recentre=False
    )
    for (home, away), weight in counts.items():
        assert built.weights[
            int(np.searchsorted(support, home)), int(np.searchsorted(support, away))
        ] == pytest.approx(weight)
        assert built.smoothed_probability(home, away) == pytest.approx(
            mode_list_probability(counts, support, home, away)
        )
    assert built.top_scores(3) == tuple(
        (home, away, pytest.approx(weight / built.weight_total))
        for home, away, weight in ranked_modes(counts, 3)
    )


def test_ranked_modes_breaks_ties_by_score_like_the_shipped_report() -> None:
    counts = {(20, 17): 3.0, (24, 21): 3.0, (13, 10): 3.0, (27, 24): 1.0}
    assert ranked_modes(counts, 3) == ((13, 10, 3.0), (20, 17, 3.0), (24, 21, 3.0))


def _hand_lattice(scores: list[int], probabilities: np.ndarray) -> ScoreLattice:
    scores_arr = np.array(scores, dtype=np.int64)
    probs_arr = np.array(probabilities, dtype=float)
    return ScoreLattice(
        scores=scores_arr,
        probabilities=probs_arr,
        weights=probs_arr,
        weight_total=float(probs_arr.sum()),
        effective_size=100.0,
        label="hand-built for test",
    )


def test_pick_consistent_top_score_never_selects_a_push_even_when_nearest_and_heaviest() -> None:

    scores = [17, 20, 23]
    probs = np.zeros((3, 3))
    probs[1, 0] = 1.0
    probs[2, 0] = 0.0
    lattice = _hand_lattice(scores, probs)

    chosen = pick_consistent_top_score(
        lattice,
        pick_side="HOME",
        spread_line=3.0,
        served_total=40.0,
        centre_margin=3.0,
    )
    assert chosen is not None
    home_score, away_score, _probability, _tolerance = chosen
    assert (home_score, away_score) == (23, 17)


def test_pick_consistent_top_score_excludes_a_final_too_far_from_the_served_total() -> None:

    scores = [17, 20, 23]
    probs = np.zeros((3, 3))
    probs[1, 0] = 0.5
    probs[2, 0] = 0.3
    probs[0, 0] = 0.2
    lattice = _hand_lattice(scores, probs)

    assert (
        pick_consistent_top_score(
            lattice,
            pick_side="HOME",
            spread_line=3.0,
            served_total=44.0,
            centre_margin=6.0,
        )
        is None
    )


def test_pick_consistent_top_score_dog_pick_picks_the_nearest_candidate_not_the_most_mass() -> None:

    guess_margin, guess_total_line = -3.19, 43.62
    home = [20.0, 20.0, 20.0, 19.0, 10.0, 10.0, 10.0, 10.0, 10.0, 24.0]
    away = [24.0, 24.0, 24.0, 24.0, 16.0, 16.0, 16.0, 16.0, 16.0, 20.0]
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0])
    neighborhood = pd.DataFrame({"home_score": home, "away_score": away})
    scores = np.array(sorted(set(home) | set(away)), dtype=np.int64)
    lattice = build_lattice(
        neighborhood, weights, guess_margin, guess_total_line, scores, recentre=False
    )
    assert lattice.probability(10, 16) > lattice.probability(20, 24)
    assert lattice.probability(24, 20) > lattice.probability(10, 16)

    chosen = pick_consistent_top_score(
        lattice,
        pick_side="AWAY",
        spread_line=3.0,
        served_total=guess_total_line,
        centre_margin=guess_margin,
    )
    assert chosen == (20, 24, pytest.approx(3.0 / 19.0), pytest.approx(1.0))


def test_pick_consistent_top_score_kc_regression_matches_the_real_centre_exactly() -> None:

    guess_margin, guess_total_line = 3.19, 43.62
    home = [24.0, 24.0, 24.0, 25.0, 16.0, 16.0, 16.0, 16.0, 16.0, 20.0]
    away = [20.0, 20.0, 20.0, 19.0, 10.0, 10.0, 10.0, 10.0, 10.0, 24.0]
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0])
    neighborhood = pd.DataFrame({"home_score": home, "away_score": away})
    scores = np.array(sorted(set(home) | set(away)), dtype=np.int64)
    lattice = build_lattice(
        neighborhood, weights, guess_margin, guess_total_line, scores, recentre=False
    )
    assert lattice.probability(16, 10) > lattice.probability(24, 20)
    assert lattice.probability(20, 24) > lattice.probability(16, 10)

    chosen = pick_consistent_top_score(
        lattice,
        pick_side="HOME",
        spread_line=3.0,
        served_total=guess_total_line,
        centre_margin=guess_margin,
    )
    assert chosen == (24, 20, pytest.approx(3.0 / 19.0), pytest.approx(1.0))


def test_pick_consistent_top_score_near_tie_broken_by_lattice_mass() -> None:

    scores = [17, 19, 20, 24, 25]
    probs = np.zeros((5, 5))
    index = {value: position for position, value in enumerate(scores)}
    probs[index[24], index[20]] = 0.2
    probs[index[25], index[19]] = 0.6
    lattice = _hand_lattice(scores, probs)

    chosen = pick_consistent_top_score(
        lattice,
        pick_side="HOME",
        spread_line=3.0,
        served_total=44.0,
        centre_margin=5.0,
    )
    assert chosen is not None
    home_score, away_score, probability, _tolerance = chosen
    assert (home_score, away_score) == (25, 19)
    assert probability == pytest.approx(0.6)


def test_pick_consistent_top_score_can_select_a_zero_mass_candidate() -> None:

    scores = [17, 20, 23]
    probs = np.zeros((3, 3))
    probs[1, 0] = 1.0
    lattice = _hand_lattice(scores, probs)
    assert lattice.probability(23, 17) == 0.0

    chosen = pick_consistent_top_score(
        lattice,
        pick_side="HOME",
        spread_line=3.0,
        served_total=40.0,
        centre_margin=6.0,
    )
    assert chosen == (23, 17, pytest.approx(0.0), pytest.approx(1.0))


def test_pick_consistent_top_score_hard_guard_refuses_a_final_too_far_from_the_centre() -> None:

    scores = [17, 20, 30]
    probs = np.zeros((3, 3))
    probs[2, 0] = 1.0
    lattice = _hand_lattice(scores, probs)
    assert (
        pick_consistent_top_score(
            lattice,
            pick_side="HOME",
            spread_line=3.0,
            served_total=45.5,
            centre_margin=4.0,
        )
        is None
    )


def test_pick_consistent_top_score_returns_none_when_no_admissible_cell_exists() -> None:

    scores = [20]
    probs = np.array([[1.0]])
    lattice = _hand_lattice(scores, probs)
    assert (
        pick_consistent_top_score(
            lattice,
            pick_side="HOME",
            spread_line=0.0,
            served_total=40.0,
            centre_margin=0.0,
        )
        is None
    )
    assert (
        pick_consistent_top_score(
            lattice,
            pick_side="AWAY",
            spread_line=0.0,
            served_total=40.0,
            centre_margin=0.0,
        )
        is None
    )


def test_pick_consistent_top_score_rejects_a_bad_pick_side() -> None:
    lattice = _hand_lattice([20], np.array([[1.0]]))
    with pytest.raises(ValueError):
        pick_consistent_top_score(
            lattice,
            pick_side="HOME_TEAM",
            spread_line=0.0,
            served_total=40.0,
            centre_margin=0.0,
        )


def test_pick_cover_probability_sums_only_the_admissible_side() -> None:
    scores = [17, 20, 23]
    probs = np.zeros((3, 3))
    probs[2, 0] = 0.3
    probs[1, 0] = 0.5
    probs[0, 1] = 0.2
    lattice = _hand_lattice(scores, probs)
    assert pick_cover_probability(lattice, pick_side="HOME", spread_line=3.0) == pytest.approx(0.3)
    assert pick_cover_probability(lattice, pick_side="AWAY", spread_line=3.0) == pytest.approx(0.2)
    assert lattice.push_probability(3.0) == pytest.approx(0.5)
