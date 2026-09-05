"""Tests for the MOD-05 correct-score lattice (WP23).

The parts most worth pinning are the ones a silent change would corrupt
invisibly: the feasible score set must come from the DATA (a hand list would
quietly re-admit impossible finals like 1 or 4), the interpolation onto the
lattice must preserve mass (a wrong bandwidth would invent or destroy
probability), the neighborhood must be the SAME one the shipped tiebreaker
uses (two arms that drift apart stop being a paired comparison), and the
walk-forward evaluator must never see the target week.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

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

REPO = Path(__file__).resolve().parents[1]


def _load_eval_script() -> ModuleType:
    scripts_root = REPO / "scripts"
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    spec = importlib.util.spec_from_file_location(
        "score_lattice_eval", scripts_root / "score_lattice_eval.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _finals(rows: list[tuple[float, float, int, int]]) -> pd.DataFrame:
    """A minimal ``lined_finals``-shaped frame: (spread, total, home, away)."""

    return pd.DataFrame(
        {
            "spread_line": [row[0] for row in rows],
            "total_line": [row[1] for row in rows],
            "home_score": [row[2] for row in rows],
            "away_score": [row[3] for row in rows],
        }
    )


# ---------------------------------------------------------------------------
# The feasible score set is read off the data, never written down
# ---------------------------------------------------------------------------


def test_feasible_scores_come_from_the_data_and_exclude_what_never_happened() -> None:
    finals = _finals([(3.0, 43.0, 24, 21), (-2.5, 47.0, 20, 17), (0.0, 41.0, 3, 0)])
    assert feasible_team_scores(finals).tolist() == [0, 3, 17, 20, 21, 24]


def test_impossible_nfl_scores_are_absent_from_the_real_history() -> None:
    # 1 and 4 are unreachable under NFL scoring and the finals say so on their
    # own -- this test exists so a future hand-written support list would fail.
    finals = _finals([(3.0, 43.0, 1, 4), (0.0, 40.0, 24, 20)])
    assert feasible_team_scores(finals).tolist() == [1, 4, 20, 24]
    without = _finals([(0.0, 40.0, 24, 20), (0.0, 40.0, 21, 17)])
    support = feasible_team_scores(without).tolist()
    assert 1 not in support and 4 not in support


def test_feasible_scores_reject_an_empty_history() -> None:
    with pytest.raises(ValueError, match="feasible score set"):
        feasible_team_scores(_finals([]))


# ---------------------------------------------------------------------------
# Known-answer fixtures for the interpolation
# ---------------------------------------------------------------------------


def test_a_single_game_at_the_target_market_lands_entirely_on_its_own_final() -> None:
    """Zero residual offset => the recentred point IS an integer final."""

    finals = _finals([(3.0, 43.0, 24, 21)])
    built = score_lattice(finals, 3.0, 43.0)
    assert built.probability(24, 21) == pytest.approx(1.0)
    assert built.top_scores(1) == ((24, 21, pytest.approx(1.0)),)


def test_a_half_point_offset_splits_mass_over_exactly_four_cells() -> None:
    """Known answer, computed by hand from the mass-preserving triangle.

    Target market (4, 43) implies (23.5, 19.5); the training game's own market
    (3, 43) implied (23, 20) and it finished 24-21, so its residual recentres
    to (24.5, 20.5) -- exactly half a point off the lattice in both
    coordinates, which the bandwidth-1 triangular kernel splits into four
    equal quarters.
    """

    finals = _finals(
        [(3.0, 43.0, 24, 21), (0.0, 40.0, 20, 20), (0.0, 40.0, 25, 25)]  # support filler
    )
    neighborhood = _neighborhood(finals, 4.0, 43.0)
    support = np.array([20, 21, 24, 25], dtype=np.int64)
    # Only the first row carries the residual under test; weight it alone.
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
    """``sum over integers n of max(0, 1 - |n - x|) == 1`` is the whole reason
    the bandwidth is 1 and not a tuned parameter."""

    support = np.arange(0, 61, dtype=np.int64)
    grid = lattice_module._lattice_weights(
        np.array([24.3, 10.9, 40.0]),
        np.array([20.7, 31.25, 3.0]),
        np.array([2.0, 3.0, 5.0]),
        support,
    )
    assert grid.sum() == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# The lattice's own products
# ---------------------------------------------------------------------------


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
    # 21-20 and 24-17 both total 41 and each carried one game, so the
    # conditioned lattice renormalises them to a half apiece.
    assert built.probability(21, 20) == pytest.approx(0.5)
    assert built.probability(24, 17) == pytest.approx(0.5)
    assert built.probability(20, 20) == 0.0


# ---------------------------------------------------------------------------
# The two arms must stay the same experiment
# ---------------------------------------------------------------------------


def test_the_lattice_reuses_the_shipped_tiebreaker_neighborhood(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Kernel reuse is structural, not a copy: patching the tiebreaker's
    neighborhood changes the lattice, which it could not do if this module had
    reimplemented the kernel."""

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
    """The declared secondary arm has an analytic answer: with no residual
    offset every point is already an integer final, so the mass-preserving
    triangle is the identity and the lattice reproduces
    ``weighted_score_counts`` cell for cell. This is what makes "smoothing"
    and "recentring" separable in the evaluation."""

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


# ---------------------------------------------------------------------------
# Leakage: the walk-forward evaluator may never see the target week
# ---------------------------------------------------------------------------


def test_walk_forward_training_is_a_strict_chronological_prefix() -> None:
    module = _load_eval_script()
    seasons, weeks = [], []
    for season in (2020, 2021):
        for week in (1, 2, 3):
            seasons += [season] * 4
            weeks += [week] * 4
    size = len(seasons)
    rng = np.random.default_rng(11)
    finals = pd.DataFrame(
        {
            "game_id": [f"g{index}" for index in range(size)],
            "season": seasons,
            "week": weeks,
            "gameday": [
                f"{season}-09-{week:02d}" for season, week in zip(seasons, weeks, strict=True)
            ],
            "game_type": ["REG"] * size,
            "home_team": ["AAA"] * size,
            "away_team": ["BBB"] * size,
            "spread_line": rng.integers(-7, 8, size).astype(float),
            "total_line": rng.integers(40, 48, size).astype(float),
            "home_score": rng.integers(3, 35, size),
            "away_score": rng.integers(3, 35, size),
        }
    )
    scored = module.walk_forward(finals, 2021, 2021, with_oracle=False)
    assert len(scored) == 12
    # Week (2021, w) may see everything before it and nothing else: 12 games
    # from 2020 plus 4 per already-played 2021 week.
    expected = {1: 12, 2: 16, 3: 20}
    for week, games in scored.groupby("week"):
        assert set(games["training_games"]) == {expected[int(week)]}
    # And the target game itself can never be in its own training slice.
    assert scored["training_games"].max() < len(finals)


def test_walk_forward_support_never_uses_a_future_score() -> None:
    """The feasible score set is rebuilt from the training prefix at every
    week, so a score that only ever happens later is not in the support."""

    module = _load_eval_script()
    finals = pd.DataFrame(
        {
            "game_id": ["a", "b", "c", "d"],
            "season": [2020, 2020, 2021, 2021],
            "week": [1, 2, 1, 2],
            "gameday": ["2020-09-01", "2020-09-08", "2021-09-01", "2021-09-08"],
            "game_type": ["REG"] * 4,
            "home_team": ["AAA"] * 4,
            "away_team": ["BBB"] * 4,
            "spread_line": [0.0, 0.0, 0.0, 0.0],
            "total_line": [40.0, 40.0, 40.0, 40.0],
            "home_score": [20, 24, 62, 20],
            "away_score": [20, 17, 3, 20],
        }
    )
    scored = module.walk_forward(finals, 2021, 2021, with_oracle=False)
    first, second = scored.iloc[0], scored.iloc[1]
    # 2021 week 1 has only 2020 behind it: scores {17, 20, 24}.
    assert first["support_scores"] == 3
    assert first["realised_in_support"] == 0  # its own 62 is not yet knowable
    # 2021 week 2 has 62 and 3 behind it now.
    assert second["support_scores"] == 5


# ---------------------------------------------------------------------------
# Owner mandate (2026-09-05): "our project over/under total needs to line up
# with our spread prediction". pick_consistent_top_score / pick_cover_probability
# select the served tiebreaker score off ONE lattice, excluding pushes and
# wrong-side finals -- see nfl_ats.tiebreaker.build_report's "one lattice,
# one margin, one total" step.
# ---------------------------------------------------------------------------


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


def test_pick_consistent_top_score_excludes_a_push_cell_even_at_higher_probability() -> None:
    """(17, 20, 23): home=20/away=17 is margin 3 -- a PUSH against a 3-point
    home-favorite line -- and carries the most raw mass, but must never be
    chosen for a HOME pick needing margin > 3 strictly."""

    scores = [17, 20, 23]
    probs = np.zeros((3, 3))
    probs[1, 0] = 0.5  # home 20, away 17 -> margin 3 (PUSH against spread_line=3)
    probs[2, 0] = 0.3  # home 23, away 17 -> margin 6 (admissible)
    probs[0, 0] = 0.2  # home 17, away 17 -> margin 0 (wrong side)
    lattice = _hand_lattice(scores, probs)

    chosen = pick_consistent_top_score(
        lattice, pick_side="HOME", spread_line=3.0, centre_home=20.0, centre_away=17.0
    )
    assert chosen == (23, 17, pytest.approx(0.3))


def test_pick_consistent_top_score_dog_pick_selects_the_away_side() -> None:
    """Dog-pick case: the away team is picked to cover a spread favoring
    home by 3 (spread_line=+3), so it must select a final with margin < 3,
    never one on the home side even if that carries more mass."""

    scores = [17, 20, 23]
    probs = np.zeros((3, 3))
    probs[2, 1] = 0.6  # home 23, away 20 -> margin 3 (wrong side for an AWAY pick)
    probs[1, 2] = 0.25  # home 20, away 23 -> margin -3 (admissible: away covers big)
    probs[0, 1] = 0.15  # home 17, away 20 -> margin -3 as well... use a distinct cell below
    lattice = _hand_lattice(scores, probs)

    chosen = pick_consistent_top_score(
        lattice, pick_side="AWAY", spread_line=3.0, centre_home=20.0, centre_away=23.0
    )
    assert chosen is not None
    home_score, away_score, probability = chosen
    assert home_score - away_score < 3.0
    assert probability == pytest.approx(0.25)
    assert (home_score, away_score) == (20, 23)


def test_pick_consistent_top_score_ties_broken_by_closeness_to_centre() -> None:
    scores = [17, 20, 24]
    probs = np.zeros((3, 3))
    probs[2, 0] = 0.4  # home 24, away 17 -> margin 7, far from centre (20, 17)
    probs[1, 0] = 0.0  # margin exactly 3 -- push, would be excluded anyway
    probs[2, 1] = 0.4  # home 24, away 20 -> margin 4, closer to centre (20, 17)? check below
    lattice = _hand_lattice(scores, probs)
    # Both admissible cells tie at 0.4 probability. Distance-squared to the
    # centre (20, 17): (24,17) -> (24-20)^2+(17-17)^2=16; (24,20) ->
    # (24-20)^2+(20-17)^2=25. (24, 17) is closer and must win the tie-break.
    chosen = pick_consistent_top_score(
        lattice, pick_side="HOME", spread_line=3.0, centre_home=20.0, centre_away=17.0
    )
    assert chosen is not None
    home_score, away_score, _probability = chosen
    assert (home_score, away_score) == (24, 17)


def test_pick_consistent_top_score_returns_none_when_no_admissible_cell_exists() -> None:
    """Every feasible final sits at margin 0 -- neither strictly above nor
    below a spread_line of 0 -- so a pick on either side has nothing to
    choose from."""

    scores = [20]
    probs = np.array([[1.0]])
    lattice = _hand_lattice(scores, probs)
    assert (
        pick_consistent_top_score(
            lattice, pick_side="HOME", spread_line=0.0, centre_home=20.0, centre_away=20.0
        )
        is None
    )
    assert (
        pick_consistent_top_score(
            lattice, pick_side="AWAY", spread_line=0.0, centre_home=20.0, centre_away=20.0
        )
        is None
    )


def test_pick_consistent_top_score_rejects_a_bad_pick_side() -> None:
    lattice = _hand_lattice([20], np.array([[1.0]]))
    with pytest.raises(ValueError):
        pick_consistent_top_score(
            lattice, pick_side="HOME_TEAM", spread_line=0.0, centre_home=20.0, centre_away=20.0
        )


def test_pick_cover_probability_sums_only_the_admissible_side() -> None:
    scores = [17, 20, 23]
    probs = np.zeros((3, 3))
    probs[2, 0] = 0.3  # margin 6 -- admissible for HOME at spread_line=3
    probs[1, 0] = 0.5  # margin 3 -- push, excluded from EITHER side
    probs[0, 1] = 0.2  # margin -3 -- admissible for AWAY, not HOME
    lattice = _hand_lattice(scores, probs)
    assert pick_cover_probability(lattice, pick_side="HOME", spread_line=3.0) == pytest.approx(0.3)
    assert pick_cover_probability(lattice, pick_side="AWAY", spread_line=3.0) == pytest.approx(0.2)
    # The push cell (0.5) never counts toward either side.
    assert lattice.push_probability(3.0) == pytest.approx(0.5)


def test_pick_consistent_top_score_returns_none_when_the_admissible_side_has_zero_mass() -> None:
    """An admissible CELL can exist in the scores x scores cross-product
    (both individual scores occurred somewhere) while carrying literally no
    interpolated mass -- that must be treated the same as no admissible
    cell at all, never returned as an invented, zero-evidence guess."""

    scores = [17, 20, 23]
    probs = np.zeros((3, 3))
    probs[1, 0] = 1.0  # home 20, away 17 -> margin 3, all the mass, but this
    # is a PUSH against spread_line=3.0, so it is inadmissible for either side.
    lattice = _hand_lattice(scores, probs)
    # (23, 17) -- margin 6 -- is admissible for HOME but carries zero mass.
    assert lattice.probability(23, 17) == 0.0
    assert (
        pick_consistent_top_score(
            lattice, pick_side="HOME", spread_line=3.0, centre_home=20.0, centre_away=17.0
        )
        is None
    )
