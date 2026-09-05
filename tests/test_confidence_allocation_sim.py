"""Release-blocking tests for LEAD-51's confidence-point allocation simulator.

Covers: rank validity (distinct confidence points 1..n every week), the
oracle positive control's dominance (it must achieve the score-maximizing
permutation for ANY fixed correctness pattern, a combinatorial fact, not a
statistical one), calibration honesty (a leakage regression test: mutating a
later week's outcomes must never change an earlier week's calibrated
probabilities), determinism under a fixed seed, the closed-form tie-credit
math against brute-force enumeration, and the archive loader's push/graded
-game-count contract.
"""

from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.confidence_allocation_sim import (
    MIN_GRADED_GAMES,
    assign_points,
    load_archive,
    measure_favorite_share,
    probability_first,
    run_simulation,
    walk_forward_calibration,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _score(confidence: np.ndarray, correct: np.ndarray, game_ids: np.ndarray) -> float:
    points = assign_points(confidence, game_ids)
    return float(np.sum(points * correct))


def _synthetic_graded(n_weeks: int, games_per_week: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for week in range(1, n_weeks + 1):
        for game in range(games_per_week):
            spread = float(rng.uniform(-10.0, 10.0))
            home_covered = bool(rng.random() < 0.5)
            favorite_side = "HOME" if spread < 0 else ("AWAY" if spread > 0 else "NONE")
            probability = float(rng.uniform(0.2, 0.8))
            picked_home = probability >= 0.5
            correct = picked_home == home_covered
            rows.append(
                {
                    "game_id": f"2024_{week:02d}_A{game:02d}_B{game:02d}",
                    "season": 2024,
                    "week": week,
                    "home_cover_probability_at_open": probability,
                    "pick_home_at_open_probability_rule": picked_home,
                    "tue_open_home_spread": spread,
                    "margin_vs_open": 1.0 if home_covered else -1.0,
                    "favorite_side": favorite_side,
                    "home_covered": home_covered,
                    "correct_at_open_probability_rule": float(correct),
                }
            )
    return pd.DataFrame(rows)


def _archive_row(*, week: int, game: int, push: bool, season: int = 2024) -> dict[str, object]:
    return {
        "game_id": f"{season}_{week:02d}_A{game:02d}_B{game:02d}",
        "season": season,
        "week": week,
        "tue_open_home_spread": 3.0,
        "margin_vs_open": 0.0 if push else 5.0,
        "home_cover_probability_at_open": 0.6,
        "correct_at_open_probability_rule": float("nan") if push else 1.0,
        "pick_home_at_open_probability_rule": True,
    }


# ---------------------------------------------------------------------------
# Rank validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 5, 14, 16])
def test_assign_points_is_a_distinct_1_to_n_ranking(n: int) -> None:
    rng = np.random.default_rng(n)
    confidence = rng.uniform(0.0, 1.0, size=n)
    game_ids = np.array([f"g{i:02d}" for i in range(n)])
    points = assign_points(confidence, game_ids)
    assert sorted(points.tolist()) == list(range(1, n + 1))


def test_assign_points_breaks_ties_by_game_id_ascending() -> None:
    confidence = np.array([0.5, 0.5, 0.9])
    game_ids = np.array(["b", "a", "z"])
    points = assign_points(confidence, game_ids)
    # "z" (index 2) has the highest confidence alone -> the top point value.
    assert points[2] == 3
    # The 0.5 tie between "a" (index 1) and "b" (index 0) breaks by game_id
    # ascending: "a" outranks "b".
    assert points[1] == 2
    assert points[0] == 1


def test_assign_points_handles_all_tied_confidence() -> None:
    n = 6
    confidence = np.zeros(n)
    game_ids = np.array([f"g{i}" for i in range(n)])
    points = assign_points(confidence, game_ids)
    assert sorted(points.tolist()) == list(range(1, n + 1))
    # Fully tied: ascending game_id order gets descending point values.
    assert points.tolist() == list(range(n, 0, -1))


# ---------------------------------------------------------------------------
# Oracle positive control: dominance is a combinatorial fact, not a
# statistical one -- it must hold for ANY confidence array on the SAME fixed
# correctness pattern.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,seed", [(1, 0), (5, 1), (14, 2), (16, 3)])
def test_oracle_achieves_the_analytic_maximum_and_dominates(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    game_ids = np.array([f"g{i:02d}" for i in range(n)])
    correct = rng.random(n) < 0.5
    k = int(correct.sum())

    edge_confidence = rng.uniform(0.0, 0.5, size=n)
    market_confidence = rng.uniform(0.0, 14.0, size=n)
    oracle_confidence = correct.astype(float)

    oracle_score = _score(oracle_confidence, correct, game_ids)
    edge_score = _score(edge_confidence, correct, game_ids)
    market_score = _score(market_confidence, correct, game_ids)
    flat_expected = k * (n + 1) / 2.0

    # Oracle assigns the top-k point values to the k correct picks -- the
    # score-maximizing permutation for this fixed pattern by construction.
    analytic_max = float(sum(range(n - k + 1, n + 1)))
    assert oracle_score == analytic_max
    assert oracle_score >= edge_score
    assert oracle_score >= market_score
    assert oracle_score >= flat_expected


def test_oracle_dominates_every_week_of_a_synthetic_multi_week_run() -> None:
    graded = _synthetic_graded(n_weeks=5, games_per_week=11, seed=42)
    result = run_simulation(graded, p_favorite=0.5, field_sizes=(10,), mc_draws=500, seed=2026)
    points_by_week = result.drop_duplicates(["season", "week", "strategy"]).pivot(
        index=["season", "week"], columns="strategy", values="expected_points"
    )
    for strategy in points_by_week.columns:
        if strategy == "oracle":
            continue
        assert (points_by_week["oracle"] >= points_by_week[strategy] - 1e-9).all()


# ---------------------------------------------------------------------------
# Calibration honesty: a leakage regression test
# ---------------------------------------------------------------------------


def test_calibration_uses_only_strictly_earlier_weeks() -> None:
    rng = np.random.default_rng(5)
    rows = []
    for week in range(1, 7):
        for _ in range(40):
            rows.append(
                {
                    "season": 2024,
                    "week": week,
                    "home_cover_probability_at_open": float(rng.uniform(0.2, 0.8)),
                    "home_covered": bool(rng.random() < 0.5),
                }
            )
    frame = pd.DataFrame(rows)

    baseline = walk_forward_calibration(frame, min_train=50)

    mutated = frame.copy()
    later = mutated["week"].ge(4)
    # Flip every later week's outcome. If an earlier week's calibrator had
    # peeked at this, its calibrated values would change.
    mutated.loc[later, "home_covered"] = ~mutated.loc[later, "home_covered"]
    perturbed = walk_forward_calibration(mutated, min_train=50)

    earlier = baseline["week"].lt(4)
    pd.testing.assert_series_equal(
        baseline.loc[earlier, "calibrated_probability_at_open"].reset_index(drop=True),
        perturbed.loc[earlier, "calibrated_probability_at_open"].reset_index(drop=True),
    )
    # Sanity: the test is not vacuous -- calibration really is applied to
    # (and could have leaked into) at least one of the earlier weeks, and at
    # least one later week's training set actually differs between the two
    # frames.
    assert bool(baseline.loc[earlier, "calibration_applied"].any())
    assert not perturbed.loc[later, "home_covered"].equals(frame.loc[later, "home_covered"])


# ---------------------------------------------------------------------------
# Determinism under a fixed seed
# ---------------------------------------------------------------------------


def test_run_simulation_is_deterministic_under_a_fixed_seed() -> None:
    graded = _synthetic_graded(n_weeks=4, games_per_week=9, seed=11)
    result_a = run_simulation(graded, p_favorite=0.55, field_sizes=(10, 30), mc_draws=400, seed=123)
    result_b = run_simulation(graded, p_favorite=0.55, field_sizes=(10, 30), mc_draws=400, seed=123)
    pd.testing.assert_frame_equal(result_a, result_b)


def test_run_simulation_differs_under_a_different_seed() -> None:
    graded = _synthetic_graded(n_weeks=4, games_per_week=9, seed=11)
    result_a = run_simulation(graded, p_favorite=0.55, field_sizes=(10,), mc_draws=400, seed=123)
    result_b = run_simulation(graded, p_favorite=0.55, field_sizes=(10,), mc_draws=400, seed=456)
    flat_a = result_a.loc[result_a["strategy"].eq("flat"), "probability_first"].to_numpy()
    flat_b = result_b.loc[result_b["strategy"].eq("flat"), "probability_first"].to_numpy()
    assert not np.array_equal(flat_a, flat_b)


# ---------------------------------------------------------------------------
# Tie-credit closed form vs. brute-force enumeration
# ---------------------------------------------------------------------------


def test_probability_first_matches_brute_force_tie_credit() -> None:
    field_values = np.array([0.0, 1.0, 2.0])
    field_pmf = np.array([0.5, 0.3, 0.2])
    field_cdf = np.cumsum(field_pmf)
    score = 1.0
    entrants = 3

    outright, shared = probability_first(score, entrants, field_values, field_pmf, field_cdf)

    total_outright = 0.0
    total_shared = 0.0
    for combo in product(range(3), repeat=entrants):
        probability = 1.0
        for index in combo:
            probability *= field_pmf[index]
        beaten = sum(1 for index in combo if field_values[index] > score)
        tied = sum(1 for index in combo if field_values[index] == score)
        if beaten == 0:
            total_shared += probability / (1 + tied)
            if tied == 0:
                total_outright += probability

    assert outright == pytest.approx(total_outright)
    assert shared == pytest.approx(total_shared)


# ---------------------------------------------------------------------------
# Archive loader contract: drop pushes, enforce the graded-game floor
# ---------------------------------------------------------------------------


def test_load_archive_drops_pushes_and_enforces_min_graded_games(tmp_path: Path) -> None:
    rows = [_archive_row(week=1, game=i, push=False) for i in range(10)]
    rows += [_archive_row(week=1, game=10 + i, push=True) for i in range(2)]
    # Week 2 has only 5 graded games, below MIN_GRADED_GAMES -> dropped whole.
    rows += [_archive_row(week=2, game=i, push=False) for i in range(5)]
    frame = pd.DataFrame(rows)
    path = tmp_path / "per_game.parquet"
    frame.to_parquet(path)

    graded = load_archive(path)

    assert set(graded["week"].unique().tolist()) == {1}
    assert len(graded) == 10
    assert MIN_GRADED_GAMES == 8


def test_measure_favorite_share_matches_a_hand_computed_case() -> None:
    frame = pd.DataFrame(
        {
            "favorite_side": ["HOME", "HOME", "AWAY", "AWAY", "NONE"],
            "pick_home_at_open_probability_rule": [True, False, False, True, True],
        }
    )
    # Favorite games only (drop the pick'em row): HOME/True match, HOME/False
    # no match, AWAY/False match, AWAY/True no match -> 2 of 4 match.
    assert measure_favorite_share(frame) == pytest.approx(0.5)
