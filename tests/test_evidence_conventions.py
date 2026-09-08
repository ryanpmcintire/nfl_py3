"""The two shared evidence conventions, and the defects they replaced.

Both were copy-pasted across ~100 call sites and both failed in the SAME
direction: they turned measurements that said nothing into resolved-looking
negatives. These are regression tests for that direction specifically.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from nfl_ats.evidence_conventions import (
    ZERO_ATOM_CREDIT,
    binomial_two_sided_p,
    probability_positive_from_draws,
)


def test_binomial_survives_the_pool_it_is_actually_run_on() -> None:
    # The defect: math.comb(n, k) is an exact int and 0.5**n underflows to 0.0,
    # so the int-to-float conversion overflowed. Measured 2026-09-08: fine at
    # n=1020, OverflowError at n>=1030. The eligible NFL pool is n=1489, so
    # `nfl-ats weak-signals pool` -- the command AGENTS.md tells every session
    # to re-run -- could not execute at all.
    assert 0.0 <= binomial_two_sided_p(576, 1489) <= 1.0
    assert 0.0 <= binomial_two_sided_p(5000, 10000) <= 1.0


def test_binomial_reproduces_the_value_agents_md_quotes() -> None:
    # AGENTS.md records 327 of 628 -> p = 0.31846963 for the NFL pool's sign
    # test. Any replacement arithmetic has to land on exactly that.
    assert binomial_two_sided_p(327, 628) == pytest.approx(0.31846963, abs=5e-9)


@pytest.mark.parametrize(
    ("favourable", "total"),
    [(0, 1), (1, 2), (3, 10), (5, 10), (0, 51), (170, 512), (576, 1281)],
)
def test_binomial_matches_the_exact_construction(favourable: int, total: int) -> None:
    # Sum every outcome no more likely than the observed one, computed the
    # slow, obviously-correct way with exact integer arithmetic.
    from fractions import Fraction

    observed = math.comb(total, favourable)
    expected = float(
        Fraction(
            sum(math.comb(total, k) for k in range(total + 1) if math.comb(total, k) <= observed),
            2**total,
        )
    )
    assert binomial_two_sided_p(favourable, total) == pytest.approx(expected, rel=1e-12)


def test_binomial_on_an_empty_pile_is_not_a_finding() -> None:
    assert binomial_two_sided_p(0, 0) == 1.0


def test_a_candidate_that_changes_nothing_scores_one_half() -> None:
    # THE defect. Two arms making identical picks produce a paired delta of
    # exactly zero in every resample. Under the strict `draws > 0` that was
    # probability_positive 0.0 -- the strongest negative the scale can express,
    # awarded to a no-op -- which is the "interval contains zero is not a
    # rejection" rule being violated through the back door.
    assert probability_positive_from_draws(np.zeros(1000)) == 0.5


def test_the_zero_atom_is_split_not_charged_to_either_arm() -> None:
    draws = np.array([0.0] * 400 + [1.0] * 400 + [-1.0] * 200)
    # 0.40 strictly positive + half of the 0.40 tied.
    assert probability_positive_from_draws(draws) == pytest.approx(0.60)
    assert ZERO_ATOM_CREDIT == 0.5


def test_unambiguous_wins_and_losses_are_unchanged() -> None:
    # The fix must not soften a real result: with no ties, the old strict
    # convention and this one agree exactly.
    assert probability_positive_from_draws(np.array([1.0, 2.0, 3.0])) == 1.0
    assert probability_positive_from_draws(np.array([-1.0, -2.0, -3.0])) == 0.0
    mixed = np.array([1.0, -1.0, 2.0, -3.0])
    assert probability_positive_from_draws(mixed) == pytest.approx(float(np.mean(mixed > 0.0)))


def test_axis_reduction_returns_one_probability_per_column() -> None:
    draws = np.array([[0.0, 1.0, -1.0], [0.0, 1.0, -1.0]])
    assert probability_positive_from_draws(draws, axis=0).tolist() == [0.5, 1.0, 0.0]


def test_failed_resamples_are_not_silently_counted_as_losses() -> None:
    draws = np.array([float("nan"), 1.0, 1.0])
    # Default keeps the previous behaviour exactly (nan > 0 is False).
    assert probability_positive_from_draws(draws) == pytest.approx(2.0 / 3.0)
    # ignore_nan drops it from the denominator instead, for the call sites
    # that used np.nanmean.
    assert probability_positive_from_draws(draws, ignore_nan=True) == 1.0


def test_an_empty_pile_reports_nothing_rather_than_a_probability() -> None:
    assert math.isnan(probability_positive_from_draws(np.array([])))
