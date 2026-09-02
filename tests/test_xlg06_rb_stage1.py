"""Tests for scripts/xlg06_rb_stage1.py (WP46).

Two things need checking, per this session's task: (1) the position filter
is correct -- the leak treatment and the correlation cell only touch/select
the requested position's rows, never another position's; (2) pointed at the
same local CFB snapshots the original ``xlg06_rookie_prior_cfb_screen.py``
run used, this thin wrapper reproduces that run's already-computed RB numbers
exactly (it calls the SAME functions with the SAME seeds -- no redesign).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from nfl_ats.cfb import latest_cfb_snapshot  # noqa: E402
from nfl_ats.data import DataContractError  # noqa: E402
from scripts.xlg06_rb_stage1 import (  # noqa: E402
    RELIABILITY_COHORT_SEED,
    RELIABILITY_PLAYER_SEED,
    SCREEN_COHORT_SEED,
    SCREEN_PLAYER_SEED,
    leak_outcome_as_monotone_predictor_function,
    run_cell,
    run_reliability,
)
from scripts.xlg06_rookie_prior_cfb_screen import (  # noqa: E402
    BOOTSTRAP_SAMPLES,
    build_true_freshman_population,
    load_sources,
)

CFB_ROOT = REPO_ROOT / "data" / "cfb"


def _synthetic_matched() -> pd.DataFrame:
    """A tiny matched-population-shaped fixture with TWO positions.

    RB rows carry an exact, deterministic positive linear rating->outcome
    relationship (Pearson r == 1.0). WR rows carry a rating-independent
    outcome pattern (alternating, not a function of rating's rank) so its
    true correlation is small and structurally different from RB's. Any
    accidental cross-position leakage in the position filter (leak function
    or correlation cell) would change one position's measured relationship
    to look like the other's, which the assertions below would catch.

    12 distinct cohort years (>= MIN_BLOCKS_FOR_INTERVAL) so the cohort-
    blocked bootstrap does not raise BootstrapDegeneracyError.
    """

    n_per_position = 14
    rb_rating = np.linspace(50.0, 99.0, n_per_position)
    rb_outcome = 0.001 * rb_rating + 0.01  # exact positive linear relationship
    wr_rating = np.linspace(50.0, 99.0, n_per_position)
    wr_outcome = np.array([0.05 if i % 2 == 0 else 0.06 for i in range(n_per_position)])

    rows: list[dict[str, object]] = []
    for i in range(n_per_position):
        cohort_year = 2000 + (i % 12)
        rows.append(
            {
                "position_usage": "RB",
                "cohort_year": cohort_year,
                "rating": rb_rating[i],
                "stars": 3,
                "usage.overall": rb_outcome[i],
                "usage.pass": rb_outcome[i] * 0.1,
                "usage.rush": rb_outcome[i] * 0.9,
            }
        )
        rows.append(
            {
                "position_usage": "WR",
                "cohort_year": cohort_year,
                "rating": wr_rating[i],
                "stars": 3,
                "usage.overall": wr_outcome[i],
                "usage.pass": wr_outcome[i] * 0.9,
                "usage.rush": wr_outcome[i] * 0.1,
            }
        )
    return pd.DataFrame(rows)


def test_leak_only_modifies_target_position() -> None:
    matched = _synthetic_matched()
    original = matched.copy(deep=True)

    leaked = leak_outcome_as_monotone_predictor_function(
        matched, position="RB", predictor="rating", outcome="usage.overall"
    )

    rb_mask = leaked["position_usage"].eq("RB")
    wr_mask = leaked["position_usage"].eq("WR")

    # A different position (WR) must be completely untouched by an RB leak.
    pd.testing.assert_series_equal(
        leaked.loc[wr_mask, "usage.overall"].reset_index(drop=True),
        original.loc[wr_mask, "usage.overall"].reset_index(drop=True),
    )
    # The caller's frame must not be mutated in place.
    pd.testing.assert_frame_equal(matched, original)

    # RB rows must have changed and be perfectly rank-monotone in the
    # predictor (Spearman rho == 1.0 by construction of the rank leak).
    assert not leaked.loc[rb_mask, "usage.overall"].equals(original.loc[rb_mask, "usage.overall"])
    rb = leaked.loc[rb_mask]
    spearman = rb["rating"].corr(rb["usage.overall"], method="spearman")
    assert spearman == pytest.approx(1.0, abs=1e-9)

    # Bounded within the RB population's own originally observed outcome range
    # (min-max rescale, not an unbounded leak).
    assert leaked.loc[rb_mask, "usage.overall"].min() == pytest.approx(
        original.loc[rb_mask, "usage.overall"].min()
    )
    assert leaked.loc[rb_mask, "usage.overall"].max() == pytest.approx(
        original.loc[rb_mask, "usage.overall"].max()
    )


def test_leak_raises_on_a_degenerate_position() -> None:
    """A position with zero predictor variance (or absent from the frame) is a
    degenerate leak target, not a silent no-op."""

    matched = _synthetic_matched()
    with pytest.raises(ValueError):
        leak_outcome_as_monotone_predictor_function(
            matched, position="QB", predictor="rating", outcome="usage.overall"
        )


def test_run_cell_position_filter_is_correct() -> None:
    """``run_cell(..., position=X)`` must score ONLY position X's rows."""

    matched = _synthetic_matched()

    rb_result = run_cell(
        matched,
        position="RB",
        predictor="rating",
        outcome="usage.overall",
        cohort_seed=1,
        player_seed=2,
        samples=200,
    )
    wr_result = run_cell(
        matched,
        position="WR",
        predictor="rating",
        outcome="usage.overall",
        cohort_seed=1,
        player_seed=2,
        samples=200,
    )

    n_rb = int(matched["position_usage"].eq("RB").sum())
    n_wr = int(matched["position_usage"].eq("WR").sum())
    assert rb_result["player_blocked_secondary"]["n"] == n_rb == 14
    assert wr_result["player_blocked_secondary"]["n"] == n_wr == 14

    # Point estimates (unaffected by bootstrap draw count/seed -- computed
    # once on the full selected subset) must reflect each position's OWN
    # constructed relationship, never the other position's.
    rb_r = rb_result["player_blocked_secondary"]["pearson_r"]
    wr_r = wr_result["player_blocked_secondary"]["pearson_r"]
    assert rb_r == pytest.approx(1.0, abs=1e-9)
    assert abs(wr_r) < 0.5
    assert rb_r != pytest.approx(wr_r, abs=1e-3)

    # Point estimate is identical across blocking schemes (both computed on
    # the same position-filtered subset before any resampling).
    assert rb_result["cohort_blocked_primary"]["pearson_r"] == pytest.approx(rb_r, abs=1e-9)


CFB_SNAPSHOTS_AVAILABLE = True
try:
    latest_cfb_snapshot(CFB_ROOT, "recruiting_players")
    latest_cfb_snapshot(CFB_ROOT, "usage")
except (DataContractError, FileNotFoundError, OSError):
    CFB_SNAPSHOTS_AVAILABLE = False


@pytest.mark.skipif(
    not CFB_SNAPSHOTS_AVAILABLE,
    reason="local CFB recruiting_players/usage snapshots not present",
)
def test_wrapper_reproduces_original_rb_artifact_numbers() -> None:
    """Pointed at the same local snapshots the 2026-08-18 run used, the
    wrapper's ``run_cell``/``run_reliability`` reproduce that run's already-
    recorded RB numbers to within 1e-6 -- same functions, same seeds, same
    data, no redesign.

    Read (this session, ``artifacts/xlg06_rookie_prior_cfb/20260818T215305Z/
    results.json``, ``secondary_position_correlations_*_blocked_*.RB`` and
    ``construct_facet_reliability_*_blocked_*.RB``):
      cohort-blocked primary correlation:  r=0.06443485228551912,
        CI [0.018736493250855902, 0.1153217444630635], P+ 0.99705,
        n=1204, blocks=13
      player-blocked secondary correlation: r=0.06443485228551912,
        CI [0.004379989934557209, 0.12326773803999294], P+ 0.98175, n=1204
      construct-facet reliability (cohort-blocked primary):
        SB(pearson)=0.8016837162101195, n=1273, blocks=13
    """

    recruiting, usage, _provenance = load_sources()
    matched, _diagnostics = build_true_freshman_population(recruiting, usage)

    correlation = run_cell(
        matched,
        position="RB",
        predictor="rating",
        outcome="usage.overall",
        cohort_seed=SCREEN_COHORT_SEED,
        player_seed=SCREEN_PLAYER_SEED,
        samples=BOOTSTRAP_SAMPLES,
    )
    cohort = correlation["cohort_blocked_primary"]
    player = correlation["player_blocked_secondary"]

    assert cohort["n"] == 1204
    assert cohort["blocks"] == 13
    assert cohort["pearson_r"] == pytest.approx(0.06443485228551912, abs=1e-6)
    assert cohort["pearson_r_ci95"][0] == pytest.approx(0.018736493250855902, abs=1e-6)
    assert cohort["pearson_r_ci95"][1] == pytest.approx(0.1153217444630635, abs=1e-6)
    assert cohort["pearson_probability_positive"] == pytest.approx(0.99705, abs=1e-6)

    assert player["n"] == 1204
    assert player["pearson_r"] == pytest.approx(0.06443485228551912, abs=1e-6)
    assert player["pearson_r_ci95"][0] == pytest.approx(0.004379989934557209, abs=1e-6)
    assert player["pearson_r_ci95"][1] == pytest.approx(0.12326773803999294, abs=1e-6)
    assert player["pearson_probability_positive"] == pytest.approx(0.98175, abs=1e-6)

    reliability = run_reliability(
        matched,
        position="RB",
        cohort_seed=RELIABILITY_COHORT_SEED,
        player_seed=RELIABILITY_PLAYER_SEED,
        samples=BOOTSTRAP_SAMPLES,
    )
    rel_cohort = reliability["cohort_blocked_primary"]
    assert rel_cohort["n"] == 1273
    assert rel_cohort["blocks"] == 13
    assert rel_cohort["spearman_brown_full_length_reliability_pearson"] == pytest.approx(
        0.8016837162101195, abs=1e-6
    )
