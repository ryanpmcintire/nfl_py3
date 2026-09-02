"""XLG-06 Stage 1: RB dedicated look (WP46, 2026-09-01).

``docs/xlg06_rookie_prior_screen.md``'s QB write-up flagged a SECONDARY read
on running backs, computed inside the same artifacts by
``scripts/xlg06_rookie_prior_cfb_screen.py``'s Step 6 (secondary positions,
never pooled with QB): ``rating -> usage.overall``, cohort-blocked, n=1204,
r=+0.0644, 95% CI [+0.0187, +0.1153], P+ 0.9971 -- an interval that sits
entirely positive. That secondary read is why this dedicated look exists;
see ``docs/xlg06_rookie_prior_screen.md``'s "RB cell: dedicated look" section
for the full disclosure that the sign was already seen before this look was
declared (sequential confirmation, not a first look).

This script does NOT redesign the cell or reimplement any statistic. It
imports ``load_sources``, ``build_true_freshman_population``,
``construct_facet_reliability`` and ``predictor_outcome_correlation`` from
``scripts/xlg06_rookie_prior_cfb_screen.py`` UNCHANGED and calls them with
``position="RB"`` -- the exact same population join, the exact same
cohort-blocked/player-blocked percentile bootstrap, the exact same
construct-facet split-half substitute, and (for ``--mode screen``) the exact
same seeds the original script already used for its RB secondary cell
(cohort seed 4000, player seed 4100, reliability cohort seed 1001, player
seed 1101 -- RB is index 0 of ``SECONDARY_POSITIONS`` for the correlation
cells and index 1 of ``(PRIMARY_POSITION, *SECONDARY_POSITIONS)`` for the
reliability cells). Pointed at the same local CFB snapshots the original run
used, ``--mode screen`` reproduces that run's RB numbers exactly.

Two modes (``--mode``):

* ``positive-control`` -- the RB outcome column is replaced by a deliberately
  LEAKED, monotone (rank-preserving, non-linear) function of the predictor:
  each RB row's ``usage.overall`` is overwritten with its predictor-rank
  rescaled into the RB population's own observed outcome range. This is a
  code-path sanity check on the bootstrap correlation machinery itself --
  "if a to-spec monotone relationship exists, does this instrument report
  r toward 1 and P+ toward 1" -- NOT a positive control sized to the real
  effect (AGENTS.md's ``bounded_by_control`` closing ground requires an
  instrument proven able to detect an effect THE SAME SIZE as the one being
  tested; this is a coarser "is the code broken" check and is never cited as
  grounds to close the RB cell).
* ``screen`` -- the real look: the RB predictor-outcome cell, unmodified,
  both cohort-blocked (primary) and player-blocked (secondary coherence
  check), plus the RB construct-facet reliability (both blockings).

Run order per this session's task: positive-control first, then screen once.
Both write an artifact under ``artifacts/xlg06_rookie_prior_cfb/rb_<stamp>/``
via ``nfl_ats.provenance.write_experiment_artifact`` (every script in this
repo that writes JSON under ``artifacts/`` must use it -- a repo test
enforces this). No registry write happens inside this script; the recording
command is run separately, under the cross-process lock, per this session's
report.

    ./.tools/uv.exe run --no-sync python scripts/xlg06_rb_stage1.py --mode positive-control
    ./.tools/uv.exe run --no-sync python scripts/xlg06_rb_stage1.py --mode screen
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from nfl_ats.io import run_id  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from scripts.xlg06_rookie_prior_cfb_screen import (  # noqa: E402
    BOOTSTRAP_SAMPLES,
    OUTCOME_COLUMNS,
    PREDICTOR_COLUMNS,
    build_true_freshman_population,
    construct_facet_reliability,
    load_sources,
    predictor_outcome_correlation,
)

ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "xlg06_rookie_prior_cfb"
POSITION = "RB"

# Same seeds the original script's Step 6 (secondary positions) and Step 3
# (reliability) already used for RB -- RB is index 0 of SECONDARY_POSITIONS =
# ("RB", "WR", "TE") and index 1 of (PRIMARY_POSITION, *SECONDARY_POSITIONS)
# = ("QB", "RB", "WR", "TE"). Reusing them is what makes --mode screen an
# exact reproduction of the already-computed RB numbers, not a new draw.
SCREEN_COHORT_SEED = 4000
SCREEN_PLAYER_SEED = 4100
RELIABILITY_COHORT_SEED = 1001
RELIABILITY_PLAYER_SEED = 1101

# New seeds for the positive-control mode -- not present in the original
# script, since it never ran a leak treatment.
POSITIVE_CONTROL_COHORT_SEED = 5000
POSITIVE_CONTROL_PLAYER_SEED = 5100


def leak_outcome_as_monotone_predictor_function(
    matched: pd.DataFrame, *, position: str, predictor: str, outcome: str
) -> pd.DataFrame:
    """Positive-control treatment: overwrite ``outcome`` with a leaked, monotone,
    NON-LINEAR function of ``predictor`` for rows at ``position`` only.

    The transform is the predictor's own within-position rank, rescaled
    (min-max) into that position's own observed outcome range -- monotone
    increasing in ``predictor`` by construction (rank is a monotone function
    of its input), but not linear, so Pearson r is expected to land strongly
    positive without being trivially forced to exactly 1.0 the way an
    identity leak (``outcome := predictor``) would. Rows outside ``position``
    are left untouched (harmless, since every caller here immediately
    subsets to ``position`` anyway).
    """

    leaked = matched.copy()
    mask = leaked["position_usage"].eq(position)
    subset = leaked.loc[mask]
    ranks = subset[predictor].rank(method="average")
    rank_span = ranks.max() - ranks.min()
    outcome_lo = subset[outcome].min()
    outcome_span = subset[outcome].max() - outcome_lo
    if rank_span <= 0 or outcome_span <= 0 or subset[predictor].isna().all():
        raise ValueError(
            f"positive-control leak degenerate for position={position!r}: "
            f"rank_span={rank_span!r} outcome_span={outcome_span!r}"
        )
    scaled = outcome_lo + (ranks - ranks.min()) / rank_span * outcome_span
    leaked.loc[mask, outcome] = scaled
    return leaked


def run_cell(
    matched: pd.DataFrame,
    *,
    position: str,
    predictor: str,
    outcome: str,
    cohort_seed: int,
    player_seed: int,
    samples: int,
) -> dict[str, Any]:
    """Cohort-blocked (primary) + player-blocked (secondary coherence check),
    calling ``predictor_outcome_correlation`` from the original script
    unmodified -- identical to how it is called for every other position.
    """

    return {
        "cohort_blocked_primary": predictor_outcome_correlation(
            matched,
            position=position,
            predictor=predictor,
            outcome=outcome,
            seed=cohort_seed,
            samples=samples,
            block="cohort",
        ),
        "player_blocked_secondary": predictor_outcome_correlation(
            matched,
            position=position,
            predictor=predictor,
            outcome=outcome,
            seed=player_seed,
            samples=samples,
            block="player",
        ),
    }


def run_reliability(
    matched: pd.DataFrame, *, position: str, cohort_seed: int, player_seed: int, samples: int
) -> dict[str, Any]:
    return {
        "cohort_blocked_primary": construct_facet_reliability(
            matched, position=position, seed=cohort_seed, samples=samples, block="cohort"
        ),
        "player_blocked_secondary": construct_facet_reliability(
            matched, position=position, seed=player_seed, samples=samples, block="player"
        ),
    }


# Seed base for the per-cohort trend diagnostic below -- the same recipe the
# original script's Step 5 uses for QB (seed = base + year), but a disjoint
# seed space (3100 vs QB's 3000) since RB's per-cohort trend was never
# computed there.
COHORT_TREND_SEED_BASE = 3100


def run_cohort_trend(
    matched: pd.DataFrame,
    *,
    position: str,
    predictor: str,
    outcome: str,
    cohort_years: list[int],
    samples: int,
) -> dict[str, dict[str, Any]]:
    """Per-cohort-year point estimates, descriptive only -- NOT itself a
    blocked interval (each cell's own bootstrap is player-blocked within
    that single cohort year, mirroring the original script's Step 5 for QB).
    """

    return {
        str(year): predictor_outcome_correlation(
            matched,
            position=position,
            predictor=predictor,
            outcome=outcome,
            seed=COHORT_TREND_SEED_BASE + year,
            samples=samples,
            cohort_year=year,
        )
        for year in cohort_years
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("positive-control", "screen"), required=True)
    parser.add_argument("--position", default=POSITION)
    parser.add_argument("--predictor", choices=PREDICTOR_COLUMNS, default="rating")
    parser.add_argument("--outcome", choices=OUTCOME_COLUMNS, default="usage.overall")
    parser.add_argument("--samples", type=int, default=BOOTSTRAP_SAMPLES)
    args = parser.parse_args()

    started = time.time()
    print("=== Step 1: load sources (unchanged load_sources()) ===", flush=True)
    recruiting, usage, provenance = load_sources()
    print(json.dumps(provenance, indent=2), flush=True)

    print(
        "\n=== Step 2: build true-freshman population (unchanged "
        "build_true_freshman_population()) ===",
        flush=True,
    )
    matched, join_diagnostics = build_true_freshman_population(recruiting, usage)
    n_cohorts = len(join_diagnostics["cohort_years_usable"])
    print(f"usable cohorts: {n_cohorts}", flush=True)

    print(
        f"\n=== Step 3: {args.position} construct-facet reliability "
        "(unchanged construct_facet_reliability()) ===",
        flush=True,
    )
    reliability = run_reliability(
        matched,
        position=args.position,
        cohort_seed=RELIABILITY_COHORT_SEED,
        player_seed=RELIABILITY_PLAYER_SEED,
        samples=args.samples,
    )
    print(json.dumps(reliability, indent=2, default=str), flush=True)

    if args.mode == "positive-control":
        print(
            f"\n=== Step 4: POSITIVE CONTROL -- {args.outcome} leaked as a monotone "
            f"function of {args.predictor} for {args.position} rows ===",
            flush=True,
        )
        scoring_matched = leak_outcome_as_monotone_predictor_function(
            matched, position=args.position, predictor=args.predictor, outcome=args.outcome
        )
        cohort_seed, player_seed = POSITIVE_CONTROL_COHORT_SEED, POSITIVE_CONTROL_PLAYER_SEED
    else:
        scoring_matched = matched
        cohort_seed, player_seed = SCREEN_COHORT_SEED, SCREEN_PLAYER_SEED

    print(
        f"\n=== Step 5: {args.predictor} -> {args.outcome}, {args.position}, "
        f"mode={args.mode} (unchanged predictor_outcome_correlation()) ===",
        flush=True,
    )
    correlation = run_cell(
        scoring_matched,
        position=args.position,
        predictor=args.predictor,
        outcome=args.outcome,
        cohort_seed=cohort_seed,
        player_seed=player_seed,
        samples=args.samples,
    )
    print(json.dumps(correlation, indent=2, default=str), flush=True)

    cohort_trend: dict[str, dict[str, Any]] | None = None
    if args.mode == "screen":
        print(
            f"\n=== Step 6: per-cohort trend diagnostic, {args.position}, "
            f"{args.predictor} -> {args.outcome} (descriptive only, not itself a "
            "blocked interval; unchanged predictor_outcome_correlation()) ===",
            flush=True,
        )
        cohort_trend = run_cohort_trend(
            scoring_matched,
            position=args.position,
            predictor=args.predictor,
            outcome=args.outcome,
            cohort_years=list(join_diagnostics["cohort_years_usable"]),
            samples=args.samples,
        )
        print(json.dumps(cohort_trend, indent=2, default=str), flush=True)

    configuration = {
        "mode": args.mode,
        "position": args.position,
        "predictor": args.predictor,
        "outcome": args.outcome,
        "samples": args.samples,
        "n_cohorts": n_cohorts,
        "cohort_seed": cohort_seed,
        "player_seed": player_seed,
        "reliability_cohort_seed": RELIABILITY_COHORT_SEED,
        "reliability_player_seed": RELIABILITY_PLAYER_SEED,
        "source_script": (
            "scripts/xlg06_rookie_prior_cfb_screen.py (functions imported unmodified)"
        ),
        "predeclaration": (
            "docs/xlg06_rookie_prior_screen.md#rb-cell-dedicated-look-2026-09-01-wp46"
        ),
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": round(time.time() - started, 1),
        **configuration,
        "join_diagnostics": join_diagnostics,
        "reliability": reliability,
        "correlation": correlation,
        "cohort_trend": cohort_trend,
        "provenance": artifact_provenance(
            configuration, REPO_ROOT / "scripts" / "xlg06_rookie_prior_cfb_screen.py", REPO_ROOT
        ),
    }

    output_dir = ARTIFACT_ROOT / f"rb_{run_id()}"
    metrics = {
        "mode": args.mode,
        "position": args.position,
        "cohort_blocked_pearson_r": correlation["cohort_blocked_primary"].get("pearson_r"),
        "cohort_blocked_probability_positive": correlation["cohort_blocked_primary"].get(
            "pearson_probability_positive"
        ),
        "reliability_cohort_blocked": reliability["cohort_blocked_primary"].get(
            "spearman_brown_full_length_reliability_pearson"
        ),
    }
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="xlg06-rb-stage1",
        metrics=metrics,
        notes=(
            "WP46 RB dedicated look, following up the QB write-up's flagged secondary RB "
            "read. Calls scripts/xlg06_rookie_prior_cfb_screen.py's functions unmodified "
            "with position='RB'. Never pooled with QB (position-scale caveat, see the "
            "original script's docstring). See "
            "docs/xlg06_rookie_prior_screen.md#rb-cell-dedicated-look-2026-09-01-wp46."
        ),
    )
    print(f"\nWrote {output_dir / 'results.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
