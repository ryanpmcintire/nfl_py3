"""Is a JOINT offence-side swap -- both offence statistics' raw triples out,
both graph katz differentials in -- better than the XLG-03 CFB benchmark
contract as it stands? (WP35)

Predeclared in ``docs/graph_team_stat_cfb_offense_replacement.md`` BEFORE any
outcome number for this comparator was computed. Read that document first: it
declares the two metrics, the five arms, the four cells, the primary grade, the
decision rule and the recording rules.

**This is a sequential confirmation, not a first look, and the script says so
where the numbers are produced.** WP24 (``docs/graph_team_stat_cfb_replacement.md``)
declared a per-cell ``replacement`` - ``ablation`` arm a DIAGNOSTIC, ran it on
all three CFB cells, and saw its signs: favoured on both offence cells
(+0.213 pts opener P+ 0.827 on ``off_epa_per_play``; +0.123 pts close P+ 0.710
on ``off_success_rate``), disfavoured on the defence cell (-0.190 close, P+
0.064). "Offence, not defence" is therefore a subset selected AFTER signs were
seen, and every result this script produces carries that multiplicity. The
genuinely new comparator is the JOINT one: no arm in WP8 or WP24 ever removed
both offence triples at once, and none ever had 31 columns.

Five arms per scored week, all fitted with the benchmark's own estimator
(``nfl_ats.cfb_benchmark.fit_cfb_residual_model``, Ridge alpha 10, out-of-time
residual distribution, 500-game training floor):

* ``benchmark``                     -- frozen ``CFB_MODEL_FEATURE_COLUMNS`` (35).
* ``offense_replacement``           -- benchmark minus BOTH offence triples,
  plus BOTH offence graph katz differentials (31).
* ``offense_ablation``              -- benchmark minus BOTH offence triples,
  nothing added (29).
* ``replacement_off_epa_per_play``  -- WP24's single-metric swap (33).
* ``replacement_off_success_rate``  -- WP24's single-metric swap (33).

The defence triple, the market/context/experience features and the other six
team-state metrics' triples are untouched in every arm.

Cells, in declared priority order (the first is the ONLY one the decision rule
reads):

1. PRIMARY   ``offense_replacement`` - ``benchmark``.
2. secondary ``replacement_off_epa_per_play`` - ``benchmark``  -- CONTINUITY
   ONLY. Comparator for comparator and window for window this is WP24's primary
   for that metric; it is re-run so the joint swap can be read against its own
   parts computed by the same code, and it is not independent evidence.
3. secondary ``replacement_off_success_rate`` - ``benchmark`` -- CONTINUITY ONLY,
   same reason.
4. secondary ``offense_replacement`` - ``offense_ablation`` -- what the two graph
   columns are worth once the six raw columns are gone.

Plus one REPORT-ONLY diagnostic, ``offense_ablation`` - ``benchmark`` (what the
two offence triples are worth inside the contract at all). Declared before the
run, gets no registry row, and cannot be promoted to a cell after its sign is
seen -- it is printed because cell 1 minus cell 4 is arithmetically that
quantity, so hiding it would be pointless as well as dishonest.

Grade: BOTH are computed. ``open`` (``result - spread_open``) is the declared
PRIMARY for the decision rule -- AGENTS.md grades decisions at the opener, and
what this decision governs is an opener-graded NFL follow-up. ``close``
(``result - spread_line``) is reported beside it because it is the frozen XLG-03
benchmark's own grade and the only figure commensurable with WP8's and WP24's
published tables. Never averaged.

Modes, to be run in this order:

* ``--mode null``             -- settle margins shuffled within each week.
* ``--mode positive-control`` -- EXACTLY ONE swapped-in column, the declared
  ``off_epa_per_play`` katz differential, becomes the realised ``ats_margin``.
  The arms that do not carry it must come back byte-identical to the screen.
* ``--mode screen``           -- the real look, once.

Closing-grounds taxonomy (binding, AGENTS.md, restated verbatim so this file
stands on its own): an interval or CI that contains zero is NEVER grounds to
reject, fail, or close an experiment. At this evaluator's ~2-point resolution,
"contains zero" is the EXPECTED outcome for a real small signal. Only two
grounds ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong
sign (whole interval on the wrong side of zero) or zero split-half reliability;
(2) bounded by a positive control proven able to detect an effect that size.
Everything else is ``unresolved_below_power``: record it with ``nfl-ats
weak-signals record``, report ``probability_positive``, never the binary
"contains zero". The registry code hard-rejects inadmissible closures; if a
record command errors, the verdict is wrong, not the validator.

This script never writes to ``registry/``: recording happens through separate
``nfl-ats weak-signals record`` calls under the session's write-lock protocol.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from nfl_ats.cfb_benchmark import (  # noqa: E402
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_CLEAN_CORE_SEASONS,
    fit_cfb_residual_model,
)
from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS  # noqa: E402
from nfl_ats.clv import pick_correct  # noqa: E402
from nfl_ats.graph_team_stat_cfb_feature import (  # noqa: E402
    CFB_GRAPH_FROZEN_STRUCTURE,
    add_cfb_graph_team_stat_feature,
    cfb_graph_column,
)
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

# WP8 and WP24 are complete and frozen; everything reusable is imported from
# them rather than re-implemented, so the three experiments cannot drift apart
# on the shared machinery (table load, paired metric, week blocks, permutation,
# bootstrap summary, seed, era boundaries, the raw-triple rule).
from scripts.graph_team_stat_cfb_replacement import (  # noqa: E402
    raw_cell_columns,
    replacement_feature_columns,
)
from scripts.graph_team_stat_cfb_replication import (  # noqa: E402
    BOOTSTRAP_SAMPLES,
    DEFAULT_PERMUTATIONS,
    ERA_1,
    ERA_2,
    FEATURES_PATH,
    GRADES,
    SEED,
    _paired_metric,
    load_cfb_table,
    permuted_margins,
    summarize_pair,
    week_positions,
)

ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "graph_team_stat_cfb_offense_replacement"
PREDECLARATION = "docs/graph_team_stat_cfb_offense_replacement.md"

#: The two OFFENCE members of WP8's frozen three-cell list. Declared in section
#: 2 of the predeclaration and not reopened: no third metric may be added after
#: these are scored, and neither may be dropped after its sign is seen.
OFFENCE_METRICS: tuple[str, ...] = ("off_epa_per_play", "off_success_rate")

#: The positive control replaces EXACTLY ONE swapped-in column, named here in
#: the predeclaration (section 5) rather than chosen at run time. The arms that
#: do not carry it must reproduce their screen values exactly.
LEAK_METRIC = "off_epa_per_play"

ARM_NAMES = (
    "benchmark",
    "offense_replacement",
    "offense_ablation",
    "replacement_off_epa_per_play",
    "replacement_off_success_rate",
)

#: ``(label, reference_arm, candidate_arm)``. The first four are the declared
#: CELLS, in priority order; only the first votes on the decision rule.
CELL_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("cell1_primary_offense_two_metric_vs_benchmark", "benchmark", "offense_replacement"),
    (
        "cell2_continuity_off_epa_per_play_alone_vs_benchmark",
        "benchmark",
        "replacement_off_epa_per_play",
    ),
    (
        "cell3_continuity_off_success_rate_alone_vs_benchmark",
        "benchmark",
        "replacement_off_success_rate",
    ),
    ("cell4_offense_two_metric_vs_ablation", "offense_ablation", "offense_replacement"),
)

#: Report only. No registry row, and it cannot be promoted to one after its sign
#: is seen.
DIAGNOSTIC_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("diagnostic_report_only_offense_ablation_vs_benchmark", "benchmark", "offense_ablation"),
)

COMPARISONS = CELL_COMPARISONS + DIAGNOSTIC_COMPARISONS


# ---------------------------------------------------------------------------
# The contract substitution -- the whole point of this work package
# ---------------------------------------------------------------------------


def validate_metric(metric: str) -> None:
    """Refuse any metric outside this work package's declared offence pair."""

    if metric not in OFFENCE_METRICS:
        raise ValueError(
            f"{metric!r} is not one of the two declared offence metrics {OFFENCE_METRICS}; "
            f"{PREDECLARATION} section 2 froze the pair before any sign was computed and "
            "WP35 does not reopen it"
        )


def offence_raw_columns() -> tuple[str, ...]:
    """The six raw contract columns the two offence metrics own.

    ``raw_cell_columns`` is WP24's, imported unchanged: the contract generates a
    ``home_``/``away_``/``diff_`` triple as one atomic unit per metric, and
    ``diff_`` equals ``home_ - away_`` exactly, so a ``diff_``-only swap would
    test nothing.
    """

    columns: list[str] = []
    for metric in OFFENCE_METRICS:
        validate_metric(metric)
        columns.extend(raw_cell_columns(metric))
    return tuple(columns)


def offence_ablation_feature_columns() -> tuple[str, ...]:
    """The benchmark contract with BOTH offence triples removed, nothing added."""

    dropped = set(offence_raw_columns())
    return tuple(column for column in CFB_MODEL_FEATURE_COLUMNS if column not in dropped)


def offence_graph_columns(*, leak: bool = False) -> tuple[str, ...]:
    """The two swapped-in graph columns, in declared metric order.

    ``leak=True`` is the positive control and substitutes the realised
    ``ats_margin`` for EXACTLY ONE of them -- ``LEAK_METRIC``'s -- leaving the
    other graph column in place.
    """

    return tuple(
        "ats_margin" if (leak and metric == LEAK_METRIC) else cfb_graph_column(metric)
        for metric in OFFENCE_METRICS
    )


def offence_replacement_feature_columns(*, leak: bool = False) -> tuple[str, ...]:
    """Benchmark minus BOTH offence triples, plus BOTH offence graph columns."""

    return (*offence_ablation_feature_columns(), *offence_graph_columns(leak=leak))


def arm_feature_columns(*, leak: bool = False) -> dict[str, tuple[str, ...]]:
    """Feature contract per arm.

    The single-metric arms reuse WP24's ``replacement_feature_columns`` verbatim
    so that cells 2 and 3 are the SAME contract WP24 fitted, which is what makes
    them a continuity check rather than a new measurement.
    """

    return {
        "benchmark": CFB_MODEL_FEATURE_COLUMNS,
        "offense_replacement": offence_replacement_feature_columns(leak=leak),
        "offense_ablation": offence_ablation_feature_columns(),
        "replacement_off_epa_per_play": replacement_feature_columns(
            "off_epa_per_play", leak=leak and LEAK_METRIC == "off_epa_per_play"
        ),
        "replacement_off_success_rate": replacement_feature_columns(
            "off_success_rate", leak=leak and LEAK_METRIC == "off_success_rate"
        ),
    }


def add_offence_graph_columns(features: pd.DataFrame) -> pd.DataFrame:
    """Attach both offence graph katz differentials by chaining WP8's builder.

    The builder returns a copy in the caller's row order and index with exactly
    one column added, so chaining it is order-preserving; the test file asserts
    that rather than assuming it.
    """

    widened = features
    for metric in OFFENCE_METRICS:
        validate_metric(metric)
        widened = add_cfb_graph_team_stat_feature(widened, metric)
    return widened


# ---------------------------------------------------------------------------
# The evaluator (WP8's walk-forward, five arms)
# ---------------------------------------------------------------------------


def run_window(
    features: pd.DataFrame,
    seasons: tuple[int, ...],
    *,
    leak: bool = False,
) -> pd.DataFrame:
    """Walk forward over ``seasons``, returning one row per scored game.

    Each week is predicted from models trained strictly on games that kicked off
    before that week's earliest kickoff, with the benchmark's own 500-game
    training floor -- ``cfb_benchmark.cfb_walk_forward_benchmark``'s scheme, and
    WP8's and WP24's, unchanged.

    Probabilities rather than correctness are returned because the permutation
    null re-grades the SAME fitted models against many shuffled outcomes. Model
    fitting never sees the grading outcome, so a permutation changes only the
    grade -- which is what makes a 200-draw null affordable.

    Two grades are carried side by side: ``close`` (the frozen benchmark's own
    ``spread_line``) and ``open`` (``spread_open`` where present). The opener
    scoring frame has its ``spread_line`` REPLACED by the opener before
    prediction, so the market feature and the grade refer to the same number.
    """

    contracts = arm_feature_columns(leak=leak)
    completed = features.loc[
        pd.to_numeric(features["result"], errors="coerce").notna()
        & pd.to_numeric(features["ats_margin"], errors="coerce").notna()
    ].copy()
    window = completed.loc[completed["season"].isin(seasons)]

    collected: list[pd.DataFrame] = []
    n_weeks = 0
    for (_season, _week), group in window.groupby(["season", "week"], sort=True):
        cutoff = group["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < CFB_BENCHMARK_MIN_TRAIN_GAMES:
            continue

        at_close = group.copy()
        # A game with no opener quote is unscorable at the opener grade, not a
        # zero: it is left NaN and drops out of the opener comparison only.
        open_available = pd.to_numeric(group["spread_open"], errors="coerce").notna().to_numpy()
        at_open = group.loc[open_available].copy()
        at_open["spread_line"] = pd.to_numeric(at_open["spread_open"], errors="coerce")

        try:
            models = {
                name: fit_cfb_residual_model(training, feature_columns=columns)
                for name, columns in contracts.items()
            }
        except ValueError:
            continue
        n_weeks += 1

        row = pd.DataFrame(
            {
                "game_id": group["game_id"].to_numpy(),
                "season": group["season"].to_numpy(),
                "week": group["week"].to_numpy(),
                "settle_margin_close": (
                    pd.to_numeric(group["result"], errors="coerce")
                    - pd.to_numeric(group["spread_line"], errors="coerce")
                ).to_numpy(),
                "settle_margin_open": (
                    pd.to_numeric(group["result"], errors="coerce")
                    - pd.to_numeric(group["spread_open"], errors="coerce")
                ).to_numpy(),
            }
        )
        for name, model in models.items():
            row[f"{name}_probability_close"] = model.predict(at_close)[
                "home_cover_probability"
            ].to_numpy()
            opener_probability = np.full(len(group), np.nan, dtype=float)
            if len(at_open):
                opener_probability[open_available] = model.predict(at_open)[
                    "home_cover_probability"
                ].to_numpy(dtype=float)
            row[f"{name}_probability_open"] = opener_probability
        collected.append(row)

    if not collected:
        raise ValueError(f"no CFB week in {min(seasons)}-{max(seasons)} had enough prior training")
    print(f"  {n_weeks} weeks fitted over seasons {min(seasons)}-{max(seasons)}", flush=True)
    return pd.concat(collected, ignore_index=True)


def grade(
    frame: pd.DataFrame,
    which: str,
    margins: pd.Series | None = None,
    arms: Iterable[str] = ARM_NAMES,
) -> pd.DataFrame:
    """Attach ``<arm>_correct`` for one grade, defaulting to its own margins.

    ``pick_correct`` documents that a NaN settle margin is UNSETTLED, not a
    push, and that callers must mask such rows themselves. Rows with no
    probability or no settle margin are masked to NaN here and drop out of every
    comparison via ``dropna``.

    ``arms`` narrows the work to the two columns a permutation draw actually
    needs; the default grades all five.
    """

    settle = frame[f"settle_margin_{which}"] if margins is None else margins
    unsettled = settle.isna().to_numpy()
    graded = frame.copy()
    for arm in arms:
        probability = graded[f"{arm}_probability_{which}"]
        correct = pick_correct(probability.ge(0.5), settle)
        graded[f"{arm}_correct"] = correct.mask(unsettled | probability.isna().to_numpy())
    return graded


def null_distribution(
    frame: pd.DataFrame,
    which: str,
    reference: str,
    candidate: str,
    *,
    permutations: int,
    seed: int = SEED,
) -> dict[str, Any]:
    """The delta's null distribution over many within-week permutations.

    **This null is deliberately NOT centred on zero, and that is a property of
    the design rather than a defect.** Within-week permutation preserves each
    week's realised home-cover rate, so two arms with different home-pick rates
    have a non-zero expected null delta. It is the CONSERVATIVE reference and is
    reported ALONGSIDE the bootstrap-versus-zero interval, never instead of it.
    ONE permutation is a single draw, not a test.
    """

    rng = np.random.default_rng(seed)
    metric = _paired_metric(f"{reference}_correct", f"{candidate}_correct")
    groups = week_positions(frame)
    pair = (reference, candidate)
    deltas = [
        metric(grade(frame, which, permuted_margins(frame, which, rng, groups), arms=pair))[
            "delta_accuracy"
        ]
        for _ in range(permutations)
    ]
    values = np.asarray(deltas, dtype=float)
    finite = values[np.isfinite(values)]
    observed = metric(grade(frame, which, arms=pair))["delta_accuracy"]
    return {
        "permutations": len(finite),
        "null_mean_delta_points": float(finite.mean()) * 100.0,
        "null_sd_delta_points": float(finite.std(ddof=1)) * 100.0,
        "null_q025_points": float(np.quantile(finite, 0.025)) * 100.0,
        "null_q975_points": float(np.quantile(finite, 0.975)) * 100.0,
        "observed_delta_points": float(observed) * 100.0,
        "observed_percentile_of_null": float((finite < observed).mean()) * 100.0,
    }


def picks_moved(graded: pd.DataFrame, which: str, reference: str, candidate: str) -> dict[str, Any]:
    """How many forced picks the candidate arm actually flips.

    A delta near zero produced by a contract that moves almost nothing is a
    different fact from one produced by a contract that moves a tenth of the
    board, so the count is reported next to every delta rather than inferred.
    """

    left = graded[f"{reference}_probability_{which}"]
    right = graded[f"{candidate}_probability_{which}"]
    usable = left.notna() & right.notna()
    moved = (left.ge(0.5) != right.ge(0.5)) & usable
    return {
        "n_comparable": int(usable.sum()),
        "n_moved": int(moved.sum()),
        "fraction_moved": float(moved.sum() / usable.sum()) if usable.any() else float("nan"),
    }


def _print_pair(label: str, summary: dict[str, Any] | None) -> None:
    if summary is None:
        print(f"  {label:<54} no usable paired games")
        return
    low, high = summary["week_blocked_ci95_points"]
    print(
        f"  {label:<54}{summary['delta_accuracy_points']:>+9.3f} pts  "
        f"P+ {summary['week_blocked_probability_positive']:.3f}  "
        f"week 95% [{low:+.3f}, {high:+.3f}]  n={summary['n_games']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("null", "positive-control", "screen"),
        required=True,
        help="null and positive-control are instrument checks; screen is the real look",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=DEFAULT_PERMUTATIONS,
        help="within-week permutations for the null (a single draw is not a test)",
    )
    args = parser.parse_args()

    started = time.time()
    leak = args.mode == "positive-control"
    contracts = arm_feature_columns(leak=leak)

    print(f"=== CFB graph team_stat OFFENCE-ONLY replacement: mode={args.mode} ===")
    print(
        "sequential confirmation: the offence side was selected after WP24's diagnostic "
        "signs were seen (see the predeclaration section 1)"
    )
    features = load_cfb_table()
    print(
        f"CFB table: {len(features)} rows, seasons "
        f"{int(features['season'].min())}-{int(features['season'].max())}"
    )

    build_started = time.time()
    widened = add_offence_graph_columns(features)
    scored_mask = widened["season"].isin(CFB_CLEAN_CORE_SEASONS)
    print(f"built both offence graph columns in {time.time() - build_started:.1f}s")
    for metric in OFFENCE_METRICS:
        column = cfb_graph_column(metric)
        print(
            f"  {column}: non-null on {int(widened[column].notna().sum())}/{len(widened)} rows "
            f"overall, {int(widened.loc[scored_mask, column].notna().sum())}/"
            f"{int(scored_mask.sum())} in the scored clean core"
        )
    print(
        "contracts: "
        + "  ".join(f"{name}={len(columns)} cols" for name, columns in contracts.items())
    )
    print(f"removed from the offence arms: {offence_raw_columns()}")
    print(f"swapped in: {offence_graph_columns(leak=leak)}")

    fitted = run_window(widened, CFB_CLEAN_CORE_SEASONS, leak=leak)

    results: dict[str, Any] = {}
    if args.mode == "null":
        print(
            f"\nNULL CHECK ({args.permutations} within-week permutations, close grade -- the "
            "grade WP8's and WP24's null tables use, so the three are comparable):"
        )
        for label, reference, candidate in COMPARISONS:
            summary = null_distribution(
                fitted, "close", reference, candidate, permutations=args.permutations
            )
            results[label] = {"close": summary}
            print(
                f"  {label:<54}null mean {summary['null_mean_delta_points']:+8.3f} "
                f"sd {summary['null_sd_delta_points']:6.3f} "
                f"95% [{summary['null_q025_points']:+.3f}, {summary['null_q975_points']:+.3f}] "
                f"observed {summary['observed_delta_points']:+8.3f}"
            )
    else:
        for which in GRADES:
            graded = grade(fitted, which)
            marker = " (PRIMARY for the decision rule)" if which == "open" else ""
            print(f"\n--- grade: {which}{marker} ---")
            print(
                "  home-pick rate: "
                + "  ".join(
                    f"{arm}={graded[f'{arm}_probability_{which}'].ge(0.5).mean():.3f}"
                    for arm in ARM_NAMES
                )
            )
            print(
                "  arm accuracy: "
                + "  ".join(f"{arm}={graded[f'{arm}_correct'].mean():.6f}" for arm in ARM_NAMES)
            )
            for label, reference, candidate in COMPARISONS:
                summary = summarize_pair(graded, reference, candidate)
                if summary is not None:
                    summary["picks_moved"] = picks_moved(graded, which, reference, candidate)
                results.setdefault(label, {})[which] = summary
                _print_pair(label, summary)

        primary_label, primary_reference, primary_candidate = CELL_COMPARISONS[0]
        for which in GRADES:
            primary_null = null_distribution(
                fitted,
                which,
                primary_reference,
                primary_candidate,
                permutations=args.permutations,
            )
            results[primary_label][f"permutation_null_{which}"] = primary_null
            print(
                f"\n  permutation null ({which}, cell 1): "
                f"mean {primary_null['null_mean_delta_points']:+.3f} "
                f"sd {primary_null['null_sd_delta_points']:.3f}; observed at the "
                f"{primary_null['observed_percentile_of_null']:.1f}th percentile of its own null"
            )

        # Era MAGNITUDES, per the owner rule -- a weaker era is a smaller
        # number, never an absence. Report only, no extra registry rows. Both
        # grades, because the decision grade is the opener and the
        # commensurable grade is the close.
        era_results: dict[str, Any] = {}
        for which in GRADES:
            graded = grade(fitted, which)
            for era_label, (start, end) in (("2012_2019", ERA_1), ("2021_2025", ERA_2)):
                era_frame = graded.loc[graded["season"].between(start, end)]
                summary = summarize_pair(era_frame, primary_reference, primary_candidate)
                era_results.setdefault(era_label, {})[which] = summary
                print(f"\n--- era {start}-{end} ({which} grade, cell 1, report only) ---")
                _print_pair(f"era {era_label} {which}", summary)
        results["era_split_cell1_report_only"] = era_results

    configuration = {
        "mode": args.mode,
        "offence_metrics": list(OFFENCE_METRICS),
        "leak_metric": LEAK_METRIC,
        "graph_columns": [cfb_graph_column(metric) for metric in OFFENCE_METRICS],
        "replaced_raw_columns": list(offence_raw_columns()),
        "league": "cfb",
        "scored_seasons": list(CFB_CLEAN_CORE_SEASONS),
        "build_seasons": [int(features["season"].min()), int(features["season"].max())],
        "primary_grade": "open (result - spread_open) -- the decision rule's grade",
        "secondary_grade": "close (the frozen XLG-03 benchmark's own spread_line)",
        "frozen_structure": CFB_GRAPH_FROZEN_STRUCTURE,
        "min_train_games": CFB_BENCHMARK_MIN_TRAIN_GAMES,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "permutations": args.permutations,
        "seed": SEED,
        "predeclaration": PREDECLARATION,
        "sequential_confirmation": (
            "the offence side was selected after WP24's diagnostic signs were seen; this is a "
            "confirmation of a selected subset, not a first look"
        ),
        "cells": [label for label, _, _ in CELL_COMPARISONS],
        "report_only": [label for label, _, _ in DIAGNOSTIC_COMPARISONS],
        "arms": {name: list(columns) for name, columns in contracts.items()},
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": round(time.time() - started, 1),
        **configuration,
        "n_scored_games": len(fitted),
        "n_scored_weeks": int(fitted[["season", "week"]].drop_duplicates().shape[0]),
        "results": results,
        "provenance": artifact_provenance(configuration, FEATURES_PATH, project_root=REPO_ROOT),
    }
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = ARTIFACT_ROOT / args.mode / timestamp
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="graph-team-stat-cfb-offense-replacement",
        metrics={
            "mode": args.mode,
            "n_scored_games": len(fitted),
        },
        notes=(
            "CFB joint OFFENCE-side swap: both offence statistics' raw triples replaced by their "
            "graph katz differentials inside the XLG-03 benchmark contract; see "
            "docs/graph_team_stat_cfb_offense_replacement.md for the predeclared design and for "
            "the sequential-selection disclosure."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")
    print(f"done in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
