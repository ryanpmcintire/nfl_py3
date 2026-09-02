"""Is the graph ``team_stat`` column a better REPRESENTATION of a statistic
inside the XLG-03 CFB benchmark than that statistic's own raw columns? (WP24)

WP8 (``docs/graph_team_stat_cfb_replication.md``,
``scripts/graph_team_stat_cfb_replication.py``) measured ADDITION: the graph
column stacked on top of a benchmark that already carries the raw statistic. It
found the graph beats the raw differential as a single feature (+0.369 / +0.291
/ +0.694 accuracy points, week-blocked P+ 0.798 / 0.765 / 0.897) but is worth
about nothing added on top (-0.011 / +0.022 / -0.179, P+ 0.467 / 0.535 /
0.266).

Addition is not substitution. A column worth nothing as an addition can still
be worth something as a SWAP -- the add-on arm pays a ridge-penalty and
collinearity cost for carrying two encodings of the same football, and the
replacement arm does not. This script measures the swap, predeclared in
``docs/graph_team_stat_cfb_replacement.md`` BEFORE any outcome number for this
comparator was computed. Read that document first.

Three arms per scored week, all fitted with the benchmark's own estimator
(``nfl_ats.cfb_benchmark.fit_cfb_residual_model``, Ridge alpha 10, out-of-time
residual distribution, 500-game training floor):

* ``benchmark``   -- the frozen ``CFB_MODEL_FEATURE_COLUMNS`` contract (35 cols).
* ``replacement`` -- the same contract with the cell's ``home_``/``away_``/
  ``diff_`` columns removed and the cell's graph katz differential appended
  (33 cols).
* ``ablation``    -- the same contract with those three columns removed and
  nothing added (32 cols).

Why all THREE raw columns go, and not just ``diff_``: the contract generates
them as one atomic triple per metric (``cfb_features.CFB_TEAM_STATE_FEATURES``),
and -- measured on all 12,500 rows -- ``diff_<stat>`` equals ``home_<stat> -
away_<stat>`` exactly, so a linear ridge with the levels still present would
reconstruct the differential and a ``diff_``-only swap would test nothing.

Comparisons, in declared priority order:

1. PRIMARY   ``replacement`` - ``benchmark``: the question of this document.
2. secondary ``ablation`` - ``benchmark``: what the raw columns are worth at all.
3. secondary ``replacement`` - ``ablation``: what the graph column is worth once
   the raw columns are gone -- the arm that disambiguates the 3-for-1 swap.

Modes, to be run in this order:

* ``--mode null``             -- settle margins shuffled within each week.
* ``--mode positive-control`` -- the graph column in the ``replacement`` arm,
  and only that column, becomes the realised ``ats_margin``.
* ``--mode screen``           -- the real look, once per cell.

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
    CFB_GRAPH_CELLS,
    CFB_GRAPH_FROZEN_STRUCTURE,
    add_cfb_graph_team_stat_feature,
    cfb_graph_column,
)
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

# WP8 is complete and frozen; everything reusable is imported from it rather
# than re-implemented, so the two experiments cannot drift apart on the shared
# machinery (table load, paired metric, week blocks, permutation, bootstrap
# summary, seed, era boundaries).
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

ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "graph_team_stat_cfb_replacement"
PREDECLARATION = "docs/graph_team_stat_cfb_replacement.md"

ARM_NAMES = ("benchmark", "replacement", "ablation")

#: ``(label, reference_arm, candidate_arm)``. The first is the headline.
COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("primary_replacement_vs_benchmark", "benchmark", "replacement"),
    ("secondary_ablation_vs_benchmark", "benchmark", "ablation"),
    ("secondary_replacement_vs_ablation", "ablation", "replacement"),
)


# ---------------------------------------------------------------------------
# The contract substitution -- the whole point of this work package
# ---------------------------------------------------------------------------


def raw_cell_columns(cell: str) -> tuple[str, str, str]:
    """The three raw contract columns one cell owns.

    ``cfb_features.CFB_TEAM_STATE_FEATURES`` generates exactly this triple per
    metric, so this is the contract's own notion of "that statistic's columns".
    """

    if cell not in CFB_GRAPH_CELLS:
        raise ValueError(
            f"{cell!r} is not one of the three predeclared cells {CFB_GRAPH_CELLS}; "
            "docs/graph_team_stat_cfb_replication.md section 3 froze the cell list "
            "before any sign was computed and WP24 does not reopen it"
        )
    return (f"home_{cell}", f"away_{cell}", f"diff_{cell}")


def ablation_feature_columns(cell: str) -> tuple[str, ...]:
    """The benchmark contract with the cell's raw triple removed, nothing added."""

    dropped = set(raw_cell_columns(cell))
    return tuple(column for column in CFB_MODEL_FEATURE_COLUMNS if column not in dropped)


def replacement_feature_columns(cell: str, *, leak: bool = False) -> tuple[str, ...]:
    """The benchmark contract with the cell's raw triple swapped for the graph.

    ``leak=True`` is the positive control: the graph column -- and only the
    graph column -- becomes the realised ``ats_margin``.
    """

    graph_column = "ats_margin" if leak else cfb_graph_column(cell)
    return (*ablation_feature_columns(cell), graph_column)


def arm_feature_columns(cell: str, *, leak: bool = False) -> dict[str, tuple[str, ...]]:
    """Feature contract per arm."""

    return {
        "benchmark": CFB_MODEL_FEATURE_COLUMNS,
        "replacement": replacement_feature_columns(cell, leak=leak),
        "ablation": ablation_feature_columns(cell),
    }


# ---------------------------------------------------------------------------
# The evaluator (WP8's walk-forward, three arms instead of five)
# ---------------------------------------------------------------------------


def run_window(
    features: pd.DataFrame,
    cell: str,
    seasons: tuple[int, ...],
    *,
    leak: bool = False,
) -> pd.DataFrame:
    """Walk forward over ``seasons``, returning one row per scored game.

    Each week is predicted from models trained strictly on games that kicked off
    before that week's earliest kickoff, with the benchmark's own 500-game
    training floor -- ``cfb_benchmark.cfb_walk_forward_benchmark``'s scheme, and
    WP8's, unchanged.

    Probabilities rather than correctness are returned because the permutation
    null re-grades the SAME fitted models against many shuffled outcomes. Model
    fitting never sees the grading outcome, so a permutation changes only the
    grade -- which is what makes a 200-draw null affordable.

    Two grades are carried side by side: ``close`` (the frozen benchmark's own
    ``spread_line``) and ``open`` (``spread_open`` where present). The opener
    scoring frame has its ``spread_line`` REPLACED by the opener before
    prediction, so the market feature and the grade refer to the same number.
    """

    contracts = arm_feature_columns(cell, leak=leak)
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


def grade(frame: pd.DataFrame, which: str, margins: pd.Series | None = None) -> pd.DataFrame:
    """Attach ``<arm>_correct`` for one grade, defaulting to its own margins.

    ``pick_correct`` documents that a NaN settle margin is UNSETTLED, not a
    push, and that callers must mask such rows themselves. Rows with no
    probability or no settle margin are masked to NaN here and drop out of every
    comparison via ``dropna``.
    """

    settle = frame[f"settle_margin_{which}"] if margins is None else margins
    unsettled = settle.isna().to_numpy()
    graded = frame.copy()
    for arm in ARM_NAMES:
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
    deltas = [
        metric(grade(frame, which, permuted_margins(frame, which, rng, groups)))["delta_accuracy"]
        for _ in range(permutations)
    ]
    values = np.asarray(deltas, dtype=float)
    finite = values[np.isfinite(values)]
    observed = metric(grade(frame, which))["delta_accuracy"]
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
    different fact from one produced by a contract that moves a quarter of the
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
        print(f"  {label:<44} no usable paired games")
        return
    low, high = summary["week_blocked_ci95_points"]
    print(
        f"  {label:<44}{summary['delta_accuracy_points']:>+9.3f} pts  "
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
        "--cell",
        choices=CFB_GRAPH_CELLS,
        required=True,
        help="one of the three cells WP8 predeclared; WP24 does not reopen the list",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=DEFAULT_PERMUTATIONS,
        help="within-week permutations for the null (a single draw is not a test)",
    )
    args = parser.parse_args()

    started = time.time()
    cell = args.cell
    leak = args.mode == "positive-control"
    contracts = arm_feature_columns(cell, leak=leak)

    print(f"=== CFB graph team_stat REPLACEMENT: cell={cell} mode={args.mode} ===")
    features = load_cfb_table()
    print(
        f"CFB table: {len(features)} rows, seasons "
        f"{int(features['season'].min())}-{int(features['season'].max())}"
    )

    build_started = time.time()
    widened = add_cfb_graph_team_stat_feature(features, cell)
    graph_column = cfb_graph_column(cell)
    scored_mask = widened["season"].isin(CFB_CLEAN_CORE_SEASONS)
    print(
        f"built {graph_column} in {time.time() - build_started:.1f}s; non-null on "
        f"{int(widened[graph_column].notna().sum())}/{len(widened)} rows overall, "
        f"{int(widened.loc[scored_mask, graph_column].notna().sum())}/"
        f"{int(scored_mask.sum())} in the scored clean core"
    )
    print(
        "contracts: "
        + "  ".join(f"{name}={len(columns)} cols" for name, columns in contracts.items())
        + f"; removed {raw_cell_columns(cell)}"
    )

    fitted = run_window(widened, cell, CFB_CLEAN_CORE_SEASONS, leak=leak)

    results: dict[str, Any] = {}
    if args.mode == "null":
        print(
            f"\nNULL CHECK ({args.permutations} within-week permutations, close grade): "
            "the distribution must be finite and its spread is the honest scale."
        )
        for label, reference, candidate in COMPARISONS:
            summary = null_distribution(
                fitted, "close", reference, candidate, permutations=args.permutations
            )
            results[label] = {"close": summary}
            print(
                f"  {label:<44}null mean {summary['null_mean_delta_points']:+8.3f} "
                f"sd {summary['null_sd_delta_points']:6.3f} "
                f"95% [{summary['null_q025_points']:+.3f}, {summary['null_q975_points']:+.3f}] "
                f"observed {summary['observed_delta_points']:+8.3f}"
            )
    else:
        for which in GRADES:
            graded = grade(fitted, which)
            print(f"\n--- grade: {which} ---")
            print(
                "  home-pick rate: "
                + "  ".join(
                    f"{arm}={graded[f'{arm}_probability_{which}'].ge(0.5).mean():.3f}"
                    for arm in ARM_NAMES
                )
            )
            for label, reference, candidate in COMPARISONS:
                summary = summarize_pair(graded, reference, candidate)
                if summary is not None:
                    summary["picks_moved"] = picks_moved(graded, which, reference, candidate)
                results.setdefault(label, {})[which] = summary
                _print_pair(label, summary)

        graded_close = grade(fitted, "close")
        primary_label, primary_reference, primary_candidate = COMPARISONS[0]
        primary_null = null_distribution(
            fitted,
            "close",
            primary_reference,
            primary_candidate,
            permutations=args.permutations,
        )
        results[primary_label]["permutation_null_close"] = primary_null
        print(
            f"\n  permutation null (close, primary): "
            f"mean {primary_null['null_mean_delta_points']:+.3f} "
            f"sd {primary_null['null_sd_delta_points']:.3f}; observed at the "
            f"{primary_null['observed_percentile_of_null']:.1f}th percentile of its own null"
        )

        # Era MAGNITUDES, per the owner rule -- a weaker era is a smaller
        # number, never an absence. Report only, no extra registry rows.
        era_results: dict[str, Any] = {}
        for era_label, (start, end) in (("2012_2019", ERA_1), ("2021_2025", ERA_2)):
            era_frame = graded_close.loc[graded_close["season"].between(start, end)]
            era_results[era_label] = summarize_pair(era_frame, primary_reference, primary_candidate)
            print(f"\n--- era {start}-{end} (report only, close grade, primary comparison) ---")
            _print_pair(f"era {era_label}", era_results[era_label])
        results["era_split_primary_close_report_only"] = era_results

    configuration = {
        "mode": args.mode,
        "cell": cell,
        "graph_column": graph_column,
        "replaced_raw_columns": list(raw_cell_columns(cell)),
        "league": "cfb",
        "scored_seasons": list(CFB_CLEAN_CORE_SEASONS),
        "build_seasons": [int(features["season"].min()), int(features["season"].max())],
        "primary_grade": "close (the frozen XLG-03 benchmark's own spread_line)",
        "secondary_grade": "open (spread_open where present)",
        "frozen_structure": CFB_GRAPH_FROZEN_STRUCTURE,
        "min_train_games": CFB_BENCHMARK_MIN_TRAIN_GAMES,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "permutations": args.permutations,
        "seed": SEED,
        "predeclaration": PREDECLARATION,
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
    output_dir = ARTIFACT_ROOT / f"{cell}_{args.mode}" / timestamp
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="graph-team-stat-cfb-replacement",
        metrics={
            "mode": args.mode,
            "cell": cell,
            "n_scored_games": len(fitted),
        },
        notes=(
            "CFB graph team_stat as a REPLACEMENT input for the raw statistic inside the "
            "XLG-03 benchmark contract; see docs/graph_team_stat_cfb_replacement.md for "
            "the predeclared design."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")
    print(f"done in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
