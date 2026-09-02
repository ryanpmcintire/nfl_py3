"""Does the graph ``team_stat`` transform add anything to a market-residual
model at all? Asked on college football, where the schedule graph is ~5x
sparser and conference-clustered, and where no NFL evaluation window is spent.

Predeclared in ``docs/graph_team_stat_cfb_replication.md`` BEFORE any outcome
sign was computed. Read that document first -- it declares the three cells, the
comparator, the grade, the frozen structural configuration, the two
adaptations, the decision rule and the recording rules.

Five arms per scored week, all fitted with the XLG-03 CFB benchmark's own
estimator (``nfl_ats.cfb_benchmark.fit_cfb_residual_model``, Ridge alpha 10,
out-of-time residual distribution, 500-game training floor) so every number
here is commensurable with that published instrument:

* ``market``               -- ``fit_market_baseline``, zero features.
* ``raw_only``             -- single feature: the raw ``diff_<cell>``.
* ``graph_only``           -- single feature: the graph katz differential.
* ``benchmark``            -- the frozen ``CFB_MODEL_FEATURE_COLUMNS`` contract.
* ``benchmark_plus_graph`` -- the same contract plus the one graph column.

The PRIMARY comparison is ``benchmark_plus_graph`` minus ``benchmark``: the CFB
analogue of the NFL "on production" question, the one that went negative on
NFL. It is deliberately strict -- ``CFB_MODEL_FEATURE_COLUMNS`` already
contains ``home_<cell>``, ``away_<cell>`` and ``diff_<cell>``, so the graph is
never credited for what the raw statistic already earned. Two secondaries
mirror the two NFL bare-baseline directions: ``graph_only`` minus ``raw_only``
(the ``graph_team_stat_screen`` direction) and ``graph_only`` minus ``market``
(the ``graph_input_screen`` direction).

Modes, to be run in this order:

* ``--mode null``             -- settle margins shuffled within each week. A
  harness that reports a real effect here is broken.
* ``--mode positive-control`` -- the graph column is replaced by the realised
  ``ats_margin``, a deliberate leak. A harness that CANNOT see it is blind, and
  a null from it would mean nothing.
* ``--mode screen``           -- the real look, once per cell.

Closing-grounds taxonomy (binding, AGENTS.md, restated verbatim so this file
stands on its own): an interval or CI that contains zero is NEVER grounds to
reject, fail, or close an experiment. At this evaluator's ~2-point resolution,
"contains zero" is the EXPECTED outcome for a real small signal. Only two
grounds ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong
sign (whole interval on the wrong side of zero) or zero split-half
reliability; (2) bounded by a positive control proven able to detect an effect
that size. Everything else is ``unresolved_below_power``: record it with
``nfl-ats weak-signals record``, report ``probability_positive``, never the
binary "contains zero". The registry code hard-rejects inadmissible closures;
if a record command errors, the verdict is wrong, not the validator.

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
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_ats.cfb_benchmark import (  # noqa: E402
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_CLEAN_CORE_SEASONS,
    fit_cfb_residual_model,
)
from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS  # noqa: E402
from nfl_ats.clv import pick_correct, week_blocked_bootstrap  # noqa: E402
from nfl_ats.graph_team_stat_cfb_feature import (  # noqa: E402
    CFB_GRAPH_CELL_NFL_COUNTERPART,
    CFB_GRAPH_CELLS,
    CFB_GRAPH_FROZEN_STRUCTURE,
    add_cfb_graph_team_stat_feature,
    cfb_graph_column,
)
from nfl_ats.margin import fit_market_baseline  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

SEED = 20260901
BOOTSTRAP_SAMPLES = 1_000
DEFAULT_PERMUTATIONS = 200

FEATURES_PATH = REPO_ROOT / "data" / "processed" / "cfb_game_features.parquet"
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "graph_team_stat_cfb_replication"
PREDECLARATION = "docs/graph_team_stat_cfb_replication.md"

#: Report-only era split. The clean core straddles the 2020 season the XLG-03
#: benchmark itself excludes, which is the obvious boundary. Owner rule: era
#: MAGNITUDE, not presence -- a weaker era is never reported as an absence.
ERA_1 = (2012, 2019)
ERA_2 = (2021, 2025)

ARM_NAMES = ("market", "raw_only", "graph_only", "benchmark", "benchmark_plus_graph")

#: ``(label, reference_arm, candidate_arm)``. The first is the headline.
COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("primary_benchmark_plus_graph_vs_benchmark", "benchmark", "benchmark_plus_graph"),
    ("secondary_graph_only_vs_raw_only", "raw_only", "graph_only"),
    ("secondary_graph_only_vs_market", "market", "graph_only"),
)

GRADES = ("close", "open")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_cfb_table() -> pd.DataFrame:
    """The XLG-03 canonical CFB game table, chronologically ordered."""

    frame = pd.read_parquet(FEATURES_PATH)
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["week"] = pd.to_numeric(frame["week"], errors="raise").astype(int)
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    return frame.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def arm_feature_columns(cell: str, *, leak: bool) -> dict[str, tuple[str, ...] | None]:
    """Feature contract per arm. ``None`` marks the zero-feature market arm.

    ``leak=True`` is the positive control: the graph column -- and only the
    graph column -- is swapped for the realised ``ats_margin``.
    """

    graph_column = "ats_margin" if leak else cfb_graph_column(cell)
    return {
        "market": None,
        "raw_only": (f"diff_{cell}",),
        "graph_only": (graph_column,),
        "benchmark": CFB_MODEL_FEATURE_COLUMNS,
        "benchmark_plus_graph": (*CFB_MODEL_FEATURE_COLUMNS, graph_column),
    }


# ---------------------------------------------------------------------------
# The evaluator
# ---------------------------------------------------------------------------


def run_window(
    features: pd.DataFrame,
    cell: str,
    seasons: tuple[int, ...],
    *,
    leak: bool = False,
) -> pd.DataFrame:
    """Walk forward over ``seasons``, returning one row per scored game.

    Each week is predicted from models trained strictly on games that kicked
    off before that week's earliest kickoff, exactly as
    ``cfb_benchmark.cfb_walk_forward_benchmark`` does, with the same 500-game
    training floor.

    Probabilities rather than correctness are returned because the permutation
    null re-grades the SAME fitted models against many shuffled outcomes. Model
    fitting never sees the grading outcome, so a permutation changes only the
    grade -- which is what makes a 200-draw null affordable.

    Two grades are carried side by side: ``close`` (the frozen benchmark's own
    ``spread_line``) and ``open`` (``spread_open`` where present). The opener
    scoring frame has its ``spread_line`` REPLACED by the opener before
    prediction, so the market feature and the grade refer to the same number --
    the convention ``nfl_ats.clv``'s opener evaluation uses.
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
        # Measured: 17 of the 9,093 clean-core rows carry no spread_open. A
        # game without an opener is unscorable at the opener grade, not a
        # zero -- it is left NaN and drops out of the opener comparison only.
        open_available = pd.to_numeric(group["spread_open"], errors="coerce").notna().to_numpy()
        at_open = group.loc[open_available].copy()
        at_open["spread_line"] = pd.to_numeric(at_open["spread_open"], errors="coerce")

        try:
            models = {
                name: (
                    fit_market_baseline(training)
                    if columns is None
                    else fit_cfb_residual_model(training, feature_columns=columns)
                )
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
    push, and that callers must mask such rows themselves -- otherwise a game
    with no opener quote would silently score as an away pick against a NaN
    margin. Rows with no probability or no settle margin are masked to NaN here
    and drop out of every comparison via ``dropna``.
    """

    settle = frame[f"settle_margin_{which}"] if margins is None else margins
    unsettled = settle.isna().to_numpy()
    graded = frame.copy()
    for arm in ARM_NAMES:
        probability = graded[f"{arm}_probability_{which}"]
        correct = pick_correct(probability.ge(0.5), settle)
        graded[f"{arm}_correct"] = correct.mask(unsettled | probability.isna().to_numpy())
    return graded


def _paired_metric(reference: str, candidate: str) -> Any:
    def metric(df: pd.DataFrame) -> dict[str, float]:
        valid = df.dropna(subset=[reference, candidate])
        if valid.empty:
            return {
                "delta_accuracy": float("nan"),
                "candidate_accuracy": float("nan"),
                "reference_accuracy": float("nan"),
            }
        return {
            "delta_accuracy": float((valid[candidate] - valid[reference]).mean()),
            "candidate_accuracy": float(valid[candidate].mean()),
            "reference_accuracy": float(valid[reference].mean()),
        }

    return metric


def summarize_pair(graded: pd.DataFrame, reference: str, candidate: str) -> dict[str, Any] | None:
    """Point estimate plus week-blocked (primary) and season-blocked
    (secondary) bootstrap for one paired comparison.

    Within-week game correlation is zero by owner mandate, so the week block is
    the honest primary and the season block is reported beside it, never
    averaged with it.
    """

    reference_column = f"{reference}_correct"
    candidate_column = f"{candidate}_correct"
    usable = graded.dropna(subset=[reference_column, candidate_column])
    if usable.empty:
        return None
    metric = _paired_metric(reference_column, candidate_column)
    point = metric(usable)
    week = week_blocked_bootstrap(
        usable, metric, block="week", samples=BOOTSTRAP_SAMPLES, seed=SEED
    )
    season = week_blocked_bootstrap(
        usable, metric, block="season", samples=BOOTSTRAP_SAMPLES, seed=SEED
    )
    week_row = week.loc[week["metric"].eq("delta_accuracy")].iloc[0]
    season_row = season.loc[season["metric"].eq("delta_accuracy")].iloc[0]
    return {
        "delta_accuracy_points": point["delta_accuracy"] * 100.0,
        "candidate_accuracy": point["candidate_accuracy"],
        "reference_accuracy": point["reference_accuracy"],
        "week_blocked_ci95_points": [
            float(week_row["lower"]) * 100.0,
            float(week_row["upper"]) * 100.0,
        ],
        "week_blocked_probability_positive": float(week_row["probability_positive"]),
        "season_blocked_ci95_points": [
            float(season_row["lower"]) * 100.0,
            float(season_row["upper"]) * 100.0,
        ],
        "season_blocked_probability_positive": float(season_row["probability_positive"]),
        "n_games": len(usable),
        "n_weeks": int(usable[["season", "week"]].drop_duplicates().shape[0]),
        "n_seasons": int(usable["season"].nunique()),
    }


def week_positions(frame: pd.DataFrame) -> list[np.ndarray]:
    """Row positions of each week, computed once and reused by every draw --
    the groupby is the expensive part, not the shuffle."""

    return [
        np.asarray(positions, dtype=np.intp)
        for positions in frame.groupby(["season", "week"], sort=False).indices.values()
    ]


def permuted_margins(
    frame: pd.DataFrame, which: str, rng: np.random.Generator, groups: list[np.ndarray]
) -> pd.Series:
    """Settle margins shuffled WITHIN each week: destroys the pick-to-outcome
    pairing while preserving week structure and the picks themselves."""

    values = frame[f"settle_margin_{which}"].to_numpy(dtype=float, copy=True)
    for positions in groups:
        values[positions] = rng.permutation(values[positions])
    return pd.Series(values, index=frame.index)


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
    have a non-zero expected null delta. It is the CONSERVATIVE reference --
    it treats week-level home tilt as noise -- and it is reported ALONGSIDE the
    bootstrap-versus-zero interval, never instead of it. ONE permutation is a
    single draw, not a test; what a null check shows is where the DISTRIBUTION
    sits and how wide it is.
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


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_pair(label: str, summary: dict[str, Any] | None) -> None:
    if summary is None:
        print(f"  {label:<46} no usable paired games")
        return
    low, high = summary["week_blocked_ci95_points"]
    print(
        f"  {label:<46}{summary['delta_accuracy_points']:>+9.3f} pts  "
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
        help="one of the three predeclared CFB team-stat cells",
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

    print(f"=== CFB graph team_stat replication: cell={cell} mode={args.mode} ===")
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
                f"  {label:<46}null mean {summary['null_mean_delta_points']:+8.3f} "
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
        "nfl_counterpart": CFB_GRAPH_CELL_NFL_COUNTERPART[cell],
        "graph_column": graph_column,
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
        "arms": {
            name: (list(cols) if cols else [])
            for name, cols in arm_feature_columns(cell, leak=leak).items()
        },
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
        command="graph-team-stat-cfb-replication",
        metrics={
            "mode": args.mode,
            "cell": cell,
            "n_scored_games": len(fitted),
        },
        notes=(
            "CFB cross-league replication of the graph team_stat arm; see "
            "docs/graph_team_stat_cfb_replication.md for the predeclared design."
        ),
    )
    print(f"\nwrote {output_dir / 'results.json'}")
    print(f"done in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
