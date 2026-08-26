"""Graph ratings v2, `team_stat` arm: does opponent-adjusting a screened
statistic beat using that statistic raw?

Predeclared in ``docs/graph_ratings_v2_screen.md`` BEFORE this script was
pointed at an outcome column. Read that document first -- it declares the
arms, the frozen structural hyperparameters and the reason they are frozen,
the grade, the window discipline, and the recording rules.

Three arms per family, one evaluator, one window:

* ``baseline``  -- market only (``fit_market_baseline``), the same reference
  the input screen used.
* ``control``   -- the raw ``home_<family> - away_<family>`` differential. This
  is the COMPARATOR OF FIRST RESORT: it is exactly what
  ``scripts/graph_input_screen.py`` already scored, so the graph is never
  credited for what the raw statistic already earned.
* ``treatment`` -- the ``team_stat`` graph's ``*_katz_diff`` column for the
  same family, at the frozen config.

The headline quantity is the PAIRED treatment-minus-control delta. Model
class, evaluator and blocking are imported from ``graph_input_screen`` rather
than reimplemented, so the three arms and the input screen's own published
numbers are commensurable by construction.

Instrument checks that run BEFORE any window is spent (``--mode``):

* ``null``             -- settle margins shuffled within each week. A harness
  that reports an effect here is broken.
* ``positive-control`` -- the treatment feature is replaced by the realized
  ``ats_margin``, a deliberate leak. A harness that CANNOT detect this is
  blind, and a "no effect" from it would mean nothing.
* ``screen``           -- the real look. Spends the family's assigned rotation
  window; only run it once, and only after both checks above pass.

Closing-grounds taxonomy (binding, restated per AGENTS.md so this file stands
on its own): an interval containing zero is NEVER grounds to reject, fail or
close an experiment. Only a RESOLVED wrong sign (whole interval on the wrong
side of zero), zero split-half reliability, or a positive control proven able
to detect an effect that size closes a line of work. Everything else is
``unresolved_below_power``: record it and report ``probability_positive``,
never the binary "contains zero".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import reliability_map as relmap  # noqa: E402
from graph_input_screen import (  # noqa: E402
    BOOTSTRAP_SAMPLES,
    RIDGE_ALPHA,
    SEED,
    fit_single_feature_market_residual_model,
)

from nfl_ats.clv import pick_correct, week_blocked_bootstrap  # noqa: E402
from nfl_ats.constants import MIN_FITTABLE_TRAIN_GAMES  # noqa: E402
from nfl_ats.graph_ratings_v2 import (  # noqa: E402
    GraphRatingV2Config,
    add_graph_ratings_v2_features,
    katz_feature_columns,
)
from nfl_ats.margin import fit_market_baseline  # noqa: E402
from nfl_ats.modeling import regular_season_rows  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

# Frozen in docs/graph_ratings_v2_screen.md section 5, before scoring, and not
# retuned on NFL. These are the module defaults, which are also the only
# structural setting the CFB grid actually resolved (+0.531 coherence on the
# raw-margin control; every residual-arm reading sat within +-0.006 of zero).
FROZEN_STRUCTURE: dict[str, Any] = {
    "alpha": 0.85,
    "half_life_weeks": 8.0,
    "max_row_l1": 1.0,
    "prior_weight": 1.0,
    "min_games": 16,
    "propagation": "signed_katz",
    "injury_beta": 0.0,
}

SCREEN_ARTIFACT = REPO_ROOT / "artifacts/graph_input_screen/20260826T163208Z/results.json"
CONTROL_PREFIX = "gts_control_"
TREATMENT_PREFIX = "gts_treatment_"


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def load_representatives(path: Path = SCREEN_ARTIFACT) -> list[str]:
    """The input screen's cluster representatives, in its own ranked order.

    Reading the screen artifact is reuse of an existing, already-recorded
    measurement, not a fresh selection step -- the ranking was fixed and
    published by that run.
    """

    screen = json.loads(path.read_text(encoding="utf-8"))
    reps = [
        (name, family)
        for name, family in screen["families"].items()
        if family.get("is_cluster_representative") and family.get("status") == "survivor"
    ]
    reps.sort(key=lambda item: -item[1]["gate2"]["holdout"]["week_blocked_probability_positive"])
    return [name for name, _ in reps]


def build_arm_columns(
    features: pd.DataFrame, family_names: list[str]
) -> tuple[pd.DataFrame, dict[str, dict[str, str]], list[str]]:
    """Attach the control and treatment columns for every family.

    Returns the widened frame, a ``{family: {"control": col, "treatment": col}}``
    map, and the families that could not be built (reported, never silently
    dropped).
    """

    dtypes = {column: features[column].dtype for column in features.columns}
    pairs, _excluded = relmap.discover_family_pairs(list(features.columns), dtypes)

    completed = features.loc[features["result"].notna()].copy()
    arms: dict[str, dict[str, str]] = {}
    skipped: list[str] = []
    additions: dict[str, pd.Series] = {}

    for name in family_names:
        pair = pairs.get(name)
        if pair is None:
            skipped.append(name)
            continue
        home_column, away_column = pair[0], pair[1]
        config = GraphRatingV2Config(
            edge_signal="team_stat",
            signal_column=name,
            signal_column_pair=(home_column, away_column),
            **FROZEN_STRUCTURE,
        )
        started = time.time()
        rated = add_graph_ratings_v2_features(completed, config)
        diff_column = katz_feature_columns(config)[2]

        control = pd.to_numeric(features[home_column], errors="coerce") - pd.to_numeric(
            features[away_column], errors="coerce"
        )
        treatment = pd.Series(np.nan, index=features.index, dtype=float)
        treatment.loc[rated.index] = pd.to_numeric(rated[diff_column], errors="coerce")

        additions[f"{CONTROL_PREFIX}{name}"] = control
        additions[f"{TREATMENT_PREFIX}{name}"] = treatment
        arms[name] = {
            "control": f"{CONTROL_PREFIX}{name}",
            "treatment": f"{TREATMENT_PREFIX}{name}",
            "home_column": home_column,
            "away_column": away_column,
        }
        print(f"  built {name:<42} {time.time() - started:5.1f}s", flush=True)

    widened = pd.concat([features, pd.DataFrame(additions, index=features.index)], axis=1)
    return widened, arms, skipped


# ---------------------------------------------------------------------------
# The evaluator
# ---------------------------------------------------------------------------


def run_window(
    features: pd.DataFrame,
    arms: dict[str, dict[str, str]],
    seasons: tuple[int, ...],
    *,
    leak_treatment: bool = False,
) -> dict[str, pd.DataFrame]:
    """Close-graded per-game rows: the realized settle margin plus one home-cover
    probability per arm (baseline, control, treatment).

    Forward-chaining only -- each week is predicted from a model trained
    strictly on games that kicked off before that week's earliest kickoff,
    exactly as ``graph_input_screen.run_close_graded_window`` does.

    Probabilities rather than correctness are returned because the null check
    needs to re-grade the SAME fitted models against many permuted outcomes.
    Model fitting never sees the grading outcome, so a permutation changes only
    the grade -- which is what makes a 200-permutation null affordable at all.

    ``leak_treatment=True`` swaps the treatment feature for the realized
    ``ats_margin``: the positive control, a deliberate leak whose effect the
    instrument must be able to see.
    """

    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    frame = frame.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    completed = frame.loc[frame["result"].notna()].copy()
    window = completed.loc[completed["season"].astype(int).isin(seasons)]

    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in arms}
    n_weeks = 0
    for (_season, _week), group in window.groupby(["season", "week"], sort=True):
        cutoff = group["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < MIN_FITTABLE_TRAIN_GAMES:
            continue
        n_weeks += 1

        settle_margin = pd.to_numeric(group["result"], errors="coerce") - pd.to_numeric(
            group["spread_line"], errors="coerce"
        )
        baseline = fit_market_baseline(training)
        baseline_probability = baseline.predict(group)["home_cover_probability"]

        for name, columns in arms.items():
            treatment_column = "ats_margin" if leak_treatment else columns["treatment"]
            try:
                control_model = fit_single_feature_market_residual_model(
                    training, columns["control"]
                )
                treatment_model = fit_single_feature_market_residual_model(
                    training, treatment_column
                )
            except ValueError:
                continue
            control_probability = control_model.predict(group)["home_cover_probability"]
            treatment_probability = treatment_model.predict(group)["home_cover_probability"]
            for game_id, season_value, week_value, margin, base, ctrl, treat in zip(
                group["game_id"],
                group["season"],
                group["week"],
                settle_margin,
                baseline_probability,
                control_probability,
                treatment_probability,
                strict=True,
            ):
                rows[name].append(
                    {
                        "game_id": game_id,
                        "season": int(str(season_value)),
                        "week": int(str(week_value)),
                        "settle_margin": margin,
                        "baseline_probability": base,
                        "control_probability": ctrl,
                        "treatment_probability": treat,
                    }
                )
    print(f"  {n_weeks} weeks fitted over seasons {min(seasons)}-{max(seasons)}")
    return {name: pd.DataFrame(collected) for name, collected in rows.items()}


ARM_PROBABILITY = {
    "baseline": "baseline_probability",
    "control": "control_probability",
    "treatment": "treatment_probability",
}


def grade(frame: pd.DataFrame, margins: pd.Series | None = None) -> pd.DataFrame:
    """Attach ``<arm>_correct`` for every arm, graded against ``margins``
    (default: the realized settle margin)."""

    settle = frame["settle_margin"] if margins is None else margins
    graded = frame.copy()
    for arm, column in ARM_PROBABILITY.items():
        graded[f"{arm}_correct"] = pick_correct(graded[column].ge(0.5), settle)
    return graded


def week_positions(frame: pd.DataFrame) -> list[np.ndarray]:
    """Row positions of each week, computed once and reused by every
    permutation -- the groupby is the expensive part, not the shuffle."""

    return [
        np.asarray(positions, dtype=np.intp)
        for positions in frame.groupby(["season", "week"], sort=False).indices.values()
    ]


def permuted_margins(
    frame: pd.DataFrame, rng: np.random.Generator, groups: list[np.ndarray]
) -> pd.Series:
    """Settle margins shuffled WITHIN each week.

    This destroys any within-week pairing of pick to outcome while preserving
    the week structure and the picks. It also PRESERVES each week's realized
    home-cover count, which is exactly why the resulting null is not centred on
    zero -- see the docstring of :func:`null_distribution`.
    """

    values = frame["settle_margin"].to_numpy(dtype=float, copy=True)
    for positions in groups:
        values[positions] = rng.permutation(values[positions])
    return pd.Series(values, index=frame.index)


def null_distribution(
    frame: pd.DataFrame,
    reference: str,
    candidate: str,
    *,
    permutations: int,
    seed: int = SEED,
) -> dict[str, Any]:
    """The delta's null distribution over many within-week permutations.

    **This null is deliberately NOT centred on zero, and that is a property of
    the design rather than a defect.** Within-week permutation preserves each
    week's realized home-cover rate ``c_w``, so an arm picking home at rate
    ``h_w`` that week has expected null accuracy ``1 - h_w - c_w + 2*h_w*c_w``,
    and the expected null DELTA between two arms is
    ``2 * mean_w[(h_w_treat - h_w_control) * (c_w - 0.5)]``. Measured on the
    2012-2014 smoke window, that closed form reproduced the Monte-Carlo null
    means to within ~0.3 points (-1.856 vs -1.892, -1.400 vs -1.266, -1.111 vs
    -0.842, -0.189 vs -0.216), because these arms carry large and differing
    home-pick rates (55-67% home against a 49.67% cover rate). Consequence for
    reading a result: this null is the CONSERVATIVE reference -- it treats
    week-level home-tilt as noise -- and it is reported ALONGSIDE the
    bootstrap-versus-zero interval the rest of the project uses, never instead
    of it.

    ONE permutation is a single draw, not a test -- an earlier version of this
    check read a single draw of -2.53 points as a broken harness. What a null
    check must show is that the DISTRIBUTION is centred on zero; the spread it
    reveals is also the honest scale against which the real delta is read.
    """

    rng = np.random.default_rng(seed)
    metric = _paired_metric(f"{reference}_correct", f"{candidate}_correct")
    groups = week_positions(frame)
    deltas = []
    for _ in range(permutations):
        graded = grade(frame, permuted_margins(frame, rng, groups))
        deltas.append(metric(graded)["delta_accuracy"])
    values = np.asarray(deltas, dtype=float)
    finite = values[np.isfinite(values)]
    observed = metric(grade(frame))["delta_accuracy"]
    return {
        "permutations": len(finite),
        "null_mean_delta": float(finite.mean()),
        "null_sd_delta": float(finite.std(ddof=1)),
        "null_q025": float(np.quantile(finite, 0.025)),
        "null_q975": float(np.quantile(finite, 0.975)),
        "observed_delta": float(observed),
        "fraction_of_null_below_observed": float((finite < observed).mean()),
    }


def _paired_metric(reference: str, candidate: str):
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


def summarize_pair(paired: pd.DataFrame, reference: str, candidate: str) -> dict[str, Any] | None:
    """Point estimate plus week- and season-blocked bootstrap for one paired
    comparison. Within-week game correlation is zero by owner mandate, so the
    week block is the honest primary and the season block is reported beside
    it, never averaged with it.
    """

    if paired.empty or paired.dropna(subset=[reference, candidate]).empty:
        return None
    metric = _paired_metric(reference, candidate)
    point = metric(paired)
    week = week_blocked_bootstrap(
        paired, metric, block="week", samples=BOOTSTRAP_SAMPLES, seed=SEED
    )
    season = week_blocked_bootstrap(
        paired, metric, block="season", samples=BOOTSTRAP_SAMPLES, seed=SEED
    )
    week_row = week.loc[week["metric"].eq("delta_accuracy")].iloc[0]
    season_row = season.loc[season["metric"].eq("delta_accuracy")].iloc[0]
    return {
        "delta_accuracy": point["delta_accuracy"],
        "candidate_accuracy": point["candidate_accuracy"],
        "reference_accuracy": point["reference_accuracy"],
        "week_blocked_ci95": [float(week_row["lower"]), float(week_row["upper"])],
        "week_blocked_probability_positive": float(week_row["probability_positive"]),
        "season_blocked_ci95": [float(season_row["lower"]), float(season_row["upper"])],
        "season_blocked_probability_positive": float(season_row["probability_positive"]),
        "n_games": len(paired.dropna(subset=[reference, candidate])),
        "n_weeks": int(paired[["season", "week"]].drop_duplicates().shape[0]),
        "n_seasons": int(paired["season"].nunique()),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("null", "positive-control", "screen"),
        required=True,
        help="null and positive-control are instrument checks; screen is the real look",
    )
    parser.add_argument(
        "--seasons",
        required=True,
        help="inclusive season range, e.g. 2012-2014 (the rotation-assigned window)",
    )
    parser.add_argument(
        "--families",
        default="",
        help="comma-separated subset; default is every cluster representative",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=200,
        help="within-week permutations for --mode null (a single draw is not a test)",
    )
    parser.add_argument("--out", default="", help="artifact directory name override")
    args = parser.parse_args()

    start, _, end = args.seasons.partition("-")
    seasons = tuple(range(int(start), int(end or start) + 1))

    representatives = load_representatives()
    if args.families:
        wanted = {name.strip() for name in args.families.split(",") if name.strip()}
        representatives = [name for name in representatives if name in wanted]
    print(f"families: {len(representatives)}")

    features = relmap.load_feature_table()
    print(
        f"feature table: {len(features)} rows, seasons "
        f"{int(features['season'].min())}-{int(features['season'].max())}"
    )

    widened, arms, skipped = build_arm_columns(features, representatives)
    if skipped:
        print(f"NOT BUILT (no discovered home/away pair): {', '.join(skipped)}")

    started = time.time()
    fitted = run_window(widened, arms, seasons, leak_treatment=args.mode == "positive-control")

    results: dict[str, Any] = {}
    for name, frame in fitted.items():
        if frame.empty:
            results[name] = {"status": "no_scored_games"}
            continue
        if args.mode == "null":
            results[name] = {
                "status": "scored",
                "treatment_vs_control": null_distribution(
                    frame, "control", "treatment", permutations=args.permutations
                ),
                "treatment_vs_baseline": null_distribution(
                    frame, "baseline", "treatment", permutations=args.permutations
                ),
                "control_vs_baseline": null_distribution(
                    frame, "baseline", "control", permutations=args.permutations
                ),
            }
            continue
        graded = grade(frame)
        results[name] = {
            "status": "scored",
            "home_pick_rate": {
                arm: float(graded[column].ge(0.5).mean()) for arm, column in ARM_PROBABILITY.items()
            },
            "permutation_null_treatment_vs_control": null_distribution(
                frame, "control", "treatment", permutations=args.permutations
            ),
            "treatment_vs_control": summarize_pair(graded, "control_correct", "treatment_correct"),
            "treatment_vs_baseline": summarize_pair(
                graded, "baseline_correct", "treatment_correct"
            ),
            "control_vs_baseline": summarize_pair(graded, "baseline_correct", "control_correct"),
        }

    configuration = {
        "mode": args.mode,
        "seasons": list(seasons),
        "grade": "close",
        "frozen_structure": FROZEN_STRUCTURE,
        "ridge_alpha": RIDGE_ALPHA,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "seed": SEED,
        "permutations": args.permutations,
        "predeclaration": "docs/graph_ratings_v2_screen.md",
        "input_screen": SCREEN_ARTIFACT.relative_to(REPO_ROOT).as_posix(),
        "families_requested": representatives,
        "families_not_built": skipped,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": round(time.time() - started, 1),
        **configuration,
        "results": results,
        "provenance": artifact_provenance(
            configuration,
            REPO_ROOT / "data" / "processed" / "game_features_weak_stack_v4.parquet",
            project_root=REPO_ROOT,
        ),
    }
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = REPO_ROOT / "artifacts" / (args.out or "graph_team_stat_screen") / timestamp
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="graph-team-stat-screen",
        metrics={
            "mode": args.mode,
            "n_families_scored": sum(
                1 for value in results.values() if value.get("status") == "scored"
            ),
            "n_families_not_built": len(skipped),
        },
        notes=(
            "Graph ratings v2 team_stat arm vs the raw differential control; "
            "see docs/graph_ratings_v2_screen.md for the predeclared design."
        ),
    )
    print("wrote " + str(output_dir / "results.json"))

    scored = [
        (name, value["treatment_vs_control"])
        for name, value in results.items()
        if value.get("status") == "scored" and value.get("treatment_vs_control")
    ]
    if args.mode == "null":
        print()
        print(
            f"NULL CHECK ({args.permutations} within-week permutations): the "
            "distribution must be centred on zero."
        )
        print(f"{'family':<42}{'null mean':>11}{'null sd':>10}{'null 95%':>22}{'observed':>11}")
        for name, summary in sorted(scored, key=lambda item: item[0]):
            print(
                f"{name:<42}{summary['null_mean_delta'] * 100:>11.3f}"
                f"{summary['null_sd_delta'] * 100:>10.3f}"
                f"   [{summary['null_q025'] * 100:+.3f}, {summary['null_q975'] * 100:+.3f}]"
                f"{summary['observed_delta'] * 100:>11.3f}"
            )
        biased = [
            name
            for name, summary in scored
            if abs(summary["null_mean_delta"]) > 0.5 * summary["null_sd_delta"]
        ]
        print(
            "null mean within half a null SD of zero for "
            f"{len(scored) - len(biased)} of {len(scored)} families"
            + (f"; CHECK: {', '.join(sorted(biased))}" if biased else "")
        )
        return 0

    scored.sort(key=lambda item: -item[1]["week_blocked_probability_positive"])
    print()
    print(f"treatment (graph) minus control (raw differential), {args.mode}:")
    print(f"{'family':<42}{'delta pts':>11}{'P+':>8}{'week 95% CI':>26}")
    for name, summary in scored:
        low, high = summary["week_blocked_ci95"]
        print(
            f"{name:<42}{summary['delta_accuracy'] * 100:>11.3f}"
            f"{summary['week_blocked_probability_positive']:>8.3f}"
            f"   [{low * 100:+.3f}, {high * 100:+.3f}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
