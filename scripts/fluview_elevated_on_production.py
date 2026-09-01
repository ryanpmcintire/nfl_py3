"""FluView elevated-illness indicators, stacked on PRODUCTION: does either
beat what is actually played, not a bare market baseline?

Predeclared in ``docs/fluview_on_production.md`` BEFORE this script was
pointed at an outcome column. Read that document first -- it declares the
two candidate columns, the two arms, the era-stratified rotation family and
window, the null, the positive control, and the recording rules.

Two cells, selected with ``--cell``:

* ``away`` (PRIMARY)   -- candidate profile ``weak_stack_fluview_away``
  (production ``weak_stack`` plus ``fluview_away_market_elevated``).
* ``home`` (secondary)  -- candidate profile ``weak_stack_fluview_home``
  (production ``weak_stack`` plus ``fluview_home_market_elevated``).

Baseline in both cases is production ``weak_stack`` unmodified. Both arms are
fit with the FULL production feature profile via
``nfl_ats.margin.fit_margin_model`` -- not a single-feature model -- so this
measures each FluView column's marginal contribution on top of everything
the production chain already explains.

**Era-stratified window, not a contiguous one** (docs/fluview_on_production.md
section 5): FluView's point-in-time-recoverable coverage is 0.0% for every
NFL season 2009-2016, so a contiguous default-size rotation block is legally
unusable for this family. The assigned window is two single-season legs,
``--legs "2011,2025"`` by default, matching the
``fluview_elevated_on_production`` family's real assigned window (confirmed
via ``nfl-ats rotation assign --stratified``, never hand-picked). Leg 2011
sits at the coverage floor (0.0% coverage -- an expected, predeclared
degenerate leg, not a bug); leg 2025 sits deep in the well-covered era
(94.0% coverage). **Per-leg walk-forward**: unlike
``nfl_ats.rotation.confirmation_split_legs``'s single-cutoff-per-leg
convention, this script refits per WEEK within each leg (every week's model
trained on all completed games strictly before that week's own earliest
kickoff) -- strictly more conservative, matching
``scripts/graph_team_stat_off_sack_rate_on_production.py``'s own
``run_window`` design generalized from "weeks in a contiguous range" to
"weeks in either leg season."

Instrument checks that run BEFORE the window is spent (``--mode``):

* ``null``             -- settle margins shuffled within each week (pooled
  across both legs). A harness that reports an effect here is broken.
* ``positive-control`` -- the candidate's one new column is replaced by the
  realized ``ats_margin``, a deliberate leak. A harness that CANNOT detect
  this, even inside the full production feature set, would be blind.
* ``screen``           -- the real look. Spends the family's assigned
  window (``fluview_elevated_on_production``); run once per cell, after both
  checks above pass.

Per-leg magnitudes are reported as first-class output (``leg_results``),
never collapsed into the pooled read alone -- the owner's binding refinement
on the era-stratified proposal (docs/era_stratified_windows_proposal.md).

Closing-grounds taxonomy (binding, restated per AGENTS.md so this file stands
on its own): an interval containing zero is NEVER grounds to reject, fail or
close an experiment. Only a RESOLVED wrong sign (whole interval on the wrong
side of zero), zero split-half reliability, or a positive control proven able
to detect an effect that size closes a line of work. Everything else is
``unresolved_below_power``: record it and report ``probability_positive``,
never the binary "contains zero". This run is CLOSE-graded, so per the
binding "grade the decision at the opener" rule, it settles no play/no-play
decision regardless of sign -- every cell here is recorded
``unresolved_below_power``.
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
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from graph_input_screen import BOOTSTRAP_SAMPLES, SEED  # noqa: E402

from nfl_ats.clv import pick_correct, week_blocked_bootstrap  # noqa: E402
from nfl_ats.constants import MIN_FITTABLE_TRAIN_GAMES  # noqa: E402
from nfl_ats.fluview_production_feature import (  # noqa: E402
    FLUVIEW_AWAY_ELEVATED_COLUMN,
    FLUVIEW_HOME_ELEVATED_COLUMN,
)
from nfl_ats.margin import MarginFeatureProfile, fit_margin_model  # noqa: E402
from nfl_ats.modeling import regular_season_rows  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

BASELINE_PROFILE: MarginFeatureProfile = "weak_stack"
REGRESSOR = "ridge"
RIDGE_ALPHA = 10.0

DEFAULT_FEATURES = REPO_ROOT / "data/processed/game_features_weak_stack_fluview.parquet"
ROTATION_FAMILY = "fluview_elevated_on_production"

CELLS: dict[str, dict[str, Any]] = {
    "away": {
        "candidate_profile": "weak_stack_fluview_away",
        "column": FLUVIEW_AWAY_ELEVATED_COLUMN,
        "role": "primary",
    },
    "home": {
        "candidate_profile": "weak_stack_fluview_home",
        "column": FLUVIEW_HOME_ELEVATED_COLUMN,
        "role": "secondary",
    },
}


# ---------------------------------------------------------------------------
# The evaluator
# ---------------------------------------------------------------------------


def run_leg(
    completed: pd.DataFrame,
    leg_season: int,
    *,
    candidate_profile: MarginFeatureProfile,
    candidate_column: str,
    leak_treatment: bool,
) -> pd.DataFrame:
    """Per-week walk-forward rows for ONE leg season.

    Every week within ``leg_season`` is predicted from a model trained
    strictly on games that kicked off before that week's earliest kickoff --
    across the WHOLE history, not just this leg (docs/fluview_on_production.md
    section 5, "resolution 4" of the era-stratified proposal: a later leg's
    training set may include an earlier leg's season; here there is only one
    leg's own season being scored per call, so this is simply ordinary
    forward-chaining applied to a single-season window).
    """

    leg = completed.loc[completed["season"].astype(int) == leg_season]

    candidate_source = completed.copy()
    if leak_treatment:
        candidate_source[candidate_column] = pd.to_numeric(
            candidate_source["ats_margin"], errors="coerce"
        )

    rows: list[dict[str, Any]] = []
    for (_season, _week), group in leg.groupby(["season", "week"], sort=True):
        cutoff = group["gameday"].min()
        baseline_training = completed.loc[completed["gameday"].lt(cutoff)]
        candidate_training = candidate_source.loc[candidate_source["gameday"].lt(cutoff)]
        if len(baseline_training) < MIN_FITTABLE_TRAIN_GAMES:
            continue

        settle_margin = pd.to_numeric(group["result"], errors="coerce") - pd.to_numeric(
            group["spread_line"], errors="coerce"
        )
        baseline_model = fit_margin_model(
            baseline_training,
            target="market_residual",
            model_name=REGRESSOR,
            ridge_alpha=RIDGE_ALPHA,
            feature_profile=BASELINE_PROFILE,
        )
        candidate_model = fit_margin_model(
            candidate_training,
            target="market_residual",
            model_name=REGRESSOR,
            ridge_alpha=RIDGE_ALPHA,
            feature_profile=candidate_profile,
        )
        candidate_scoring = (
            group
            if not leak_treatment
            else group.assign(
                **{candidate_column: pd.to_numeric(group["ats_margin"], errors="coerce")}
            )
        )
        baseline_probability = baseline_model.predict(group)["home_cover_probability"]
        candidate_probability = candidate_model.predict(candidate_scoring)["home_cover_probability"]

        for game_id, season_value, week_value, margin, base, cand in zip(
            group["game_id"],
            group["season"],
            group["week"],
            settle_margin,
            baseline_probability,
            candidate_probability,
            strict=True,
        ):
            rows.append(
                {
                    "game_id": game_id,
                    "season": int(str(season_value)),
                    "week": int(str(week_value)),
                    "leg": leg_season,
                    "settle_margin": margin,
                    "baseline_probability": base,
                    "candidate_probability": cand,
                }
            )
    return pd.DataFrame(rows)


def run_window(
    features: pd.DataFrame,
    legs: tuple[int, int],
    *,
    candidate_profile: MarginFeatureProfile,
    candidate_column: str,
    leak_treatment: bool = False,
) -> pd.DataFrame:
    """Pooled per-game rows across BOTH legs. Forward-chaining only, per leg
    (see ``run_leg``)."""

    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    frame = frame.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    completed = frame.loc[frame["result"].notna()].copy()

    leg_frames = []
    for leg_season in legs:
        leg_rows = run_leg(
            completed,
            leg_season,
            candidate_profile=candidate_profile,
            candidate_column=candidate_column,
            leak_treatment=leak_treatment,
        )
        print(f"  leg {leg_season}: {len(leg_rows)} games scored")
        leg_frames.append(leg_rows)
    return pd.concat(leg_frames, ignore_index=True) if leg_frames else pd.DataFrame()


ARM_PROBABILITY = {"baseline": "baseline_probability", "candidate": "candidate_probability"}


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
    """Settle margins shuffled WITHIN each week, pooled across both legs (see
    docs/fluview_on_production.md section 6 for why this null is not centred
    on zero by design)."""

    values = frame["settle_margin"].to_numpy(dtype=float, copy=True)
    for positions in groups:
        values[positions] = rng.permutation(values[positions])
    return pd.Series(values, index=frame.index)


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


def null_distribution(
    frame: pd.DataFrame,
    *,
    permutations: int,
    seed: int = SEED,
) -> dict[str, Any]:
    """The paired candidate-minus-baseline delta's null distribution over many
    within-week permutations, pooled across both legs."""

    rng = np.random.default_rng(seed)
    metric = _paired_metric("baseline_correct", "candidate_correct")
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


def summarize_pair(paired: pd.DataFrame, samples: int, seed: int) -> dict[str, Any] | None:
    """Point estimate plus week- and season-blocked bootstrap for the paired
    candidate-minus-baseline comparison, pooled across both legs."""

    if paired.empty or paired.dropna(subset=["baseline_correct", "candidate_correct"]).empty:
        return None
    metric = _paired_metric("baseline_correct", "candidate_correct")
    point = metric(paired)
    week = week_blocked_bootstrap(paired, metric, block="week", samples=samples, seed=seed)
    season = week_blocked_bootstrap(paired, metric, block="season", samples=samples, seed=seed)
    week_row = week.loc[week["metric"].eq("delta_accuracy")].iloc[0]
    season_row = season.loc[season["metric"].eq("delta_accuracy")].iloc[0]
    return {
        "delta_accuracy": point["delta_accuracy"],
        "candidate_accuracy": point["candidate_accuracy"],
        "baseline_accuracy": point["reference_accuracy"],
        "week_blocked_ci95": [float(week_row["lower"]), float(week_row["upper"])],
        "week_blocked_probability_positive": float(week_row["probability_positive"]),
        "season_blocked_ci95": [float(season_row["lower"]), float(season_row["upper"])],
        "season_blocked_probability_positive": float(season_row["probability_positive"]),
        "n_games": len(paired.dropna(subset=["baseline_correct", "candidate_correct"])),
        "n_weeks": int(paired[["season", "week"]].drop_duplicates().shape[0]),
        "n_seasons": int(paired["season"].nunique()),
    }


def summarize_leg(leg_frame: pd.DataFrame, samples: int, seed: int) -> dict[str, Any] | None:
    """One leg's own magnitude -- the era-stratified proposal's binding
    refinement: per-leg magnitudes are first-class, never collapsed into the
    pooled read alone."""

    if leg_frame.empty:
        return None
    graded = grade(leg_frame)
    return summarize_pair(graded, samples=samples, seed=seed)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", choices=tuple(CELLS), required=True)
    parser.add_argument(
        "--mode",
        choices=("null", "positive-control", "screen"),
        required=True,
        help="null and positive-control are instrument checks; screen is the real look",
    )
    parser.add_argument(
        "--legs",
        default="2011,2025",
        help="comma-separated leg seasons (the rotation-assigned stratified window; "
        "default matches the fluview_elevated_on_production family's assigned (2011, 2025))",
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument(
        "--permutations",
        type=int,
        default=200,
        help="within-week permutations for --mode null (a single draw is not a test)",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default="", help="artifact directory name override")
    args = parser.parse_args()

    legs = tuple(int(part.strip()) for part in args.legs.split(","))
    if len(legs) != 2:
        raise SystemExit(f"--legs must name exactly two seasons, got {args.legs!r}")

    cell = CELLS[args.cell]
    candidate_profile: MarginFeatureProfile = cell["candidate_profile"]
    candidate_column: str = cell["column"]

    features = pd.read_parquet(args.features)
    print(
        f"feature table: {len(features)} rows, seasons "
        f"{int(features['season'].min())}-{int(features['season'].max())}"
    )
    print(f"cell={args.cell} ({cell['role']}) candidate_profile={candidate_profile}")

    started = time.time()
    fitted = run_window(
        features,
        legs,
        candidate_profile=candidate_profile,
        candidate_column=candidate_column,
        leak_treatment=args.mode == "positive-control",
    )

    result: dict[str, Any] = {"status": "no_scored_games"}
    if not fitted.empty:
        if args.mode == "null":
            result = {
                "status": "scored",
                "null": null_distribution(fitted, permutations=args.permutations, seed=args.seed),
            }
        else:
            graded = grade(fitted)
            leg_results = {}
            for leg_season in legs:
                leg_summary = summarize_leg(
                    fitted.loc[fitted["leg"] == leg_season],
                    samples=args.bootstrap_samples,
                    seed=args.seed,
                )
                leg_results[str(leg_season)] = leg_summary
            result = {
                "status": "scored",
                "home_pick_rate": {
                    arm: float(graded[column].ge(0.5).mean())
                    for arm, column in ARM_PROBABILITY.items()
                },
                "permutation_null": null_distribution(
                    fitted, permutations=args.permutations, seed=args.seed
                ),
                "candidate_vs_baseline_pooled": summarize_pair(
                    graded, samples=args.bootstrap_samples, seed=args.seed
                ),
                "leg_results": leg_results,
            }

    configuration = {
        "cell": args.cell,
        "role": cell["role"],
        "mode": args.mode,
        "legs": list(legs),
        "grade": "close",
        "rotation_family": ROTATION_FAMILY,
        "baseline_profile": BASELINE_PROFILE,
        "candidate_profile": candidate_profile,
        "candidate_column": candidate_column,
        "regressor": REGRESSOR,
        "ridge_alpha": RIDGE_ALPHA,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "permutations": args.permutations,
        "predeclaration": "docs/fluview_on_production.md",
        "features_path": str(args.features),
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": round(time.time() - started, 1),
        **configuration,
        "result": result,
        "provenance": artifact_provenance(
            configuration,
            args.features,
            project_root=REPO_ROOT,
        ),
    }
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = (
        REPO_ROOT / "artifacts" / (args.out or "fluview_elevated_on_production") / timestamp
    )
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="fluview-elevated-on-production",
        metrics={"cell": args.cell, "mode": args.mode, "status": result.get("status", "unknown")},
        notes=(
            "FluView elevated-illness indicator vs PRODUCTION weak_stack (not a bare "
            "market baseline), era-stratified window; see docs/fluview_on_production.md."
        ),
    )
    print("wrote " + str(output_dir / "results.json"))

    if args.mode == "null" and result.get("status") == "scored":
        null = result["null"]
        print()
        print(
            f"NULL CHECK ({args.permutations} within-week permutations, pooled across legs "
            f"{legs}): the distribution must be centred near its own closed-form "
            "expectation, not necessarily zero."
        )
        print(
            f"null mean {null['null_mean_delta'] * 100:+.3f} pts, sd "
            f"{null['null_sd_delta'] * 100:.3f}, 95% [{null['null_q025'] * 100:+.3f}, "
            f"{null['null_q975'] * 100:+.3f}], observed {null['observed_delta'] * 100:+.3f}"
        )
    elif result.get("status") == "scored":
        pair = result["candidate_vs_baseline_pooled"]
        null = result["permutation_null"]
        print()
        print(f"candidate ({candidate_profile}) minus baseline (weak_stack), {args.mode}, pooled:")
        if pair is not None:
            low, high = pair["week_blocked_ci95"]
            print(
                f"delta {pair['delta_accuracy'] * 100:+.3f} pts  P+ "
                f"{pair['week_blocked_probability_positive']:.3f}  "
                f"week 95% CI [{low * 100:+.3f}, {high * 100:+.3f}]  n={pair['n_games']} games, "
                f"{pair['n_weeks']} weeks"
            )
        print(
            f"permutation null: mean {null['null_mean_delta'] * 100:+.3f} pts, observed at the "
            f"{null['fraction_of_null_below_observed'] * 100:.1f}th percentile of its own null"
        )
        for leg_season, leg_summary in result["leg_results"].items():
            if leg_summary is None:
                print(f"  leg {leg_season}: no scored games")
                continue
            low, high = leg_summary["week_blocked_ci95"]
            print(
                f"  leg {leg_season}: delta {leg_summary['delta_accuracy'] * 100:+.3f} pts  P+ "
                f"{leg_summary['week_blocked_probability_positive']:.3f}  "
                f"week 95% CI [{low * 100:+.3f}, {high * 100:+.3f}]  "
                f"n={leg_summary['n_games']} games"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
