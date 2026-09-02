"""Graph `team_stat` off_rush_epa_per_play, stacked on PRODUCTION: does it beat
what is actually played, not a bare market baseline?

Predeclared in ``docs/graph_team_stat_off_rush_epa_on_production.md`` BEFORE
this script was pointed at an outcome column. Read that document first -- it
declares the candidate feature, the two arms, the rotation family/window, the
null, the positive control, and the recording rules. Mirrors
``scripts/graph_team_stat_def_yards_per_play_on_production.py`` (the
def_yards_per_play sibling) exactly, substituting the ``off_rush_epa_per_play``
team_stat column -- the family carrying the HIGHEST split-half reliability in
the whole graph line of work (registry
``graph_input_screen_off_rush_epa_per_play``, 0.987, opener-graded 2020-2025)
but whose 38-family screen row is the one the permutation null discounts most
(docs/graph_ratings_v2_screen.md section 8, 53.5th percentile of its own null).

Two arms, one evaluator, one rotation-assigned window, close-graded:

* ``baseline``  -- production ``weak_stack`` (``fit_margin_model``, ridge
  alpha 10.0, target ``market_residual`` -- the active model's own recipe,
  see ``artifacts/active_ats_model.json``).
* ``candidate`` -- ``weak_stack_graph_off_rush_epa``: the SAME production
  feature set plus exactly one new column,
  ``graph_v2_team_stat_off_rush_epa_per_play_katz_diff``.

Both arms are fit with the FULL production feature profile via
``nfl_ats.margin.fit_margin_model`` -- not a single-feature model -- so this
measures the graph column's marginal contribution on top of everything the
production chain already explains, per the project's own "composition is not
the signal" lesson.

Instrument checks that run BEFORE the window is spent (``--mode``):

* ``null``             -- settle margins shuffled within each week. A harness
  that reports an effect here is broken.
* ``positive-control`` -- the candidate's one new column is replaced by the
  realized ``ats_margin``, a deliberate leak. A harness that CANNOT detect
  this, even inside the full production feature set, would be blind, and a
  "no effect" reading from it would mean nothing.
* ``screen``           -- the real look. Spends the family's assigned
  rotation window (``graph_off_rush_epa_on_production``); run once, after both
  checks above pass.

Closing-grounds taxonomy (binding, restated per AGENTS.md so this file stands
on its own): an interval containing zero is NEVER grounds to reject, fail or
close an experiment. Only a RESOLVED wrong sign (whole interval on the wrong
side of zero), zero split-half reliability, or a positive control proven able
to detect an effect that size closes a line of work. Everything else is
``unresolved_below_power``: record it and report ``probability_positive``,
never the binary "contains zero". This run is CLOSE-graded, so per the binding
"grade the decision at the opener" rule, it settles no play/no-play decision
regardless of sign -- every family here is recorded
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
from nfl_ats.graph_team_stat_off_rush_epa_production_feature import (  # noqa: E402
    GRAPH_OFF_RUSH_EPA_COLUMN,
)
from nfl_ats.margin import MarginFeatureProfile, fit_margin_model  # noqa: E402
from nfl_ats.modeling import regular_season_rows  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402

BASELINE_PROFILE: MarginFeatureProfile = "weak_stack"
CANDIDATE_PROFILE: MarginFeatureProfile = "weak_stack_graph_off_rush_epa"
REGRESSOR = "ridge"
RIDGE_ALPHA = 10.0

DEFAULT_FEATURES = REPO_ROOT / "data/processed/game_features_weak_stack_graph_off_rush_epa.parquet"
ROTATION_FAMILY = "graph_off_rush_epa_on_production"

# Rotation-assigned window (docs/graph_team_stat_off_rush_epa_on_production.md
# section 7); overridable via --seasons for the instrument checks, which run on
# the same window for comparability with both siblings' own instrument checks.
DEFAULT_SEASONS = "2014-2016"


# ---------------------------------------------------------------------------
# The evaluator
# ---------------------------------------------------------------------------


def run_window(
    features: pd.DataFrame,
    seasons: tuple[int, ...],
    *,
    leak_treatment: bool = False,
) -> pd.DataFrame:
    """Close-graded per-game rows: the realized settle margin plus one
    home-cover probability per arm (baseline, candidate).

    Forward-chaining only -- each week is predicted from a model trained
    strictly on games that kicked off before that week's earliest kickoff.
    Probabilities rather than correctness are returned because the null check
    needs to re-grade the SAME fitted models against many permuted outcomes:
    model fitting never sees the grading outcome, so a permutation changes only
    the grade -- what makes a 200-permutation null affordable.

    ``leak_treatment=True`` swaps the candidate's one new column for the
    realized ``ats_margin``: the positive control, a deliberate leak whose
    effect the instrument must be able to see even inside the full production
    feature set.
    """

    frame = regular_season_rows(features).copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="raise")
    frame = frame.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    completed = frame.loc[frame["result"].notna()].copy()
    window = completed.loc[completed["season"].astype(int).isin(seasons)]

    candidate_source = completed.copy()
    if leak_treatment:
        candidate_source[GRAPH_OFF_RUSH_EPA_COLUMN] = pd.to_numeric(
            candidate_source["ats_margin"], errors="coerce"
        )

    rows: list[dict[str, Any]] = []
    n_weeks = 0
    for (_season, _week), group in window.groupby(["season", "week"], sort=True):
        cutoff = group["gameday"].min()
        baseline_training = completed.loc[completed["gameday"].lt(cutoff)]
        candidate_training = candidate_source.loc[candidate_source["gameday"].lt(cutoff)]
        if len(baseline_training) < MIN_FITTABLE_TRAIN_GAMES:
            continue
        n_weeks += 1

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
            feature_profile=CANDIDATE_PROFILE,
        )
        candidate_scoring = (
            group
            if not leak_treatment
            else group.assign(
                **{GRAPH_OFF_RUSH_EPA_COLUMN: pd.to_numeric(group["ats_margin"], errors="coerce")}
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
                    "settle_margin": margin,
                    "baseline_probability": base,
                    "candidate_probability": cand,
                }
            )
    print(f"  {n_weeks} weeks fitted over seasons {min(seasons)}-{max(seasons)}")
    return pd.DataFrame(rows)


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
    """Settle margins shuffled WITHIN each week (see
    ``docs/graph_ratings_v2_screen.md`` section 6 for why this null is not
    centred on zero by design: it preserves each week's realized home-cover
    rate, and the two arms may carry different home-pick rates)."""

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
    within-week permutations. See module docstring / predeclaration doc
    section 6 -- this null is deliberately NOT centred on zero."""

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
    candidate-minus-baseline comparison. Within-week game correlation is zero
    by owner mandate, so the week block is the honest primary and the season
    block is reported beside it, never averaged with it."""

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
        default=DEFAULT_SEASONS,
        help="inclusive season range (the rotation-assigned window; default matches "
        "the graph_off_rush_epa_on_production family's assigned window)",
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

    start, _, end = args.seasons.partition("-")
    seasons = tuple(range(int(start), int(end or start) + 1))

    features = pd.read_parquet(args.features)
    print(
        f"feature table: {len(features)} rows, seasons "
        f"{int(features['season'].min())}-{int(features['season'].max())}"
    )

    started = time.time()
    fitted = run_window(features, seasons, leak_treatment=args.mode == "positive-control")

    result: dict[str, Any] = {"status": "no_scored_games"}
    if not fitted.empty:
        if args.mode == "null":
            result = {
                "status": "scored",
                "null": null_distribution(fitted, permutations=args.permutations, seed=args.seed),
            }
        else:
            graded = grade(fitted)
            result = {
                "status": "scored",
                "home_pick_rate": {
                    arm: float(graded[column].ge(0.5).mean())
                    for arm, column in ARM_PROBABILITY.items()
                },
                "permutation_null": null_distribution(
                    fitted, permutations=args.permutations, seed=args.seed
                ),
                "candidate_vs_baseline": summarize_pair(
                    graded, samples=args.bootstrap_samples, seed=args.seed
                ),
            }

    configuration = {
        "mode": args.mode,
        "seasons": list(seasons),
        "grade": "close",
        "rotation_family": ROTATION_FAMILY,
        "baseline_profile": BASELINE_PROFILE,
        "candidate_profile": CANDIDATE_PROFILE,
        "regressor": REGRESSOR,
        "ridge_alpha": RIDGE_ALPHA,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "permutations": args.permutations,
        "predeclaration": "docs/graph_team_stat_off_rush_epa_on_production.md",
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
        REPO_ROOT
        / "artifacts"
        / (args.out or "graph_team_stat_off_rush_epa_per_play_on_production")
        / timestamp
    )
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="graph-team-stat-off-rush-epa-per-play-on-production",
        metrics={"mode": args.mode, "status": result.get("status", "unknown")},
        notes=(
            "Graph ratings v2 team_stat off_rush_epa_per_play feature vs PRODUCTION "
            "weak_stack (not a bare market baseline); see "
            "docs/graph_team_stat_off_rush_epa_on_production.md."
        ),
    )
    print("wrote " + str(output_dir / "results.json"))

    if args.mode == "null" and result.get("status") == "scored":
        null = result["null"]
        print()
        print(
            f"NULL CHECK ({args.permutations} within-week permutations): the "
            "distribution must be centred near its own closed-form expectation, "
            "not necessarily zero."
        )
        print(
            f"null mean {null['null_mean_delta'] * 100:+.3f} pts, sd "
            f"{null['null_sd_delta'] * 100:.3f}, 95% [{null['null_q025'] * 100:+.3f}, "
            f"{null['null_q975'] * 100:+.3f}], observed {null['observed_delta'] * 100:+.3f}"
        )
    elif result.get("status") == "scored":
        pair = result["candidate_vs_baseline"]
        null = result["permutation_null"]
        print()
        print(
            f"candidate (weak_stack_graph_off_rush_epa) minus baseline (weak_stack), {args.mode}:"
        )
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
