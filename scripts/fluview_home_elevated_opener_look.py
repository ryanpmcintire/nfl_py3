"""FluView home-market elevated-illness indicator, OPENER-graded confirmation
on PRODUCTION `weak_stack`.

Predeclared in ``docs/fluview_opener_look.md`` BEFORE this script was pointed
at an outcome column. Read that document first -- it declares the candidate
column/profile, the opener grade (production probability rule primary, sign
rule secondary), the new rotation family (``fluview_home_elevated_opener``,
grade ``opener``, inheriting the close-graded ``fluview_elevated_on_production``),
and the recording rules.

This is the deciding, OPENER-graded look at the home-market cell that the
close-graded look (``docs/fluview_on_production.md`` section 7: pooled +0.969
accuracy points, week-blocked P+ 0.792) earned. Only the home-market cell is
scored here -- the away-market cell reads as a coin flip at the close grade
and is not re-run.

Baseline arm: production ``weak_stack``. Candidate arm: ``weak_stack_fluview_home``
(``weak_stack`` plus exactly ``fluview_home_market_elevated``). Both graded via
``nfl_ats.clv.opener_pick_evaluation`` -- the exact machinery behind
``docs/opener_evaluation.md``'s incumbent numbers -- restricted to the
family's rotation-assigned window via ``nfl_ats.rotation.confirmation_split``
(a CONTIGUOUS window, not stratified -- ``confirmation_split``, not
``confirmation_split_legs``), exactly as ``scripts/mod07_weak_stack.py``'s
own ``arm()`` pattern: hand the evaluator ``training + window`` concatenated
so every earlier completed game remains available to the walk-forward fit,
then filter the scored frame down to the window's own seasons.

Three modes (``--mode``), mirroring ``scripts/fluview_elevated_on_production.py``'s
own instrument-check discipline, adapted to the opener grade (no precedent
opener script runs these checks):

* ``null``             -- the realized settlement outcome at the opener
  (``margin_vs_open``) is shuffled WITHIN each week, 200 draws, after the real
  models are fit once (only the grading outcome is permuted, so this costs no
  extra model fits). Not centred on zero by design.
* ``positive-control`` -- ``fluview_home_market_elevated`` is temporarily
  replaced by the realized ``ats_margin`` (a deliberate, large leak) across
  the WHOLE scoped table (training and scoring rows alike) before calling
  ``opener_pick_evaluation``. Proves the harness can detect a real effect of
  meaningful size at this window's sample.
* ``screen``           -- the real look. Spends the family's assigned window.

Closing-grounds taxonomy (binding, restated per AGENTS.md so this file stands
on its own): an interval containing zero is NEVER grounds to reject, fail or
close an experiment. Only a RESOLVED wrong sign (whole interval on the wrong
side of zero), zero split-half reliability, or a positive control proven able
to detect an effect that size closes a line of work. Everything else is
``unresolved_below_power``: record it and report ``probability_positive``,
never the binary "contains zero".

Run:  .\\.tools\\uv.exe run --no-sync python scripts/fluview_home_elevated_opener_look.py \
        --mode screen
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

from nfl_ats.clv import (  # noqa: E402
    opener_evaluation_metrics,
    opener_pick_evaluation,
    week_blocked_bootstrap,
)
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES  # noqa: E402
from nfl_ats.fluview_production_feature import FLUVIEW_HOME_ELEVATED_COLUMN  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from nfl_ats.rotation import confirmation_split, load_registry  # noqa: E402

BASELINE_PROFILE = "weak_stack"
CANDIDATE_PROFILE = "weak_stack_fluview_home"
CANDIDATE_COLUMN = FLUVIEW_HOME_ELEVATED_COLUMN
REGRESSOR = "ridge"
RIDGE_ALPHA = 10.0
ROTATION_FAMILY = "fluview_home_elevated_opener"

DEFAULT_FEATURES = REPO_ROOT / "data/processed/game_features_weak_stack_fluview.parquet"
DEFAULT_MARKET_ROOT = REPO_ROOT / "data/market/raw"

# Matches docs/opener_evaluation.md's and scripts/surface_profile_opener_eval.py's
# own opener-grade convention -- comparable to the other opener confirmations
# in this tree, not to the close-graded sibling's graph_input_screen constants.
OPENER_BOOTSTRAP_SAMPLES = 20_000
OPENER_BOOTSTRAP_SEED = 20260817
NULL_PERMUTATIONS = 200


def _config(profile: str) -> dict[str, Any]:
    return {
        "feature_profile": profile,
        "regressor": REGRESSOR,
        "ridge_alpha": RIDGE_ALPHA,
        "target": "market_residual",
    }


def scoped_window_frame(
    features: pd.DataFrame, registry: Any, family: str
) -> tuple[pd.DataFrame, tuple[int, ...]]:
    """(training + window) concatenated, plus the window's own seasons.

    Mirrors scripts/mod07_weak_stack.py's arm() exactly: handing the
    evaluator training+window means every earlier completed game remains
    available to the walk-forward fit, while only the window's own seasons
    can be scored (the opener archive does not reach the training seasons).
    """

    training, window = confirmation_split(features, registry, family)
    scoped = pd.concat([training, window], ignore_index=True)
    seasons = tuple(sorted(int(s) for s in window["season"].astype(int).unique()))
    return scoped, seasons


def run_arm(
    features: pd.DataFrame,
    *,
    market_root: Path,
    profile: str,
    seasons: tuple[int, ...],
    min_train_games: int,
    leak_column: str | None = None,
) -> pd.DataFrame:
    """Score one arm at the opener grade, restricted to ``seasons``.

    ``leak_column``, when given, is the positive-control treatment: that
    column is replaced by the realized ``ats_margin`` across the WHOLE
    scoped frame (training and scoring rows alike) before scoring -- the
    same deliberate leak scripts/fluview_elevated_on_production.py applies
    at the close grade.
    """

    source = features
    if leak_column is not None:
        source = features.copy()
        source[leak_column] = pd.to_numeric(source["ats_margin"], errors="coerce")
    scored = opener_pick_evaluation(
        market_root,
        source,
        active_model_config=_config(profile),
        min_train_games=min_train_games,
    )
    return scored.loc[scored["season"].astype(int).isin(seasons)].reset_index(drop=True)


def paired_frame(baseline: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    """One row per game scored by BOTH arms, both pick rules carried."""

    keep = [
        "game_id",
        "season",
        "week",
        "margin_vs_open",
        "margin_vs_close",
        "pick_home_at_open",
        "pick_home_at_open_probability_rule",
        "correct_at_open",
        "correct_at_close",
        "correct_at_open_probability_rule",
        "correct_at_close_probability_rule",
    ]
    left = baseline[keep].rename(
        columns={
            "pick_home_at_open": "baseline_pick_home",
            "pick_home_at_open_probability_rule": "baseline_pick_home_pr",
            "correct_at_open": "baseline_correct_open",
            "correct_at_close": "baseline_correct_close",
            "correct_at_open_probability_rule": "baseline_correct_open_pr",
            "correct_at_close_probability_rule": "baseline_correct_close_pr",
        }
    )
    right = candidate[
        [
            "game_id",
            "pick_home_at_open",
            "pick_home_at_open_probability_rule",
            "correct_at_open",
            "correct_at_close",
            "correct_at_open_probability_rule",
            "correct_at_close_probability_rule",
        ]
    ].rename(
        columns={
            "pick_home_at_open": "candidate_pick_home",
            "pick_home_at_open_probability_rule": "candidate_pick_home_pr",
            "correct_at_open": "candidate_correct_open",
            "correct_at_close": "candidate_correct_close",
            "correct_at_open_probability_rule": "candidate_correct_open_pr",
            "correct_at_close_probability_rule": "candidate_correct_close_pr",
        }
    )
    merged = left.merge(right, on="game_id", how="inner")
    for column in (
        "baseline_correct_open",
        "baseline_correct_close",
        "baseline_correct_open_pr",
        "baseline_correct_close_pr",
        "candidate_correct_open",
        "candidate_correct_close",
        "candidate_correct_open_pr",
        "candidate_correct_close_pr",
    ):
        merged[column] = merged[column].astype(float)
    return merged.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


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


def summarize_pair(paired: pd.DataFrame, reference: str, candidate: str, samples: int, seed: int):
    if paired.empty or paired.dropna(subset=[reference, candidate]).empty:
        return None
    metric = _paired_metric(reference, candidate)
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
        "n_games": len(paired.dropna(subset=[reference, candidate])),
        "n_weeks": int(paired[["season", "week"]].drop_duplicates().shape[0]),
        "n_seasons": int(paired["season"].nunique()),
    }


def week_positions(frame: pd.DataFrame) -> list[np.ndarray]:
    return [
        np.asarray(positions, dtype=np.intp)
        for positions in frame.groupby(["season", "week"], sort=False).indices.values()
    ]


def permuted_margins(
    frame: pd.DataFrame, column: str, rng: np.random.Generator, groups: list[np.ndarray]
) -> pd.Series:
    """``column`` shuffled WITHIN each week (docs/fluview_opener_look.md section 5)."""

    values = frame[column].to_numpy(dtype=float, copy=True)
    for positions in groups:
        values[positions] = rng.permutation(values[positions])
    return pd.Series(values, index=frame.index)


def pick_correct_flag(pick_home: pd.Series, margin: pd.Series) -> pd.Series:
    """Local re-derivation of nfl_ats.clv.pick_correct's rule, applied to a
    (possibly permuted) margin series -- keeps this null a pure re-grading of
    FIXED picks against a shuffled outcome, never a re-fit."""

    covered_home = margin.gt(0.0)
    correct = np.where(pick_home.astype(bool), covered_home, ~covered_home).astype(float)
    return pd.Series(np.where(margin.eq(0.0), np.nan, correct), index=margin.index)


def null_distribution(
    paired: pd.DataFrame, *, rule_suffix: str, permutations: int, seed: int
) -> dict[str, Any]:
    """Paired candidate-minus-baseline delta's null distribution over within
    -week permutations of the realized opener-settlement margin."""

    pick_col = {"": "pick_home", "_pr": "pick_home_pr"}[rule_suffix]
    baseline_pick = paired[f"baseline_{pick_col}"]
    candidate_pick = paired[f"candidate_{pick_col}"]
    rng = np.random.default_rng(seed)
    groups = week_positions(paired)
    deltas = []
    for _ in range(permutations):
        margin = permuted_margins(paired, "margin_vs_open", rng, groups)
        base_correct = pick_correct_flag(baseline_pick, margin)
        cand_correct = pick_correct_flag(candidate_pick, margin)
        valid = pd.DataFrame({"b": base_correct, "c": cand_correct}).dropna()
        deltas.append(float((valid["c"] - valid["b"]).mean()) if len(valid) else float("nan"))
    values = np.asarray(deltas, dtype=float)
    finite = values[np.isfinite(values)]
    observed_margin = paired["margin_vs_open"]
    observed_base = pick_correct_flag(baseline_pick, observed_margin)
    observed_cand = pick_correct_flag(candidate_pick, observed_margin)
    observed_valid = pd.DataFrame({"b": observed_base, "c": observed_cand}).dropna()
    observed = float((observed_valid["c"] - observed_valid["b"]).mean())
    return {
        "permutations": len(finite),
        "null_mean_delta": float(finite.mean()),
        "null_sd_delta": float(finite.std(ddof=1)),
        "null_q025": float(np.quantile(finite, 0.025)),
        "null_q975": float(np.quantile(finite, 0.975)),
        "observed_delta": observed,
        "fraction_of_null_below_observed": float((finite < observed).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("null", "positive-control", "screen"), required=True)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--market-root", type=Path, default=DEFAULT_MARKET_ROOT)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--family", default=ROTATION_FAMILY)
    parser.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    parser.add_argument("--permutations", type=int, default=NULL_PERMUTATIONS)
    parser.add_argument("--bootstrap-samples", type=int, default=OPENER_BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=OPENER_BOOTSTRAP_SEED)
    parser.add_argument("--out", default="", help="artifact directory name override")
    args = parser.parse_args()

    features = pd.read_parquet(args.features)
    registry = load_registry(args.registry)
    scoped, seasons = scoped_window_frame(features, registry, args.family)
    print(f"family={args.family} window seasons={seasons}")

    started = time.time()
    baseline_scored = run_arm(
        scoped,
        market_root=args.market_root,
        profile=BASELINE_PROFILE,
        seasons=seasons,
        min_train_games=args.min_train_games,
    )
    candidate_scored = run_arm(
        scoped,
        market_root=args.market_root,
        profile=CANDIDATE_PROFILE,
        seasons=seasons,
        min_train_games=args.min_train_games,
        leak_column=CANDIDATE_COLUMN if args.mode == "positive-control" else None,
    )
    paired = paired_frame(baseline_scored, candidate_scored)
    print(f"paired games: {len(paired)}  weeks: {paired.groupby(['season', 'week']).ngroups}")

    result: dict[str, Any] = {"status": "no_scored_games" if paired.empty else "scored"}
    if not paired.empty:
        if args.mode == "null":
            result["null_production_rule"] = null_distribution(
                paired, rule_suffix="_pr", permutations=args.permutations, seed=args.seed
            )
            result["null_sign_rule"] = null_distribution(
                paired, rule_suffix="", permutations=args.permutations, seed=args.seed
            )
        else:
            result["home_pick_rate"] = {
                "baseline_pr": float(paired["baseline_pick_home_pr"].mean()),
                "candidate_pr": float(paired["candidate_pick_home_pr"].mean()),
                "baseline_sign": float(paired["baseline_pick_home"].mean()),
                "candidate_sign": float(paired["candidate_pick_home"].mean()),
            }
            result["permutation_null_production_rule"] = null_distribution(
                paired, rule_suffix="_pr", permutations=args.permutations, seed=args.seed
            )
            result["permutation_null_sign_rule"] = null_distribution(
                paired, rule_suffix="", permutations=args.permutations, seed=args.seed
            )
            result["opener_production_rule"] = summarize_pair(
                paired,
                "baseline_correct_open_pr",
                "candidate_correct_open_pr",
                samples=args.bootstrap_samples,
                seed=args.seed,
            )
            result["opener_sign_rule"] = summarize_pair(
                paired,
                "baseline_correct_open",
                "candidate_correct_open",
                samples=args.bootstrap_samples,
                seed=args.seed,
            )
            result["close_production_rule"] = summarize_pair(
                paired,
                "baseline_correct_close_pr",
                "candidate_correct_close_pr",
                samples=args.bootstrap_samples,
                seed=args.seed,
            )
            result["close_sign_rule"] = summarize_pair(
                paired,
                "baseline_correct_close",
                "candidate_correct_close",
                samples=args.bootstrap_samples,
                seed=args.seed,
            )
            result["baseline_metrics"] = opener_evaluation_metrics(baseline_scored)
            result["candidate_metrics"] = opener_evaluation_metrics(candidate_scored)
            disagreements_pr = paired.loc[
                paired["baseline_pick_home_pr"].ne(paired["candidate_pick_home_pr"])
            ]
            result["picks_disagreeing_production_rule"] = len(disagreements_pr)

    configuration = {
        "mode": args.mode,
        "family": args.family,
        "window_seasons": list(seasons),
        "grade": "opener",
        "rotation_family": args.family,
        "baseline_profile": BASELINE_PROFILE,
        "candidate_profile": CANDIDATE_PROFILE,
        "candidate_column": CANDIDATE_COLUMN,
        "regressor": REGRESSOR,
        "ridge_alpha": RIDGE_ALPHA,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "permutations": args.permutations,
        "min_train_games": args.min_train_games,
        "predeclaration": "docs/fluview_opener_look.md",
        "features_path": str(args.features),
        "market_root": str(args.market_root),
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": round(time.time() - started, 1),
        **configuration,
        "result": result,
        "provenance": artifact_provenance(configuration, args.features, project_root=REPO_ROOT),
    }
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = (
        REPO_ROOT / "artifacts" / (args.out or "fluview_home_elevated_opener_look") / timestamp
    )
    write_experiment_artifact(
        output_dir,
        "results.json",
        payload,
        command="fluview-home-elevated-opener-look",
        metrics={"mode": args.mode, "status": result.get("status", "unknown")},
        notes=(
            "FluView home-market elevated-illness indicator vs PRODUCTION weak_stack, "
            "OPENER-graded (production probability rule primary), rotation-family window; "
            "see docs/fluview_opener_look.md."
        ),
    )
    print("wrote " + str(output_dir / "results.json"))

    if args.mode == "null" and result.get("status") == "scored":
        for label, key in (
            ("production rule", "null_production_rule"),
            ("sign rule", "null_sign_rule"),
        ):
            null = result[key]
            print(
                f"NULL ({label}, {args.permutations} within-week permutations): "
                f"mean {null['null_mean_delta'] * 100:+.3f} pts, "
                f"sd {null['null_sd_delta'] * 100:.3f}, "
                f"95% [{null['null_q025'] * 100:+.3f}, {null['null_q975'] * 100:+.3f}], "
                f"observed {null['observed_delta'] * 100:+.3f}"
            )
    elif result.get("status") == "scored":
        pair = result["opener_production_rule"]
        null = result["permutation_null_production_rule"]
        print(
            f"\ncandidate ({CANDIDATE_PROFILE}) minus baseline (weak_stack), "
            f"{args.mode}, OPENER, production rule:"
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
            f"permutation null (production rule): mean {null['null_mean_delta'] * 100:+.3f} pts, "
            f"observed at the {null['fraction_of_null_below_observed'] * 100:.1f}th percentile"
        )
        sign_pair = result["opener_sign_rule"]
        if sign_pair is not None:
            low, high = sign_pair["week_blocked_ci95"]
            print(
                f"(sign rule) delta {sign_pair['delta_accuracy'] * 100:+.3f} pts  P+ "
                f"{sign_pair['week_blocked_probability_positive']:.3f}  "
                f"week 95% CI [{low * 100:+.3f}, {high * 100:+.3f}]"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
