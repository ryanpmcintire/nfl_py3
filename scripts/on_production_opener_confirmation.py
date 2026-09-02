"""Run one predeclared, rotation-assigned opener confirmation on production.

The four choices are deliberately fixed in :data:`CANDIDATES`.  Each applies
the same estimator as the played ``weak_stack`` chain, retains game-level
paired predictions, grades at the Tuesday opener, and refuses to infer a
window from command-line seasons.  ``null`` and ``positive-control`` are
instrument checks; only ``screen`` is the single outcome look.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
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
from nfl_ats.margin import margin_feature_columns  # noqa: E402
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact  # noqa: E402
from nfl_ats.rotation import confirmation_split, load_registry  # noqa: E402

BASELINE_PROFILE = "weak_stack"
REGRESSOR = "ridge"
RIDGE_ALPHA = 10.0
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260902
NULL_PERMUTATIONS = 200
DEFAULT_MARKET_ROOT = REPO_ROOT / "data/market/raw"


@dataclass(frozen=True)
class Candidate:
    family: str
    inherits: str
    profile: str
    column: str
    features: Path
    predeclaration: str
    artifact_dir: str


CANDIDATES = {
    "reddit": Candidate(
        "reddit_home_comment_ratio_elevated_on_production_opener",
        "reddit_attention_on_production",
        "weak_stack_reddit_ratio_home",
        "reddit_home_comment_ratio_elevated",
        REPO_ROOT / "data/processed/game_features_weak_stack_reddit.parquet",
        "docs/reddit_attention_opener_confirmation.md",
        "reddit_attention_opener_confirmation",
    ),
    "pace": Candidate(
        "team_style_pace_mismatch_on_production_opener",
        "team_style_pace_on_production",
        "weak_stack_team_style_pace",
        "team_style_pace_mismatch_flag",
        REPO_ROOT / "data/processed/game_features_weak_stack_team_style_pace.parquet",
        "docs/team_style_pace_opener_confirmation.md",
        "team_style_pace_opener_confirmation",
    ),
    "illness": Candidate(
        "illness_away_active_ge1_on_production_opener",
        "illness_on_production",
        "weak_stack_illness_away",
        "illness_away_active_ge1",
        REPO_ROOT / "data/processed/game_features_weak_stack_illness.parquet",
        "docs/illness_away_opener_confirmation.md",
        "illness_away_opener_confirmation",
    ),
    "redzone": Candidate(
        "redzone_third_down_over_fade_on_production_opener",
        "redzone_reversion_on_production",
        "weak_stack_redzone_third_down",
        "redzone_third_down_over_fade_diff",
        REPO_ROOT / "data/processed/game_features_weak_stack_redzone_third_down.parquet",
        "docs/redzone_third_down_opener_confirmation.md",
        "redzone_third_down_opener_confirmation",
    ),
}


def model_config(profile: str) -> dict[str, Any]:
    return {
        "feature_profile": profile,
        "regressor": REGRESSOR,
        "ridge_alpha": RIDGE_ALPHA,
        "target": "market_residual",
    }


def profile_identity(candidate: Candidate, features: pd.DataFrame) -> dict[str, Any]:
    """Fail closed unless the candidate profile is production plus one column."""
    baseline = set(margin_feature_columns("market_residual", BASELINE_PROFILE))
    treatment = set(margin_feature_columns("market_residual", candidate.profile))
    if treatment - baseline != {candidate.column} or baseline - treatment:
        raise ValueError(f"{candidate.profile} is not {BASELINE_PROFILE} plus {candidate.column}")
    missing = sorted(treatment.difference(features.columns))
    if missing:
        raise ValueError(f"Feature table lacks required candidate inputs: {missing}")
    return {
        "baseline_columns": len(baseline),
        "candidate_columns": len(treatment),
        "only_added_column": candidate.column,
    }


def scoped_window_frame(
    features: pd.DataFrame, registry: Any, family: str
) -> tuple[pd.DataFrame, tuple[int, ...]]:
    """Use only the CLI-assigned contiguous window and strictly earlier training."""
    training, window = confirmation_split(features, registry, family)
    if pd.to_datetime(training["gameday"]).max() >= pd.to_datetime(window["gameday"]).min():
        raise ValueError("confirmation split leaked a training row into the assigned window")
    seasons = tuple(sorted(int(x) for x in window["season"].unique()))
    return pd.concat([training, window], ignore_index=True), seasons


def run_arm(
    features: pd.DataFrame,
    candidate: Candidate,
    *,
    market_root: Path,
    profile: str,
    seasons: tuple[int, ...],
    min_train_games: int,
    leak: bool,
) -> pd.DataFrame:
    source = features.copy() if leak else features
    if leak:
        # The only permitted treatment leak, used solely by --mode positive-control.
        source[candidate.column] = pd.to_numeric(source["ats_margin"], errors="raise")
    scored = opener_pick_evaluation(
        market_root,
        source,
        active_model_config=model_config(profile),
        min_train_games=min_train_games,
    )
    return scored.loc[scored["season"].astype(int).isin(seasons)].reset_index(drop=True)


def paired_frame(baseline: pd.DataFrame, treatment: pd.DataFrame) -> pd.DataFrame:
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
    right = treatment[
        [x for x in keep if x not in {"season", "week", "margin_vs_open", "margin_vs_close"}]
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
    paired = left.merge(right, on="game_id", validate="one_to_one").sort_values(
        ["season", "week", "game_id"]
    )
    for column in paired.columns:
        if (
            column.endswith("_correct_open")
            or column.endswith("_correct_close")
            or column.endswith("_correct_open_pr")
            or column.endswith("_correct_close_pr")
        ):
            paired[column] = paired[column].astype(float)
    return paired.reset_index(drop=True)


def metric(reference: str, treatment: str) -> Any:
    def calculate(frame: pd.DataFrame) -> dict[str, float]:
        valid = frame.dropna(subset=[reference, treatment])
        return {
            "delta_accuracy": float((valid[treatment] - valid[reference]).mean()),
            "candidate_accuracy": float(valid[treatment].mean()),
            "reference_accuracy": float(valid[reference].mean()),
        }

    return calculate


def summarize(
    paired: pd.DataFrame, reference: str, treatment: str, samples: int, seed: int
) -> dict[str, Any]:
    fn = metric(reference, treatment)
    point = fn(paired)
    week = week_blocked_bootstrap(paired, fn, block="week", samples=samples, seed=seed)
    season = week_blocked_bootstrap(paired, fn, block="season", samples=samples, seed=seed)
    w = week.loc[week["metric"].eq("delta_accuracy")].iloc[0]
    s = season.loc[season["metric"].eq("delta_accuracy")].iloc[0]
    return {
        **point,
        "week_blocked_ci95": [float(w["lower"]), float(w["upper"])],
        "week_blocked_probability_positive": float(w["probability_positive"]),
        "season_blocked_ci95": [float(s["lower"]), float(s["upper"])],
        "season_blocked_probability_positive": float(s["probability_positive"]),
        "n_games": int(paired.dropna(subset=[reference, treatment]).shape[0]),
        "n_weeks": int(paired[["season", "week"]].drop_duplicates().shape[0]),
        "n_seasons": int(paired["season"].nunique()),
    }


def correct(pick_home: pd.Series, margin: pd.Series) -> pd.Series:
    values = np.where(pick_home.astype(bool), margin.gt(0), ~margin.gt(0)).astype(float)
    return pd.Series(np.where(margin.eq(0), np.nan, values), index=margin.index)


def null_distribution(
    paired: pd.DataFrame, *, probability_rule: bool, permutations: int, seed: int
) -> dict[str, Any]:
    suffix = "_pr" if probability_rule else ""
    base_pick, cand_pick = (
        paired[f"baseline_pick_home{suffix}"],
        paired[f"candidate_pick_home{suffix}"],
    )
    positions = [
        np.asarray(values, dtype=np.intp)
        for values in paired.groupby(["season", "week"], sort=False).indices.values()
    ]
    rng, deltas = np.random.default_rng(seed), []
    original = paired["margin_vs_open"].to_numpy(dtype=float)
    for _ in range(permutations):
        shuffled = original.copy()
        for group in positions:
            shuffled[group] = rng.permutation(shuffled[group])
        margin = pd.Series(shuffled, index=paired.index)
        value = pd.DataFrame(
            {"b": correct(base_pick, margin), "c": correct(cand_pick, margin)}
        ).dropna()
        deltas.append(float((value.c - value.b).mean()))
    values = np.asarray(deltas, dtype=float)
    observed = pd.DataFrame(
        {
            "b": correct(base_pick, paired["margin_vs_open"]),
            "c": correct(cand_pick, paired["margin_vs_open"]),
        }
    ).dropna()
    delta = float((observed.c - observed.b).mean())
    return {
        "permutations": permutations,
        "null_mean_delta": float(values.mean()),
        "null_sd_delta": float(values.std(ddof=1)),
        "null_q025": float(np.quantile(values, 0.025)),
        "null_q975": float(np.quantile(values, 0.975)),
        "observed_delta": delta,
        "fraction_of_null_below_observed": float((values < delta).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), required=True)
    parser.add_argument("--mode", choices=("null", "positive-control", "screen"), required=True)
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--market-root", type=Path, default=DEFAULT_MARKET_ROOT)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--permutations", type=int, default=NULL_PERMUTATIONS)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--min-train-games", type=int, default=DEFAULT_MIN_TRAIN_GAMES)
    args = parser.parse_args()
    candidate = CANDIDATES[args.candidate]
    features = pd.read_parquet(args.features or candidate.features)
    identity = profile_identity(candidate, features)
    scoped, seasons = scoped_window_frame(features, load_registry(args.registry), candidate.family)
    started = time.time()
    baseline = run_arm(
        scoped,
        candidate,
        market_root=args.market_root,
        profile=BASELINE_PROFILE,
        seasons=seasons,
        min_train_games=args.min_train_games,
        leak=False,
    )
    treatment = run_arm(
        scoped,
        candidate,
        market_root=args.market_root,
        profile=candidate.profile,
        seasons=seasons,
        min_train_games=args.min_train_games,
        leak=args.mode == "positive-control",
    )
    paired = paired_frame(baseline, treatment)
    if paired.empty:
        raise RuntimeError("No paired opener-grade games were scored")
    result: dict[str, Any] = {
        "status": "scored",
        "profile_identity": identity,
        "paired_games": len(paired),
        "paired_weeks": int(paired.groupby(["season", "week"]).ngroups),
    }
    if args.mode == "null":
        result["null_production_rule"] = null_distribution(
            paired, probability_rule=True, permutations=args.permutations, seed=args.seed
        )
        result["null_sign_rule"] = null_distribution(
            paired, probability_rule=False, permutations=args.permutations, seed=args.seed
        )
    else:
        for label, reference, treatment_col in (
            ("opener_production_rule", "baseline_correct_open_pr", "candidate_correct_open_pr"),
            ("opener_sign_rule", "baseline_correct_open", "candidate_correct_open"),
            ("close_production_rule", "baseline_correct_close_pr", "candidate_correct_close_pr"),
            ("close_sign_rule", "baseline_correct_close", "candidate_correct_close"),
        ):
            result[label] = summarize(
                paired, reference, treatment_col, args.bootstrap_samples, args.seed
            )
        result["permutation_null_production_rule"] = null_distribution(
            paired, probability_rule=True, permutations=args.permutations, seed=args.seed
        )
        result["baseline_metrics"] = opener_evaluation_metrics(baseline)
        result["candidate_metrics"] = opener_evaluation_metrics(treatment)
        result["picks_disagreeing_production_rule"] = int(
            (paired.baseline_pick_home_pr != paired.candidate_pick_home_pr).sum()
        )
    configuration = {
        "candidate": args.candidate,
        "mode": args.mode,
        "family": candidate.family,
        "inherits": candidate.inherits,
        "window_seasons": list(seasons),
        "grade": "opener",
        "baseline_profile": BASELINE_PROFILE,
        "candidate_profile": candidate.profile,
        "candidate_column": candidate.column,
        "predeclaration": candidate.predeclaration,
        "features_path": str(args.features or candidate.features),
        "market_root": str(args.market_root),
        "bootstrap_samples": args.bootstrap_samples,
        "permutations": args.permutations,
        "seed": args.seed,
    }
    payload = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "elapsed_seconds": round(time.time() - started, 1),
        **configuration,
        "result": result,
        "provenance": artifact_provenance(
            configuration, args.features or candidate.features, project_root=REPO_ROOT
        ),
    }
    output = (
        REPO_ROOT
        / "artifacts"
        / candidate.artifact_dir
        / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )
    write_experiment_artifact(
        output,
        "results.json",
        payload,
        command="on-production-opener-confirmation",
        metrics={"mode": args.mode, "status": "scored"},
        notes="Rotation-assigned opener confirmation; prediction-level paired output retained.",
    )
    paired.to_csv(output / "paired_predictions.csv", index=False)
    print(f"wrote {output / 'results.json'}")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
