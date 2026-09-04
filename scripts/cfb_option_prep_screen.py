"""CFB triple-option prep-deficit screen (LEAD-45, docs/cfb_option_prep_screen.md).

Predeclared before any outcome is scored: BACK the option-team side via
one signed column (``cfb_option_side``: +1 home option, -1 away option,
0 otherwise) on top of the frozen XLG-03 benchmark arm. Spends no NFL
evaluation window and no rotation window. All cells are recorded
regardless of sign; an interval crossing zero is never a rejection.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_ats.cfb_benchmark import (
    CFB_BENCHMARK_MIN_TRAIN_GAMES,
    CFB_BENCHMARK_RIDGE_ALPHA,
    CFB_CLEAN_CORE_SEASONS,
    fit_cfb_residual_model,
)
from nfl_ats.cfb_features import CFB_MODEL_FEATURE_COLUMNS
from nfl_ats.clv import pick_correct, week_blocked_bootstrap
from nfl_ats.provenance import artifact_provenance, write_experiment_artifact

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = REPO_ROOT / "data" / "processed" / "cfb_game_features.parquet"
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "cfb_option_prep"
CANDIDATE_COLUMN = "cfb_option_side"
BOOTSTRAP_SAMPLES = 1_000
SEED = 20260904
PERMUTATIONS = 200
ERAS: tuple[tuple[str, int, int], ...] = (("2012_2019", 2012, 2019), ("2021_2025", 2021, 2025))

#: Permanent triple-option programs, plus Georgia Tech for the Paul
#: Johnson era (2008-2018). Frozen by the predeclaration, never fitted.
ALWAYS_OPTION = frozenset({"Army", "Navy", "Air Force"})


def is_option_team(team: str, season: int) -> bool:
    """Deterministic identity flag (zero measurement error)."""

    if team in ALWAYS_OPTION:
        return True
    return team == "Georgia Tech" and 2008 <= season <= 2018


def attach_option_flag(features: pd.DataFrame) -> pd.DataFrame:
    """Signed prep-asymmetry column, computed from identity only."""

    frame = features.copy()
    seasons = pd.to_numeric(frame["season"], errors="coerce").fillna(0).astype(int)
    home_option = [
        is_option_team(str(team), int(season))
        for team, season in zip(frame["home_team"], seasons, strict=True)
    ]
    away_option = [
        is_option_team(str(team), int(season))
        for team, season in zip(frame["away_team"], seasons, strict=True)
    ]
    frame[CANDIDATE_COLUMN] = np.asarray(home_option, dtype=float) - np.asarray(
        away_option, dtype=float
    )
    return frame


def run_walk_forward(
    attached: pd.DataFrame,
    scored_seasons: tuple[int, ...],
    *,
    leak_treatment: bool,
) -> pd.DataFrame:
    completed = attached.loc[
        pd.to_numeric(attached["result"], errors="coerce").notna()
        & pd.to_numeric(attached["ats_margin"], errors="coerce").notna()
    ].copy()
    candidate_source = completed
    if leak_treatment:
        candidate_source = completed.copy()
        candidate_source[CANDIDATE_COLUMN] = pd.to_numeric(
            candidate_source["ats_margin"], errors="coerce"
        )
    candidate_columns = (*CFB_MODEL_FEATURE_COLUMNS, CANDIDATE_COLUMN)
    scored = completed.loc[completed["season"].astype(int).isin(scored_seasons)]
    rows: list[dict[str, Any]] = []
    for (season_value, week_value), group in scored.groupby(["season", "week"], sort=True):
        cutoff = group["gameday"].min()
        baseline_training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(baseline_training) < CFB_BENCHMARK_MIN_TRAIN_GAMES:
            continue
        candidate_training = candidate_source.loc[candidate_source["gameday"].lt(cutoff)]
        baseline_model = fit_cfb_residual_model(
            baseline_training,
            ridge_alpha=CFB_BENCHMARK_RIDGE_ALPHA,
            feature_columns=CFB_MODEL_FEATURE_COLUMNS,
        )
        candidate_model = fit_cfb_residual_model(
            candidate_training,
            ridge_alpha=CFB_BENCHMARK_RIDGE_ALPHA,
            feature_columns=candidate_columns,
        )
        candidate_scoring = (
            group
            if not leak_treatment
            else group.assign(
                **{CANDIDATE_COLUMN: pd.to_numeric(group["ats_margin"], errors="coerce")}
            )
        )
        settle_margin = pd.to_numeric(group["result"], errors="coerce") - pd.to_numeric(
            group["spread_line"], errors="coerce"
        )
        baseline_probability = baseline_model.predict(group)["home_cover_probability"]
        candidate_probability = candidate_model.predict(candidate_scoring)["home_cover_probability"]
        for game_id, margin, base, cand, feature_value in zip(
            group["game_id"],
            settle_margin,
            baseline_probability,
            candidate_probability,
            group[CANDIDATE_COLUMN],
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
                    "feature_value": feature_value,
                }
            )
    return pd.DataFrame(rows)


def grade(frame: pd.DataFrame, margins: pd.Series | None = None) -> pd.DataFrame:
    settle = frame["settle_margin"] if margins is None else margins
    graded = frame.copy()
    for arm, column in (
        ("baseline", "baseline_probability"),
        ("candidate", "candidate_probability"),
    ):
        graded[f"{arm}_correct"] = pick_correct(graded[column].ge(0.5), settle)
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


def week_positions(frame: pd.DataFrame) -> list[np.ndarray]:
    return [
        np.asarray(positions, dtype=np.intp)
        for positions in frame.groupby(["season", "week"], sort=False).indices.values()
    ]


def null_distribution(frame: pd.DataFrame, *, permutations: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    metric = _paired_metric("baseline_correct", "candidate_correct")
    groups = week_positions(frame)
    deltas = []
    for _ in range(permutations):
        values = frame["settle_margin"].to_numpy(dtype=float, copy=True)
        for positions in groups:
            values[positions] = rng.permutation(values[positions])
        permuted = pd.Series(values, index=frame.index)
        deltas.append(metric(grade(frame, permuted))["delta_accuracy"])
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
    if paired.empty or paired.dropna(subset=["baseline_correct", "candidate_correct"]).empty:
        return None
    metric = _paired_metric("baseline_correct", "candidate_correct")
    point = metric(paired)
    week = week_blocked_bootstrap(paired, metric, block="week", samples=samples, seed=seed)
    week_row = week.loc[week["metric"].eq("delta_accuracy")].iloc[0]
    summary: dict[str, Any] = {
        "delta_accuracy": point["delta_accuracy"],
        "candidate_accuracy": point["candidate_accuracy"],
        "baseline_accuracy": point["reference_accuracy"],
        "week_blocked_ci95": [float(week_row["lower"]), float(week_row["upper"])],
        "week_blocked_probability_positive": float(week_row["probability_positive"]),
        "n_games": len(paired.dropna(subset=["baseline_correct", "candidate_correct"])),
        "n_weeks": int(paired[["season", "week"]].drop_duplicates().shape[0]),
        "n_seasons": int(paired["season"].nunique()),
        "n_flagged_nonzero": int(
            (pd.to_numeric(paired["feature_value"], errors="coerce") != 0).sum()
        ),
    }
    if paired["season"].nunique() >= 2:
        season = week_blocked_bootstrap(paired, metric, block="season", samples=samples, seed=seed)
        season_row = season.loc[season["metric"].eq("delta_accuracy")].iloc[0]
        summary["season_blocked_ci95"] = [float(season_row["lower"]), float(season_row["upper"])]
        summary["season_blocked_probability_positive"] = float(season_row["probability_positive"])
    else:
        summary["season_blocked_ci95"] = None
        summary["season_blocked_probability_positive"] = None
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("coverage", "null", "positive-control", "screen"), required=True
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print("=== loading features ===", flush=True)
    features = pd.read_parquet(args.features)
    features["gameday"] = pd.to_datetime(features["gameday"], errors="raise")
    attached = attach_option_flag(features)
    attached = attached.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    if args.mode == "coverage":
        clean = attached.loc[attached["season"].astype(int).isin(CFB_CLEAN_CORE_SEASONS)]
        flagged = clean.loc[clean[CANDIDATE_COLUMN] != 0]
        print(f"clean-core games: {len(clean)}")
        print(f"flagged (sole option side): {len(flagged)}")
        both = clean.loc[
            clean.apply(
                lambda row: (
                    is_option_team(str(row["home_team"]), int(row["season"]))
                    and is_option_team(str(row["away_team"]), int(row["season"]))
                ),
                axis=1,
            )
        ]
        print(f"option-vs-option games (flag 0, kept as baseline): {len(both)}")
        print("flagged by season:")
        print(flagged.groupby("season").size().to_string())
        return 0

    fitted = run_walk_forward(
        attached,
        tuple(CFB_CLEAN_CORE_SEASONS),
        leak_treatment=args.mode == "positive-control",
    )
    if fitted.empty:
        print("no scored games")
        return 1
    if args.mode == "null":
        null = null_distribution(fitted, permutations=args.permutations, seed=args.seed)
        print(json.dumps(null, indent=2))
        return 0

    graded = grade(fitted)
    result: dict[str, Any] = {
        "status": "scored",
        "candidate_column": CANDIDATE_COLUMN,
        "grade": "close_proxy_median_book_spread_line",
        "league": "cfb",
        "seed": args.seed,
        "candidate_vs_baseline_pooled": summarize_pair(
            graded, samples=args.bootstrap_samples, seed=args.seed
        ),
        "permutation_null": null_distribution(
            fitted, permutations=args.permutations, seed=args.seed
        ),
        "era_results": {
            label: (
                summarize_pair(
                    graded.loc[graded["season"].between(start, end)],
                    samples=args.bootstrap_samples,
                    seed=args.seed,
                )
            )
            for label, start, end in ERAS
        },
        "home_pick_rate": {
            "baseline": float(graded["baseline_probability"].ge(0.5).mean()),
            "candidate": float(graded["candidate_probability"].ge(0.5).mean()),
        },
    }
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = ARTIFACT_ROOT / stamp
    configuration = {
        "cell": "option_side",
        "predicted_direction": "back the option-team side",
        "mode": args.mode,
        "scored_seasons": list(CFB_CLEAN_CORE_SEASONS),
        "grade": "close_proxy_median_book_spread_line",
        "league": "cfb",
        "baseline_feature_columns": list(CFB_MODEL_FEATURE_COLUMNS),
        "candidate_column": CANDIDATE_COLUMN,
        "regressor": "ridge",
        "ridge_alpha": CFB_BENCHMARK_RIDGE_ALPHA,
        "min_train_games": CFB_BENCHMARK_MIN_TRAIN_GAMES,
        "bootstrap_samples": args.bootstrap_samples,
        "permutations": args.permutations,
        "seed": args.seed,
        "predeclaration": "docs/cfb_option_prep_screen.md",
        "features_path": str(args.features),
    }
    payload = {
        **configuration,
        "result": result,
        "provenance": artifact_provenance(configuration, args.features, project_root=REPO_ROOT),
    }
    write_experiment_artifact(
        out_dir,
        "results.json",
        payload,
        command="cfb-option-prep-screen",
        metrics={"cell": "option_side", "mode": args.mode, "status": result.get("status")},
        notes=(
            "LEAD-45 triple-option prep-deficit screen on the frozen XLG-03 "
            "benchmark arm; no NFL window spent. See docs/cfb_option_prep_screen.md."
        ),
    )
    pooled = result["candidate_vs_baseline_pooled"]
    low, high = pooled["week_blocked_ci95"]
    print(
        f"pooled: delta {pooled['delta_accuracy'] * 100:+.3f} pts  P+ "
        f"{pooled['week_blocked_probability_positive']:.3f}  week 95% "
        f"[{low * 100:+.3f}, {high * 100:+.3f}]  n={pooled['n_games']} games, "
        f"{pooled['n_weeks']} weeks, flagged={pooled['n_flagged_nonzero']}"
    )
    null = result["permutation_null"]
    print(
        f"null: mean {null['null_mean_delta'] * 100:+.3f}, observed at the "
        f"{null['fraction_of_null_below_observed'] * 100:.1f}th percentile"
    )
    for label, _start, _end in ERAS:
        era = result["era_results"][label]
        if era is None:
            print(f"era {label}: no scored games")
        else:
            elow, ehigh = era["week_blocked_ci95"]
            print(
                f"era {label}: delta {era['delta_accuracy'] * 100:+.3f} pts  P+ "
                f"{era['week_blocked_probability_positive']:.3f}  "
                f"[{elow * 100:+.3f}, {ehigh * 100:+.3f}]"
            )
    print(f"wrote {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
