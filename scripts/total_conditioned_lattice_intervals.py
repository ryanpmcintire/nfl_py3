from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO / "artifacts" / "total_conditioned_lattice" / "20260923T205550Z"
OUT_ROOT = REPO / "artifacts" / "total_conditioned_lattice_intervals"
N_DRAWS = 2000
SEED = 20260923


def load(label: str) -> pd.DataFrame:
    return pd.read_parquet(ARTIFACT_DIR / f"{label}.parquet")


def block_index_map(keys: np.ndarray) -> dict:
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    boundaries = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
    groups = np.split(order, boundaries)
    unique = sorted_keys[np.concatenate(([0], boundaries))] if len(sorted_keys) else np.array([])
    return dict(zip(unique.tolist(), groups, strict=True))


def block_metrics(df: pd.DataFrame, idx: np.ndarray) -> dict:
    served_ll = df["served_log_loss"].to_numpy()[idx]
    challenger_ll = df["challenger_log_loss"].to_numpy()[idx]
    served_br = df["served_brier"].to_numpy()[idx]
    challenger_br = df["challenger_brier"].to_numpy()[idx]
    outcome = df["outcome"].to_numpy()[idx]
    served_cover = df["served_cover"].to_numpy()[idx]
    served_loss = df["served_loss"].to_numpy()[idx]
    challenger_cover = df["challenger_cover"].to_numpy()[idx]
    challenger_loss = df["challenger_loss"].to_numpy()[idx]
    is_integer = df["is_integer_line"].to_numpy()[idx]
    served_push = df["served_push"].to_numpy()[idx]
    challenger_push = df["challenger_push"].to_numpy()[idx]

    log_loss_improvement = float(served_ll.mean() - challenger_ll.mean())
    brier_improvement = float(served_br.mean() - challenger_br.mean())

    decisive = outcome != 1
    served_pick_home = served_cover[decisive] >= served_loss[decisive]
    challenger_pick_home = challenger_cover[decisive] >= challenger_loss[decisive]
    actual_home = outcome[decisive] == 0
    served_acc = float((served_pick_home == actual_home).mean())
    challenger_acc = float((challenger_pick_home == actual_home).mean())
    accuracy_points = (challenger_acc - served_acc) * 100.0

    integer_rows = is_integer.astype(bool)
    served_push_pred = float(served_push[integer_rows].mean())
    challenger_push_pred = float(challenger_push[integer_rows].mean())
    actual_push = float((outcome[integer_rows] == 1).mean())
    served_push_error = abs(served_push_pred - actual_push)
    challenger_push_error = abs(challenger_push_pred - actual_push)
    push_calibration_improvement = served_push_error - challenger_push_error

    return {
        "log_loss_improvement": log_loss_improvement,
        "brier_improvement": brier_improvement,
        "accuracy_points": accuracy_points,
        "push_calibration_improvement": push_calibration_improvement,
    }


def block_bootstrap(df: pd.DataFrame, block_keys: np.ndarray, *, n_draws: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    idx_map = block_index_map(block_keys)
    unique_blocks = np.array(list(idx_map.keys()), dtype=object)
    n_blocks = len(unique_blocks)
    draws = {
        "log_loss_improvement": np.empty(n_draws),
        "brier_improvement": np.empty(n_draws),
        "accuracy_points": np.empty(n_draws),
        "push_calibration_improvement": np.empty(n_draws),
    }
    for d in range(n_draws):
        chosen = rng.choice(unique_blocks, size=n_blocks, replace=True)
        idx = np.concatenate([idx_map[b] for b in chosen])
        metrics = block_metrics(df, idx)
        for key, value in metrics.items():
            draws[key][d] = value
    return draws


def summarize_bootstrap(observed: float, draws: np.ndarray) -> dict:
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "observed": observed,
        "bootstrap_mean": float(draws.mean()),
        "standard_error": float(draws.std(ddof=1)),
        "interval_low": float(low),
        "interval_high": float(high),
        "probability_positive": float((draws > 0).mean()),
        "n_draws": int(draws.size),
    }


def disagreement_exact_null(df: pd.DataFrame) -> dict:
    outcome = df["outcome"].to_numpy()
    decisive = outcome != 1
    served_pick_home = (
        df["served_cover"].to_numpy()[decisive] >= df["served_loss"].to_numpy()[decisive]
    )
    challenger_pick_home = (
        df["challenger_cover"].to_numpy()[decisive] >= df["challenger_loss"].to_numpy()[decisive]
    )
    actual_home = outcome[decisive] == 0
    disagree = served_pick_home != challenger_pick_home
    n = int(disagree.sum())
    challenger_correct = int(((challenger_pick_home == actual_home) & disagree).sum())
    served_correct = int(((served_pick_home == actual_home) & disagree).sum())
    result = stats.binomtest(challenger_correct, n=n, p=0.5, alternative="two-sided")
    ci = result.proportion_ci(confidence_level=0.95)
    return {
        "n_disagreement_games": n,
        "challenger_correct": challenger_correct,
        "served_correct": served_correct,
        "challenger_win_rate": challenger_correct / n,
        "exact_binomial_p_two_sided": float(result.pvalue),
        "challenger_win_rate_ci95_low": float(ci.low),
        "challenger_win_rate_ci95_high": float(ci.high),
    }


def split_half_reliability(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["improvement"] = df["served_log_loss"] - df["challenger_log_loss"]
    odd_means = []
    even_means = []
    per_season = []
    for season, group in df.groupby("season", sort=True):
        odd = group.loc[group["week"] % 2 == 1, "improvement"]
        even = group.loc[group["week"] % 2 == 0, "improvement"]
        if odd.empty or even.empty:
            continue
        odd_means.append(float(odd.mean()))
        even_means.append(float(even.mean()))
        per_season.append(
            {
                "season": int(season),
                "odd_week_mean_improvement": float(odd.mean()),
                "even_week_mean_improvement": float(even.mean()),
                "odd_n": int(odd.size),
                "even_n": int(even.size),
            }
        )
    odd_arr = np.array(odd_means)
    even_arr = np.array(even_means)
    r, p = stats.pearsonr(odd_arr, even_arr)
    r = float(r)
    spearman_brown = float(2 * r / (1 + r)) if (1 + r) != 0 else float("nan")
    return {
        "method": "per-season mean log_loss_improvement, odd-parity weeks vs even-parity weeks, "
        "Pearson r across season units, Spearman-Brown corrected for full-length reliability",
        "n_season_units": int(odd_arr.size),
        "pearson_r": r,
        "pearson_p": float(p),
        "spearman_brown_corrected_r": spearman_brown,
        "per_season": per_season,
    }


def main() -> None:
    primary = load("primary_2020_2025")
    extended = load("extended_2009_2025")

    observed_primary = block_metrics(primary, np.arange(len(primary)))
    observed_extended = block_metrics(extended, np.arange(len(extended)))

    season_draws = block_bootstrap(
        primary, primary["season"].to_numpy(), n_draws=N_DRAWS, seed=SEED
    )
    week_keys = (primary["season"].astype(str) + "_" + primary["week"].astype(str)).to_numpy(
        dtype=object
    )
    week_draws = block_bootstrap(primary, week_keys, n_draws=N_DRAWS, seed=SEED)

    season_block_summary = {
        key: summarize_bootstrap(observed_primary[key], season_draws[key])
        for key in observed_primary
    }
    week_block_summary = {
        key: summarize_bootstrap(observed_primary[key], week_draws[key]) for key in observed_primary
    }

    disagreement = disagreement_exact_null(primary)
    reliability = split_half_reliability(primary)

    out = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "source_artifact_dir": str(ARTIFACT_DIR),
        "n_draws": N_DRAWS,
        "seed": SEED,
        "sign_convention": "positive = challenger (total-conditioned) favorable; "
        "log_loss_improvement = served_log_loss - challenger_log_loss; "
        "brier_improvement = served_brier - challenger_brier; "
        "accuracy_points = challenger_accuracy_pct - served_accuracy_pct on decisive games; "
        "push_calibration_improvement = abs(served_predicted_minus_actual_push) - "
        "abs(challenger_predicted_minus_actual_push) at integer lines",
        "primary_2020_2025": {
            "games": len(primary),
            "observed_point_estimates": observed_primary,
            "season_block_bootstrap": season_block_summary,
            "week_block_bootstrap": week_block_summary,
            "disagreement_exact_null": disagreement,
            "split_half_reliability_log_loss_improvement": reliability,
        },
        "extended_2009_2025_reported_only": {
            "games": len(extended),
            "observed_point_estimates": observed_extended,
        },
    }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ts = out["generated_at_utc"]
    out_dir = OUT_ROOT / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps(out, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
