from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.pick_probability_fit import FIT_FEATURES, FIT_RIDGE, _fit_logit

SEED = 20260920
BOOTSTRAPS = 10000
BANDS = (0.5, 0.52, 0.55, 0.58, 0.62, 1.0)
ARMS = ("raw", "simple", "combined", "neutral")


def fit_predict(
    train: pd.DataFrame, test: pd.DataFrame, features: tuple[str, ...]
) -> tuple[np.ndarray, dict[str, float]]:
    means = train[list(features)].mean()
    scales = train[list(features)].std(ddof=0).replace(0.0, 1.0)
    x = np.column_stack([np.ones(len(train)), (train[list(features)] - means) / scales])
    beta = _fit_logit(x, train["home_covered"].to_numpy(dtype=float), FIT_RIDGE)
    z = np.column_stack([np.ones(len(test)), (test[list(features)] - means) / scales]) @ beta
    natural = beta[1:] / scales.to_numpy()
    coefficients = dict(zip(features, natural.tolist(), strict=True))
    coefficients["intercept"] = float(beta[0] - np.sum(natural * means.to_numpy()))
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0))), coefficients


def interval(values: np.ndarray, *, estimate: float) -> dict[str, float]:
    values = values[np.isfinite(values)]
    return {
        "effect": float(estimate),
        "interval_low": float(np.quantile(values, 0.025)),
        "interval_high": float(np.quantile(values, 0.975)),
        "standard_error": float(np.std(values, ddof=1)),
        "probability_positive": float(np.mean(values > 0) + 0.5 * np.mean(values == 0)),
    }


def blocked_mean(frame: pd.DataFrame, values: np.ndarray) -> dict[str, float]:
    work = frame[["season", "week"]].copy()
    work["value"] = values
    groups = work.groupby(["season", "week"])["value"].agg(["sum", "count"])
    indices = np.random.default_rng(SEED).integers(0, len(groups), (BOOTSTRAPS, len(groups)))
    sums = groups["sum"].to_numpy()
    counts = groups["count"].to_numpy()
    samples = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return interval(samples, estimate=float(np.mean(values)))


def within_week_slope(frame: pd.DataFrame, arm: str) -> dict[str, float]:
    work = frame[["season", "week", f"{arm}_confidence", f"{arm}_correct"]].copy()
    group = work.groupby(["season", "week"])
    x = work[f"{arm}_confidence"] - group[f"{arm}_confidence"].transform("mean")
    y = work[f"{arm}_correct"] - group[f"{arm}_correct"].transform("mean")
    work["xy"] = x * y
    work["xx"] = x * x
    sums = work.groupby(["season", "week"])[["xy", "xx"]].sum()
    indices = np.random.default_rng(SEED).integers(0, len(sums), (BOOTSTRAPS, len(sums)))
    xy = sums["xy"].to_numpy()
    xx = sums["xx"].to_numpy()
    samples = 10.0 * xy[indices].sum(axis=1) / xx[indices].sum(axis=1)
    return interval(samples, estimate=float(10.0 * xy.sum() / xx.sum()))


def reliability(frame: pd.DataFrame, arm: str) -> list[dict[str, float | int]]:
    result = []
    for lower, upper in pairwise(BANDS):
        selected = frame.loc[
            frame[f"{arm}_confidence"].ge(lower) & frame[f"{arm}_confidence"].lt(upper)
        ]
        if selected.empty:
            continue
        actual = selected[f"{arm}_correct"].to_numpy(dtype=float)
        estimate = selected[f"{arm}_confidence"].to_numpy(dtype=float)
        result.append(
            {
                "lower": lower,
                "upper": upper,
                "games": len(selected),
                "predicted": float(estimate.mean()),
                "actual": float(actual.mean()),
                **blocked_mean(selected, (actual - estimate) * 100.0),
            }
        )
    return result


def score(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    frame = frame.copy()
    target = frame["home_covered"].to_numpy(dtype=float)
    metrics = {}
    for arm in ARMS:
        probability = np.clip(frame[f"{arm}_p"].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
        frame[f"{arm}_correct"] = (probability >= 0.5) == target
        frame[f"{arm}_confidence"] = np.maximum(probability, 1.0 - probability)
        frame[f"{arm}_brier"] = (probability - target) ** 2
        frame[f"{arm}_log_loss"] = -(
            target * np.log(probability) + (1.0 - target) * np.log1p(-probability)
        )
        metrics[arm] = {
            "games": len(frame),
            "correct": int(frame[f"{arm}_correct"].sum()),
            "accuracy": float(frame[f"{arm}_correct"].mean()),
            "brier": float(frame[f"{arm}_brier"].mean()),
            "log_loss": float(frame[f"{arm}_log_loss"].mean()),
            "reliability": reliability(frame, arm) if arm != "neutral" else [],
        }
    comparisons = {}
    for baseline in ("raw", "simple", "neutral"):
        for metric in ("brier", "log_loss"):
            values = (frame[f"{baseline}_{metric}"] - frame[f"combined_{metric}"]).to_numpy()
            comparisons[f"combined_vs_{baseline}_{metric}"] = blocked_mean(frame, values)
    weekly = frame.groupby(["season", "week"]).size().rename("games").reset_index()
    for arm in ("raw", "simple", "combined"):
        top = frame.sort_values(
            ["season", "week", f"{arm}_confidence", "game_id"],
            ascending=[True, True, False, True],
        ).drop_duplicates(["season", "week"])
        top = top[["season", "week", "game_id", f"{arm}_confidence", f"{arm}_correct"]]
        top = top.rename(columns={"game_id": f"{arm}_game_id"})
        weekly = weekly.merge(top, on=["season", "week"], validate="one_to_one")
    sums = frame.groupby(["season", "week"])["combined_correct"].sum().rename("all_correct")
    weekly = weekly.merge(sums, on=["season", "week"], validate="one_to_one")
    weekly["rest_accuracy"] = (weekly["all_correct"] - weekly["combined_correct"]) / (
        weekly["games"] - 1
    )
    weekly["top_vs_rest"] = 100.0 * (
        weekly["combined_correct"].astype(float) - weekly["rest_accuracy"]
    )
    comparisons["combined_top_vs_rest_accuracy"] = blocked_mean(
        weekly, weekly["top_vs_rest"].to_numpy()
    )
    comparisons["combined_top_vs_raw_top_accuracy"] = blocked_mean(
        weekly,
        100.0
        * (
            weekly["combined_correct"].astype(float) - weekly["raw_correct"].astype(float)
        ).to_numpy(),
    )
    different = weekly["combined_correct"].ne(weekly["raw_correct"])
    decisive = weekly.loc[different]
    slopes = {arm: within_week_slope(frame, arm) for arm in ("raw", "simple", "combined")}
    seasons = []
    for season, rows in frame.groupby("season"):
        stars = weekly.loc[weekly["season"].eq(season)]
        seasons.append(
            {
                "season": int(season),
                "games": len(rows),
                "combined_accuracy": float(rows["combined_correct"].mean()),
                "combined_top_record": f"{int(stars['combined_correct'].sum())}/{len(stars)}",
                "combined_top_confidence": float(stars["combined_confidence"].mean()),
                "combined_brier": float(rows["combined_brier"].mean()),
                "raw_brier": float(rows["raw_brier"].mean()),
                "within_week_slope": within_week_slope(rows, "combined"),
            }
        )
    summary = {
        "metrics": metrics,
        "comparisons": comparisons,
        "within_week_slope_accuracy_points_per_10_confidence_points": slopes,
        "weekly": {
            "weeks": len(weekly),
            "combined_correct": int(weekly["combined_correct"].sum()),
            "raw_correct": int(weekly["raw_correct"].sum()),
            "combined_mean_estimate": float(weekly["combined_confidence"].mean()),
            "rest_accuracy": float(weekly["rest_accuracy"].mean()),
            "different_nominees": int(weekly["combined_game_id"].ne(weekly["raw_game_id"]).sum()),
            "decisive_weeks": len(decisive),
            "decisive_combined_wins": int(decisive["combined_correct"].sum()),
            "decisive_raw_wins": int(decisive["raw_correct"].sum()),
        },
        "seasons": seasons,
    }
    return frame, summary, weekly


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a frozen probability population without activation"
    )
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    population = pd.read_parquet(args.directory / "population.parquet")
    seasons = sorted(population["season"].unique())
    coefficients = []
    summaries = {}
    for protocol in ("in_sample", "leave_one_season_out", "chronological"):
        frame = population.copy()
        frame["raw_p"] = frame["model_probability"]
        frame["neutral_p"] = 0.5
        for arm in ("simple", "combined"):
            frame[f"{arm}_p"] = np.nan
        for held in seasons:
            if protocol == "chronological":
                train = population.loc[population["season"].lt(held)]
                if train["season"].nunique() < 2:
                    continue
            elif protocol == "leave_one_season_out":
                train = population.loc[population["season"].ne(held)]
            else:
                train = population
            test = population.loc[population["season"].eq(held)]
            if protocol == "chronological" and train["season"].max() >= held:
                raise ValueError("Chronological training reached the held-out season")
            for arm, features in (("simple", ("model_logit",)), ("combined", FIT_FEATURES)):
                predictions, fitted = fit_predict(train, test, features)
                frame.loc[test.index, f"{arm}_p"] = predictions
                coefficients.append({"protocol": protocol, "arm": arm, "held": int(held), **fitted})
        scored, summary, weekly = score(frame.dropna(subset=["combined_p"]).reset_index(drop=True))
        scored.to_parquet(args.directory / f"{protocol}_predictions.parquet", index=False)
        weekly.to_csv(args.directory / f"{protocol}_weekly.csv", index=False)
        summaries[protocol] = summary
    summaries["coefficient_fits"] = len(coefficients)
    summaries["seed"] = SEED
    summaries["bootstrap_samples"] = BOOTSTRAPS
    summaries["look_inventory"] = {
        "probability_arms": len(ARMS),
        "protocols": 3,
        "coefficient_fit_calls": len(coefficients),
        "arm_metric_cells": len(ARMS) * 3 * 3,
        "reliability_bands": sum(
            len(summaries[p]["metrics"][a]["reliability"])
            for p in ("in_sample", "leave_one_season_out", "chronological")
            for a in ARMS
        ),
        "paired_comparisons": 3 * 8,
        "pooled_slopes": 3 * 3,
        "season_slopes": sum(
            len(summaries[p]["seasons"])
            for p in ("in_sample", "leave_one_season_out", "chronological")
        ),
    }
    summaries["limitations"] = [
        "Earlier feature and policy selection reused these seasons; no untouched outer test.",
        "Bootstrap conditions on replay predictions and resamples week blocks, "
        "not feature discovery.",
        "Neutral market reference is 0.5, not archived no-vig prices.",
        "Source policy is recorded in population_metadata.json; legacy smooth-input replay "
        "does not validate discrete serving.",
    ]
    (args.directory / "audit.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(coefficients).to_csv(args.directory / "coefficients.csv", index=False)
    for protocol in ("in_sample", "leave_one_season_out", "chronological"):
        summary = summaries[protocol]
        print(
            json.dumps(
                {
                    "protocol": protocol,
                    "weekly": summary["weekly"],
                    "comparisons": summary["comparisons"],
                    "slopes": summary["within_week_slope_accuracy_points_per_10_confidence_points"],
                }
            )
        )


if __name__ == "__main__":
    main()
