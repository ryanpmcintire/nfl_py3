"""Measure whether the ATS evaluator recovers deliberately injected signal.

This is a positive-control experiment, not a candidate football feature. Each
synthetic feature is generated independently of real outcomes, then given a
known counterfactual effect on ATS margin. The same chronological weekly Ridge
path used by the active market-residual model must learn that effect using only
earlier games.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.margin import make_margin_estimator, margin_feature_columns

EFFECTS = (0.0, 0.5, 1.0, 2.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _blocked_interval(
    frame: pd.DataFrame,
    *,
    candidate: str,
    block: str,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    candidate_correct = frame[f"{candidate}_correct"].to_numpy(dtype=float)
    baseline_correct = frame["baseline_correct"].to_numpy(dtype=float)
    differences = candidate_correct - baseline_correct
    point = float(differences.mean())
    keys = (
        frame["season"].astype(str) + "-" + frame["week"].astype(str)
        if block == "week"
        else frame["season"].astype(str)
    )
    grouped = (
        pd.DataFrame({"key": keys, "difference": differences})
        .groupby("key", sort=True)["difference"]
        .agg(["sum", "size"])
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(grouped), size=(samples, len(grouped)))
    draws = grouped["sum"].to_numpy()[indices].sum(axis=1) / grouped["size"].to_numpy()[
        indices
    ].sum(axis=1)
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _evaluate_replica(
    predictions: pd.DataFrame,
    *,
    replica: int,
    effect: float,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    actual_margin = predictions["ats_margin"] + effect * predictions[f"signal_{replica}"]
    keep = actual_margin.ne(0.0)
    frame = predictions.loc[keep].copy()
    actual_home = actual_margin.loc[keep].gt(0.0)
    effect_index = EFFECTS.index(effect)
    frame["baseline_correct"] = (
        frame[f"baseline_probability_{effect_index}_{replica}"].ge(0.5).eq(actual_home)
    )
    frame["signal_correct"] = (
        frame[f"signal_probability_{effect_index}_{replica}"].ge(0.5).eq(actual_home)
    )
    frame["permuted_correct"] = (
        frame[f"permuted_probability_{effect_index}_{replica}"].ge(0.5).eq(actual_home)
    )
    row: dict[str, Any] = {
        "replica": replica,
        "effect_points_per_sd": effect,
        "games": len(frame),
        "baseline_accuracy": float(frame["baseline_correct"].mean()),
        "signal_accuracy": float(frame["signal_correct"].mean()),
        "permuted_accuracy": float(frame["permuted_correct"].mean()),
    }
    for candidate in ("signal", "permuted"):
        for block in ("week", "season"):
            point, lower, upper = _blocked_interval(
                frame,
                candidate=candidate,
                block=block,
                samples=samples,
                seed=seed + replica * 101 + (0 if block == "week" else 10_000),
            )
            row[f"{candidate}_lift"] = point
            row[f"{candidate}_{block}_lower"] = lower
            row[f"{candidate}_{block}_upper"] = upper
    return row


def _fit_components(
    training: pd.DataFrame,
    scoring: pd.DataFrame,
    feature_columns: list[str],
    target: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    distribution_rows = int(len(training) * 0.20)
    split = len(training) - distribution_rows
    temporary = make_margin_estimator("ridge", ridge_alpha=10.0)
    temporary.fit(training.iloc[:split][feature_columns], target[:split])
    residuals = target[split:] - np.asarray(
        temporary.predict(training.iloc[split:][feature_columns]), dtype=float
    )
    final = make_margin_estimator("ridge", ridge_alpha=10.0)
    final.fit(training[feature_columns], target)
    prediction = np.asarray(final.predict(scoring[feature_columns]), dtype=float)
    return prediction, residuals


def _smoothed_probabilities(
    center: np.ndarray[Any, Any], residuals: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    successes = (residuals[None, :] > -center[:, None]).sum(axis=1)
    return (successes + 0.5) / (len(residuals) + 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/game_features_player_value.parquet"),
    )
    parser.add_argument(
        "--active-predictions",
        type=Path,
        default=Path("artifacts/margins/20260812T205551Z/predictions.parquet"),
    )
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.replicas < 1 or args.bootstrap_samples < 100:
        raise ValueError("replicas must be positive and bootstrap samples must be at least 100")
    features = pd.read_parquet(args.features)
    features["gameday"] = pd.to_datetime(features["gameday"], errors="raise")
    completed = features.loc[features["result"].notna()].copy()
    completed = completed.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    signals = rng.standard_normal((len(completed), args.replicas))
    permutations = np.column_stack(
        [rng.permutation(signals[:, replica]) for replica in range(args.replicas)]
    )
    for replica in range(args.replicas):
        completed[f"signal_{replica}"] = signals[:, replica]
        completed[f"permuted_{replica}"] = permutations[:, replica]

    base_columns = list(margin_feature_columns("market_residual", "player"))
    batches: list[pd.DataFrame] = []
    test = completed.loc[completed["season"].ge(2018)]
    target_columns = ["ats_margin", *[f"signal_{i}" for i in range(args.replicas)]]
    for (_, _), weekly_games in test.groupby(["season", "week"], sort=True):
        cutoff = weekly_games["gameday"].min()
        training = completed.loc[completed["gameday"].lt(cutoff)]
        if len(training) < 500:
            continue
        target = training[target_columns].to_numpy(dtype=float)
        baseline_prediction, baseline_residuals = _fit_components(
            training, weekly_games, base_columns, target
        )
        batch = weekly_games[
            ["game_id", "season", "week", "gameday", "ats_margin", *target_columns[1:]]
        ].copy()
        batch["baseline_yhat"] = baseline_prediction[:, 0]
        for replica in range(args.replicas):
            for effect_index, effect in enumerate(EFFECTS):
                baseline_center = (
                    baseline_prediction[:, 0] + effect * baseline_prediction[:, replica + 1]
                )
                baseline_residual = (
                    baseline_residuals[:, 0] + effect * baseline_residuals[:, replica + 1]
                )
                batch[f"baseline_probability_{effect_index}_{replica}"] = _smoothed_probabilities(
                    baseline_center, baseline_residual
                )
            for kind in ("signal", "permuted"):
                synthetic_column = f"{kind}_{replica}"
                prediction, residuals = _fit_components(
                    training,
                    weekly_games,
                    [*base_columns, synthetic_column],
                    target[:, [0, replica + 1]],
                )
                for effect_index, effect in enumerate(EFFECTS):
                    center = prediction[:, 0] + effect * prediction[:, 1]
                    residual = residuals[:, 0] + effect * residuals[:, 1]
                    batch[f"{kind}_probability_{effect_index}_{replica}"] = _smoothed_probabilities(
                        center, residual
                    )
        batches.append(batch)
    predictions = pd.concat(batches, ignore_index=True)

    active = pd.read_parquet(args.active_predictions)
    active = active.loc[active["method"].eq("market_residual")]
    reproduction = predictions.merge(
        active[["game_id", "predicted_market_residual", "home_cover_probability"]],
        on="game_id",
        validate="one_to_one",
    )
    max_prediction_error = float(
        np.abs(reproduction["baseline_yhat"] - reproduction["predicted_market_residual"]).max()
    )
    max_probability_error = float(
        np.abs(
            reproduction["baseline_probability_0_0"] - reproduction["home_cover_probability"]
        ).max()
    )
    zero_nonpush = reproduction.loc[reproduction["ats_margin"].ne(0.0)]
    reproduced_correct = int(
        zero_nonpush["baseline_probability_0_0"]
        .ge(0.5)
        .eq(zero_nonpush["ats_margin"].gt(0.0))
        .sum()
    )
    if (
        len(reproduction) != len(predictions)
        or max_prediction_error > 1e-9
        or max_probability_error > 1e-12
        or reproduced_correct != 1080
    ):
        raise RuntimeError(
            "Positive-control evaluator did not reproduce the active evaluation: "
            f"rows={len(reproduction)}/{len(predictions)}, "
            f"prediction_error={max_prediction_error}, probability_error={max_probability_error}, "
            f"correct={reproduced_correct}/1080"
        )

    rows = [
        _evaluate_replica(
            predictions,
            replica=replica,
            effect=effect,
            samples=args.bootstrap_samples,
            seed=args.seed,
        )
        for effect in EFFECTS
        for replica in range(args.replicas)
    ]
    details = pd.DataFrame(rows)
    summary = (
        details.groupby("effect_points_per_sd", as_index=False)
        .agg(
            replicas=("replica", "size"),
            mean_baseline_accuracy=("baseline_accuracy", "mean"),
            mean_signal_accuracy=("signal_accuracy", "mean"),
            mean_signal_lift=("signal_lift", "mean"),
            min_signal_lift=("signal_lift", "min"),
            max_signal_lift=("signal_lift", "max"),
            positive_signal_replicas=("signal_lift", lambda value: int(value.gt(0).sum())),
            week_detected_replicas=("signal_week_lower", lambda value: int(value.gt(0).sum())),
            season_detected_replicas=("signal_season_lower", lambda value: int(value.gt(0).sum())),
            mean_permuted_lift=("permuted_lift", "mean"),
            permuted_week_false_positives=(
                "permuted_week_lower",
                lambda value: int(value.gt(0).sum()),
            ),
            permuted_season_false_positives=(
                "permuted_season_lower",
                lambda value: int(value.gt(0).sum()),
            ),
        )
        .sort_values("effect_points_per_sd")
    )
    run = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("artifacts/sensitivity_audits") / run
    output.mkdir(parents=True, exist_ok=False)
    details.to_csv(output / "replica_results.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "positive control only; synthetic target signal is not a football feature",
        "features": str(args.features),
        "features_sha256": _sha256(args.features),
        "active_predictions": str(args.active_predictions),
        "active_predictions_sha256": _sha256(args.active_predictions),
        "active_profile": "player",
        "target": "market_residual",
        "regressor": "ridge",
        "ridge_alpha": 10.0,
        "start_season": 2018,
        "replicas": args.replicas,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "synthetic_effects_points_per_feature_sd": list(EFFECTS),
        "active_prediction_reproduction_max_absolute_error": max_prediction_error,
        "active_probability_reproduction_max_absolute_error": max_probability_error,
        "active_classification_reproduction_correct": reproduced_correct,
        "prediction_rows": len(predictions),
        "nonpush_games_at_zero_effect": int(predictions["ats_margin"].ne(0).sum()),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({**metadata, "artifact_directory": str(output)}, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
