from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from nfl_ats.pick_probability import (
    BASE_PROBABILITY_POLICY,
    COUNTED_FLAG_COLUMNS,
    FLAG_SUM_COLUMN,
    MOVE_AVAILABLE_COLUMN,
    MOVE_COLUMN,
    PICK_PROBABILITY_POLICY,
    PROBABILITY_EPSILON,
)
from nfl_ats.pick_probability_fit import (
    FIT_FEATURES,
    FIT_ITERATIONS,
    FIT_RIDGE,
    MODEL_PROBABILITY_SOURCE,
    _design,
    _fit_logit,
    _natural_coefficients,
    _predict,
    _standardisers,
)

SEASONS = tuple(range(2020, 2026))
METRICS = ("accuracy", "log_loss", "brier")
ARMS = {
    "loso_base": "base_home_probability",
    "served_evaluation": "served_evaluation_home_probability",
    "raw_model": "model_probability",
    "market_even": "market_even_home_probability",
}
BIN_EDGES = (0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 1.0)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260929


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def frame_digest(frame: pd.DataFrame) -> str:
    return digest(frame.to_json(orient="split", index=False, double_precision=15).encode())


def read_input(path: Path, hashes: dict[str, str]) -> bytes:
    payload = path.read_bytes()
    hashes[path.as_posix()] = digest(payload)
    return payload


def metric_rows(target: np.ndarray, probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    return np.column_stack(
        (
            (probability >= 0.5) == target,
            -(target * np.log(p) + (1.0 - target) * np.log1p(-p)),
            (probability - target) ** 2,
        )
    )


def metric_summary(rows: np.ndarray, directional: bool = True) -> dict:
    result = {name: float(rows[:, index].mean()) for index, name in enumerate(METRICS)}
    result["games"] = len(rows)
    result["wins"] = int(rows[:, 0].sum()) if directional else None
    result["losses"] = len(rows) - result["wins"] if directional else None
    if not directional:
        result["accuracy"] = None
    return result


def interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def diagnostics(frame: pd.DataFrame, training_rows: np.ndarray) -> dict:
    target = frame["home_covered"].to_numpy(dtype=float)
    counts = np.array([frame["season"].eq(season).sum() for season in SEASONS])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(SEASONS), size=(BOOTSTRAP_DRAWS, len(SEASONS)))
    weights = np.eye(len(SEASONS), dtype=int)[draws].sum(axis=1)
    denominator = weights @ counts
    overall, by_season, reliability, bootstrap = {}, {}, {}, {}
    for arm, column in ARMS.items():
        rows = metric_rows(target, frame[column].to_numpy(dtype=float))
        directional = arm != "market_even"
        overall[arm] = metric_summary(rows, directional)
        totals = np.stack([rows[frame["season"].eq(season)].sum(axis=0) for season in SEASONS])
        bootstrap[arm] = (weights @ totals) / denominator[:, None]
        overall[arm]["interval_95"] = {
            name: interval(bootstrap[arm][:, index]) if directional or name != "accuracy" else None
            for index, name in enumerate(METRICS)
        }
        by_season[arm] = {
            str(season): metric_summary(rows[frame["season"].eq(season)], directional)
            for season in SEASONS
        }
        reliability[arm] = []
        for lower, upper in pairwise(BIN_EDGES):
            mask = frame[column].ge(lower) & (
                frame[column].lt(upper) if upper < 1.0 else frame[column].le(upper)
            )
            reliability[arm].append(
                {
                    "lower": lower,
                    "upper": upper,
                    "games": int(mask.sum()),
                    "mean_probability": float(frame.loc[mask, column].mean())
                    if mask.any()
                    else None,
                    "home_cover_rate": float(frame.loc[mask, "home_covered"].mean())
                    if mask.any()
                    else None,
                }
            )
    contrasts = {}
    for arm in ("served_evaluation", "raw_model", "market_even"):
        contrasts[arm] = {}
        for index, name in enumerate(METRICS):
            if arm == "market_even" and name == "accuracy":
                contrasts[arm][name] = None
                continue
            sign = 1.0 if name == "accuracy" else -1.0
            gains = sign * (bootstrap["loso_base"][:, index] - bootstrap[arm][:, index])
            gains[np.abs(gains) < 1e-12] = 0.0
            contrasts[arm][name] = {
                "improvement": sign * (overall["loso_base"][name] - overall[arm][name]),
                "interval_95": interval(gains),
                "probability_positive": float(np.mean(gains > 0) + 0.5 * np.mean(gains == 0)),
            }
    training = metric_summary(training_rows)
    return {
        "overall": overall,
        "by_season": by_season,
        "reliability": reliability,
        "pooled_fold_training": training,
        "held_out_minus_training": {
            name: overall["loso_base"][name] - training[name] for name in METRICS
        },
        "paired_improvement_over": contrasts,
    }


def build(source_directory: Path | None, output_directory: Path | None) -> tuple[Path, dict]:
    hashes: dict[str, str] = {}
    pointer = json.loads(read_input(Path("artifacts/active_pick_probability.json"), hashes))
    active = json.loads(read_input(Path("artifacts/active_ats_model.json"), hashes))
    if source_directory is None:
        source_directory = Path("artifacts") / pointer["artifact"]
    require(
        source_directory.resolve().is_relative_to(Path("artifacts/pick_probability").resolve()),
        "Source must be a served pick_probability artifact in this repository",
    )
    source_metadata = json.loads(read_input(source_directory / "metadata.json", hashes))
    coefficients = json.loads(read_input(source_directory / "coefficients.json", hashes))
    source = pd.read_parquet(io.BytesIO(read_input(source_directory / "per_game.parquet", hashes)))
    for path in (
        Path("scripts/build_loso_base.py"),
        Path("src/nfl_ats/pick_probability_fit.py"),
        Path("src/nfl_ats/pick_probability.py"),
    ):
        read_input(path, hashes)
    require(source_metadata["command"] == "nfl-ats fit-pick-probability", "Wrong source command")
    require(source_metadata["features"] == list(FIT_FEATURES), "Served features changed")
    require(float(source_metadata["ridge"]) == FIT_RIDGE, "Served ridge changed")
    require(source_metadata["policy"] == PICK_PROBABILITY_POLICY, "Wrong probability policy")
    require(
        source_metadata["base_probability_policy"] == BASE_PROBABILITY_POLICY, "Wrong base policy"
    )
    require(
        source_metadata["active_model_id"] == active["model_id"], "Source differs from active model"
    )
    require(
        source_metadata["counted_flag_columns"] == list(COUNTED_FLAG_COLUMNS), "Composition changed"
    )
    require(
        coefficients["market_move_feature_version"]
        == source_metadata["market_move_feature_version"],
        "Move versions differ",
    )
    required = {
        "game_id",
        "season",
        "week",
        "game_type",
        "home_covered",
        "margin_vs_open",
        "tue_open_home_spread",
        "model_probability",
        MODEL_PROBABILITY_SOURCE,
        "out_of_season_home_probability",
        *FIT_FEATURES,
        *COUNTED_FLAG_COLUMNS,
    }
    require(
        required.issubset(source.columns),
        f"Missing source columns: {sorted(required - set(source.columns))}",
    )
    require(not source["game_id"].duplicated().any(), "Duplicate source game IDs")
    require(source["game_id"].notna().all(), "Missing source game IDs")
    eligible = source["season"].isin(SEASONS) & source["game_type"].eq("REG")
    frame = source.loc[eligible].copy().reset_index(drop=True)
    require(set(frame["season"].unique()) == set(SEASONS), "All six seasons must be present")
    require(frame["home_covered"].isin([0.0, 1.0]).all(), "Invalid home-cover target")
    require(
        frame["margin_vs_open"].notna().all() and frame["margin_vs_open"].ne(0).all(),
        "Ungraded or push row in fit population",
    )
    require(
        np.array_equal(
            frame["home_covered"].to_numpy(), frame["margin_vs_open"].gt(0).astype(float).to_numpy()
        ),
        "Target differs from opener grade",
    )
    numeric = [
        *FIT_FEATURES,
        "model_probability",
        "out_of_season_home_probability",
        "tue_open_home_spread",
    ]
    require(np.isfinite(frame[numeric].to_numpy(dtype=float)).all(), "Nonfinite fit inputs")
    p = frame[MODEL_PROBABILITY_SOURCE].clip(PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON)
    require(
        np.allclose(frame["model_probability"], p, rtol=0, atol=1e-12),
        "Model input differs from opener",
    )
    require(
        np.allclose(frame["model_logit"], np.log(p / (1.0 - p)), rtol=0, atol=1e-12),
        "Model logit differs from served transform",
    )
    require(
        np.array_equal(frame[FLAG_SUM_COLUMN], frame[list(COUNTED_FLAG_COLUMNS)].sum(axis=1)),
        "Composition sum differs from served flags",
    )
    require(frame[MOVE_AVAILABLE_COLUMN].isin([0.0, 1.0]).all(), "Invalid move availability")
    require(
        frame.loc[frame[MOVE_AVAILABLE_COLUMN].eq(0), MOVE_COLUMN].eq(0).all(),
        "Missing moves must use served zero fill",
    )
    require(
        frame["base_probability_policy"].eq(BASE_PROBABILITY_POLICY).all(),
        "Non-discrete base probability",
    )
    opener_directory = Path("artifacts") / source_metadata["opener_evaluation"]
    require(
        opener_directory.resolve().is_relative_to(Path("artifacts/opener_evaluation").resolve()),
        "Invalid opener artifact",
    )
    opener = pd.read_parquet(io.BytesIO(read_input(opener_directory / "per_game.parquet", hashes)))
    opener_metadata = json.loads(read_input(opener_directory / "metadata.json", hashes))
    opener = opener.loc[opener["season"].isin(SEASONS)]
    opener_nonpush = opener.loc[opener["margin_vs_open"].notna() & opener["margin_vs_open"].ne(0)]
    require(
        set(frame["game_id"]) == set(opener_nonpush["game_id"]), "Incomplete opener fit population"
    )
    alignment = frame.merge(
        opener_nonpush[
            [
                "game_id",
                "season",
                "tue_open_home_spread",
                "margin_vs_open",
                MODEL_PROBABILITY_SOURCE,
            ]
        ],
        on=["game_id", "season"],
        validate="one_to_one",
        suffixes=("", "_opener"),
    )
    require(len(alignment) == len(frame), "Opener season mismatch")
    for column in ("tue_open_home_spread", "margin_vs_open", MODEL_PROBABILITY_SOURCE):
        require(
            np.allclose(alignment[column], alignment[f"{column}_opener"], rtol=0, atol=1e-12),
            f"Opener mismatch: {column}",
        )
    input_columns = ["game_id", "season", "home_covered", *FIT_FEATURES]
    frame["base_home_probability"] = np.nan
    frame["base_training_games"] = 0
    frame["base_training_seasons"] = ""
    folds, training = {}, []
    for held in SEASONS:
        train = frame.loc[frame["season"].ne(held)]
        test = frame.loc[frame["season"].eq(held)]
        train_seasons = sorted(int(value) for value in train["season"].unique())
        require(set(train_seasons) == set(SEASONS) - {held}, "Training season set is incorrect")
        require(
            not train["season"].isin(test["season"].unique()).any(),
            "Training shares a prediction season",
        )
        require(
            not test["season"].isin(train_seasons).any(),
            "A prediction row shares its training season",
        )
        require(
            not train["game_id"].isin(test["game_id"]).any(), "Training shares prediction games"
        )
        means, stds = _standardisers(train)
        beta = _fit_logit(
            _design(train, means, stds), train["home_covered"].to_numpy(dtype=float), FIT_RIDGE
        )
        natural = _natural_coefficients(beta, means, stds)
        frame.loc[test.index, "base_home_probability"] = _predict(test, beta, means, stds)
        frame.loc[test.index, "base_training_games"] = len(train)
        frame.loc[test.index, "base_training_seasons"] = ",".join(map(str, train_seasons))
        training.append(
            metric_rows(
                train["home_covered"].to_numpy(dtype=float), _predict(train, beta, means, stds)
            )
        )
        folds[str(held)] = {
            "held_out_season": held,
            "training_seasons": train_seasons,
            "training_games": len(train),
            "prediction_games": len(test),
            "same_season_training_rows": 0,
            "training_inputs_sha256": frame_digest(train[input_columns]),
            "prediction_inputs_sha256": frame_digest(test[input_columns]),
            "training_game_ids_sha256": digest(
                "\n".join(sorted(train["game_id"].astype(str))).encode()
            ),
            "coefficients": natural,
            "standardized_coefficients": beta.tolist(),
            "training_means": means,
            "training_stds": stds,
        }
    require(
        frame["base_home_probability"].between(0, 1, inclusive="neither").all(),
        "Incomplete or invalid LOSO predictions",
    )
    frame["served_evaluation_home_probability"] = frame["out_of_season_home_probability"]
    probability_error = float(
        np.max(np.abs(frame["base_home_probability"] - frame["served_evaluation_home_probability"]))
    )
    coefficient_error = max(
        abs(value - source_metadata["fold_coefficients"][held][name])
        for held, fold in folds.items()
        for name, value in fold["coefficients"].items()
    )
    require(probability_error < 1e-10, "Independent refit differs from served LOSO evaluation")
    require(
        coefficient_error < 1e-10, "Independent fold coefficients differ from served evaluation"
    )
    frame["held_out_season"] = frame["season"].astype(int)
    frame["same_season_training_rows"] = 0
    frame["base_pick_home"] = frame["base_home_probability"].ge(0.5)
    frame["base_pick_probability"] = np.maximum(
        frame["base_home_probability"], 1 - frame["base_home_probability"]
    )
    frame["base_correct"] = frame["base_pick_home"].eq(frame["home_covered"])
    frame["market_even_home_probability"] = 0.5
    selected = [
        "game_id",
        "season",
        "week",
        "game_type",
        "gameday",
        "home_team",
        "away_team",
        "tue_open_home_spread",
        "margin_vs_open",
        "home_covered",
        "base_probability_policy",
        MODEL_PROBABILITY_SOURCE,
        "model_probability",
        *FIT_FEATURES,
        *[column for column in frame.columns if column.startswith("flag_")],
        "base_home_probability",
        "base_pick_home",
        "base_pick_probability",
        "base_correct",
        "held_out_season",
        "base_training_games",
        "base_training_seasons",
        "same_season_training_rows",
        "served_evaluation_home_probability",
        "market_even_home_probability",
    ]
    predictions = frame[selected].sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    metrics = diagnostics(predictions, np.concatenate(training))
    stamp = datetime.now(UTC)
    if output_directory is None:
        output_directory = Path("artifacts/loso_base") / stamp.strftime("%Y%m%dT%H%M%S%fZ")
    require(
        output_directory.resolve().is_relative_to(Path("artifacts/loso_base").resolve()),
        "Output must stay under artifacts/loso_base",
    )
    require(not output_directory.exists(), "Output directory already exists")
    for path, expected_hash in hashes.items():
        require(
            digest(Path(path).read_bytes()) == expected_hash, f"Input changed during build: {path}"
        )
    metadata = {
        "schema_version": 1,
        "created_at_utc": stamp.isoformat(),
        "source_artifact": source_directory.as_posix(),
        "active_model_id": source_metadata["active_model_id"],
        "base_probability_policy": BASE_PROBABILITY_POLICY,
        "market_move_feature_version": source_metadata["market_move_feature_version"],
        "target": "Home covers historical opener, conditional on no push",
        "seasons": list(SEASONS),
        "features": list(FIT_FEATURES),
        "ridge": FIT_RIDGE,
        "iterations": FIT_ITERATIONS,
        "input_hashes": hashes,
        "fit_inputs_sha256": frame_digest(frame[input_columns]),
        "folds": folds,
        "coverage": {
            "source_fit_rows": len(source),
            "eligible_rows": len(predictions),
            "source_rows_outside_population": int((~eligible).sum()),
            "opener_rows_2020_2025": len(opener),
            "opener_pushes_excluded": int(opener["margin_vs_open"].eq(0).sum()),
            "opener_ungraded_excluded": int(opener["margin_vs_open"].isna().sum()),
            "missing_eligible_predictions": 0,
        },
        "verification": {
            "same_season_training_rows": 0,
            "served_probability_max_abs_error": probability_error,
            "served_coefficient_max_abs_error": coefficient_error,
        },
        "upstream_validation_limitations": opener_metadata.get("validation_limitations"),
        "limitations": [
            (
                "Season exclusion is enforced for the four-term calibration only; "
                "upstream model inputs are reused, not retrained."
            ),
            (
                "Underlying model fits may include earlier games in the prediction "
                "season; this is not an end-to-end season-held-out model."
            ),
            (
                "Retrospective LOSO uses later seasons to predict earlier seasons; it "
                "is not a prospective rolling evaluation."
            ),
            (
                "Features were selected using these seasons; this is not an untouched "
                "outer test or promotion evidence."
            ),
            (
                "Only eligible non-push opener games are predicted; pushes have no "
                "binary cover target and are excluded by the served fitter."
            ),
            (
                "Six-season bootstrap intervals hold fitted predictions fixed and do "
                "not include refit uncertainty."
            ),
            (
                "Served market-move inputs retain their Sunday pre-kick availability; "
                "opener grading is not an opener-time information claim."
            ),
        ],
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "unit": "whole season",
            "interval": "95% percentile",
        },
        "look_count": 162,
        "look_family": (
            "One baseline construction: six fits and 156 prespecified diagnostic cells, "
            "including null market accuracy cells"
        ),
        "metrics": metrics,
    }
    output_directory.mkdir(parents=True, exist_ok=False)
    prediction_path = output_directory / "predictions.parquet"
    predictions.to_parquet(prediction_path, index=False)
    metadata["predictions_sha256"] = digest(prediction_path.read_bytes())
    metadata["prediction_columns"] = list(predictions.columns)
    reread = pd.read_parquet(prediction_path)
    require(reread.equals(predictions), "Parquet round-trip changed predictions")
    (output_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return output_directory, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a shared four-term season-held-out research base "
            "without changing serving artifacts."
        )
    )
    parser.add_argument(
        "--source-artifact",
        type=Path,
        help="Frozen artifacts/pick_probability directory; defaults to active pointer",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="New directory under artifacts/loso_base; defaults to a UTC timestamp",
    )
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        directory, metadata = build(args.source_artifact, args.output_dir)
    print(
        json.dumps(
            {
                "artifact": directory.as_posix(),
                "coverage": metadata["coverage"],
                "verification": metadata["verification"],
                "overall": metadata["metrics"]["overall"],
                "pooled_fold_training": metadata["metrics"]["pooled_fold_training"],
                "held_out_minus_training": metadata["metrics"]["held_out_minus_training"],
                "paired_improvement_over": metadata["metrics"]["paired_improvement_over"],
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
