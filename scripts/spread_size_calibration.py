from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

from nfl_ats.evidence_conventions import probability_positive_from_draws
from nfl_ats.mass_preserving_lattice import (
    BAND_HALF_WIDTH,
    prior_pool,
    walk_forward_reads,
)
from nfl_ats.spread_regime import BUCKETS, spread_bucket

PROBABILITY_FLOOR = 1e-6
PRIOR_WEIGHT_GAMES = 100.0
SLOPE_FLOOR = 0.05
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260817
REPLAY_TOLERANCE = 1e-9
KEY_LINE_ATOMS = (3.0, 7.0)
FIT_SEASONS = (2020, 2021, 2022)
SCORE_SEASONS = (2023, 2024, 2025)


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        np.asarray(probability, dtype=float), PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR
    )
    return np.log(clipped / (1.0 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float))), dtype=float)


FIT_MODES = ("shift_slope", "shift_only", "slope_only")


def _fit_logistic(x: np.ndarray, y: np.ndarray, *, mode: str) -> tuple[float, float]:
    if x.size == 0:
        return 0.0, 1.0
    free_intercept = mode in {"shift_slope", "shift_only"}
    free_slope = mode in {"shift_slope", "slope_only"}

    def unpack(params: np.ndarray) -> tuple[float, float]:
        intercept = float(params[0]) if free_intercept else 0.0
        slope = float(params[1]) if free_slope else 1.0
        return intercept, slope

    def negative_log_likelihood(params: np.ndarray) -> float:
        intercept, slope = unpack(params)
        linear = intercept + slope * x
        return float(np.sum(np.logaddexp(0.0, linear) - y * linear))

    def gradient(params: np.ndarray) -> np.ndarray:
        intercept, slope = unpack(params)
        residual = _sigmoid(intercept + slope * x) - y
        return np.array(
            [
                float(residual.sum()) if free_intercept else 0.0,
                float((residual * x).sum()) if free_slope else 0.0,
            ]
        )

    fitted = optimize.minimize(
        negative_log_likelihood, np.array([0.0, 1.0]), jac=gradient, method="BFGS"
    )
    intercept, slope = unpack(fitted.x)
    if not np.isfinite(intercept) or not np.isfinite(slope):
        return 0.0, 1.0
    return intercept, slope


def fit_bucket_recalibration(
    frame: pd.DataFrame, probability_column: str, *, mode: str
) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for bucket in BUCKETS:
        rows = frame.loc[frame["bucket"].eq(bucket) & frame["home_cover"].notna()]
        x = _logit(rows[probability_column].to_numpy(dtype=float))
        y = rows["home_cover"].to_numpy(dtype=float)
        raw_intercept, raw_slope = _fit_logistic(x, y, mode=mode)
        games = float(len(rows))
        weight = games / (games + PRIOR_WEIGHT_GAMES) if games > 0 else 0.0
        table[bucket] = {
            "games": games,
            "weight": weight,
            "raw_intercept": raw_intercept,
            "raw_slope": raw_slope,
            "intercept": weight * raw_intercept,
            "slope": weight * raw_slope + (1.0 - weight) * 1.0,
        }
    return table


def apply_bucket_recalibration(
    frame: pd.DataFrame, probability_column: str, table: dict[str, dict[str, float]]
) -> np.ndarray:
    x = _logit(frame[probability_column].to_numpy(dtype=float))
    intercept = frame["bucket"].map(lambda b: table[b]["intercept"]).to_numpy(dtype=float)
    slope = frame["bucket"].map(lambda b: table[b]["slope"]).to_numpy(dtype=float)
    return _sigmoid(intercept + slope * x)


def load_active_config(artifacts_root: Path) -> dict[str, Any]:
    payload = json.loads((artifacts_root / "active_ats_model.json").read_text(encoding="utf-8"))
    return dict(payload)


def matching_opener_evaluation(artifacts_root: Path, active: dict[str, Any]) -> Path:
    candidates = sorted(
        (artifacts_root / "opener_evaluation").glob("*/metadata.json"), reverse=True
    )
    for path in candidates:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        same_model = metadata.get("active_model_id") == active.get("model_id")
        same_features = metadata.get("feature_table_sha256") == active.get("feature_table_sha256")
        if same_model and same_features:
            return path.parent
    raise FileNotFoundError("No opener evaluation matches the active model")


def replay_served_points(per_game: pd.DataFrame) -> pd.DataFrame:
    frame = per_game.copy()
    frame["median_residual"] = np.nan
    frame["residual_std"] = np.nan
    frame["replay_gap"] = np.nan
    for _, group in frame.groupby(["season", "week"], sort=True):
        residual = group["residual_at_open_served"].to_numpy(dtype=float)
        probability = group["home_cover_probability_at_open"].to_numpy(dtype=float)
        z = stats.norm.ppf(np.clip(probability, PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR))
        slope, intercept = np.polyfit(residual, z, 1)
        scale = 1.0 / float(slope)
        location = float(intercept) * scale
        replayed = stats.norm.sf(-residual, loc=location, scale=scale)
        frame.loc[group.index, "median_residual"] = location
        frame.loc[group.index, "residual_std"] = scale
        frame.loc[group.index, "replay_gap"] = np.abs(replayed - probability)
    return frame


def build_scored_frame(evaluation: Path, features: pd.DataFrame, cache: Path) -> pd.DataFrame:
    if cache.is_file():
        return pd.read_parquet(cache)
    per_game = pd.read_parquet(evaluation / "per_game.parquet")
    replayed = replay_served_points(per_game)
    worst = float(replayed["replay_gap"].max())
    if worst > REPLAY_TOLERANCE:
        raise ValueError(f"Served-probability replay failed: worst gap {worst:.3e}")
    schedule = features[["game_id", "gameday"]].drop_duplicates("game_id").copy()
    schedule["game_id"] = schedule["game_id"].astype(str)
    replayed["game_id"] = replayed["game_id"].astype(str)
    replayed = replayed.merge(schedule, on="game_id", how="left")
    replayed["gameday"] = pd.to_datetime(replayed["gameday"])
    replayed["line"] = pd.to_numeric(replayed["tue_open_home_spread"], errors="raise")
    replayed["point"] = (
        replayed["line"] + replayed["residual_at_open_served"] + replayed["median_residual"]
    )
    opener_lines = pd.Series(
        replayed["line"].to_numpy(dtype=float), index=replayed["game_id"].astype(str)
    )
    pool = prior_pool(features, opener_lines)
    reads = walk_forward_reads(
        pool,
        replayed[["game_id", "season", "week", "gameday", "line", "point"]],
        BAND_HALF_WIDTH,
    )
    reads["game_id"] = reads["game_id"].astype(str)
    merged = replayed.merge(
        reads[
            [
                "game_id",
                "cover",
                "push",
                "loss",
                "home_cover_probability",
                "theta",
                "band",
                "band_games",
                "key_mass_3",
            ]
        ],
        on="game_id",
        how="inner",
    )
    merged = merged.rename(
        columns={
            "cover": "cover_lattice",
            "push": "push_lattice",
            "loss": "loss_lattice",
            "home_cover_probability": "p_lattice",
        }
    )
    merged["p_s3"] = merged["home_cover_probability_at_open"].astype(float)
    on_atom = np.isin(np.abs(merged["line"].to_numpy(dtype=float)), np.asarray(KEY_LINE_ATOMS))
    merged["on_key_atom"] = on_atom
    merged["p_kl1b"] = np.where(on_atom, merged["p_lattice"], merged["p_s3"])
    merged["bucket"] = spread_bucket(merged["line"])
    merged["settle_margin"] = merged["margin_vs_open"].astype(float)
    merged["home_cover"] = np.where(
        merged["settle_margin"].eq(0.0), np.nan, (merged["settle_margin"] > 0.0).astype(float)
    )
    merged.to_parquet(cache, index=False)
    return merged


def pick_accuracy(probability: np.ndarray, home_cover: np.ndarray) -> np.ndarray:
    pick_home = np.asarray(probability, dtype=float) >= 0.5
    cover = np.asarray(home_cover, dtype=float) > 0.5
    return np.where(pick_home, cover, ~cover).astype(float)


def brier(probability: np.ndarray, home_cover: np.ndarray) -> np.ndarray:
    return (np.asarray(probability, dtype=float) - np.asarray(home_cover, dtype=float)) ** 2


def block_arrays(frame: pd.DataFrame) -> list[np.ndarray]:
    return [
        np.asarray(index) for index in frame.groupby(["season", "week"], sort=True).indices.values()
    ]


def paired_bootstrap(
    frame: pd.DataFrame, baseline: np.ndarray, candidate: np.ndarray, *, metric: str
) -> dict[str, float]:
    cover = frame["home_cover"].to_numpy(dtype=float)
    if metric == "accuracy_points":
        base_value = pick_accuracy(baseline, cover) * 100.0
        cand_value = pick_accuracy(candidate, cover) * 100.0
    else:
        base_value = -brier(baseline, cover)
        cand_value = -brier(candidate, cover)
    blocks = block_arrays(frame.reset_index(drop=True))
    counts = np.array([len(block) for block in blocks], dtype=float)
    base_sums = np.array([base_value[block].sum() for block in blocks], dtype=float)
    cand_sums = np.array([cand_value[block].sum() for block in blocks], dtype=float)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    draws_index = generator.integers(0, len(blocks), size=(BOOTSTRAP_SAMPLES, len(blocks)))
    sampled_counts = counts[draws_index].sum(axis=1)
    sampled_base = base_sums[draws_index].sum(axis=1)
    sampled_candidate = cand_sums[draws_index].sum(axis=1)
    draws = (sampled_candidate - sampled_base) / sampled_counts
    estimate = float(cand_value.mean() - base_value.mean())
    return {
        "delta": estimate,
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "standard_error": float(np.std(draws, ddof=1)),
        "probability_positive": float(probability_positive_from_draws(draws)),
        "n": len(frame),
        "blocks": len(blocks),
        "baseline": float(base_value.mean() if metric == "accuracy_points" else -base_value.mean()),
        "candidate": float(
            cand_value.mean() if metric == "accuracy_points" else -cand_value.mean()
        ),
        "flips": int(np.count_nonzero((baseline >= 0.5) != (candidate >= 0.5))),
    }


def reliability_rows(frame: pd.DataFrame, probability_column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    probability = frame[probability_column].to_numpy(dtype=float)
    cover = frame["home_cover"].to_numpy(dtype=float)
    for bucket in (*BUCKETS, "ALL"):
        mask = (
            np.ones(len(frame), dtype=bool)
            if bucket == "ALL"
            else frame["bucket"].eq(bucket).to_numpy()
        )
        if not mask.any():
            continue
        stated = np.maximum(probability[mask], 1.0 - probability[mask])
        rows.append(
            {
                "bucket": bucket,
                "n": int(mask.sum()),
                "stated_confidence": float(stated.mean()),
                "accuracy": float(pick_accuracy(probability[mask], cover[mask]).mean()),
                "stated_home_cover": float(probability[mask].mean()),
                "realised_home_cover": float(cover[mask].mean()),
                "brier": float(brier(probability[mask], cover[mask]).mean()),
                "home_picks": int(np.count_nonzero(probability[mask] >= 0.5)),
            }
        )
    return rows


def week_one_side_changes(forecast: Path, tables: dict[str, dict[str, float]]) -> dict[str, Any]:
    payload = json.loads((forecast / "discrete_push_read.json").read_text(encoding="utf-8"))
    games = payload.get("games") or []
    if not games:
        return {"games": 0, "changed": 0, "rows": []}
    frame = pd.DataFrame(
        {
            "game_id": [str(row["game_id"]) for row in games],
            "line": [float(row["spread_line"]) for row in games],
            "p_served": [float(row["home_cover_probability"]) for row in games],
            "p_lattice": [
                float(row["served"]["cover"]) + 0.5 * float(row["served"]["push"]) for row in games
            ],
        }
    )
    frame["bucket"] = spread_bucket(frame["line"])
    frame["p_candidate"] = apply_bucket_recalibration(frame, "p_lattice", tables)
    frame["changed"] = (frame["p_served"] >= 0.5) != (frame["p_candidate"] >= 0.5)
    return {
        "forecast": str(forecast),
        "games": len(frame),
        "changed": int(frame["changed"].sum()),
        "rows": frame.to_dict("records"),
    }


def arm_definitions() -> dict[str, dict[str, Any]]:
    return {
        "A0_lattice_raw": {"source": "p_lattice", "mode": None},
        "A1_lattice_shift_slope": {"source": "p_lattice", "mode": "shift_slope"},
        "A2_lattice_shift_only": {"source": "p_lattice", "mode": "shift_only"},
        "A3_smooth_shift_slope": {"source": "p_s3", "mode": "shift_slope"},
        "A4_smooth_shift_only": {"source": "p_s3", "mode": "shift_only"},
        "A5_smooth_slope_only": {"source": "p_s3", "mode": "slope_only"},
        "A6_lattice_slope_only": {"source": "p_lattice", "mode": "slope_only"},
        "A7_smooth_slope_only_floored": {
            "source": "p_s3",
            "mode": "slope_only",
            "slope_floor": SLOPE_FLOOR,
        },
        "A8_played_slope_only_floored": {
            "source": "p_kl1b",
            "mode": "slope_only",
            "slope_floor": SLOPE_FLOOR,
        },
    }


def run(output: Path, evaluation: Path | None) -> dict[str, Any]:
    artifacts_root = Path("artifacts")
    active = load_active_config(artifacts_root)
    features_path = Path("data/processed/game_features_weak_stack.parquet")
    digest = hashlib.sha256(features_path.read_bytes()).hexdigest()
    if digest != active.get("feature_table_sha256"):
        raise ValueError("Feature table does not match the active model")
    evaluation_dir = evaluation or matching_opener_evaluation(artifacts_root, active)
    features = pd.read_parquet(features_path)
    output.mkdir(parents=True, exist_ok=True)
    scored = build_scored_frame(evaluation_dir, features, output / "scored.parquet")
    graded = scored.loc[scored["home_cover"].notna()].reset_index(drop=True)
    fit = graded.loc[graded["season"].isin(FIT_SEASONS)].reset_index(drop=True)
    score = graded.loc[graded["season"].isin(SCORE_SEASONS)].reset_index(drop=True)

    tables: dict[str, Any] = {}
    probabilities: dict[str, np.ndarray] = {}
    for name, spec in arm_definitions().items():
        source = str(spec["source"])
        if spec["mode"] is None:
            probabilities[name] = score[source].to_numpy(dtype=float)
            continue
        table = fit_bucket_recalibration(fit, source, mode=str(spec["mode"]))
        floor = spec.get("slope_floor")
        if floor is not None:
            table = {
                bucket: {**values, "slope": max(float(values["slope"]), float(floor))}
                for bucket, values in table.items()
            }
        tables[name] = table
        probabilities[name] = apply_bucket_recalibration(score, source, table)

    baselines = {
        "S3": score["p_s3"].to_numpy(dtype=float),
        "KL1b": score["p_kl1b"].to_numpy(dtype=float),
        "A0": probabilities["A0_lattice_raw"],
    }

    cells: dict[str, Any] = {}
    for baseline_name, baseline in baselines.items():
        for arm_name, candidate in probabilities.items():
            if baseline_name == "A0" and arm_name not in {
                "A1_lattice_shift_slope",
                "A2_lattice_shift_only",
            }:
                continue
            for metric in ("accuracy_points", "brier_improvement"):
                key = f"{arm_name}_vs_{baseline_name}_overall_{metric}"
                cells[key] = paired_bootstrap(
                    score,
                    baseline,
                    candidate,
                    metric="accuracy_points" if metric == "accuracy_points" else "brier",
                )
            for bucket in BUCKETS:
                mask = score["bucket"].eq(bucket).to_numpy()
                if mask.sum() < 2:
                    continue
                subset = score.loc[mask].reset_index(drop=True)
                for metric in ("accuracy_points", "brier_improvement"):
                    key = f"{arm_name}_vs_{baseline_name}_bucket_{bucket}_{metric}"
                    cells[key] = paired_bootstrap(
                        subset,
                        baseline[mask],
                        candidate[mask],
                        metric="accuracy_points" if metric == "accuracy_points" else "brier",
                    )

    reliability = {
        "incumbent_S3": reliability_rows(score, "p_s3"),
        "incumbent_KL1b": reliability_rows(score.assign(p_kl1b=baselines["KL1b"]), "p_kl1b"),
    }
    for arm_name, candidate in probabilities.items():
        reliability[arm_name] = reliability_rows(score.assign(**{arm_name: candidate}), arm_name)

    archive_reliability = {
        "raw_no_offset": reliability_rows(graded, "home_cover_probability_at_open_raw"),
        "served_S3": reliability_rows(graded, "p_s3"),
        "served_KL1b": reliability_rows(graded, "p_kl1b"),
        "lattice_every_game": reliability_rows(graded, "p_lattice"),
    }
    fit_reliability = {
        "raw_no_offset": reliability_rows(fit, "home_cover_probability_at_open_raw"),
        "served_S3": reliability_rows(fit, "p_s3"),
    }
    full_archive_tables = {
        "A1_lattice_shift_slope": fit_bucket_recalibration(graded, "p_lattice", mode="shift_slope"),
        "A2_lattice_shift_only": fit_bucket_recalibration(graded, "p_lattice", mode="shift_only"),
        "A5_smooth_slope_only": fit_bucket_recalibration(graded, "p_s3", mode="slope_only"),
    }
    forecasts = sorted((artifacts_root / "margin_predictions").glob("*/discrete_push_read.json"))
    week_one = (
        week_one_side_changes(forecasts[-1].parent, full_archive_tables["A1_lattice_shift_slope"])
        if forecasts
        else {"games": 0, "changed": 0, "rows": []}
    )

    payload = {
        "archive_reliability": archive_reliability,
        "fit_reliability": fit_reliability,
        "full_archive_tables": full_archive_tables,
        "week_one": week_one,
        "active_model_id": active.get("model_id"),
        "feature_table_sha256": active.get("feature_table_sha256"),
        "evaluation": str(evaluation_dir),
        "fit_seasons": list(FIT_SEASONS),
        "score_seasons": list(SCORE_SEASONS),
        "fit_games": len(fit),
        "score_games": len(score),
        "graded_games": len(graded),
        "archive_games": len(scored),
        "prior_weight_games": PRIOR_WEIGHT_GAMES,
        "bootstrap": {"samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED},
        "recalibration_tables": tables,
        "reliability": reliability,
        "cells": cells,
    }
    (output / "cells.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MOD-18 C2: spread-conditional recalibration on the discrete margin lattice"
    )
    parser.add_argument("--output", type=Path, required=True, help="directory for the run outputs")
    parser.add_argument("--evaluation", type=Path, default=None, help="opener evaluation directory")
    args = parser.parse_args()
    payload = run(args.output, args.evaluation)
    skip = {"cells", "reliability", "archive_reliability", "fit_reliability", "week_one"}
    summary = {k: v for k, v in payload.items() if k not in skip}
    summary["week_one_changed"] = payload["week_one"]["changed"]
    summary["week_one_games"] = payload["week_one"]["games"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
