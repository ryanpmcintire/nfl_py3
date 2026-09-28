from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nfl_ats.clv import week_blocked_bootstrap
from nfl_ats.io import atomic_json
from nfl_ats.pick_probability import BASE_PROBABILITY_POLICY
from nfl_ats.pick_probability_fit import _fit_logit
from nfl_ats.provenance import sha256_file


def fit_matrix(x: pd.DataFrame, y: pd.Series, ridge: float) -> dict[str, Any]:
    values = x.to_numpy(dtype=float)
    means = values.mean(axis=0)
    stds = values.std(axis=0)
    stds[stds == 0] = 1
    design = np.column_stack([np.ones(len(x)), (values - means) / stds])
    beta = _fit_logit(design, y.to_numpy(dtype=float), ridge)
    slopes = beta[1:] / stds
    natural = dict(zip(x.columns, slopes.tolist(), strict=True))
    natural["intercept"] = float(beta[0] - np.dot(slopes, means))
    return {
        "columns": list(x.columns),
        "means": means.tolist(),
        "stds": stds.tolist(),
        "beta": beta.tolist(),
        "coefficients": natural,
    }


def predict_matrix(model: dict[str, Any], x: pd.DataFrame) -> np.ndarray:
    values = x[model["columns"]].to_numpy(dtype=float)
    design = np.column_stack(
        [np.ones(len(x)), (values - np.array(model["means"])) / np.array(model["stds"])]
    )
    z = np.clip(design @ np.array(model["beta"]), -35, 35)
    return np.clip(1 / (1 + np.exp(-z)), 1e-9, 1 - 1e-9)


def features(frame: pd.DataFrame, flags: tuple[str, ...], market: bool) -> pd.DataFrame:
    result = frame[["model_logit"]].copy()
    if flags:
        result["selected_flag_sum"] = frame[list(flags)].sum(axis=1)
    if market:
        result["market_move_toward_home"] = frame.market_move_toward_home
        result["market_move_available"] = frame.market_move_available
    return result


def logit_frame(probability: np.ndarray, index: pd.Index) -> pd.DataFrame:
    p = np.clip(probability, 1e-9, 1 - 1e-9)
    return pd.DataFrame({"base_logit": np.log(p / (1 - p))}, index=index)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    hit = (p >= 0.5) == y
    return {
        "games": len(y),
        "correct": int(hit.sum()),
        "accuracy": float(hit.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p))),
    }


def fit_calibrated(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    evaluation: pd.DataFrame,
    flags: tuple[str, ...],
    market: bool,
    ridge: float,
    calibration_ridge: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    base = fit_matrix(features(train, flags, market), train.home_covered, ridge)
    calibration_p = predict_matrix(base, features(calibration, flags, market))
    calibrator = fit_matrix(
        logit_frame(calibration_p, calibration.index),
        calibration.home_covered,
        calibration_ridge,
    )
    evaluation_p = predict_matrix(base, features(evaluation, flags, market))
    final = predict_matrix(calibrator, logit_frame(evaluation_p, evaluation.index))
    combined = {
        key: float(value * calibrator["coefficients"]["base_logit"])
        for key, value in base["coefficients"].items()
    }
    combined["intercept"] += calibrator["coefficients"]["intercept"]
    return final, {
        "flags": list(flags),
        "market": market,
        "ridge": ridge,
        "base": base,
        "calibrator": calibrator,
        "combined_coefficients": combined,
    }


def reliability(y: np.ndarray, p: np.ndarray, edges: list[float]) -> list[dict[str, Any]]:
    confidence = np.maximum(p, 1 - p)
    hit = (p >= 0.5) == y
    result = []
    for low, high in itertools.pairwise(edges):
        mask = (confidence >= low) & ((confidence < high) | ((high == 1) & (confidence == 1)))
        result.append(
            {
                "low": low,
                "high": high,
                "games": int(mask.sum()),
                "predicted": float(confidence[mask].mean()) if mask.any() else None,
                "observed": float(hit[mask].mean()) if mask.any() else None,
            }
        )
    return result


def exact_disagreement(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a_side, b_side = a >= 0.5, b >= 0.5
    disagreed = a_side != b_side
    n = int(disagreed.sum())
    a_wins = int((a_side[disagreed] == y[disagreed]).sum())
    tail = min(a_wins, n - a_wins)
    null_p = min(1.0, 2 * sum(math.comb(n, k) for k in range(tail + 1)) * 2.0 ** (-n))
    return {
        "games": n,
        "candidate_correct": a_wins,
        "baseline_correct": n - a_wins,
        "exact_two_sided_null_p": null_p,
    }


def run(protocol_path: Path, destination: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if destination.exists():
        raise ValueError("Refusing to replace a historical evaluation")
    for key in ("input", "opener_input"):
        if sha256_file(Path(protocol[key])) != protocol[key + "_sha256"]:
            raise ValueError(f"Declared source changed: {key}")
    frame = pd.read_parquet(protocol["input"])
    frame = frame.loc[frame.game_type.eq(protocol["game_type"])].copy()
    flags = tuple(protocol["flags"])
    wanted = ["model_logit", "model_probability", "home_covered", "margin_vs_open", *flags]
    wanted += ["market_move_toward_home", "market_move_available"]
    if frame.game_id.duplicated().any() or not np.isfinite(frame[wanted].to_numpy()).all():
        raise ValueError("Historical input is duplicated or incomplete")
    if not frame.base_probability_policy.eq(BASE_PROBABILITY_POLICY).all():
        raise ValueError("Historical input must use the served discrete probability")
    if (
        not frame.home_covered.eq(frame.margin_vs_open.gt(0)).all()
        or frame.margin_vs_open.eq(0).any()
    ):
        raise ValueError("Historical targets must be decisive opener grades")
    mass = frame[
        [
            "home_cover_probability_excluding_push_at_open",
            "push_probability_at_open",
            "home_loss_probability_at_open",
        ]
    ]
    if (
        not np.isfinite(mass.to_numpy()).all()
        or (mass < 0).any(axis=None)
        or not np.allclose(mass.sum(axis=1), 1)
        or not np.allclose(mass.iloc[:, 0] / (1 - mass.iloc[:, 1]), frame.model_probability)
    ):
        raise ValueError("Historical probabilities do not match discrete nonpush mass")
    if not np.allclose(frame[list(flags)].sum(axis=1), frame.composition_flag_sum):
        raise ValueError("Declared flags do not reproduce the recorded combination")
    candidates = []
    for mask in range(2 ** len(flags)):
        subset = tuple(flag for position, flag in enumerate(flags) if mask & (1 << position))
        for market in (False, True):
            candidates.append((f"flags_{mask:03d}_market_{int(market)}", subset, market))
    if len(candidates) != protocol["candidates_per_selection_fold"]:
        raise ValueError("Candidate count differs from declaration")
    rows, folds, selection_scores, validation_scores = [], [], [], []
    calibration_ridge = float(protocol["calibration_ridge"])
    for outer in protocol["outer_test_seasons"]:
        order = frame.season * 100 + frame.week
        previous = (outer - 1) * 100
        training = frame.loc[order.le(previous + 6)].copy()
        selection = frame.loc[order.between(previous + 7, previous + 10)].copy()
        validation = frame.loc[order.between(previous + 11, previous + 14)].copy()
        calibration = frame.loc[order.between(previous + 15, previous + 18)].copy()
        test = frame.loc[frame.season.eq(outer)].copy()
        windows = [training, selection, validation, calibration, test]
        if any(part.empty for part in windows):
            raise ValueError(f"Incomplete chronological windows for {outer}")
        for earlier, later in itertools.pairwise(windows):
            if (earlier.season * 100 + earlier.week).max() >= (
                later.season * 100 + later.week
            ).min():
                raise ValueError("Historical windows overlap")
        ranked = []
        for candidate_id, subset, market in candidates:
            model = fit_matrix(features(training, subset, market), training.home_covered, 0.001)
            p = predict_matrix(model, features(selection, subset, market))
            loss = float(np.mean((p - selection.home_covered.to_numpy()) ** 2))
            row = {
                "outer_season": outer,
                "candidate_id": candidate_id,
                "flags": "|".join(subset),
                "market": market,
                "signals": len(subset) + 2 * int(market),
                "selection_brier": loss,
            }
            selection_scores.append(row)
            ranked.append((loss, row["signals"], market, candidate_id, subset))
        chosen = min(ranked)
        subset, market = chosen[4], chosen[2]
        validation_train = frame.loc[order.le(previous + 10)]
        ridge_scores = []
        for ridge in protocol["ridge_grid"]:
            model = fit_matrix(
                features(validation_train, subset, market), validation_train.home_covered, ridge
            )
            p = predict_matrix(model, features(validation, subset, market))
            loss = float(np.mean((p - validation.home_covered.to_numpy()) ** 2))
            ridge_scores.append((loss, -ridge))
            validation_scores.append(
                {"outer_season": outer, "ridge": ridge, "validation_brier": loss}
            )
        ridge = -min(ridge_scores)[1]
        final_train = frame.loc[order.le(previous + 14)]
        evaluation = frame.loc[frame.season.le(outer)].copy()
        development_mask = evaluation.season.lt(outer).to_numpy()
        test_mask = evaluation.season.eq(outer).to_numpy()
        fitted = {}
        probabilities = {}
        for arm, arm_flags, arm_market, arm_ridge in (
            ("nested_selected", subset, market, ridge),
            ("fixed_combination", flags, True, 0.001),
        ):
            probabilities[arm], fitted[arm] = fit_calibrated(
                final_train,
                calibration,
                evaluation,
                arm_flags,
                arm_market,
                arm_ridge,
                calibration_ridge,
            )
        raw_calibration = fit_matrix(
            calibration[["model_logit"]], calibration.home_covered, calibration_ridge
        )
        probabilities["raw_calibrated"] = predict_matrix(
            raw_calibration, evaluation[["model_logit"]]
        )
        fitted["raw_calibrated"] = raw_calibration
        y = evaluation.home_covered.to_numpy(dtype=float)
        gaps = {}
        for arm, p in probabilities.items():
            development_metrics = metrics(y[development_mask], p[development_mask])
            outer_metrics = metrics(y[test_mask], p[test_mask])
            gaps[arm] = {
                "development_apparent": development_metrics,
                "outer_test": outer_metrics,
                "apparent_minus_outer_accuracy_points": 100
                * (development_metrics["accuracy"] - outer_metrics["accuracy"]),
                "outer_minus_apparent_brier": outer_metrics["brier"] - development_metrics["brier"],
            }
            test[arm] = p[test_mask]
        test["raw"] = test.model_probability
        test["neutral_market"] = 0.5
        test["fixed_forward"] = test.chronological_home_probability
        if not np.isfinite(test[protocol["arms"]].to_numpy()).all():
            raise ValueError("An outer arm is missing predictions")
        kept = [
            "game_id",
            "season",
            "week",
            "gameday",
            "home_team",
            "away_team",
            "tue_open_home_spread",
            "margin_vs_open",
            "home_covered",
            "push_probability_at_open",
            *protocol["arms"],
        ]
        rows.append(test[kept])
        folds.append(
            {
                "outer_season": outer,
                "windows": {
                    name: {
                        "seasons": sorted(part.season.unique().tolist()),
                        "first_season_week": int((part.season * 100 + part.week).min()),
                        "last_season_week": int((part.season * 100 + part.week).max()),
                        "games": len(part),
                    }
                    for name, part in zip(
                        ["training", "selection", "validation", "calibration", "outer"],
                        windows,
                        strict=True,
                    )
                },
                "selected_candidate": chosen[3],
                "selected_flags": list(subset),
                "selected_market": market,
                "selected_ridge": ridge,
                "selection_brier": chosen[0],
                "validation_brier": min(ridge_scores)[0],
                "fitted_models": fitted,
                "gaps": gaps,
            }
        )
        print(f"Completed declared outer season {outer}", flush=True)
    predictions = pd.concat(rows, ignore_index=True)
    if predictions.game_id.duplicated().any():
        raise ValueError("Outer games are repeated")
    y = predictions.home_covered.to_numpy(dtype=float)
    pooled = {}
    seasons = {}
    for arm in protocol["arms"]:
        p = predictions[arm].to_numpy(dtype=float)
        pooled[arm] = {
            **metrics(y, p),
            "reliability": reliability(y, p, protocol["reliability_bands"]),
        }
        seasons[arm] = {
            str(season): metrics(
                part.home_covered.to_numpy(dtype=float), part[arm].to_numpy(dtype=float)
            )
            for season, part in predictions.groupby("season")
        }
    contrasts = predictions[["season", "week"]].copy()
    disagreements = {}
    for candidate, baseline in protocol["contrasts"]:
        name = f"{candidate}_vs_{baseline}"
        a = predictions[candidate].to_numpy(dtype=float)
        b = predictions[baseline].to_numpy(dtype=float)
        contrasts[name + "__brier_improvement"] = (b - y) ** 2 - (a - y) ** 2
        contrasts[name + "__log_loss_improvement"] = (
            -y * np.log(b) - (1 - y) * np.log1p(-b) + y * np.log(a) + (1 - y) * np.log1p(-a)
        )
        contrasts[name + "__accuracy_points"] = 100 * (
            ((a >= 0.5) == y).astype(float) - ((b >= 0.5) == y).astype(float)
        )
        disagreements[name] = exact_disagreement(y, a, b)
    metric_names = list(contrasts.columns[2:])
    values = contrasts[metric_names].to_numpy(dtype=float)

    def mean_metrics(sample: pd.DataFrame) -> dict[str, float]:
        return {name: float(sample[name].mean()) for name in metric_names}

    def draw_factory(valid: pd.DataFrame) -> Any:
        def draw(positions: np.ndarray) -> dict[str, float]:
            return dict(zip(metric_names, values[positions].mean(axis=0).tolist(), strict=True))

        return draw

    bootstrap = protocol["bootstrap"]
    intervals = week_blocked_bootstrap(
        contrasts,
        mean_metrics,
        samples=bootstrap["samples"],
        confidence=bootstrap["confidence"],
        seed=bootstrap["seed"],
        metric_columns=metric_names,
        metric_draw_factory=draw_factory,
    )
    opener = pd.read_parquet(protocol["opener_input"])
    opener_outer = opener.loc[opener.season.isin(protocol["outer_test_seasons"])]
    expected = set(opener_outer.loc[opener_outer.margin_vs_open.ne(0), "game_id"])
    missing = sorted(expected - set(predictions.game_id))
    if missing:
        raise ValueError(f"Missing decisive opener games: {missing}")
    destination.mkdir(parents=True)
    (destination / "replay_source.py").write_bytes(Path(__file__).read_bytes())
    predictions.to_parquet(destination / "predictions.parquet", index=False)
    predictions.to_csv(destination / "predictions.csv", index=False)
    pd.DataFrame(selection_scores).to_csv(destination / "selection_scores.csv", index=False)
    pd.DataFrame(validation_scores).to_csv(destination / "validation_scores.csv", index=False)
    intervals.to_csv(destination / "paired_intervals.csv", index=False)
    report = {
        "study_id": protocol["study_id"],
        "protocol_path": protocol_path.as_posix(),
        "protocol_sha256": sha256_file(protocol_path),
        "script_sha256": sha256_file(Path(__file__)),
        "fitter_sha256": sha256_file(Path("src/nfl_ats/pick_probability_fit.py")),
        "input_sha256": protocol["input_sha256"],
        "games": len(predictions),
        "weeks": len(predictions[["season", "week"]].drop_duplicates()),
        "pushes_excluded": int(opener_outer.margin_vs_open.eq(0).sum()),
        "missing_decisive_games": missing,
        "arms": pooled,
        "seasons": seasons,
        "folds": folds,
        "paired_intervals": intervals.to_dict(orient="records"),
        "disagreements": disagreements,
        "look_accounting": protocol["look_accounting"],
        "limitations": protocol["limitations"],
        "automatic_serving_change": False,
    }
    atomic_json(report, destination / "report.json")
    cells = []
    for row in intervals.to_dict(orient="records"):
        name, units = row["metric"].split("__")
        cells.append(
            {
                "name": f"{protocol['study_id']}_{name}_{units}",
                "description": f"Declared chronological selection audit: {name}, {units}",
                "effect": row["estimate"],
                "effect_units": units,
                "interval_low": row["lower"],
                "interval_high": row["upper"],
                "probability_positive": row["probability_positive"],
            }
        )
    batch = {
        "defaults": {
            "league": "nfl",
            "season_start": min(protocol["outer_test_seasons"]),
            "season_end": max(protocol["outer_test_seasons"]),
            "family": protocol["study_id"],
            "category": "modeling",
            "classification": "unresolved_below_power",
            "classification_evidence": "Conditional historical audit with prior feature-design "
            "selection outside the replay; no admissible mechanism refutation or power bound.",
            "source": (destination / "report.json").as_posix(),
            "sample_games": len(predictions),
            "sample_blocks": report["weeks"],
            "plain_summary": "Feature subsets and ridge were selected before each outer season. "
            "Earlier research choices remain outside this replay.",
            "notes": "Primary contrast is nested-selected versus raw Brier. All 15 declared "
            "paired metrics are retained; no post-result subgroup or serving change.",
        },
        "cells": cells,
    }
    shared = batch.pop("defaults")
    batch["family"] = shared["family"]
    batch["cells"] = [{**shared, **cell} for cell in batch["cells"]]
    atomic_json(batch, destination / "weak_signals_batch.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the declared chronological feature-selection audit "
        "without changing serving"
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("registry/studies/combined_vs_raw_historical_nested_20260928.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/independent_historical_validation/20260928_nested"),
    )
    args = parser.parse_args()
    report = run(args.protocol, args.output)
    summary = {
        "games": report["games"],
        "weeks": report["weeks"],
        "pushes_excluded": report["pushes_excluded"],
        "arms": {
            arm: {key: value for key, value in result.items() if key != "reliability"}
            for arm, result in report["arms"].items()
        },
        "paired_intervals": report["paired_intervals"],
        "disagreements": report["disagreements"],
        "report": (args.output / "report.json").as_posix(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
