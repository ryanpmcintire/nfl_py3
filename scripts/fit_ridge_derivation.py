from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.pick_probability_fit import (
    FIT_FEATURES,
    FIT_RIDGE,
    PickProbabilitySourceError,
    _confidence_bands,
    _design,
    _fit_logit,
    _predict,
    _standardisers,
    build_fit_population,
)

RIDGE_GRID = tuple(float(value) for value in np.logspace(-4, 2, 13))
REQUIRED_COLUMNS = ("game_id", "season", "home_covered", *FIT_FEATURES)


def _pooled_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    clipped = np.clip(p, 1e-9, 1.0 - 1e-9)
    return float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped)))


def _pooled_brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _decisive_record(y: np.ndarray, p: np.ndarray) -> str:
    picked_home = p >= 0.5
    correct = picked_home.astype(float) == y
    wins = int(correct.sum())
    return f"{wins}-{len(correct) - wins}"


def _select_ridge(train: pd.DataFrame, grid: tuple[float, ...]) -> tuple[float, dict[float, float]]:
    inner_seasons = sorted(train["season"].unique())
    scores: dict[float, float] = {}
    for ridge in grid:
        preds = []
        actuals = []
        for held in inner_seasons:
            inner_train = train.loc[train["season"].ne(held)]
            inner_test = train.loc[train["season"].eq(held)]
            if inner_train.empty or inner_test.empty or inner_train["season"].nunique() < 2:
                continue
            means, stds = _standardisers(inner_train)
            beta = _fit_logit(
                _design(inner_train, means, stds),
                inner_train["home_covered"].astype(float).to_numpy(),
                ridge,
            )
            preds.append(_predict(inner_test, beta, means, stds))
            actuals.append(inner_test["home_covered"].astype(float).to_numpy())
        if not preds:
            continue
        p = np.concatenate(preds)
        y = np.concatenate(actuals)
        scores[ridge] = _pooled_log_loss(y, p)
    if not scores:
        return grid[len(grid) // 2], scores
    best = min(scores, key=scores.get)
    return float(best), scores


def _run_nested(
    population: pd.DataFrame, grid: tuple[float, ...], served_ridge: float
) -> tuple[pd.DataFrame, dict[str, float], dict[str, dict[float, float]]]:
    seasons = sorted(population["season"].unique())
    chosen: dict[str, float] = {}
    inner_scores: dict[str, dict[float, float]] = {}
    outer_rows = []
    for held in seasons:
        train = population.loc[population["season"].ne(held)]
        test = population.loc[population["season"].eq(held)]
        if train.empty or test.empty or train["season"].nunique() < 2:
            continue
        best_ridge, scores = _select_ridge(train, grid)
        chosen[str(held)] = best_ridge
        inner_scores[str(held)] = scores
        means, stds = _standardisers(train)
        target = train["home_covered"].astype(float).to_numpy()
        beta_nested = _fit_logit(_design(train, means, stds), target, best_ridge)
        beta_served = _fit_logit(_design(train, means, stds), target, served_ridge)
        rows = test[["game_id", "season", "home_covered"]].copy()
        rows["nested_probability"] = _predict(test, beta_nested, means, stds)
        rows["served_probability"] = _predict(test, beta_served, means, stds)
        rows["chosen_ridge"] = best_ridge
        outer_rows.append(rows)
    if not outer_rows:
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS)), chosen, inner_scores
    return pd.concat(outer_rows, ignore_index=True), chosen, inner_scores


def _reliability_table(frame: pd.DataFrame, column: str) -> list[dict[str, object]]:
    bands = _confidence_bands(frame, column)
    return [band.to_dict() for band in bands]


def _summarise(frame: pd.DataFrame, column: str) -> dict[str, object]:
    if frame.empty:
        return {"games": 0}
    y = frame["home_covered"].astype(float).to_numpy()
    p = frame[column].astype(float).to_numpy()
    return {
        "games": len(frame),
        "decisive_record": _decisive_record(y, p),
        "accuracy": float(np.mean((p >= 0.5) == y)),
        "brier": _pooled_brier(y, p),
        "log_loss": _pooled_log_loss(y, p),
        "reliability": _reliability_table(frame, column),
    }


def _load_extended_population(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"extended population is missing columns: {', '.join(missing)}")
    return frame[list(REQUIRED_COLUMNS)].copy()


def _load_served_population(artifacts_root: Path, data_root: Path) -> pd.DataFrame | None:
    try:
        population, _provenance = build_fit_population(artifacts_root, data_root)
    except PickProbabilitySourceError as error:
        print(f"served population unavailable: {error}")
        return None
    return population[list(REQUIRED_COLUMNS)].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--extended-population",
        default="artifacts/extended_fit_population/20260923T205910Z/population.parquet",
    )
    parser.add_argument("--output-root", default="artifacts/fit_ridge_derivation")
    args = parser.parse_args()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_root) / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "ridge_grid": list(RIDGE_GRID),
        "served_ridge": FIT_RIDGE,
        "datasets": {},
    }

    served_population = _load_served_population(Path(args.artifacts_root), Path(args.data_root))
    if served_population is not None and not served_population.empty:
        rows, chosen, inner_scores = _run_nested(served_population, RIDGE_GRID, FIT_RIDGE)
        rows.to_parquet(output_dir / "served_population_predictions.parquet", index=False)
        report["datasets"]["served_population_2020_2025"] = {
            "games": len(served_population),
            "seasons": sorted(int(v) for v in served_population["season"].unique()),
            "chosen_ridge_per_fold": chosen,
            "inner_grid_scores": inner_scores,
            "nested": _summarise(rows, "nested_probability"),
            "served_constant": _summarise(rows, "served_probability"),
        }

    extended_path = Path(args.extended_population)
    if extended_path.is_file():
        extended_population = _load_extended_population(extended_path)
        rows, chosen, inner_scores = _run_nested(extended_population, RIDGE_GRID, FIT_RIDGE)
        rows.to_parquet(output_dir / "extended_population_predictions.parquet", index=False)
        report["datasets"]["extended_population_2011_2025"] = {
            "games": len(extended_population),
            "seasons": sorted(int(v) for v in extended_population["season"].unique()),
            "chosen_ridge_per_fold": chosen,
            "inner_grid_scores": inner_scores,
            "nested": _summarise(rows, "nested_probability"),
            "served_constant": _summarise(rows, "served_probability"),
        }

    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(str(output_dir))


if __name__ == "__main__":
    main()
