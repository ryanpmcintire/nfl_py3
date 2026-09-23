from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from nfl_ats.pick_probability import (
    PROBABILITY_EPSILON,
    STRENGTH_BAND_QUANTILES,
    STRENGTH_ROUNDING_PLACES,
)
from nfl_ats.pick_probability_fit import (
    FIT_FEATURES,
    FIT_RIDGE,
    MODEL_PROBABILITY_SOURCE,
    PickProbabilitySourceError,
    _confidence_bands,
    _design,
    _fit_logit,
    _predict,
    _standardisers,
    _strength_bands,
    build_fit_population,
)

REQUIRED_COLUMNS = ("game_id", "season", "home_covered", *FIT_FEATURES)
BOOTSTRAP_REPS = 20000
RNG_SEED = 20260923
WORDS = ("slight", "lean", "strong")


def _loso_out_of_season(population: pd.DataFrame, ridge: float) -> pd.DataFrame:
    seasons = sorted(population["season"].unique())
    out = pd.Series(np.nan, index=population.index, dtype=float)
    for held in seasons:
        train = population.loc[population["season"].ne(held)]
        test = population.loc[population["season"].eq(held)]
        if train.empty or test.empty:
            continue
        means, stds = _standardisers(train)
        beta = _fit_logit(
            _design(train, means, stds), train["home_covered"].astype(float).to_numpy(), ridge
        )
        out.loc[test.index] = _predict(test, beta, means, stds)
    result = population.copy()
    result["out_of_season_home_probability"] = out
    return result.loc[result["out_of_season_home_probability"].notna()].copy()


def _shown_and_correct(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    probability = pd.to_numeric(frame[column], errors="coerce")
    picked_home = probability.ge(0.5)
    correct = picked_home.astype(float).eq(frame["home_covered"]).astype(float)
    shown = probability.where(picked_home, 1.0 - probability)
    out = frame[["game_id", "season"]].copy()
    out["shown"] = shown
    out["correct"] = correct
    return out


def _nested_strength_words(scored: pd.DataFrame) -> pd.DataFrame:
    base = _shown_and_correct(scored, "out_of_season_home_probability")
    seasons = sorted(base["season"].unique())
    rows = []
    for held in seasons:
        pool = base.loc[base["season"].ne(held), "shown"].to_numpy(dtype=float)
        test = base.loc[base["season"].eq(held)].copy()
        if len(pool) == 0 or test.empty:
            continue
        lower, upper = np.quantile(pool, STRENGTH_BAND_QUANTILES)
        lower_r = round(float(lower), STRENGTH_ROUNDING_PLACES)
        upper_r = round(float(upper), STRENGTH_ROUNDING_PLACES)
        rounded = test["shown"].round(STRENGTH_ROUNDING_PLACES)
        test["word"] = np.select(
            [rounded.ge(upper_r), rounded.ge(lower_r)], ["strong", "lean"], default="slight"
        )
        test["fold_lean_edge"] = lower_r
        test["fold_strong_edge"] = upper_r
        rows.append(test)
    return pd.concat(rows, ignore_index=True) if rows else base.iloc[0:0].assign(word=[])


def _fixed_strength_words(scored: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    frame = _shown_and_correct(scored, "out_of_season_home_probability")
    rounded = frame["shown"].round(STRENGTH_ROUNDING_PLACES)
    frame["word"] = np.select(
        [rounded.ge(upper), rounded.ge(lower)], ["strong", "lean"], default="slight"
    )
    return frame


def _season_block_bootstrap(word_frame: pd.DataFrame, reps: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    seasons = sorted(word_frame["season"].unique())
    n = len(seasons)
    by_season = {s: word_frame.loc[word_frame["season"].eq(s)] for s in seasons}
    reps_rates: dict[str, list[float]] = {w: [] for w in WORDS}
    reps_diff = []
    for _ in range(reps):
        draw = rng.choice(seasons, size=n, replace=True)
        pooled = pd.concat([by_season[s] for s in draw], ignore_index=True)
        rates = {}
        for w in WORDS:
            sub = pooled.loc[pooled["word"].eq(w)]
            rate = float(sub["correct"].mean()) if len(sub) else np.nan
            rates[w] = rate
            reps_rates[w].append(rate)
        reps_diff.append(rates["strong"] - rates["slight"])
    summary: dict[str, object] = {}
    for w in WORDS:
        arr = np.array(reps_rates[w], dtype=float)
        arr = arr[np.isfinite(arr)]
        summary[w] = {
            "mean": float(np.mean(arr)) if len(arr) else None,
            "ci_low": float(np.quantile(arr, 0.025)) if len(arr) else None,
            "ci_high": float(np.quantile(arr, 0.975)) if len(arr) else None,
        }
    diff = np.array(reps_diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    summary["strong_minus_slight"] = {
        "mean": float(np.mean(diff)) if len(diff) else None,
        "ci_low": float(np.quantile(diff, 0.025)) if len(diff) else None,
        "ci_high": float(np.quantile(diff, 0.975)) if len(diff) else None,
        "probability_positive": float(np.mean(diff > 0)) if len(diff) else None,
    }
    return summary


def _band_table(word_frame: pd.DataFrame) -> dict[str, object]:
    table: dict[str, object] = {}
    for w in WORDS:
        sub = word_frame.loc[word_frame["word"].eq(w)]
        table[w] = {
            "games": len(sub),
            "cover_rate": float(sub["correct"].mean()) if len(sub) else None,
        }
    return table


def _iteration_trace(
    population: pd.DataFrame, ridge: float, max_iter: int = 50
) -> list[dict[str, object]]:
    means, stds = _standardisers(population)
    x = _design(population, means, stds)
    y = population["home_covered"].astype(float).to_numpy()
    beta = np.zeros(x.shape[1])
    trace = []
    for i in range(max_iter):
        z = np.clip(x @ beta, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.clip(p * (1.0 - p), 1e-6, None)
        gradient = x.T @ (y - p) - ridge * beta
        hessian = (x.T * w) @ x + ridge * np.eye(x.shape[1])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        beta = beta + step
        step_norm = float(np.linalg.norm(step))
        beta_norm = float(np.linalg.norm(beta)) or 1.0
        trace.append(
            {"iteration": i + 1, "step_norm": step_norm, "relative_step": step_norm / beta_norm}
        )
    return trace


def _epsilon_binding_dataset_b(extended: pd.DataFrame) -> dict[str, object]:
    boundary_logit = float(np.log((1.0 - PROBABILITY_EPSILON) / PROBABILITY_EPSILON))
    logit = pd.to_numeric(extended["model_logit"], errors="coerce")
    at_boundary = int(np.isclose(logit.abs(), boundary_logit, atol=1e-9).sum())
    return {
        "games": len(logit),
        "min_logit": float(logit.min()),
        "max_logit": float(logit.max()),
        "boundary_logit": boundary_logit,
        "at_boundary": at_boundary,
        "binds": bool(at_boundary),
    }


def _confidence_edge_report(scored: pd.DataFrame) -> dict[str, object]:
    bands = _confidence_bands(scored, "out_of_season_home_probability")
    rows = [band.to_dict() for band in bands]
    rates = [row["accuracy"] for row in rows if row["accuracy"] is not None]
    monotone = all(a <= b for a, b in pairwise(rates))
    return {"bands": rows, "monotone_nondecreasing": bool(monotone)}


def _run_dataset(name: str, population: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    scored = _loso_out_of_season(population, FIT_RIDGE)
    scored.to_parquet(output_dir / f"{name}_loso_predictions.parquet", index=False)

    served_bands = _strength_bands(scored, "out_of_season_home_probability")
    served_lookup = {band.word: band for band in served_bands}
    served_frame = _fixed_strength_words(
        scored, served_lookup["lean"].minimum, served_lookup["strong"].minimum
    )
    served_bootstrap = _season_block_bootstrap(served_frame, BOOTSTRAP_REPS, RNG_SEED)
    served_table = _band_table(served_frame)

    nested_frame = _nested_strength_words(scored)
    nested_bootstrap = _season_block_bootstrap(nested_frame, BOOTSTRAP_REPS, RNG_SEED + 1)
    nested_table = _band_table(nested_frame)
    fold_edges = (
        nested_frame[["season", "fold_lean_edge", "fold_strong_edge"]]
        .drop_duplicates()
        .sort_values("season")
        .to_dict("records")
    )

    confidence = _confidence_edge_report(scored)
    iterations = _iteration_trace(population, FIT_RIDGE)
    converged_iteration = next(
        (row["iteration"] for row in iterations if row["relative_step"] < 1e-6), None
    )

    return {
        "games": len(scored),
        "seasons": sorted(int(v) for v in scored["season"].unique()),
        "served_strength_bands": [band.to_dict() for band in served_bands],
        "served_band_table_out_of_fold": served_table,
        "served_band_bootstrap": served_bootstrap,
        "nested_band_table": nested_table,
        "nested_band_bootstrap": nested_bootstrap,
        "nested_fold_edges": fold_edges,
        "confidence_band_edges_report": confidence,
        "fit_iterations_trace_last5": iterations[-5:],
        "fit_iterations_converged_at_relative_step_1e-6": converged_iteration,
    }


def _load_extended_population(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"extended population is missing columns: {', '.join(missing)}")
    return frame[list(REQUIRED_COLUMNS)].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--extended-population",
        default="artifacts/extended_fit_population/20260923T205910Z/population.parquet",
    )
    parser.add_argument("--output-root", default="artifacts/confidence_band_derivation")
    args = parser.parse_args()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.output_root) / stamp
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "predeclared_band_meaning": (
            "slight/lean/strong split the picked-side held-out probability into equal-count "
            "thirds (STRENGTH_BAND_QUANTILES=(1/3,2/3)); success means held-out cover rate is "
            "monotone strong>lean>slight and adjacent bands are statistically distinguishable "
            "(season-block bootstrap interval on strong-minus-slight excludes zero). Edges are "
            "derived nested leave-one-season-out: for each held-out test season, the tertile "
            "edges come only from the pooled held-out probabilities of the OTHER seasons, never "
            "from the test season itself."
        ),
        "served_ridge": FIT_RIDGE,
        "confidence_band_edges_note": (
            "CONFIDENCE_BAND_EDGES is read-only: measured this session that it is never used to "
            "build the card/board Slight/Lean/Strong text (only STRENGTH_BAND_QUANTILES via "
            "displayed_confidence.StrengthBands feeds board_content.py and publishing.py); it "
            "only feeds the fit-pick-probability CLI JSON diagnostic (cli_commands/prediction.py)."
        ),
        "datasets": {},
    }

    artifacts_root = Path(args.artifacts_root)
    data_root = Path(args.data_root)
    try:
        served_population, _provenance = build_fit_population(artifacts_root, data_root)
        report["datasets"]["served_population_2020_2025"] = _run_dataset(
            "served", served_population[list(REQUIRED_COLUMNS)].copy(), output_dir
        )
        raw = pd.to_numeric(served_population[MODEL_PROBABILITY_SOURCE], errors="coerce")
        below = int(raw.lt(PROBABILITY_EPSILON).sum())
        above = int(raw.gt(1.0 - PROBABILITY_EPSILON).sum())
        report["datasets"]["served_population_2020_2025"]["epsilon_binding"] = {
            "games": len(raw),
            "min": float(raw.min()),
            "max": float(raw.max()),
            "below_epsilon": below,
            "above_one_minus_epsilon": above,
            "binds": bool(below or above),
        }
    except PickProbabilitySourceError as error:
        report["datasets"]["served_population_2020_2025"] = {"error": str(error)}

    extended_path = Path(args.extended_population)
    if extended_path.is_file():
        extended_population = _load_extended_population(extended_path)
        report["datasets"]["extended_population_2011_2025"] = _run_dataset(
            "extended", extended_population, output_dir
        )
        report["datasets"]["extended_population_2011_2025"]["epsilon_binding"] = (
            _epsilon_binding_dataset_b(extended_population)
        )

    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(str(output_dir))


if __name__ == "__main__":
    main()
