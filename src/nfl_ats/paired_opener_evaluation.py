from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.linear_model import LogisticRegression

from nfl_ats.data import DataContractError
from nfl_ats.evidence_conventions import probability_positive_from_draws

PAIR_KEYS = ("season", "week", "game_id")
REQUIRED_COLUMNS = (*PAIR_KEYS, "line", "outcome", "raw_probability", "provenance")
BASELINE_NAMES = ("market", "model_only")
_PROBABILITY_EPSILON = 1e-6


@dataclass(frozen=True)
class PairedOpenerEvaluation:
    predictions: pd.DataFrame
    fold_diagnostics: pd.DataFrame
    metrics: pd.DataFrame
    reliability: pd.DataFrame
    bootstrap_summary: pd.DataFrame
    bootstrap_draws: pd.DataFrame
    metadata: dict[str, object]


def _contains_boolean(series: pd.Series) -> bool:
    return any(isinstance(value, (bool, np.bool_)) for value in series.dropna().tolist())


def _integer_series(frame: pd.DataFrame, column: str, arm_name: str) -> pd.Series:
    if _contains_boolean(frame[column]):
        raise DataContractError(f"{arm_name} {column} must contain finite integers")
    numeric = pd.to_numeric(frame[column], errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise DataContractError(f"{arm_name} {column} must contain finite integers")
    values = numeric.to_numpy(dtype=float)
    if not np.equal(values, np.floor(values)).all():
        raise DataContractError(f"{arm_name} {column} must contain finite integers")
    return pd.Series(values.astype(np.int64), index=frame.index)


def _probability_series(frame: pd.DataFrame, column: str, arm_name: str) -> pd.Series:
    if column not in frame.columns:
        raise DataContractError(f"{arm_name} is missing probability column {column!r}")
    if _contains_boolean(frame[column]):
        raise DataContractError(f"{arm_name} {column} must contain finite probabilities in [0, 1]")
    numeric = pd.to_numeric(frame[column], errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise DataContractError(f"{arm_name} {column} must contain finite probabilities in [0, 1]")
    return pd.Series(values, index=frame.index)


def _validate_arm(
    frame: pd.DataFrame,
    *,
    arm_name: str,
    completed_seasons: frozenset[int],
    baseline_columns: Mapping[str, tuple[str, str]],
    baseline_index: int,
) -> pd.DataFrame:
    duplicate_columns = frame.columns[frame.columns.duplicated()].tolist()
    if duplicate_columns:
        raise DataContractError(f"{arm_name} contains duplicate column labels: {duplicate_columns}")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataContractError(f"{arm_name} is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise DataContractError(f"{arm_name} has no rows")
    result = frame.copy()
    result["season"] = _integer_series(result, "season", arm_name)
    result["week"] = _integer_series(result, "week", arm_name)
    result["game_id"] = result["game_id"].astype("string")
    if result["game_id"].isna().any() or result["game_id"].str.strip().eq("").any():
        raise DataContractError(f"{arm_name} game_id must contain nonempty values")
    if result.duplicated(list(PAIR_KEYS)).any():
        examples = result.loc[result.duplicated(list(PAIR_KEYS), keep=False), list(PAIR_KEYS)]
        raise DataContractError(
            f"{arm_name} contains duplicate game keys: {examples.head(3).to_dict('records')}"
        )
    if _contains_boolean(result["line"]):
        raise DataContractError(f"{arm_name} line must contain finite values")
    line = pd.to_numeric(result["line"], errors="coerce")
    if line.isna().any() or not np.isfinite(line.to_numpy(dtype=float)).all():
        raise DataContractError(f"{arm_name} line must contain finite values")
    result["line"] = line.astype(float)
    raw_outcome = result["outcome"]
    push = raw_outcome.isna()
    if _contains_boolean(raw_outcome):
        raise DataContractError(f"{arm_name} outcome must be 0, 1, or missing for a push")
    outcome = pd.to_numeric(raw_outcome, errors="coerce")
    non_push = ~push
    non_push_values = outcome.loc[non_push].to_numpy(dtype=float)
    if (
        outcome.loc[non_push].isna().any()
        or not np.isfinite(non_push_values).all()
        or not outcome.loc[non_push].isin([0.0, 1.0]).all()
    ):
        raise DataContractError(f"{arm_name} outcome must be 0, 1, or missing for a push")
    outcome.loc[push] = np.nan
    result["outcome"] = outcome.astype(float)
    result["raw_probability"] = _probability_series(result, "raw_probability", arm_name)
    provenance = result["provenance"].astype("string")
    if provenance.isna().any() or provenance.str.strip().eq("").any():
        raise DataContractError(f"{arm_name} provenance must contain nonempty values")
    result["provenance"] = provenance
    undeclared = sorted(set(result["season"].astype(int)) - completed_seasons)
    if undeclared:
        raise DataContractError(f"{arm_name} contains seasons not declared completed: {undeclared}")
    selected = list(REQUIRED_COLUMNS)
    for baseline_name in BASELINE_NAMES:
        if baseline_name not in baseline_columns:
            raise DataContractError(f"baseline_probability_columns must include {baseline_name!r}")
        columns = baseline_columns[baseline_name]
        if len(columns) != 2:
            raise DataContractError(
                f"baseline {baseline_name!r} must provide one column for each arm"
            )
        column = columns[baseline_index]
        result[column] = _probability_series(result, column, arm_name)
        if column not in selected:
            selected.append(column)
    return result.loc[:, selected].sort_values(list(PAIR_KEYS)).reset_index(drop=True)


def _normalize_completed_seasons(seasons: Sequence[int]) -> frozenset[int]:
    message = "completed_seasons must contain integer seasons"
    try:
        values = iter(seasons)
    except TypeError as error:
        raise DataContractError(message) from error
    normalized: set[int] = set()
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            raise DataContractError(message)
        try:
            season = int(value)
            if season != value:
                raise DataContractError(message)
        except (TypeError, ValueError, OverflowError) as error:
            raise DataContractError(message) from error
        normalized.add(season)
    if not normalized:
        raise DataContractError("completed_seasons must not be empty")
    return frozenset(normalized)


def _merge_arms(
    arm_a: pd.DataFrame,
    arm_b: pd.DataFrame,
    *,
    arm_a_name: str,
    arm_b_name: str,
    baseline_columns: Mapping[str, tuple[str, str]],
) -> pd.DataFrame:
    keys_a = pd.MultiIndex.from_frame(arm_a.loc[:, list(PAIR_KEYS)])
    keys_b = pd.MultiIndex.from_frame(arm_b.loc[:, list(PAIR_KEYS)])
    missing_from_b = keys_a.difference(keys_b)
    missing_from_a = keys_b.difference(keys_a)
    if len(missing_from_a) or len(missing_from_b):
        raise DataContractError(
            "paired arms must contain identical game keys; "
            f"missing_from_{arm_b_name}={len(missing_from_b)}, "
            f"missing_from_{arm_a_name}={len(missing_from_a)}"
        )
    rename_a = {
        "line": "line_a",
        "outcome": "outcome_a",
        "raw_probability": "raw_probability_a",
        "provenance": "provenance_a",
    }
    rename_b = {
        "line": "line_b",
        "outcome": "outcome_b",
        "raw_probability": "raw_probability_b",
        "provenance": "provenance_b",
    }
    for baseline_name, columns in baseline_columns.items():
        rename_a[columns[0]] = f"{baseline_name}_probability_a"
        rename_b[columns[1]] = f"{baseline_name}_probability_b"
    left = arm_a.rename(columns=rename_a)
    right = arm_b.rename(columns=rename_b)
    merged = left.merge(right, on=list(PAIR_KEYS), how="inner", validate="one_to_one")
    return merged.sort_values(list(PAIR_KEYS)).reset_index(drop=True)


def _logit_features(probability: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    clipped = np.clip(probability, _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON)
    return np.asarray(np.log(clipped / (1.0 - clipped)).reshape(-1, 1), dtype=float)


def _score_values(
    probability: npt.NDArray[np.float64], outcome: npt.NDArray[np.float64]
) -> dict[str, float]:
    clipped = np.clip(probability, _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON)
    side = probability >= 0.5
    return {
        "accuracy": float(np.mean(side == outcome.astype(bool))),
        "brier": float(np.mean((probability - outcome) ** 2)),
        "log_loss": float(
            -np.mean(outcome * np.log(clipped) + (1.0 - outcome) * np.log1p(-clipped))
        ),
    }


def _fit_fold(
    merged: pd.DataFrame,
    *,
    evaluation_season: int,
    arm_suffix: str,
    arm_name: str,
) -> tuple[npt.NDArray[np.float64], dict[str, object]]:
    outcome_column = f"outcome_{arm_suffix}"
    probability_column = f"raw_probability_{arm_suffix}"
    training = merged.loc[(merged["season"] < evaluation_season) & merged[outcome_column].notna()]
    held_out = merged.loc[merged["season"] == evaluation_season]
    held_out_decisive = held_out.loc[held_out[outcome_column].notna()]
    if training.empty:
        raise DataContractError(
            f"{arm_name} season {evaluation_season} has no completed prior-season calibration rows"
        )
    training_outcome = training[outcome_column].to_numpy(dtype=float)
    if len(np.unique(training_outcome)) != 2:
        raise DataContractError(
            f"{arm_name} season {evaluation_season} calibration rows lack both outcome classes"
        )
    if held_out_decisive.empty:
        raise DataContractError(
            f"{arm_name} season {evaluation_season} has no decisive held-out rows"
        )
    training_raw = training[probability_column].to_numpy(dtype=float)
    held_out_raw = held_out[probability_column].to_numpy(dtype=float)
    calibrator = LogisticRegression(C=1_000_000.0, max_iter=1_000, solver="lbfgs")
    calibrator.fit(_logit_features(training_raw), training_outcome)
    training_calibrated = np.asarray(
        calibrator.predict_proba(_logit_features(training_raw))[:, 1], dtype=float
    )
    held_out_calibrated = np.asarray(
        calibrator.predict_proba(_logit_features(held_out_raw))[:, 1], dtype=float
    )
    decisive_mask = held_out[outcome_column].notna().to_numpy(dtype=bool)
    held_out_outcome = held_out.loc[held_out[outcome_column].notna(), outcome_column].to_numpy(
        dtype=float
    )
    training_scores = _score_values(training_calibrated, training_outcome)
    held_out_scores = _score_values(held_out_calibrated[decisive_mask], held_out_outcome)
    training_seasons = sorted(training["season"].astype(int).unique().tolist())
    diagnostic: dict[str, object] = {
        "season": evaluation_season,
        "arm": arm_name,
        "training_seasons": ",".join(str(value) for value in training_seasons),
        "training_rows": len(training),
        "held_out_rows": len(held_out),
        "held_out_decisive_rows": len(held_out_decisive),
        "coefficient": float(calibrator.coef_[0, 0]),
        "intercept": float(calibrator.intercept_[0]),
        "iterations": int(calibrator.n_iter_[0]),
    }
    for metric_name in ("accuracy", "brier", "log_loss"):
        in_sample = training_scores[metric_name]
        out_of_sample = held_out_scores[metric_name]
        diagnostic[f"in_sample_{metric_name}"] = in_sample
        diagnostic[f"out_of_sample_{metric_name}"] = out_of_sample
        diagnostic[f"out_minus_in_{metric_name}"] = out_of_sample - in_sample
    return held_out_calibrated, diagnostic


def _build_predictions(
    merged: pd.DataFrame,
    *,
    evaluation_seasons: tuple[int, ...],
    arm_a_name: str,
    arm_b_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_parts: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    for season in evaluation_seasons:
        held_out = merged.loc[merged["season"] == season].copy()
        if held_out.empty:
            raise DataContractError(f"evaluation season {season} has no paired rows")
        for suffix, arm_name in (("a", arm_a_name), ("b", arm_b_name)):
            calibrated, diagnostic = _fit_fold(
                merged,
                evaluation_season=season,
                arm_suffix=suffix,
                arm_name=arm_name,
            )
            held_out[f"calibrated_probability_{suffix}"] = calibrated
            held_out[f"selected_side_{suffix}"] = np.where(calibrated >= 0.5, "home", "away")
            diagnostics.append(diagnostic)
        prediction_parts.append(held_out)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    return (
        predictions.sort_values(list(PAIR_KEYS)).reset_index(drop=True),
        pd.DataFrame(diagnostics).sort_values(["season", "arm"]).reset_index(drop=True),
    )


def _paired_decisive(predictions: pd.DataFrame) -> pd.DataFrame:
    decisive = predictions.loc[
        predictions["outcome_a"].notna() & predictions["outcome_b"].notna()
    ].copy()
    if decisive.empty:
        raise DataContractError("paired evaluation has no games decisive for both arms")
    return decisive


def _metric_table(decisive: pd.DataFrame, arm_names: tuple[str, str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sources = ("calibrated", *BASELINE_NAMES)
    for source in sources:
        scores: dict[str, dict[str, float]] = {}
        for suffix, arm_name in zip(("a", "b"), arm_names, strict=True):
            probability_column = (
                f"calibrated_probability_{suffix}"
                if source == "calibrated"
                else f"{source}_probability_{suffix}"
            )
            scores[suffix] = _score_values(
                decisive[probability_column].to_numpy(dtype=float),
                decisive[f"outcome_{suffix}"].to_numpy(dtype=float),
            )
            for metric, value in scores[suffix].items():
                rows.append(
                    {
                        "source": source,
                        "arm": arm_name,
                        "metric": metric,
                        "value": value,
                        "rows": len(decisive),
                    }
                )
        for metric in ("accuracy", "brier", "log_loss"):
            rows.append(
                {
                    "source": source,
                    "arm": f"{arm_names[0]}_minus_{arm_names[1]}",
                    "metric": metric,
                    "value": scores["a"][metric] - scores["b"][metric],
                    "rows": len(decisive),
                }
            )
    return pd.DataFrame(rows)


def _reliability_table(
    decisive: pd.DataFrame, arm_names: tuple[str, str], bins: int
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for source in ("calibrated", *BASELINE_NAMES):
        for suffix, arm_name in zip(("a", "b"), arm_names, strict=True):
            probability_column = (
                f"calibrated_probability_{suffix}"
                if source == "calibrated"
                else f"{source}_probability_{suffix}"
            )
            probability = decisive[probability_column].to_numpy(dtype=float)
            outcome = decisive[f"outcome_{suffix}"].to_numpy(dtype=float)
            assignments = np.minimum(np.floor(probability * bins).astype(int), bins - 1)
            for bin_index in range(bins):
                selected = assignments == bin_index
                if not selected.any():
                    continue
                rows.append(
                    {
                        "source": source,
                        "arm": arm_name,
                        "bin": bin_index,
                        "lower": float(edges[bin_index]),
                        "upper": float(edges[bin_index + 1]),
                        "rows": int(selected.sum()),
                        "mean_probability": float(np.mean(probability[selected])),
                        "outcome_rate": float(np.mean(outcome[selected])),
                    }
                )
    return pd.DataFrame(rows)


def _paired_deltas(frame: pd.DataFrame) -> dict[str, float]:
    scores_a = _score_values(
        frame["calibrated_probability_a"].to_numpy(dtype=float),
        frame["outcome_a"].to_numpy(dtype=float),
    )
    scores_b = _score_values(
        frame["calibrated_probability_b"].to_numpy(dtype=float),
        frame["outcome_b"].to_numpy(dtype=float),
    )
    return {metric: scores_a[metric] - scores_b[metric] for metric in scores_a}


def _joint_season_bootstrap(
    decisive: pd.DataFrame,
    *,
    samples: int,
    seed: int,
    confidence: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = np.sort(decisive["season"].unique())
    rng = np.random.default_rng(seed)
    metric_names = ("accuracy", "brier", "log_loss")
    season_indices = {season: index for index, season in enumerate(seasons)}
    season_rows: npt.NDArray[np.int64] = np.empty(len(seasons), dtype=np.int64)
    sums_a: npt.NDArray[np.float64] = np.empty((len(seasons), len(metric_names)), dtype=float)
    sums_b: npt.NDArray[np.float64] = np.empty((len(seasons), len(metric_names)), dtype=float)
    for season_index, season in enumerate(seasons):
        frame = decisive.loc[decisive["season"] == season]
        season_rows[season_index] = len(frame)
        for suffix, sums in (("a", sums_a), ("b", sums_b)):
            probability = frame[f"calibrated_probability_{suffix}"].to_numpy(dtype=float)
            outcome = frame[f"outcome_{suffix}"].to_numpy(dtype=float)
            clipped = np.clip(probability, _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON)
            sums[season_index] = (
                float(np.count_nonzero((probability >= 0.5) == outcome.astype(bool))),
                float(np.sum((probability - outcome) ** 2)),
                float(-np.sum(outcome * np.log(clipped) + (1.0 - outcome) * np.log1p(-clipped))),
            )
    draw_rows: list[dict[str, object]] = []
    for draw in range(samples):
        sampled = rng.choice(seasons, size=len(seasons), replace=True)
        sampled_indices: npt.NDArray[np.intp] = np.fromiter(
            (season_indices[season] for season in sampled),
            dtype=np.intp,
            count=len(seasons),
        )
        rows = int(season_rows[sampled_indices].sum())
        deltas = (
            sums_a[sampled_indices].sum(axis=0) / rows - sums_b[sampled_indices].sum(axis=0) / rows
        )
        for metric, delta in zip(metric_names, deltas, strict=True):
            draw_rows.append({"draw": draw, "metric": metric, "delta": float(delta)})
    draws = pd.DataFrame(draw_rows)
    alpha = (1.0 - confidence) / 2.0
    point = _paired_deltas(decisive)
    summary_rows: list[dict[str, object]] = []
    for metric in metric_names:
        values = draws.loc[draws["metric"] == metric, "delta"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "metric": metric,
                "point": point[metric],
                "lower": float(np.quantile(values, alpha)),
                "upper": float(np.quantile(values, 1.0 - alpha)),
                "probability_positive": probability_positive_from_draws(values),
                "samples": samples,
            }
        )
    return pd.DataFrame(summary_rows), draws


def evaluate_paired_openers(
    arm_a: pd.DataFrame,
    arm_b: pd.DataFrame,
    *,
    arm_a_name: str,
    arm_b_name: str,
    completed_seasons: Sequence[int],
    baseline_probability_columns: Mapping[str, tuple[str, str]],
    evaluation_seasons: Sequence[int] | None = None,
    family: str = "MOD18_FANDUEL_SINGLE_BOOK_V1",
    reliability_bins: int = 10,
    bootstrap_samples: int = 20_000,
    bootstrap_seed: int = 20_260_817,
    confidence: float = 0.95,
) -> PairedOpenerEvaluation:
    names = (arm_a_name.strip(), arm_b_name.strip())
    if not all(names) or names[0] == names[1]:
        raise DataContractError("arm names must be nonempty and distinct")
    if not family.strip():
        raise DataContractError("family must be nonempty")
    if set(baseline_probability_columns) != set(BASELINE_NAMES):
        raise DataContractError(
            "baseline_probability_columns must contain exactly market and model_only"
        )
    for baseline_name, columns in baseline_probability_columns.items():
        if len(columns) != 2:
            raise DataContractError(
                f"baseline {baseline_name!r} must provide one column for each arm"
            )
    for arm_index, name in enumerate(names):
        selected = [
            baseline_probability_columns[baseline][arm_index] for baseline in BASELINE_NAMES
        ]
        if len(set(selected)) != len(selected) or set(selected) & set(REQUIRED_COLUMNS):
            raise DataContractError(
                f"{name} baseline columns must be distinct from each other and required columns"
            )
    completed = _normalize_completed_seasons(completed_seasons)
    validated_a = _validate_arm(
        arm_a,
        arm_name=names[0],
        completed_seasons=completed,
        baseline_columns=baseline_probability_columns,
        baseline_index=0,
    )
    validated_b = _validate_arm(
        arm_b,
        arm_name=names[1],
        completed_seasons=completed,
        baseline_columns=baseline_probability_columns,
        baseline_index=1,
    )
    merged = _merge_arms(
        validated_a,
        validated_b,
        arm_a_name=names[0],
        arm_b_name=names[1],
        baseline_columns=baseline_probability_columns,
    )
    available = tuple(sorted(merged["season"].astype(int).unique().tolist()))
    if len(available) < 2:
        raise DataContractError("paired evaluation requires at least two completed seasons")
    if evaluation_seasons is None:
        selected_evaluation_seasons = available[1:]
    else:
        selected_evaluation_seasons = tuple(
            sorted(_normalize_completed_seasons(evaluation_seasons))
        )
    unavailable = sorted(set(selected_evaluation_seasons) - set(available))
    if unavailable:
        raise DataContractError(f"evaluation seasons are absent from paired inputs: {unavailable}")
    for season in selected_evaluation_seasons:
        if not any(prior < season for prior in available):
            raise DataContractError(
                f"evaluation season {season} has no completed prior-season training data"
            )
    if (
        isinstance(reliability_bins, bool)
        or not isinstance(reliability_bins, (int, np.integer))
        or reliability_bins < 1
    ):
        raise DataContractError("reliability_bins must be a positive integer")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, (int, np.integer))
        or bootstrap_samples < 1
    ):
        raise DataContractError("bootstrap_samples must be a positive integer")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, (int, np.integer)):
        raise DataContractError("bootstrap_seed must be an integer")
    if not 0.0 < confidence < 1.0:
        raise DataContractError("confidence must be between zero and one")
    reliability_bins = int(reliability_bins)
    bootstrap_samples = int(bootstrap_samples)
    bootstrap_seed = int(bootstrap_seed)
    predictions, fold_diagnostics = _build_predictions(
        merged,
        evaluation_seasons=selected_evaluation_seasons,
        arm_a_name=names[0],
        arm_b_name=names[1],
    )
    decisive = _paired_decisive(predictions)
    metrics = _metric_table(decisive, names)
    reliability = _reliability_table(decisive, names, reliability_bins)
    bootstrap_summary, bootstrap_draws = _joint_season_bootstrap(
        decisive,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    metadata: dict[str, object] = {
        "family": family.strip(),
        "primary_look": f"{names[0]}_minus_{names[1]}_calibrated_accuracy",
        "primary_look_count": 1,
        "diagnostic_proper_score_count": 2,
        "diagnostic_proper_scores": ["brier", "log_loss"],
        "delta_direction": "arm_a_minus_arm_b_for_all_metrics",
        "proper_score_preference": "lower_is_better",
        "arm_a": names[0],
        "arm_b": names[1],
        "available_completed_seasons": list(available),
        "evaluation_seasons": list(selected_evaluation_seasons),
        "calibration_rule": "strictly_prior_completed_seasons_arm_specific_platt",
        "selection_rule": "calibrated_probability_at_least_0.5_selects_home",
        "paired_rows": len(predictions),
        "paired_decisive_rows": len(decisive),
        "arm_a_push_rows": int(predictions["outcome_a"].isna().sum()),
        "arm_b_push_rows": int(predictions["outcome_b"].isna().sum()),
        "bootstrap_unit": "season",
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "confidence": confidence,
        "provenance_a": sorted(predictions["provenance_a"].astype(str).unique().tolist()),
        "provenance_b": sorted(predictions["provenance_b"].astype(str).unique().tolist()),
    }
    return PairedOpenerEvaluation(
        predictions=predictions,
        fold_diagnostics=fold_diagnostics,
        metrics=metrics,
        reliability=reliability,
        bootstrap_summary=bootstrap_summary,
        bootstrap_draws=bootstrap_draws,
        metadata=metadata,
    )


__all__ = ["PairedOpenerEvaluation", "evaluate_paired_openers"]
