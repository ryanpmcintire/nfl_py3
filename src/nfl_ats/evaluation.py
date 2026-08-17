"""Nested chronological model selection and outer-fold evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from nfl_ats.backtest import BacktestResult, summarize_predictions, walk_forward_backtest
from nfl_ats.constants import DEFAULT_MIN_TRAIN_GAMES, FEATURE_SETS
from nfl_ats.modeling import MODEL_NAMES

SELECTION_METRICS = ("brier_score", "log_loss")


@dataclass(frozen=True, order=True)
class EvaluationCandidate:
    model_name: str
    feature_set: str

    def __post_init__(self) -> None:
        if self.model_name not in MODEL_NAMES:
            raise ValueError(f"Unknown model in evaluation candidate: {self.model_name}")
        if self.feature_set not in FEATURE_SETS:
            raise ValueError(f"Unknown feature set in evaluation candidate: {self.feature_set}")

    @property
    def candidate_id(self) -> str:
        return f"{self.model_name}:{self.feature_set}"


# This fixed, reviewable search budget is intentionally small. Adding a
# candidate changes the protocol and therefore the experiment configuration.
DEFAULT_EVALUATION_CANDIDATES = (
    EvaluationCandidate("logistic", "market"),
    EvaluationCandidate("logistic", "market_context"),
    EvaluationCandidate("logistic", "market_elo"),
    EvaluationCandidate("logistic", "football"),
    EvaluationCandidate("logistic", "full_without_ats"),
    EvaluationCandidate("logistic", "full"),
    EvaluationCandidate("hgb", "market_context"),
    EvaluationCandidate("hgb", "football"),
    EvaluationCandidate("hgb", "full"),
)


@dataclass(frozen=True)
class NestedEvaluationResult:
    candidate_validation: pd.DataFrame
    fold_summary: pd.DataFrame
    predictions: pd.DataFrame
    metrics: dict[str, Any]


def parse_candidates(value: str) -> tuple[EvaluationCandidate, ...]:
    """Parse ``model:feature_set`` candidates from a comma-separated CLI value."""

    candidates: list[EvaluationCandidate] = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        parts = text.split(":", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid candidate {text!r}; expected model:feature_set")
        candidates.append(EvaluationCandidate(parts[0].strip(), parts[1].strip()))
    if not candidates:
        raise ValueError("At least one evaluation candidate is required")
    if len(set(candidates)) != len(candidates):
        raise ValueError("Evaluation candidates must be unique")
    return tuple(candidates)


def format_candidates(candidates: tuple[EvaluationCandidate, ...]) -> str:
    return ",".join(candidate.candidate_id for candidate in candidates)


def _evaluate_candidate(
    features: pd.DataFrame,
    candidate: EvaluationCandidate,
    *,
    start_season: int,
    end_season: int,
    min_edge: float,
    min_train_games: int,
) -> BacktestResult:
    return walk_forward_backtest(
        features,
        start_season=start_season,
        end_season=end_season,
        model_name=candidate.model_name,
        feature_set=candidate.feature_set,
        min_edge=min_edge,
        min_train_games=min_train_games,
    )


def _slice_candidate_result(
    stream: BacktestResult,
    candidate: EvaluationCandidate,
    *,
    start_season: int,
    end_season: int,
    min_edge: float,
    min_train_games: int,
) -> BacktestResult:
    """Summarize a season range from a precomputed chronological stream."""

    predictions = stream.predictions.loc[
        stream.predictions["season"].between(start_season, end_season)
    ].copy()
    if predictions.empty:
        raise ValueError(f"No cached predictions found from {start_season} through {end_season}")
    metrics = summarize_predictions(predictions)
    metrics.update(
        {
            "model_name": candidate.model_name,
            "feature_set": candidate.feature_set,
            "start_season": start_season,
            "end_season": end_season,
            "min_edge": min_edge,
            "min_train_games": min_train_games,
            "first_test_gameday": predictions["gameday"].min().date().isoformat(),
            "last_test_gameday": predictions["gameday"].max().date().isoformat(),
        }
    )
    return BacktestResult(predictions.reset_index(drop=True), metrics)


def nested_walk_forward_evaluation(
    features: pd.DataFrame,
    *,
    first_test_season: int,
    last_test_season: int,
    validation_seasons: int = 2,
    candidates: tuple[EvaluationCandidate, ...] = DEFAULT_EVALUATION_CANDIDATES,
    selection_metric: str = "brier_score",
    min_edge: float = 0.02,
    min_train_games: int = DEFAULT_MIN_TRAIN_GAMES,
) -> NestedEvaluationResult:
    """Select on prior seasons, then score each outer season exactly once.

    For outer season ``Y``, candidate configurations are compared only on
    ``Y-validation_seasons`` through ``Y-1``. The winner is frozen before any
    rows from season ``Y`` are scored. Weekly refitting inside the outer season
    may use results from earlier weeks, matching the intended deployment loop.
    """

    if last_test_season < first_test_season:
        raise ValueError("last_test_season cannot be earlier than first_test_season")
    if validation_seasons < 1:
        raise ValueError("validation_seasons must be positive")
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}")
    if not candidates:
        raise ValueError("At least one evaluation candidate is required")
    if len(set(candidates)) != len(candidates):
        raise ValueError("Evaluation candidates must be unique")

    # Each candidate's prediction for a given week depends only on games before
    # that week, not on which outer fold later consumes it. Compute that stream
    # once, then take immutable season slices below. This is statistically
    # identical to refitting overlapping validation windows independently and
    # avoids repeating the same candidate/week fit two or three times.
    stream_start_season = first_test_season - validation_seasons
    candidate_streams = {
        candidate: _evaluate_candidate(
            features,
            candidate,
            start_season=stream_start_season,
            end_season=last_test_season,
            min_edge=min_edge,
            min_train_games=min_train_games,
        )
        for candidate in candidates
    }

    validation_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    outer_predictions: list[pd.DataFrame] = []
    for test_season in range(first_test_season, last_test_season + 1):
        validation_start = test_season - validation_seasons
        validation_end = test_season - 1
        candidate_results: list[tuple[EvaluationCandidate, BacktestResult]] = []
        for candidate in candidates:
            result = _slice_candidate_result(
                candidate_streams[candidate],
                candidate,
                start_season=validation_start,
                end_season=validation_end,
                min_edge=min_edge,
                min_train_games=min_train_games,
            )
            candidate_results.append((candidate, result))
            validation_rows.append(
                {
                    "outer_test_season": test_season,
                    "validation_start_season": validation_start,
                    "validation_end_season": validation_end,
                    "candidate_id": candidate.candidate_id,
                    "model_name": candidate.model_name,
                    "feature_set": candidate.feature_set,
                    "feature_count": len(FEATURE_SETS[candidate.feature_set]),
                    **result.metrics,
                }
            )

        selected, selected_validation = min(
            candidate_results,
            key=lambda item: (
                float(item[1].metrics[selection_metric]),
                float(item[1].metrics["log_loss"]),
                item[0].candidate_id,
            ),
        )
        outer = _slice_candidate_result(
            candidate_streams[selected],
            selected,
            start_season=test_season,
            end_season=test_season,
            min_edge=min_edge,
            min_train_games=min_train_games,
        )
        predictions = outer.predictions.copy()
        predictions["outer_test_season"] = test_season
        predictions["validation_start_season"] = validation_start
        predictions["validation_end_season"] = validation_end
        predictions["selected_candidate_id"] = selected.candidate_id
        predictions["selected_model_name"] = selected.model_name
        predictions["selected_feature_set"] = selected.feature_set
        predictions["selection_metric"] = selection_metric
        predictions["validation_selection_score"] = selected_validation.metrics[selection_metric]
        outer_predictions.append(predictions)
        fold_rows.append(
            {
                "outer_test_season": test_season,
                "validation_start_season": validation_start,
                "validation_end_season": validation_end,
                "selected_candidate_id": selected.candidate_id,
                "selected_model_name": selected.model_name,
                "selected_feature_set": selected.feature_set,
                "selection_metric": selection_metric,
                "validation_selection_score": selected_validation.metrics[selection_metric],
                **{f"test_{key}": value for key, value in outer.metrics.items()},
            }
        )

    predictions = pd.concat(outer_predictions, ignore_index=True).sort_values(
        ["gameday", "game_id"], ignore_index=True
    )
    candidate_validation = pd.DataFrame(validation_rows)
    candidate_validation["selection_rank"] = candidate_validation.groupby("outer_test_season")[
        selection_metric
    ].rank(method="min", ascending=True)
    fold_summary = pd.DataFrame(fold_rows).sort_values("outer_test_season", ignore_index=True)
    selected_by_season = fold_summary.set_index("outer_test_season")[
        "selected_candidate_id"
    ].to_dict()
    candidate_validation["selected"] = candidate_validation.apply(
        lambda row: row["candidate_id"] == selected_by_season[int(row["outer_test_season"])],
        axis=1,
    )
    metrics = summarize_predictions(predictions)
    metrics.update(
        {
            "protocol": "nested_rolling_origin",
            "first_outer_test_season": first_test_season,
            "last_outer_test_season": last_test_season,
            "outer_folds": len(fold_summary),
            "validation_seasons": validation_seasons,
            "selection_metric": selection_metric,
            "candidate_count": len(candidates),
            "candidates": [candidate.candidate_id for candidate in candidates],
            "selection_counts": fold_summary["selected_candidate_id"].value_counts().to_dict(),
        }
    )
    return NestedEvaluationResult(
        candidate_validation=candidate_validation.sort_values(
            ["outer_test_season", "selection_rank", "candidate_id"], ignore_index=True
        ),
        fold_summary=fold_summary,
        predictions=predictions,
        metrics=metrics,
    )
