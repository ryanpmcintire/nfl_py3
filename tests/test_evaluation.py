from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from nfl_ats.constants import MODEL_FEATURE_COLUMNS
from nfl_ats.evaluation import (
    EvaluationCandidate,
    nested_walk_forward_evaluation,
    parse_candidates,
)


@pytest.fixture
def evaluation_frame() -> pd.DataFrame:
    rows_per_season = 60
    seasons = np.repeat(np.arange(2017, 2022), rows_per_season)
    index = np.arange(len(seasons))
    frame = pd.DataFrame(
        {
            "game_id": [f"evaluation_{value:03d}" for value in index],
            "season": seasons,
            "week": np.tile(np.repeat(np.arange(1, 7), 10), 5),
            "gameday": [date(2017, 1, 1) + timedelta(days=int(value)) for value in index],
            "away_team": "AWY",
            "home_team": "HME",
            "home_spread_odds": -110.0,
            "away_spread_odds": -110.0,
        }
    )
    for feature_index, column in enumerate(MODEL_FEATURE_COLUMNS, start=1):
        frame[column] = np.sin(index / feature_index) + (index % 7) / 10.0
    frame["spread_line"] = np.where(index % 2 == 0, 2.5, -2.5)
    frame["home_cover"] = ((index + seasons) % 3 != 0).astype(float)
    frame["ats_margin"] = np.where(frame["home_cover"].eq(1), 3.0, -3.0)
    frame["result"] = frame["spread_line"] + frame["ats_margin"]
    return frame


def test_nested_evaluation_selects_before_each_outer_season(
    evaluation_frame: pd.DataFrame,
) -> None:
    candidates = (
        EvaluationCandidate("logistic", "market"),
        EvaluationCandidate("logistic", "market_context"),
    )
    result = nested_walk_forward_evaluation(
        evaluation_frame,
        first_test_season=2020,
        last_test_season=2021,
        validation_seasons=1,
        candidates=candidates,
        min_train_games=80,
    )

    assert len(result.candidate_validation) == 4
    assert result.candidate_validation.groupby("outer_test_season")["selected"].sum().eq(1).all()
    assert set(result.predictions["season"]) == {2020, 2021}
    assert result.metrics["protocol"] == "nested_rolling_origin"
    assert result.metrics["outer_folds"] == 2
    for test_season, fold in result.fold_summary.set_index("outer_test_season").iterrows():
        assert fold["validation_end_season"] == test_season - 1
        predictions = result.predictions.loc[result.predictions["season"].eq(test_season)]
        assert predictions["selected_candidate_id"].nunique() == 1
        assert predictions["selected_candidate_id"].iloc[0] == fold["selected_candidate_id"]


def test_candidate_parser_and_protocol_guards(evaluation_frame: pd.DataFrame) -> None:
    assert parse_candidates("logistic:market,hgb:full")[1].candidate_id == "hgb:full"
    with pytest.raises(ValueError, match="expected model:feature_set"):
        parse_candidates("logistic")
    with pytest.raises(ValueError, match="unique"):
        parse_candidates("logistic:market,logistic:market")
    with pytest.raises(ValueError, match="last_test_season"):
        nested_walk_forward_evaluation(
            evaluation_frame,
            first_test_season=2021,
            last_test_season=2020,
        )
