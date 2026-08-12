from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.backtest import score_week, summarize_predictions, walk_forward_backtest


def test_score_week_has_strict_training_cutoff(model_frame: pd.DataFrame) -> None:
    predictions, model = score_week(
        model_frame,
        season=2020,
        week=1,
        min_train_games=80,
    )
    test_start = pd.to_datetime(predictions["gameday"]).min()
    assert pd.Timestamp(model.training_max_gameday) < test_start
    assert predictions["train_max_gameday"].nunique() == 1
    assert set(predictions["bet_side"]).issubset({"HOME", "AWAY", "PASS"})


def test_week_prediction_is_invariant_to_target_outcomes_and_row_order(
    model_frame: pd.DataFrame,
) -> None:
    baseline, _ = score_week(model_frame, season=2020, week=1, min_train_games=80)
    altered = model_frame.sample(frac=1.0, random_state=17).copy()
    target = altered["season"].eq(2020) & altered["week"].eq(1)
    altered.loc[target, "home_cover"] = 1.0 - altered.loc[target, "home_cover"]
    altered.loc[target, "ats_margin"] = 999.0
    altered.loc[target, "result"] = -999.0
    rescored, _ = score_week(altered, season=2020, week=1, min_train_games=80)

    columns = ["game_id", "home_cover_probability", "pick", "bet_side", "edge"]
    pd.testing.assert_frame_equal(
        baseline.loc[:, columns].sort_values("game_id").reset_index(drop=True),
        rescored.loc[:, columns].sort_values("game_id").reset_index(drop=True),
    )


def test_walk_forward_backtest_outputs_probability_and_betting_metrics(
    model_frame: pd.DataFrame,
) -> None:
    result = walk_forward_backtest(
        model_frame,
        start_season=2020,
        min_edge=0.0,
        min_train_games=80,
    )
    assert len(result.predictions) == 60
    assert result.metrics["games_evaluated"] == 60
    assert 0 < result.metrics["bets"] <= 60
    assert 0.0 <= result.metrics["brier_score"] <= 1.0
    for _, group in result.predictions.groupby(["season", "week"]):
        assert pd.Timestamp(group["train_max_gameday"].iloc[0]) < group["gameday"].min()


def test_backtest_guards(model_frame: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="No games found"):
        score_week(model_frame, season=2030, week=1)
    with pytest.raises(ValueError, match="need 500"):
        score_week(model_frame, season=2020, week=1)
    with pytest.raises(ValueError, match="No completed games"):
        walk_forward_backtest(model_frame, start_season=2030)

    missing_line = model_frame.copy()
    missing_line.loc[
        missing_line["season"].eq(2020) & missing_line["week"].eq(1), "spread_line"
    ] = np.nan
    with pytest.raises(ValueError, match="missing spread lines"):
        score_week(missing_line, season=2020, week=1, min_train_games=80)
    with pytest.raises(ValueError, match="no non-push"):
        summarize_predictions(pd.DataFrame({"home_cover": [float("nan")]}))
