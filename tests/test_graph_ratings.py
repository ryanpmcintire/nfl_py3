from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.constants import GRAPH_FEATURE_COLUMNS
from nfl_ats.graph_ratings import GraphRatingConfig, add_schedule_strength_features


def _games() -> pd.DataFrame:
    matchups = (
        (1, "B", "A", 10.0),
        (1, "D", "C", 3.0),
        (2, "C", "A", 7.0),
        (2, "D", "B", 6.0),
        (3, "D", "A", 14.0),
        (3, "C", "B", 4.0),
        (4, "A", "C", -3.0),
        (4, "B", "D", -7.0),
        (5, "B", "A", 8.0),
        (5, "D", "C", 1.0),
    )
    rows = []
    for week, away, home, margin in matchups:
        home_score = 24.0 + max(0.0, margin)
        away_score = home_score - margin
        rows.append(
            {
                "game_id": f"2022_{week:02d}_{away}_{home}",
                "season": 2022,
                "week": week,
                "gameday": pd.Timestamp("2022-09-01") + pd.Timedelta(days=7 * week),
                "away_team": away,
                "home_team": home,
                "away_score": away_score,
                "home_score": home_score,
                "result": margin,
            }
        )
    return pd.DataFrame(rows)


def _config() -> GraphRatingConfig:
    return GraphRatingConfig(min_games=2, half_life_weeks=4.0, ridge_alpha=2.0)


def test_graph_features_are_continuous_and_pregame() -> None:
    ratings = add_schedule_strength_features(_games(), _config())
    assert set(GRAPH_FEATURE_COLUMNS).issubset(ratings.columns)
    assert ratings.loc[ratings["week"].eq(1), list(GRAPH_FEATURE_COLUMNS)].isna().all(axis=None)
    assert ratings.loc[ratings["week"].eq(3), list(GRAPH_FEATURE_COLUMNS)].notna().all(axis=None)

    late_week = ratings.loc[ratings["week"].eq(5)]
    late = late_week.iloc[0]
    assert late["home_graph_pagerank"] > late["away_graph_pagerank"]
    assert late["graph_pagerank_diff"] == pytest.approx(
        late["home_graph_pagerank"] - late["away_graph_pagerank"]
    )
    home_field_estimates = (
        late_week["schedule_predicted_margin"] - late_week["schedule_rating_diff"]
    )
    assert np.ptp(home_field_estimates.to_numpy()) < 1e-12


def test_current_week_outcomes_cannot_change_current_ratings() -> None:
    games = _games()
    baseline = add_schedule_strength_features(games, _config())
    changed_games = games.copy()
    current = changed_games["week"].eq(3)
    changed_games.loc[current, "result"] *= -10.0
    changed_games.loc[current, "home_score"] = 3.0
    changed_games.loc[current, "away_score"] = 70.0
    changed = add_schedule_strength_features(changed_games, _config())

    columns = list(GRAPH_FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(
        baseline.loc[baseline["week"].eq(3), columns].reset_index(drop=True),
        changed.loc[changed["week"].eq(3), columns].reset_index(drop=True),
    )
    assert not np.allclose(
        baseline.loc[baseline["week"].eq(4), columns],
        changed.loc[changed["week"].eq(4), columns],
    )


def test_future_outcomes_and_input_order_cannot_change_prior_ratings() -> None:
    games = _games()
    baseline = add_schedule_strength_features(games, _config())
    changed_games = games.sample(frac=1.0, random_state=91).copy()
    future = changed_games["week"].eq(5)
    changed_games.loc[future, ["result", "home_score", "away_score"]] = np.nan
    changed = add_schedule_strength_features(changed_games, _config())

    columns = ["game_id", *GRAPH_FEATURE_COLUMNS]
    expected = baseline.loc[baseline["week"].le(5), columns].sort_values("game_id")
    actual = changed.loc[changed["week"].le(5), columns].sort_values("game_id")
    pd.testing.assert_frame_equal(expected.reset_index(drop=True), actual.reset_index(drop=True))


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("half_life_weeks", 0.0, "half_life"),
        ("offseason_retention", 1.1, "offseason_retention"),
        ("damping", 1.0, "damping"),
        ("prior_weight", 0.0, "prior_weight"),
        ("ridge_alpha", 0.0, "ridge_alpha"),
        ("min_games", 0, "min_games"),
        ("offseason_age_weeks", -1, "offseason_age_weeks"),
    ],
)
def test_graph_configuration_guards(setting: str, value: float, message: str) -> None:
    values = GraphRatingConfig().__dict__ | {setting: value}
    with pytest.raises(ValueError, match=message):
        add_schedule_strength_features(_games(), GraphRatingConfig(**values))


def test_graph_schema_guard() -> None:
    with pytest.raises(ValueError, match="Graph ratings require"):
        add_schedule_strength_features(pd.DataFrame({"game_id": ["missing"]}))
