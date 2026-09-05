"""PER-14 annual aggregation and production-fold leakage regressions."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nfl_ats import margin
from nfl_ats.unit_prior_features import UNIT_PRIOR_COLUMNS, attach_unit_prior_features
from scripts.unit_apm_team_ratings import aggregate_team_unit, correlation_summary
from scripts.unit_prior_features_on_production import (
    PROFILE,
    candidate_profile,
    oracle_features,
    run_folds,
)


def test_snap_weights_trades_unique_keys_and_team_effect_exclusion():
    table = pd.DataFrame(
        {
            "season": [2020] * 3,
            "posteam": ["BUF", "BUF", "KC"],
            "defteam": ["KC", "KC", "BUF"],
            "offense_players": ["a;b;skill", "a;skill", "a"],
            "defense_players": ["d", "d", "e"],
        }
    )
    lookup = {("a", 2020): "OFF_OL", ("b", 2020): "OFF_OL", ("skill", 2020): "OFF_SKILL"}
    coefs = {
        "unit_player::a": 1.0,
        "unit_player::b": 7.0,
        "unit_player::skill": 999.0,
        "unit_team::BUF": 9999.0,
    }
    ratings = aggregate_team_unit(table, "OFF_OL", lookup, coefs)
    assert not ratings.duplicated(["season", "team", "unit"]).any()
    buf = ratings.set_index("team").loc["BUF"]
    assert buf.rating == pytest.approx(3.0)  # (2*1 + 1*7)/3
    assert (buf.players, buf.snaps, buf.members) == (2, 3, "a;b")
    assert ratings.set_index("team").loc["KC", "rating"] == 1.0


def test_annual_producer_to_join_excludes_current_and_future_poison():
    ratings = []
    for season, scale in ((2020, 1.0), (2021, 10000.0), (2022, -99999.0)):
        table = pd.DataFrame(
            {
                "season": [season, season],
                "posteam": ["BUF", "KC"],
                "offense_players": ["a;b", "c;d"],
            }
        )
        lookup = {
            (p, season): unit
            for p, unit in (
                ("a", "OFF_OL"),
                ("b", "OFF_SKILL"),
                ("c", "OFF_OL"),
                ("d", "OFF_SKILL"),
            )
        }
        coef = {
            f"unit_player::{p}": scale * value
            for p, value in (("a", 3), ("b", 4), ("c", 1), ("d", 1))
        }
        for unit in ("OFF_OL", "OFF_SKILL"):
            ratings.append(aggregate_team_unit(table, unit, lookup, coef))
    source = pd.concat(ratings, ignore_index=True)
    games = pd.DataFrame(
        {
            "season": [2021],
            "home_team": ["BUF"],
            "away_team": ["KC"],
            "prediction_timestamp": ["2021-09-01"],
        }
    )
    result = attach_unit_prior_features(games, source)
    assert result[list(UNIT_PRIOR_COLUMNS)].iloc[0].tolist() == [2.0, 3.0]
    oracle = oracle_features(games, source)
    assert oracle[list(UNIT_PRIOR_COLUMNS)].iloc[0].tolist() == [20000.0, 30000.0]
    games["prediction_timestamp"] = "2021-03-01"
    assert attach_unit_prior_features(games, source)[list(UNIT_PRIOR_COLUMNS)].isna().all().all()


def test_season_fold_training_calibration_boundary_and_opener(monkeypatch):
    base = pd.DataFrame(
        {
            "game_id": ["old", "a", "push", "b", "next"],
            "season": [2019, 2020, 2020, 2020, 2021],
            "week": [1, 1, 1, 10, 1],
            "gameday": pd.to_datetime(
                ["2019-09-01", "2020-09-01", "2020-09-01", "2020-11-01", "2021-09-01"]
            ),
            "result": [1.0, 1.0, 3.0, -2.0, 99999.0],
            "spread_line": [7.0] * 5,
        }
    )
    candidate = base.assign(**dict.fromkeys(UNIT_PRIOR_COLUMNS, 2.0))
    oracle = base.assign(**dict.fromkeys(UNIT_PRIOR_COLUMNS, 9000.0))
    lines = pd.DataFrame({"game_id": ["a", "push", "b"], "tue_open_home_spread": [0.0, 3.0, 0.0]})
    calls = []

    def fit(training, **kwargs):
        calls.append(training.game_id.tolist())
        assert training.game_id.tolist() == ["old"]
        assert kwargs["ridge_alpha"] == 10.0

        def predict(scoring, **options):
            assert options["probability_method"] == "gaussian"
            assert scoring.spread_line.tolist() == [0.0, 0.0]
            return pd.DataFrame({"home_cover_probability": [0.5, 0.49]})

        return SimpleNamespace(predict=predict, distribution_rows=1)

    monkeypatch.setattr(margin, "fit_margin_model", fit)
    paired, audit = run_folds(base, candidate, oracle, lines, "gaussian")
    assert len(calls) == 3
    assert paired.game_id.tolist() == ["a", "b"]
    assert paired.baseline_correct_open_pr.tolist() == [1.0, 1.0]
    assert all(row["training_max_season"] == 2019 for row in audit)
    assert base.spread_line.eq(7).all()


def test_runtime_profile_restored_and_exactly_two_columns():
    profiles = margin.MARGIN_FEATURE_PROFILES
    with candidate_profile():
        assert margin.margin_feature_columns("market_residual", PROFILE) == (
            *margin.margin_feature_columns("market_residual", "weak_stack"),
            *UNIT_PRIOR_COLUMNS,
        )
    assert profiles == margin.MARGIN_FEATURE_PROFILES
    assert PROFILE not in margin._MARGIN_PROFILE_FEATURE_SETS


def test_correlation_bootstrap_is_reproducible_and_reports_missing_variation():
    frame = pd.DataFrame(
        {
            "team": ["A", "B", "C", "A", "B", "C"],
            "x": [1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
            "y": [2.0, 4.0, 5.0, 3.0, 5.0, 7.0],
        }
    )
    result = correlation_summary(frame, "x", "y")
    assert result == correlation_summary(frame, "x", "y")
    assert result["pairs"] == 6
    assert result["pearson"] > 0.9
    assert correlation_summary(frame.assign(x=1.0), "x", "y")["pearson"] is None
