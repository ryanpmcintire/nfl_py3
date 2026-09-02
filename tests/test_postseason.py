"""Postseason rows are servable, but never train or grade anything.

The canonical feature table can carry WC/DIV/CON/SB rows so playoff weeks can
be predicted, while every training and evaluation path stays regular-season
only. These tests pin both halves of that contract: the regular-season table is
bit-identical whether or not postseason rows are requested, and no fitting,
backtest, or evaluation path can see a postseason row.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats import cli
from nfl_ats.backtest import walk_forward_backtest
from nfl_ats.clv import upcoming_week
from nfl_ats.constants import GRAPH_FEATURE_COLUMNS, MODEL_FEATURE_COLUMNS
from nfl_ats.features import POSTSEASON_GAME_TYPES, build_game_features
from nfl_ats.margin import fit_margin_model, fit_market_baseline
from nfl_ats.modeling import fit_cover_model, regular_season_rows
from nfl_ats.outcomes import OUTCOME_METHODS, score_outcome_week, walk_forward_outcomes

# ---------------------------------------------------------------------------
# A tiny two-season league with a full postseason bracket
# ---------------------------------------------------------------------------

FIRST_SEASON = 2021
SECOND_SEASON = 2022
REGULAR_SEASON_WEEKS = 18
SEASON_OPENERS = {FIRST_SEASON: date(2021, 9, 12), SECOND_SEASON: date(2022, 9, 11)}

# (game_type, week, gameday, home, away, margin, spread_line)
POSTSEASON_BRACKET: tuple[tuple[str, int, date, str, str, float, float], ...] = (
    ("WC", 19, date(2022, 1, 16), "A", "C", 49.0, 2.5),
    ("WC", 19, date(2022, 1, 16), "B", "D", 6.0, -1.5),
    ("DIV", 20, date(2022, 1, 23), "A", "B", -4.0, 3.5),
    ("CON", 21, date(2022, 1, 30), "A", "D", 10.0, -2.5),
    ("SB", 22, date(2022, 2, 13), "A", "B", 3.0, 1.5),
)
WILD_CARD_GAME_ID = "2021_19_C_A"
DIVISIONAL_GAME_ID = "2021_20_B_A"
SUPER_BOWL_GAME_ID = "2021_22_B_A"
WEEK_18_GAME_ID = "2021_18_A_B"
BRACKET_TEAM = "A"
# The comparison bracket: the same wild-card game lost by three touchdowns
# instead of won by seven. Elo is margin-blind, so flipping the winner is what
# makes the divisional round's rating provably a function of the wild-card
# result; the margin-weighted schedule graph moves either way.
UPSET_MARGIN = -21.0
# ``add_elo_features``' default home-field term, removed when recovering a
# team-oriented Elo gap from the published ``elo_diff`` column.
HOME_FIELD_ELO = 55.0

# Small enough to keep the fixture cheap; the postseason contract does not
# depend on the rolling-window sizes.
BUILD_KWARGS = {"span": 3, "min_periods": 1, "graph_min_games": 4}


def _schedule_row(
    *,
    season: int,
    game_type: str,
    week: int,
    gameday: date,
    home: str,
    away: str,
    margin: float,
    spread: float,
) -> dict[str, object]:
    away_score = 17 + (week % 5)
    return {
        "game_id": f"{season}_{week:02d}_{away}_{home}",
        "season": season,
        "game_type": game_type,
        "week": week,
        "gameday": pd.Timestamp(gameday),
        "away_team": away,
        "home_team": home,
        "away_score": float(away_score),
        "home_score": float(away_score) + margin,
        "result": margin,
        "spread_line": spread,
        "total_line": 42.0 + (week % 6),
        "home_rest": 7,
        "away_rest": 7,
        "location": "Neutral" if game_type == "SB" else "Home",
        "div_game": 1 if game_type == "REG" else 0,
        "temp": 60.0,
        "wind": 7.0,
        "home_spread_odds": -110.0,
        "away_spread_odds": -110.0,
    }


def _team_stat_rows(schedules: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for position, game in enumerate(schedules.itertuples(index=False)):
        season_type = "REG" if game.game_type == "REG" else "POST"
        for team, direction in ((game.home_team, 1.0), (game.away_team, -1.0)):
            rows.append(
                {
                    "game_id": game.game_id,
                    "season": game.season,
                    "season_type": season_type,
                    "week": game.week,
                    "team": team,
                    "attempts": 30 + position % 7,
                    "carries": 22 + position % 5,
                    "passing_epa": direction * (2.0 + (position % 9) / 3.0),
                    "rushing_epa": direction * (1.0 + (position % 4) / 4.0),
                    "passing_cpoe": direction * (1.5 + (position % 6) / 2.0),
                    "passing_yards": 230 + direction * 12 + position % 11,
                    "rushing_yards": 100 + direction * 6 + position % 8,
                    "sacks_suffered": 2,
                    "passing_interceptions": int(direction < 0),
                    "sack_fumbles_lost": 0,
                    "rushing_fumbles_lost": 0,
                    "receiving_fumbles_lost": 0,
                }
            )
    return rows


def _schedule_and_stats(wild_card_margin: float = 49.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two full regular seasons plus one postseason bracket between them.

    The bracket is played in January/February of ``FIRST_SEASON + 1``, so the
    second regular season is chronologically after it: any playoff result that
    leaked into the regular-season build would be visible in the next season's
    rows.
    """

    rows: list[dict[str, object]] = []
    for season in (FIRST_SEASON, SECOND_SEASON):
        opener = SEASON_OPENERS[season]
        for week in range(1, REGULAR_SEASON_WEEKS + 1):
            gameday = opener + timedelta(days=7 * (week - 1))
            pairs = (("A", "B"), ("C", "D")) if week % 2 else (("B", "A"), ("D", "C"))
            for slot, (home, away) in enumerate(pairs, start=1):
                rows.append(
                    _schedule_row(
                        season=season,
                        game_type="REG",
                        week=week,
                        gameday=gameday,
                        home=home,
                        away=away,
                        margin=float(((week * 3 + slot * 5) % 17) - 8),
                        spread=float(((week + slot) % 7) - 3) + 0.5,
                    )
                )
    for position, (game_type, week, gameday, home, away, margin, spread) in enumerate(
        POSTSEASON_BRACKET
    ):
        rows.append(
            _schedule_row(
                season=FIRST_SEASON,
                game_type=game_type,
                week=week,
                gameday=gameday,
                home=home,
                away=away,
                margin=wild_card_margin if position == 0 else margin,
                spread=spread,
            )
        )
    schedules = pd.DataFrame(rows)
    return schedules, pd.DataFrame(_team_stat_rows(schedules))


def _elo_gap(features: pd.DataFrame, game_id: str, team: str) -> float:
    """The team's pregame Elo minus its opponent's, for one game.

    The published table exposes ``elo_diff`` (home Elo plus home-field minus
    away Elo) rather than the two raw ratings, so the team-oriented gap is
    recovered by removing the home-field term and orienting the sign.
    """

    row = features.loc[features["game_id"].eq(game_id)].iloc[0]
    home_field = 0.0 if int(row["neutral_site"]) else HOME_FIELD_ELO
    signed = float(row["elo_diff"]) - home_field
    return signed if row["home_team"] == team else -signed


def _schedule_rating_diff(features: pd.DataFrame, game_id: str) -> float:
    return float(features.loc[features["game_id"].eq(game_id), "schedule_rating_diff"].iloc[0])


def _team_season_elo(features: pd.DataFrame, team: str, season: int) -> pd.DataFrame:
    """One pregame Elo view per regular-season game the team played."""

    involved = features["home_team"].eq(team) | features["away_team"].eq(team)
    rows = features.loc[
        involved & features["season"].eq(season) & features["game_type"].eq("REG")
    ].sort_values("game_id")
    return rows.loc[:, ["game_id", "elo_diff", "elo_home_win_prob"]].reset_index(drop=True)


@pytest.fixture(scope="module")
def postseason_build() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(schedules, regular-only features, features including the bracket)."""

    schedules, stats = _schedule_and_stats()
    regular = build_game_features(schedules, stats, **BUILD_KWARGS)  # type: ignore[arg-type]
    combined = build_game_features(
        schedules,
        stats,
        include_postseason=True,
        **BUILD_KWARGS,  # type: ignore[arg-type]
    )
    return schedules, regular, combined


# ---------------------------------------------------------------------------
# 1. Regular-season rows are bit-identical either way
# ---------------------------------------------------------------------------


def test_regular_season_rows_are_bit_identical_with_postseason_included(
    postseason_build: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    _, regular, combined = postseason_build
    subset = combined.loc[combined["game_type"].eq("REG")].reset_index(drop=True)
    pd.testing.assert_frame_equal(regular.reset_index(drop=True), subset)
    # Both seasons are present, so the offseason carry-over paths (Elo reset,
    # team-state regression, graph decay) actually ran inside the comparison.
    assert sorted(regular["season"].unique()) == [FIRST_SEASON, SECOND_SEASON]
    assert regular["game_type"].eq("REG").all()


# ---------------------------------------------------------------------------
# 2. Postseason rows are present, labeled, and encoded
# ---------------------------------------------------------------------------


def test_postseason_rows_are_appended_and_encoded(
    postseason_build: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    schedules, regular, combined = postseason_build
    postseason = combined.loc[combined["game_type"].ne("REG")]

    assert len(combined) == len(regular) + len(POSTSEASON_BRACKET) == len(schedules)
    assert set(postseason["game_type"]) == set(POSTSEASON_GAME_TYPES)
    assert combined["game_id"].is_unique

    super_bowl = combined.loc[combined["game_id"].eq(SUPER_BOWL_GAME_ID)].iloc[0]
    assert super_bowl["neutral_site"] == 1
    assert regular["neutral_site"].eq(0).all()

    # Playoff weeks clamp to the top of the regular-season cycle rather than
    # wrapping the cyclic encoding back to September.
    week_18 = combined.loc[combined["game_id"].eq(WEEK_18_GAME_ID)].iloc[0]
    assert postseason["week"].min() == 19
    assert postseason["week_sin"].eq(week_18["week_sin"]).all()
    assert postseason["week_cos"].eq(week_18["week_cos"]).all()

    expected_margin = postseason["result"] - postseason["spread_line"]
    assert postseason["ats_margin"].eq(expected_margin).all()
    assert postseason["home_cover"].eq(expected_margin.gt(0).astype(float)).all()
    assert postseason["home_cover"].notna().all()
    assert combined.loc[combined["game_id"].eq(WILD_CARD_GAME_ID), "ats_margin"].iloc[
        0
    ] == pytest.approx(46.5)


# ---------------------------------------------------------------------------
# 3. Playoff results reach postseason rows only
# ---------------------------------------------------------------------------


def test_postseason_results_reach_postseason_rows_only(
    postseason_build: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    _, regular, blowout = postseason_build
    upset_schedules, upset_stats = _schedule_and_stats(wild_card_margin=UPSET_MARGIN)
    upset = build_game_features(
        upset_schedules,
        upset_stats,
        include_postseason=True,
        **BUILD_KWARGS,  # type: ignore[arg-type]
    )

    # Pass two carried the wild-card result forward into the divisional row:
    # the team's Elo standing there is neither its week-18 standing nor the
    # standing it would have had after losing that wild-card game instead.
    week_18_gap = _elo_gap(blowout, WEEK_18_GAME_ID, BRACKET_TEAM)
    blowout_gap = _elo_gap(blowout, DIVISIONAL_GAME_ID, BRACKET_TEAM)
    upset_gap = _elo_gap(upset, DIVISIONAL_GAME_ID, BRACKET_TEAM)
    assert blowout_gap != pytest.approx(week_18_gap)
    assert blowout_gap > upset_gap
    # The blowout's size (not just its winner) also reaches the later rounds,
    # through the margin-weighted schedule graph.
    assert _schedule_rating_diff(blowout, DIVISIONAL_GAME_ID) != pytest.approx(
        _schedule_rating_diff(upset, DIVISIONAL_GAME_ID)
    )

    # ...and nowhere else: every regular-season row is untouched by the change,
    # including the season played after the bracket.
    pd.testing.assert_frame_equal(
        blowout.loc[blowout["game_type"].eq("REG")].reset_index(drop=True),
        upset.loc[upset["game_type"].eq("REG")].reset_index(drop=True),
    )
    next_season_elo = _team_season_elo(regular, BRACKET_TEAM, SECOND_SEASON)
    assert len(next_season_elo) == REGULAR_SEASON_WEEKS
    pd.testing.assert_frame_equal(
        _team_season_elo(blowout, BRACKET_TEAM, SECOND_SEASON), next_season_elo
    )
    pd.testing.assert_frame_equal(
        _team_season_elo(upset, BRACKET_TEAM, SECOND_SEASON), next_season_elo
    )


# ---------------------------------------------------------------------------
# 4. The guard itself
# ---------------------------------------------------------------------------


def test_regular_season_rows_drops_postseason_and_passes_through() -> None:
    frame = pd.DataFrame(
        {
            "game_id": ["reg", "wc", "div", "con", "sb"],
            "game_type": ["REG", "WC", "DIV", "CON", "SB"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    assert set(POSTSEASON_GAME_TYPES) == {"WC", "DIV", "CON", "SB"}
    assert regular_season_rows(frame)["game_id"].tolist() == ["reg"]

    without_game_type = frame.drop(columns="game_type")
    pd.testing.assert_frame_equal(regular_season_rows(without_game_type), without_game_type)


# ---------------------------------------------------------------------------
# A model table with postseason rows the fitters must not see
# ---------------------------------------------------------------------------

FILL_COLUMNS = (*MODEL_FEATURE_COLUMNS, *GRAPH_FEATURE_COLUMNS)


def _model_frame_with_postseason() -> pd.DataFrame:
    """160 regular-season rows over two seasons plus nine postseason rows.

    The first season's bracket sits chronologically between the two regular
    seasons and carries wildly out-of-scale features and outcomes, so any
    training path that admitted it would move visibly. The second season's
    bracket is the latest block in the table, so a leak would also change every
    reported training cutoff.
    """

    specs: list[dict[str, object]] = []
    cursor = date(2019, 9, 1)

    def _append(season: int, week: int, game_type: str, count: int) -> None:
        nonlocal cursor
        for slot in range(count):
            specs.append(
                {
                    "game_id": f"{season}_{week:02d}_{game_type}_{slot}",
                    "season": season,
                    "week": week,
                    "game_type": game_type,
                    "gameday": cursor,
                }
            )
            cursor += timedelta(days=1)

    for week in range(1, 11):
        _append(2019, week, "REG", 10)
    for week, game_type in ((19, "WC"), (20, "DIV"), (21, "CON"), (22, "SB")):
        _append(2019, week, game_type, 1)
    for week in range(1, 5):
        _append(2020, week, "REG", 15)
    _append(2020, 19, "WC", 2)
    for week, game_type in ((20, "DIV"), (21, "CON"), (22, "SB")):
        _append(2020, week, game_type, 1)

    frame = pd.DataFrame(specs)
    frame["away_team"] = "AWY"
    frame["home_team"] = "HME"
    frame["home_spread_odds"] = -110.0
    frame["away_spread_odds"] = -110.0

    index = np.arange(len(frame))
    for feature_index, column in enumerate(FILL_COLUMNS, start=1):
        frame[column] = np.sin(index / feature_index) + (index % 5) / 10.0
    poison = frame["season"].eq(2019) & frame["game_type"].ne("REG")
    frame.loc[poison, list(FILL_COLUMNS)] += 100.0

    postseason = frame["game_type"].ne("REG")
    frame["spread_line"] = np.where(index % 2 == 0, 2.5, -2.5)
    frame["home_cover"] = (index % 3 != 0).astype(float)
    frame.loc[postseason, "home_cover"] = 1.0
    frame["ats_margin"] = np.where(frame["home_cover"].eq(1), 3.0, -3.0)
    frame.loc[postseason, "ats_margin"] = 250.0
    frame["result"] = frame["spread_line"] + frame["ats_margin"]
    return frame


@pytest.fixture
def postseason_model_frame() -> pd.DataFrame:
    return _model_frame_with_postseason()


# ---------------------------------------------------------------------------
# 5. Fitters are immune to postseason rows
# ---------------------------------------------------------------------------


def test_margin_and_market_fits_ignore_postseason_rows(
    postseason_model_frame: pd.DataFrame,
) -> None:
    frame = postseason_model_frame
    regular = frame.loc[frame["game_type"].eq("REG")]
    target = regular.tail(3)
    last_regular_gameday = max(regular["gameday"])

    for model_target in ("margin", "market_residual"):
        with_postseason = fit_margin_model(frame, target=model_target)  # type: ignore[arg-type]
        regular_only = fit_margin_model(regular, target=model_target)  # type: ignore[arg-type]
        assert with_postseason.training_rows == regular_only.training_rows == len(regular)
        assert with_postseason.training_max_gameday == regular_only.training_max_gameday
        assert with_postseason.training_max_gameday == last_regular_gameday.isoformat()
        assert np.array_equal(with_postseason.residuals, regular_only.residuals)
        pd.testing.assert_frame_equal(with_postseason.predict(target), regular_only.predict(target))

    market_with = fit_market_baseline(frame)
    market_regular = fit_market_baseline(regular)
    assert market_with.training_rows == market_regular.training_rows == len(regular)
    assert market_with.training_max_gameday == market_regular.training_max_gameday
    assert np.array_equal(market_with.residuals, market_regular.residuals)
    # A leaked 250-point ATS margin would be impossible to miss in the residuals.
    assert float(np.abs(market_with.residuals).max()) == pytest.approx(3.0)


def test_cover_model_fit_ignores_postseason_rows(postseason_model_frame: pd.DataFrame) -> None:
    frame = postseason_model_frame
    regular = frame.loc[frame["game_type"].eq("REG")]
    with_postseason = fit_cover_model(frame)
    regular_only = fit_cover_model(regular)

    assert with_postseason.training_rows == regular_only.training_rows == len(regular)
    assert with_postseason.training_max_gameday == regular_only.training_max_gameday
    assert with_postseason.calibration_rows == regular_only.calibration_rows
    assert np.array_equal(
        with_postseason.predict_home_cover(regular.tail(5)),
        regular_only.predict_home_cover(regular.tail(5)),
    )


# ---------------------------------------------------------------------------
# 6. Walk-forward evaluation is unchanged by appended postseason rows
# ---------------------------------------------------------------------------


def test_walk_forward_backtest_ignores_postseason_rows(
    postseason_model_frame: pd.DataFrame,
) -> None:
    frame = postseason_model_frame
    regular = frame.loc[frame["game_type"].eq("REG")]
    postseason_ids = set(frame.loc[frame["game_type"].ne("REG"), "game_id"])

    with_postseason = walk_forward_backtest(
        frame, start_season=2020, min_edge=0.0, min_train_games=80
    )
    regular_only = walk_forward_backtest(
        regular, start_season=2020, min_edge=0.0, min_train_games=80
    )

    assert set(with_postseason.predictions["game_id"]) == set(regular_only.predictions["game_id"])
    assert set(with_postseason.predictions["game_id"]).isdisjoint(postseason_ids)
    assert with_postseason.metrics == regular_only.metrics
    assert with_postseason.metrics["games_scored"] == 60
    pd.testing.assert_frame_equal(with_postseason.predictions, regular_only.predictions)


def test_walk_forward_outcomes_ignores_postseason_rows(
    postseason_model_frame: pd.DataFrame,
) -> None:
    frame = postseason_model_frame
    regular = frame.loc[frame["game_type"].eq("REG")]
    postseason_ids = set(frame.loc[frame["game_type"].ne("REG"), "game_id"])

    with_postseason = walk_forward_outcomes(frame, start_season=2020, min_train_games=80)
    regular_only = walk_forward_outcomes(regular, start_season=2020, min_train_games=80)

    assert set(with_postseason.predictions["game_id"]) == set(regular_only.predictions["game_id"])
    assert set(with_postseason.predictions["game_id"]).isdisjoint(postseason_ids)
    assert set(with_postseason.predictions["game_type"]) == {"REG"}
    pd.testing.assert_frame_equal(with_postseason.summary, regular_only.summary)
    pd.testing.assert_frame_equal(with_postseason.season_summary, regular_only.season_summary)


# ---------------------------------------------------------------------------
# 7. A playoff week can still be served
# ---------------------------------------------------------------------------


def test_outcome_week_scores_a_playoff_week_without_training_on_postseason(
    postseason_model_frame: pd.DataFrame,
) -> None:
    frame = postseason_model_frame
    playoff_ids = set(frame.loc[frame["season"].eq(2020) & frame["week"].eq(19), "game_id"])
    assert len(playoff_ids) == 2

    predictions = score_outcome_week(frame, season=2020, week=19, min_train_games=80)
    assert set(predictions["game_id"]) == playoff_ids
    assert set(predictions["game_type"]) == {"WC"}
    assert len(predictions) == len(playoff_ids) * len(OUTCOME_METHODS)

    # Training stopped at the last regular-season game, not the 2019 bracket
    # that sits chronologically between the two regular seasons. Every
    # regular-season row precedes the target week, so all of them -- and only
    # them -- are eligible.
    regular = frame.loc[frame["game_type"].eq("REG")]
    assert set(predictions["train_max_gameday"]) == {max(regular["gameday"]).isoformat()}
    assert set(predictions["train_rows"]) == {len(regular)}

    # Dropping the earlier postseason "poison" rows changes nothing, which is
    # only possible if they were never in the training set.
    without_poison = frame.loc[~(frame["season"].eq(2019) & frame["game_type"].ne("REG"))]
    clean = score_outcome_week(without_poison, season=2020, week=19, min_train_games=80)
    pd.testing.assert_frame_equal(predictions.reset_index(drop=True), clean.reset_index(drop=True))


def test_margin_predict_cli_preserves_postseason_round_in_artifact_metadata(
    postseason_model_frame: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """January serving keeps the safety-validated round visible at artifact level."""

    features = tmp_path / "postseason_features.parquet"
    postseason_model_frame.to_parquet(features, index=False)
    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("NFL_ATS_ARTIFACTS_DIR", str(artifacts))
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))

    assert (
        cli.main(
            [
                "margin-predict",
                "--features",
                str(features),
                "--feature-profile",
                "base",
                "--season",
                "2020",
                "--week",
                "19",
                "--min-train-games",
                "80",
                "--no-line-sweep",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    artifact = Path(output["artifact_directory"])
    metadata = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
    safety = json.loads((artifact / "prediction_safety.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(artifact / "predictions.csv")

    assert output["game_type"] == metadata["game_type"] == "WC"
    assert output["games"] == predictions["game_id"].nunique() == 2
    assert set(predictions["game_type"]) == {"WC"}
    assert safety["status"] == "PASS"
    assert "card_scope" in safety["checks_passed"]
    assert output["synchronization_status"] == "UNLINKED"
    assert not (artifacts / "active_ats_model.json").exists()


# ---------------------------------------------------------------------------
# 8. Weekly serving skips unplayed postseason rows
# ---------------------------------------------------------------------------


def test_upcoming_week_skips_unplayed_postseason_rows() -> None:
    frame = pd.DataFrame(
        {
            "game_id": ["2024_22_SB", "2024_21_CON", "2025_05_REG", "2025_06_REG"],
            "season": [2024, 2024, 2025, 2025],
            "week": [22, 21, 5, 6],
            "game_type": ["SB", "CON", "REG", "REG"],
            "result": [np.nan, np.nan, np.nan, np.nan],
        }
    )
    # Sorted by (season, week) the postseason rows come first; the guard keeps
    # them from claiming the upcoming week.
    assert upcoming_week(frame) == (2025, 5)

    only_postseason_unplayed = frame.copy()
    only_postseason_unplayed.loc[only_postseason_unplayed["game_type"].eq("REG"), "result"] = 3.0
    with pytest.raises(ValueError, match="no unplayed games"):
        upcoming_week(only_postseason_unplayed)
