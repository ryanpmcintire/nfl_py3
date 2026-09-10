from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.pbp import PBP_SNAPSHOT_COLUMNS
from nfl_ats.pbp_coaching_traits import (
    build_fourth_down_opportunities,
    build_fourth_down_rolling,
    build_fourth_down_team_games,
    build_fourth_down_team_seasons,
    build_odd_even_halves,
    build_opening_drive_rolling,
    build_opening_drive_team_games,
    build_opening_drive_team_seasons,
    build_season_to_season_pairs,
    build_third_quarter_point_diff_team_games,
    build_third_quarter_rolling,
    compute_trait_reliability,
    paired_split_half_reliability,
    run_all_trait_reliabilities,
)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys(PBP_SNAPSHOT_COLUMNS, np.nan)
    row.update(
        {
            "season_type": "REG",
            "home_team": "A",
            "away_team": "B",
            "down": 1,
            "ydstogo": 10,
            "yardline_100": 50,
            "qtr": 1,
            "play_type": "pass",
            "yards_gained": 6,
            "pass_attempt": 1,
            "rush_attempt": 0,
            "qb_dropback": 1,
            "qb_kneel": 0,
            "qb_spike": 0,
            "aborted_play": 0,
            "complete_pass": 1,
            "interception": 0,
            "fumble_lost": 0,
            "sack": 0,
            "qb_hit": 0,
            "touchdown": 0,
            "first_down": 0,
            "epa": 0.0,
            "success": 0,
            "wp": 0.5,
            "score_differential": 0,
            "penalty": 0,
            "penalty_yards": 0,
            "fixed_drive_result": "Punt",
            "posteam_score": 0,
            "posteam_score_post": 0,
            "play": 1,
        }
    )
    row.update(overrides)
    return row


def test_opening_drive_team_games_picks_min_fixed_drive_and_scores_td_and_epa() -> None:
    rows = [
        _row(
            game_id="G1",
            season=2022,
            week=1,
            play_id=1,
            posteam="A",
            defteam="B",
            fixed_drive=1,
            fixed_drive_result="Touchdown",
            epa=1.0,
        ),
        _row(
            game_id="G1",
            season=2022,
            week=1,
            play_id=2,
            posteam="A",
            defteam="B",
            fixed_drive=1,
            fixed_drive_result="Touchdown",
            epa=2.0,
        ),
        _row(
            game_id="G1",
            season=2022,
            week=1,
            play_id=3,
            posteam="A",
            defteam="B",
            fixed_drive=1,
            fixed_drive_result="Touchdown",
            epa=3.0,
        ),
        _row(
            game_id="G1",
            season=2022,
            week=1,
            play_id=7,
            posteam="A",
            defteam="B",
            fixed_drive=3,
            fixed_drive_result="Punt",
            epa=99.0,
        ),
        _row(
            game_id="G1",
            season=2022,
            week=1,
            play_id=4,
            posteam="B",
            defteam="A",
            fixed_drive=2,
            fixed_drive_result="Punt",
            epa=-1.0,
        ),
        _row(
            game_id="G1",
            season=2022,
            week=1,
            play_id=5,
            posteam="B",
            defteam="A",
            fixed_drive=2,
            fixed_drive_result="Punt",
            epa=0.0,
        ),
    ]
    pbp = pd.DataFrame(rows)
    team_games = build_opening_drive_team_games(pbp)
    team_games = team_games.set_index("team")

    assert team_games.loc["A", "opening_drive_td"] == 1.0
    assert team_games.loc["A", "opening_drive_plays"] == 3
    assert team_games.loc["A", "opening_drive_epa"] == pytest.approx(6.0)

    assert team_games.loc["B", "opening_drive_td"] == 0.0
    assert team_games.loc["B", "opening_drive_plays"] == 2
    assert team_games.loc["B", "opening_drive_epa"] == pytest.approx(-1.0)


def test_opening_drive_team_seasons_is_play_weighted() -> None:
    rows = []
    for play_id, epa in enumerate((1.0, 2.0, 3.0), start=1):
        rows.append(
            _row(
                game_id="G1",
                season=2022,
                week=1,
                play_id=play_id,
                posteam="A",
                defteam="B",
                fixed_drive=1,
                fixed_drive_result="Touchdown",
                epa=epa,
            )
        )
    for play_id, epa in enumerate((-0.5, -0.5), start=10):
        rows.append(
            _row(
                game_id="G2",
                season=2022,
                week=2,
                play_id=play_id,
                posteam="A",
                defteam="B",
                fixed_drive=1,
                fixed_drive_result="Punt",
                epa=epa,
            )
        )
    pbp = pd.DataFrame(rows)
    seasons = build_opening_drive_team_seasons(pbp).set_index("team")
    assert seasons.loc["A", "n_games"] == 2
    assert seasons.loc["A", "opening_drive_td_rate"] == pytest.approx(0.5)
    assert seasons.loc["A", "opening_drive_epa_per_play"] == pytest.approx(1.0)


def test_opening_drive_rolling_is_leak_safe() -> None:
    rows = []
    games = [
        ("G1", 1, (1.0, 2.0, 3.0), "Touchdown"),
        ("G2", 2, (-1.0, 0.0), "Punt"),
        ("G3", 3, (2.0, 2.0), "Touchdown"),
    ]
    play_id = 1
    for game_id, week, epas, result in games:
        for epa in epas:
            rows.append(
                _row(
                    game_id=game_id,
                    season=2022,
                    week=week,
                    play_id=play_id,
                    posteam="A",
                    defteam="B",
                    fixed_drive=1,
                    fixed_drive_result=result,
                    epa=epa,
                )
            )
            play_id += 1
    pbp = pd.DataFrame(rows)
    rolling = build_opening_drive_rolling(pbp).set_index("game_id")

    assert np.isnan(rolling.loc["G1", "rolling_opening_drive_td_rate"])
    assert np.isnan(rolling.loc["G1", "rolling_opening_drive_epa_per_play"])

    assert rolling.loc["G2", "rolling_opening_drive_td_rate"] == pytest.approx(1.0)
    assert rolling.loc["G2", "rolling_opening_drive_epa_per_play"] == pytest.approx(2.0)

    assert rolling.loc["G3", "rolling_opening_drive_td_rate"] == pytest.approx(0.5)
    assert rolling.loc["G3", "rolling_opening_drive_epa_per_play"] == pytest.approx(1.0)


def test_third_quarter_point_diff_sign_and_magnitude() -> None:
    rows = [
        _row(
            game_id="G1", season=2022, week=1, play_id=1, qtr=3, posteam="A", score_differential=0
        ),
        _row(
            game_id="G1", season=2022, week=1, play_id=2, qtr=4, posteam="A", score_differential=3
        ),
    ]
    pbp = pd.DataFrame(rows)
    team_games = build_third_quarter_point_diff_team_games(pbp).set_index("team")
    assert team_games.loc["A", "q3_point_diff"] == pytest.approx(3.0)
    assert team_games.loc["B", "q3_point_diff"] == pytest.approx(-3.0)


def test_third_quarter_point_diff_reads_defensive_scores_via_score_differential() -> None:
    rows = [
        _row(
            game_id="G1", season=2022, week=1, play_id=1, qtr=3, posteam="B", score_differential=-7
        ),
        _row(
            game_id="G1", season=2022, week=1, play_id=2, qtr=4, posteam="B", score_differential=0
        ),
    ]
    pbp = pd.DataFrame(rows)
    team_games = build_third_quarter_point_diff_team_games(pbp).set_index("team")
    assert team_games.loc["A", "q3_point_diff"] == pytest.approx(-7.0)
    assert team_games.loc["B", "q3_point_diff"] == pytest.approx(7.0)


def test_third_quarter_rolling_is_leak_safe() -> None:
    rows = []
    for game_id, week, q3_diff in (("G1", 1, 0.0), ("G2", 2, 5.0), ("G3", 3, -1.0)):
        rows.append(
            _row(
                game_id=game_id,
                season=2022,
                week=week,
                play_id=1,
                qtr=3,
                posteam="A",
                score_differential=q3_diff,
            )
        )
    for game_id, week, q4_diff in (("G1", 1, 3.0), ("G2", 2, 3.0), ("G3", 3, 4.0)):
        rows.append(
            _row(
                game_id=game_id,
                season=2022,
                week=week,
                play_id=2,
                qtr=4,
                posteam="A",
                score_differential=q4_diff,
            )
        )
    pbp = pd.DataFrame(rows)
    team_games = build_third_quarter_point_diff_team_games(pbp)
    team_a = team_games.loc[team_games["team"] == "A"].set_index("game_id")["q3_point_diff"]
    assert team_a.loc["G1"] == pytest.approx(3.0)
    assert team_a.loc["G2"] == pytest.approx(-2.0)
    assert team_a.loc["G3"] == pytest.approx(5.0)

    rolling = build_third_quarter_rolling(pbp)
    rolling_a = rolling.loc[rolling["team"] == "A"].set_index("game_id")
    assert np.isnan(rolling_a.loc["G1", "rolling_q3_point_diff"])
    assert rolling_a.loc["G2", "rolling_q3_point_diff"] == pytest.approx(3.0)
    assert rolling_a.loc["G3", "rolling_q3_point_diff"] == pytest.approx(0.5)


def test_fourth_down_opportunity_filter_boundaries() -> None:
    rows = [
        _row(
            game_id="G1",
            season=2022,
            week=1,
            play_id=1,
            posteam="A",
            down=4,
            ydstogo=3,
            yardline_100=30,
            play_type="run",
        ),
        _row(
            game_id="G1",
            season=2022,
            week=1,
            play_id=2,
            posteam="A",
            down=4,
            ydstogo=1,
            yardline_100=70,
            play_type="pass",
        ),
        _row(
            game_id="G1",
            season=2022,
            week=1,
            play_id=3,
            posteam="A",
            down=4,
            ydstogo=4,
            yardline_100=50,
            play_type="run",
        ),
        _row(
            game_id="G1",
            season=2022,
            week=1,
            play_id=4,
            posteam="A",
            down=4,
            ydstogo=1,
            yardline_100=29,
            play_type="run",
        ),
        _row(
            game_id="G1",
            season=2022,
            week=1,
            play_id=5,
            posteam="A",
            down=4,
            ydstogo=1,
            yardline_100=71,
            play_type="run",
        ),
        _row(
            game_id="G1",
            season=2022,
            week=1,
            play_id=6,
            posteam="A",
            down=4,
            ydstogo=2,
            yardline_100=50,
            play_type="no_play",
        ),
        _row(
            game_id="G1",
            season=2022,
            week=1,
            play_id=7,
            posteam="A",
            down=4,
            ydstogo=2,
            yardline_100=50,
            play_type="run",
            qb_kneel=1,
        ),
        _row(
            game_id="G1",
            season=2022,
            week=1,
            play_id=8,
            posteam="A",
            down=3,
            ydstogo=2,
            yardline_100=50,
            play_type="run",
        ),
        _row(
            game_id="G1",
            season=2022,
            week=1,
            play_id=9,
            posteam="A",
            down=4,
            ydstogo=2,
            yardline_100=50,
            play_type="punt",
            play=0,
        ),
        _row(
            game_id="G1",
            season=2022,
            week=1,
            play_id=10,
            posteam="A",
            down=4,
            ydstogo=2,
            yardline_100=50,
            play_type="field_goal",
            play=0,
        ),
    ]
    pbp = pd.DataFrame(rows)
    opportunities = build_fourth_down_opportunities(pbp)
    assert len(opportunities) == 4
    assert opportunities["go_for_it"].sum() == pytest.approx(2.0)
    assert (opportunities["go_for_it"] == 0.0).sum() == 2


def test_fourth_down_rolling_is_leak_safe() -> None:
    rows = [
        _row(
            game_id="G1",
            season=2022,
            week=1,
            play_id=1,
            posteam="A",
            down=4,
            ydstogo=2,
            yardline_100=50,
            play_type="run",
        ),
        _row(
            game_id="G1",
            season=2022,
            week=1,
            play_id=2,
            posteam="A",
            down=4,
            ydstogo=1,
            yardline_100=40,
            play_type="punt",
            play=0,
        ),
        _row(
            game_id="G2",
            season=2022,
            week=2,
            play_id=1,
            posteam="A",
            down=4,
            ydstogo=3,
            yardline_100=60,
            play_type="field_goal",
            play=0,
        ),
        _row(
            game_id="G3",
            season=2022,
            week=3,
            play_id=1,
            posteam="A",
            down=4,
            ydstogo=1,
            yardline_100=35,
            play_type="pass",
        ),
    ]
    pbp = pd.DataFrame(rows)
    team_games = build_fourth_down_team_games(pbp).set_index("game_id")
    assert team_games.loc["G1", "go_count"] == 1
    assert team_games.loc["G1", "eligible_count"] == 2

    rolling = build_fourth_down_rolling(pbp).set_index("game_id")
    assert np.isnan(rolling.loc["G1", "rolling_fourth_down_go_rate"])
    assert rolling.loc["G2", "rolling_fourth_down_go_rate"] == pytest.approx(0.5)
    assert rolling.loc["G3", "rolling_fourth_down_go_rate"] == pytest.approx(1.0 / 3.0)


def test_fourth_down_team_seasons_aggregates_across_games() -> None:
    rows = [
        _row(
            game_id="G1",
            season=2022,
            week=1,
            play_id=1,
            posteam="A",
            down=4,
            ydstogo=2,
            yardline_100=50,
            play_type="run",
        ),
        _row(
            game_id="G2",
            season=2022,
            week=2,
            play_id=1,
            posteam="A",
            down=4,
            ydstogo=1,
            yardline_100=40,
            play_type="punt",
            play=0,
        ),
    ]
    pbp = pd.DataFrame(rows)
    seasons = build_fourth_down_team_seasons(pbp).set_index("team")
    assert seasons.loc["A", "go_count"] == 1
    assert seasons.loc["A", "eligible_count"] == 2
    assert seasons.loc["A", "fourth_down_go_rate"] == pytest.approx(0.5)


def test_require_columns_raises_on_malformed_pbp() -> None:
    with pytest.raises(DataContractError):
        build_fourth_down_opportunities(pd.DataFrame({"game_id": ["G1"]}))


def test_build_odd_even_halves_known_frame() -> None:
    long = pd.DataFrame(
        {
            "team": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "season": [2022] * 8,
            "week": [1, 2, 3, 4, 1, 2, 3, 4],
            "value": [10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    halves = build_odd_even_halves(long, "value", min_per_half=1).set_index("team")
    assert halves.loc["A", "value_a"] == pytest.approx(20.0)
    assert halves.loc["A", "value_b"] == pytest.approx(30.0)
    assert halves.loc["B", "value_a"] == pytest.approx(2.0)
    assert halves.loc["B", "value_b"] == pytest.approx(3.0)


def test_build_odd_even_halves_respects_min_per_half() -> None:
    long = pd.DataFrame(
        {"team": ["A", "A"], "season": [2022, 2022], "week": [1, 2], "value": [1.0, 2.0]}
    )
    halves = build_odd_even_halves(long, "value", min_per_half=2)
    assert halves.empty


def test_build_season_to_season_pairs_only_joins_adjacent_seasons() -> None:
    team_season = pd.DataFrame(
        {
            "team": ["A", "A", "A", "B", "B"],
            "season": [2019, 2020, 2021, 2019, 2021],
            "value": [1.0, 2.0, 3.0, 5.0, 9.0],
        }
    )
    pairs = build_season_to_season_pairs(team_season, "value").set_index(["team", "season"])
    assert pairs.loc[("A", 2019), "value_a"] == pytest.approx(1.0)
    assert pairs.loc[("A", 2019), "value_b"] == pytest.approx(2.0)
    assert pairs.loc[("A", 2020), "value_a"] == pytest.approx(2.0)
    assert pairs.loc[("A", 2020), "value_b"] == pytest.approx(3.0)
    assert "B" not in pairs.index.get_level_values("team")


def test_paired_split_half_reliability_perfect_correlation() -> None:
    pairs = pd.DataFrame(
        {
            "team": [f"T{i}" for i in range(12)],
            "season": [2019 + (i % 4) for i in range(12)],
            "block_season": [2019 + (i % 4) for i in range(12)],
            "value_a": [float(i) for i in range(12)],
            "value_b": [float(i) for i in range(12)],
        }
    )
    result = paired_split_half_reliability(
        pairs, metric="known", method="test", seed=1, n_boot=200, n_null=200, spearman_brown=True
    )
    assert result["status"] == "measured"
    assert result["pearson_r"] == pytest.approx(1.0)
    assert result["spearman_rho"] == pytest.approx(1.0)
    assert result["pearson_probability_positive"] == pytest.approx(1.0)
    assert result["spearman_brown_full_length_reliability"] == pytest.approx(1.0)
    assert result["pearson_r_ci95"][0] == pytest.approx(1.0)
    assert result["pearson_r_ci95"][1] == pytest.approx(1.0)


def test_paired_split_half_reliability_insufficient_units_returns_nan() -> None:
    pairs = pd.DataFrame(
        {
            "team": ["A", "B"],
            "season": [2020, 2020],
            "block_season": [2020, 2020],
            "value_a": [1.0, 2.0],
            "value_b": [1.0, 2.0],
        }
    )
    result = paired_split_half_reliability(
        pairs, metric="known", method="test", seed=1, n_boot=50, n_null=50, spearman_brown=True
    )
    assert result["status"] == "insufficient_units"
    assert np.isnan(result["pearson_r"])
    assert result["spearman_brown_full_length_reliability"] is None


def test_null_shuffle_destroys_real_structure_but_leaves_real_r_intact() -> None:
    rng = np.random.default_rng(20260905)
    teams = [f"T{i}" for i in range(6)]
    seasons = list(range(2018, 2023))
    rows = []
    for season in seasons:
        for i, team in enumerate(teams):
            base = float(i)
            rows.append(
                {
                    "team": team,
                    "season": season,
                    "block_season": season,
                    "value_a": base + rng.normal(0, 0.15),
                    "value_b": base + rng.normal(0, 0.15),
                }
            )
    pairs = pd.DataFrame(rows)
    result = paired_split_half_reliability(
        pairs, metric="known", method="test", seed=7, n_boot=300, n_null=500, spearman_brown=True
    )
    assert result["status"] == "measured"
    assert result["pearson_r"] > 0.8
    assert abs(result["null_mean_r"]) < 0.35
    assert abs(result["null_mean_r"]) < result["pearson_r"]


def test_compute_trait_reliability_returns_both_methods() -> None:
    long = pd.DataFrame(
        {
            "team": ["A", "A", "A", "A", "B", "B", "B", "B"] * 2,
            "season": [2021] * 8 + [2022] * 8,
            "week": [1, 2, 3, 4] * 4,
            "value": [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0] * 2,
        }
    )
    team_season = long.groupby(["team", "season"])["value"].mean().reset_index()
    result = compute_trait_reliability(long, team_season, metric="known", n_boot=100, n_null=100)
    assert set(result) == {
        "metric",
        "within_season_odd_even_week",
        "season_to_season_same_franchise",
    }
    assert result["within_season_odd_even_week"]["method"] == "within_season_odd_even_week"
    assert result["season_to_season_same_franchise"]["method"] == "season_to_season_same_franchise"


def test_run_all_trait_reliabilities_smoke() -> None:
    rows = []
    play_id = 1
    for season in (2021, 2022):
        for week in range(1, 4):
            game_id = f"{season}_{week:02d}_A_B"
            rows.append(
                _row(
                    game_id=game_id,
                    season=season,
                    week=week,
                    play_id=play_id,
                    posteam="A",
                    defteam="B",
                    fixed_drive=1,
                    fixed_drive_result="Touchdown",
                    epa=1.0,
                )
            )
            play_id += 1
            rows.append(
                _row(
                    game_id=game_id,
                    season=season,
                    week=week,
                    play_id=play_id,
                    qtr=3,
                    posteam="A",
                    score_differential=0.0,
                )
            )
            play_id += 1
            rows.append(
                _row(
                    game_id=game_id,
                    season=season,
                    week=week,
                    play_id=play_id,
                    qtr=4,
                    posteam="A",
                    score_differential=3.0,
                )
            )
            play_id += 1
            rows.append(
                _row(
                    game_id=game_id,
                    season=season,
                    week=week,
                    play_id=play_id,
                    posteam="A",
                    down=4,
                    ydstogo=2,
                    yardline_100=50,
                    play_type="run",
                )
            )
            play_id += 1
    pbp = pd.DataFrame(rows)
    results = run_all_trait_reliabilities(pbp, n_boot=50, n_null=50)
    assert set(results) == {
        "opening_drive_td_rate",
        "opening_drive_epa_per_play",
        "q3_point_diff",
        "fourth_down_go_rate",
    }
    for payload in results.values():
        assert "within_season_odd_even_week" in payload
        assert "season_to_season_same_franchise" in payload
        assert payload["within_season_odd_even_week"]["status"] in {
            "measured",
            "insufficient_units",
        }
