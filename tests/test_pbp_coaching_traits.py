"""Phase 12 lane J: LEAD-26/27/30 PBP coaching-trait reliability tests.

Covers each trait builder on a tiny synthetic play-by-play frame, the
LEAD-30 opportunity filter's boundaries, the strictly-before-cutoff
(leakage) guarantee of every rolling team-week builder, and the split-half
reliability engine's math on a known frame (perfect correlation, and a
label-shuffle null that must destroy real structure).
"""

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


# ---------------------------------------------------------------------------
# LEAD-26: opening drive
# ---------------------------------------------------------------------------


def test_opening_drive_team_games_picks_min_fixed_drive_and_scores_td_and_epa() -> None:
    rows = [
        # Team A's OPENING drive (fixed_drive=1): 3 plays, ends in a TD.
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
        # Team A's SECOND drive (fixed_drive=3): must NOT be picked as "opening".
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
        # Team B's opening drive (fixed_drive=2): 2 plays, ends in a punt.
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
    # Game 1: 3-play TD drive (epa sum 6.0).
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
    # Game 2: 2-play punt drive (epa sum -1.0).
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
    # play-weighted: (6.0 + -1.0) / (3 + 2) == 1.0, NOT the mean-of-means (2.0 + -0.5)/2 = 0.75
    assert seasons.loc["A", "opening_drive_epa_per_play"] == pytest.approx(1.0)


def test_opening_drive_rolling_is_leak_safe() -> None:
    rows = []
    games = [
        ("G1", 1, (1.0, 2.0, 3.0), "Touchdown"),  # epa sum 6.0, 3 plays
        ("G2", 2, (-1.0, 0.0), "Punt"),  # epa sum -1.0, 2 plays
        ("G3", 3, (2.0, 2.0), "Touchdown"),  # epa sum 4.0, 2 plays
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

    # G2's rolling state uses ONLY G1 (td_rate 1/1=1.0, epa/play 6.0/3=2.0) --
    # NEVER G2's own -1.0/0-TD outcome.
    assert rolling.loc["G2", "rolling_opening_drive_td_rate"] == pytest.approx(1.0)
    assert rolling.loc["G2", "rolling_opening_drive_epa_per_play"] == pytest.approx(2.0)

    # G3's rolling state uses G1+G2 ONLY (td_rate 1/2=0.5, epa/play 5.0/5=1.0) --
    # NEVER G3's own Touchdown/4.0 outcome, which would otherwise pull it up.
    assert rolling.loc["G3", "rolling_opening_drive_td_rate"] == pytest.approx(0.5)
    assert rolling.loc["G3", "rolling_opening_drive_epa_per_play"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# LEAD-27: third-quarter point differential
# ---------------------------------------------------------------------------


def test_third_quarter_point_diff_sign_and_magnitude() -> None:
    # Home (A) leads 0-0 entering Q3, leads by 3 entering Q4: home_q3_diff = +3.
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
    # Away (B) has the ball at the start of both quarters; score_differential is
    # recorded from B's own perspective, so home's lead must be negated.
    # B trails by 7 entering Q3 (home leads by 7) and trails by 0 entering Q4
    # (home lead fell to 0 -- e.g. a defensive/ST score credited to the away
    # side) -- home_q3_diff = 0 - 7 = -7.
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
    # Team A is home in all three games; home_q3_diff = [3, -2, 5].
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
    # sanity: q3_point_diff per game is [3-0, 3-5, 4-(-1)] = [3, -2, 5]
    team_games = build_third_quarter_point_diff_team_games(pbp)
    team_a = team_games.loc[team_games["team"] == "A"].set_index("game_id")["q3_point_diff"]
    assert team_a.loc["G1"] == pytest.approx(3.0)
    assert team_a.loc["G2"] == pytest.approx(-2.0)
    assert team_a.loc["G3"] == pytest.approx(5.0)

    rolling = build_third_quarter_rolling(pbp)
    rolling_a = rolling.loc[rolling["team"] == "A"].set_index("game_id")
    assert np.isnan(rolling_a.loc["G1", "rolling_q3_point_diff"])
    assert rolling_a.loc["G2", "rolling_q3_point_diff"] == pytest.approx(3.0)
    # G3's rolling value averages G1+G2 (3, -2) == 0.5, NEVER G3's own +5.
    assert rolling_a.loc["G3", "rolling_q3_point_diff"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# LEAD-30: fourth-down aggressiveness
# ---------------------------------------------------------------------------


def test_fourth_down_opportunity_filter_boundaries() -> None:
    rows = [
        # eligible, go (boundary: ydstogo=3, yardline=30)
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
        # eligible, go (boundary: yardline=70)
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
        # NOT eligible: ydstogo=4 (> max)
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
        # NOT eligible: yardline=29 (< low)
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
        # NOT eligible: yardline=71 (> high)
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
        # NOT eligible: no_play (penalty-nullified)
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
        # NOT eligible: qb_kneel flag set even though play_type says run
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
        # NOT eligible: down != 4
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
        # eligible, NOT go (punt). ``play=0`` matches real nflverse data --
        # kicking plays are NOT flagged in the ``play`` indicator -- and is
        # the exact regression case for a 2026-09-05 bug where gating on
        # ``play == 1`` silently discarded every punt/FG, making go_for_it
        # come back constant at 1.0 on the real data run.
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
        # eligible, NOT go (field goal), same ``play=0`` regression case.
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
        # G1: 2 eligible, 1 go (go_count=1, eligible_count=2)
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
        # G2: 1 eligible, 0 go
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
        # G3: 1 eligible, 1 go
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
    # G3's rolling rate uses G1+G2 ONLY: (1 + 0) / (2 + 1) == 1/3, NEVER G3's own go.
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


# ---------------------------------------------------------------------------
# Reliability engine: split-half math on a known frame
# ---------------------------------------------------------------------------


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
    # odd weeks (1, 3) for A: mean(10, 30) = 20; even weeks (2, 4): mean(20, 40) = 30
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
    # A has two adjacent transitions: 2019->2020 and 2020->2021.
    assert pairs.loc[("A", 2019), "value_a"] == pytest.approx(1.0)
    assert pairs.loc[("A", 2019), "value_b"] == pytest.approx(2.0)
    assert pairs.loc[("A", 2020), "value_a"] == pytest.approx(2.0)
    assert pairs.loc[("A", 2020), "value_b"] == pytest.approx(3.0)
    # B has NO adjacent transition (2019, 2021 -- a gap year) so B is absent.
    assert "B" not in pairs.index.get_level_values("team")


def test_paired_split_half_reliability_perfect_correlation() -> None:
    # 12 team-season rows across 4 block seasons, value_a == value_b exactly.
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
    # Spearman-Brown of r=1.0 is (2*1)/(1+1) == 1.0.
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
    # 6 teams x 5 seasons; a real, strong team-level effect (base level per
    # team) plus small per-half noise, so pairing by team gives a strong real
    # correlation -- but shuffling WHICH team's b-half pairs with which
    # team's a-half, within each season, destroys that pairing and should
    # collapse the null distribution's mean toward zero.
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
    # The real, team-paired correlation is strong.
    assert result["pearson_r"] > 0.8
    # The label-shuffled null, which breaks team pairing within season,
    # must center much closer to zero than the real correlation does.
    assert abs(result["null_mean_r"]) < 0.35
    assert abs(result["null_mean_r"]) < result["pearson_r"]


# ---------------------------------------------------------------------------
# Wiring smoke tests
# ---------------------------------------------------------------------------


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
