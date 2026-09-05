from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats.cfb_role_features import (
    CFB_ROLE_FEATURE_COLUMNS,
    CONTINUITY_NEUTRAL,
    FROZEN_STREAK_CAP,
    ROLE_BENCHMARK_BASELINE_ARM,
    ROLE_BENCHMARK_CANDIDATE_ARM,
    absence_separation_study,
    attach_role_continuity,
    build_role_continuity,
    cfb_role_benchmark,
)
from nfl_ats.data import DataContractError

# ---------------------------------------------------------------------------
# Fixtures: one team's dropback history spanning a QB change and a season break
# ---------------------------------------------------------------------------


def _dropback_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Team A: P1 starts G1-G3, then P2 takes over G4-G8; P2 returns in 2023.

    P1 qualifies (3 appearances, share 1.0) entering G4 and then never
    appears again. P2 accumulates appearances from G4 and qualifies entering
    G7. The 2023 season opens with nobody yet appeared that season.
    """

    games = [
        ("G1", 2022, 1),
        ("G2", 2022, 2),
        ("G3", 2022, 3),
        ("G4", 2022, 4),
        ("G5", 2022, 5),
        ("G6", 2022, 6),
        ("G7", 2022, 7),
        ("G8", 2022, 8),
        ("H1", 2023, 1),
        ("H2", 2023, 2),
    ]
    appearances = {
        "P1": {"G1", "G2", "G3"},
        "P2": {"G4", "G5", "G6", "G7", "G8", "H1", "H2"},
    }
    action_rows = []
    team_game_rows = []
    for order, (game_id, season, week) in enumerate(games, start=1):
        team_game_rows.append(
            {
                "game_id": game_id,
                "season": season,
                "week": week,
                "order_key": order,
                "team": "A",
                "action_type": "dropback",
                "team_total": 20.0,
            }
        )
        for player, played in appearances.items():
            if game_id in played:
                action_rows.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "week": week,
                        "order_key": order,
                        "team": "A",
                        "player_id": player,
                        "action_type": "dropback",
                        "count": 20.0,
                        "team_total": 20.0,
                    }
                )
    return pd.DataFrame(action_rows), pd.DataFrame(team_game_rows)


# ---------------------------------------------------------------------------
# 1. build_role_continuity
# ---------------------------------------------------------------------------


def test_role_continuity_hand_computed_sequence() -> None:
    actions, team_games = _dropback_history()
    continuity = build_role_continuity(actions, team_games)
    by_game = continuity.set_index("game_id")["continuity"]

    # Nobody qualified yet: neutral.
    assert by_game.loc["G1"] == pytest.approx(CONTINUITY_NEUTRAL)
    assert by_game.loc["G2"] == pytest.approx(CONTINUITY_NEUTRAL)
    assert by_game.loc["G3"] == pytest.approx(CONTINUITY_NEUTRAL)
    # G4: P1 qualified (3 appearances, state 1.0) and appeared in G3.
    assert by_game.loc["G4"] == pytest.approx(1.0)
    # G5-G6: P1 is the only qualified holder and missed the previous game.
    assert by_game.loc["G5"] == pytest.approx(0.0)
    assert by_game.loc["G6"] == pytest.approx(0.0)
    # G7: P2 now qualified too (state 1.0, streak 0); P1 still inside the
    # streak cap with state 1.0 -> continuity is the appeared half of mass.
    assert by_game.loc["G7"] == pytest.approx(0.5)
    # G8: P1 has now missed FROZEN_STREAK_CAP straight valid games and leaves
    # the mass; P2 alone remains and appeared last game.
    assert FROZEN_STREAK_CAP == 4
    assert by_game.loc["G8"] == pytest.approx(1.0)
    # 2023 opener: nobody has appeared this season yet -> neutral.
    assert by_game.loc["H1"] == pytest.approx(CONTINUITY_NEUTRAL)
    # After P2's first 2023 appearance the mass is active again.
    assert by_game.loc["H2"] == pytest.approx(1.0)


def test_role_continuity_ignores_receptions() -> None:
    actions, team_games = _dropback_history()
    actions["action_type"] = "reception"
    team_games["action_type"] = "reception"
    assert build_role_continuity(actions, team_games).empty


# ---------------------------------------------------------------------------
# 2. absence_separation_study
# ---------------------------------------------------------------------------


def test_absence_separation_labels_departure_and_temporary() -> None:
    actions, team_games = _dropback_history()

    # Add a carry role holder with a one-game temporary absence: P3 carries
    # in G1-G3, misses G4, returns G5-G8.
    carry_games = {"G1", "G2", "G3", "G5", "G6", "G7", "G8"}
    carry_actions = []
    carry_team_games = []
    order_by_game = {row["game_id"]: row["order_key"] for _, row in team_games.iterrows()}
    for game_id, season, week in [
        ("G1", 2022, 1),
        ("G2", 2022, 2),
        ("G3", 2022, 3),
        ("G4", 2022, 4),
        ("G5", 2022, 5),
        ("G6", 2022, 6),
        ("G7", 2022, 7),
        ("G8", 2022, 8),
    ]:
        carry_team_games.append(
            {
                "game_id": game_id,
                "season": season,
                "week": week,
                "order_key": order_by_game[game_id],
                "team": "A",
                "action_type": "carry",
                "team_total": 20.0,
            }
        )
        if game_id in carry_games:
            carry_actions.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "order_key": order_by_game[game_id],
                    "team": "A",
                    "player_id": "P3",
                    "action_type": "carry",
                    "count": 10.0,
                    "team_total": 20.0,
                }
            )
    actions = pd.concat([actions, pd.DataFrame(carry_actions)], ignore_index=True)
    team_games = pd.concat([team_games, pd.DataFrame(carry_team_games)], ignore_index=True)

    study = absence_separation_study(actions, team_games)
    episodes = study["episodes"]

    p1 = episodes.loc[episodes["player_id"].eq("P1")].iloc[0]
    assert p1["action_type"] == "dropback"
    assert p1["start_season"] == 2022 and p1["start_week"] == 4
    # P1 misses G4-G8 (five valid 2022 games) and never reappears; the 2023
    # games are also valid team-games, so the open episode kept counting.
    assert p1["length_valid_games"] == 7
    assert bool(p1["never_reappeared"])
    assert not bool(p1["reappeared_same_season"])

    p3 = episodes.loc[episodes["player_id"].eq("P3")].iloc[0]
    assert p3["action_type"] == "carry"
    assert p3["length_valid_games"] == 1
    assert bool(p3["reappeared_same_season"])

    carryover = study["carryover"]
    dropback_2022 = carryover.loc[
        carryover["season"].eq(2022) & carryover["action_type"].eq("dropback")
    ].set_index("player_id")
    assert bool(dropback_2022.loc["P2", "appeared_next_season"])
    assert not bool(dropback_2022.loc["P1", "appeared_next_season"])
    assert bool(dropback_2022["next_season_observed"].all())
    # 2023 is the final observed season: transitions out of it are censored.
    final_season = carryover.loc[carryover["season"].eq(2023)]
    assert not final_season.empty
    assert not final_season["next_season_observed"].any()

    episode_summary = study["episode_summary"]
    dropback_k1 = episode_summary.loc[
        episode_summary["action_type"].eq("dropback") & episode_summary["reached_length"].eq(1)
    ].iloc[0]
    assert dropback_k1["episodes"] == 1
    assert dropback_k1["reappeared_same_season_rate"] == pytest.approx(0.0)
    carry_k1 = episode_summary.loc[
        episode_summary["action_type"].eq("carry") & episode_summary["reached_length"].eq(1)
    ].iloc[0]
    assert carry_k1["reappeared_same_season_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 3. attach_role_continuity
# ---------------------------------------------------------------------------


def test_attach_role_continuity_imputes_neutral() -> None:
    # Canonical sides carry ids, not pbp display names; the pbp name "A Team"
    # maps to id 11 via the team-id frame (never by name).
    canonical = pd.DataFrame(
        {
            "game_id": ["G1", "G2"],
            "home_id": [11, 31],
            "away_id": [21, 41],
        }
    )
    continuity = pd.DataFrame(
        {
            "game_id": ["G1"],
            "season": [2022],
            "week": [1],
            "team": ["A Team"],
            "action_type": ["dropback"],
            "continuity": [0.4],
            "active_mass": [1.0],
            "absent_mass": [0.6],
            "qualified_players": [2],
        }
    )
    team_ids = pd.DataFrame({"game_id": ["G1"], "team": ["A Team"], "team_id": [11.0]})
    result = attach_role_continuity(canonical, continuity, team_ids)
    for column in CFB_ROLE_FEATURE_COLUMNS:
        assert column in result.columns
    first = result.set_index("game_id").loc["G1"]
    assert first["home_dropback_continuity"] == pytest.approx(0.4)
    assert first["away_dropback_continuity"] == pytest.approx(CONTINUITY_NEUTRAL)
    assert first["diff_dropback_continuity"] == pytest.approx(0.4 - CONTINUITY_NEUTRAL)
    assert first["home_carry_continuity"] == pytest.approx(CONTINUITY_NEUTRAL)
    assert first["diff_carry_continuity"] == pytest.approx(0.0)
    second = result.set_index("game_id").loc["G2"]
    assert second["home_dropback_continuity"] == pytest.approx(CONTINUITY_NEUTRAL)
    assert second["diff_dropback_continuity"] == pytest.approx(0.0)


def test_attach_role_continuity_requires_columns() -> None:
    with pytest.raises(DataContractError, match="missing"):
        attach_role_continuity(pd.DataFrame({"game_id": []}), pd.DataFrame(), pd.DataFrame())


def test_attach_role_continuity_rejects_unmapped_teams() -> None:
    canonical = pd.DataFrame({"game_id": ["G1"], "home_id": [11], "away_id": [21]})
    continuity = pd.DataFrame(
        {
            "game_id": ["G1"],
            "season": [2022],
            "week": [1],
            "team": ["Unmapped Team"],
            "action_type": ["dropback"],
            "continuity": [0.4],
            "active_mass": [1.0],
            "absent_mass": [0.6],
            "qualified_players": [2],
        }
    )
    team_ids = pd.DataFrame({"game_id": ["G1"], "team": ["Other Team"], "team_id": [11]})
    with pytest.raises(DataContractError, match="Unmapped Team"):
        attach_role_continuity(canonical, continuity, team_ids)


# ---------------------------------------------------------------------------
# 4. The three-arm benchmark run
# ---------------------------------------------------------------------------


@pytest.mark.full  # ENG-11: dominates --durations; full CFB benchmark fit
def test_cfb_role_benchmark_three_matched_arms(cfb_features_frame: pd.DataFrame) -> None:
    features = cfb_features_frame.copy()
    generator = np.random.default_rng(7)
    for action_type in ("dropback", "carry"):
        home = generator.uniform(0.6, 1.0, size=len(features))
        away = generator.uniform(0.6, 1.0, size=len(features))
        features[f"home_{action_type}_continuity"] = home
        features[f"away_{action_type}_continuity"] = away
        features[f"diff_{action_type}_continuity"] = home - away

    result = cfb_role_benchmark(
        features,
        start_season=2014,
        end_season=2014,
        min_train_games=50,
        bootstrap_samples=25,
    )
    methods = set(result.predictions["method"])
    assert methods == {"market", "market_residual", "market_residual_roles"}

    # Identical weeks: every arm scored exactly the same games.
    per_method = result.predictions.groupby("method")["game_id"].apply(set)
    assert per_method["market_residual_roles"] == per_method["market_residual"]

    paired = result.paired
    assert set(paired["block"]) == {"week", "season"}
    assert set(paired["candidate_feature_set"]) == {ROLE_BENCHMARK_CANDIDATE_ARM}
    assert set(paired["baseline_feature_set"]) == {ROLE_BENCHMARK_BASELINE_ARM}
    assert {"accuracy_improvement", "brier_improvement", "log_loss_improvement"}.issubset(
        set(paired["metric"])
    )
    assert (paired["paired_games"] > 0).all()


def test_cfb_role_benchmark_requires_role_columns(cfb_features_frame: pd.DataFrame) -> None:
    with pytest.raises(DataContractError, match="missing columns"):
        cfb_role_benchmark(cfb_features_frame)


def test_cfb_role_benchmark_rejects_constant_role_columns(
    cfb_features_frame: pd.DataFrame,
) -> None:
    # The failure mode that voided the first real run: an all-neutral join
    # makes the candidate arm identical to the baseline. Must fail closed.
    features = cfb_features_frame.copy()
    for column in CFB_ROLE_FEATURE_COLUMNS:
        features[column] = CONTINUITY_NEUTRAL if not column.startswith("diff_") else 0.0
    with pytest.raises(DataContractError, match="constant"):
        cfb_role_benchmark(features)
