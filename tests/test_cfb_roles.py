from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from nfl_ats.cfb_roles import (
    ABSENCE_COLUMNS,
    FROZEN_MIN_PRIOR_APPEARANCES,
    FROZEN_MIN_TEAM_ACTIONS,
    FROZEN_ROLE_SPAN,
    FROZEN_ROLE_THRESHOLDS,
    ROLE_ACTION_COLUMNS,
    build_absence_frame,
    build_delivery_frame,
    build_role_states,
    cfb_role_actions,
    evaluate_replication_gates,
    nfl_role_actions,
    run_role_replication,
    summarize_delivery,
)

_ALPHA = 2.0 / (FROZEN_ROLE_SPAN + 1.0)  # 2/9, matching FROZEN_ROLE_SPAN=8


def _action_row(
    *,
    game_id: str,
    season: int,
    week: int,
    order_key: int,
    team: str,
    player_id: str,
    action_type: str,
    count: float,
    team_total: float,
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "order_key": order_key,
        "team": team,
        "player_id": player_id,
        "action_type": action_type,
        "count": count,
        "team_total": team_total,
    }


# ---------------------------------------------------------------------------
# 1. build_role_states: share/prior_share/prior_appearances EWM mirror
# ---------------------------------------------------------------------------


def test_build_role_states_hand_computed_ewm_sequence() -> None:
    actions = pd.DataFrame(
        [
            _action_row(
                game_id="G1",
                season=2022,
                week=1,
                order_key=1,
                team="A",
                player_id="P1",
                action_type="dropback",
                count=20,
                team_total=40,
            ),
            _action_row(
                game_id="G2",
                season=2022,
                week=2,
                order_key=2,
                team="A",
                player_id="P1",
                action_type="dropback",
                count=25,
                team_total=40,
            ),
            _action_row(
                game_id="G3",
                season=2022,
                week=3,
                order_key=3,
                team="A",
                player_id="P1",
                action_type="dropback",
                count=30,
                team_total=40,
            ),
            # Same player, different team: state must start fresh (team-keyed).
            _action_row(
                game_id="G4",
                season=2022,
                week=1,
                order_key=1,
                team="B",
                player_id="P1",
                action_type="dropback",
                count=5,
                team_total=10,
            ),
        ]
    )
    states = build_role_states(actions)
    by_game = states.set_index("game_id")

    share1, share2, share3 = 0.5, 0.625, 0.75
    expected_state_after_g1 = share1
    expected_state_after_g2 = _ALPHA * share2 + (1.0 - _ALPHA) * expected_state_after_g1
    expected_state_after_g3 = _ALPHA * share3 + (1.0 - _ALPHA) * expected_state_after_g2

    assert by_game.loc["G1", "share"] == pytest.approx(share1)
    assert math.isnan(by_game.loc["G1", "prior_share"])
    assert by_game.loc["G1", "prior_appearances"] == 0

    assert by_game.loc["G2", "share"] == pytest.approx(share2)
    assert by_game.loc["G2", "prior_share"] == pytest.approx(expected_state_after_g1)
    assert by_game.loc["G2", "prior_appearances"] == 1

    assert by_game.loc["G3", "share"] == pytest.approx(share3)
    assert by_game.loc["G3", "prior_share"] == pytest.approx(expected_state_after_g2)
    assert by_game.loc["G3", "prior_appearances"] == 2

    # Team B's row for the same player has no prior state of its own.
    assert math.isnan(by_game.loc["G4", "prior_share"])
    assert by_game.loc["G4", "prior_appearances"] == 0

    assert list(states.columns) == [
        *ROLE_ACTION_COLUMNS,
        "share",
        "prior_share",
        "prior_appearances",
    ]
    # Sanity: expected_state_after_g3 is only used to seed the delivery test below.
    assert expected_state_after_g3 > expected_state_after_g2


def test_build_role_states_rejects_zero_count_rows() -> None:
    actions = pd.DataFrame(
        [
            _action_row(
                game_id="G1",
                season=2022,
                week=1,
                order_key=1,
                team="A",
                player_id="P1",
                action_type="dropback",
                count=0,
                team_total=10,
            )
        ]
    )
    from nfl_ats.data import DataContractError

    with pytest.raises(DataContractError):
        build_role_states(actions)


# ---------------------------------------------------------------------------
# 2. build_delivery_frame: threshold / min-prior-appearances gating + ratio
# ---------------------------------------------------------------------------


def _states_row(
    *, action_type: str, share: float, prior_share: float, prior_appearances: int
) -> dict[str, object]:
    return {
        "game_id": "G",
        "season": 2022,
        "week": 1,
        "order_key": 1,
        "team": "A",
        "player_id": "P",
        "action_type": action_type,
        "count": 1,
        "team_total": 1,
        "share": share,
        "prior_share": prior_share,
        "prior_appearances": prior_appearances,
    }


def test_build_delivery_frame_filters_and_computes_ratio() -> None:
    states = pd.DataFrame(
        [
            # Qualifies: prior_share >= 0.50, prior_appearances >= 3.
            {
                **_states_row(
                    action_type="dropback", share=0.5, prior_share=0.6, prior_appearances=3
                )
            },
            # Fails: prior_appearances below the minimum.
            {
                **_states_row(
                    action_type="dropback", share=0.5, prior_share=0.6, prior_appearances=2
                )
            },
            # Fails: prior_share below the dropback threshold.
            {
                **_states_row(
                    action_type="dropback", share=0.5, prior_share=0.4, prior_appearances=5
                )
            },
            # Qualifies: carry threshold is 0.20.
            {**_states_row(action_type="carry", share=0.3, prior_share=0.25, prior_appearances=4)},
            # Fails: no prior appearance at all (NaN prior_share).
            {
                **_states_row(
                    action_type="carry", share=0.3, prior_share=math.nan, prior_appearances=0
                )
            },
        ]
    )
    delivery = build_delivery_frame(states)
    assert len(delivery) == 2
    dropback_row = delivery.loc[delivery["action_type"].eq("dropback")].iloc[0]
    assert dropback_row["ratio"] == pytest.approx(0.5 / 0.6)
    carry_row = delivery.loc[delivery["action_type"].eq("carry")].iloc[0]
    assert carry_row["ratio"] == pytest.approx(0.3 / 0.25)


def test_summarize_delivery_aggregates_per_action_type() -> None:
    states = pd.DataFrame(
        [
            {
                **_states_row(
                    action_type="dropback", share=0.5, prior_share=0.5, prior_appearances=3
                )
            },
            {
                **_states_row(
                    action_type="dropback", share=0.25, prior_share=0.5, prior_appearances=4
                )
            },
        ]
    )
    delivery = build_delivery_frame(states)
    summary = summarize_delivery(delivery)
    row = summary.loc[summary["action_type"].eq("dropback")].iloc[0]
    assert row["n"] == 2
    assert row["median_ratio"] == pytest.approx(0.75)  # median of ratios 1.0 and 0.5
    assert row["fraction_at_or_above_one"] == pytest.approx(0.5)
    assert row["fraction_severe_under"] == pytest.approx(0.5)  # the 0.5 ratio row


# ---------------------------------------------------------------------------
# 3. build_absence_frame: proxy absence events, state frozen during a streak
# ---------------------------------------------------------------------------


def test_build_absence_frame_detects_streak_without_updating_state() -> None:
    # Team A, action_type "carry": P1 appears G1-G3, is absent G4-G5; P2
    # appears every game (also the top replacement while P1 is absent).
    rows = []
    p1_counts = {"G1": 10, "G2": 12, "G3": 11}
    p2_counts = {"G1": 5, "G2": 4, "G3": 5, "G4": 8, "G5": 7}
    order_keys = {"G1": 1, "G2": 2, "G3": 3, "G4": 4, "G5": 5}
    for game_id, count in p1_counts.items():
        rows.append(
            _action_row(
                game_id=game_id,
                season=2022,
                week=order_keys[game_id],
                order_key=order_keys[game_id],
                team="A",
                player_id="P1",
                action_type="carry",
                count=count,
                team_total=20,
            )
        )
    for game_id, count in p2_counts.items():
        rows.append(
            _action_row(
                game_id=game_id,
                season=2022,
                week=order_keys[game_id],
                order_key=order_keys[game_id],
                team="A",
                player_id="P2",
                action_type="carry",
                count=count,
                team_total=20,
            )
        )
    actions = pd.DataFrame(rows)
    states = build_role_states(actions)

    team_games = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "season": 2022,
                "week": order_keys[game_id],
                "order_key": order_keys[game_id],
                "team": "A",
                "action_type": "carry",
                "team_total": 20,
            }
            for game_id in ("G1", "G2", "G3", "G4", "G5")
        ]
    )

    absences = build_absence_frame(team_games, states)
    assert list(absences.columns) == list(ABSENCE_COLUMNS)
    p1_absences = absences.loc[absences["player_id"].eq("P1")].sort_values("game_id")
    assert p1_absences["game_id"].tolist() == ["G4", "G5"]

    # The state is frozen during the absence streak: identical prior_share
    # (and hence identical top_replacement lookups keyed off the same state
    # formula) on both absence rows.
    prior_shares = p1_absences["prior_share"].tolist()
    assert prior_shares[0] == pytest.approx(prior_shares[1])

    # Hand-recompute P1's state after G1-G3 to cross-check the frozen value.
    alpha = _ALPHA
    state = 10.0 / 20.0
    state = alpha * (12.0 / 20.0) + (1.0 - alpha) * state
    state = alpha * (11.0 / 20.0) + (1.0 - alpha) * state
    assert prior_shares[0] == pytest.approx(state)

    # top_replacement on G4 is P2, whose own share that game was 8/20.
    g4_row = p1_absences.loc[p1_absences["game_id"].eq("G4")].iloc[0]
    assert g4_row["top_replacement_share"] == pytest.approx(8.0 / 20.0)
    assert not math.isnan(g4_row["top_replacement_prior_share"])

    # team_total_trailing equals the (constant) prior team_total of 20 once
    # there is history to trail.
    assert g4_row["team_total_trailing"] == pytest.approx(20.0)


def test_build_absence_frame_excludes_team_games_below_minimum() -> None:
    # Same qualifying player as above, but the only "absence" opportunity is
    # a team-game whose team_total is below FROZEN_MIN_TEAM_ACTIONS["carry"].
    rows = [
        _action_row(
            game_id=f"G{index}",
            season=2022,
            week=index,
            order_key=index,
            team="A",
            player_id="P1",
            action_type="carry",
            count=10,
            team_total=20,
        )
        for index in range(1, 4)
    ]
    actions = pd.DataFrame(rows)
    states = build_role_states(actions)

    team_games = pd.DataFrame(
        [
            {
                "game_id": f"G{index}",
                "season": 2022,
                "week": index,
                "order_key": index,
                "team": "A",
                "action_type": "carry",
                "team_total": 20,
            }
            for index in range(1, 4)
        ]
        + [
            {
                "game_id": "G4-low-volume",
                "season": 2022,
                "week": 4,
                "order_key": 4,
                "team": "A",
                "action_type": "carry",
                "team_total": FROZEN_MIN_TEAM_ACTIONS["carry"] - 1,
            }
        ]
    )
    absences = build_absence_frame(team_games, states)
    assert absences.empty


# ---------------------------------------------------------------------------
# 4. cfb_role_actions: coverage gate and action definitions
# ---------------------------------------------------------------------------


def _canonical_games(rows: list[tuple[object, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [game_id for game_id, _ in rows],
            "gameday": pd.to_datetime([gameday for _, gameday in rows]),
        }
    )


def _base_pbp_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "game_id": "G1",
        "season": 2020,
        "week": 1,
        "seasonType": 2,
        "pos_team": "A",
        "pass": False,
        "rush": False,
        "type.text": "No Play",
        "passer_player_id": np.nan,
        "rusher_player_id": np.nan,
        "receiver_player_id": np.nan,
    }
    row.update(overrides)
    return row


def test_cfb_role_actions_coverage_gate_excludes_low_coverage_season() -> None:
    canonical_games = _canonical_games([("G2013", "2013-09-01"), ("G2014", "2014-09-01")])
    rows = [
        _base_pbp_row(
            game_id="G2013",
            season=2013,
            **{"type.text": "Pass Reception"},
            receiver_player_id="WR1",
        )
        for _ in range(10)
    ]
    rows += [
        _base_pbp_row(
            game_id="G2014",
            season=2014,
            **{"type.text": "Pass Reception"},
            receiver_player_id="WR1" if index < 4 else np.nan,
        )
        for index in range(10)
    ]
    pbp = pd.DataFrame(rows)

    actions, _team_games, coverage = cfb_role_actions(pbp, canonical_games)

    reception_coverage = coverage.loc[coverage["action_type"].eq("reception")].set_index("season")
    assert reception_coverage.loc[2013, "coverage"] == pytest.approx(1.0)
    assert bool(reception_coverage.loc[2013, "excluded"]) is False
    assert reception_coverage.loc[2014, "coverage"] == pytest.approx(0.4)
    assert bool(reception_coverage.loc[2014, "excluded"]) is True

    reception_actions = actions.loc[actions["action_type"].eq("reception")]
    assert set(reception_actions["season"]) == {2013}
    assert reception_actions["count"].iloc[0] == 10


def test_cfb_role_actions_definitions_and_uncredited_play() -> None:
    # pbp game_id is numeric; canonical_games' game_id is a string -- the
    # adapter must normalize both to str before joining.
    canonical_games = pd.DataFrame({"game_id": ["1"], "gameday": pd.to_datetime(["2020-09-05"])})
    rows = []
    for index in range(20):
        rows.append(
            _base_pbp_row(
                game_id=1,
                **{"pass": True},
                passer_player_id="QB1" if index < 19 else np.nan,  # one uncredited dropback
            )
        )
    for _ in range(12):
        rows.append(_base_pbp_row(game_id=1, rush=True, rusher_player_id="RB1"))
    for _ in range(6):
        rows.append(
            _base_pbp_row(game_id=1, **{"type.text": "Pass Reception"}, receiver_player_id="WR1")
        )
    pbp = pd.DataFrame(rows)

    actions, _team_games, coverage = cfb_role_actions(pbp, canonical_games)

    by_action = actions.set_index("action_type")
    assert by_action.loc["dropback", "count"] == 19
    assert by_action.loc["dropback", "team_total"] == 19
    assert by_action.loc["dropback", "player_id"] == "QB1"
    assert by_action.loc["carry", "count"] == 12
    assert by_action.loc["reception", "count"] == 6

    coverage_by_type = coverage.set_index("action_type")
    assert coverage_by_type.loc["dropback", "coverage"] == pytest.approx(19 / 20)
    assert bool(coverage_by_type.loc["dropback", "excluded"]) is False
    assert coverage_by_type.loc["carry", "coverage"] == pytest.approx(1.0)
    assert coverage_by_type.loc["reception", "coverage"] == pytest.approx(1.0)

    # order_key comes from the canonical games' gameday.
    assert (actions["order_key"] == pd.Timestamp("2020-09-05")).all()


# ---------------------------------------------------------------------------
# 5. nfl_role_actions: dropback = attempts + sacks_taken
# ---------------------------------------------------------------------------


def test_nfl_role_actions_dropback_is_attempts_plus_sacks_taken() -> None:
    role_stats = pd.DataFrame(
        [
            {
                "player_id": "QB1",
                "season": 2021,
                "week": 1,
                "game_id": "N1",
                "team": "A",
                "attempts": 25,
                "carries": 2,
                "receptions": 0,
                "sacks_taken": 3,
            },
            {
                "player_id": "RB1",
                "season": 2021,
                "week": 1,
                "game_id": "N1",
                "team": "A",
                "attempts": 0,
                "carries": 15,
                "receptions": 3,
                "sacks_taken": 0,
            },
        ]
    )
    actions, _team_games, coverage = nfl_role_actions(role_stats)

    dropback = actions.loc[actions["action_type"].eq("dropback") & actions["player_id"].eq("QB1")]
    assert dropback["count"].iloc[0] == 28
    assert dropback["team_total"].iloc[0] == 28

    carry_team_total = actions.loc[
        actions["action_type"].eq("carry") & actions["player_id"].eq("RB1"), "team_total"
    ].iloc[0]
    assert carry_team_total == 17  # 2 (QB1) + 15 (RB1)

    official_note = coverage.loc[coverage["action_type"].eq("dropback")].iloc[0]
    assert official_note["coverage"] == pytest.approx(1.0)
    assert official_note["note"] == "official_stats"
    assert bool(official_note["excluded"]) is False


# ---------------------------------------------------------------------------
# 6. evaluate_replication_gates
# ---------------------------------------------------------------------------


def _summary_row(
    action_type: str, median_ratio: float, fraction_severe_under: float
) -> dict[str, object]:
    return {
        "action_type": action_type,
        "n": 100,
        "median_ratio": median_ratio,
        "mean_ratio": median_ratio,
        "p25_ratio": median_ratio - 0.1,
        "p75_ratio": median_ratio + 0.1,
        "fraction_at_or_above_one": 0.5,
        "fraction_severe_under": fraction_severe_under,
    }


def test_evaluate_replication_gates_mixed_pass_fail() -> None:
    cfb_summary = pd.DataFrame(
        [
            _summary_row("dropback", median_ratio=0.95, fraction_severe_under=0.05),
            _summary_row("carry", median_ratio=1.25, fraction_severe_under=0.05),  # band fails
            _summary_row("reception", median_ratio=1.00, fraction_severe_under=0.05),
        ]
    )
    nfl_summary = pd.DataFrame(
        [
            _summary_row("dropback", median_ratio=1.00, fraction_severe_under=0.05),
            _summary_row("carry", median_ratio=1.20, fraction_severe_under=0.05),
            _summary_row("reception", median_ratio=1.25, fraction_severe_under=0.05),  # gap fails
        ]
    )
    gates = evaluate_replication_gates(cfb_summary, nfl_summary)

    assert gates["dropback"]["replicated"] is True
    assert gates["carry"]["passed_median_band"] is False
    assert gates["carry"]["replicated"] is False
    assert gates["reception"]["passed_league_gap"] is False
    assert gates["reception"]["replicated"] is False


def test_evaluate_replication_gates_severe_under_delivery_failure() -> None:
    cfb_summary = pd.DataFrame(
        [_summary_row("dropback", median_ratio=1.0, fraction_severe_under=0.20)]
    )
    nfl_summary = pd.DataFrame(
        [_summary_row("dropback", median_ratio=1.0, fraction_severe_under=0.05)]
    )
    gates = evaluate_replication_gates(cfb_summary, nfl_summary)
    assert gates["dropback"]["passed_median_band"] is True
    assert gates["dropback"]["passed_league_gap"] is True
    assert gates["dropback"]["passed_severe_under"] is False
    assert gates["dropback"]["replicated"] is False


def test_evaluate_replication_gates_missing_action_type() -> None:
    cfb_summary = pd.DataFrame(
        [_summary_row("dropback", median_ratio=1.0, fraction_severe_under=0.0)]
    )
    nfl_summary = pd.DataFrame(columns=cfb_summary.columns)
    gates = evaluate_replication_gates(cfb_summary, nfl_summary)
    assert gates["dropback"]["replicated"] is False
    assert "note" in gates["dropback"]
    assert gates["carry"]["replicated"] is False


# ---------------------------------------------------------------------------
# 7. run_role_replication end-to-end
# ---------------------------------------------------------------------------


def test_run_role_replication_end_to_end_produces_all_keys() -> None:
    game_ids = [f"G{index}" for index in range(1, 6)]
    canonical_games = pd.DataFrame(
        {
            "game_id": game_ids,
            "gameday": pd.to_datetime([f"2020-09-{day:02d}" for day in range(1, 5 + 1)]),
        }
    )
    pbp_rows: list[dict[str, object]] = []
    for week, game_id in enumerate(game_ids, start=1):
        for _ in range(20):
            pbp_rows.append(
                _base_pbp_row(
                    game_id=game_id,
                    season=2020,
                    week=week,
                    **{"pass": True},
                    passer_player_id="QB1",
                )
            )
    cfb_pbp = pd.DataFrame(pbp_rows)

    nfl_role_stats = pd.DataFrame(
        [
            {
                "player_id": "QB1",
                "season": 2020,
                "week": week,
                "game_id": f"N{week}",
                "team": "A",
                "attempts": 20,
                "carries": 0,
                "receptions": 0,
                "sacks_taken": 0,
            }
            for week in range(1, 6)
        ]
    )

    result = run_role_replication(cfb_pbp, canonical_games, nfl_role_stats)

    expected_keys = {
        "cfb_summary",
        "nfl_summary",
        "gates",
        "coverage",
        "cfb_delivery",
        "nfl_delivery",
        "cfb_absences",
        "nfl_absences",
        "configuration",
    }
    assert expected_keys == set(result.keys())
    assert isinstance(result["cfb_delivery"], pd.DataFrame)
    assert isinstance(result["nfl_delivery"], pd.DataFrame)
    assert isinstance(result["cfb_absences"], pd.DataFrame)
    assert isinstance(result["nfl_absences"], pd.DataFrame)
    assert result["configuration"]["hypothesis_frozen_before_scoring"] is True
    assert result["configuration"]["min_prior_appearances"] == FROZEN_MIN_PRIOR_APPEARANCES
    assert set(FROZEN_ROLE_THRESHOLDS) == set(result["configuration"]["role_thresholds"])

    # QB1 delivers 100% of both leagues' dropbacks every week, so the two
    # trailing appearances (weeks 4-5, once min_prior=3 is satisfied) qualify
    # with ratio == 1.0 in both leagues, and the gate is a clean replication.
    dropback_gate = result["gates"]["dropback"]
    assert dropback_gate["replicated"] is True
    assert dropback_gate["cfb_median_ratio"] == pytest.approx(1.0)
    assert dropback_gate["nfl_median_ratio"] == pytest.approx(1.0)
