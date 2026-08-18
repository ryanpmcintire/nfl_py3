from __future__ import annotations

import pandas as pd
import pytest

from nfl_ats.cfb_qb_dependence import (
    CFB_QB_DEPENDENCE_COLUMNS,
    CFB_QB_MIN_DROPBACKS,
    build_and_attach_cfb_qb_dependence,
    build_cfb_pass_rate_team_games,
    build_cfb_qb_game_metrics,
    build_cfb_qb_states,
)

# ---------------------------------------------------------------------------
# Synthetic CFB universe: three teams, controlled passer identities/EPA, and
# enough competitive plays per team-game to exercise both the per-game floor
# (CFB_QB_MIN_GAME_DROPBACKS=5) and the career gate (CFB_QB_MIN_DROPBACKS=20).
# ---------------------------------------------------------------------------

TEAM_HOME = 1
TEAM_AWAY = 2
TEAM_THIRD = 3


def _play_row(
    *,
    game_id: int,
    season: int,
    week: int,
    team_id: int,
    opponent_id: int,
    is_home: bool,
    pass_flag: bool,
    epa: float,
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "seasonType": 2,
        "pos_team_id": team_id,
        "homeTeamId": team_id if is_home else opponent_id,
        "awayTeamId": opponent_id if is_home else team_id,
        "is_home": is_home,
        "EPA": epa,
        "EPA_success": epa > 0,
        "rush": not pass_flag,
        "pass": pass_flag,
        "kneel_down": False,
        "statYardage": 6.0,
        "home_wp_before": 0.5,
        "away_wp_before": 0.5,
        "passer_player_id": None,
    }


def _team_game_plays(
    *,
    game_id: int,
    season: int,
    week: int,
    team_id: int,
    opponent_id: int,
    is_home: bool,
    passer_id: str | None,
    n_dropbacks: int,
    n_rushes: int,
    dropback_epa: float,
) -> list[dict[str, object]]:
    """``n_dropbacks`` pass plays (credited to ``passer_id`` if not None) plus
    ``n_rushes`` rush plays (never credited -- rush identity is out of scope)."""

    rows: list[dict[str, object]] = []
    for _ in range(n_dropbacks):
        row = _play_row(
            game_id=game_id,
            season=season,
            week=week,
            team_id=team_id,
            opponent_id=opponent_id,
            is_home=is_home,
            pass_flag=True,
            epa=dropback_epa,
        )
        row["passer_player_id"] = passer_id
        rows.append(row)
    for _ in range(n_rushes):
        rows.append(
            _play_row(
                game_id=game_id,
                season=season,
                week=week,
                team_id=team_id,
                opponent_id=opponent_id,
                is_home=is_home,
                pass_flag=False,
                epa=0.0,
            )
        )
    return rows


def _canonical_games(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["gameday"] = pd.to_datetime(frame["gameday"])
    return frame


def _base_universe() -> tuple[pd.DataFrame, pd.DataFrame]:
    """QB1 (home team 1) crosses the career gate by game 4; QB3 (team 3) never does.

    Six games, one per week, home team 1 hosts every game against a rotating
    opponent so QB1's career dropbacks accumulate across games. QB1 throws 8
    dropbacks/game at EPA 0.30/dropback (well above the per-game floor of 5),
    crossing the 20-dropback career gate partway through game 3 (8+8=16 after
    game 2, 24 after game 3) -- so ``state_qb_epa_per_dropback`` should first
    be non-NaN on game 3's own qb_games row and every game after. Team 3's QB3
    plays exactly two low-volume games (6 dropbacks each, 12 career total),
    always below the 20-dropback gate, so its state must stay NaN throughout.
    """

    games = []
    plays: list[dict[str, object]] = []
    opponents = [TEAM_AWAY, TEAM_AWAY, TEAM_AWAY, TEAM_AWAY, TEAM_THIRD, TEAM_THIRD]
    for week, opponent in enumerate(opponents, start=1):
        game_id = 20200000 + week
        games.append(
            {
                "game_id": game_id,
                "season": 2020,
                "week": week,
                "gameday": pd.Timestamp("2020-09-01") + pd.Timedelta(days=7 * (week - 1)),
                "home_id": TEAM_HOME,
                "away_id": opponent,
            }
        )
        plays.extend(
            _team_game_plays(
                game_id=game_id,
                season=2020,
                week=week,
                team_id=TEAM_HOME,
                opponent_id=opponent,
                is_home=True,
                passer_id="QB1",
                n_dropbacks=8,
                n_rushes=4,
                dropback_epa=0.30,
            )
        )
        if opponent == TEAM_AWAY:
            plays.extend(
                _team_game_plays(
                    game_id=game_id,
                    season=2020,
                    week=week,
                    team_id=TEAM_AWAY,
                    opponent_id=TEAM_HOME,
                    is_home=False,
                    passer_id="QB2",
                    n_dropbacks=6,
                    n_rushes=6,
                    dropback_epa=-0.10,
                )
            )
        else:
            plays.extend(
                _team_game_plays(
                    game_id=game_id,
                    season=2020,
                    week=week,
                    team_id=TEAM_THIRD,
                    opponent_id=TEAM_HOME,
                    is_home=False,
                    passer_id="QB3",
                    n_dropbacks=6,
                    n_rushes=6,
                    dropback_epa=0.05,
                )
            )
    return _canonical_games(games), pd.DataFrame(plays)


# ---------------------------------------------------------------------------
# 1. off_pass_rate: hand-computed share on a tiny synthetic PBP frame
# ---------------------------------------------------------------------------


def test_off_pass_rate_matches_hand_computed_share() -> None:
    _games, pbp = _base_universe()
    team_games = build_cfb_pass_rate_team_games(pbp)
    game1_home = team_games.loc[
        team_games["game_id"].eq(20200001) & team_games["team_id"].eq(TEAM_HOME)
    ]
    assert len(game1_home) == 1
    # 8 pass plays, 4 rush plays -> 8 / 12 = 0.6666...
    assert game1_home["off_pass_rate"].iloc[0] == pytest.approx(8.0 / 12.0)

    game1_away = team_games.loc[
        team_games["game_id"].eq(20200001) & team_games["team_id"].eq(TEAM_AWAY)
    ]
    # 6 pass plays, 6 rush plays -> 0.5
    assert game1_away["off_pass_rate"].iloc[0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 2. Dropback-gate: a low-volume backup never crosses the career threshold
# ---------------------------------------------------------------------------


def test_dropback_gate_excludes_low_volume_backup_from_state() -> None:
    games, pbp = _base_universe()
    qb_games = build_cfb_qb_game_metrics(pbp)
    qb_states = build_cfb_qb_states(qb_games, games)

    qb3_rows = qb_states.loc[qb_states["passer_player_id"].eq("QB3")].sort_values("gameday")
    assert len(qb3_rows) == 2
    # QB3's career dropbacks: 6, then 12 -- both strictly below CFB_QB_MIN_DROPBACKS (20).
    assert qb3_rows["career_dropbacks"].tolist() == [6.0, 12.0]
    assert qb3_rows["state_qb_epa_per_dropback"].isna().all()
    assert CFB_QB_MIN_DROPBACKS > 12.0

    qb1_rows = qb_states.loc[qb_states["passer_player_id"].eq("QB1")].sort_values("gameday")
    # QB1's career dropbacks: 8, 16, 24, 32, 40, 48 -- crosses the gate at game 3 (24 >= 20).
    assert qb1_rows["career_dropbacks"].tolist() == [8.0, 16.0, 24.0, 32.0, 40.0, 48.0]
    assert qb1_rows["state_qb_epa_per_dropback"].isna().tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]


# ---------------------------------------------------------------------------
# 3. Leak-safety: a game's own result never touches its own pregame feature
# ---------------------------------------------------------------------------


def test_week_n_interaction_unaffected_by_week_n_own_result() -> None:
    games, pbp = _base_universe()
    baseline = build_and_attach_cfb_qb_dependence(games, pbp)

    perturbed_pbp = pbp.copy()
    # Perturb ONLY game 3's own plays: triple QB1's EPA that game and swap
    # its pass/rush mix, changing that game's own realized off_pass_rate and
    # qb_epa_per_dropback substantially.
    game3_home_pass = (
        perturbed_pbp["game_id"].eq(20200003)
        & perturbed_pbp["pos_team_id"].eq(TEAM_HOME)
        & perturbed_pbp["pass"]
    )
    perturbed_pbp.loc[game3_home_pass, "EPA"] = 5.0
    game3_home_rush = (
        perturbed_pbp["game_id"].eq(20200003)
        & perturbed_pbp["pos_team_id"].eq(TEAM_HOME)
        & perturbed_pbp["rush"]
    )
    perturbed_pbp.loc[perturbed_pbp.loc[game3_home_rush].index[:2], "pass"] = True
    perturbed_pbp.loc[perturbed_pbp.loc[game3_home_rush].index[:2], "rush"] = False
    perturbed_pbp.loc[perturbed_pbp.loc[game3_home_rush].index[:2], "passer_player_id"] = "QB1"

    perturbed = build_and_attach_cfb_qb_dependence(games, perturbed_pbp)

    game3_columns = list(CFB_QB_DEPENDENCE_COLUMNS)
    baseline_game3 = baseline.loc[baseline["game_id"].eq(20200003), game3_columns].reset_index(
        drop=True
    )
    perturbed_game3 = perturbed.loc[perturbed["game_id"].eq(20200003), game3_columns].reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(baseline_game3, perturbed_game3)

    # Sanity check: the perturbation DID propagate forward to a later game
    # (game 4's home_qb_starter_epa_per_dropback reads QB1's post-game-3
    # state), proving the leak-safety result above is not vacuous.
    baseline_game4 = baseline.loc[
        baseline["game_id"].eq(20200004), "home_qb_starter_epa_per_dropback"
    ].iloc[0]
    perturbed_game4 = perturbed.loc[
        perturbed["game_id"].eq(20200004), "home_qb_starter_epa_per_dropback"
    ].iloc[0]
    assert baseline_game4 != pytest.approx(perturbed_game4)


# ---------------------------------------------------------------------------
# 4. REG bit-identity: attaching the new columns never touches an existing one
# ---------------------------------------------------------------------------


def test_reg_bit_identity_of_existing_cfb_columns(
    cfb_inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    cfb_features_frame: pd.DataFrame,
) -> None:
    from nfl_ats.cfb_qb_dependence import CFB_QB_DEPENDENCE_COLUMNS

    _, _, pbp = cfb_inputs
    # The shared fixture's synthetic pbp carries no passer identity at all;
    # add a deterministic one so the module's functions have something to
    # read, without touching the shared conftest fixture itself.
    pbp_with_passers = pbp.copy()
    pbp_with_passers["passer_player_id"] = None
    pass_rows = pbp_with_passers["pass"].astype(bool)
    pbp_with_passers.loc[pass_rows, "passer_player_id"] = "QB_" + pbp_with_passers.loc[
        pass_rows, "pos_team_id"
    ].astype(str)

    existing_columns = list(cfb_features_frame.columns)
    joined = build_and_attach_cfb_qb_dependence(cfb_features_frame, pbp_with_passers)

    pd.testing.assert_frame_equal(
        joined.loc[:, existing_columns],
        cfb_features_frame.loc[:, existing_columns],
        check_exact=True,
    )
    for column in CFB_QB_DEPENDENCE_COLUMNS:
        assert column in joined.columns
        assert column not in existing_columns
