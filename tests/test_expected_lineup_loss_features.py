from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nfl_ats import expected_lineup_loss_features as loss
from nfl_ats import play_probability as play
from nfl_ats.data import DataContractError
from nfl_ats.play_probability import DEPTH_CHART_HISTORY_OUTPUT_COLUMNS

_SEASONS = (2020, 2021, 2022, 2023)
_WEEKS = (1, 2, 3, 4)
_TEAMS = tuple(f"T{index:02d}" for index in range(6))
_ROLES = (("QB", 1), ("QB", 2), ("QB", 3), ("WR", 1), ("WR", 2), ("WR", 3))


def _build_synthetic_sources(
    *, seed: int = 0, extra_week: dict[str, object] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    rng = np.random.default_rng(seed)
    depth_rows: list[dict[str, object]] = []
    roster_rows: list[dict[str, object]] = []
    snap_rows: list[dict[str, object]] = []
    injury_rows: list[dict[str, object]] = []
    extra_week = extra_week or {}

    for season in _SEASONS:
        for team in _TEAMS:
            for week in _WEEKS:
                key = (season, team, week)
                baseline_draw = bool(rng.random() < 0.3)
                qb1_out = bool(extra_week[key]) if key in extra_week else baseline_draw
                for position, rank in _ROLES:
                    gsis_id = f"{team}-{position}{rank}"
                    depth_rows.append(
                        {
                            "season": season,
                            "week": week,
                            "team": team,
                            "gsis_id": gsis_id,
                            "player_name": gsis_id,
                            "position": position,
                            "position_group": "skill",
                            "depth_rank": rank,
                            "source_schema": "legacy_week",
                        }
                    )
                    roster_rows.append(
                        {
                            "season": season,
                            "week": week,
                            "team": team,
                            "position": position,
                            "status": "ACT",
                            "full_name": gsis_id,
                            "gsis_id": gsis_id,
                            "pfr_id": gsis_id,
                            "years_exp": 3.0,
                            "game_type": "REG",
                        }
                    )
                    if position == "QB":
                        played = (rank == 1 and not qb1_out) or (rank == 2 and qb1_out)
                    else:
                        played = bool(rng.random() < {1: 0.9, 2: 0.6, 3: 0.2}[rank])
                    if played:
                        snap_rows.append(
                            {
                                "game_id": f"{season}_{week:02d}_{team}",
                                "season": season,
                                "game_type": "REG",
                                "week": week,
                                "player": gsis_id,
                                "pfr_player_id": gsis_id,
                                "position": position,
                                "team": team,
                                "offense_snaps": 55.0,
                                "offense_pct": 0.85,
                                "defense_snaps": 0.0,
                                "defense_pct": 0.0,
                                "st_snaps": 0.0,
                                "st_pct": 0.0,
                            }
                        )
                    if position == "QB" and rank == 1 and qb1_out:
                        injury_rows.append(
                            {
                                "season": season,
                                "game_type": "REG",
                                "team": team,
                                "week": week,
                                "gsis_id": gsis_id,
                                "position": position,
                                "report_status": "Out",
                                "practice_status": "Did Not Participate In Practice",
                                "date_modified": pd.Timestamp(f"{season}-01-01T00:00:00Z"),
                            }
                        )
    depth_history = pd.DataFrame(depth_rows)[list(DEPTH_CHART_HISTORY_OUTPUT_COLUMNS)]
    depth_history["decision_at"] = pd.to_datetime(
        depth_history["season"].astype(str) + "-09-01", utc=True
    )
    rosters = pd.DataFrame(roster_rows)
    snaps = pd.DataFrame(snap_rows)
    injuries = pd.DataFrame(injury_rows)
    return depth_history, rosters, snaps, injuries


@pytest.fixture
def sources():
    depth, rosters, snaps, injuries = _build_synthetic_sources()
    defense = depth.gsis_id.str.endswith("WR2")
    depth.loc[defense, ["position", "position_group", "depth_rank"]] = ["CB", "secondary", 1]
    depth["decision_at"] = pd.to_datetime(depth.season.astype(str) + "-09-10T19:00:00Z", utc=True)
    games = pd.DataFrame(
        [
            {
                "season": season,
                "week": week,
                "home_team": "T00",
                "away_team": "T01",
                "kickoff": f"{season}-09-10T19:00:00Z",
                "game_id": f"{season}_{week}",
            }
            for season in (2020, 2021, 2022, 2023)
            for week in (1, 2, 3, 4)
        ]
    )
    decisions = loss.team_week_decision_instants(games)
    depth = depth.drop(columns="decision_at").merge(
        decisions[["season", "week", "team", "decision_at"]],
        on=["season", "week", "team"],
        how="inner",
    )
    injuries["effective_observed_at"] = injuries.date_modified
    return depth, rosters, snaps, injuries, games


def test_team_week_decisions_use_shared_pool_cutoff():
    games = pd.DataFrame(
        [
            {"season": 2026, "week": week, "home_team": "H", "away_team": "A", "kickoff": kickoff}
            for week, kickoff in enumerate(
                [
                    "2026-09-10T00:20:00Z",
                    "2026-09-13T17:00:00Z",
                    "2026-09-13T20:25:00Z",
                    "2026-09-15T00:20:00Z",
                    "2026-11-09T01:20:00Z",
                ],
                1,
            )
        ]
    )
    got = loss.team_week_decision_instants(games).query("team == 'H'")
    assert list(got.decision_at) == list(
        pd.to_datetime(
            [
                "2026-09-10T00:20:00Z",
                "2026-09-13T17:00:00Z",
                "2026-09-13T20:00:00Z",
                "2026-09-13T20:00:00Z",
                "2026-11-08T21:00:00Z",
            ],
            utc=True,
        )
    )


def test_visible_injury_lookup_latest_visible_revision(sources):
    _, _, _, injuries, games = sources
    decisions = loss.team_week_decision_instants(games)
    row = injuries.query("team == 'T00'").iloc[[0]].copy()
    cutoff = (
        decisions.query("team == 'T00'")
        .set_index(["season", "week"])
        .loc[(row.iloc[0].season, row.iloc[0].week), "decision_at"]
    )
    row["effective_observed_at"] = cutoff
    row["report_status"] = "Questionable"
    late = row.assign(effective_observed_at=cutoff + pd.Timedelta(seconds=1), report_status="Out")
    got = loss.visible_injury_lookup(pd.concat([row, late]), decisions)
    assert got.report_status.tolist() == ["Questionable"]
    assert loss.visible_injury_lookup(row.assign(effective_observed_at=pd.NaT), decisions).empty


def test_probability_fit_receives_only_prior_seasons(sources, monkeypatch):
    depth, rosters, snaps, injuries, _ = sources
    panel = play.build_player_week_panel(depth, rosters, snaps, injuries)
    seen = []

    def fit(training, *, scored_season):
        assert training.season.max() < scored_season
        seen.append(scored_season)
        return object()

    monkeypatch.setattr(loss, "fit_play_probability_model", fit)
    monkeypatch.setattr(
        loss,
        "predict_play_probabilities",
        lambda model, rows: pd.DataFrame(
            {"play_probability": np.full(len(rows), 0.75)}, index=rows.index
        ),
    )
    got = loss.attach_play_probabilities(loss.select_week_starters(panel), panel)
    assert seen == [2021, 2022, 2023]
    assert got.loc[got.season.eq(2020), "play_probability"].isna().all()
    assert got.loc[got.season.gt(2020), "play_probability"].eq(0.75).all()


def test_end_to_end_late_injury_depth_snaps_and_outcomes_cannot_change_features(sources):
    depth, rosters, snaps, injuries, games = sources
    panel = play.build_player_week_panel(depth, rosters, snaps, injuries)
    targets = games.query("season == 2023 and week == 4")
    baseline = loss.attach_expected_lineup_loss_features(
        targets, panel=panel, injuries=injuries, scored_seasons=[2023]
    )
    changed = panel.copy()
    changed.loc[changed.season.eq(2023), ["played", "started"]] = True
    changed["play_probability"] = 0.0
    late_depth = changed.query("season == 2023 and week == 4").copy()
    late_depth["depth_observed_at"] = late_depth.decision_at + pd.Timedelta(seconds=1)
    late_depth["source_schema"] = "daily_dt"
    late_depth["depth_rank"] = 2
    late_depth["trailing4_snap_share"] = 1000.0
    late_depth["play_probability"] = 0.0
    changed = pd.concat([changed, late_depth], ignore_index=True)
    late_injury = late_depth[["season", "week", "team", "gsis_id", "decision_at"]].rename(
        columns={"decision_at": "effective_observed_at"}
    )
    late_injury["effective_observed_at"] += pd.Timedelta(seconds=1)
    late_injury["report_status"] = "Out"
    late_injury["practice_status"] = "Did Not Participate In Practice"
    result = loss.attach_expected_lineup_loss_features(
        targets, panel=changed, injuries=pd.concat([injuries, late_injury]), scored_seasons=[2023]
    )
    pd.testing.assert_frame_equal(baseline, result)
    for group in loss.LINEUP_GROUPS:
        assert baseline[f"diff_expected_lineup_loss_{group}"].iloc[0] == pytest.approx(
            baseline[f"home_expected_lineup_loss_{group}"].iloc[0]
            - baseline[f"away_expected_lineup_loss_{group}"].iloc[0]
        )
    poisoned_snaps = snaps.copy()
    mask = poisoned_snaps.season.eq(2023) & poisoned_snaps.week.eq(4)
    poisoned_snaps.loc[mask, ["offense_pct", "defense_pct"]] = 999.0
    rebuilt = play.build_player_week_panel(depth, rosters, poisoned_snaps, injuries)
    history_columns = ["weeks_since_last_snap", "trailing4_snap_share"]
    pd.testing.assert_frame_equal(panel[history_columns], rebuilt[history_columns])


def test_mismatched_panel_decision_fails_closed(sources):
    depth, rosters, snaps, injuries, games = sources
    panel = play.build_player_week_panel(depth, rosters, snaps, injuries)
    panel["decision_at"] += pd.Timedelta(hours=24)
    with pytest.raises(DataContractError, match="pool decision cutoff"):
        loss.attach_expected_lineup_loss_features(games, panel=panel, injuries=injuries)
