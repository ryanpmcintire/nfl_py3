"""LEAD-62 arithmetic, chronological fitting, and decision-time leakage contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from test_play_probability import _build_synthetic_sources

from nfl_ats import expected_lineup_loss_features as loss
from nfl_ats import play_probability as play
from nfl_ats.data import DataContractError


@pytest.fixture
def sources():
    depth, rosters, snaps, injuries = _build_synthetic_sources()
    # Exercise all three loss groups with actual player history and model inputs.
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


def test_starters_groups_and_daily_observation_filter(sources):
    depth, rosters, snaps, injuries, _ = sources
    panel = play.build_player_week_panel(depth, rosters, snaps, injuries)
    extra = panel.iloc[[0]].assign(position="K", position_group="other", gsis_id="kicker")
    late = panel.iloc[[0]].assign(source_schema="daily_dt", gsis_id="late")
    late["depth_observed_at"] = late.decision_at + pd.Timedelta(seconds=1)
    unknown = late.assign(gsis_id="unknown", depth_observed_at=pd.NaT)
    got = loss.select_week_starters(pd.concat([panel, extra, late, unknown]))
    assert set(got.lineup_group) == {"qb", "offense", "defense"}
    assert got.depth_rank.eq(1).all()
    assert not set(got.gsis_id) & {"kicker", "late", "unknown"}


def test_asof_inputs_replace_panel_statuses(sources):
    depth, rosters, snaps, injuries, games = sources
    panel = play.build_player_week_panel(depth, rosters, snaps, injuries)
    starters = loss.select_week_starters(panel).assign(
        report_category="out",
        practice_category="dnp",
        roster_status="INA",
        has_injury_designation=True,
    )
    lookup = loss.visible_injury_lookup(injuries.iloc[:0], loss.team_week_decision_instants(games))
    got = loss.attach_asof_injury_features(starters, lookup)
    assert got.roster_status.eq("ACT").all()
    assert not got.has_injury_designation.any()
    assert got.report_category.eq(play.report_category(None)).all()
    assert (
        got.loc[got.lineup_group.ne("qb"), "qb1_report_category"].eq(play.QB1_NOT_APPLICABLE).all()
    )


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


def test_group_sums_missing_history_and_missing_probability():
    rows = pd.DataFrame(
        {
            "season": [2020] * 5,
            "week": [1] * 5,
            "team": ["H"] * 5,
            "lineup_group": ["qb", "offense", "offense", "defense", "defense"],
            "trailing4_snap_share": [0.8, 0.5, np.nan, 0.6, 0.9],
            "play_probability": [0.25, 0.5, 0.1, 0.5, np.nan],
        }
    )
    got = loss.team_week_expected_loss(rows).iloc[0]
    assert got.expected_lineup_loss_qb == pytest.approx(0.6)
    assert got.expected_lineup_loss_offense == pytest.approx(0.25)
    assert got.expected_lineup_loss_defense == pytest.approx(0.3)
    assert loss.team_week_expected_loss(rows.iloc[:0]).empty


def test_reliability_odd_even_team_season_means():
    rows = pd.DataFrame(
        [
            {
                "season": 2020,
                "week": week,
                "team": str(team),
                "expected_lineup_loss_qb": team * (1 if week % 2 else 2),
                "expected_lineup_loss_offense": 0,
                "expected_lineup_loss_defense": 0,
            }
            for team in range(4)
            for week in range(1, 5)
        ]
    )
    assert loss.team_season_split_half_reliability(rows) == {
        "n_team_seasons": 4,
        "reliability": pytest.approx(1.0),
    }
    assert np.isnan(loss.team_season_split_half_reliability(rows.iloc[:1])["reliability"])


def test_end_to_end_late_injury_depth_snaps_and_outcomes_cannot_change_features(sources):
    depth, rosters, snaps, injuries, games = sources
    panel = play.build_player_week_panel(depth, rosters, snaps, injuries)
    targets = games.query("season == 2023 and week == 4")
    baseline = loss.attach_expected_lineup_loss_features(
        targets, panel=panel, injuries=injuries, scored_seasons=[2023]
    )
    # Poison scored-season outcomes: neither booster nor calibration may use them.
    changed = panel.copy()
    changed.loc[changed.season.eq(2023), ["played", "started"]] = True
    # Cached probabilities have no decision-time guarantee and must be ignored.
    changed["play_probability"] = 0.0
    # Append revisions that would change every group if made visible.
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
    # Shared history helper must ignore own current-week snaps.
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


def test_harness_retains_all_production_training_and_pairs_only_covered_games(
    tmp_path, monkeypatch
):
    import json
    import sys
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import expected_lineup_loss_on_production as harness

    features = pd.DataFrame(
        {
            "game_id": ["old_training", "covered", "uncovered"],
            "season": [2009, 2020, 2020],
            **{column: [np.nan, 0.1, np.nan] for column in loss.EXPECTED_LINEUP_LOSS_COLUMNS},
        }
    )
    path = tmp_path / "features.parquet"
    features.to_parquet(path)
    (tmp_path / "build.json").write_text(json.dumps({"reliability": {}, "coverage": {}}))
    monkeypatch.setattr(harness, "load_registry", lambda: {})
    monkeypatch.setattr(
        harness.confirmation, "scoped_window_frame", lambda rows, registry, family: (rows, (2020,))
    )
    profiles = []

    def run_arm(rows, candidate, **kwargs):
        pd.testing.assert_frame_equal(rows, features)
        profiles.append(kwargs["profile"])
        return pd.DataFrame({"game_id": ["covered", "uncovered"]})

    monkeypatch.setattr(harness.confirmation, "run_arm", run_arm)
    monkeypatch.setattr(harness.confirmation, "paired_frame", lambda baseline, candidate: baseline)

    def null_distribution(paired, **kwargs):
        assert paired.game_id.tolist() == ["covered"]
        return {}

    monkeypatch.setattr(harness.confirmation, "null_distribution", null_distribution)
    monkeypatch.setattr(harness, "artifact_provenance", lambda *args: {})
    monkeypatch.setattr(harness, "write_experiment_artifact", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "harness",
            "--mode",
            "null",
            "--features",
            str(path),
            "--output",
            str(tmp_path / "output"),
        ],
    )
    assert harness.main() == 0
    assert profiles == ["weak_stack", harness.PROFILE]
    assert harness.PROFILE not in harness.margin.MARGIN_FEATURE_PROFILES
