"""UI-20-AB: a real per-player, per-game play/start probability forecast.

Owner directive (2026-09-05, verbatim): "the percentages should obviously
make sense my dude... it needs to be a forecast about the game and it needs
to consider depth chart." Covers:

* Depth-chart history canonicalization on both nflverse schemas (legacy
  week-labelled rows, seasons <= 2024; daily dt-timestamped rows, seasons
  >= 2025 -- see the `nfl_ats.play_probability` module docstring for why
  these differ and how they are unified).
* Feature construction (`build_player_week_panel`) on a synthetic
  roster/snap/injury/depth-history panel large enough for a real
  walk-forward fit.
* Depth-rank ordering monotonicity for healthy players (rank 1 > rank 2 >
  rank 3, all else equal).
* The QB2-rises-when-QB1-is-out behaviour.
* Calibration table shape.
* The leakage test: a later week's snap, injury revision, or depth change
  never changes an earlier week's prediction/feature row.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nfl_ats.play_probability import (
    DEPTH_CHART_HISTORY_OUTPUT_COLUMNS,
    FEATURE_COLUMNS,
    LABEL_PLAYED,
    LABEL_STARTED,
    QB1_NOT_APPLICABLE,
    build_player_week_panel,
    calibration_slot,
    calibration_table,
    canonicalize_depth_chart_history,
    depth_rank_bucket,
    fit_play_probability_model,
    predict_play_probabilities,
    season_blocked_bootstrap,
    serving_feature_frame,
    serving_player_history,
)

SEASONS = (2020, 2021, 2022, 2023)
WEEKS = (1, 2, 3, 4)
TEAMS = tuple(f"T{index:02d}" for index in range(6))
_ROLES = (("QB", 1), ("QB", 2), ("QB", 3), ("WR", 1), ("WR", 2), ("WR", 3))


def _build_synthetic_sources(
    *, seed: int = 0, extra_week: dict[str, object] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A depth_history/rosters/snaps/injuries panel with a clean, strong,
    hand-designed signal: QB1 plays unless marked "Out" on the injury
    report that week, in which case QB2 plays instead; QB3 never plays.
    WR1/WR2/WR3 play with fixed, decreasing probabilities. ``extra_week``
    optionally overrides one (season, week) tuple's QB1-out draw -- used by
    the leakage test to mutate only the LAST week without touching earlier
    ones.
    """

    rng = np.random.default_rng(seed)
    depth_rows: list[dict[str, object]] = []
    roster_rows: list[dict[str, object]] = []
    snap_rows: list[dict[str, object]] = []
    injury_rows: list[dict[str, object]] = []
    extra_week = extra_week or {}

    for season in SEASONS:
        for team in TEAMS:
            for week in WEEKS:
                key = (season, team, week)
                # Always draw, even when about to override the result --
                # otherwise overriding one (season, team, week)'s draw
                # consumes a different number of `rng` calls than the
                # baseline run, desyncing the RNG stream for every LATER
                # team/season and changing rows the mutation was never
                # meant to touch (measured this session: a naive
                # short-circuited draw made 42% of week<4 rows differ,
                # which looked exactly like a leakage bug but was a test
                # fixture bug instead).
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


# ---------------------------------------------------------------------------
# Depth-chart history canonicalization: both nflverse schemas.
# ---------------------------------------------------------------------------


def test_canonicalize_depth_chart_history_legacy_schema() -> None:
    frame = pd.DataFrame(
        [
            {
                "season": 2020,
                "club_code": "KC",
                "week": 1,
                "game_type": "REG",
                "depth_team": 1,
                "position": "QB",
                "formation": "Offense",
                "depth_position": "QB",
                "gsis_id": "qb1",
                "full_name": "QB One",
            },
            {
                "season": 2020,
                "club_code": "KC",
                "week": 1,
                "game_type": "REG",
                "depth_team": 1,
                "position": "WR",
                "formation": "Special Teams",
                "depth_position": "KR",
                "gsis_id": "wr1",
                "full_name": "WR One",
            },
            {
                "season": 2020,
                "club_code": "KC",
                "week": 1,
                "game_type": "REG",
                "depth_team": 2,
                "position": "WR",
                "formation": "Offense",
                "depth_position": "WR",
                "gsis_id": "wr1",
                "full_name": "WR One",
            },
        ]
    )
    schedule = pd.DataFrame(
        [{"season": 2020, "week": 1, "home_team": "KC", "away_team": "DEN", "game_type": "REG"}]
    )
    result = canonicalize_depth_chart_history(frame, schedule)
    assert set(result.columns) == set(DEPTH_CHART_HISTORY_OUTPUT_COLUMNS)
    assert (result["source_schema"] == "legacy_week").all()
    # The Special-Teams-only KR row is dropped in favour of wr1's real
    # Offense/WR row (depth rank 2) -- a return-specialist listing must
    # never stand in for a receiver's real depth rank.
    wr_row = result.loc[result["gsis_id"].eq("wr1")]
    assert len(wr_row) == 1
    assert int(wr_row.iloc[0]["depth_rank"]) == 2
    assert wr_row.iloc[0]["position_group"] == "skill"


def test_canonicalize_depth_chart_history_daily_schema_aligns_to_weeks_by_kickoff() -> None:
    frame = pd.DataFrame(
        [
            {
                "dt": "2025-09-01T00:00:00Z",
                "team": "KC",
                "player_name": "QB One",
                "gsis_id": "qb1",
                "pos_abb": "QB",
                "pos_rank": 1,
            },
            {
                "dt": "2025-09-10T00:00:00Z",
                "team": "KC",
                "player_name": "QB Two",
                "gsis_id": "qb2",
                "pos_abb": "QB",
                "pos_rank": 1,
            },
        ]
    )
    schedule = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "home_team": "KC",
                "away_team": "DEN",
                "game_type": "REG",
                "gameday": "2025-09-07",
                "gametime": "13:00",
            },
            {
                "season": 2025,
                "week": 2,
                "home_team": "KC",
                "away_team": "BUF",
                "game_type": "REG",
                "gameday": "2025-09-14",
                "gametime": "13:00",
            },
        ]
    )
    result = canonicalize_depth_chart_history(frame, schedule)
    assert (result["source_schema"] == "daily_dt").all()
    week1 = result.loc[result["week"].eq(1)]
    week2 = result.loc[result["week"].eq(2)]
    # Week 1's kickoff (2025-09-07) is before the Sep-10 depth-chart update
    # (qb2) -- only the Sep-1 snapshot (qb1) was visible.
    assert week1["gsis_id"].tolist() == ["qb1"]
    # Week 2's kickoff (2025-09-14) is after the Sep-10 update.
    assert week2["gsis_id"].tolist() == ["qb2"]


# ---------------------------------------------------------------------------
# Feature construction on the synthetic panel.
# ---------------------------------------------------------------------------


def test_build_player_week_panel_produces_every_feature_and_label_column() -> None:
    depth_history, rosters, snaps, injuries = _build_synthetic_sources()
    panel = build_player_week_panel(depth_history, rosters, snaps, injuries)

    assert len(panel) == len(SEASONS) * len(TEAMS) * len(WEEKS) * len(_ROLES)
    for column in (*FEATURE_COLUMNS, LABEL_PLAYED, LABEL_STARTED):
        assert column in panel.columns

    # QB1 should play close to 70% of the time (not marked "Out" ~70% of
    # team-weeks, by construction).
    qb1_rate = panel.loc[panel["position"].eq("QB") & panel["depth_rank"].eq(1), "played"].mean()
    assert 0.55 < qb1_rate < 0.85

    # QB3 never plays by construction.
    qb3_rate = panel.loc[panel["position"].eq("QB") & panel["depth_rank"].eq(3), "played"].mean()
    assert qb3_rate == 0.0

    # Non-QB rows carry the "not applicable" QB1-status sentinel.
    wr_rows = panel.loc[panel["position"].eq("WR")]
    assert (wr_rows["qb1_report_category"] == QB1_NOT_APPLICABLE).all()
    assert (wr_rows["qb1_practice_category"] == QB1_NOT_APPLICABLE).all()

    # QB rows see a real (non-"not_applicable") QB1 status -- "out" on the
    # team-weeks QB1 was actually marked out, "none" otherwise.
    qb_rows = panel.loc[panel["position"].eq("QB")]
    assert set(qb_rows["qb1_report_category"].unique()) <= {"out", "none"}
    assert (qb_rows["qb1_report_category"] == "out").any()


# ---------------------------------------------------------------------------
# Depth-rank ordering monotonicity for healthy players.
# ---------------------------------------------------------------------------


def test_depth_rank_bucket_orders_1_2_3plus() -> None:
    assert depth_rank_bucket(1) == "1"
    assert depth_rank_bucket(2) == "2"
    assert depth_rank_bucket(3) == "3+"
    assert depth_rank_bucket(7) == "3+"
    assert depth_rank_bucket(None) == "unknown"
    assert depth_rank_bucket(float("nan")) == "unknown"


def test_healthy_player_probability_decreases_with_depth_rank() -> None:
    depth_history, rosters, snaps, injuries = _build_synthetic_sources()
    panel = build_player_week_panel(depth_history, rosters, snaps, injuries)
    model = fit_play_probability_model(panel, scored_season=2023)

    depth_rows = pd.DataFrame(
        {
            "gsis_id": ["wr-healthy-1", "wr-healthy-2", "wr-healthy-3"],
            "position": ["WR", "WR", "WR"],
            "depth_rank": [1, 2, 3],
        }
    )
    features = serving_feature_frame(
        depth_rows,
        week=1,
        current_injuries=None,
        player_history={},
    )
    predictions = predict_play_probabilities(model, features)
    probabilities = predictions["play_probability"].to_numpy()
    # Non-increasing with depth rank, and strictly lower at rank 3 than
    # rank 1. Not a strict `>` at every step: isotonic calibration fit on
    # this test's small calibration season can plateau at 1.0 across a
    # range of raw scores that are themselves strictly ordered (measured
    # this session -- the RAW booster scores are 0.99/0.84/0.09, correctly
    # ordered, but the calibrator maps both of the first two to 1.0). A
    # real production calibration set (hundreds of thousands of rows) does
    # not saturate this way -- see docs/play_probability_model.md's
    # measured 2026 Week 1 distribution.
    assert probabilities[0] >= probabilities[1] >= probabilities[2]
    assert probabilities[0] > probabilities[2]


# ---------------------------------------------------------------------------
# QB2's probability rises when QB1 is out.
# ---------------------------------------------------------------------------


def test_qb2_probability_rises_when_qb1_is_out() -> None:
    depth_history, rosters, snaps, injuries = _build_synthetic_sources()
    panel = build_player_week_panel(depth_history, rosters, snaps, injuries)
    model = fit_play_probability_model(panel, scored_season=2023)

    depth_rows = pd.DataFrame(
        {
            "gsis_id": ["qb-1", "qb-2"],
            "position": ["QB", "QB"],
            "depth_rank": [1, 2],
        }
    )

    features_healthy = serving_feature_frame(
        depth_rows, week=1, current_injuries=None, player_history={}
    )
    healthy = predict_play_probabilities(model, features_healthy)

    qb1_out = pd.DataFrame(
        [
            {
                "report_status": "Out",
                "practice_status": "Did Not Participate In Practice",
            }
        ],
        index=pd.Index(["qb-1"], name="gsis_id"),
    )
    features_qb1_out = serving_feature_frame(
        depth_rows, week=1, current_injuries=qb1_out, player_history={}
    )
    with_qb1_out = predict_play_probabilities(model, features_qb1_out)

    qb2_healthy = healthy.loc[1, "play_probability"]
    qb2_when_qb1_out = with_qb1_out.loc[1, "play_probability"]
    assert qb2_when_qb1_out > qb2_healthy

    # And QB1's own probability drops when he is the one marked "Out".
    assert with_qb1_out.loc[0, "play_probability"] < healthy.loc[0, "play_probability"]


# ---------------------------------------------------------------------------
# Calibration table shape.
# ---------------------------------------------------------------------------


def test_calibration_slot_covers_the_named_slots() -> None:
    assert calibration_slot("QB", 1) == "QB1"
    assert calibration_slot("QB", 2) == "QB2"
    assert calibration_slot("QB", 5) == "QB3+"
    assert calibration_slot("RB", 1) == "RB1"
    assert calibration_slot("RB", 2) == "RB2+"
    assert calibration_slot("WR", 2) == "WR2"
    assert calibration_slot("T", 1) == "OL"
    assert calibration_slot("DE", 1) == "DL"
    assert calibration_slot("LB", 1) == "LB"
    assert calibration_slot("CB", 1) == "CB"
    assert calibration_slot("S", 1) == "S"
    assert calibration_slot("K", 1) == "K/P"
    assert calibration_slot("P", 1) == "K/P"


def test_calibration_table_shape() -> None:
    frame = pd.DataFrame(
        {
            "position": ["QB", "QB", "WR", "WR"],
            "depth_rank": [1, 2, 1, 2],
            "predicted": [0.9, 0.2, 0.85, 0.5],
            "actual": [1.0, 0.0, 1.0, 1.0],
        }
    )
    table = calibration_table(frame, prediction_column="predicted", actual_column="actual")
    assert list(table.columns) == ["slot", "n", "mean_predicted", "mean_observed", "gap"]
    assert set(table["slot"]) == {"QB1", "QB2", "WR1", "WR2"}
    assert (table["n"] == 1).all()


def test_season_blocked_bootstrap_reports_probability_positive() -> None:
    always_positive = pd.Series([0.05, 0.03, 0.07, 0.04], index=[2020, 2021, 2022, 2023])
    result = season_blocked_bootstrap(always_positive, n_bootstrap=200, random_state=0)
    assert result["point_estimate"] == pytest.approx(always_positive.mean())
    assert result["probability_positive"] == 1.0
    assert result["interval_low"] <= result["point_estimate"] <= result["interval_high"]


# ---------------------------------------------------------------------------
# Leakage: a later week's snap/injury/depth revision never changes an
# earlier week's feature row.
# ---------------------------------------------------------------------------


def test_a_later_weeks_outcome_never_changes_an_earlier_weeks_features() -> None:
    baseline_sources = _build_synthetic_sources(seed=1)
    baseline_panel = build_player_week_panel(*baseline_sources)

    # Mutate ONLY the very last chronological week in the whole synthetic
    # dataset (week 4 of the final season) -- every row strictly before it,
    # by (season, week) ORDER rather than by the "week" column's own value
    # (which resets to 1 every season, so "week < 4" alone would wrongly
    # include season 2021's week 1-3 even though those come AFTER season
    # 2020's week 4), must come out byte-identical.
    last_season = max(SEASONS)
    mutated = {(last_season, team, 4): True for team in TEAMS}
    mutated_sources = _build_synthetic_sources(seed=1, extra_week=mutated)
    mutated_panel = build_player_week_panel(*mutated_sources)

    def _ordinal(frame: pd.DataFrame) -> pd.Series:
        return frame["season"] * 100 + frame["week"]

    earlier_baseline = (
        baseline_panel.loc[_ordinal(baseline_panel).lt(last_season * 100 + 4)]
        .sort_values(["season", "week", "team", "gsis_id"])
        .reset_index(drop=True)
    )
    earlier_mutated = (
        mutated_panel.loc[_ordinal(mutated_panel).lt(last_season * 100 + 4)]
        .sort_values(["season", "week", "team", "gsis_id"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(earlier_baseline, earlier_mutated)

    # Confirm the mutation actually changed something in the mutated week --
    # a leakage test that passes because nothing changed anywhere proves
    # nothing.
    later_baseline = baseline_panel.loc[
        baseline_panel["season"].eq(last_season) & baseline_panel["week"].eq(4)
    ]
    later_mutated = mutated_panel.loc[
        mutated_panel["season"].eq(last_season) & mutated_panel["week"].eq(4)
    ]
    assert not later_baseline["qb1_report_category"].equals(later_mutated["qb1_report_category"])


def test_serving_player_history_only_uses_strictly_earlier_weeks() -> None:
    _depth_history, rosters, snaps, _injuries = _build_synthetic_sources()
    history_before_week1 = serving_player_history(rosters, snaps, as_of_season=2020, as_of_week=1)
    # Nobody has played yet as of week 1 of the very first synthetic season.
    assert history_before_week1 == {}

    history_before_week3 = serving_player_history(rosters, snaps, as_of_season=2020, as_of_week=3)
    # Anyone with a recorded snap in week 1 or 2 of 2020 appears; nobody's
    # value can depend on week 3 itself or later.
    some_gsis_id = f"{TEAMS[0]}-WR1"
    if some_gsis_id in history_before_week3:
        assert history_before_week3[some_gsis_id]["weeks_since_last_snap"] >= 1.0


@pytest.mark.parametrize("timestamp_column", ["date_modified", "effective_observed_at"])
def test_post_decision_injury_revision_cannot_change_features(timestamp_column: str) -> None:
    depth, rosters, snaps, injuries = _build_synthetic_sources()
    injuries = injuries.rename(columns={"date_modified": timestamp_column})
    injuries.loc[injuries.index[0], "report_status"] = "Questionable"
    before = build_player_week_panel(depth, rosters, snaps, injuries)
    revision = injuries.iloc[[0]].copy()
    revision[timestamp_column] = pd.Timestamp("2030-01-01", tz="UTC")
    revision["report_status"] = "Out"
    revision["practice_status"] = "Full Participation in Practice"
    after = build_player_week_panel(depth, rosters, snaps, pd.concat([injuries, revision]))
    pd.testing.assert_frame_equal(before, after)
    revision[timestamp_column] = pd.Timestamp(f"{int(revision.iloc[0]['season'])}-09-01", tz="UTC")
    at_deadline = build_player_week_panel(depth, rosters, snaps, pd.concat([injuries, revision]))
    assert not before["report_category"].equals(at_deadline["report_category"])


def test_target_week_inactive_roster_status_is_not_a_feature() -> None:
    depth, rosters, snaps, injuries = _build_synthetic_sources()
    before = build_player_week_panel(depth, rosters, snaps, injuries)
    rosters["status"] = "INA"
    after = build_player_week_panel(depth, rosters, snaps, injuries)
    pd.testing.assert_frame_equal(before[list(FEATURE_COLUMNS)], after[list(FEATURE_COLUMNS)])
    assert "roster_status" not in FEATURE_COLUMNS


def test_daily_depth_uses_strict_pool_decision_time_and_retains_observation() -> None:
    frame = pd.DataFrame(
        [
            {
                "dt": dt,
                "team": "KC",
                "player_name": name,
                "gsis_id": name,
                "pos_abb": "QB",
                "pos_rank": 1,
            }
            for dt, name in [
                ("2025-09-07T19:59:59Z", "before"),
                ("2025-09-07T20:00:00Z", "exact"),
                ("2025-09-07T20:05:00Z", "after"),
            ]
        ]
    )
    schedule = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "home_team": "KC",
                "away_team": "DEN",
                "game_type": "REG",
                "gameday": "2025-09-07",
                "gametime": "16:25",
            }
        ]
    )
    result = canonicalize_depth_chart_history(frame, schedule)
    assert result["gsis_id"].tolist() == ["before"]
    assert result.iloc[0]["depth_observed_at"] == pd.Timestamp("2025-09-07T19:59:59Z")
    assert result.iloc[0]["decision_at"] == pd.Timestamp("2025-09-07T20:00:00Z")


def test_missing_snap_coverage_does_not_create_negative_labels() -> None:
    depth, rosters, snaps, injuries = _build_synthetic_sources()
    assert build_player_week_panel(depth, rosters, snaps.iloc[:0], injuries).empty
    missing = snaps.loc[
        ~(snaps["season"].eq(2020) & snaps["team"].eq(TEAMS[0]) & snaps["week"].eq(1))
    ]
    result = build_player_week_panel(depth, rosters, missing, injuries)
    assert not (
        result["season"].eq(2020) & result["team"].eq(TEAMS[0]) & result["week"].eq(1)
    ).any()


def test_starting_slots_follow_snap_leaders_not_listed_qb1() -> None:
    from nfl_ats.play_probability import _started_label

    depth, rosters, snaps, _ = _build_synthetic_sources()
    population = depth.loc[
        depth["season"].eq(2020) & depth["week"].eq(1) & depth["team"].eq(TEAMS[0])
    ]
    template = snaps.iloc[0].to_dict()
    rows = []
    for player, count in [("QB1", 1), ("QB2", 59), ("WR1", 60), ("WR2", 60), ("WR3", 60)]:
        gsis_id = f"{TEAMS[0]}-{player}"
        rows.append(
            {
                **template,
                "season": 2020,
                "week": 1,
                "team": TEAMS[0],
                "player": gsis_id,
                "pfr_player_id": gsis_id,
                "position": player[:2],
                "offense_snaps": count,
            }
        )
    labels = _started_label(population, rosters, pd.DataFrame(rows))
    actual = dict(zip(population["gsis_id"], labels, strict=True))
    assert not actual[f"{TEAMS[0]}-QB1"]
    assert actual[f"{TEAMS[0]}-QB2"]
    assert all(actual[f"{TEAMS[0]}-WR{i}"] for i in (1, 2, 3))
    rows[0]["offense_snaps"] = 59
    tied = _started_label(population, rosters, pd.DataFrame(rows))
    actual = dict(zip(population["gsis_id"], tied, strict=True))
    assert actual[f"{TEAMS[0]}-QB1"] and not actual[f"{TEAMS[0]}-QB2"]


def test_calibration_fold_is_disjoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nfl_ats.play_probability as module

    panel = build_player_week_panel(*_build_synthetic_sources())
    calls = []

    def spy(train: pd.DataFrame, *, label: str, calibration: pd.DataFrame) -> tuple[None, None]:
        calls.append((train.copy(), calibration.copy()))
        assert set(train.index).isdisjoint(calibration.index)
        return None, None

    monkeypatch.setattr(module, "_fit_one_label", spy)
    model = module.fit_play_probability_model(panel, scored_season=2023)
    assert model.train_seasons == (2020, 2021)
    assert model.calibration_season == 2022
    assert model.calibration_status == "held_out_previous_season"
    assert all(
        set(train["season"]) == {2020, 2021} and set(cal["season"]) == {2022}
        for train, cal in calls
    )
    fallback = module.fit_play_probability_model(panel, scored_season=2021)
    assert fallback.calibration_season is None
    assert fallback.calibration_status == "uncalibrated_insufficient_history"
    assert calls[-1][1].empty


def test_untimestamped_daily_archive_cannot_reenter_training() -> None:
    depth, rosters, snaps, injuries = _build_synthetic_sources()
    depth["source_schema"] = "daily_dt"
    assert build_player_week_panel(depth, rosters, snaps, injuries).empty


def test_panel_builder_resolves_newest_injuries_and_accepts_pin(tmp_path: Path) -> None:
    builder = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/build_play_probability_panel.py")
    )
    resolve = builder["resolve_injuries_path"]
    resolve.__globals__["RAW_INJURIES_ROOT"] = tmp_path
    with pytest.raises(FileNotFoundError, match="ingest separately"):
        resolve()
    old = tmp_path / "20260826T122850Z" / "injuries.parquet"
    new = tmp_path / "20260905T211248Z" / "injuries.parquet"
    for path in (new, old):
        path.parent.mkdir()
        pd.DataFrame({"season": [2025]}).to_parquet(path)
    # A newer incomplete snapshot must not shadow a usable archive.
    (tmp_path / "20260906T000000Z").mkdir()
    assert resolve() == new
    assert resolve(old) == old
    with pytest.raises(FileNotFoundError, match="No local injury archive"):
        resolve(tmp_path / "missing.parquet")


def test_newest_injury_snapshot_cannot_leak_later_revisions(tmp_path: Path) -> None:
    builder = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/build_play_probability_panel.py")
    )
    resolve = builder["resolve_injuries_path"]
    resolve.__globals__["RAW_INJURIES_ROOT"] = tmp_path
    depth, rosters, snaps, injuries = _build_synthetic_sources()
    old = tmp_path / "20260826T122850Z" / "injuries.parquet"
    new = tmp_path / "20260905T211248Z" / "injuries.parquet"
    old.parent.mkdir()
    new.parent.mkdir()
    injuries.to_parquet(old)
    revision = injuries.iloc[[0]].copy()
    revision["date_modified"] = pd.Timestamp("2030-01-01", tz="UTC")
    revision["report_status"] = "Out"
    pd.concat([injuries, revision], ignore_index=True).to_parquet(new)
    before = build_player_week_panel(depth, rosters, snaps, pd.read_parquet(resolve(old)))
    after = build_player_week_panel(depth, rosters, snaps, pd.read_parquet(resolve()))
    pd.testing.assert_frame_equal(before, after)


def test_uncalibrated_early_season_still_predicts_valid_subset_probabilities() -> None:
    panel = build_player_week_panel(*_build_synthetic_sources())
    model = fit_play_probability_model(panel, scored_season=2021)
    assert model.played_calibrator is None and model.started_calibrator is None
    predicted = predict_play_probabilities(model, panel.loc[panel["season"].eq(2021)])
    assert predicted["play_probability"].between(0, 1).all()
    assert predicted["start_probability"].between(0, predicted["play_probability"]).all()


def test_panel_daily_snapshot_2025_excludes_future_dt_and_preserves_history(tmp_path: Path) -> None:
    builder = runpy.run_path(
        str(Path(__file__).parents[1] / "scripts/build_play_probability_panel.py")
    )
    loader = builder["load_panel_depth_history"]
    legacy = pd.DataFrame(
        [{"season": 2024, "gsis_id": "legacy"}, {"season": 2025, "gsis_id": "unverified"}]
    )
    loader.__globals__["_load_or_fetch_depth_history"] = lambda *args: legacy.copy()
    loader.__globals__["RAW_DEPTH_ROOT"] = tmp_path
    schedule = pd.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "home_team": "KC",
                "away_team": "DEN",
                "game_type": "REG",
                "gameday": "2025-09-07",
                "gametime": "16:25",
            }
        ]
    )

    def snapshot(stamp: str, rows: list[dict[str, object]]) -> None:
        directory = tmp_path / stamp
        directory.mkdir()
        pd.DataFrame(rows).to_parquet(directory / "depth_charts.parquet")
        (directory / "manifest.json").write_text(json.dumps({"requested_seasons": [2025]}))

    before = {
        "dt": "2025-09-07T19:59:59Z",
        "team": "KC",
        "player_name": "before",
        "gsis_id": "before",
        "pos_abb": "QB",
        "pos_rank": 1,
        "upstream_extra": "retained",
    }
    snapshot("20260905T000000Z", [before])
    expected = loader(2024, 2025, schedule)
    snapshot(
        "20260906T000000Z",
        [before, {**before, "dt": "2025-09-07T20:01:00Z", "gsis_id": "future", "pos_rank": 2}],
    )
    (tmp_path / "20260907T000000Z").mkdir()
    actual = loader(2024, 2025, schedule)
    pd.testing.assert_frame_equal(expected, actual)
    assert actual["gsis_id"].tolist() == ["legacy", "before"]
    assert actual.iloc[1]["depth_observed_at"] < actual.iloc[1]["decision_at"]
    assert "20260906" in actual.attrs["raw_2025_depth_source"]
    pd.testing.assert_frame_equal(loader(2024, 2024, schedule), legacy.iloc[:1])
