"""Tests for the inactives refresh-time overlay (WP41).

The rule under test is frozen in ``docs/inactives_channel.md``'s
"Prospective wiring predeclaration (2026-09-01, WP41)" section, written before
``src/nfl_ats/inactives_refresh_overlay.py`` existed.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from nfl_ats.active_model import ACTIVE_ATS_MODEL_VERSION
from nfl_ats.availability import fixed_unavailability
from nfl_ats.clv import PAPER_DECISION_COLUMNS, paper_decision_ledger_path
from nfl_ats.constants import FEATURE_SETS
from nfl_ats.inactives_refresh_overlay import (
    CHALLENGER_ID,
    INACTIVES_LEAD_MINUTES,
    INACTIVES_REFRESH_OVERLAY_COLUMNS,
    SOURCE_NO_SNAPSHOT,
    SOURCE_STRUCTURALLY_EXCLUDED,
    apply_inactives_increments,
    build_inactives_refresh_overlay_rows,
    load_inactives_snapshots,
    load_player_context,
    newest_snapshot_before,
    record_inactives_refresh_overlay,
    snapshot_source_tag,
    structurally_excluded,
    team_unavailability_increments,
)
from nfl_ats.io import atomic_json, atomic_parquet
from nfl_ats.pick_refresh import plan_refresh
from nfl_ats.players import PLAYER_INJURY_STATE_METRICS
from nfl_ats.prospective_scoring import (
    ACTIVE_CHALLENGER_STATUS,
    config_fingerprint,
    find_challenger,
)

MIN_TRAIN_GAMES = 50
SEASON, WEEK = 2026, 2
FEATURE_PROFILE = "player_injuries"

# A Tue..Mon NFL week: Tuesday 2026-09-15 through Monday 2026-09-21, in UTC
# (September is EDT, UTC-4). Mirrors tests/test_pick_refresh.py's own anchors
# so the two files agree on what a "week" looks like.
TNF_KICKOFF = pd.Timestamp("2026-09-18T00:15:00+00:00")  # Thu 8:15pm ET
SUN_EARLY_KICKOFF = pd.Timestamp("2026-09-20T17:00:00+00:00")  # Sun 1:00pm ET
SUN_LATE_KICKOFF = pd.Timestamp("2026-09-20T20:25:00+00:00")  # Sun 4:25pm ET
SNF_KICKOFF = pd.Timestamp("2026-09-21T00:20:00+00:00")  # Sun 8:20pm ET
MNF_KICKOFF = pd.Timestamp("2026-09-22T00:15:00+00:00")  # Mon 8:15pm ET
SUNDAY_LOCK = pd.Timestamp("2026-09-20T20:00:00+00:00")  # Sun 4:00pm ET

NOW = pd.Timestamp("2026-09-20T16:00:00+00:00")  # Sun noon ET, before every deadline

POLICY_ID = "overlay_union_coach_division_revenge_player_arrests_spread_gap_v1"

GAMES: list[dict[str, Any]] = [
    {
        "game_id": "2026_02_AAA_BBB",
        "away_team": "AAA",
        "home_team": "BBB",
        "kickoff": SUN_EARLY_KICKOFF,
        "gameday": pd.Timestamp("2026-09-20"),
    },
    {
        "game_id": "2026_02_CCC_DDD",
        "away_team": "CCC",
        "home_team": "DDD",
        "kickoff": SUN_LATE_KICKOFF,
        "gameday": pd.Timestamp("2026-09-20"),
    },
    {
        "game_id": "2026_02_EEE_FFF",
        "away_team": "EEE",
        "home_team": "FFF",
        "kickoff": SNF_KICKOFF,
        "gameday": pd.Timestamp("2026-09-20"),
    },
    {
        "game_id": "2026_02_GGG_HHH",
        "away_team": "GGG",
        "home_team": "HHH",
        "kickoff": MNF_KICKOFF,
        "gameday": pd.Timestamp("2026-09-21"),
    },
]

INJURY_DIFF_DRIVER = "diff_injury_offense_unavailability"


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _feature_table(target_injury_diff: dict[str, float]) -> pd.DataFrame:
    """Training history plus this week's four unplayed games.

    Training ``ats_margin`` is a strong, monotone, DETERMINISTIC function of
    ``diff_injury_offense_unavailability`` (home hurt more -> home covers
    less), so the fitted ridge has a large known coefficient on the one column
    the overlay moves. That is what makes "the pick flips only when the
    recomputed probability crosses 0.5" testable at all: with a flat model the
    probability could never move.
    """

    rows = 160
    index = np.arange(rows)
    start = date(2024, 9, 1)
    frame = pd.DataFrame(
        {
            "game_id": [f"train_{value:03d}" for value in index],
            "season": np.where(index < 100, 2024, 2025),
            "week": np.where(index < 100, (index // 10) + 1, ((index - 100) // 15) + 1),
            "gameday": [start + timedelta(days=int(value)) for value in index],
            "kickoff": [
                pd.Timestamp(start + timedelta(days=int(value)), tz="UTC") for value in index
            ],
            "away_team": "AWY",
            "home_team": "HME",
            "home_spread_odds": -110.0,
            "away_spread_odds": -110.0,
            "game_type": "REG",
        }
    )
    # Every feature except the driver is held flat, so the fitted ridge is a
    # one-variable model and the direction of the injury effect is
    # unambiguous: more home unavailability -> a worse home margin.
    for column in FEATURE_SETS[f"full_{FEATURE_PROFILE}"]:
        frame[column] = 0.0
    # The one driver column: a clean sweep across the range a real inactives
    # increment moves (a full-time offensive player is 1.0/11 = 0.0909).
    driver = np.linspace(-0.25, 0.25, rows)
    frame[INJURY_DIFF_DRIVER] = driver
    for metric in PLAYER_INJURY_STATE_METRICS:
        frame[f"home_{metric}"] = 0.0
        frame[f"away_{metric}"] = 0.0
    frame["home_injury_offense_unavailability"] = np.clip(driver, 0.0, None)
    frame["away_injury_offense_unavailability"] = np.clip(-driver, 0.0, None)
    frame["spread_line"] = 0.0
    # Deterministic residual spread (sd ~2.1 points) so the gaussian
    # probability method returns something smooth rather than saturating.
    frame["ats_margin"] = -60.0 * driver + 3.0 * np.sin(index)
    frame["result"] = frame["spread_line"] + frame["ats_margin"]
    frame["home_cover"] = (frame["ats_margin"] > 0).astype(float)

    template = frame.iloc[0]
    feature_columns = [
        column
        for column in frame.columns
        if column
        not in {
            "game_id",
            "season",
            "week",
            "gameday",
            "kickoff",
            "away_team",
            "home_team",
            "game_type",
            "home_spread_odds",
            "away_spread_odds",
            "spread_line",
            "home_cover",
            "ats_margin",
            "result",
        }
    ]
    target_rows = []
    for game in GAMES:
        row = {column: float(template[column]) for column in feature_columns}
        for metric in PLAYER_INJURY_STATE_METRICS:
            row[f"home_{metric}"] = 0.0
            row[f"away_{metric}"] = 0.0
            row[f"diff_{metric}"] = 0.0
        value = target_injury_diff[game["game_id"]]
        row["home_injury_offense_unavailability"] = max(value, 0.0)
        row["away_injury_offense_unavailability"] = max(-value, 0.0)
        row[INJURY_DIFF_DRIVER] = value
        row.update(
            {
                "game_id": game["game_id"],
                "season": SEASON,
                "week": WEEK,
                "gameday": game["gameday"],
                "kickoff": game["kickoff"],
                "away_team": game["away_team"],
                "home_team": game["home_team"],
                "game_type": "REG",
                "home_spread_odds": -110.0,
                "away_spread_odds": -110.0,
                "spread_line": 0.0,
                "home_cover": np.nan,
                "ats_margin": np.nan,
                "result": np.nan,
            }
        )
        target_rows.append(row)
    return pd.concat([frame, pd.DataFrame(target_rows)], ignore_index=True, sort=False)


def _write_active_manifest(artifacts_root: Path) -> None:
    atomic_json(
        {
            "version": ACTIVE_ATS_MODEL_VERSION,
            "status": "SYNCHRONIZED",
            "method": "market_residual",
            "feature_profile": FEATURE_PROFILE,
            "regressor": "ridge",
            "ridge_alpha": 10.0,
            "probability_method": "gaussian",
            "model_id": "model-inactives-test",
        },
        artifacts_root / "active_ats_model.json",
    )


def _write_original_card(artifacts_root: Path, *, pick_sides: dict[str, str]) -> pd.DataFrame:
    rows = []
    for game in GAMES:
        rows.append(
            {
                "recorded_at_utc": pd.Timestamp("2026-09-15T13:00:00+00:00"),
                "forecast_artifact": "margin_predictions/test",
                "forecast_created_at_utc": pd.Timestamp("2026-09-15T13:00:00+00:00"),
                "model_id": "model-inactives-test",
                "method": "market_residual",
                "decision_policy_id": POLICY_ID,
                "decision_policy_fingerprint": "test-policy-fingerprint",
                "game_id": game["game_id"],
                "season": SEASON,
                "week": WEEK,
                "kickoff": game["kickoff"],
                "away_team": game["away_team"],
                "home_team": game["home_team"],
                "model_pick_side": pick_sides[game["game_id"]],
                "pre_arrest_pick_side": pick_sides[game["game_id"]],
                "former_policy_pick_side": pick_sides[game["game_id"]],
                "pick_side": pick_sides[game["game_id"]],
                "coach_fade_flip": False,
                "division_revenge_flip": False,
                "player_arrests_flip": False,
                "spread_gap_zone_flip": False,
                "composed_overlay_flip": False,
                "player_arrests_home_flag": False,
                "player_arrests_away_flag": False,
                "player_arrests_snapshot_id": "snapshot-tuesday",
                "player_arrests_snapshot_fetched_at_utc": pd.Timestamp("2026-09-15T12:00:00+00:00"),
                "player_arrests_safe_index_sha256": "safe-index-hash",
                "schedule_snapshot_id": "schedule-tuesday",
                "schedule_parquet_sha256": "schedule-hash",
                "bet_side": "PASS",
                "decision_home_spread": 0.0,
                "edge": np.nan,
                "is_best_pick": False,
            }
        )
    frame = pd.DataFrame(rows)
    atomic_parquet(frame[list(PAPER_DECISION_COLUMNS)], paper_decision_ledger_path(artifacts_root))
    return frame


def _write_player_snapshot(
    data_root: Path,
    *,
    snapshot_id: str = "20260915T000000Z",
    injuries: list[dict[str, Any]] | None = None,
    players: list[dict[str, Any]] | None = None,
) -> None:
    """A minimal but contract-complete injuries/rosters/snaps snapshot.

    ``players`` rows carry ``gsis_id``, ``full_name``, ``team``, ``position``
    and the three prior-week snap shares the overlay reads as role shares.
    """

    players = players or []
    roster_rows = [
        {
            "season": SEASON,
            "team": player["team"],
            "position": player["position"],
            "status": "ACT",
            "full_name": player["full_name"],
            "gsis_id": player["gsis_id"],
            "pfr_id": player["gsis_id"],
            "years_exp": 3,
            "week": 1,
            "game_type": "REG",
        }
        for player in players
    ]
    snap_rows = [
        {
            "game_id": f"2026_01_{player['team']}",
            "season": SEASON,
            "game_type": "REG",
            "week": 1,
            "player": player["full_name"],
            "pfr_player_id": player["gsis_id"],
            "position": player["position"],
            "team": player["team"],
            "offense_snaps": 60,
            "offense_pct": player.get("offense_pct", 0.0),
            "defense_snaps": 0,
            "defense_pct": player.get("defense_pct", 0.0),
            "st_snaps": 0,
            "st_pct": player.get("st_pct", 0.0),
        }
        for player in players
    ]
    injury_rows = injuries or []
    root = data_root / "players" / "raw" / snapshot_id
    atomic_parquet(
        pd.DataFrame(
            injury_rows,
            columns=[
                "season",
                "game_type",
                "team",
                "week",
                "gsis_id",
                "position",
                "report_status",
                "practice_status",
                "date_modified",
            ],
        ),
        root / "injuries.parquet",
    )
    atomic_parquet(pd.DataFrame(roster_rows), root / "weekly_rosters.parquet")
    atomic_parquet(pd.DataFrame(snap_rows), root / "snap_counts.parquet")
    atomic_json(
        {
            "snapshot_id": snapshot_id,
            "injury_seasons": [SEASON],
            "roster_seasons": [SEASON],
            "snap_seasons": [SEASON],
        },
        root / "manifest.json",
    )


def _write_inactives_snapshot(
    data_root: Path,
    *,
    snapshot_id: str,
    captured_at: pd.Timestamp,
    rows: list[dict[str, Any]],
    season: int | None = SEASON,
    week: int | None = WEEK,
    empty_reason: str | None = None,
) -> Path:
    root = data_root / "players" / "inactives" / snapshot_id
    frame = pd.DataFrame(
        rows,
        columns=[
            "captured_at_utc",
            "season",
            "week",
            "game_id",
            "home_team",
            "away_team",
            "team",
            "player_name",
            "position",
            "status",
            "source_url",
        ],
    )
    atomic_parquet(frame, root / "inactives.parquet")
    atomic_json(
        {
            "schema": "nflcom_inactives_snapshot/1",
            "snapshot_id": snapshot_id,
            "captured_at_utc": pd.Timestamp(captured_at).isoformat(),
            "slot": "sun_early",
            "season": season,
            "week": week,
            "source_used": "primary" if rows else "none",
            "row_count": len(rows),
            "teams_seen": sorted({str(row["team"]) for row in rows}),
            "empty_reason": empty_reason,
            "warnings": [],
            "ok": True,
        },
        root / "manifest.json",
    )
    return root


def _inactive_row(team: str, player_name: str, position: str) -> dict[str, Any]:
    game = next(
        (
            candidate
            for candidate in GAMES
            if team in {candidate["away_team"], candidate["home_team"]}
        ),
        None,
    )
    return {
        "captured_at_utc": "2026-09-20T15:30:00Z",
        "season": SEASON,
        "week": WEEK,
        "game_id": None if game is None else game["game_id"],
        "home_team": None if game is None else game["home_team"],
        "away_team": None if game is None else game["away_team"],
        "team": team,
        "player_name": player_name,
        "position": position,
        "status": "Inactive",
        "source_url": "https://www.nfl.com/inactives/",
    }


@pytest.fixture
def env(tmp_path: Path) -> tuple[Path, Path, Path]:
    """artifacts_root, data_root, features_path with the whole week wired up."""

    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_active_manifest(artifacts_root)
    # Every target game sits just far enough onto the AWAY side of 0.5 that a
    # full-time away-team inactive crosses it and a 1%-share one does not --
    # the exact pair the "flips only when the probability crosses 0.5" test
    # needs. The value is read off the fixture's own fitted response, not
    # tuned against any real data.
    features = _feature_table({game["game_id"]: -0.01 for game in GAMES})
    features_path = tmp_path / "features.parquet"
    atomic_parquet(features, features_path)
    _write_original_card(artifacts_root, pick_sides={game["game_id"]: "AWAY" for game in GAMES})
    return artifacts_root, data_root, features_path


def _plan(artifacts_root: Path, data_root: Path, features_path: Path) -> Any:
    return plan_refresh(
        artifacts_root,
        data_root,
        season=SEASON,
        week=WEEK,
        features_path=features_path,
        min_train_games=MIN_TRAIN_GAMES,
        now=NOW.to_pydatetime(),
    )


def _rows(
    artifacts_root: Path, data_root: Path, features_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    plan = _plan(artifacts_root, data_root, features_path)
    return build_inactives_refresh_overlay_rows(
        plan,
        artifacts_root=artifacts_root,
        data_root=data_root,
        min_train_games=MIN_TRAIN_GAMES,
    )


# ---------------------------------------------------------------------------
# Deadline arithmetic: SNF/MNF are structurally excluded, the 4:25 slot is not
# ---------------------------------------------------------------------------


def test_structural_exclusion_matches_the_measured_slot_table() -> None:
    """docs/inactives_channel.md Section 2's measured playability, in code."""

    assert not structurally_excluded(SUN_EARLY_KICKOFF, SUN_EARLY_KICKOFF)
    # The 4:25 ET slot: deadline is the 4:00 ET lock, EARLIER than kickoff, but
    # T-90 (2:55 ET) still precedes it -- the naive "deadline < kickoff" test
    # would wrongly exclude this slot.
    assert SUNDAY_LOCK < SUN_LATE_KICKOFF
    assert not structurally_excluded(SUN_LATE_KICKOFF, SUNDAY_LOCK)
    assert structurally_excluded(SNF_KICKOFF, SUNDAY_LOCK)
    assert structurally_excluded(MNF_KICKOFF, SUNDAY_LOCK)


def test_inactives_lead_is_the_reported_ninety_minute_convention() -> None:
    assert INACTIVES_LEAD_MINUTES == 90


# ---------------------------------------------------------------------------
# Snapshot selection: in-window, out-of-window, anti-backdating
# ---------------------------------------------------------------------------


def test_only_the_snapshot_captured_before_the_deadline_is_selected(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_inactives_snapshot(
        data_root,
        snapshot_id="20260920T153000Z",
        captured_at=pd.Timestamp("2026-09-20T15:30:00+00:00"),
        rows=[_inactive_row("BBB", "Real Starter", "WR")],
    )
    _write_inactives_snapshot(
        data_root,
        snapshot_id="20260920T190000Z",
        captured_at=pd.Timestamp("2026-09-20T19:00:00+00:00"),
        rows=[_inactive_row("BBB", "Later Starter", "WR")],
    )
    snapshots = load_inactives_snapshots(data_root)
    assert [snapshot.snapshot_id for snapshot in snapshots] == [
        "20260920T153000Z",
        "20260920T190000Z",
    ]
    chosen = newest_snapshot_before(
        snapshots, SUN_EARLY_KICKOFF, season=SEASON, week=WEEK
    )  # deadline == the 1:00pm ET kickoff
    assert chosen is not None
    assert chosen.snapshot_id == "20260920T153000Z"


def test_a_snapshot_captured_after_kickoff_never_applies(tmp_path: Path) -> None:
    """Anti-backdating: the deadline is at most the kickoff, and selection is
    STRICTLY before the deadline, so a post-kickoff capture is unreachable."""

    data_root = tmp_path / "data"
    _write_inactives_snapshot(
        data_root,
        snapshot_id="20260920T180000Z",
        captured_at=SUN_EARLY_KICKOFF + pd.Timedelta(hours=1),
        rows=[_inactive_row("BBB", "Too Late", "WR")],
    )
    snapshots = load_inactives_snapshots(data_root)
    assert newest_snapshot_before(snapshots, SUN_EARLY_KICKOFF, season=SEASON, week=WEEK) is None
    # A capture at exactly the deadline is also refused (strict inequality).
    _write_inactives_snapshot(
        data_root,
        snapshot_id="20260920T170000Z",
        captured_at=SUN_EARLY_KICKOFF,
        rows=[_inactive_row("BBB", "Exactly At Kickoff", "WR")],
    )
    snapshots = load_inactives_snapshots(data_root)
    assert newest_snapshot_before(snapshots, SUN_EARLY_KICKOFF, season=SEASON, week=WEEK) is None


def test_a_zero_row_snapshot_is_not_an_in_window_snapshot(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_inactives_snapshot(
        data_root,
        snapshot_id="20260901T120000Z",
        captured_at=pd.Timestamp("2026-09-20T15:30:00+00:00"),
        rows=[],
        empty_reason="primary_offseason_placeholder",
    )
    snapshots = load_inactives_snapshots(data_root)
    assert len(snapshots) == 1
    assert not snapshots[0].reported_inactives
    assert newest_snapshot_before(snapshots, SUN_EARLY_KICKOFF, season=SEASON, week=WEEK) is None


def test_a_snapshot_for_a_different_week_is_ignored(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_inactives_snapshot(
        data_root,
        snapshot_id="20260913T153000Z",
        captured_at=pd.Timestamp("2026-09-20T15:30:00+00:00"),
        rows=[_inactive_row("BBB", "Last Week", "WR")],
        week=WEEK - 1,
    )
    snapshots = load_inactives_snapshots(data_root)
    assert newest_snapshot_before(snapshots, SUN_EARLY_KICKOFF, season=SEASON, week=WEEK) is None


def test_stale_and_future_dated_snapshots_are_not_available_at_decision_time(
    tmp_path: Path,
) -> None:
    """A same-week manifest is insufficient: it must be today's, already-seen report."""

    data_root = tmp_path / "data"
    _write_inactives_snapshot(
        data_root,
        snapshot_id="20260919T153000Z",
        captured_at=pd.Timestamp("2026-09-19T15:30:00+00:00"),
        rows=[_inactive_row("BBB", "Yesterday Starter", "WR")],
    )
    _write_inactives_snapshot(
        data_root,
        snapshot_id="20260920T163000Z",
        captured_at=pd.Timestamp("2026-09-20T16:30:00+00:00"),
        rows=[_inactive_row("BBB", "Future Starter", "WR")],
    )
    snapshots = load_inactives_snapshots(data_root)

    assert (
        newest_snapshot_before(
            snapshots,
            SUN_EARLY_KICKOFF,
            season=SEASON,
            week=WEEK,
            now=NOW,
            game_day=SUN_EARLY_KICKOFF,
        )
        is None
    )


# ---------------------------------------------------------------------------
# The P(plays) = 0 override, through production's own aggregation
# ---------------------------------------------------------------------------


def test_out_is_already_credited_so_the_increment_is_zero(tmp_path: Path) -> None:
    """A player the report already ruled Out adds NOTHING -- no double-count.

    ``fixed_unavailability("Out", ...) == 1.0`` is production's own mapping, so
    the P(plays)=0 override's increment for that player is exactly 0.0.
    """

    assert fixed_unavailability("Out", "") == 1.0
    data_root = tmp_path / "data"
    _write_player_snapshot(
        data_root,
        players=[
            {
                "gsis_id": "00-0000001",
                "full_name": "Already Out",
                "team": "BBB",
                "position": "WR",
                "offense_pct": 1.0,
            },
            {
                "gsis_id": "00-0000002",
                "full_name": "Surprise Absence",
                "team": "BBB",
                "position": "WR",
                "offense_pct": 1.0,
            },
        ],
        injuries=[
            {
                "season": SEASON,
                "game_type": "REG",
                "team": "BBB",
                "week": WEEK,
                "gsis_id": "00-0000001",
                "position": "WR",
                "report_status": "Out",
                "practice_status": "Did Not Participate In Practice",
                "date_modified": pd.Timestamp("2026-09-19T12:00:00+00:00"),
            }
        ],
    )
    context = load_player_context(
        data_root, season=SEASON, week=WEEK, cutoff=pd.Timestamp("2026-09-20T15:30:00+00:00")
    )
    assert context is not None

    already_out = pd.DataFrame([_inactive_row("BBB", "Already Out", "WR")])
    increments, listed = team_unavailability_increments(
        already_out, context, season=SEASON, week=WEEK, team="BBB"
    )
    assert listed == 1
    assert increments == dict.fromkeys(PLAYER_INJURY_STATE_METRICS, 0.0)

    surprise = pd.DataFrame([_inactive_row("BBB", "Surprise Absence", "WR")])
    increments, listed = team_unavailability_increments(
        surprise, context, season=SEASON, week=WEEK, team="BBB"
    )
    assert listed == 1
    # production's own arithmetic: severity 1.0 x offense share 1.0 / 11
    assert increments["injury_offense_unavailability"] == pytest.approx(1.0 / 11.0)
    assert increments["injury_skill_unavailability"] == pytest.approx(1.0 / 6.0)


def test_a_questionable_player_contributes_only_the_remaining_increment(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_player_snapshot(
        data_root,
        players=[
            {
                "gsis_id": "00-0000003",
                "full_name": "Questionable Starter",
                "team": "BBB",
                "position": "WR",
                "offense_pct": 1.0,
            }
        ],
        injuries=[
            {
                "season": SEASON,
                "game_type": "REG",
                "team": "BBB",
                "week": WEEK,
                "gsis_id": "00-0000003",
                "position": "WR",
                "report_status": "Questionable",
                "practice_status": "Limited Participation In Practice",
                "date_modified": pd.Timestamp("2026-09-19T12:00:00+00:00"),
            }
        ],
    )
    context = load_player_context(
        data_root, season=SEASON, week=WEEK, cutoff=pd.Timestamp("2026-09-20T15:30:00+00:00")
    )
    assert context is not None
    increments, _ = team_unavailability_increments(
        pd.DataFrame([_inactive_row("BBB", "Questionable Starter", "WR")]),
        context,
        season=SEASON,
        week=WEEK,
        team="BBB",
    )
    remaining = 1.0 - fixed_unavailability("Questionable", "")
    assert increments["injury_offense_unavailability"] == pytest.approx(remaining / 11.0)


def test_apply_increments_moves_only_the_named_game(tmp_path: Path) -> None:
    features = _feature_table({game["game_id"]: 0.0 for game in GAMES})
    increments = {
        GAMES[0]["game_id"]: {
            "home": dict.fromkeys(PLAYER_INJURY_STATE_METRICS, 0.0)
            | {"injury_offense_unavailability": 0.25},
            "away": dict.fromkeys(PLAYER_INJURY_STATE_METRICS, 0.0),
        }
    }
    adjusted = apply_inactives_increments(features, increments)
    target = adjusted["game_id"].astype(str).eq(GAMES[0]["game_id"])
    assert adjusted.loc[target, "home_injury_offense_unavailability"].iloc[0] == pytest.approx(0.25)
    assert adjusted.loc[target, INJURY_DIFF_DRIVER].iloc[0] == pytest.approx(0.25)
    untouched = ~target
    pd.testing.assert_frame_equal(
        adjusted.loc[untouched].reset_index(drop=True),
        features.loc[untouched].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# The full pass
# ---------------------------------------------------------------------------


def test_no_snapshot_store_leaves_every_game_on_the_tuesday_card(
    env: tuple[Path, Path, Path],
) -> None:
    artifacts_root, data_root, features_path = env
    rows, diagnostics = _rows(artifacts_root, data_root, features_path)

    assert list(rows.columns) == list(INACTIVES_REFRESH_OVERLAY_COLUMNS)
    assert len(rows) == len(GAMES)
    snf_mnf = {GAMES[2]["game_id"], GAMES[3]["game_id"]}
    for row in rows.to_dict("records"):
        expected = SOURCE_STRUCTURALLY_EXCLUDED if row["game_id"] in snf_mnf else SOURCE_NO_SNAPSHOT
        assert row["source"] == expected
        assert row["inactives_pick_side"] == row["played_pick_side"]
        assert row["inactives_pick_side"] == row["tuesday_pick_side"]
        assert not row["inactives_flip_vs_played"]
        assert not row["inactives_flip_vs_tuesday"]
    assert diagnostics["games_adjusted"] == []
    assert diagnostics["would_flip_vs_played_game_ids"] == []


def test_snf_and_mnf_are_untouched_even_with_a_live_snapshot(
    env: tuple[Path, Path, Path],
) -> None:
    artifacts_root, data_root, features_path = env
    _write_player_snapshot(
        data_root,
        players=[
            {
                "gsis_id": "00-0000010",
                "full_name": "Snf Starter",
                "team": "EEE",
                "position": "WR",
                "offense_pct": 1.0,
            },
            {
                "gsis_id": "00-0000011",
                "full_name": "Mnf Starter",
                "team": "GGG",
                "position": "WR",
                "offense_pct": 1.0,
            },
        ],
    )
    _write_inactives_snapshot(
        data_root,
        snapshot_id="20260920T153000Z",
        captured_at=pd.Timestamp("2026-09-20T15:30:00+00:00"),
        rows=[
            _inactive_row("EEE", "Snf Starter", "WR"),
            _inactive_row("GGG", "Mnf Starter", "WR"),
        ],
    )
    rows, diagnostics = _rows(artifacts_root, data_root, features_path)
    indexed = rows.set_index("game_id")
    for game_id in (GAMES[2]["game_id"], GAMES[3]["game_id"]):
        assert indexed.loc[game_id, "source"] == SOURCE_STRUCTURALLY_EXCLUDED
        assert (
            indexed.loc[game_id, "inactives_pick_side"] == indexed.loc[game_id, "played_pick_side"]
        )
        assert indexed.loc[game_id, "home_inactives_listed"] == 0
    assert diagnostics["games_adjusted"] == []


def test_a_team_or_game_misaligned_snapshot_fails_closed(
    env: tuple[Path, Path, Path],
) -> None:
    """A slate-wide capture may only affect the game it explicitly identifies."""

    artifacts_root, data_root, features_path = env
    bad = _inactive_row("AAA", "Wrong Game Starter", "WR")
    bad["game_id"] = "2026_02_CCC_DDD"
    bad["home_team"] = "DDD"
    bad["away_team"] = "CCC"
    _write_inactives_snapshot(
        data_root,
        snapshot_id="20260920T143000Z",
        captured_at=pd.Timestamp("2026-09-20T14:30:00+00:00"),
        rows=[bad],
    )

    rows, diagnostics = _rows(artifacts_root, data_root, features_path)
    first = rows.set_index("game_id").loc[GAMES[0]["game_id"]]
    assert first["source"] == SOURCE_NO_SNAPSHOT
    assert first["inactives_pick_side"] == first["played_pick_side"]
    assert diagnostics["games_adjusted"] == []


def test_an_inactive_starter_flips_the_pick_only_when_the_probability_crosses_half(
    env: tuple[Path, Path, Path],
) -> None:
    """The flip is a consequence of the recomputed probability, not of the list.

    Same player, same game, same snapshot -- only the player's prior-week snap
    share differs. The tiny-share arm moves the probability and does NOT cross
    0.5, so the pick holds; the full-time arm crosses and the pick flips.
    """

    artifacts_root, data_root, features_path = env
    game_id = GAMES[0]["game_id"]

    baseline, _ = _rows(artifacts_root, data_root, features_path)
    played = baseline.set_index("game_id").loc[game_id]
    assert played["played_pick_side"] == "AWAY"
    assert float(played["played_home_cover_probability"]) < 0.5

    def _run(offense_pct: float) -> pd.Series:
        for stale in (data_root / "players" / "inactives").glob("*"):
            for path in stale.iterdir():
                path.unlink()
            stale.rmdir()
        _write_player_snapshot(
            data_root,
            snapshot_id="20260915T000000Z",
            players=[
                {
                    "gsis_id": "00-0000020",
                    "full_name": "Away Starter",
                    "team": "AAA",
                    "position": "WR",
                    "offense_pct": offense_pct,
                }
            ],
        )
        _write_inactives_snapshot(
            data_root,
            snapshot_id="20260920T153000Z",
            captured_at=pd.Timestamp("2026-09-20T15:30:00+00:00"),
            rows=[_inactive_row("AAA", "Away Starter", "WR")],
        )
        rows, _diagnostics = _rows(artifacts_root, data_root, features_path)
        return rows.set_index("game_id").loc[game_id]

    tiny = _run(0.01)
    assert tiny["source"] == snapshot_source_tag("20260920T153000Z")
    assert tiny["away_inactives_listed"] == 1
    assert float(tiny["inactives_home_cover_probability"]) > float(
        played["played_home_cover_probability"]
    )
    assert float(tiny["inactives_home_cover_probability"]) < 0.5
    assert tiny["inactives_pick_side"] == "AWAY"
    assert not tiny["inactives_flip_vs_played"]

    full = _run(1.0)
    assert float(full["inactives_home_cover_probability"]) >= 0.5
    assert full["inactives_pick_side"] == "HOME"
    assert bool(full["inactives_flip_vs_played"])
    assert bool(full["inactives_flip_vs_tuesday"])


def test_the_pick_always_follows_the_recomputed_probability(
    env: tuple[Path, Path, Path],
) -> None:
    """No market quotes are written, so every game runs the model-only arm and
    the >= 0.5 forced-pick rule must hold on every recorded row."""

    artifacts_root, data_root, features_path = env
    _write_player_snapshot(
        data_root,
        players=[
            {
                "gsis_id": "00-0000030",
                "full_name": "Away Starter",
                "team": "AAA",
                "position": "WR",
                "offense_pct": 1.0,
            }
        ],
    )
    _write_inactives_snapshot(
        data_root,
        snapshot_id="20260920T153000Z",
        captured_at=pd.Timestamp("2026-09-20T15:30:00+00:00"),
        rows=[_inactive_row("AAA", "Away Starter", "WR")],
    )
    rows, _diagnostics = _rows(artifacts_root, data_root, features_path)
    for row in rows.to_dict("records"):
        expected = "HOME" if float(row["inactives_home_cover_probability"]) >= 0.5 else "AWAY"
        assert row["inactives_pick_side"] == expected


def test_a_zero_row_snapshot_reproduces_the_incumbent_card(
    env: tuple[Path, Path, Path],
) -> None:
    artifacts_root, data_root, features_path = env
    _write_inactives_snapshot(
        data_root,
        snapshot_id="20260920T153000Z",
        captured_at=pd.Timestamp("2026-09-20T15:30:00+00:00"),
        rows=[],
        empty_reason="primary_offseason_placeholder",
    )
    rows, diagnostics = _rows(artifacts_root, data_root, features_path)
    assert diagnostics["games_adjusted"] == []
    assert set(rows["inactives_pick_side"]) == {"AWAY"}
    assert (rows["inactives_pick_side"] == rows["tuesday_pick_side"]).all()
    assert not rows["inactives_flip_vs_tuesday"].any()
    playable = rows.loc[~rows["source"].eq(SOURCE_STRUCTURALLY_EXCLUDED)]
    assert set(playable["source"]) == {SOURCE_NO_SNAPSHOT}


def test_a_missing_player_snapshot_fails_open(env: tuple[Path, Path, Path]) -> None:
    artifacts_root, data_root, features_path = env
    _write_inactives_snapshot(
        data_root,
        snapshot_id="20260920T153000Z",
        captured_at=pd.Timestamp("2026-09-20T15:30:00+00:00"),
        rows=[_inactive_row("AAA", "Nobody Known", "WR")],
    )
    rows, diagnostics = _rows(artifacts_root, data_root, features_path)
    assert diagnostics["no_adjustment_reason"] == "no_player_snapshot"
    assert (rows["inactives_pick_side"] == rows["played_pick_side"]).all()


def test_recording_is_opt_in_and_writes_its_own_ledger(env: tuple[Path, Path, Path]) -> None:
    artifacts_root, data_root, features_path = env
    plan = _plan(artifacts_root, data_root, features_path)

    skipped = record_inactives_refresh_overlay(artifacts_root, data_root, plan)
    assert skipped["recorded"] == 0
    assert skipped["skipped"] is True
    assert skipped["challenger_id"] == CHALLENGER_ID
    assert not (artifacts_root / "prospective" / "inactives_refresh_decisions.parquet").is_file()

    recorded = record_inactives_refresh_overlay(
        artifacts_root,
        data_root,
        plan,
        record_decisions=True,
        min_train_games=MIN_TRAIN_GAMES,
    )
    assert recorded["recorded"] == len(GAMES)
    ledger = pd.read_parquet(artifacts_root / "prospective" / "inactives_refresh_decisions.parquet")
    assert list(ledger.columns) == list(INACTIVES_REFRESH_OVERLAY_COLUMNS)
    # The played pick ledger is never touched by this overlay.
    assert not (artifacts_root / "prospective" / "pick_revisions.parquet").is_file()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _repo_artifacts_root() -> Path:
    return Path(__file__).resolve().parents[1] / "artifacts"


def test_challenger_is_registered_active_prospective() -> None:
    entry = find_challenger(_repo_artifacts_root(), CHALLENGER_ID)
    assert entry["status"] == ACTIVE_CHALLENGER_STATUS


def test_challenger_fingerprint_is_stable() -> None:
    """The registered fingerprint must be the digest of its own model block.

    The same guard every sibling overlay challenger carries: a registration
    whose declared fingerprint does not match its declared configuration would
    silently mis-pin the arm to a model it was never registered against.
    """

    entry = find_challenger(_repo_artifacts_root(), CHALLENGER_ID)
    assert config_fingerprint(entry["model"]) == entry["config_fingerprint"]
    # Stable across repeated digests of an equivalent (10 vs 10.0) config.
    equivalent = dict(entry["model"])
    equivalent["ridge_alpha"] = float(equivalent["ridge_alpha"])
    equivalent["min_train_games"] = int(equivalent["min_train_games"])
    assert config_fingerprint(equivalent) == entry["config_fingerprint"]


def test_registration_does_not_claim_the_publish_recording_path() -> None:
    """`cli.py` was off-limits, so this arm must NOT register under
    `publish-predictions --record-decisions` -- that would break
    tests/test_cli.py's PUBLISH_CHALLENGER_RESULT_KEYS coverage assertion."""

    entry = find_challenger(_repo_artifacts_root(), CHALLENGER_ID)
    command = str(entry["weekly_recording_command"])
    # The exact string tests/test_cli.py keys off (read, tests/test_cli.py:667).
    assert "nfl-ats publish-predictions --record-decisions" not in command
    assert "refresh-picks" in command


def test_registry_json_is_still_well_formed() -> None:
    payload = json.loads(
        (_repo_artifacts_root() / "prospective" / "challengers.json").read_text(encoding="utf-8")
    )
    ids = [entry["challenger_id"] for entry in payload["challengers"]]
    assert len(ids) == len(set(ids))
    assert CHALLENGER_ID in ids
