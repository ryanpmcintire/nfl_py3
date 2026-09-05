"""Tests for ENG-08 timing-policy instrumentation (``nfl_ats.refresh_triggers``).

Binding closing-grounds taxonomy (AGENTS.md), restated verbatim per this
project's rule for any test that scores or adjudicates an experiment: an
interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. Only two grounds ever close a line of work: (1) refuted
mechanism -- a RESOLVED wrong sign (whole interval on the wrong side of
zero) or zero split-half reliability; (2) bounded by a positive control
proven able to detect an effect that size. Everything else is
``unresolved_below_power``. This module is pure instrumentation: no test
here records to a registry, and every ledger row below is synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.io import atomic_json, atomic_parquet
from nfl_ats.refresh_triggers import (
    CLOCK_CHECKPOINT_NAMES,
    TRIGGER_CLOCK_CHECKPOINT,
    TRIGGER_CLOCK_DISPATCH,
    TRIGGER_INACTIVES_POSTED,
    TRIGGER_INJURY_REPORT_POSTED,
    TRIGGER_LINE_MOVE,
    TRIGGER_LINEUP_CHANGE,
    TRIGGER_MANUAL,
    TRIGGER_NEWS_EVENT,
    TRIGGER_UNKNOWN,
    RefreshTrigger,
    append_triggers_to_evidence_log,
    archive_lineup_snapshot,
    compare_trigger_vs_checkpoint,
    detect_clock_checkpoint_triggers,
    detect_inactives_triggers,
    detect_injury_report_triggers,
    detect_line_move_triggers,
    detect_lineup_change_triggers,
    evidence_log_path,
    mkt08_trigger_type,
    schedule_game_windows,
)

SEASON, WEEK = 2026, 2

# A Tue..Mon week: Thursday 2026-09-17 through Monday 2026-09-21 ET
# (September is EDT, UTC-4). Mirrors tests/test_inactives_refresh_overlay.py's
# own anchors.
THU_GAME_ID = "2026_02_AAA_THU"
SUN_EARLY_GAME_ID = "2026_02_BBB_SUNEARLY"
SNF_GAME_ID = "2026_02_CCC_SNF"
MNF_GAME_ID = "2026_02_DDD_MNF"

THU_KICKOFF = pd.Timestamp("2026-09-18T00:15:00Z")  # Thu 8:15pm ET
SUN_EARLY_KICKOFF = pd.Timestamp("2026-09-20T17:00:00Z")  # Sun 1:00pm ET
SNF_KICKOFF = pd.Timestamp("2026-09-21T00:20:00Z")  # Sun 8:20pm ET
MNF_KICKOFF = pd.Timestamp("2026-09-22T00:15:00Z")  # Mon 8:15pm ET
SUNDAY_LOCK = pd.Timestamp("2026-09-20T20:00:00Z")  # Sun 4:00pm ET


def _write_schedule(repo_root: Path) -> None:
    frame = pd.DataFrame(
        {
            "season": [SEASON] * 4,
            "week": [WEEK] * 4,
            "game_type": ["REG"] * 4,
            "game_id": [THU_GAME_ID, SUN_EARLY_GAME_ID, SNF_GAME_ID, MNF_GAME_ID],
            "home_team": ["THU_H", "SUN_H", "SNF_H", "MNF_H"],
            "away_team": ["THU_A", "SUN_A", "SNF_A", "MNF_A"],
            "gameday": ["2026-09-17", "2026-09-20", "2026-09-20", "2026-09-21"],
            "gametime": ["20:15", "13:00", "20:20", "20:15"],
        }
    )
    out = repo_root / "data" / "raw" / "20260901T000000Z" / "schedules.parquet"
    atomic_parquet(frame, out)


# ---------------------------------------------------------------------------
# Deadlines: Sunday 1pm, SNF, MNF (early Sunday-4pm lock), Thursday
# ---------------------------------------------------------------------------


def test_schedule_game_windows_deadline_arithmetic(tmp_path: Path) -> None:
    _write_schedule(tmp_path)
    windows = {w.game_id: w for w in schedule_game_windows(tmp_path, season=SEASON, week=WEEK)}
    assert set(windows) == {THU_GAME_ID, SUN_EARLY_GAME_ID, SNF_GAME_ID, MNF_GAME_ID}

    # Thursday: deadline is its own kickoff (well before the week's Sunday lock).
    assert windows[THU_GAME_ID].deadline == THU_KICKOFF

    # Sunday 1pm: deadline is its own kickoff (earlier than the 4pm lock).
    assert windows[SUN_EARLY_GAME_ID].deadline == SUN_EARLY_KICKOFF

    # SNF: locks EARLY at Sunday 4pm ET, not at its own 8:20pm kickoff.
    assert windows[SNF_GAME_ID].deadline == SUNDAY_LOCK
    assert windows[SNF_GAME_ID].deadline < SNF_KICKOFF

    # MNF: also locks EARLY at the same Sunday 4pm ET instant, a full day
    # before its own Monday kickoff.
    assert windows[MNF_GAME_ID].deadline == SUNDAY_LOCK
    assert windows[MNF_GAME_ID].deadline < MNF_KICKOFF


def test_deadline_validation_sunday_1pm_game(tmp_path: Path) -> None:
    _write_schedule(tmp_path)
    window = next(
        w
        for w in schedule_game_windows(tmp_path, season=SEASON, week=WEEK)
        if w.game_id == SUN_EARLY_GAME_ID
    )
    before = window.deadline - pd.Timedelta(hours=1)
    trigger_ok = RefreshTrigger(
        trigger_source=TRIGGER_INACTIVES_POSTED,
        game_id=window.game_id,
        season=SEASON,
        week=WEEK,
        observation_time=pd.Timestamp.now(tz="UTC"),
        source_capture_time=before,
        checkpoint_name=None,
        deadline=window.deadline,
        deadline_valid=before < window.deadline,
        deadline_reason="",
    )
    assert trigger_ok.deadline_valid is True

    after = window.deadline + pd.Timedelta(minutes=30)
    assert not (after < window.deadline)


def test_deadline_validation_snf_locks_at_sunday_4pm(tmp_path: Path) -> None:
    """SNF: a capture AFTER the 4pm ET lock but well BEFORE its own 8:20pm
    kickoff must still be a deadline violation -- the early-lock rule, not a
    kickoff-relative one."""

    _write_schedule(tmp_path)
    window = next(
        w
        for w in schedule_game_windows(tmp_path, season=SEASON, week=WEEK)
        if w.game_id == SNF_GAME_ID
    )
    after_lock_before_kickoff = SUNDAY_LOCK + pd.Timedelta(hours=1)
    assert after_lock_before_kickoff < SNF_KICKOFF
    assert not (after_lock_before_kickoff < window.deadline)

    before_lock = SUNDAY_LOCK - pd.Timedelta(hours=1)
    assert before_lock < window.deadline


def test_deadline_validation_mnf_locks_at_sunday_4pm(tmp_path: Path) -> None:
    """MNF: a Monday-morning capture, hours before its own Monday-night
    kickoff, is still a deadline violation -- the lock is Sunday afternoon."""

    _write_schedule(tmp_path)
    window = next(
        w
        for w in schedule_game_windows(tmp_path, season=SEASON, week=WEEK)
        if w.game_id == MNF_GAME_ID
    )
    monday_morning = pd.Timestamp("2026-09-21T14:00:00Z")  # Mon 10am ET
    assert monday_morning < MNF_KICKOFF
    assert not (monday_morning < window.deadline)

    sunday_afternoon = pd.Timestamp("2026-09-20T19:00:00Z")  # Sun 3pm ET
    assert sunday_afternoon < window.deadline


def test_deadline_validation_thursday_game(tmp_path: Path) -> None:
    _write_schedule(tmp_path)
    window = next(
        w
        for w in schedule_game_windows(tmp_path, season=SEASON, week=WEEK)
        if w.game_id == THU_GAME_ID
    )
    wednesday = pd.Timestamp("2026-09-16T18:00:00Z")
    assert wednesday < window.deadline
    at_kickoff = THU_KICKOFF
    assert not (at_kickoff < window.deadline)  # strict inequality: at-kickoff is invalid


# ---------------------------------------------------------------------------
# Detector 1: clock checkpoints
# ---------------------------------------------------------------------------


def test_detect_clock_checkpoint_triggers(tmp_path: Path) -> None:
    _write_schedule(tmp_path)
    games = schedule_game_windows(tmp_path, season=SEASON, week=WEEK)
    assert CLOCK_CHECKPOINT_NAMES  # non-empty, sanity
    state = {
        "runs": {
            "refresh_sun@2026-09-20": {
                "status": "OK",
                "window_start": "2026-09-20T10:00:00-04:00",
                "ran_at": "2026-09-20T10:03:11-04:00",
            },
            # An unrelated job name must never leak into clock_checkpoint triggers.
            "odds_sun_close@2026-09-20": {
                "status": "OK",
                "ran_at": "2026-09-20T12:31:00-04:00",
            },
            # A MISSED occurrence never becomes a trigger.
            "refresh_sat@2026-09-19": {
                "status": "MISSED",
                "window_start": "2026-09-19T10:30:00-04:00",
            },
        }
    }
    triggers = detect_clock_checkpoint_triggers(state, games, season=SEASON, week=WEEK)
    assert len(triggers) == len(games)
    assert all(t.trigger_source == TRIGGER_CLOCK_CHECKPOINT for t in triggers)
    assert all(t.checkpoint_name == "refresh_sun" for t in triggers)
    # ran_at 2026-09-20T10:03:11-04:00 -> 14:03:11Z: after the STATE's own
    # window_start test above; the fixture's real checkpoint below uses a
    # 3pm ET run instead, so it lands strictly between the Sunday-1pm
    # kickoff and the SNF/MNF 4pm-ET early lock -- the illustrative case.
    by_game = {t.game_id: t for t in triggers}
    assert by_game[THU_GAME_ID].deadline_valid is False  # Thursday's own kickoff already passed
    assert by_game[SUN_EARLY_GAME_ID].deadline_valid is True  # before its own 1pm ET kickoff


def test_detect_clock_checkpoint_triggers_between_sunday_kickoff_and_lock(tmp_path: Path) -> None:
    """A checkpoint that runs at 3pm ET Sunday: the 1pm-ET game's own kickoff
    has already passed (invalid), but SNF/MNF's EARLY 4pm-ET lock has not yet
    arrived (valid) -- the exact contrast the early-lock rule predicts."""

    _write_schedule(tmp_path)
    games = schedule_game_windows(tmp_path, season=SEASON, week=WEEK)
    state = {
        "runs": {
            "refresh_sun@2026-09-20": {
                "status": "OK",
                "ran_at": "2026-09-20T15:00:00-04:00",  # 19:00Z
            }
        }
    }
    triggers = detect_clock_checkpoint_triggers(state, games, season=SEASON, week=WEEK)
    by_game = {t.game_id: t for t in triggers}
    assert by_game[THU_GAME_ID].deadline_valid is False
    assert by_game[SUN_EARLY_GAME_ID].deadline_valid is False  # own 1pm ET kickoff already passed
    assert by_game[SNF_GAME_ID].deadline_valid is True  # 4pm ET lock not yet reached
    assert by_game[MNF_GAME_ID].deadline_valid is True  # same lock, not yet reached


def test_detect_clock_checkpoint_triggers_before_any_deadline(tmp_path: Path) -> None:
    _write_schedule(tmp_path)
    games = schedule_game_windows(tmp_path, season=SEASON, week=WEEK)
    state = {
        "runs": {
            "refresh_thu@2026-09-17": {
                "status": "CAUGHT_UP",
                "ran_at": "2026-09-16T09:00:00-04:00",  # Wednesday: before every deadline
            }
        }
    }
    triggers = detect_clock_checkpoint_triggers(state, games, season=SEASON, week=WEEK)
    assert len(triggers) == len(games)
    assert all(t.deadline_valid for t in triggers)


# ---------------------------------------------------------------------------
# Detector 2: inactives-posted
# ---------------------------------------------------------------------------


def _write_inactives_snapshot(
    data_root: Path, *, snapshot_id: str, captured_at: pd.Timestamp, rows: list[dict[str, Any]]
) -> None:
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
            "season": SEASON,
            "week": WEEK,
            "source_used": "primary",
            "row_count": len(rows),
            "teams_seen": sorted({row["team"] for row in rows}),
            "empty_reason": None,
            "ok": True,
        },
        root / "manifest.json",
    )


def test_detect_inactives_triggers(tmp_path: Path) -> None:
    _write_schedule(tmp_path)
    data_root = tmp_path / "data"
    games = schedule_game_windows(tmp_path, season=SEASON, week=WEEK)
    captured = SUN_EARLY_KICKOFF - pd.Timedelta(hours=2)
    _write_inactives_snapshot(
        data_root,
        snapshot_id="20260920T110000Z",
        captured_at=captured,
        rows=[
            {
                "captured_at_utc": captured.isoformat(),
                "season": SEASON,
                "week": WEEK,
                "game_id": SUN_EARLY_GAME_ID,
                "home_team": "SUN_H",
                "away_team": "SUN_A",
                "team": "SUN_H",
                "player_name": "Real Starter",
                "position": "WR",
                "status": "Inactive",
                "source_url": "https://www.nfl.com/inactives/",
            }
        ],
    )
    triggers = detect_inactives_triggers(data_root, games, season=SEASON, week=WEEK)
    assert len(triggers) == 1
    trigger = triggers[0]
    assert trigger.trigger_source == TRIGGER_INACTIVES_POSTED
    assert trigger.game_id == SUN_EARLY_GAME_ID
    assert trigger.source_capture_time == captured.tz_convert("UTC")
    assert trigger.deadline_valid is True
    assert trigger.checkpoint_name is None


# ---------------------------------------------------------------------------
# Detector 3: injury-report-posted (nflverse + Sportradar)
# ---------------------------------------------------------------------------


def test_detect_injury_report_triggers(tmp_path: Path) -> None:
    _write_schedule(tmp_path)
    data_root = tmp_path / "data"
    games = schedule_game_windows(tmp_path, season=SEASON, week=WEEK)

    nflverse_dir = data_root / "players" / "raw" / "20260916T120000Z"
    nflverse_dir.mkdir(parents=True)
    atomic_json(
        {"created_at_utc": "2026-09-16T12:00:00+00:00", "snapshot_id": "20260916T120000Z"},
        nflverse_dir / "manifest.json",
    )

    sportradar_dir = data_root / "raw" / "sportradar_injuries" / "20260917T170000Z"
    sportradar_dir.mkdir(parents=True)
    atomic_json(
        {
            "status": "complete",
            "schema": "sportradar_nfl_injuries_snapshot/1",
            "captured_at_utc": "2026-09-17T17:00:00+00:00",
            "season": SEASON,
            "week": WEEK,
        },
        sportradar_dir / "manifest.json",
    )
    # A different week's Sportradar snapshot must never leak in.
    other_week_dir = data_root / "raw" / "sportradar_injuries" / "20260910T170000Z"
    other_week_dir.mkdir(parents=True)
    atomic_json(
        {
            "status": "complete",
            "schema": "sportradar_nfl_injuries_snapshot/1",
            "captured_at_utc": "2026-09-10T17:00:00+00:00",
            "season": SEASON,
            "week": WEEK - 1,
        },
        other_week_dir / "manifest.json",
    )

    triggers = detect_injury_report_triggers(data_root, games, season=SEASON, week=WEEK)
    assert all(t.trigger_source == TRIGGER_INJURY_REPORT_POSTED for t in triggers)
    detail_snapshots = {t.detail for t in triggers}
    assert any("nflverse" in detail for detail in detail_snapshots)
    assert any("sportradar" in detail for detail in detail_snapshots)
    assert any("20260910T170000Z" in detail for detail in detail_snapshots) is False
    # One row per game per snapshot: 4 games x 2 in-scope snapshots.
    assert len(triggers) == len(games) * 2


# ---------------------------------------------------------------------------
# Detector 4: lineup change between consecutive archived captures
# ---------------------------------------------------------------------------


def _lineup_payload(
    generated_at: str, sun_early_home_players: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "season": SEASON,
        "week": WEEK,
        "generated_at": generated_at,
        "games": {
            SUN_EARLY_GAME_ID: {
                "home": {"players": sun_early_home_players},
                "away": {"players": [{"slot": "QB1", "name": "Away QB", "gsis_id": "away-qb"}]},
            },
            THU_GAME_ID: {
                "home": {"players": [{"slot": "QB1", "name": "Thu QB", "gsis_id": "thu-qb"}]},
                "away": {
                    "players": [{"slot": "QB1", "name": "Thu Away QB", "gsis_id": "thu-away-qb"}]
                },
            },
        },
    }


def test_detect_lineup_change_triggers(tmp_path: Path) -> None:
    _write_schedule(tmp_path)
    games = schedule_game_windows(tmp_path, season=SEASON, week=WEEK)
    archive_dir = tmp_path / "lineup_archive"
    archive_dir.mkdir()

    first = _lineup_payload(
        "20260916T120000Z", [{"slot": "QB1", "name": "Starter QB", "gsis_id": "qb-1"}]
    )
    second = _lineup_payload(
        "20260917T120000Z", [{"slot": "QB1", "name": "Backup QB", "gsis_id": "qb-2"}]
    )
    (archive_dir / "a.json").write_text(json.dumps(first), encoding="utf-8")
    (archive_dir / "b.json").write_text(json.dumps(second), encoding="utf-8")

    triggers = detect_lineup_change_triggers(archive_dir, games, season=SEASON, week=WEEK)
    assert len(triggers) == 1
    trigger = triggers[0]
    assert trigger.trigger_source == TRIGGER_LINEUP_CHANGE
    assert trigger.game_id == SUN_EARLY_GAME_ID  # only this game's home side changed
    assert "home" in trigger.detail
    assert trigger.source_capture_time == pd.Timestamp("2026-09-17T12:00:00Z")


def test_detect_lineup_change_triggers_no_change_no_trigger(tmp_path: Path) -> None:
    _write_schedule(tmp_path)
    games = schedule_game_windows(tmp_path, season=SEASON, week=WEEK)
    archive_dir = tmp_path / "lineup_archive"
    archive_dir.mkdir()
    payload = _lineup_payload(
        "20260916T120000Z", [{"slot": "QB1", "name": "Starter QB", "gsis_id": "qb-1"}]
    )
    (archive_dir / "a.json").write_text(json.dumps(payload), encoding="utf-8")
    identical = dict(payload)
    identical["generated_at"] = "20260917T120000Z"
    (archive_dir / "b.json").write_text(json.dumps(identical), encoding="utf-8")

    triggers = detect_lineup_change_triggers(archive_dir, games, season=SEASON, week=WEEK)
    assert triggers == ()


def test_archive_lineup_snapshot_is_idempotent_by_generated_at(tmp_path: Path) -> None:
    source = tmp_path / "lineups.json"
    source.write_text(
        json.dumps(
            {"season": SEASON, "week": WEEK, "generated_at": "20260916T120000Z", "games": {}}
        ),
        encoding="utf-8",
    )
    archive_dir = tmp_path / "archive"
    first = archive_lineup_snapshot(source, archive_dir)
    assert first is not None
    assert first.is_file()
    second = archive_lineup_snapshot(source, archive_dir)
    assert second is None  # same generated_at already archived
    assert len(list(archive_dir.glob("*.json"))) == 1


# ---------------------------------------------------------------------------
# Detector 5: line move beyond MOVEMENT_POLICY_THRESHOLD
# ---------------------------------------------------------------------------


def test_detect_line_move_triggers(tmp_path: Path) -> None:
    from nfl_ats.clv import PAPER_DECISION_COLUMNS
    from nfl_ats.market_data import QUOTE_COLUMNS, write_market_snapshot

    _write_schedule(tmp_path)
    games = schedule_game_windows(tmp_path, season=SEASON, week=WEEK)
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"

    decisions = pd.DataFrame({column: [None] for column in PAPER_DECISION_COLUMNS})
    decisions = decisions.loc[decisions.index.repeat(1)].reset_index(drop=True)
    decisions["game_id"] = [SUN_EARLY_GAME_ID]
    decisions["season"] = [SEASON]
    decisions["week"] = [WEEK]
    decisions["decision_home_spread"] = [-3.0]
    decisions["pick_side"] = ["HOME"]
    decisions["is_best_pick"] = [False]
    for flag in (
        "coach_fade_flip",
        "division_revenge_flip",
        "player_arrests_flip",
        "spread_gap_zone_flip",
        "composed_overlay_flip",
    ):
        decisions[flag] = [False]
    ledger_path = artifacts_root / "clv_ledger" / "decisions.parquet"
    atomic_parquet(decisions, ledger_path)

    now = SUN_EARLY_KICKOFF - pd.Timedelta(hours=5)
    quote_row = dict.fromkeys(QUOTE_COLUMNS)
    quote_row.update(
        {
            "observed_at_utc": now.isoformat(),
            "provider": "the_odds_api",
            "provider_event_id": "evt-1",
            "commence_time_utc": SUN_EARLY_KICKOFF.isoformat(),
            "nflverse_game_id": SUN_EARLY_GAME_ID,
            "bookmaker_key": "book_a",
            "market": "spreads",
            "outcome_side": "HOME",
            "home_spread_line": -5.0,  # moved 2.0 pts from the -3.0 opener
        }
    )
    quotes = pd.DataFrame([quote_row], columns=list(QUOTE_COLUMNS))
    write_market_snapshot(
        b"{}",
        quotes,
        data_root / "market" / "raw",
        observed_at=now.to_pydatetime(),
        request_metadata={},
    )

    triggers = detect_line_move_triggers(
        artifacts_root, data_root, games, season=SEASON, week=WEEK, now=now
    )
    assert len(triggers) == 1
    trigger = triggers[0]
    assert trigger.trigger_source == TRIGGER_LINE_MOVE
    assert trigger.game_id == SUN_EARLY_GAME_ID
    assert trigger.deadline_valid is True
    assert "-3.0" in trigger.detail or "-3" in trigger.detail


# ---------------------------------------------------------------------------
# Idempotent JSONL append
# ---------------------------------------------------------------------------


def _trigger(game_id: str, source_capture_time: pd.Timestamp) -> RefreshTrigger:
    return RefreshTrigger(
        trigger_source=TRIGGER_INACTIVES_POSTED,
        game_id=game_id,
        season=SEASON,
        week=WEEK,
        observation_time=pd.Timestamp.now(tz="UTC"),
        source_capture_time=source_capture_time,
        checkpoint_name=None,
        deadline=source_capture_time + pd.Timedelta(hours=1),
        deadline_valid=True,
        deadline_reason="ok",
        detail="test",
    )


def test_append_triggers_to_evidence_log_is_idempotent(tmp_path: Path) -> None:
    path = evidence_log_path(tmp_path, season=SEASON, week=WEEK)
    triggers = (
        _trigger("g1", pd.Timestamp("2026-09-20T11:00:00Z")),
        _trigger("g2", pd.Timestamp("2026-09-20T11:00:00Z")),
    )
    written, skipped = append_triggers_to_evidence_log(path, triggers)
    assert (written, skipped) == (2, 0)
    lines_after_first = path.read_text(encoding="utf-8").splitlines()
    assert len(lines_after_first) == 2

    # Re-running the exact same scan must not duplicate anything.
    written2, skipped2 = append_triggers_to_evidence_log(path, triggers)
    assert (written2, skipped2) == (0, 2)
    lines_after_second = path.read_text(encoding="utf-8").splitlines()
    assert lines_after_second == lines_after_first

    # A genuinely new trigger (different source_capture_time) does append.
    written3, _ = append_triggers_to_evidence_log(
        path, (_trigger("g1", pd.Timestamp("2026-09-20T12:00:00Z")),)
    )
    assert written3 == 1
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


# ---------------------------------------------------------------------------
# mkt08_trigger_type mapping
# ---------------------------------------------------------------------------


def test_mkt08_trigger_type_mapping() -> None:
    assert mkt08_trigger_type(TRIGGER_CLOCK_CHECKPOINT) == TRIGGER_CLOCK_DISPATCH
    assert mkt08_trigger_type(TRIGGER_MANUAL) == TRIGGER_UNKNOWN
    assert mkt08_trigger_type(TRIGGER_INACTIVES_POSTED) == TRIGGER_NEWS_EVENT
    assert mkt08_trigger_type(TRIGGER_INJURY_REPORT_POSTED) == TRIGGER_NEWS_EVENT
    assert mkt08_trigger_type(TRIGGER_LINEUP_CHANGE) == TRIGGER_NEWS_EVENT


# ---------------------------------------------------------------------------
# The prospective comparison scaffold, synthetic rows only
# ---------------------------------------------------------------------------


def _valid_trigger(game_id: str, week: int, *, deadline_valid: bool = True) -> RefreshTrigger:
    return RefreshTrigger(
        trigger_source=TRIGGER_INJURY_REPORT_POSTED,
        game_id=game_id,
        season=SEASON,
        week=week,
        observation_time=pd.Timestamp.now(tz="UTC"),
        source_capture_time=pd.Timestamp("2026-09-17T12:00:00Z"),
        checkpoint_name=None,
        deadline=pd.Timestamp("2026-09-20T17:00:00Z"),
        deadline_valid=deadline_valid,
        deadline_reason="ok" if deadline_valid else "deadline_violation: too late",
    )


def test_compare_trigger_vs_checkpoint_interval_containing_zero_is_unresolved() -> None:
    """A synthetic population built to sum to exactly zero -- 11 weeks (at
    the estimator's own MIN_BLOCKS_FOR_INTERVAL floor, so the interval is a
    genuine, non-degenerate one): the checkpoint always picks HOME; the
    trigger agrees on 5 games (improvement 0 regardless of outcome) and
    disagrees on 6, split 3-3 between the trigger being right (home does NOT
    cover, improvement +1) and wrong (home covers, improvement -1). The
    point estimate is exactly 0 and the bootstrap interval straddles it --
    per AGENTS.md that is NEVER grounds to reject or close this line of
    work: classification must stay unresolved_below_power."""

    # (checkpoint_home, trigger_home, home_covers) per game.
    specs = [
        (True, True, True),  # agree -> 0
        (True, False, True),  # disagree, home covers -> checkpoint right, trigger wrong: -1
        (True, True, True),  # agree -> 0
        (True, False, False),  # disagree, away covers -> trigger right, checkpoint wrong: +1
        (True, True, False),  # agree -> 0
        (True, False, True),  # disagree -> -1
        (True, True, False),  # agree -> 0
        (True, False, False),  # disagree -> +1
        (True, True, True),  # agree -> 0
        (True, False, True),  # disagree -> -1
        (True, False, False),  # disagree -> +1
    ]
    rows = []
    for i, (checkpoint_home, trigger_home, home_covers) in enumerate(specs):
        rows.append(
            {
                "game_id": f"g{i}",
                "season": SEASON,
                "week": i + 1,
                "checkpoint_pick_home": checkpoint_home,
                "trigger_pick_home": trigger_home,
                "settle_margin": 3.0 if home_covers else -3.0,
            }
        )
    ledger_rows = pd.DataFrame(rows)
    triggers = tuple(_valid_trigger(row["game_id"], row["week"]) for row in rows)

    result = compare_trigger_vs_checkpoint(ledger_rows, triggers, samples=5000, seed=1)
    assert result.n_games == len(rows)
    assert result.n_weeks == len(rows)
    assert result.estimate == 0.0
    assert result.lower <= 0.0 <= result.upper  # the interval genuinely contains zero
    assert result.classification == "unresolved_below_power"
    assert result.closing_ground is None
    assert 0.0 <= result.probability_positive <= 1.0
    # The taxonomy: report probability_positive, never a binary "contains zero".
    assert "probability_positive" in result.detail


def test_compare_trigger_vs_checkpoint_excludes_deadline_violations() -> None:
    rows = [
        {
            "game_id": "g_valid",
            "season": SEASON,
            "week": 1,
            "checkpoint_pick_home": True,
            "trigger_pick_home": True,
            "settle_margin": 3.0,
        },
        {
            "game_id": "g_violation",
            "season": SEASON,
            "week": 1,
            "checkpoint_pick_home": True,
            "trigger_pick_home": False,
            "settle_margin": -3.0,
        },
    ]
    ledger_rows = pd.DataFrame(rows)
    triggers = (
        _valid_trigger("g_valid", 1, deadline_valid=True),
        _valid_trigger("g_violation", 1, deadline_valid=False),
    )
    result = compare_trigger_vs_checkpoint(ledger_rows, triggers, samples=200, seed=1)
    assert result.excluded_deadline_violations == ("g_violation",)
    assert result.n_games == 1


def test_compare_trigger_vs_checkpoint_pushes_are_excluded() -> None:
    rows = [
        {
            "game_id": "g_push",
            "season": SEASON,
            "week": 1,
            "checkpoint_pick_home": True,
            "trigger_pick_home": False,
            "settle_margin": 0.0,
        }
    ]
    ledger_rows = pd.DataFrame(rows)
    triggers = (_valid_trigger("g_push", 1),)
    result = compare_trigger_vs_checkpoint(ledger_rows, triggers, samples=200, seed=1)
    assert result.n_games == 0
    assert result.classification == "unresolved_below_power"
