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

THU_GAME_ID = "2026_02_AAA_THU"
SUN_EARLY_GAME_ID = "2026_02_BBB_SUNEARLY"
SNF_GAME_ID = "2026_02_CCC_SNF"
MNF_GAME_ID = "2026_02_DDD_MNF"

THU_KICKOFF = pd.Timestamp("2026-09-18T00:15:00Z")
SUN_EARLY_KICKOFF = pd.Timestamp("2026-09-20T17:00:00Z")
SNF_KICKOFF = pd.Timestamp("2026-09-21T00:20:00Z")
MNF_KICKOFF = pd.Timestamp("2026-09-22T00:15:00Z")
SUNDAY_LOCK = pd.Timestamp("2026-09-20T20:00:00Z")


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


def test_schedule_game_windows_deadline_arithmetic(tmp_path: Path) -> None:
    _write_schedule(tmp_path)
    windows = {w.game_id: w for w in schedule_game_windows(tmp_path, season=SEASON, week=WEEK)}
    assert set(windows) == {THU_GAME_ID, SUN_EARLY_GAME_ID, SNF_GAME_ID, MNF_GAME_ID}

    assert windows[THU_GAME_ID].deadline == THU_KICKOFF

    assert windows[SUN_EARLY_GAME_ID].deadline == SUN_EARLY_KICKOFF

    assert windows[SNF_GAME_ID].deadline == SUNDAY_LOCK
    assert windows[SNF_GAME_ID].deadline < SNF_KICKOFF

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

    _write_schedule(tmp_path)
    window = next(
        w
        for w in schedule_game_windows(tmp_path, season=SEASON, week=WEEK)
        if w.game_id == MNF_GAME_ID
    )
    monday_morning = pd.Timestamp("2026-09-21T14:00:00Z")
    assert monday_morning < MNF_KICKOFF
    assert not (monday_morning < window.deadline)

    sunday_afternoon = pd.Timestamp("2026-09-20T19:00:00Z")
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
    assert not (at_kickoff < window.deadline)


def test_detect_clock_checkpoint_triggers(tmp_path: Path) -> None:
    _write_schedule(tmp_path)
    games = schedule_game_windows(tmp_path, season=SEASON, week=WEEK)
    assert CLOCK_CHECKPOINT_NAMES
    state = {
        "runs": {
            "refresh_sun@2026-09-20": {
                "status": "OK",
                "window_start": "2026-09-20T10:00:00-04:00",
                "ran_at": "2026-09-20T10:03:11-04:00",
            },
            "odds_sun_close@2026-09-20": {
                "status": "OK",
                "ran_at": "2026-09-20T12:31:00-04:00",
            },
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
    by_game = {t.game_id: t for t in triggers}
    assert by_game[THU_GAME_ID].deadline_valid is False
    assert by_game[SUN_EARLY_GAME_ID].deadline_valid is True


def test_detect_clock_checkpoint_triggers_between_sunday_kickoff_and_lock(tmp_path: Path) -> None:

    _write_schedule(tmp_path)
    games = schedule_game_windows(tmp_path, season=SEASON, week=WEEK)
    state = {
        "runs": {
            "refresh_sun@2026-09-20": {
                "status": "OK",
                "ran_at": "2026-09-20T15:00:00-04:00",
            }
        }
    }
    triggers = detect_clock_checkpoint_triggers(state, games, season=SEASON, week=WEEK)
    by_game = {t.game_id: t for t in triggers}
    assert by_game[THU_GAME_ID].deadline_valid is False
    assert by_game[SUN_EARLY_GAME_ID].deadline_valid is False
    assert by_game[SNF_GAME_ID].deadline_valid is True
    assert by_game[MNF_GAME_ID].deadline_valid is True


def test_detect_clock_checkpoint_triggers_before_any_deadline(tmp_path: Path) -> None:
    _write_schedule(tmp_path)
    games = schedule_game_windows(tmp_path, season=SEASON, week=WEEK)
    state = {
        "runs": {
            "refresh_thu@2026-09-17": {
                "status": "CAUGHT_UP",
                "ran_at": "2026-09-16T09:00:00-04:00",
            }
        }
    }
    triggers = detect_clock_checkpoint_triggers(state, games, season=SEASON, week=WEEK)
    assert len(triggers) == len(games)
    assert all(t.deadline_valid for t in triggers)


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
    assert len(triggers) == len(games) * 2


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
    assert trigger.game_id == SUN_EARLY_GAME_ID
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
    assert second is None
    assert len(list(archive_dir.glob("*.json"))) == 1


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
            "home_spread_line": -5.0,
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

    written2, skipped2 = append_triggers_to_evidence_log(path, triggers)
    assert (written2, skipped2) == (0, 2)
    lines_after_second = path.read_text(encoding="utf-8").splitlines()
    assert lines_after_second == lines_after_first

    written3, _ = append_triggers_to_evidence_log(
        path, (_trigger("g1", pd.Timestamp("2026-09-20T12:00:00Z")),)
    )
    assert written3 == 1
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def test_mkt08_trigger_type_mapping() -> None:
    assert mkt08_trigger_type(TRIGGER_CLOCK_CHECKPOINT) == TRIGGER_CLOCK_DISPATCH
    assert mkt08_trigger_type(TRIGGER_MANUAL) == TRIGGER_UNKNOWN
    assert mkt08_trigger_type(TRIGGER_INACTIVES_POSTED) == TRIGGER_NEWS_EVENT
    assert mkt08_trigger_type(TRIGGER_INJURY_REPORT_POSTED) == TRIGGER_NEWS_EVENT
    assert mkt08_trigger_type(TRIGGER_LINEUP_CHANGE) == TRIGGER_NEWS_EVENT


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

    specs = [
        (True, True, True),
        (True, False, True),
        (True, True, True),
        (True, False, False),
        (True, True, False),
        (True, False, True),
        (True, True, False),
        (True, False, False),
        (True, True, True),
        (True, False, True),
        (True, False, False),
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
    assert result.lower <= 0.0 <= result.upper
    assert result.classification == "unresolved_below_power"
    assert result.closing_ground is None
    assert 0.0 <= result.probability_positive <= 1.0
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
