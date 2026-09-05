"""Specialist (long-snapper/punter) absence fade refresh-path challenger
(LEAD-17, docs/schedule_flag_battery.md "Wave 7").

Mirrors ``tests/test_crew_tilt_refresh_overlay.py``'s structure (the refresh-
path precedent this module follows): a pure ``RefreshResult`` -> rows
computation, FAIL-OPEN on every missing-data path, opt-in recording, and a
registration self-consistency check against the TRACKED registry.

Load-bearing here:

1. :func:`live_specialist_out_qualifying` -- the deliberate, disclosed
   deviation from ``roster_availability_flag_features.weekly_specialist_out_qualifying``:
   no season cap.
2. :func:`build_specialist_absence_fade_refresh_rows` -- fades the team
   missing its LS/P (backs the opponent), leaves a both-out or neither-out
   game at the incumbent pick, and is FAIL-OPEN when no injury snapshot
   exists or none resolves for the exact (season, week).
3. :func:`record_specialist_absence_fade_refresh_overlay` -- opt-in
   recording, anti-backdating via the week's ORIGINAL card kickoffs, never
   touches ``pick_revisions.parquet`` or the published card.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import nfl_ats.specialist_absence_fade_refresh_overlay as specialist_module
from nfl_ats.data import DataContractError
from nfl_ats.pick_refresh import (
    MOVEMENT_POLICY_MODEL_ONLY,
    RefreshedGame,
    RefreshResult,
    pick_deadline,
    sunday_pick_lock,
)
from nfl_ats.prospective_scoring import config_fingerprint
from nfl_ats.specialist_absence_fade_refresh_overlay import (
    CHALLENGER_ID,
    OVERLAY_STATUS_APPLIED,
    OVERLAY_STATUS_BOTH_OUT,
    OVERLAY_STATUS_NEITHER_OUT,
    OVERLAY_STATUS_NO_REPORT_FOR_WEEK,
    OVERLAY_STATUS_NO_SNAPSHOT,
    SPECIALIST_ABSENCE_REFRESH_COLUMNS,
    build_specialist_absence_fade_refresh_rows,
    latest_nflverse_injuries_snapshot,
    live_specialist_out_qualifying,
    load_specialist_absence_fade_refresh_decisions,
    record_specialist_absence_fade_refresh_overlay,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

SEASON = 2026
WEEK = 2
KICKOFF = pd.Timestamp("2026-09-20T17:00:00+00:00")
SATURDAY_PASS = pd.Timestamp("2026-09-19T15:00:00+00:00")
TUESDAY_RECORD = pd.Timestamp("2026-09-15T16:00:00+00:00")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _game(
    *,
    game_id: str,
    home_team: str = "HME",
    away_team: str = "AWY",
    new_pick_side: str = "HOME",
    probability: float = 0.52,
    eligible: bool = True,
) -> RefreshedGame:
    lock = sunday_pick_lock(pd.Series([KICKOFF]))
    return RefreshedGame(
        game_id=game_id,
        home_team=home_team,
        away_team=away_team,
        kickoff=KICKOFF,
        deadline=pick_deadline(KICKOFF, lock),
        decision_home_spread=-2.5,
        original_recorded_at_utc=TUESDAY_RECORD,
        previous_pick_side=new_pick_side,
        previous_home_cover_probability=None,
        new_pick_side=new_pick_side,
        new_home_cover_probability=probability,
        decision_policy_id="policy-1",
        decision_policy_fingerprint="fingerprint-1",
        coach_fade_flip=False,
        division_revenge_flip=False,
        player_arrests_flip=False,
        spread_gap_zone_flip=False,
        composed_overlay_flip=False,
        player_arrests_snapshot_id="snapshot-tuesday",
        player_arrests_safe_index_sha256="safe-index-sha",
        movement_policy=MOVEMENT_POLICY_MODEL_ONLY,
        movement_delta=None,
        movement_pick_side="",
        model_only_pick_side="HOME" if probability >= 0.5 else "AWAY",
        eligible=eligible,
        ineligible_reason="" if eligible else "kickoff_passed",
        changed=False,
    )


def _plan(games: tuple[RefreshedGame, ...]) -> RefreshResult:
    return RefreshResult(
        season=SEASON,
        week=WEEK,
        refresh_run_id="20260919T150000Z",
        computed_at_utc=SATURDAY_PASS,
        model_id="model-1",
        feature_table_path="unused",
        feature_table_sha256="feature-sha",
        games=games,
        unrefreshable_game_ids=(),
        missing_from_features_game_ids=(),
    )


def _write_injuries_snapshot(
    data_root: Path,
    *,
    snapshot_id: str,
    rows: list[dict[str, object]],
) -> Path:
    snapshot = data_root / "raw" / "nflverse_injuries" / snapshot_id
    snapshot.mkdir(parents=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(snapshot / "injuries.parquet", index=False)
    return snapshot / "injuries.parquet"


def _out_row(*, season: int, week: int, team: str, position: str = "LS") -> dict[str, object]:
    return {
        "season": season,
        "week": week,
        "team": team,
        "position": position,
        "report_status": "Out",
        "game_type": "REG",
        "full_name": f"{team} {position} Player",
    }


# ---------------------------------------------------------------------------
# 1. live_specialist_out_qualifying -- the disclosed, un-capped deviation
# ---------------------------------------------------------------------------


def test_refresh_ignores_a_future_injury_capture(tmp_path: Path) -> None:
    _write_injuries_snapshot(
        tmp_path,
        snapshot_id="20260920T000000Z",
        rows=[_out_row(season=SEASON, week=WEEK, team="HME")],
    )
    rows, diagnostics = build_specialist_absence_fade_refresh_rows(
        _plan((_game(game_id="future"),)), data_root=tmp_path
    )
    assert rows.empty
    assert diagnostics["skipped"] is True


def test_refresh_ignores_post_recording_report_updates(tmp_path: Path) -> None:
    report = _out_row(season=SEASON, week=WEEK, team="HME")
    report["date_modified"] = "2026-09-20T00:00:00Z"
    _write_injuries_snapshot(tmp_path, snapshot_id="20260901T000000Z", rows=[report])
    rows, diagnostics = build_specialist_absence_fade_refresh_rows(
        _plan((_game(game_id="future-update"),)), data_root=tmp_path
    )
    assert rows.empty
    assert diagnostics["skipped"] is True


def test_current_report_without_specialist_absences_still_pairs_every_game(tmp_path: Path) -> None:
    _write_injuries_snapshot(
        tmp_path,
        snapshot_id="20260901T000000Z",
        rows=[_out_row(season=SEASON, week=WEEK, team="HME", position="WR")],
    )
    rows, diagnostics = build_specialist_absence_fade_refresh_rows(
        _plan((_game(game_id="no-specialist-out"),)), data_root=tmp_path
    )
    assert not diagnostics["skipped"]
    assert len(rows) == 1
    assert rows.iloc[0]["played_pick_side"] == rows.iloc[0]["specialist_would_be_pick_side"]
    assert not rows.iloc[0]["specialist_fade_flip"]


def test_live_qualifying_has_no_season_cap_unlike_the_historical_builder() -> None:
    injuries = pd.DataFrame(
        [_out_row(season=2026, week=1, team="AWY"), _out_row(season=2025, week=1, team="HME")]
    )
    qualifying = live_specialist_out_qualifying(injuries)
    assert set(qualifying["season"]) == {2025, 2026}


def test_live_qualifying_restricts_to_lsp_out_reg() -> None:
    injuries = pd.DataFrame(
        [
            _out_row(season=2026, week=1, team="A"),
            {
                "season": 2026,
                "week": 1,
                "team": "B",
                "position": "QB",
                "report_status": "Out",
                "game_type": "REG",
                "full_name": "B QB Player",
            },
            {
                "season": 2026,
                "week": 1,
                "team": "C",
                "position": "P",
                "report_status": "Questionable",
                "game_type": "REG",
                "full_name": "C P Player",
            },
            {
                "season": 2026,
                "week": 1,
                "team": "D",
                "position": "LS",
                "report_status": "Out",
                "game_type": "POST",
                "full_name": "D LS Player",
            },
        ]
    )
    qualifying = live_specialist_out_qualifying(injuries)
    assert set(qualifying["team"]) == {"A"}


def test_live_qualifying_requires_its_columns() -> None:
    with pytest.raises(DataContractError, match="injuries is missing"):
        live_specialist_out_qualifying(pd.DataFrame({"season": [2026]}))


def test_latest_nflverse_injuries_snapshot_picks_the_newest(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_injuries_snapshot(
        data_root, snapshot_id="20260101T000000Z", rows=[_out_row(season=2025, week=1, team="A")]
    )
    newest = _write_injuries_snapshot(
        data_root, snapshot_id="20260901T000000Z", rows=[_out_row(season=2026, week=1, team="A")]
    )
    assert latest_nflverse_injuries_snapshot(data_root) == newest


def test_latest_nflverse_injuries_snapshot_is_none_when_absent(tmp_path: Path) -> None:
    assert latest_nflverse_injuries_snapshot(tmp_path / "data") is None


# ---------------------------------------------------------------------------
# 2. build_specialist_absence_fade_refresh_rows
# ---------------------------------------------------------------------------


def test_fades_the_away_team_missing_its_specialist(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    games = (_game(game_id="g1", new_pick_side="HOME", probability=0.55),)
    plan = _plan(games)
    _write_injuries_snapshot(
        data_root,
        snapshot_id="20260901T000000Z",
        rows=[_out_row(season=SEASON, week=WEEK, team="AWY")],
    )

    frame, diagnostics = build_specialist_absence_fade_refresh_rows(plan, data_root=data_root)

    assert not diagnostics["skipped"]
    row = frame.set_index("game_id").loc["g1"]
    assert bool(row["away_specialist_out"]) is True
    assert bool(row["home_specialist_out"]) is False
    assert row["specialist_would_be_pick_side"] == "HOME"
    assert bool(row["specialist_fade_flip"]) is False  # already HOME
    assert row["overlay_status"] == OVERLAY_STATUS_APPLIED
    assert row["played_pick_side"] == "HOME"


def test_flips_the_pick_when_the_played_side_is_the_missing_teams_side(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    games = (_game(game_id="g1", new_pick_side="AWAY", probability=0.35),)
    plan = _plan(games)
    _write_injuries_snapshot(
        data_root,
        snapshot_id="20260901T000000Z",
        rows=[_out_row(season=SEASON, week=WEEK, team="AWY")],
    )

    frame, _diag = build_specialist_absence_fade_refresh_rows(plan, data_root=data_root)
    row = frame.set_index("game_id").loc["g1"]
    assert row["specialist_would_be_pick_side"] == "HOME"
    assert bool(row["specialist_fade_flip"]) is True


def test_both_teams_missing_a_specialist_keeps_the_incumbent_pick(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    games = (_game(game_id="g1", new_pick_side="HOME", probability=0.55),)
    plan = _plan(games)
    _write_injuries_snapshot(
        data_root,
        snapshot_id="20260901T000000Z",
        rows=[
            _out_row(season=SEASON, week=WEEK, team="AWY"),
            _out_row(season=SEASON, week=WEEK, team="HME", position="P"),
        ],
    )

    frame, _diag = build_specialist_absence_fade_refresh_rows(plan, data_root=data_root)
    row = frame.set_index("game_id").loc["g1"]
    assert row["overlay_status"] == OVERLAY_STATUS_BOTH_OUT
    assert bool(row["specialist_fade_flip"]) is False
    assert row["specialist_would_be_pick_side"] == "HOME"


def test_neither_team_missing_a_specialist_keeps_the_incumbent_pick(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    games = (_game(game_id="g1", new_pick_side="HOME", probability=0.55),)
    plan = _plan(games)
    _write_injuries_snapshot(
        data_root,
        snapshot_id="20260901T000000Z",
        rows=[_out_row(season=SEASON, week=WEEK, team="SOMEOTHER")],
    )

    frame, _diag = build_specialist_absence_fade_refresh_rows(plan, data_root=data_root)
    row = frame.set_index("game_id").loc["g1"]
    assert row["overlay_status"] == OVERLAY_STATUS_NEITHER_OUT
    assert bool(row["specialist_fade_flip"]) is False


def test_no_snapshot_is_a_documented_skip_never_an_exception(tmp_path: Path) -> None:
    games = (_game(game_id="g1"),)
    plan = _plan(games)
    frame, diagnostics = build_specialist_absence_fade_refresh_rows(
        plan, data_root=tmp_path / "data"
    )
    assert frame.empty
    assert list(frame.columns) == list(SPECIALIST_ABSENCE_REFRESH_COLUMNS)
    assert diagnostics["skipped"] is True
    assert diagnostics["reason"] == OVERLAY_STATUS_NO_SNAPSHOT


def test_no_report_for_the_exact_week_is_a_documented_skip(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    games = (_game(game_id="g1"),)
    plan = _plan(games)
    # A real report exists, but for a DIFFERENT week -- must not leak across.
    _write_injuries_snapshot(
        data_root,
        snapshot_id="20260901T000000Z",
        rows=[_out_row(season=SEASON, week=WEEK + 1, team="AWY")],
    )

    frame, diagnostics = build_specialist_absence_fade_refresh_rows(plan, data_root=data_root)
    assert frame.empty
    assert diagnostics["skipped"] is True
    assert diagnostics["reason"] == OVERLAY_STATUS_NO_REPORT_FOR_WEEK


def test_no_eligible_games_is_a_documented_skip() -> None:
    games = (_game(game_id="g1", eligible=False),)
    plan = _plan(games)
    frame, diagnostics = build_specialist_absence_fade_refresh_rows(plan, data_root=Path("unused"))
    assert frame.empty
    assert diagnostics["skipped"] is True


def test_a_malformed_snapshot_is_fail_open_not_an_exception(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    games = (_game(game_id="g1"),)
    plan = _plan(games)
    # Missing required columns -> DataContractError inside, caught, folded
    # into a documented skip.
    snapshot = data_root / "raw" / "nflverse_injuries" / "20260901T000000Z"
    snapshot.mkdir(parents=True)
    pd.DataFrame({"season": [2026]}).to_parquet(snapshot / "injuries.parquet", index=False)

    frame, diagnostics = build_specialist_absence_fade_refresh_rows(plan, data_root=data_root)
    assert frame.empty
    assert diagnostics["skipped"] is True
    assert diagnostics["reason"] == OVERLAY_STATUS_NO_SNAPSHOT


# ---------------------------------------------------------------------------
# 3. record_specialist_absence_fade_refresh_overlay
# ---------------------------------------------------------------------------


def test_recording_is_opt_in(tmp_path: Path) -> None:
    result = record_specialist_absence_fade_refresh_overlay(
        tmp_path / "artifacts",
        tmp_path / "data",
        _plan((_game(game_id="g"),)),
        record_decisions=False,
    )
    assert result == {
        "challenger_id": CHALLENGER_ID,
        "recorded": 0,
        "skipped": True,
        "reason": (
            "pass --record-decisions to append this pass's would-be picks to the "
            "specialist-absence-fade refresh ledger"
        ),
    }
    assert not (tmp_path / "artifacts").exists()


def test_recording_outside_the_lock_window_writes_no_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts_root = tmp_path / "artifacts"
    plan = replace(
        _plan((_game(game_id="g"),)), computed_at_utc=pd.Timestamp("2026-09-01T15:00:00Z")
    )
    monkeypatch.setattr(
        specialist_module,
        "original_card",
        lambda *_args, **_kwargs: pd.DataFrame({"kickoff": [KICKOFF]}),
    )

    with pytest.raises(ValueError, match="RECORDING_LOCK_WINDOW"):
        record_specialist_absence_fade_refresh_overlay(
            artifacts_root, tmp_path / "data", plan, record_decisions=True
        )

    assert not (
        artifacts_root / "prospective" / "specialist_absence_fade_refresh_decisions.parquet"
    ).exists()


def test_recording_writes_a_row_recording_both_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    games = (_game(game_id="g1", new_pick_side="AWAY", probability=0.35),)
    plan = _plan(games)
    _write_injuries_snapshot(
        data_root,
        snapshot_id="20260901T000000Z",
        rows=[_out_row(season=SEASON, week=WEEK, team="AWY")],
    )

    monkeypatch.setattr(
        specialist_module,
        "original_card",
        lambda *_args, **_kwargs: pd.DataFrame({"kickoff": [KICKOFF]}),
    )

    result = record_specialist_absence_fade_refresh_overlay(
        artifacts_root, data_root, plan, record_decisions=True
    )
    assert result["recorded"] == 1
    ledger = load_specialist_absence_fade_refresh_decisions(artifacts_root)
    assert len(ledger) == 1
    row = ledger.iloc[0]
    # Both arms in one row.
    assert row["played_pick_side"] == "AWAY"
    assert row["specialist_would_be_pick_side"] == "HOME"
    assert bool(row["specialist_fade_flip"]) is True


# ---------------------------------------------------------------------------
# 4. Registration: fingerprint stability, refresh-path recording command
# ---------------------------------------------------------------------------


def _registered_entry() -> dict:
    payload = json.loads(
        (_REPO_ROOT / "artifacts" / "prospective" / "challengers.json").read_text(encoding="utf-8")
    )
    entries = [entry for entry in payload["challengers"] if entry["challenger_id"] == CHALLENGER_ID]
    assert len(entries) == 1, f"{CHALLENGER_ID} must be registered exactly once"
    return dict(entries[0])


def test_registered_challenger_fingerprint_is_stable() -> None:
    entry = _registered_entry()
    assert entry["status"] == "ACTIVE_PROSPECTIVE"
    assert entry["config_fingerprint"] == config_fingerprint(entry["model"])


def test_registration_names_the_refresh_recording_path() -> None:
    """The specialist-fade arm is late-refresh only, never a Tuesday publish
    recorder -- its source postdates the Tuesday lock."""

    entry = _registered_entry()
    assert "publish-predictions --record-decisions" not in entry["weekly_recording_command"]
    assert "refresh-picks --record-decisions" in entry["weekly_recording_command"]
