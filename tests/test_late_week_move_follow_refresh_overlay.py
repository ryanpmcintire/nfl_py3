from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import nfl_ats.late_week_move_follow_refresh_overlay as movement
from nfl_ats.data import DataContractError
from nfl_ats.pick_refresh import (
    MOVEMENT_POLICY_MODEL_ONLY,
    RefreshedGame,
    RefreshResult,
    pick_deadline,
    sunday_pick_lock,
)

SEASON = 2025
WEEK = 2
KICKOFF = pd.Timestamp("2025-09-21T17:00:00+00:00")
SATURDAY_PASS = pd.Timestamp("2025-09-20T15:00:00+00:00")
TUESDAY_RECORD = pd.Timestamp("2025-09-16T16:00:00+00:00")


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
        refresh_run_id="20250919T150000Z",
        computed_at_utc=SATURDAY_PASS,
        model_id="model-1",
        feature_table_path="unused",
        feature_table_sha256="feature-sha",
        games=games,
        unrefreshable_game_ids=(),
        missing_from_features_game_ids=(),
    )


def _original():
    return pd.DataFrame(
        [
            {
                "game_id": "g",
                "kickoff": KICKOFF,
                "recorded_at_utc": TUESDAY_RECORD,
                "pick_side": "AWAY",
                "decision_home_spread": 3.0,
            }
        ]
    )


def _quotes(move=0.5):
    return pd.DataFrame(
        [
            {
                "nflverse_game_id": "g",
                "bookmaker_key": book,
                "market": "spreads",
                "observed_at_utc": time,
                "snapshot_timestamp_utc": time,
                "bookmaker_last_update_utc": time,
                "home_spread_line": line,
            }
            for book in ["bovada", "fanduel"]
            for time, line in [("2025-09-16T18:00Z", 3.0), ("2025-09-19T18:00Z", 3.0 + move)]
        ]
    )


def _build(quotes=None, plan=None, original=None):
    return movement.build_late_week_move_follow_refresh_rows(
        plan or _plan((_game(game_id="g", new_pick_side="HOME"),)),
        original=_original() if original is None else original,
        quotes=_quotes() if quotes is None else quotes,
    )


@pytest.mark.parametrize(
    "move,side,flips", [(0.5, "HOME", 1), (0.49, "AWAY", 0), (-0.5, "AWAY", 0), (0, "AWAY", 0)]
)
def test_frozen_threshold_and_tuesday_baseline(move, side, flips):
    rows, info = _build(_quotes(move))
    assert rows.iloc[0].tuesday_pick_side == "AWAY"
    assert rows.iloc[0].movement_would_be_pick_side == side
    assert rows.iloc[0].decision_home_spread == 3.0
    assert info["flips"] == flips


def test_equal_books_not_capture_frequency():
    q = _quotes(1.0)
    q.loc[
        (q.bookmaker_key == "fanduel") & q.observed_at_utc.eq("2025-09-19T18:00Z"),
        "home_spread_line",
    ] = 2.0
    q = pd.concat([q, q.iloc[[1]]], ignore_index=True)
    rows, _ = _build(q)
    assert rows.iloc[0].equal_net_move == 0
    assert rows.iloc[0].movement_would_be_pick_side == "AWAY"


@pytest.mark.parametrize(
    "column", ["observed_at_utc", "snapshot_timestamp_utc", "bookmaker_last_update_utc"]
)
def test_future_quote_and_snapshot_leakage(column):
    q = _quotes()
    q.loc[q.home_spread_line.eq(3.5), column] = "2025-09-21T18:00Z"
    rows, info = _build(q)
    assert rows.empty
    assert info["skipped"]
    assert info["refused_quote_rows"] == 2


def test_sunday_quotes_do_not_change_frozen_construct():
    q = _quotes()
    future = q.iloc[[1]].copy()
    for c in ["observed_at_utc", "snapshot_timestamp_utc", "bookmaker_last_update_utc"]:
        future[c] = "2025-09-21T10:00Z"
    future["home_spread_line"] = -100
    plan = replace(_plan((_game(game_id="g"),)), computed_at_utc=pd.Timestamp("2025-09-21T14:00Z"))
    rows, _ = _build(pd.concat([q, future]), plan)
    assert rows.iloc[0].equal_net_move == 0.5


@pytest.mark.parametrize("instant", ["2025-09-21T17:00Z", "2025-09-22T00:00Z"])
def test_deadline_cannot_be_bypassed_with_eligible_flag(instant):
    plan = replace(_plan((_game(game_id="g"),)), computed_at_utc=pd.Timestamp(instant))
    rows, info = _build(plan=plan)
    assert rows.empty and info["skipped"]


def test_future_plan_and_future_original_refused():
    with pytest.raises(DataContractError, match="future"):
        _build(
            plan=replace(
                _plan((_game(game_id="g"),)),
                computed_at_utc=pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1),
            )
        )
    original = _original()
    original["recorded_at_utc"] = "2025-09-21T12:00Z"
    with pytest.raises(DataContractError, match="future"):
        _build(original=original)


def test_absent_archive_and_opt_in(tmp_path):
    rows, info = _build(pd.DataFrame())
    assert rows.empty and info["skipped"]
    result = movement.record_late_week_move_follow_refresh_overlay(tmp_path, tmp_path, _plan(()))
    assert result["recorded"] == 0
    assert not list(tmp_path.rglob("*.parquet"))


def test_recorder_paired_and_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(movement, "original_card", lambda *a, **kw: _original())
    monkeypatch.setattr(movement, "load_decision_quotes", lambda *a, **kw: _quotes())
    monkeypatch.setattr(movement, "refuse_if_outside_recording_lock_window", lambda *a, **kw: None)
    plan = _plan((_game(game_id="g"),))
    assert (
        movement.record_late_week_move_follow_refresh_overlay(
            tmp_path, tmp_path, plan, record_decisions=True
        )["recorded"]
        == 1
    )
    assert (
        movement.record_late_week_move_follow_refresh_overlay(
            tmp_path, tmp_path, plan, record_decisions=True
        )["recorded"]
        == 0
    )
    rows = pd.read_parquet(tmp_path / "prospective" / movement.LEDGER_NAME)
    assert rows.iloc[0].tuesday_pick_side == "AWAY"
    assert rows.iloc[0].movement_would_be_pick_side == "HOME"
    assert not (tmp_path / "prospective/pick_revisions.parquet").exists()


def test_display_name():
    source = (
        Path(__file__).resolve().parents[1] / "src/nfl_ats/dashboard/findings_content.py"
    ).read_text(encoding="utf-8")
    assert '"late_week_move_follow_refresh_v1": "Follow late-week line moves"' in source


def test_sunday_pool_deadline_precedes_monday_kickoff():
    game = replace(
        _game(game_id="g"),
        kickoff=pd.Timestamp("2025-09-23T00:15Z"),
        deadline=pd.Timestamp("2025-09-23T00:15Z"),
    )
    original = _original()
    original["kickoff"] = game.kickoff
    plan = replace(_plan((game,)), computed_at_utc=pd.Timestamp("2025-09-21T20:00Z"))
    rows, info = _build(plan=plan, original=original)
    assert rows.empty and info["skipped"]


def test_unknown_books_cannot_change_the_frozen_universe():
    q = _quotes()
    unknown = q.iloc[[0, 1]].copy()
    unknown["bookmaker_key"] = "unlisted_book"
    unknown.loc[unknown.home_spread_line.eq(3.5), "home_spread_line"] = -100
    rows, _ = _build(pd.concat([q, unknown], ignore_index=True))
    assert rows.iloc[0].equal_net_move == 0.5
    assert rows.iloc[0].eligible_books == 2


def test_local_live_archive_loading(tmp_path, monkeypatch):
    import json

    q = _quotes()
    q["commence_time_utc"] = KICKOFF
    for index, (time, group) in enumerate(q.groupby("observed_at_utc")):
        directory = tmp_path / "market/raw" / str(index)
        directory.mkdir(parents=True)
        group.to_parquet(directory / "quotes.parquet", index=False)
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "capture_kind": "live",
                    "observed_at_utc": time,
                    "request": {"season": 2025, "week": 2},
                }
            ),
            encoding="utf-8",
        )
    monkeypatch.setattr(movement, "original_card", lambda *a, **kw: _original())
    monkeypatch.setattr(movement, "refuse_if_outside_recording_lock_window", lambda *a, **kw: None)
    result = movement.record_late_week_move_follow_refresh_overlay(
        tmp_path, tmp_path, _plan((_game(game_id="g"),)), record_decisions=True
    )
    assert result["recorded"] == 1
    assert result["flips"] == 1
