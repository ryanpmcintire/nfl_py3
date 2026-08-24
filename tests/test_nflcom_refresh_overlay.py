"""Refresh-path wiring of the NFL.com Friday out>=2-starters fade challenger.

Pins, per the task's contract and the repo's leakage invariant:
- the flag computation (2+ starter-caliber Outs on the picked team, opponent
  unflagged -> flip; both flagged -> keep; nothing flagged -> keep);
- played-pick INVARIANCE (the overlay cannot alter any RefreshedGame side or
  the pick-revision ledger -- it only ever writes its own separate ledger);
- FAIL-OPEN no-ops (no snapshot / no snap counts / page absent / page stale /
  page fetched at-or-after kickoff -- never an error, never a flip);
- challenger record emission (opt-in recording, append-only across passes).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.clv import PAPER_DECISION_COLUMNS, paper_decision_ledger_path
from nfl_ats.data import DataContractError
from nfl_ats.four_overlay_composition import POLICY_FINGERPRINT, POLICY_ID
from nfl_ats.io import atomic_parquet
from nfl_ats.nflcom_refresh_overlay import (
    NFLCOM_REFRESH_OVERLAY_COLUMNS,
    build_nflcom_refresh_overlay_rows,
    load_nflcom_refresh_overlay_decisions,
    nflcom_refresh_overlay_ledger_path,
    record_nflcom_refresh_overlay,
)
from nfl_ats.pick_refresh import (
    MOVEMENT_POLICY_MODEL_ONLY,
    MOVEMENT_POLICY_MOVEMENT,
    RefreshedGame,
    RefreshResult,
    load_pick_revisions,
)

SEASON = 2026
WEEK = 2
# Sunday 1:00 PM ET kickoff: late enough in its Tue..Mon week that a FINAL
# Friday injury page can legitimately precede it (a Thursday-night week
# cannot, and correctly gates to a documented skip).
KICKOFF = pd.Timestamp("2026-09-20T17:00:00+00:00")
SATURDAY_PASS = pd.Timestamp("2026-09-19T15:00:00+00:00")
FINAL_PAGE_FETCHED = "2026-09-18T21:00:00+00:00"  # Friday ~5:00 PM ET
TUESDAY_RECORD = pd.Timestamp("2026-09-15T16:00:00+00:00")


def _game(
    *,
    game_id: str,
    home_team: str,
    away_team: str,
    new_pick_side: str,
    eligible: bool = True,
) -> RefreshedGame:
    return RefreshedGame(
        game_id=game_id,
        home_team=home_team,
        away_team=away_team,
        kickoff=KICKOFF,
        deadline=KICKOFF,
        decision_home_spread=-2.5,
        original_recorded_at_utc=TUESDAY_RECORD,
        previous_pick_side=new_pick_side,
        previous_home_cover_probability=None,
        new_pick_side=new_pick_side,
        new_home_cover_probability=0.55 if new_pick_side == "HOME" else 0.45,
        decision_policy_id=POLICY_ID,
        decision_policy_fingerprint=POLICY_FINGERPRINT,
        coach_fade_flip=False,
        division_revenge_flip=False,
        player_arrests_flip=False,
        spread_gap_zone_flip=False,
        composed_overlay_flip=False,
        player_arrests_snapshot_id="snapshot-tuesday",
        player_arrests_safe_index_sha256="safe-index-sha",
        movement_policy=(MOVEMENT_POLICY_MODEL_ONLY if eligible else MOVEMENT_POLICY_MOVEMENT),
        movement_delta=None,
        movement_pick_side="" if eligible else "HOME",
        model_only_pick_side=new_pick_side,
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


def _write_nflcom_snapshot(data_root: Path, *, fetched_at_utc: str) -> Path:
    snapshot = data_root / "raw" / "nflcom_injuries" / "20260918T210000Z"
    snapshot.mkdir(parents=True)
    pd.DataFrame(
        [
            # g_flip's picked team TST: exactly two starter-caliber Outs.
            {
                "player": "Alpha One",
                "game_status": "Out",
                "season": SEASON,
                "week": WEEK,
                "team": "TST",
            },
            {
                "player": "Beta Two",
                "game_status": "Out",
                "season": SEASON,
                "week": WEEK,
                "team": "TST",
            },
            # g_tie: BOTH sides carry two starter-caliber Outs -> frozen tie rule keeps.
            {
                "player": "Gamma Three",
                "game_status": "Out",
                "season": SEASON,
                "week": WEEK,
                "team": "AAA",
            },
            {
                "player": "Delta Four",
                "game_status": "Out",
                "season": SEASON,
                "week": WEEK,
                "team": "AAA",
            },
            {
                "player": "Eps Five",
                "game_status": "Out",
                "season": SEASON,
                "week": WEEK,
                "team": "BBB",
            },
            {
                "player": "Zeta Six",
                "game_status": "Out",
                "season": SEASON,
                "week": WEEK,
                "team": "BBB",
            },
        ]
    ).to_parquet(snapshot / "injuries.parquet", index=False)
    (snapshot / "manifest.json").write_text(
        json.dumps({"pages": [{"season": SEASON, "week": WEEK, "fetched_at_utc": fetched_at_utc}]}),
        encoding="utf-8",
    )
    players_raw = data_root / "players" / "raw" / "20260915T000000Z"
    players_raw.mkdir(parents=True)
    rows = []
    for team, names in {
        "TST": ["Alpha One", "Beta Two"],
        "AAA": ["Gamma Three", "Delta Four"],
        "BBB": ["Eps Five", "Zeta Six"],
    }.items():
        for name in names:
            rows.append(
                {
                    "season": SEASON,
                    "game_type": "REG",
                    "week": WEEK - 1,
                    "team": team,
                    "player": name,
                    "offense_pct": 0.8,
                    "defense_pct": 0.0,
                }
            )
    pd.DataFrame(rows).to_parquet(players_raw / "snap_counts.parquet", index=False)
    return snapshot


def _write_original_card(artifacts_root: Path, kickoff: pd.Timestamp = KICKOFF) -> None:
    frame = pd.DataFrame(
        {
            "recorded_at_utc": [TUESDAY_RECORD],
            "forecast_artifact": ["margin_predictions/test"],
            "forecast_created_at_utc": [TUESDAY_RECORD],
            "model_id": ["model-1"],
            "method": ["market_residual"],
            "decision_policy_id": [POLICY_ID],
            "decision_policy_fingerprint": [POLICY_FINGERPRINT],
            "game_id": ["g_flip"],
            "season": [SEASON],
            "week": [WEEK],
            "kickoff": [kickoff],
            "away_team": ["MOV"],
            "home_team": ["TST"],
            "model_pick_side": ["HOME"],
            "pre_arrest_pick_side": ["HOME"],
            "former_policy_pick_side": ["HOME"],
            "pick_side": ["HOME"],
            "coach_fade_flip": [False],
            "division_revenge_flip": [False],
            "player_arrests_flip": [False],
            "spread_gap_zone_flip": [False],
            "composed_overlay_flip": [False],
            "player_arrests_home_flag": [False],
            "player_arrests_away_flag": [False],
            "player_arrests_snapshot_id": [""],
            "player_arrests_snapshot_fetched_at_utc": [pd.NaT],
            "player_arrests_safe_index_sha256": [""],
            "schedule_snapshot_id": ["schedule-tuesday"],
            "schedule_parquet_sha256": ["schedule-hash"],
            "bet_side": ["PASS"],
            "decision_home_spread": [-2.5],
            "edge": [float("nan")],
            "is_best_pick": [False],
        }
    )
    frame["recorded_at_utc"] = pd.to_datetime(frame["recorded_at_utc"], utc=True)
    frame["kickoff"] = pd.to_datetime(frame["kickoff"], utc=True)
    frame["forecast_created_at_utc"] = pd.to_datetime(frame["forecast_created_at_utc"], utc=True)
    atomic_parquet(frame[list(PAPER_DECISION_COLUMNS)], paper_decision_ledger_path(artifacts_root))


def _default_games() -> tuple[RefreshedGame, ...]:
    return (
        _game(game_id="g_flip", home_team="TST", away_team="MOV", new_pick_side="HOME"),
        # Played AWAY so the picked (flagged) team is BBB and the tie rule,
        # not the flip rule, is what keeps the pick unchanged.
        _game(game_id="g_tie", home_team="AAA", away_team="BBB", new_pick_side="AWAY"),
        _game(game_id="g_keep", home_team="CCC", away_team="DDD", new_pick_side="HOME"),
        _game(
            game_id="g_ineligible",
            home_team="EEE",
            away_team="FFF",
            new_pick_side="HOME",
            eligible=False,
        ),
    )


def _rows_for(data_root: Path) -> pd.DataFrame:
    rows, diagnostics = build_nflcom_refresh_overlay_rows(
        _plan(_default_games()), data_root=data_root
    )
    assert diagnostics["skipped"] is False
    return rows


# ---------------------------------------------------------------------------
# 1. Flag computation
# ---------------------------------------------------------------------------


def test_two_starter_outs_on_picked_team_and_none_on_opponent_flips(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_nflcom_snapshot(data_root, fetched_at_utc=FINAL_PAGE_FETCHED)

    rows = _rows_for(data_root)
    row = rows.set_index("game_id").loc["g_flip"]

    assert row["played_pick_side"] == "HOME"
    assert row["nflcom_would_be_pick_side"] == "AWAY"
    assert bool(row["nflcom_flip"]) is True
    assert row["picked_team"] == "TST"
    assert row["opponent_team"] == "MOV"
    assert row["picked_starter_out"] == 2
    assert row["opponent_starter_out"] == 0
    assert bool(row["picked_flag_ge_threshold"]) is True
    assert bool(row["opponent_flag_ge_threshold"]) is False
    assert row["injury_page_snapshot"] == "20260918T210000Z"
    assert pd.Timestamp(row["injury_page_fetched_at_utc"]) == pd.Timestamp(FINAL_PAGE_FETCHED)


def test_both_teams_flagged_keeps_the_played_pick(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_nflcom_snapshot(data_root, fetched_at_utc=FINAL_PAGE_FETCHED)

    rows = _rows_for(data_root)
    row = rows.set_index("game_id").loc["g_tie"]

    assert bool(row["picked_flag_ge_threshold"]) is True
    assert bool(row["opponent_flag_ge_threshold"]) is True
    assert row["picked_starter_out"] == 2
    assert row["opponent_starter_out"] == 2
    assert row["nflcom_would_be_pick_side"] == row["played_pick_side"] == "AWAY"
    assert bool(row["nflcom_flip"]) is False


def test_no_flagged_team_keeps_the_played_pick_and_week1_proxy_is_unavailable(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    _write_nflcom_snapshot(data_root, fetched_at_utc=FINAL_PAGE_FETCHED)

    rows = _rows_for(data_root)
    row = rows.set_index("game_id").loc["g_keep"]
    assert bool(row["picked_flag_ge_threshold"]) is False
    assert bool(row["opponent_flag_ge_threshold"]) is False
    assert row["nflcom_would_be_pick_side"] == "HOME"
    assert bool(row["nflcom_flip"]) is False

    # Week 1 has no prior-week snaps, so the starter proxy cannot flag anyone:
    # every count must come out 0 and every pick must keep.
    plan = RefreshResult(
        season=SEASON,
        week=1,
        refresh_run_id="20260912T150000Z",
        computed_at_utc=pd.Timestamp("2026-09-12T15:00:00+00:00"),
        model_id="model-1",
        feature_table_path="unused",
        feature_table_sha256="feature-sha",
        games=(_game(game_id="g_flip", home_team="TST", away_team="MOV", new_pick_side="HOME"),),
        unrefreshable_game_ids=(),
        missing_from_features_game_ids=(),
    )
    week1_rows, diagnostics = build_nflcom_refresh_overlay_rows(plan, data_root=data_root)
    assert diagnostics["skipped"] is True  # manifest has no (2026, week 1) page
    assert week1_rows.empty


def test_ineligible_games_are_excluded_from_the_challenger_record(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_nflcom_snapshot(data_root, fetched_at_utc=FINAL_PAGE_FETCHED)

    rows = _rows_for(data_root)
    assert set(rows["game_id"].astype(str)) == {"g_flip", "g_tie", "g_keep"}


# ---------------------------------------------------------------------------
# 2. Documented NO-OPs (fail-open; never an error, never a flip)
# ---------------------------------------------------------------------------


def test_noop_when_no_injuries_snapshot_exists(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _write_original_card(artifacts_root)

    result = record_nflcom_refresh_overlay(
        artifacts_root, tmp_path / "data", _plan(_default_games()), record_decisions=True
    )

    assert result["recorded"] == 0
    assert result["skipped"] is True
    assert result["reason"] == "no_nflcom_injuries_snapshot"
    assert not nflcom_refresh_overlay_ledger_path(artifacts_root).is_file()


def test_noop_when_snap_counts_are_absent(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_original_card(artifacts_root)
    snapshot = data_root / "raw" / "nflcom_injuries" / "20260918T210000Z"
    snapshot.mkdir(parents=True)
    (snapshot / "manifest.json").write_text(json.dumps({"pages": []}), encoding="utf-8")

    result = record_nflcom_refresh_overlay(
        artifacts_root, data_root, _plan(_default_games()), record_decisions=True
    )

    assert result["recorded"] == 0
    assert result["skipped"] is True
    assert result["reason"] == "no_snap_counts_snapshot"


def test_leakage_regression_page_fetched_at_or_after_kickoff_is_a_documented_noop(
    tmp_path: Path,
) -> None:
    """The flag may ONLY consume the pre-kickoff Friday-final page: a page
    stamped at (or after) the week's earliest kickoff carries post-kickoff
    information and must produce a skipped no-op with zero ledger writes --
    never a fallback read, never a flip."""

    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_original_card(artifacts_root)
    _write_nflcom_snapshot(data_root, fetched_at_utc=KICKOFF.isoformat())

    result = record_nflcom_refresh_overlay(
        artifacts_root, data_root, _plan(_default_games()), record_decisions=True
    )

    assert result["recorded"] == 0
    assert result["skipped"] is True
    assert "at or after the week's earliest kickoff" in result["reason"]
    assert not nflcom_refresh_overlay_ledger_path(artifacts_root).is_file()


def test_pre_friday_page_is_a_documented_noop(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_original_card(artifacts_root)
    _write_nflcom_snapshot(data_root, fetched_at_utc="2026-09-15T12:00:00+00:00")

    result = record_nflcom_refresh_overlay(
        artifacts_root, data_root, _plan(_default_games()), record_decisions=True
    )

    assert result["recorded"] == 0
    assert result["skipped"] is True
    assert "before Friday 16:00 ET" in result["reason"]
    assert not nflcom_refresh_overlay_ledger_path(artifacts_root).is_file()


def test_page_absent_from_manifest_is_a_documented_noop(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_original_card(artifacts_root)
    _write_nflcom_snapshot(data_root, fetched_at_utc=FINAL_PAGE_FETCHED)

    plan = RefreshResult(
        season=SEASON,
        week=3,  # not in the manifest
        refresh_run_id="20260926T150000Z",
        computed_at_utc=pd.Timestamp("2026-09-26T15:00:00+00:00"),
        model_id="model-1",
        feature_table_path="unused",
        feature_table_sha256="feature-sha",
        games=_default_games(),
        unrefreshable_game_ids=(),
        missing_from_features_game_ids=(),
    )
    rows, diagnostics = build_nflcom_refresh_overlay_rows(plan, data_root=data_root)
    assert rows.empty
    assert diagnostics["skipped"] is True
    assert "absent from snapshot manifest" in diagnostics["reason"]


# ---------------------------------------------------------------------------
# 3. Played-pick invariance (pinned)
# ---------------------------------------------------------------------------


def test_overlay_can_never_alter_the_played_pick_or_the_revision_ledger(
    tmp_path: Path,
) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_original_card(artifacts_root)
    _write_nflcom_snapshot(data_root, fetched_at_utc=FINAL_PAGE_FETCHED)
    plan = _plan(_default_games())
    sides_before = {game.game_id: game.new_pick_side for game in plan.games}
    revisions_before = load_pick_revisions(artifacts_root)

    result = record_nflcom_refresh_overlay(artifacts_root, data_root, plan, record_decisions=True)

    # The plan object itself is untouched...
    assert {game.game_id: game.new_pick_side for game in plan.games} == sides_before
    assert all(game.changed is False for game in plan.games)
    # ...the production revision ledger is untouched...
    pd.testing.assert_frame_equal(load_pick_revisions(artifacts_root), revisions_before)
    assert not (artifacts_root / "prospective" / "pick_revisions.parquet").is_file()
    # ...and the overlay ledger records would-be picks that differ from the
    # played side ONLY as separate challenger rows.
    assert result["recorded"] == 3
    ledger = load_nflcom_refresh_overlay_decisions(artifacts_root)
    by_game = ledger.set_index("game_id")
    assert by_game.loc["g_flip", "played_pick_side"] == sides_before["g_flip"] == "HOME"
    assert by_game.loc["g_flip", "nflcom_would_be_pick_side"] == "AWAY"
    assert by_game.loc["g_tie", "nflcom_would_be_pick_side"] == sides_before["g_tie"]
    assert set(ledger.columns) == set(NFLCOM_REFRESH_OVERLAY_COLUMNS)


# ---------------------------------------------------------------------------
# 4. Challenger record emission (opt-in, append-only)
# ---------------------------------------------------------------------------


def test_record_skips_the_ledger_write_by_default(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_original_card(artifacts_root)

    result = record_nflcom_refresh_overlay(
        artifacts_root, data_root, _plan(_default_games()), record_decisions=False
    )

    assert result == {
        "challenger_id": "nflcom_friday_refresh_out2_starters_v1",
        "recorded": 0,
        "skipped": True,
        "reason": (
            "pass --record-decisions to append this pass's would-be picks to the "
            "NFL.com refresh-overlay ledger"
        ),
    }
    assert not nflcom_refresh_overlay_ledger_path(artifacts_root).is_file()


def test_record_appends_across_passes_append_only(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_original_card(artifacts_root)
    _write_nflcom_snapshot(data_root, fetched_at_utc=FINAL_PAGE_FETCHED)
    first = _plan(_default_games())
    second = RefreshResult(
        **{
            **first.__dict__,
            "refresh_run_id": "20260920T130000Z",
            "computed_at_utc": pd.Timestamp("2026-09-20T13:00:00+00:00"),
        }
    )

    first_result = record_nflcom_refresh_overlay(
        artifacts_root, data_root, first, record_decisions=True
    )
    second_result = record_nflcom_refresh_overlay(
        artifacts_root, data_root, second, record_decisions=True
    )

    assert first_result["recorded"] == 3
    assert first_result["would_flip_game_ids"] == ["g_flip"]
    assert second_result["ledger_rows"] == 6
    ledger = load_nflcom_refresh_overlay_decisions(artifacts_root)
    assert len(ledger) == 6
    assert set(ledger["refresh_run_id"].astype(str)) == {
        "20260919T150000Z",
        "20260920T130000Z",
    }


def test_record_refuses_far_ahead_of_kickoff(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    far_future_kickoff = pd.Timestamp("2026-12-06T18:00:00+00:00")
    _write_original_card(artifacts_root, kickoff=far_future_kickoff)
    _write_nflcom_snapshot(data_root, fetched_at_utc=FINAL_PAGE_FETCHED)

    with pytest.raises(ValueError, match="nflcom-refresh-overlay"):
        record_nflcom_refresh_overlay(
            artifacts_root, data_root, _plan(_default_games()), record_decisions=True
        )
    assert not nflcom_refresh_overlay_ledger_path(artifacts_root).is_file()


def test_load_returns_empty_frame_when_absent(tmp_path: Path) -> None:
    ledger = load_nflcom_refresh_overlay_decisions(tmp_path)
    assert ledger.empty
    assert list(ledger.columns) == list(NFLCOM_REFRESH_OVERLAY_COLUMNS)


def test_load_raises_on_missing_columns(tmp_path: Path) -> None:
    bad_path = nflcom_refresh_overlay_ledger_path(tmp_path)
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_parquet(pd.DataFrame({"game_id": ["g1"]}), bad_path)

    with pytest.raises(DataContractError):
        load_nflcom_refresh_overlay_decisions(tmp_path)
