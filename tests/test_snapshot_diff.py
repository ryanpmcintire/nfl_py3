"""ENG-18: tests for ``nfl_ats.snapshot_diff``.

**Binding closing-grounds taxonomy (AGENTS.md), restated verbatim per this
project's rule for any module that touches an experiment:** an interval or CI
that contains zero is NEVER grounds to reject, fail, or close an experiment.
At this evaluator's ~2-point resolution, "contains zero" is the EXPECTED
outcome for a real small signal. Only two grounds ever close a line of work:
(1) refuted mechanism -- a RESOLVED wrong sign (whole interval on the wrong
side of zero) or zero split-half reliability; (2) bounded by a positive
control proven able to detect an effect that size. Everything else is
``unresolved_below_power``: record it with ``nfl-ats weak-signals record``,
report ``probability_positive``, never the binary "contains zero." **None of
that applies to this test file** -- ``nfl_ats.snapshot_diff`` is a diff, not
a verdict; it never adjudicates a signal and these tests never touch
``registry/``.

Synthetic fixtures only, all under ``tmp_path``. Two games (a Sunday-early
kickoff and a Sunday-slightly-later kickoff, same week) with a paper-decision
ledger row each and a matching ``margin_predictions`` artifact, then one or
more pick-revision-ledger and/or later-forecast-artifact refresh passes
layered on top per test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from nfl_ats.clv import PAPER_DECISION_COLUMNS, paper_decision_ledger_path
from nfl_ats.io import atomic_parquet
from nfl_ats.pick_refresh import (
    PICK_REVISION_COLUMNS,
    TRIGGER_NEWS_EVENT,
    pick_revision_ledger_path,
)
from nfl_ats.refresh_triggers import (
    TRIGGER_CLOCK_CHECKPOINT,
    TRIGGER_CLOCK_DISPATCH,
    TRIGGER_INJURY_REPORT_POSTED,
    evidence_log_path,
)
from nfl_ats.snapshot_diff import (
    PASS_ORIGIN_FORECAST_ARTIFACT,
    PASS_ORIGIN_PICK_REVISION,
    STATE_CHANGED,
    STATE_NO_DATA,
    STATE_UNCHANGED,
    GameSnapshot,
    _diff_game,
    build_snapshot_diff,
    render_markdown,
    resolve_tuesday_lock,
    to_dict,
    to_json,
)

SEASON = 2024
WEEK = 1

GAME_A = "2024_01_AAA_BBB"  # away AAA at home BBB, Sunday early window
GAME_B = "2024_01_CCC_DDD"  # away CCC at home DDD, Sunday early window (later kickoff than A)

KICKOFF_A = pd.Timestamp("2024-09-08T17:00:00Z")
KICKOFF_B = pd.Timestamp("2024-09-08T18:00:00Z")

#: Well before both kickoffs and the week's Sunday 16:00 ET (~20:00 UTC in
#: September, EDT) pick lock.
FRIDAY = pd.Timestamp("2024-09-06T12:00:00Z")

#: After both kickoffs AND after the Sunday pick lock.
SUNDAY_LATE = pd.Timestamp("2024-09-08T20:30:00Z")

FORECAST_DIR_NAME = "2024-week-01-20240903T140000Z"


# ---------------------------------------------------------------------------
# fixture builders
# ---------------------------------------------------------------------------


def _write_artifact(directory: Path, *, created_at_utc: str, games: list[dict[str, Any]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "ats_method": "market_residual",
        "created_at_utc": created_at_utc,
        "season": SEASON,
        "week": WEEK,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    rows = [
        {
            "game_id": g["game_id"],
            "method": "market_residual",
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "kickoff": g["kickoff"],
            "spread_line": g["spread_line"],
            "home_cover_probability": g["home_cover_probability"],
            "bet_side": g["bet_side"],
        }
        for g in games
    ]
    pd.DataFrame(rows).to_csv(directory / "predictions.csv", index=False)


def _paper_row(
    *,
    game_id: str,
    home_team: str,
    away_team: str,
    kickoff: pd.Timestamp,
    decision_home_spread: float,
    pick_side: str,
    forecast_artifact: str,
) -> dict[str, Any]:
    return {
        "recorded_at_utc": pd.Timestamp("2024-09-03T14:00:00Z"),
        "forecast_artifact": forecast_artifact,
        "forecast_created_at_utc": pd.Timestamp("2024-09-03T14:00:00Z"),
        "model_id": "test_model",
        "method": "market_residual",
        "decision_policy_id": "overlay_union_coach_division_revenge_player_arrests_spread_gap_v1",
        "decision_policy_fingerprint": "fp",
        "game_id": game_id,
        "season": SEASON,
        "week": WEEK,
        "kickoff": kickoff,
        "away_team": away_team,
        "home_team": home_team,
        "model_pick_side": pick_side,
        "pre_arrest_pick_side": pick_side,
        "former_policy_pick_side": pick_side,
        "pick_side": pick_side,
        "coach_fade_flip": False,
        "division_revenge_flip": False,
        "player_arrests_flip": False,
        "spread_gap_zone_flip": False,
        "composed_overlay_flip": False,
        "player_arrests_home_flag": False,
        "player_arrests_away_flag": False,
        "player_arrests_snapshot_id": "",
        "player_arrests_snapshot_fetched_at_utc": pd.NaT,
        "player_arrests_safe_index_sha256": "",
        "schedule_snapshot_id": "",
        "schedule_parquet_sha256": "",
        "bet_side": pick_side,
        "decision_home_spread": decision_home_spread,
        "edge": 0.05,
        "is_best_pick": False,
    }


def _write_paper_ledger(artifacts_root: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)[list(PAPER_DECISION_COLUMNS)]
    atomic_parquet(frame, paper_decision_ledger_path(artifacts_root))


def _revision_row(
    *,
    refresh_run_id: str,
    game_id: str,
    home_team: str,
    away_team: str,
    kickoff: pd.Timestamp,
    decision_home_spread: float,
    previous_pick_side: str,
    new_pick_side: str,
    new_home_cover_probability: float,
    revision_recorded_at_utc: pd.Timestamp,
    trigger_type: str = "",
    trigger_source: str = "",
    trigger_observed_at_utc: pd.Timestamp = pd.NaT,
) -> dict[str, Any]:
    return {
        "revision_recorded_at_utc": revision_recorded_at_utc,
        "refresh_run_id": refresh_run_id,
        "season": SEASON,
        "week": WEEK,
        "game_id": game_id,
        "home_team": home_team,
        "away_team": away_team,
        "kickoff": kickoff,
        "decision_home_spread": decision_home_spread,
        "original_recorded_at_utc": pd.Timestamp("2024-09-03T14:00:00Z"),
        "previous_pick_side": previous_pick_side,
        "previous_home_cover_probability": float("nan"),
        "new_pick_side": new_pick_side,
        "new_home_cover_probability": new_home_cover_probability,
        "decision_policy_id": "overlay_union_coach_division_revenge_player_arrests_spread_gap_v1",
        "decision_policy_fingerprint": "fp",
        "coach_fade_flip": False,
        "division_revenge_flip": False,
        "player_arrests_flip": False,
        "spread_gap_zone_flip": False,
        "composed_overlay_flip": False,
        "player_arrests_snapshot_id": "",
        "player_arrests_safe_index_sha256": "",
        "movement_policy": "model_only",
        "movement_delta": float("nan"),
        "movement_pick_side": "",
        "model_only_pick_side": new_pick_side,
        "model_id": "test_model",
        "feature_table_sha256": "abc123",
        "reason": "test",
        "trigger_type": trigger_type,
        "trigger_source": trigger_source,
        "trigger_observed_at_utc": trigger_observed_at_utc,
    }


def _write_pick_revisions(artifacts_root: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)[list(PICK_REVISION_COLUMNS)]
    atomic_parquet(frame, pick_revision_ledger_path(artifacts_root))


def _setup_tuesday(artifacts_root: Path, *, forecast_dir_name: str = FORECAST_DIR_NAME) -> None:
    _write_artifact(
        artifacts_root / "margin_predictions" / forecast_dir_name,
        created_at_utc="2024-09-03T14:00:00+00:00",
        games=[
            {
                "game_id": GAME_A,
                "home_team": "BBB",
                "away_team": "AAA",
                "kickoff": "2024-09-08T17:00:00+00:00",
                "spread_line": -3.0,
                "home_cover_probability": 0.40,
                "bet_side": "AWAY",
            },
            {
                "game_id": GAME_B,
                "home_team": "DDD",
                "away_team": "CCC",
                "kickoff": "2024-09-08T18:00:00+00:00",
                "spread_line": 1.5,
                "home_cover_probability": 0.55,
                "bet_side": "HOME",
            },
        ],
    )
    _write_paper_ledger(
        artifacts_root,
        [
            _paper_row(
                game_id=GAME_A,
                home_team="BBB",
                away_team="AAA",
                kickoff=KICKOFF_A,
                decision_home_spread=-3.0,
                pick_side="AWAY",
                forecast_artifact=f"margin_predictions/{forecast_dir_name}",
            ),
            _paper_row(
                game_id=GAME_B,
                home_team="DDD",
                away_team="CCC",
                kickoff=KICKOFF_B,
                decision_home_spread=1.5,
                pick_side="HOME",
                forecast_artifact=f"margin_predictions/{forecast_dir_name}",
            ),
        ],
    )


def _write_evidence_log(artifacts_root: Path, rows: list[dict[str, Any]]) -> None:
    path = evidence_log_path(artifacts_root, season=SEASON, week=WEEK)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_resolves_via_paper_decision_ledger_and_reads_tuesday_state(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _setup_tuesday(artifacts_root)

    lock = resolve_tuesday_lock(artifacts_root, season=SEASON, week=WEEK)
    assert lock.resolved
    assert lock.basis == "paper_decision_ledger"
    assert lock.ledger_rows == 2
    assert len(lock.games) == 2
    by_game = {game.game_id: game for game in lock.games}
    assert by_game[GAME_A].pick_side == "AWAY"
    assert by_game[GAME_A].pick_basis == "paper_decision_ledger"
    assert by_game[GAME_A].market_line == -3.0
    # model_probability always comes from the forecast artifact, never the ledger
    assert by_game[GAME_A].model_probability == 0.40
    assert by_game[GAME_A].overlays_fired == ()


def test_resolve_falls_back_to_earliest_artifact_when_no_ledger(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _write_artifact(
        artifacts_root / "margin_predictions" / FORECAST_DIR_NAME,
        created_at_utc="2024-09-03T14:00:00+00:00",
        games=[
            {
                "game_id": GAME_A,
                "home_team": "BBB",
                "away_team": "AAA",
                "kickoff": "2024-09-08T17:00:00+00:00",
                "spread_line": -3.0,
                "home_cover_probability": 0.4,
                "bet_side": "AWAY",
            }
        ],
    )

    lock = resolve_tuesday_lock(artifacts_root, season=SEASON, week=WEEK)
    assert lock.resolved
    assert lock.basis == "forecast_artifact_earliest"
    assert lock.ledger_rows == 0
    assert len(lock.games) == 1
    # no ledger row -> overlay knowledge is genuinely unknown, not "none fired"
    assert lock.games[0].overlays_fired is None
    assert lock.games[0].pick_basis == "forecast_artifact_raw"


def test_unresolved_when_nothing_on_disk(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    diff = build_snapshot_diff(9999, 1, artifacts_root=artifacts_root)
    assert not diff.tuesday.resolved
    assert diff.refresh_passes == ()
    markdown = render_markdown(diff)
    assert "UNRESOLVED" in markdown


def test_flipped_pick_and_inferred_unchanged_pick(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _setup_tuesday(artifacts_root)
    _write_pick_revisions(
        artifacts_root,
        [
            _revision_row(
                refresh_run_id="run1",
                game_id=GAME_A,
                home_team="BBB",
                away_team="AAA",
                kickoff=KICKOFF_A,
                decision_home_spread=-3.0,
                previous_pick_side="AWAY",
                new_pick_side="HOME",
                new_home_cover_probability=0.62,
                revision_recorded_at_utc=FRIDAY,
                trigger_type=TRIGGER_NEWS_EVENT,
                trigger_source="injury_report_posted",
                trigger_observed_at_utc=FRIDAY - pd.Timedelta(hours=1),
            ),
            # GAME_B intentionally absent: still eligible at FRIDAY -> inferred unchanged.
        ],
    )

    diff = build_snapshot_diff(SEASON, WEEK, artifacts_root=artifacts_root)
    pick_revision_passes = [p for p in diff.refresh_passes if p.origin == PASS_ORIGIN_PICK_REVISION]
    assert len(pick_revision_passes) == 1
    refresh_pass = pick_revision_passes[0]
    assert refresh_pass.trigger_basis == "ledger_recorded"
    assert refresh_pass.trigger_source == "injury_report_posted"

    by_game = {row.game_id: row for row in refresh_pass.games}

    flipped = by_game[GAME_A]
    assert flipped.tuesday_pick_side == "AWAY"
    assert flipped.refresh_pick_side == "HOME"
    assert flipped.pick_state == "flipped_away_to_home"
    assert flipped.probability_delta is not None
    assert flipped.probability_state == STATE_CHANGED
    # the frozen-line invariant: market line never moves on a pick-revision pass
    assert flipped.market_line_state == STATE_UNCHANGED
    # overlays are frozen by design for this channel
    assert flipped.overlay_state == STATE_UNCHANGED

    unchanged = by_game[GAME_B]
    assert unchanged.tuesday_pick_side == "HOME"
    assert unchanged.refresh_pick_side == "HOME"
    assert unchanged.pick_state == "same"
    assert "inferred_unchanged" in unchanged.refresh_pick_basis
    # the recomputed probability for an unchanged, absent game is genuinely unrecorded
    assert unchanged.probability_state == STATE_NO_DATA


def test_absent_game_ineligible_after_its_own_deadline(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _setup_tuesday(artifacts_root)
    _write_pick_revisions(
        artifacts_root,
        [
            _revision_row(
                refresh_run_id="run_late",
                game_id=GAME_A,
                home_team="BBB",
                away_team="AAA",
                kickoff=KICKOFF_A,
                decision_home_spread=-3.0,
                previous_pick_side="AWAY",
                new_pick_side="HOME",
                new_home_cover_probability=0.70,
                revision_recorded_at_utc=SUNDAY_LATE,
                trigger_type=TRIGGER_NEWS_EVENT,
                trigger_source="line_move",
                trigger_observed_at_utc=SUNDAY_LATE,
            ),
            # GAME_B absent, and by SUNDAY_LATE its own deadline has passed.
        ],
    )

    diff = build_snapshot_diff(SEASON, WEEK, artifacts_root=artifacts_root)
    refresh_pass = next(p for p in diff.refresh_passes if p.refresh_run_id == "run_late")
    by_game = {row.game_id: row for row in refresh_pass.games}
    assert "ineligible_at_this_pass" in by_game[GAME_B].refresh_pick_basis
    assert by_game[GAME_B].pick_state == STATE_NO_DATA


def test_trigger_resolved_from_evidence_log_news_vs_clock_checkpoint(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _setup_tuesday(artifacts_root)

    run1_time = pd.Timestamp("2024-09-06T12:00:00Z")
    run2_time = pd.Timestamp("2024-09-07T09:00:00Z")

    _write_pick_revisions(
        artifacts_root,
        [
            _revision_row(
                refresh_run_id="run1",
                game_id=GAME_A,
                home_team="BBB",
                away_team="AAA",
                kickoff=KICKOFF_A,
                decision_home_spread=-3.0,
                previous_pick_side="AWAY",
                new_pick_side="HOME",
                new_home_cover_probability=0.6,
                revision_recorded_at_utc=run1_time,
                # blank trigger fields -> must be enriched from the evidence log
            ),
            _revision_row(
                refresh_run_id="run2",
                game_id=GAME_B,
                home_team="DDD",
                away_team="CCC",
                kickoff=KICKOFF_B,
                decision_home_spread=1.5,
                previous_pick_side="HOME",
                new_pick_side="AWAY",
                new_home_cover_probability=0.35,
                revision_recorded_at_utc=run2_time,
            ),
        ],
    )

    _write_evidence_log(
        artifacts_root,
        [
            {
                "trigger_source": TRIGGER_INJURY_REPORT_POSTED,
                "game_id": GAME_A,
                "season": SEASON,
                "week": WEEK,
                "observation_time": "2024-09-06T11:00:00+00:00",
                "source_capture_time": "2024-09-06T11:30:00+00:00",
                "checkpoint_name": None,
                "deadline": "2024-09-08T17:00:00+00:00",
                "deadline_valid": True,
                "deadline_reason": "ok",
                "detail": "injury report posted",
            },
            {
                "trigger_source": TRIGGER_CLOCK_CHECKPOINT,
                "game_id": GAME_B,
                "season": SEASON,
                "week": WEEK,
                "observation_time": "2024-09-07T08:00:00+00:00",
                "source_capture_time": "2024-09-07T08:30:00+00:00",
                "checkpoint_name": "refresh_sat",
                "deadline": "2024-09-08T18:00:00+00:00",
                "deadline_valid": True,
                "deadline_reason": "ok",
                "detail": "scheduled clock checkpoint",
            },
        ],
    )

    diff = build_snapshot_diff(SEASON, WEEK, artifacts_root=artifacts_root)
    passes = {
        p.refresh_run_id: p for p in diff.refresh_passes if p.origin == PASS_ORIGIN_PICK_REVISION
    }

    run1 = passes["run1"]
    assert run1.trigger_basis == "evidence_log_nearest"
    assert run1.trigger_source == TRIGGER_INJURY_REPORT_POSTED
    assert run1.trigger_type == TRIGGER_NEWS_EVENT

    run2 = passes["run2"]
    assert run2.trigger_basis == "evidence_log_nearest"
    assert run2.trigger_source == TRIGGER_CLOCK_CHECKPOINT
    assert run2.trigger_type == TRIGGER_CLOCK_DISPATCH


def test_diff_game_overlay_added_and_removed() -> None:
    """Direct unit test of the overlay-diff primitive.

    This project's only two live refresh channels never actually surface an
    ``added`` overlay in production today (``refresh-picks`` freezes overlays
    by design; a ``margin-predict`` artifact carries no overlay data at all),
    so this exercises :func:`nfl_ats.snapshot_diff._diff_game` directly to
    prove the underlying added/removed/unchanged computation is correct for
    whichever future channel does supply two differing overlay sets.
    """

    tuesday = GameSnapshot(
        game_id="g1",
        home_team="H",
        away_team="A",
        kickoff=None,
        market_line=-3.0,
        market_line_basis="test",
        model_probability=0.50,
        model_probability_basis="test",
        pick_side="HOME",
        pick_basis="test",
        overlays_fired=("coach_fade",),
        overlays_basis="test",
    )
    refresh = GameSnapshot(
        game_id="g1",
        home_team="H",
        away_team="A",
        kickoff=None,
        market_line=-3.0,
        market_line_basis="test",
        model_probability=0.55,
        model_probability_basis="test",
        pick_side="HOME",
        pick_basis="test",
        overlays_fired=("division_revenge", "coach_fade"),
        overlays_basis="test",
    )
    row = _diff_game(tuesday, refresh)
    assert row.overlays_added == ("division_revenge",)
    assert row.overlays_removed == ()
    assert row.overlays_unchanged == ("coach_fade",)
    assert row.overlay_state == STATE_CHANGED


def test_forecast_artifact_pass_has_real_deltas_but_no_data_sources_and_overlays(
    tmp_path: Path,
) -> None:
    artifacts_root = tmp_path / "artifacts"
    _setup_tuesday(artifacts_root)

    later_dir_name = "2024-week-01-20240905T090000Z"
    _write_artifact(
        artifacts_root / "margin_predictions" / later_dir_name,
        created_at_utc="2024-09-05T09:00:00+00:00",
        games=[
            {
                "game_id": GAME_A,
                "home_team": "BBB",
                "away_team": "AAA",
                "kickoff": "2024-09-08T17:00:00+00:00",
                "spread_line": -3.0,
                "home_cover_probability": 0.60,
                "bet_side": "HOME",
            },
            {
                "game_id": GAME_B,
                "home_team": "DDD",
                "away_team": "CCC",
                "kickoff": "2024-09-08T18:00:00+00:00",
                "spread_line": 1.5,
                "home_cover_probability": 0.55,
                "bet_side": "HOME",
            },
        ],
    )

    diff = build_snapshot_diff(SEASON, WEEK, artifacts_root=artifacts_root)
    forecast_passes = [p for p in diff.refresh_passes if p.origin == PASS_ORIGIN_FORECAST_ARTIFACT]
    assert len(forecast_passes) == 1
    refresh_pass = forecast_passes[0]

    # source-timestamp cells: no lineage.json on either side -> every cell no_data,
    # but never an empty tuple (there's always at least the required fields).
    assert refresh_pass.sources
    assert all(cell.state == STATE_NO_DATA for cell in refresh_pass.sources)
    assert all(cell.source_id for cell in refresh_pass.sources)

    by_game = {row.game_id: row for row in refresh_pass.games}
    game_a = by_game[GAME_A]
    # market line and probability ARE real, comparable data for this channel
    assert game_a.market_line_state == STATE_UNCHANGED  # -3.0 == -3.0
    assert game_a.tuesday_model_probability == 0.40
    assert game_a.refresh_model_probability == 0.60
    assert game_a.probability_state == STATE_CHANGED
    assert game_a.pick_state == "flipped_away_to_home"
    # overlays are never observable from a margin-predict artifact alone
    assert game_a.overlay_state == STATE_NO_DATA


def test_render_markdown_never_leaves_a_blank_cell(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _setup_tuesday(artifacts_root)
    _write_pick_revisions(
        artifacts_root,
        [
            _revision_row(
                refresh_run_id="run1",
                game_id=GAME_A,
                home_team="BBB",
                away_team="AAA",
                kickoff=KICKOFF_A,
                decision_home_spread=-3.0,
                previous_pick_side="AWAY",
                new_pick_side="HOME",
                new_home_cover_probability=0.62,
                revision_recorded_at_utc=FRIDAY,
                trigger_type=TRIGGER_NEWS_EVENT,
                trigger_source="injury_report_posted",
                trigger_observed_at_utc=FRIDAY - pd.Timedelta(hours=1),
            ),
        ],
    )

    diff = build_snapshot_diff(SEASON, WEEK, artifacts_root=artifacts_root)
    markdown = render_markdown(diff)

    assert "None" not in markdown
    table_rows_checked = 0
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        interior = stripped.strip("|")
        if set(interior.replace("-", "").replace(":", "").strip()) == set():
            continue  # a "|---|---|" separator row
        cells = [cell.strip() for cell in interior.split("|")]
        assert all(cells), f"blank cell found in rendered row: {line!r}"
        table_rows_checked += 1
    assert table_rows_checked > 0


def test_to_dict_and_to_json_round_trip(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _setup_tuesday(artifacts_root)
    _write_pick_revisions(
        artifacts_root,
        [
            _revision_row(
                refresh_run_id="run1",
                game_id=GAME_A,
                home_team="BBB",
                away_team="AAA",
                kickoff=KICKOFF_A,
                decision_home_spread=-3.0,
                previous_pick_side="AWAY",
                new_pick_side="HOME",
                new_home_cover_probability=0.62,
                revision_recorded_at_utc=FRIDAY,
            ),
        ],
    )
    diff = build_snapshot_diff(SEASON, WEEK, artifacts_root=artifacts_root)

    payload = to_dict(diff)
    assert payload["season"] == SEASON
    assert payload["week"] == WEEK
    assert len(payload["refresh_passes"]) == len(diff.refresh_passes)

    text = to_json(diff)
    reparsed = json.loads(text)
    assert reparsed["season"] == SEASON
    assert reparsed["tuesday"]["resolved"] is True
