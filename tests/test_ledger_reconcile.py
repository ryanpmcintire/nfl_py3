"""ENG-15: ledger reconciliation and recovery.

Every test builds a synthetic ``artifacts/`` tree under ``tmp_path`` -- no
test here reads or writes the real repository's ``artifacts/`` or ``data/``.
``repo_root`` is deliberately still the REAL repository root for every call
into :func:`reconcile`: that is how it locates the real, tracked
``scripts/lockday_verify.py`` (read-only source code, not data) to resolve
the dedicated-ledger map, exactly the way the production CLI does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nfl_ats.clv import paper_decision_ledger_path
from nfl_ats.inactives_refresh_overlay import inactives_refresh_overlay_ledger_path
from nfl_ats.ledger_reconcile import (
    STATUS_CARD_MISMATCH,
    STATUS_CONSISTENT,
    STATUS_DUPLICATE_ROWS,
    STATUS_MISSING_ROWS,
    STATUS_NOT_RUN,
    STATUS_ORPHAN_ROWS,
    Declaration,
    _derive_rerun_command,
    parse_published_card,
    reconcile,
    render,
)
from nfl_ats.prospective_scoring import challenger_ledger_path, challenger_registry_path

REPO_ROOT = Path(__file__).resolve().parents[1]
SEASON = 2026
WEEK = 1


# ---------------------------------------------------------------------------
# fixtures / builders
# ---------------------------------------------------------------------------


def _registry(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"challengers": list(entries)}


def _entry(challenger_id: str, command: str, status: str = "ACTIVE_PROSPECTIVE") -> dict[str, Any]:
    return {
        "challenger_id": challenger_id,
        "status": status,
        "weekly_recording_command": command,
    }


def _write_registry(artifacts_root: Path, *entries: dict[str, Any]) -> None:
    path = challenger_registry_path(artifacts_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_registry(*entries)), encoding="utf-8")


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


CARD_TEXT = """# NFL ATS predictions: 2026 Week 1

Published from synchronized model `deadbeef12345678` at `2026-09-08T16:00:00+00:00`.

Active model: `market_residual` with `weak_stack` features.

| Date        | Matchup    | ATS prediction   | Decision score   |
|:------------|:-----------|:-----------------|:-----------------|
| Sun, Sep 13 | AAA at BBB | BBB -3.5         | 51.6%            |
| Sun, Sep 13 | CCC at DDD | ★ CCC +3.5       | 54.2%            |
"""


def _write_card(path: Path, text: str = CARD_TEXT) -> None:
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# parse_published_card
# ---------------------------------------------------------------------------


def test_parse_published_card_reads_heading_model_and_picks(tmp_path: Path) -> None:
    card_path = tmp_path / "CURRENT_PREDICTIONS.md"
    _write_card(card_path)

    parsed = parse_published_card(card_path)

    assert parsed["exists"] is True
    assert parsed["season"] == 2026
    assert parsed["week"] == 1
    assert parsed["model_id"] == "deadbeef12345678"
    games = parsed["games"]
    assert games["2026_01_AAA_BBB"]["pick_side"] == "HOME"
    assert games["2026_01_AAA_BBB"]["is_best_pick"] is False
    assert games["2026_01_CCC_DDD"]["pick_side"] == "AWAY"
    assert games["2026_01_CCC_DDD"]["is_best_pick"] is True
    assert games["2026_01_CCC_DDD"]["pick_team"] == "CCC"


def test_parse_published_card_missing_file_reports_exists_false(tmp_path: Path) -> None:
    parsed = parse_published_card(tmp_path / "does_not_exist.md")
    assert parsed == {
        "exists": False,
        "path": str(tmp_path / "does_not_exist.md"),
        "season": None,
        "week": None,
        "model_id": None,
        "games": {},
    }


def test_parse_published_card_rejects_a_file_without_the_heading(tmp_path: Path) -> None:
    from nfl_ats.ledger_reconcile import CardParseError

    path = tmp_path / "bad.md"
    path.write_text("not a card\n", encoding="utf-8")
    with pytest.raises(CardParseError):
        parse_published_card(path)


# ---------------------------------------------------------------------------
# _derive_rerun_command
# ---------------------------------------------------------------------------


def test_derive_rerun_command_substitutes_season_and_week() -> None:
    raw = (
        ".\\.tools\\uv.exe run python -m nfl_ats prospective-record "
        "--challenger mod07_weak_signal_stack --season 2026 --week <N>"
    )
    command, raw_out = _derive_rerun_command(raw, season=2025, week=18)
    assert command is not None
    assert "--season 2025" in command
    assert "--week 18" in command
    assert raw_out == raw


def test_derive_rerun_command_returns_none_for_na_prose() -> None:
    raw = "N/A -- there is no dedicated command; see refresh-picks --record-decisions"
    command, raw_out = _derive_rerun_command(raw, season=2026, week=1)
    assert command is None
    assert raw_out == raw


def test_derive_rerun_command_strips_trailing_parenthetical() -> None:
    raw = "nfl-ats publish-predictions --record-decisions (also records the paper ledger)"
    command, _ = _derive_rerun_command(raw, season=2026, week=1)
    assert command == "nfl-ats publish-predictions --record-decisions"


# ---------------------------------------------------------------------------
# reconcile: the six classifications
# ---------------------------------------------------------------------------


def test_reconcile_active_model_consistent_when_ledger_matches_card(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _write_card(tmp_path / "CURRENT_PREDICTIONS.md")
    _write_parquet(
        paper_decision_ledger_path(artifacts_root),
        pd.DataFrame(
            {
                "season": [SEASON],
                "week": [WEEK],
                "game_id": ["2026_01_AAA_BBB"],
                "pick_side": ["HOME"],
                "forecast_artifact": ["2026-week-01-abc123"],
            }
        ),
    )

    report = reconcile(
        artifacts_root,
        season=SEASON,
        week=WEEK,
        repo_root=REPO_ROOT,
        card_path=tmp_path / "CURRENT_PREDICTIONS.md",
    )

    active_model = next(row for row in report["recorders"] if row["recorder_id"] == "active_model")
    assert active_model["status"] == STATUS_CONSISTENT
    assert report["card_available_for_requested_week"] is True


def test_reconcile_active_model_card_mismatch(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _write_card(tmp_path / "CURRENT_PREDICTIONS.md")
    _write_parquet(
        paper_decision_ledger_path(artifacts_root),
        pd.DataFrame(
            {
                "season": [SEASON],
                "week": [WEEK],
                "game_id": ["2026_01_AAA_BBB"],
                "pick_side": ["AWAY"],  # card says HOME (BBB) for this game
                "forecast_artifact": ["2026-week-01-abc123"],
            }
        ),
    )

    report = reconcile(
        artifacts_root,
        season=SEASON,
        week=WEEK,
        repo_root=REPO_ROOT,
        card_path=tmp_path / "CURRENT_PREDICTIONS.md",
    )

    active_model = next(row for row in report["recorders"] if row["recorder_id"] == "active_model")
    assert active_model["status"] == STATUS_CARD_MISMATCH
    assert active_model["card_mismatches"][0]["game_id"] == "2026_01_AAA_BBB"
    assert report["all_consistent"] is False
    plan = {entry["recorder_id"]: entry for entry in report["recovery_plan"]}
    assert "EXPECTED" in plan["active_model"]["rerun_safety_note"]


def test_reconcile_active_model_duplicate_rows(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _write_parquet(
        paper_decision_ledger_path(artifacts_root),
        pd.DataFrame(
            {
                "season": [SEASON, SEASON],
                "week": [WEEK, WEEK],
                "game_id": ["2026_01_AAA_BBB", "2026_01_AAA_BBB"],
                "pick_side": ["HOME", "AWAY"],
            }
        ),
    )

    report = reconcile(
        artifacts_root,
        season=SEASON,
        week=WEEK,
        repo_root=REPO_ROOT,
        card_path=tmp_path / "CURRENT_PREDICTIONS.md",
    )

    active_model = next(row for row in report["recorders"] if row["recorder_id"] == "active_model")
    assert active_model["status"] == STATUS_DUPLICATE_ROWS
    assert active_model["duplicate_keys"][0]["game_id"] == "2026_01_AAA_BBB"
    plan = {entry["recorder_id"]: entry for entry in report["recovery_plan"]}
    assert plan["active_model"]["rerun_is_safe"] is False


def test_reconcile_active_model_missing_rows_from_run_summary(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    run_summary_path = tmp_path / "run_summary.json"
    run_summary_path.write_text(
        json.dumps({"clv_ledger": {"recorded": 3, "season": SEASON, "week": WEEK}}),
        encoding="utf-8",
    )
    # no ledger file written at all -> zero rows this week

    report = reconcile(
        artifacts_root,
        season=SEASON,
        week=WEEK,
        repo_root=REPO_ROOT,
        run_summary_path=run_summary_path,
    )

    active_model = next(row for row in report["recorders"] if row["recorder_id"] == "active_model")
    assert active_model["status"] == STATUS_MISSING_ROWS
    assert active_model["declared_recorded"] == 3


def test_reconcile_shared_ledger_challenger_orphan_rows(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _write_registry(artifacts_root)  # empty registry: nothing is registered
    _write_parquet(
        challenger_ledger_path(artifacts_root),
        pd.DataFrame(
            {
                "season": [SEASON],
                "week": [WEEK],
                "challenger_id": ["ghost_challenger"],
                "game_id": ["2026_01_AAA_BBB"],
            }
        ),
    )

    report = reconcile(
        artifacts_root,
        season=SEASON,
        week=WEEK,
        repo_root=REPO_ROOT,
        card_path=tmp_path / "CURRENT_PREDICTIONS.md",
    )

    ghost = next(row for row in report["recorders"] if row["recorder_id"] == "ghost_challenger")
    assert ghost["status"] == STATUS_ORPHAN_ROWS
    assert ghost["kind"] == "orphan_shared_ledger_challenger"


def test_reconcile_shared_ledger_duplicate_rows(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _write_registry(
        artifacts_root,
        _entry("dup_challenger", "nfl-ats prospective-record --challenger dup_challenger"),
    )
    _write_parquet(
        challenger_ledger_path(artifacts_root),
        pd.DataFrame(
            {
                "season": [SEASON, SEASON],
                "week": [WEEK, WEEK],
                "challenger_id": ["dup_challenger", "dup_challenger"],
                "game_id": ["2026_01_AAA_BBB", "2026_01_AAA_BBB"],
            }
        ),
    )

    report = reconcile(
        artifacts_root,
        season=SEASON,
        week=WEEK,
        repo_root=REPO_ROOT,
        card_path=tmp_path / "CURRENT_PREDICTIONS.md",
    )

    row = next(row for row in report["recorders"] if row["recorder_id"] == "dup_challenger")
    assert row["status"] == STATUS_DUPLICATE_ROWS


def test_reconcile_shared_ledger_challenger_not_run_and_na_command(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _write_registry(
        artifacts_root,
        _entry(
            "na_challenger",
            "N/A -- there is no dedicated command; see refresh-picks --record-decisions",
        ),
    )
    # no ledger rows at all, no run summary/package

    report = reconcile(
        artifacts_root,
        season=SEASON,
        week=WEEK,
        repo_root=REPO_ROOT,
        card_path=tmp_path / "CURRENT_PREDICTIONS.md",
    )

    row = next(row for row in report["recorders"] if row["recorder_id"] == "na_challenger")
    assert row["status"] == STATUS_NOT_RUN
    plan = {entry["recorder_id"]: entry for entry in report["recovery_plan"]}
    assert plan["na_challenger"]["rerun_command"] is None
    assert "N/A" in plan["na_challenger"]["rerun_command_raw"]


def test_reconcile_inactive_registry_status_is_informational_not_orphan(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _write_registry(
        artifacts_root,
        _entry(
            "superseded_challenger",
            "nfl-ats publish-predictions --record-decisions",
            status="SUPERSEDED_BY_PROMOTION",
        ),
    )
    _write_parquet(
        challenger_ledger_path(artifacts_root),
        pd.DataFrame(
            {
                "season": [SEASON],
                "week": [WEEK],
                "challenger_id": ["superseded_challenger"],
                "game_id": ["2026_01_AAA_BBB"],
            }
        ),
    )

    report = reconcile(
        artifacts_root,
        season=SEASON,
        week=WEEK,
        repo_root=REPO_ROOT,
        card_path=tmp_path / "CURRENT_PREDICTIONS.md",
    )

    recorder_ids = {row["recorder_id"] for row in report["recorders"]}
    assert "superseded_challenger" not in recorder_ids
    info = report["informational_inactive_challengers_with_rows"]
    assert info[0]["challenger_id"] == "superseded_challenger"
    assert info[0]["registry_status"] == "SUPERSEDED_BY_PROMOTION"


def test_reconcile_shared_ledger_consistent(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _write_registry(
        artifacts_root,
        _entry(
            "good_challenger",
            "nfl-ats prospective-record --challenger good_challenger --season 2026 --week <N>",
        ),
    )
    _write_parquet(
        challenger_ledger_path(artifacts_root),
        pd.DataFrame(
            {
                "season": [SEASON],
                "week": [WEEK],
                "challenger_id": ["good_challenger"],
                "game_id": ["2026_01_AAA_BBB"],
                "source_artifact": ["2026-week-01-abc123"],
            }
        ),
    )

    report = reconcile(
        artifacts_root,
        season=SEASON,
        week=WEEK,
        repo_root=REPO_ROOT,
        card_path=tmp_path / "CURRENT_PREDICTIONS.md",
    )

    row = next(row for row in report["recorders"] if row["recorder_id"] == "good_challenger")
    assert row["status"] == STATUS_CONSISTENT
    plan_ids = {entry["recorder_id"] for entry in report["recovery_plan"]}
    assert "good_challenger" not in plan_ids


# ---------------------------------------------------------------------------
# dedicated (revision-log) ledgers: looser (game_id, refresh_run_id) key
# ---------------------------------------------------------------------------


def test_dedicated_ledger_different_refresh_run_id_is_not_a_duplicate(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _write_registry(
        artifacts_root,
        _entry("inactives_refresh_v1", "nfl-ats refresh-picks --record-decisions"),
    )
    _write_parquet(
        inactives_refresh_overlay_ledger_path(artifacts_root),
        pd.DataFrame(
            {
                "season": [SEASON, SEASON],
                "week": [WEEK, WEEK],
                "game_id": ["2026_01_AAA_BBB", "2026_01_AAA_BBB"],
                "refresh_run_id": ["run-1", "run-2"],
            }
        ),
    )

    report = reconcile(
        artifacts_root,
        season=SEASON,
        week=WEEK,
        repo_root=REPO_ROOT,
        card_path=tmp_path / "CURRENT_PREDICTIONS.md",
    )

    row = next(row for row in report["recorders"] if row["recorder_id"] == "inactives_refresh_v1")
    assert row["status"] == STATUS_CONSISTENT
    assert row["rows_this_week"] == 2


def test_dedicated_ledger_same_refresh_run_id_twice_is_a_duplicate(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _write_registry(
        artifacts_root,
        _entry("inactives_refresh_v1", "nfl-ats refresh-picks --record-decisions"),
    )
    _write_parquet(
        inactives_refresh_overlay_ledger_path(artifacts_root),
        pd.DataFrame(
            {
                "season": [SEASON, SEASON],
                "week": [WEEK, WEEK],
                "game_id": ["2026_01_AAA_BBB", "2026_01_AAA_BBB"],
                "refresh_run_id": ["run-1", "run-1"],
            }
        ),
    )

    report = reconcile(
        artifacts_root,
        season=SEASON,
        week=WEEK,
        repo_root=REPO_ROOT,
        card_path=tmp_path / "CURRENT_PREDICTIONS.md",
    )

    row = next(row for row in report["recorders"] if row["recorder_id"] == "inactives_refresh_v1")
    assert row["status"] == STATUS_DUPLICATE_ROWS


def test_dedicated_ledger_unwired_and_empty_is_not_run(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _write_registry(
        artifacts_root,
        _entry("inactives_refresh_v1", "N/A YET -- refresh-picks pending"),
    )
    # no ledger file at all -> zero rows

    report = reconcile(
        artifacts_root,
        season=SEASON,
        week=WEEK,
        repo_root=REPO_ROOT,
        card_path=tmp_path / "CURRENT_PREDICTIONS.md",
    )

    row = next(row for row in report["recorders"] if row["recorder_id"] == "inactives_refresh_v1")
    assert row["status"] == STATUS_NOT_RUN
    assert "not wired" in row["note"]


# ---------------------------------------------------------------------------
# run_id filter
# ---------------------------------------------------------------------------


def test_run_id_filter_restricts_ledger_rows(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _write_parquet(
        paper_decision_ledger_path(artifacts_root),
        pd.DataFrame(
            {
                "season": [SEASON, SEASON],
                "week": [WEEK, WEEK],
                "game_id": ["2026_01_AAA_BBB", "2026_01_CCC_DDD"],
                "pick_side": ["HOME", "AWAY"],
                "forecast_artifact": ["run-old", "run-new"],
            }
        ),
    )

    card_path = tmp_path / "CURRENT_PREDICTIONS.md"
    report_all = reconcile(
        artifacts_root, season=SEASON, week=WEEK, repo_root=REPO_ROOT, card_path=card_path
    )
    report_filtered = reconcile(
        artifacts_root,
        season=SEASON,
        week=WEEK,
        repo_root=REPO_ROOT,
        card_path=card_path,
        run_id="run-new",
    )

    all_rows = next(r for r in report_all["recorders"] if r["recorder_id"] == "active_model")
    filtered_rows = next(
        r for r in report_filtered["recorders"] if r["recorder_id"] == "active_model"
    )
    assert all_rows["rows_this_week"] == 2
    assert filtered_rows["rows_this_week"] == 1


# ---------------------------------------------------------------------------
# package manifest integration (ENG-01)
# ---------------------------------------------------------------------------


def test_reconcile_uses_package_manifest_for_missing_rows(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    package_path = tmp_path / "manifest.json"
    package_path.write_text(
        json.dumps(
            {
                "kind": "lockday_decision_package",
                "ledgers": [
                    {
                        "ledger": "paper_decisions",
                        "path": str(paper_decision_ledger_path(artifacts_root)),
                        "appended_rows": 5,
                    }
                ],
                "recorders": {"by_challenger_id": {}},
            }
        ),
        encoding="utf-8",
    )
    # no ledger file written -> zero rows this week, package says 5 appended

    report = reconcile(
        artifacts_root,
        season=SEASON,
        week=WEEK,
        repo_root=REPO_ROOT,
        card_path=tmp_path / "CURRENT_PREDICTIONS.md",
        package_path=package_path,
    )

    active_model = next(row for row in report["recorders"] if row["recorder_id"] == "active_model")
    assert active_model["status"] == STATUS_MISSING_ROWS
    assert active_model["declared_source"] == "package_ledger_diff"


def test_reconcile_tolerates_a_missing_or_malformed_package_path(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    bogus = tmp_path / "does_not_exist" / "manifest.json"

    report = reconcile(
        artifacts_root,
        season=SEASON,
        week=WEEK,
        repo_root=REPO_ROOT,
        card_path=tmp_path / "CURRENT_PREDICTIONS.md",
        package_path=bogus,
    )

    assert report["package_path"] == str(bogus)
    active_model = next(row for row in report["recorders"] if row["recorder_id"] == "active_model")
    assert active_model["declared_source"] == "none"


def test_build_declarations_prefers_package_over_run_summary() -> None:
    from nfl_ats.ledger_reconcile import build_declarations

    run_summary = {"clv_ledger": {"recorded": 1}}
    package_manifest = {
        "ledgers": [{"ledger": "paper_decisions", "appended_rows": 9}],
        "recorders": {"by_challenger_id": {}},
    }
    declared = build_declarations(run_summary=run_summary, package_manifest=package_manifest)
    assert declared["active_model"].recorded == 9
    assert declared["active_model"].source == "package_ledger_diff"


# ---------------------------------------------------------------------------
# idempotency: the binding guarantee
# ---------------------------------------------------------------------------


def test_reconcile_is_idempotent_and_never_writes(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _write_card(tmp_path / "CURRENT_PREDICTIONS.md")
    _write_registry(
        artifacts_root,
        _entry(
            "good_challenger",
            "nfl-ats prospective-record --challenger good_challenger --season 2026 --week <N>",
        ),
        _entry("na_challenger", "N/A -- see refresh-picks --record-decisions"),
    )
    paper_path = paper_decision_ledger_path(artifacts_root)
    _write_parquet(
        paper_path,
        pd.DataFrame(
            {
                "season": [SEASON],
                "week": [WEEK],
                "game_id": ["2026_01_AAA_BBB"],
                "pick_side": ["HOME"],
            }
        ),
    )
    shared_path = challenger_ledger_path(artifacts_root)
    _write_parquet(
        shared_path,
        pd.DataFrame(
            {
                "season": [SEASON],
                "week": [WEEK],
                "challenger_id": ["good_challenger"],
                "game_id": ["2026_01_AAA_BBB"],
            }
        ),
    )

    paper_bytes_before = paper_path.read_bytes()
    shared_bytes_before = shared_path.read_bytes()

    report_1 = reconcile(
        artifacts_root,
        season=SEASON,
        week=WEEK,
        repo_root=REPO_ROOT,
        card_path=tmp_path / "CURRENT_PREDICTIONS.md",
    )
    report_2 = reconcile(
        artifacts_root,
        season=SEASON,
        week=WEEK,
        repo_root=REPO_ROOT,
        card_path=tmp_path / "CURRENT_PREDICTIONS.md",
    )

    assert report_1 == report_2
    assert paper_path.read_bytes() == paper_bytes_before
    assert shared_path.read_bytes() == shared_bytes_before
    # render() must also work off either report without raising
    assert render(report_1) == render(report_2)


def test_reconcile_works_with_no_ledgers_no_registry_no_card(tmp_path: Path) -> None:
    """A brand-new artifacts tree (Week 1's real state) must not crash."""

    artifacts_root = tmp_path / "artifacts"
    report = reconcile(
        artifacts_root,
        season=SEASON,
        week=WEEK,
        repo_root=REPO_ROOT,
        card_path=tmp_path / "CURRENT_PREDICTIONS.md",
    )

    assert report["registry_available"] is False
    assert report["card_available_for_requested_week"] is False
    active_model = next(row for row in report["recorders"] if row["recorder_id"] == "active_model")
    assert active_model["status"] == STATUS_NOT_RUN
    assert report["all_consistent"] is False
    rendered = render(report)
    assert "active_model" in rendered


# ---------------------------------------------------------------------------
# render()
# ---------------------------------------------------------------------------


def test_render_includes_recovery_plan_for_non_consistent_recorders(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    report = reconcile(
        artifacts_root,
        season=SEASON,
        week=WEEK,
        repo_root=REPO_ROOT,
        card_path=tmp_path / "CURRENT_PREDICTIONS.md",
    )
    rendered = render(report)
    assert "recovery plan" in rendered
    assert "re-run:" in rendered


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------


def test_declaration_as_dict_round_trips() -> None:
    declaration = Declaration(recorded=2, reason="gated", error=None, source="run_summary")
    payload = declaration.as_dict()
    assert payload == {
        "declared_recorded": 2,
        "declared_reason": "gated",
        "declared_error": None,
        "declared_source": "run_summary",
    }
