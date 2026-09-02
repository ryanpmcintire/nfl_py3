"""Contracts for the lock-day ledger coverage report."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import lockday_verify


def _registry(*commands: tuple[str, str]) -> dict[str, object]:
    return {
        "challengers": [
            {
                "challenger_id": challenger_id,
                "status": "ACTIVE_PROSPECTIVE",
                "weekly_recording_command": command,
            }
            for challenger_id, command in commands
        ]
    }


def _week_rows(*challenger_ids: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "challenger_id": list(challenger_ids),
            "season": [2026] * len(challenger_ids),
            "week": [1] * len(challenger_ids),
        }
    )


def test_verify_separates_missing_publish_from_unwired_paths(monkeypatch, tmp_path: Path) -> None:
    active = [
        "wired_publish",
        "bye_edge_fade_overlay",
        "inactives_refresh_v1",
        "crew_tilt_refresh_v1",
    ]
    monkeypatch.setattr(lockday_verify, "active_challenger_ids", lambda _root: active)
    monkeypatch.setattr(
        lockday_verify,
        "load_challenger_registry",
        lambda _root: _registry(
            ("wired_publish", "nfl-ats publish-predictions --record-decisions"),
            ("bye_edge_fade_overlay", "python scripts/record_bye_edge_fade_challenger.py"),
            ("inactives_refresh_v1", "python -c record_inactives_refresh_overlay"),
            ("crew_tilt_refresh_v1", "N/A YET -- refresh-picks"),
            ("retired_arm", "nfl-ats publish-predictions --record-decisions"),
        ),
    )
    monkeypatch.setattr(lockday_verify, "load_challenger_decisions", lambda _root: _week_rows())
    monkeypatch.setattr(lockday_verify, "load_paper_decisions", lambda _root: pd.DataFrame())

    def refresh_rows(_root: Path) -> pd.DataFrame:
        return pd.DataFrame({"season": [2026], "week": [1]})

    monkeypatch.setitem(
        lockday_verify.PENDING_REFRESH_LEDGERS["inactives_refresh_v1"], "loader", refresh_rows
    )
    monkeypatch.setitem(
        lockday_verify.PENDING_REFRESH_LEDGERS["crew_tilt_refresh_v1"],
        "loader",
        lambda _root: pd.DataFrame(columns=["season", "week"]),
    )

    report = lockday_verify.verify(tmp_path, season=2026, week=1)
    rows = {row["challenger_id"]: row for row in report["challengers"]}

    assert report["active_registered"] == 4
    assert report["missing"] == ["wired_publish"]
    assert report["pending_wiring"] == [
        "bye_edge_fade_overlay",
        "crew_tilt_refresh_v1",
        "inactives_refresh_v1",
    ]
    assert rows["wired_publish"]["recording_path"] == "publish"
    assert rows["bye_edge_fade_overlay"]["recording_path"] == "standalone_pending_wiring"
    assert rows["bye_edge_fade_overlay"]["status"] == "PENDING_WIRING"
    assert rows["inactives_refresh_v1"]["recording_path"] == "refresh/dedicated"
    assert rows["inactives_refresh_v1"]["status"] == "recorded"
    assert rows["crew_tilt_refresh_v1"]["status"] == "PENDING_WIRING"
    assert "retired_arm" not in rows


def test_verify_keeps_legitimate_wired_refresh_empty_as_skipped(
    monkeypatch, tmp_path: Path
) -> None:
    challenger_id = "injury_signal_refresh_tilt"
    monkeypatch.setattr(lockday_verify, "active_challenger_ids", lambda _root: [challenger_id])
    monkeypatch.setattr(lockday_verify, "load_challenger_registry", lambda _root: _registry())
    monkeypatch.setattr(lockday_verify, "load_challenger_decisions", lambda _root: pd.DataFrame())
    monkeypatch.setattr(lockday_verify, "load_paper_decisions", lambda _root: pd.DataFrame())
    monkeypatch.setattr(
        lockday_verify,
        "load_injury_signal_decisions",
        lambda _root: pd.DataFrame(columns=["season", "week"]),
    )

    report = lockday_verify.verify(tmp_path, season=2026, week=1)
    row = report["challengers"][0]

    assert row["recording_path"] == "refresh/dedicated"
    assert row["status"] == "skipped"
    assert report["missing"] == []
    assert report["pending_wiring"] == []


def test_render_shows_pending_wiring_separately() -> None:
    rendered = lockday_verify.render(
        {
            "season": 2026,
            "week": 1,
            "artifacts_root": "artifacts",
            "paper_ledger_rows": 0,
            "paper_best_pick": None,
            "recorded": 0,
            "skipped": 0,
            "missing": [],
            "pending_wiring": ["arm"],
            "active_registered": 1,
            "challengers": [
                {
                    "challenger_id": "arm",
                    "rows": 0,
                    "status": "PENDING_WIRING",
                    "note": "wire it",
                }
            ],
        }
    )

    assert "1 pending wiring" in rendered
    assert "?? arm" in rendered
