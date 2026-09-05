"""Tests for the MOD-17 side-ledger challenger (``nfl_ats.served_total_challenger``).

Covers registration (the fast lock-day static wiring audit,
``scripts/lockday_rehearsal.py``, already covers CLI dispatch -- these tests
cover the registry entry's own shape) and the recorder's behaviour: one row
per week, idempotent, pre-kickoff only, and realised-total backfill.
``tiebreaker_report`` itself is monkeypatched so these tests need no
production data root -- the recorder's OWN plumbing is what is under test,
not the tiebreaker pipeline (covered by ``tests/test_tiebreaker.py`` and
``tests/test_served_total.py``).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

import nfl_ats.served_total_challenger as challenger_module
from nfl_ats.served_total_challenger import (
    CHALLENGER_ID,
    LEDGER_COLUMNS,
    load_decisions,
    record_totals_served_method_decisions,
    settle_realised_totals,
)
from nfl_ats.tiebreaker import MarketConsensus, TiebreakerReport

# ---------------------------------------------------------------------------
# 0. Registration shape (the fast lockday audit, scripts/lockday_rehearsal.py,
#    already checks the CLI-dispatch contract separately).
# ---------------------------------------------------------------------------


def test_challenger_is_registered_active_prospective_on_the_publish_path() -> None:
    registry = json.loads(
        Path("artifacts/prospective/challengers.json").read_text(encoding="utf-8")
    )
    entries = [
        entry for entry in registry["challengers"] if entry.get("challenger_id") == CHALLENGER_ID
    ]
    assert len(entries) == 1, "totals_served_method must be registered exactly once"
    entry = entries[0]
    assert entry["status"] == "ACTIVE_PROSPECTIVE"
    assert "publish-predictions --record-decisions" in entry["weekly_recording_command"]


# ---------------------------------------------------------------------------
# 1. Fixtures
# ---------------------------------------------------------------------------


def _write_registry(artifacts_root: Path, *, status: str = "ACTIVE_PROSPECTIVE") -> None:
    prospective = artifacts_root / "prospective"
    prospective.mkdir(parents=True, exist_ok=True)
    registry = {
        "challengers": [
            {
                "challenger_id": CHALLENGER_ID,
                "status": status,
                "weekly_recording_command": "nfl-ats publish-predictions --record-decisions",
                "model": {},
            }
        ]
    }
    (prospective / "challengers.json").write_text(json.dumps(registry), encoding="utf-8")


def _write_schedules(
    data_root: Path,
    *,
    game_id: str = "2026_01_DEN_KC",
    gameday: str = "2026-09-10",
    gametime: str = "20:15",
    home_score: float | None = None,
    away_score: float | None = None,
) -> None:
    raw = data_root / "raw" / "20260901T000000Z"
    raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "game_id": [game_id],
            "season": [2026],
            "week": [1],
            "game_type": ["REG"],
            "gameday": [gameday],
            "gametime": [gametime],
            "home_team": ["KC"],
            "away_team": ["DEN"],
            "home_score": [home_score],
            "away_score": [away_score],
            "spread_line": [2.5],
            "total_line": [43.0],
        }
    ).to_parquet(raw / "schedules.parquet")


def _fixed_report(
    *,
    game_id: str = "2026_01_DEN_KC",
    served_total_method: str = "joint_residual",
    served_total: float = 43.73,
    comparison_total_blend_k01: float = 43.62,
) -> TiebreakerReport:
    consensus = MarketConsensus(
        game_id=game_id, home_expected_margin=2.5, total_line=43.0, source="test"
    )
    return TiebreakerReport(
        game_id=game_id,
        home="KC",
        away="DEN",
        consensus=consensus,
        model_view=None,
        totals_view=None,
        guess_margin=2.5,
        guess_total_line=served_total,
        served_total_method=served_total_method,  # type: ignore[arg-type]
        comparison_total_blend_k01=comparison_total_blend_k01,
        implied_home=23.1,
        implied_away=20.6,
        neighborhood_games=150,
        neighborhood_window="test",
        median_total=43.0,
        median_home_margin=3.0,
        guess_home=24,
        guess_away=20,
        common_scores=((24, 20, 2.0),),
        total_mae=10.5,
        total_median_ae=9.0,
        total_bias=0.5,
        implied_score_mae=7.4,
    )


# ---------------------------------------------------------------------------
# 2. record_totals_served_method_decisions
# ---------------------------------------------------------------------------


def test_record_refuses_when_challenger_is_not_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_registry(artifacts_root, status="CLOSED_BEFORE_ACTIVATION")
    _write_schedules(data_root)
    monkeypatch.setattr(challenger_module, "tiebreaker_report", lambda *a, **k: _fixed_report())

    with pytest.raises(ValueError, match="only ACTIVE_PROSPECTIVE"):
        record_totals_served_method_decisions(
            artifacts_root, data_root, now=datetime(2026, 9, 8, tzinfo=UTC)
        )


def test_record_writes_one_row_pre_kickoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_registry(artifacts_root)
    _write_schedules(data_root)  # kickoff 2026-09-10 20:15 ET, unplayed
    monkeypatch.setattr(challenger_module, "tiebreaker_report", lambda *a, **k: _fixed_report())

    result = record_totals_served_method_decisions(
        artifacts_root, data_root, now=datetime(2026, 9, 8, 16, 0, tzinfo=UTC)
    )

    assert result["recorded"] == 1
    assert result["already_recorded"] == 0
    assert result["post_kickoff_skipped"] == 0
    assert result["served_total_method"] == "joint_residual"
    assert result["served_total_blend_k01"] == pytest.approx(43.62)
    assert result["served_total_joint_residual"] == pytest.approx(43.73)

    decisions = load_decisions(artifacts_root)
    assert len(decisions) == 1
    row = decisions.iloc[0]
    assert row["challenger_id"] == CHALLENGER_ID
    assert row["game_id"] == "2026_01_DEN_KC"
    assert row["season"] == 2026
    assert row["week"] == 1
    assert row["served_total_method"] == "joint_residual"
    assert row["market_total"] == pytest.approx(43.0)
    assert row["served_total_blend_k01"] == pytest.approx(43.62)
    assert row["served_total_joint_residual"] == pytest.approx(43.73)
    assert pd.isna(row["realised_total"])
    assert set(decisions.columns) == set(LEDGER_COLUMNS)


def test_record_carries_nan_joint_column_when_blend_k01_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the joint model could not price the game, the report itself
    already degraded to blend_k01 (nfl_ats.served_total.served_total's
    fallback) -- the ledger must not invent a joint number that was never
    actually served."""

    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_registry(artifacts_root)
    _write_schedules(data_root)
    monkeypatch.setattr(
        challenger_module,
        "tiebreaker_report",
        lambda *a, **k: _fixed_report(served_total_method="blend_k01", served_total=43.62),
    )

    record_totals_served_method_decisions(
        artifacts_root, data_root, now=datetime(2026, 9, 8, 16, 0, tzinfo=UTC)
    )
    row = load_decisions(artifacts_root).iloc[0]
    assert row["served_total_method"] == "blend_k01"
    assert pd.isna(row["served_total_joint_residual"])


def test_record_is_idempotent_across_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_registry(artifacts_root)
    _write_schedules(data_root)
    monkeypatch.setattr(challenger_module, "tiebreaker_report", lambda *a, **k: _fixed_report())

    now = datetime(2026, 9, 8, 16, 0, tzinfo=UTC)
    first = record_totals_served_method_decisions(artifacts_root, data_root, now=now)
    second = record_totals_served_method_decisions(artifacts_root, data_root, now=now)

    assert first["recorded"] == 1
    assert second["recorded"] == 0
    assert second["already_recorded"] == 1
    assert len(load_decisions(artifacts_root)) == 1


def test_record_skips_without_writing_once_kickoff_has_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_registry(artifacts_root)
    _write_schedules(data_root)  # kickoff 2026-09-10T20:15 America/New_York
    monkeypatch.setattr(challenger_module, "tiebreaker_report", lambda *a, **k: _fixed_report())

    result = record_totals_served_method_decisions(
        artifacts_root, data_root, now=datetime(2026, 9, 11, 4, 0, tzinfo=UTC)
    )

    assert result["recorded"] == 0
    assert result["post_kickoff_skipped"] == 1
    assert load_decisions(artifacts_root).empty


def test_record_refuses_far_outside_the_recording_lock_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts_root = tmp_path / "artifacts"
    data_root = tmp_path / "data"
    _write_registry(artifacts_root)
    _write_schedules(data_root)  # kickoff 2026-09-10
    monkeypatch.setattr(challenger_module, "tiebreaker_report", lambda *a, **k: _fixed_report())

    with pytest.raises(ValueError, match="Refusing to record"):
        record_totals_served_method_decisions(
            artifacts_root, data_root, now=datetime(2026, 7, 1, tzinfo=UTC)
        )


# ---------------------------------------------------------------------------
# 3. settle_realised_totals
# ---------------------------------------------------------------------------


def test_settle_realised_totals_backfills_only_pending_rows() -> None:
    decisions = pd.DataFrame(
        [
            {
                "recorded_at_utc": pd.Timestamp("2026-09-08T16:00:00Z"),
                "challenger_id": CHALLENGER_ID,
                "served_total_method": "joint_residual",
                "game_id": "2026_01_DEN_KC",
                "season": 2026,
                "week": 1,
                "kickoff": pd.Timestamp("2026-09-11T00:15:00Z"),
                "home_team": "KC",
                "away_team": "DEN",
                "market_total": 43.0,
                "served_total_blend_k01": 43.62,
                "served_total_joint_residual": 43.73,
                "realised_total": float("nan"),
            },
            {
                "recorded_at_utc": pd.Timestamp("2026-09-01T16:00:00Z"),
                "challenger_id": CHALLENGER_ID,
                "served_total_method": "blend_k01",
                "game_id": "2025_18_A_B",
                "season": 2025,
                "week": 18,
                "kickoff": pd.Timestamp("2026-01-04T18:00:00Z"),
                "home_team": "B",
                "away_team": "A",
                "market_total": 40.0,
                "served_total_blend_k01": 40.5,
                "served_total_joint_residual": float("nan"),
                "realised_total": 47.0,  # already settled -- must not change
            },
        ]
    )[list(LEDGER_COLUMNS)]
    schedules = pd.DataFrame(
        {
            "game_id": ["2026_01_DEN_KC", "2025_18_A_B"],
            "home_score": [24.0, 999.0],  # 999 must be ignored -- already settled
            "away_score": [20.0, 999.0],
        }
    )

    settled = settle_realised_totals(decisions, schedules)

    pending_row = settled.loc[settled["game_id"] == "2026_01_DEN_KC"].iloc[0]
    assert pending_row["realised_total"] == pytest.approx(44.0)
    settled_row = settled.loc[settled["game_id"] == "2025_18_A_B"].iloc[0]
    assert settled_row["realised_total"] == pytest.approx(47.0)  # unchanged
    assert list(settled.columns) == list(LEDGER_COLUMNS)


def test_settle_realised_totals_is_a_noop_on_an_empty_ledger() -> None:
    empty = pd.DataFrame(columns=list(LEDGER_COLUMNS))
    schedules = pd.DataFrame({"game_id": [], "home_score": [], "away_score": []})
    assert settle_realised_totals(empty, schedules).empty
