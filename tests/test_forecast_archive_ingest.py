from __future__ import annotations

import importlib.util
import json
import urllib.error
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ingest_forecast_archive_test", REPO / "scripts" / "ingest_forecast_archive.py"
)
assert SPEC is not None and SPEC.loader is not None
INGEST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INGEST)


def test_pool_decision_mode_uses_gfs_and_caps_snf_at_sunday_1600_et() -> None:
    kickoff = pd.Timestamp("2025-09-08T00:20:00Z")
    cutoff = INGEST.decision_cutoff_utc(kickoff, "pool_decision")
    assert INGEST.MOS_MODEL_BY_CUTOFF_MODE["pool_decision"] == "GFS"
    assert cutoff == pd.Timestamp("2025-09-07T20:00:00Z")
    assert max(INGEST.candidate_runtimes(cutoff, 10)) <= cutoff


def test_pool_decision_mode_keeps_early_games_at_their_own_kickoff() -> None:
    kickoff = pd.Timestamp("2025-09-07T17:00:00Z")
    assert INGEST.decision_cutoff_utc(kickoff, "pool_decision") == kickoff


def test_unknown_cutoff_mode_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unknown cutoff mode"):
        INGEST.decision_cutoff_utc(pd.Timestamp("2025-09-07T17:00:00Z"), "typo")


def test_ingest_rejects_a_bulletin_labeled_after_the_pool_cutoff(monkeypatch) -> None:
    def malformed(station: str, runtime_utc, *, model: str):
        return [
            {
                "runtime_utc": "2025-09-08T00:00:00Z",
                "ftime_utc": "2025-09-08T00:00:00Z",
                "tmp": 70,
                "wsp": 5,
            }
        ]

    monkeypatch.setattr(INGEST, "fetch_mos_bulletin", malformed)
    kickoff = pd.Timestamp("2025-09-08T00:20:00Z")
    cutoff = INGEST.decision_cutoff_utc(kickoff, "pool_decision")
    result = INGEST.fetch_one_game(
        "KXYZ",
        kickoff,
        cutoff,
        model="GFS",
        max_lookback_steps=1,
        delay_seconds=0.0,
    )
    assert result["fetch_status"] == "invalid_issuance_timestamp"


def test_http_404_is_an_expected_missing_bulletin(monkeypatch) -> None:
    def missing(*args, **kwargs):
        raise urllib.error.HTTPError("https://example.test", 404, "missing", {}, None)

    monkeypatch.setattr(INGEST.urllib.request, "urlopen", missing)
    assert (
        INGEST.fetch_mos_bulletin(
            "KSEA", pd.Timestamp("2010-12-05T12:00:00Z").to_pydatetime(), model="GFS"
        )
        == []
    )


def test_missing_runtime_continues_the_backward_search(monkeypatch) -> None:
    calls = []

    def fetch(station: str, runtime_utc, *, model: str):
        calls.append(runtime_utc)
        if len(calls) == 1:
            return []
        return [
            {
                "runtime_utc": runtime_utc.isoformat(),
                "ftime_utc": "2010-12-05T18:00:00",
                "tmp": 42,
                "wsp": 5,
            }
        ]

    monkeypatch.setattr(INGEST, "fetch_mos_bulletin", fetch)
    kickoff = pd.Timestamp("2010-12-05T21:00:00Z")
    result = INGEST.fetch_one_game(
        "KSEA",
        kickoff,
        kickoff,
        model="GFS",
        max_lookback_steps=2,
        delay_seconds=0.0,
    )
    assert result["fetch_status"] == "ok"
    assert result["lookback_steps_used"] == 1
    assert calls == [
        pd.Timestamp("2010-12-05T12:00:00Z").to_pydatetime(),
        pd.Timestamp("2010-12-05T00:00:00Z").to_pydatetime(),
    ]


def test_resume_rejects_rows_from_a_different_cutoff_mode(tmp_path: Path) -> None:
    (tmp_path / "run_config.json").write_text(
        json.dumps({"cutoff_mode": "pool_decision", "mos_model": "GFS"}),
        encoding="utf-8",
    )
    (tmp_path / "results.jsonl").write_text(
        json.dumps({"game_id": "snf", "cutoff_mode": "kickoff_nearest"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected 'pool_decision'"):
        INGEST.load_resume_cache(tmp_path, cutoff_mode="pool_decision", mos_model="GFS")


def test_resume_reuses_only_completed_rows_and_retries_failures(tmp_path: Path) -> None:
    (tmp_path / "run_config.json").write_text(
        json.dumps({"cutoff_mode": "pool_decision", "mos_model": "GFS"}),
        encoding="utf-8",
    )
    records = [
        {"game_id": "ok", "cutoff_mode": "pool_decision", "fetch_status": "ok"},
        {
            "game_id": "international",
            "cutoff_mode": "pool_decision",
            "fetch_status": "unmappable_international_stadium",
        },
        {
            "game_id": "transport",
            "cutoff_mode": "pool_decision",
            "fetch_status": "transport_error",
        },
        {
            "game_id": "invalid",
            "cutoff_mode": "pool_decision",
            "fetch_status": "invalid_issuance_timestamp",
        },
    ]
    (tmp_path / "results.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    cache = INGEST.load_resume_cache(tmp_path, cutoff_mode="pool_decision", mos_model="GFS")
    assert set(cache) == {"ok", "international"}

    # In-place resume rewrites the same attempt log before appending retries,
    # so failed/superseded rows cannot survive into the final parquet.
    INGEST.rewrite_resume_cache(tmp_path / "results.jsonl", cache)
    rewritten = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["game_id"] for row in rewritten] == ["ok", "international"]
