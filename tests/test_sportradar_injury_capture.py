from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.capture_scheduler as scheduler  # noqa: E402
import scripts.capture_sportradar_injuries as capture_module  # noqa: E402


def _schedule(path: Path) -> Path:
    pd.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "game_type": ["REG"],
            "gameday": ["2026-09-09"],
            "gametime": ["20:20"],
            "away_team": ["ARI"],
            "home_team": ["ATL"],
        }
    ).to_parquet(path, index=False)
    return path


def _payload(
    *, generated_at: str = "2026-09-02T15:30:00Z", teams: tuple[str, ...] = ("ARI", "ATL")
) -> bytes:
    blocks = []
    for number, team in enumerate(teams):
        blocks.append(
            {
                "id": f"team-{team}",
                "alias": team,
                "name": team,
                "market": team,
                "players": [
                    {
                        "id": f"player-{number}",
                        "sr_id": f"sr:player:{number}",
                        "name": f"Player {number}",
                        "position": "QB",
                        "injury": {
                            "primary": "Knee",
                            "status": "Questionable",
                            "status_date": "2026-09-02T00:00:00Z",
                            "practice": {"status": "Limited Participation In Practice"},
                        },
                    }
                ],
            }
        )
    return json.dumps(
        {
            "generated_at": generated_at,
            "season": {"year": 2026, "type": "REG", "id": "season"},
            "week": {"sequence": 1, "id": "week"},
            "injuries": blocks,
        }
    ).encode()


def test_capture_is_immutable_complete_and_does_not_persist_secret(
    private_raw_root: Path,
) -> None:
    # ENG-30: `capture()` enforces `source_policy.require_private_raw_destination`
    # on its output root before writing anything -- plain `tmp_path` trips that
    # guard when `--basetemp` is pointed in-repo. See conftest.py's
    # `private_raw_root` fixture.
    now = datetime(2026, 9, 2, 16, tzinfo=UTC)
    schedule = _schedule(private_raw_root / "schedule.parquet")
    calls: list[tuple[str, str]] = []

    def fetch(url: str, key: str) -> bytes:
        calls.append((url, key))
        return _payload()

    snapshot = capture_module.capture(
        private_raw_root / "captures",
        now=now,
        api_key="private-key",
        schedule_path=schedule,
        fetcher=fetch,
    )
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(snapshot / "injuries.parquet")

    assert manifest["status"] == "complete"
    assert manifest["coverage"]["required_teams"] == ["ARI", "ATL"]
    assert manifest["available_at_policy"] == "capture_time_not_provider_status_date"
    assert calls == [(manifest["source_url"], "private-key")]
    assert "private-key" not in json.dumps(manifest)
    assert frame["available_at_utc"].eq(pd.Timestamp(now)).all()
    for entry in manifest["files"]:
        assert (
            entry["sha256"] == hashlib.sha256((snapshot / entry["path"]).read_bytes()).hexdigest()
        )


def test_missing_credential_fails_before_output_and_network(
    private_raw_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ENG-30: `capture()` checks `require_private_raw_destination` before the
    # credential check, so this needs the out-of-repo fixture too, or an
    # in-repo `--basetemp` raises the wrong error before "is required" is hit.
    monkeypatch.delenv(capture_module.API_KEY_ENV, raising=False)
    called = False

    def fetch(_url: str, _key: str) -> bytes:
        nonlocal called
        called = True
        return b"{}"

    out = private_raw_root / "captures"
    with pytest.raises(capture_module.SportradarInjuryCaptureError, match="is required"):
        capture_module.capture(out, api_key=None, fetcher=fetch)
    assert not out.exists()
    assert not called


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_payload(teams=("ARI",)), "missing=\\['ATL'\\]"),
        (_payload(generated_at="2026-09-01T00:00:00Z"), "stale/future"),
        (b"{}", "Malformed Weekly Injuries response"),
    ],
)
def test_bad_response_leaves_failed_manifest_without_canonical_table(
    private_raw_root: Path, payload: bytes, message: str
) -> None:
    # ENG-30: see test_capture_is_immutable_complete_and_does_not_persist_secret.
    schedule = _schedule(private_raw_root / "schedule.parquet")
    out = private_raw_root / "captures"
    with pytest.raises(capture_module.SportradarInjuryCaptureError, match=message):
        capture_module.capture(
            out,
            now=datetime(2026, 9, 2, 16, tzinfo=UTC),
            api_key="key",
            schedule_path=schedule,
            fetcher=lambda _url, _key: payload,
        )
    snapshot = out / "20260902T160000Z"
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert not (snapshot / "injuries.parquet").exists()


def test_decision_loader_ignores_later_revision_and_verifies_hashes(
    private_raw_root: Path,
) -> None:
    # ENG-30: see test_capture_is_immutable_complete_and_does_not_persist_secret.
    schedule = _schedule(private_raw_root / "schedule.parquet")
    out = private_raw_root / "captures"
    first = capture_module.capture(
        out,
        now=datetime(2026, 9, 2, 16, tzinfo=UTC),
        api_key="key",
        schedule_path=schedule,
        fetcher=lambda _url, _key: _payload(generated_at="2026-09-02T15:30:00Z"),
    )
    capture_module.capture(
        out,
        now=datetime(2026, 9, 2, 17, tzinfo=UTC),
        api_key="key",
        schedule_path=schedule,
        fetcher=lambda _url, _key: _payload(generated_at="2026-09-02T16:30:00Z").replace(
            b"Questionable", b"Out"
        ),
    )

    selected, frame = capture_module.load_for_decision(
        out, datetime(2026, 9, 2, 16, 30, tzinfo=UTC)
    )
    assert selected == first
    assert frame["game_status"].eq("Questionable").all()

    (first / "injuries.parquet").write_bytes(b"corrupt")
    with pytest.raises(capture_module.SportradarInjuryCaptureError, match="SHA-256"):
        capture_module.load_for_decision(out, datetime(2026, 9, 2, 16, 30, tzinfo=UTC))


def test_scheduler_replacement_jobs_are_credential_gated_and_nflcom_stays_paused() -> None:
    nflcom = [job for job in scheduler.SCHEDULE if job.name.startswith("injuries_")]
    replacements = [
        job for job in scheduler.SCHEDULE if job.name.startswith("sportradar_injuries_")
    ]

    assert len(nflcom) == len(replacements) == 4
    assert all(not job.enabled for job in nflcom)
    assert all("ingest_nflcom_injuries.py" in " ".join(job.command) for job in nflcom)
    assert all("capture_sportradar_injuries.py" in " ".join(job.command) for job in replacements)
    assert all(job.enabled == scheduler.SPORTRADAR_INJURY_CAPTURE_ENABLED for job in replacements)
