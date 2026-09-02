from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nfl_ats.data import DataContractError
from nfl_ats.public_forecast_archive import (
    append_public_forecast_record,
    build_public_forecast_record,
    main,
    verify_public_forecast_archive,
)


def _forecasts(*, reverse: bool = False) -> pd.DataFrame:
    rows = [
        {
            "game_id": "2026_01_B_A",
            "season": 2026,
            "week": 1,
            "home_team": "A",
            "away_team": "B",
            "kickoff_utc": "2026-09-13T17:00:00Z",
            "spread_line": -3.0,
            "predicted_margin": 3.5,
            "home_win_probability": 0.61,
            "home_cover_probability": 0.52,
            "home_cover_probability_excluding_push": 0.50,
            "push_probability": 0.05,
            "home_loss_probability": 0.45,
        },
        {
            "game_id": "2026_01_D_C",
            "season": 2026,
            "week": 1,
            "home_team": "C",
            "away_team": "D",
            "kickoff_utc": "2026-09-14T00:20:00Z",
            "spread_line": 2.5,
            "predicted_margin": -1.0,
            "home_win_probability": 0.47,
            "home_cover_probability": 0.56,
            "home_cover_probability_excluding_push": 0.56,
            "push_probability": 0.0,
            "home_loss_probability": 0.44,
        },
    ]
    return pd.DataFrame(list(reversed(rows)) if reverse else rows)


def _provenance() -> dict[str, str]:
    return {
        "model_id": "model-1",
        "feature_profile": "weak_stack",
        "probability_method": "gaussian",
        "model_configuration_sha256": "a" * 64,
        "feature_table_sha256": "b" * 64,
        "prediction_artifact_sha256": "c" * 64,
    }


def _build(
    forecasts: pd.DataFrame | None = None,
    *,
    publication_id: str = "publication-1",
    published_at: str = "2026-09-08T16:00:00Z",
    previous: str | None = None,
) -> dict[str, object]:
    return build_public_forecast_record(
        _forecasts() if forecasts is None else forecasts,
        publication_id=publication_id,
        published_at_utc=published_at,
        decision_label="tuesday_lock",
        decision_at_utc="2026-09-08T16:00:00Z",
        inputs_observed_through_utc="2026-09-08T15:59:00Z",
        provenance=_provenance(),
        previous_record_sha256=previous,
    )


def _append(path: Path, publication_id: str, published_at: str) -> dict[str, object]:
    return append_public_forecast_record(
        path,
        _forecasts(reverse=publication_id.endswith("2")),
        publication_id=publication_id,
        published_at_utc=published_at,
        decision_label="tuesday_lock",
        decision_at_utc=published_at,
        inputs_observed_through_utc=published_at,
        provenance=_provenance(),
    )


def test_canonical_hash_is_deterministic_and_append_builds_chain(tmp_path: Path) -> None:
    assert _build()["content_sha256"] == _build(_forecasts(reverse=True))["content_sha256"]
    path = tmp_path / "public_forecasts.jsonl"
    first = _append(path, "publication-1", "2026-09-08T16:00:00Z")
    first_bytes = path.read_bytes()
    second = _append(path, "publication-2", "2026-09-09T16:00:00Z")

    assert path.read_bytes().startswith(first_bytes)
    assert second["previous_record_sha256"] == first["content_sha256"]
    assert "signature" not in json.dumps(second).lower()
    verified = verify_public_forecast_archive(
        path, expected_head_sha256=str(second["content_sha256"])
    )
    assert verified.records == 2
    assert verified.forecasts == 4
    assert verified.head_sha256 == second["content_sha256"]


def test_strict_decision_and_prekickoff_provenance() -> None:
    with pytest.raises(DataContractError, match="observed after the decision"):
        build_public_forecast_record(
            _forecasts(),
            publication_id="bad-input-cutoff",
            published_at_utc="2026-09-08T16:00:00Z",
            decision_label="lock",
            decision_at_utc="2026-09-08T15:00:00Z",
            inputs_observed_through_utc="2026-09-08T15:01:00Z",
            provenance=_provenance(),
        )
    with pytest.raises(DataContractError, match="strictly pre-kickoff"):
        _build(published_at="2026-09-13T17:00:00Z")

    malformed = _forecasts()
    malformed.loc[0, "home_loss_probability"] = 0.40
    with pytest.raises(DataContractError, match="must sum to one"):
        _build(malformed)


def test_tampering_noncanonical_encoding_and_broken_links_fail_verification(
    tmp_path: Path,
) -> None:
    path = tmp_path / "public_forecasts.jsonl"
    first = _append(path, "publication-1", "2026-09-08T16:00:00Z")
    _append(path, "publication-2", "2026-09-09T16:00:00Z")
    original = path.read_bytes()

    records = [json.loads(line) for line in original.splitlines()]
    records[0]["forecasts"][0]["home_cover_probability"] = 0.99
    path.write_bytes(
        (
            "\n".join(
                json.dumps(record, sort_keys=True, separators=(",", ":")) for record in records
            )
            + "\n"
        ).encode()
    )
    with pytest.raises(DataContractError, match="content hash mismatch"):
        verify_public_forecast_archive(path)

    path.write_bytes((json.dumps(first) + "\n").encode())
    with pytest.raises(DataContractError, match="not canonical JSON"):
        verify_public_forecast_archive(path)

    forged_second = _build(
        publication_id="publication-2",
        published_at="2026-09-09T16:00:00Z",
        previous="0" * 64,
    )
    path.write_bytes(
        json.dumps(first, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
        + json.dumps(forged_second, sort_keys=True, separators=(",", ":")).encode()
        + b"\n"
    )
    with pytest.raises(DataContractError, match="breaks the previous-record chain"):
        verify_public_forecast_archive(path)

    valid_path = tmp_path / "valid.jsonl"
    valid = _append(valid_path, "publication-1", "2026-09-08T16:00:00Z")
    with pytest.raises(DataContractError, match="pinned hash"):
        verify_public_forecast_archive(
            valid_path,
            expected_head_sha256=(
                "0" * 64 if str(valid["content_sha256"]) != "0" * 64 else "1" * 64
            ),
        )


def test_verification_command_and_append_guards(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "public_forecasts.jsonl"
    _append(path, "publication-1", "2026-09-08T16:00:00Z")
    assert main(["verify", str(path)]) == 0
    assert '"status": "valid"' in capsys.readouterr().out

    with pytest.raises(DataContractError, match="already exists"):
        _append(path, "publication-1", "2026-09-09T16:00:00Z")
    with pytest.raises(DataContractError, match="timestamps must increase"):
        _append(path, "publication-2", "2026-09-07T16:00:00Z")

    assert main(["verify", str(tmp_path / "missing.jsonl")]) == 1
    assert '"status": "invalid"' in capsys.readouterr().err
