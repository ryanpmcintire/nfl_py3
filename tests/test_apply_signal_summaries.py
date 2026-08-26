from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.apply_signal_summaries as apply_signal_summaries
from nfl_ats.weak_signals import (
    WEAK_SIGNAL_REGISTRY_VERSION,
    Registry,
    load_registry,
    save_registry,
    signal_from_payload,
)


def _signal_payload(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "recorded_at": "2026-08-17",
        "description": "a small measured effect, raw technical wording",
        "source": "docs/example.md",
        "effect": 0.10,
        "effect_units": "accuracy_points",
        "classification": "unresolved_below_power",
        "league": "nfl",
        "seasons": [2009, 2017],
        "standard_error": 0.30,
    }
    body.update(overrides)
    return body


def _write_registry(path: Path, **signals: dict[str, Any]) -> Registry:
    registry = Registry(
        version=WEAK_SIGNAL_REGISTRY_VERSION,
        notes=(),
        signals={name: signal_from_payload(name, body) for name, body in signals.items()},
    )
    save_registry(registry, path)
    return registry


def test_missing_summaries_file_reports_zero_applied_not_an_error(tmp_path: Path) -> None:
    registry_path = tmp_path / "weak_signals.json"
    _write_registry(registry_path, alpha=_signal_payload())

    report = apply_signal_summaries.apply_signal_summaries(
        tmp_path / "does_not_exist.json", registry_path
    )
    assert report["summaries_found"] is False
    assert report["applied"] == 0
    assert report["remaining_without_plain_summary_or_category"] == 1


def test_applies_matches_reports_unmatched_and_rejects_bad_category(tmp_path: Path) -> None:
    registry_path = tmp_path / "weak_signals.json"
    _write_registry(
        registry_path,
        alpha=_signal_payload(),
        beta=_signal_payload(description="beta's own raw description"),
    )
    summaries_path = tmp_path / "signal_summaries.json"
    summaries_path.write_text(
        json.dumps(
            {
                "alpha": {
                    "category": "onfield",
                    "plain_summary": "A short sentence a fan can read on its own.",
                },
                "beta": {"category": "not_a_real_category", "plain_summary": "Also short."},
                "ghost_signal_not_in_registry": {
                    "category": "onfield",
                    "plain_summary": "Never applied -- no matching registry row.",
                },
            }
        ),
        encoding="utf-8",
    )

    report = apply_signal_summaries.apply_signal_summaries(summaries_path, registry_path)

    assert report["summaries_found"] is True
    # alpha gets both fields; beta's plain_summary applies but its category is
    # rejected (invalid), so beta still changed (plain_summary alone) -> counted
    # in applied, and the bad category is reported separately.
    assert set(report["applied_names"]) == {"alpha", "beta"}
    assert report["unmatched_names"] == ["ghost_signal_not_in_registry"]
    assert len(report["rejected_invalid_category"]) == 1
    assert report["rejected_invalid_category"][0]["name"] == "beta"

    updated = load_registry(registry_path)
    assert updated.signals["alpha"].category == "onfield"
    assert updated.signals["alpha"].plain_summary == "A short sentence a fan can read on its own."
    # beta's category was rejected, so it stays unset even though plain_summary applied.
    assert updated.signals["beta"].category is None
    assert updated.signals["beta"].plain_summary == "Also short."


def test_never_alters_any_field_other_than_plain_summary_and_category(tmp_path: Path) -> None:
    registry_path = tmp_path / "weak_signals.json"
    original = _write_registry(
        registry_path,
        alpha=_signal_payload(effect=1.75, notes="a load-bearing note", sample_games=456),
    )
    summaries_path = tmp_path / "signal_summaries.json"
    summaries_path.write_text(
        json.dumps({"alpha": {"category": "health", "plain_summary": "Plain words."}}),
        encoding="utf-8",
    )

    apply_signal_summaries.apply_signal_summaries(summaries_path, registry_path)

    updated = load_registry(registry_path)
    before = original.signals["alpha"]
    after = updated.signals["alpha"]
    assert after.effect == before.effect
    assert after.notes == before.notes
    assert after.sample_games == before.sample_games
    assert after.description == before.description
    assert after.source == before.source
    assert after.interval == before.interval
    assert after.classification == before.classification
    # Only these two changed.
    assert before.category is None and after.category == "health"
    assert before.plain_summary is None and after.plain_summary == "Plain words."


def test_is_idempotent(tmp_path: Path) -> None:
    registry_path = tmp_path / "weak_signals.json"
    _write_registry(registry_path, alpha=_signal_payload())
    summaries_path = tmp_path / "signal_summaries.json"
    summaries_path.write_text(
        json.dumps({"alpha": {"category": "onfield", "plain_summary": "Plain words."}}),
        encoding="utf-8",
    )

    first = apply_signal_summaries.apply_signal_summaries(summaries_path, registry_path)
    assert first["applied"] == 1

    second = apply_signal_summaries.apply_signal_summaries(summaries_path, registry_path)
    assert second["applied"] == 0
    assert second["already_up_to_date"] == 1


def test_dry_run_reports_without_writing(tmp_path: Path) -> None:
    registry_path = tmp_path / "weak_signals.json"
    _write_registry(registry_path, alpha=_signal_payload())
    summaries_path = tmp_path / "signal_summaries.json"
    summaries_path.write_text(
        json.dumps({"alpha": {"category": "onfield", "plain_summary": "Plain words."}}),
        encoding="utf-8",
    )

    report = apply_signal_summaries.apply_signal_summaries(
        summaries_path, registry_path, dry_run=True
    )
    assert report["applied"] == 1

    unchanged = load_registry(registry_path)
    assert unchanged.signals["alpha"].category is None
    assert unchanged.signals["alpha"].plain_summary is None


def test_accepts_the_produced_envelope_shape(tmp_path: Path) -> None:
    """The real file this script consumes wraps the mapping in a
    ``{"generated_at_utc": ..., "source_registry_sha256": ..., "summaries": {...}}``
    envelope; the bare mapping must also still work."""

    registry_path = tmp_path / "weak_signals.json"
    _write_registry(registry_path, alpha=_signal_payload())
    summaries_path = tmp_path / "signal_summaries.json"
    summaries_path.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-08-26T00:00:00Z",
                "source_registry_sha256": "deadbeef",
                "summaries": {"alpha": {"category": "onfield", "plain_summary": "Plain words."}},
            }
        ),
        encoding="utf-8",
    )

    report = apply_signal_summaries.apply_signal_summaries(summaries_path, registry_path)
    assert report["applied"] == 1
    updated = load_registry(registry_path)
    assert updated.signals["alpha"].category == "onfield"


def test_main_prints_json_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    registry_path = tmp_path / "weak_signals.json"
    _write_registry(registry_path, alpha=_signal_payload())
    summaries_path = tmp_path / "signal_summaries.json"
    summaries_path.write_text(
        json.dumps({"alpha": {"category": "onfield", "plain_summary": "Plain words."}}),
        encoding="utf-8",
    )

    sys.argv = [
        "apply_signal_summaries.py",
        "--summaries",
        str(summaries_path),
        "--registry",
        str(registry_path),
    ]
    apply_signal_summaries.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["applied"] == 1
