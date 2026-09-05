from dataclasses import replace

import pytest

from nfl_ats import cli
from nfl_ats import weak_signals as ws


def signal(name="valid", **kwargs):
    return ws.WeakSignal(
        name=name,
        recorded_at="2026-09-05",
        description="pregame screen",
        source="fixture",
        effect=0.1,
        effect_units="accuracy_points",
        classification="unresolved_below_power",
        league="nfl",
        seasons=(2020, 2025),
        standard_error=0.3,
        **kwargs,
    )


def registry():
    return ws.Registry(
        version=1,
        notes=(),
        signals={"valid": signal(), "leaked": replace(signal("leaked"), effect=90.0)},
    )


def test_invalidated_post_cutoff_measurement_cannot_leak_into_pool_or_signs(tmp_path):
    original = registry()
    updated = ws.invalidate_signal(
        original,
        name="leaked",
        reason="Inputs were after the prediction cutoff",
        superseded_by="valid",
    )
    invalid = updated.signals["leaked"]
    assert invalid.effect == original.signals["leaked"].effect
    assert invalid.standard_error == original.signals["leaked"].standard_error
    assert invalid.recorded_at == original.signals["leaked"].recorded_at
    assert invalid.classification == "unresolved_below_power"
    assert invalid.closing_ground is None
    assert "Inputs were after" in invalid.notes
    assert (
        ws.invalidate_signal(
            updated, name="leaked", reason=invalid.invalidated_reason, superseded_by="valid"
        )
        == updated
    )
    path = tmp_path / "weak_signals.json"
    ws.save_registry(updated, path)
    assert ws.load_registry(path) == updated
    report = ws.combination_report(updated, league="nfl", effect_units="accuracy_points")
    assert report["eligible"] == ["valid"]
    assert report["excluded_invalidated"] == 1
    assert report["pooled_by_unit"]["accuracy_points"]["pooled_effect"] == pytest.approx(0.1)
    entries = list(updated.signals.values())
    assert ws.sign_test(entries)["signals"] == 1
    assert ws.sign_test(entries)["excluded_invalidated"] == 1
    assert ws.pooled_effect(entries)["pooled_effect"] == pytest.approx(0.1)
    assert ws.pooled_effect(entries)["excluded_invalidated"] == 1
    assert ws.pooled_effect([invalid])["signals"] == 0
    assert ws.family_overlap_warnings(entries)["families"] == 1
    assert ws.overlap_warnings(entries) == []
    assert ws.combination_report(updated, league="cfb")["excluded_invalidated"] == 0
    with pytest.raises(ws.WeakSignalError, match="Keep invalidated history"):
        ws.record_signal(updated, signal("leaked"), replace=True)


@pytest.mark.parametrize("classification", ws.TERMINAL_CLASSIFICATIONS)
def test_invalidation_never_accepts_terminal_classification(classification):
    original = registry()
    original = replace(
        original, signals={"valid": replace(signal(), classification=classification)}
    )
    with pytest.raises(ws.WeakSignalError, match="terminal classification"):
        ws.invalidate_signal(original, name="valid", reason="Bad data")
    payload = ws.registry_to_payload(registry())["signals"]["valid"]
    payload.update(
        status="invalidated",
        invalidated_reason="Bad data",
        superseded_by=None,
        classification=classification,
    )
    with pytest.raises(ws.WeakSignalError, match="terminal classification"):
        ws.signal_from_payload("valid", payload)


@pytest.mark.parametrize(
    "metadata",
    [
        {"status": "invalidated", "superseded_by": None},
        {"status": "invalidated", "invalidated_reason": " ", "superseded_by": None},
        {"status": "invalidated", "invalidated_reason": "Bad data"},
        {"status": "active", "invalidated_reason": "Bad data"},
        {"status": "unknown"},
    ],
)
def test_invalid_invalidation_metadata_rejected(metadata):
    payload = ws.registry_to_payload(registry())["signals"]["valid"]
    with pytest.raises(ws.WeakSignalError):
        ws.signal_from_payload("valid", {**payload, **metadata})


@pytest.mark.parametrize("replacement", ["missing", "valid", " "])
def test_invalid_replacement_rejected(replacement):
    with pytest.raises(ws.WeakSignalError):
        ws.invalidate_signal(registry(), name="valid", reason="Bad data", superseded_by=replacement)


def test_cli_invalidate_and_listing(tmp_path, monkeypatch, capsys):
    import json

    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path))
    ws.save_registry(registry(), tmp_path / "weak_signals.json")
    assert (
        cli.main(
            ["weak-signals", "invalidate", "--name", "leaked", "--reason", "Post-cutoff input"]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "invalidated"
    for command in ("list", "status"):
        assert cli.main(["weak-signals", command]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["excluded_invalidated"] == 1
        assert [s["name"] for s in result["signals"]] == ["valid"]
        assert cli.main(["weak-signals", command, "--include-invalidated"]) == 0
        result = json.loads(capsys.readouterr().out)
        assert result["excluded_invalidated"] == 0
        assert len(result["signals"]) == 2
    assert (
        cli.main(["weak-signals", "pool", "--league", "nfl", "--effect-units", "accuracy_points"])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["excluded_invalidated"] == 1
