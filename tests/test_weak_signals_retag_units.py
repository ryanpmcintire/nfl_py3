"""CLI coverage for ``nfl-ats weak-signals retag-units`` (WP16).

The library-level behaviour (new units, sign convention, pool robustness on
empty/one-entry buckets, the ``retag_effect_units`` helper itself) is covered
in ``tests/test_weak_signals.py``. This file covers the CLI wiring: the new
``--effect-units`` choices reaching ``record``, and the ``retag-units``
subcommand end to end -- registry on disk in, registry on disk out, nothing
but the unit and an audit note changed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nfl_ats import cli


def _record_args(name: str, **extra: str) -> list[str]:
    args = [
        "weak-signals",
        "record",
        "--name",
        name,
        "--description",
        "a technical description of the measurement",
        "--source",
        "docs/example.md",
        "--effect",
        "0.25",
        "--effect-units",
        "accuracy_points",
        "--classification",
        "unresolved_below_power",
        "--league",
        "nfl",
        "--season-start",
        "2020",
        "--season-end",
        "2024",
    ]
    for flag, value in extra.items():
        args += [f"--{flag.replace('_', '-')}", value]
    return args


def _retag_args(name: str, *, effect_units: str, reason: str) -> list[str]:
    return [
        "weak-signals",
        "retag-units",
        "--name",
        name,
        "--effect-units",
        effect_units,
        "--reason",
        reason,
    ]


def _registry_payload(tmp_path: Path) -> dict[str, object]:
    registry_path = tmp_path / "registry" / "weak_signals.json"
    return json.loads(registry_path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "unit", ["correlation", "mae_improvement", "brier_improvement", "log_loss_improvement"]
)
def test_record_accepts_each_new_effect_unit(
    unit: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))
    name = f"cli_new_unit_{unit}"
    assert cli.main(_record_args(name, effect_units=unit)) == 0
    stored = _registry_payload(tmp_path)
    assert stored["signals"][name]["effect_units"] == unit


def test_record_still_rejects_an_unknown_effect_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))
    with pytest.raises(SystemExit):
        cli.main(_record_args("cli_bad_unit_demo", effect_units="vibes"))


def test_retag_units_changes_only_the_unit_and_appends_an_audit_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))
    name = "cli_retag_demo"
    assert (
        cli.main(
            _record_args(
                name,
                effect_units="mae",
                effect="0.00082",
                notes="positive already means the candidate's MAE was lower (better)",
            )
        )
        == 0
    )
    capsys.readouterr()  # discard the record command's own JSON/warnings

    before = _registry_payload(tmp_path)["signals"][name]

    assert (
        cli.main(
            _retag_args(
                name,
                effect_units="mae_improvement",
                reason="was mae with the sign convention explained only in notes",
            )
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["retagged"] == name
    assert out["previous_effect_units"] == "mae"
    assert out["effect_units"] == "mae_improvement"
    assert "mae_improvement" in out["notes"]

    after = _registry_payload(tmp_path)["signals"][name]
    assert after["effect_units"] == "mae_improvement"
    assert before["notes"] in after["notes"]
    assert "sign convention explained only in notes" in after["notes"]

    # Everything else on the entry is byte-identical to before the retag.
    unchanged_fields = set(before) - {"effect_units", "notes"}
    for field in unchanged_fields:
        assert after[field] == before[field], field


def test_retag_units_rejects_an_unknown_unit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))
    assert cli.main(_record_args("cli_retag_bad_unit")) == 0
    with pytest.raises(SystemExit):
        cli.main(_retag_args("cli_retag_bad_unit", effect_units="vibes", reason="typo"))


def test_retag_units_rejects_a_missing_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))
    # An empty registry (no prior record call) still resolves to a valid,
    # empty ledger -- the retag itself is what must refuse the unknown name.
    with pytest.raises(SystemExit):
        cli.main(_retag_args("does_not_exist", effect_units="correlation", reason="n/a"))


def test_pool_on_a_new_unit_handles_an_empty_bucket_from_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(tmp_path / "registry"))
    assert cli.main(_record_args("cli_pool_seed", effect_units="accuracy_points")) == 0
    capsys.readouterr()
    assert cli.main(["weak-signals", "pool", "--effect-units", "correlation"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["eligible"] == []
    assert out["pooled_by_unit"] == {}
