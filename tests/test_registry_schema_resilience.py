from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import pytest

from nfl_ats import cli, rotation
from nfl_ats.findings_registry import load_rotation_registry, load_weak_signal_registry
from nfl_ats.rotation import RegistryError as RotationRegistryError
from nfl_ats.weak_signals import (
    WEAK_SIGNAL_REGISTRY_VERSION,
    ImplausibleStandardErrorWarning,
    Registry,
    UnknownRegistryFieldWarning,
    WeakSignalError,
    load_registry,
    load_registry_permissive,
    record_signal,
    registry_from_payload,
    registry_from_payload_permissive,
    save_registry_preserving_quarantine,
    signal_from_payload,
)


def _signal(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "recorded_at": "2026-09-08",
        "description": "a small measured effect, held for the resilience test",
        "source": "tests/test_registry_schema_resilience.py",
        "effect": 0.42,
        "effect_units": "accuracy_points",
        "classification": "unresolved_below_power",
        "league": "nfl",
        "seasons": [2020, 2021],
        "standard_error": 0.90,
    }
    body.update(overrides)
    return body


def _payload(**signals: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": WEAK_SIGNAL_REGISTRY_VERSION,
        "notes": ["resilience-test ledger"],
        "signals": signals,
    }


def _write_registry(path: Path, payload: dict[str, Any]) -> Path:
    destination = path / "weak_signals.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return destination


def test_strict_default_still_raises_on_an_unrecognised_signal_field(tmp_path: Path) -> None:

    destination = _write_registry(
        tmp_path, _payload(holdout_slow_start_on_production=_signal(reviewer_notes="pending"))
    )

    with pytest.raises(WeakSignalError, match="unknown fields: reviewer_notes"):
        load_registry(destination)


def test_warn_mode_tolerates_the_unrecognised_field_and_still_loads_the_signal(
    tmp_path: Path,
) -> None:
    destination = _write_registry(
        tmp_path, _payload(holdout_slow_start_on_production=_signal(reviewer_notes="pending"))
    )

    with pytest.warns(UnknownRegistryFieldWarning, match="reviewer_notes"):
        registry = load_registry(destination, on_unknown_field="warn")

    assert set(registry.signals) == {"holdout_slow_start_on_production"}
    signal = registry.signals["holdout_slow_start_on_production"]
    assert signal.effect == pytest.approx(0.42)
    assert signal.effect_units == "accuracy_points"
    assert signal.classification == "unresolved_below_power"


def test_registry_from_payload_warn_mode_tolerates_a_top_level_unknown_field() -> None:
    payload = {**_payload(alpha=_signal()), "future_top_level_field": True}

    with pytest.raises(WeakSignalError, match="unknown top-level fields"):
        registry_from_payload(payload)

    with pytest.warns(UnknownRegistryFieldWarning, match="future_top_level_field"):
        registry = registry_from_payload(payload, on_unknown_field="warn")
    assert set(registry.signals) == {"alpha"}


def test_correction_entry_unknown_field_is_tolerated_only_in_warn_mode() -> None:
    entry = {
        "at": "2026-09-08T18:00:00+00:00",
        "field": "probability_positive",
        "from": 0.0,
        "to": 0.5,
        "reason": "every resample was an exact tie; the old convention charged the zero atom",
        "approved_by": "a field this build does not recognise yet",
    }
    payload = _signal(
        effect=0.0,
        interval=[0.0, 0.0],
        probability_positive=0.5,
        corrections=[entry],
    )

    with pytest.raises(WeakSignalError, match="unknown fields: approved_by"):
        signal_from_payload("noop", payload)

    with pytest.warns(UnknownRegistryFieldWarning, match="approved_by"):
        signal = signal_from_payload("noop", payload, on_unknown_field="warn")
    assert signal.probability_positive == pytest.approx(0.5)


def test_publish_board_registry_reader_survives_a_schema_addition(tmp_path: Path) -> None:

    _write_registry(
        tmp_path,
        _payload(
            holdout_slow_start_on_production=_signal(
                reviewer_notes="pending re-measurement",
            )
        ),
    )

    registry = load_weak_signal_registry(registry_root=tmp_path)

    assert isinstance(registry, Registry)
    assert set(registry.signals) == {"holdout_slow_start_on_production"}


def test_publish_board_registry_reader_still_raises_on_real_corruption(tmp_path: Path) -> None:

    _write_registry(
        tmp_path,
        _payload(broken=_signal(classification="not_a_real_classification")),
    )

    with pytest.raises(WeakSignalError, match="unknown classification"):
        load_weak_signal_registry(registry_root=tmp_path)


def test_missing_registry_file_still_loads_as_empty_through_the_site_reader(
    tmp_path: Path,
) -> None:

    registry = load_weak_signal_registry(registry_root=tmp_path)
    assert registry.signals == {}


def _rotation_window(**overrides: Any) -> dict[str, Any]:
    window: dict[str, Any] = {
        "seasons": [2009, 2011],
        "state": "assigned",
        "assigned_at": "2026-09-08",
        "spent_at": None,
        "artifact": None,
        "verdict": None,
        "probability_positive": None,
        "effect": None,
        "effect_units": None,
        "interval": None,
        "standard_error": None,
        "sample_blocks": None,
        "notes": "",
    }
    window.update(overrides)
    return window


def _rotation_family(**overrides: Any) -> dict[str, Any]:
    family: dict[str, Any] = {
        "declared_at": "2026-09-08",
        "description": "test family, held for the resilience test",
        "grade": "nflverse_spread",
        "status": "open",
        "inherits": [],
        "acknowledges_mined_2018_2025": False,
        "windows": [],
    }
    family.update(overrides)
    return family


def _rotation_payload(**families: dict[str, Any]) -> dict[str, Any]:
    return {"version": 1, "notes": ["resilience-test ledger"], "families": families}


def _write_rotation_registry(path: Path, payload: dict[str, Any]) -> Path:
    destination = path / rotation.ROTATION_REGISTRY_FILENAME
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return destination


def test_rotation_strict_default_still_raises_on_an_unrecognised_family_field() -> None:

    payload = _rotation_payload(alpha=_rotation_family(reviewer_notes="pending"))

    with pytest.raises(RotationRegistryError, match="unknown fields: \\['reviewer_notes'\\]"):
        rotation.registry_from_payload(payload)


def test_rotation_warn_mode_tolerates_the_unrecognised_field_and_still_loads_the_family() -> None:
    payload = _rotation_payload(alpha=_rotation_family(reviewer_notes="pending"))

    with pytest.warns(UnknownRegistryFieldWarning, match="reviewer_notes"):
        registry = rotation.registry_from_payload(payload, on_unknown_field="warn")

    assert set(registry.families) == {"alpha"}
    assert registry.families["alpha"].grade == "nflverse_spread"


def test_rotation_warn_mode_tolerates_an_unrecognised_field_on_a_nested_window() -> None:

    window = _rotation_window(campaign_id="future-metadata")
    payload = _rotation_payload(alpha=_rotation_family(windows=[window]))

    with pytest.raises(RotationRegistryError, match="unknown window fields"):
        rotation.registry_from_payload(payload)

    with pytest.warns(UnknownRegistryFieldWarning, match="campaign_id"):
        registry = rotation.registry_from_payload(payload, on_unknown_field="warn")
    assert registry.families["alpha"].windows[0].seasons == (2009, 2011)


def test_rotation_publish_board_registry_reader_survives_a_schema_addition(
    tmp_path: Path,
) -> None:

    _write_rotation_registry(
        tmp_path,
        _rotation_payload(alpha=_rotation_family(reviewer_notes="pending re-measurement")),
    )

    registry = load_rotation_registry(registry_root=tmp_path)

    assert isinstance(registry, rotation.Registry)
    assert set(registry.families) == {"alpha"}


def test_rotation_publish_board_registry_reader_still_raises_on_inadmissible_closing_ground(
    tmp_path: Path,
) -> None:

    bad_window = _rotation_window(
        state="spent",
        spent_at="2026-09-08",
        artifact="docs/alpha.md",
        verdict="closed_negative",
        probability_positive=0.04,
        closing_ground="vibes",
    )
    _write_rotation_registry(
        tmp_path,
        _rotation_payload(alpha=_rotation_family(status="closed_negative", windows=[bad_window])),
    )

    with pytest.raises(RotationRegistryError, match="unknown closing_ground"):
        load_rotation_registry(registry_root=tmp_path)


def test_rotation_missing_registry_file_still_loads_as_empty_through_the_site_reader(
    tmp_path: Path,
) -> None:

    registry = load_rotation_registry(registry_root=tmp_path)
    assert registry.families == {}


def _record_args(name: str, *, replace: bool = False, **extra: str) -> list[str]:
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
        "0.42",
        "--effect-units",
        "accuracy_points",
        "--classification",
        "unresolved_below_power",
        "--league",
        "nfl",
        "--season-start",
        "2020",
        "--season-end",
        "2021",
        "--standard-error",
        "0.9",
    ]
    for flag, value in extra.items():
        args += [f"--{flag.replace('_', '-')}", value]
    if replace:
        args.append("--replace")
    return args


def _write_weak_signals_registry_dir(tmp_path: Path, payload: dict[str, Any]) -> Path:
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir(exist_ok=True)
    (registry_dir / "weak_signals.json").write_text(json.dumps(payload), encoding="utf-8")
    return registry_dir


def test_record_replace_repairs_an_entry_that_currently_fails_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    registry_dir = _write_weak_signals_registry_dir(
        tmp_path,
        _payload(broken=_signal(standard_error=0.0), healthy=_signal(effect=0.1)),
    )
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(registry_dir))
    registry_path = registry_dir / "weak_signals.json"

    with pytest.raises(WeakSignalError, match="non-positive standard_error"):
        load_registry(registry_path)

    with pytest.raises(SystemExit):
        cli.main(_record_args("broken", standard_error="1.2"))
    with pytest.raises(WeakSignalError, match="non-positive standard_error"):
        load_registry(registry_path)

    assert cli.main(_record_args("broken", standard_error="1.2", replace=True)) == 0

    repaired = load_registry(registry_path)
    assert set(repaired.signals) == {"broken", "healthy"}
    assert repaired.signals["broken"].standard_error == pytest.approx(1.2)
    assert repaired.signals["healthy"].effect == pytest.approx(0.1)


def test_record_replace_still_rejects_a_non_positive_standard_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    registry_dir = _write_weak_signals_registry_dir(
        tmp_path, _payload(broken=_signal(standard_error=0.0))
    )
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(registry_dir))
    registry_path = registry_dir / "weak_signals.json"

    with pytest.raises(SystemExit):
        cli.main(_record_args("broken", standard_error="0.0", replace=True))

    with pytest.raises(WeakSignalError, match="non-positive standard_error"):
        load_registry(registry_path)


def test_record_replace_still_rejects_an_inverted_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_dir = _write_weak_signals_registry_dir(
        tmp_path, _payload(broken=_signal(standard_error=0.0))
    )
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(registry_dir))

    with pytest.raises(SystemExit):
        cli.main(
            _record_args(
                "broken",
                replace=True,
                interval_low="2.0",
                interval_high="-1.0",
            )
        )


def test_record_replace_still_rejects_an_inadmissible_closing_ground(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    registry_dir = _write_weak_signals_registry_dir(
        tmp_path, _payload(broken=_signal(standard_error=0.0))
    )
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(registry_dir))

    with pytest.raises(SystemExit):
        cli.main(
            _record_args(
                "broken",
                replace=True,
                classification="refuted_mechanism",
                classification_evidence="the interval merely crosses zero",
            )
        )


def test_load_registry_permissive_quarantines_only_the_invalid_entry() -> None:

    payload = _payload(broken=_signal(standard_error=0.0), healthy=_signal(effect=0.1))

    registry, quarantined = registry_from_payload_permissive(payload)
    assert set(registry.signals) == {"healthy"}
    assert set(quarantined) == {"broken"}
    assert "non-positive standard_error" in quarantined["broken"].error
    assert quarantined["broken"].payload["standard_error"] == 0.0


def test_load_registry_permissive_reads_from_disk_and_save_preserves_untouched_rows(
    tmp_path: Path,
) -> None:
    destination = _write_registry(
        tmp_path, _payload(broken=_signal(standard_error=0.0), healthy=_signal(effect=0.1))
    )

    registry, quarantined = load_registry_permissive(destination)
    assert set(registry.signals) == {"healthy"}
    assert set(quarantined) == {"broken"}

    save_registry_preserving_quarantine(registry, quarantined, destination)
    on_disk = json.loads(destination.read_text(encoding="utf-8"))
    assert set(on_disk["signals"]) == {"broken", "healthy"}
    assert on_disk["signals"]["broken"]["standard_error"] == 0.0

    reloaded, still_quarantined = load_registry_permissive(destination)
    assert set(reloaded.signals) == {"healthy"}
    assert set(still_quarantined) == {"broken"}


def test_save_registry_preserving_quarantine_drops_the_now_repaired_entry(
    tmp_path: Path,
) -> None:

    destination = _write_registry(tmp_path, _payload(broken=_signal(standard_error=0.0)))
    registry, quarantined = load_registry_permissive(destination)
    assert set(quarantined) == {"broken"}

    fixed = signal_from_payload("broken", _signal(standard_error=1.2))
    registry = record_signal(registry, fixed, replace=True)
    save_registry_preserving_quarantine(registry, quarantined, destination)

    reloaded = load_registry(destination)
    assert set(reloaded.signals) == {"broken"}
    assert reloaded.signals["broken"].standard_error == pytest.approx(1.2)


def test_record_signal_warns_and_widens_a_standard_error_narrower_than_the_pool_supports() -> None:

    registry = Registry(version=WEAK_SIGNAL_REGISTRY_VERSION, notes=(), signals={})
    for name, games, se in (
        ("baseline_a", 500, 0.6),
        ("baseline_b", 1000, 0.45),
        ("baseline_c", 2000, 0.30),
    ):
        baseline_signal = signal_from_payload(
            name, _signal(effect=0.2, standard_error=se, sample_games=games)
        )
        registry = record_signal(registry, baseline_signal)

    narrow = signal_from_payload(
        "degenerate_cell",
        _signal(effect=100.0, standard_error=0.5, sample_games=15),
    )
    with pytest.warns(ImplausibleStandardErrorWarning, match="degenerate_cell"):
        registry = record_signal(registry, narrow)

    stored = registry.signals["degenerate_cell"]
    assert stored.standard_error is not None
    assert stored.standard_error > 0.5
    assert stored.standard_error == pytest.approx(3.2176, rel=1e-3)
    assert stored.effect == pytest.approx(100.0)


def test_record_signal_does_not_floor_when_the_pool_is_too_thin_to_say() -> None:

    registry = Registry(version=WEAK_SIGNAL_REGISTRY_VERSION, notes=(), signals={})
    only_other = signal_from_payload(
        "only_other", _signal(effect=0.2, standard_error=0.6, sample_games=500)
    )
    registry = record_signal(registry, only_other)

    narrow = signal_from_payload(
        "degenerate_cell", _signal(effect=100.0, standard_error=0.001, sample_games=15)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ImplausibleStandardErrorWarning)
        registry = record_signal(registry, narrow)

    assert registry.signals["degenerate_cell"].standard_error == pytest.approx(0.001)
