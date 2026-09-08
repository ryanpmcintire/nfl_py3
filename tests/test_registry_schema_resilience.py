"""Regression coverage for the 2026-09-08 ``publish-board`` outage.

A session added a ``corrections`` field to entries in
``registry/weak_signals.json`` and the matching entry to
``weak_signals._SIGNAL_FIELDS`` in the SAME commit, but the scheduled
``lineups_tue`` job ran between the two writes and read the data file one
field ahead of the code that had to parse it. ``signal_from_payload`` raised
``WeakSignalError: Signal '...' has unknown fields: corrections``, uncaught,
all the way up through ``weekly-run``'s ``publish-board`` step -- aborting
the entire scheduled run and leaving the public site unbuilt.

Both writes are committed now (that specific field is recognised), but the
CLASS of bug is not fixed by that alone: the next additive field lands the
same way (data ahead of code, or a mid-session gap a scheduled job races).
This file pins the fix -- ``weak_signals.load_registry``/``registry_from_payload``/
``signal_from_payload`` accept ``on_unknown_field="warn"`` to tolerate a
schema addition instead of raising, and
``findings_registry.load_weak_signal_registry`` (the traced, single choke
point every public-site reader of this registry goes through --
``board_site_content.py``'s findings and signal-ledger-summary loaders,
``public_board.py``'s findings and signal-ledger pages) now reads with that
flag -- and pins the boundary: every OTHER validation error (missing
required field, bad classification, incoherent effect/interval, inadmissible
closing ground) must still raise, in EITHER mode, because those are real
data corruption, not additive drift.

The SAME failure shape was traced (2026-09-08, same session) into
``rotation.py``: ``board_site_content._load_findings_content``'s "Research
this week" section -- reached from the live ``publish-board`` handler via
``board_site.build_site`` -> ``load_site_content`` -- calls
``findings_registry.load_rotation_registry``, which called
``rotation.load_registry`` unguarded, with the identical
``"has unknown fields"`` raises (``_no_rotation_record_from_payload``,
``_family_from_payload``, and, threaded the same way as
``weak_signals._validate_corrections``, the nested ``_window_from_payload``
and ``_leg_result_from_payload``). The second half of this file mirrors
every case above for the rotation registry, with one addition the
coordinator called for explicitly: a pin that the rotation registry's
closing-ground taxonomy (``rotation._validate_closing_ground`` and the
inline closing_ground checks in ``_window_from_payload`` -- release-blocking
per AGENTS.md's binding "an interval crossing zero is not grounds for
rejection" rule) is NOT weakened by any of this -- an inadmissible
``closing_ground`` still raises through the tolerant site-build reader
exactly as it did before.
"""

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


# ---------------------------------------------------------------------------
# 1. The exact shape of the outage: an unrecognised field on a signal entry.
# ---------------------------------------------------------------------------


def test_strict_default_still_raises_on_an_unrecognised_signal_field(tmp_path: Path) -> None:
    """Unchanged behaviour: the CLI's read path (``weak-signals status`` /
    ``pool`` / ``record``, via ``cli_commands/registry.py``, and every
    existing caller of ``load_registry``) must keep hearing about a typo'd
    field immediately -- this is what caught the real ``corrections`` gap in
    the first place, before it was reclassified as a recognised field."""

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


# ---------------------------------------------------------------------------
# 2. The actual regression pin: the public site's own entry point.
# ---------------------------------------------------------------------------


def test_publish_board_registry_reader_survives_a_schema_addition(tmp_path: Path) -> None:
    """Pins the fix at the traced choke point (``findings_registry.
    load_weak_signal_registry``), not merely at the low-level parser.

    Every reader on the live ``publish-board`` path --
    ``board_site_content._load_findings_content``,
    ``board_site_content._load_signal_ledger_summary``,
    ``public_board.render_findings_page``,
    ``public_board.render_signal_ledger_page`` -- calls this function rather
    than ``weak_signals.load_registry`` directly, so fixing it here is what
    actually stops a schema addition from aborting the scheduled
    ``weekly-run`` at its ``publish-board`` step, which is exactly how the
    2026-09-08 outage happened.
    """

    _write_registry(
        tmp_path,
        _payload(
            holdout_slow_start_on_production=_signal(
                # A field shaped exactly like the real incident, but under a
                # name this build has never heard of -- the next one of these,
                # not the one already fixed.
                reviewer_notes="pending re-measurement",
            )
        ),
    )

    registry = load_weak_signal_registry(registry_root=tmp_path)

    assert isinstance(registry, Registry)
    assert set(registry.signals) == {"holdout_slow_start_on_production"}


def test_publish_board_registry_reader_still_raises_on_real_corruption(tmp_path: Path) -> None:
    """The tolerant read path must not swallow an actual data problem: a
    build that would publish a signal with an inadmissible classification (or
    any other genuine corruption) should still refuse, exactly as before --
    only ADDITIVE schema drift gets tolerated."""

    _write_registry(
        tmp_path,
        _payload(broken=_signal(classification="not_a_real_classification")),
    )

    with pytest.raises(WeakSignalError, match="unknown classification"):
        load_weak_signal_registry(registry_root=tmp_path)


def test_missing_registry_file_still_loads_as_empty_through_the_site_reader(
    tmp_path: Path,
) -> None:
    """Unrelated to unknown fields, but pins that the tolerant path did not
    disturb the existing "no file yet" contract (a fresh checkout with no
    registry history)."""

    registry = load_weak_signal_registry(registry_root=tmp_path)
    assert registry.signals == {}


# ---------------------------------------------------------------------------
# 3. The rotation registry: the same failure shape, traced and fixed the
#    same session, after the coordinator authorised editing rotation.py.
# ---------------------------------------------------------------------------


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
    """Unchanged behaviour: the ``rotation`` CLI (``cli_commands/registry.py``)
    calls ``rotation.load_registry``/``registry_from_payload`` directly and
    must keep hearing about a typo'd field immediately."""

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
    """The same tolerance must reach NESTED unknown fields, not just the
    top-level ``Family`` object -- mirrors the weak-signal registry's
    ``corrections``-entry case: an unrecognised field on a ``Window`` is the
    same additive-drift shape one level down."""

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
    """Pins the fix at the traced choke point
    (``findings_registry.load_rotation_registry``), reached from the live
    ``publish-board`` handler via ``board_site.build_site`` ->
    ``load_site_content`` -> ``board_site_content._load_findings_content``'s
    "Research this week" section -- not merely at the low-level parser."""

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
    """The tolerant read path must not weaken the closing-ground taxonomy.

    Repo memory, verbatim: "Directives now enforced in code -- never weaken
    the validators." A ``closed_negative`` verdict naming an inadmissible
    ``closing_ground`` (here: not one of AGENTS.md's admissible grounds) is
    genuine corruption -- release-blocking -- and must still abort the site
    build through the SAME reader that now tolerates an unrecognised field.
    """

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
    """Unrelated to unknown fields, but pins that the tolerant path did not
    disturb the existing "no file yet" contract, matching the weak-signal
    registry's own such pin above."""

    registry = load_rotation_registry(registry_root=tmp_path)
    assert registry.families == {}


# ---------------------------------------------------------------------------
# 4. The repair path: a single invalid registry entry must not block its
#    own repair. Coordinator-reported incident, 2026-09-08: a lane recorded
#    ``mod18_discrete_side_read_v1_ds_pcsmall_vs_s3_touched_standalone_2020_2025``
#    with ``standard_error: 0.0`` (a genuinely degenerate 15-game block
#    bootstrap: the leaked arm was right on all 15 games, the baseline wrong
#    on all 15, so every resample returned exactly 100.0). Consequence 2 of
#    that incident: ``nfl-ats weak-signals record --replace`` -- the one
#    sanctioned repair tool -- could not fix it, because the record command
#    loads and validates the WHOLE registry before writing, and the invalid
#    entry made that load fail. The coordinator had to hand-delete the key
#    with a throwaway script and re-record it -- exactly the hand-editing
#    AGENTS.md exists to prevent. This section pins the fix: a tolerant load
#    that quarantines an individually-invalid entry instead of refusing the
#    whole file, paired with a save that never silently drops an untouched
#    quarantined entry -- while the WRITE side stays exactly as strict as
#    before (unchanged validators, only reachable again).
# ---------------------------------------------------------------------------


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
    """The coordinator's exact incident, reproduced and fixed: a
    ``standard_error: 0.0`` entry sits in the registry alongside a healthy
    one; ``record --replace`` on the broken name repairs it in place through
    the CLI -- the sanctioned tool, no hand-editing."""

    registry_dir = _write_weak_signals_registry_dir(
        tmp_path,
        _payload(broken=_signal(standard_error=0.0), healthy=_signal(effect=0.1)),
    )
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(registry_dir))
    registry_path = registry_dir / "weak_signals.json"

    # The strict load a read-only command (status/pool) uses is refused, as
    # before -- pins that the incident's own failure mode still fires.
    with pytest.raises(WeakSignalError, match="non-positive standard_error"):
        load_registry(registry_path)

    # Without --replace, the CLI still refuses: a repair is a deliberate
    # act, not an accident triggered by recording under the same name.
    with pytest.raises(SystemExit):
        cli.main(_record_args("broken", standard_error="1.2"))
    # Nothing was written by the refused attempt.
    with pytest.raises(WeakSignalError, match="non-positive standard_error"):
        load_registry(registry_path)

    # `record --replace` -- the one sanctioned repair tool -- now works.
    assert cli.main(_record_args("broken", standard_error="1.2", replace=True)) == 0

    # The repaired registry loads strictly clean.
    repaired = load_registry(registry_path)
    assert set(repaired.signals) == {"broken", "healthy"}
    assert repaired.signals["broken"].standard_error == pytest.approx(1.2)
    # The untouched entry survived byte-for-byte, not silently dropped while
    # the load had to quarantine its way past the broken one.
    assert repaired.signals["healthy"].effect == pytest.approx(0.1)


def test_record_replace_still_rejects_a_non_positive_standard_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write side stays strict: a --replace that would store a
    non-positive standard_error still fails loudly, exactly as it does for a
    brand-new record. Nothing here weakens that validator -- the repair path
    only lets a WRITE reach the registry despite an existing invalid row; it
    never accepts an invalid VALUE."""

    registry_dir = _write_weak_signals_registry_dir(
        tmp_path, _payload(broken=_signal(standard_error=0.0))
    )
    monkeypatch.setenv("NFL_ATS_REGISTRY_DIR", str(registry_dir))
    registry_path = registry_dir / "weak_signals.json"

    with pytest.raises(SystemExit):
        cli.main(_record_args("broken", standard_error="0.0", replace=True))

    # The broken entry is exactly as it was -- the rejected write changed
    # nothing.
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
    """Repo memory, verbatim: "Directives now enforced in code -- never
    weaken the validators." A terminal classification with no admissible
    closing_ground is refused through the repair path exactly as it is on a
    fresh record."""

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
    """Library-level pin of the mechanism underneath the CLI test above."""

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

    # A write that touches neither name (a no-op save here) must still keep
    # the untouched-but-broken row in the file rather than dropping it.
    save_registry_preserving_quarantine(registry, quarantined, destination)
    on_disk = json.loads(destination.read_text(encoding="utf-8"))
    assert set(on_disk["signals"]) == {"broken", "healthy"}
    assert on_disk["signals"]["broken"]["standard_error"] == 0.0

    # And it round-trips through the SAME permissive reader.
    reloaded, still_quarantined = load_registry_permissive(destination)
    assert set(reloaded.signals) == {"healthy"}
    assert set(still_quarantined) == {"broken"}


def test_save_registry_preserving_quarantine_drops_the_now_repaired_entry(
    tmp_path: Path,
) -> None:
    """The one entry a caller actually repairs must end up validated and
    live in ``signals``, not duplicated into the quarantine leftovers."""

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
    """docs/weak_signal_pooling.md defect 4's floor -- a band narrower than
    its own sample size can support is a block-bootstrap artifact, not
    power, and is floored rather than trusted at POOL time -- applied at
    RECORD time too, so a future lane cannot store a zero-width (or simply
    implausibly narrow) band in the first place. Widening only, always with
    a warning: this is the exact shape of the incident that motivated the
    whole repair-path fix above, caught before it can be written at all."""

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
    assert stored.standard_error > 0.5  # offered value was widened, never narrowed
    assert stored.standard_error == pytest.approx(3.2176, rel=1e-3)
    # The measurement itself -- the effect -- is untouched by the widening.
    assert stored.effect == pytest.approx(100.0)


def test_record_signal_does_not_floor_when_the_pool_is_too_thin_to_say() -> None:
    """Fewer than three usable same-unit entries elsewhere in the registry:
    nothing to floor against, so the offered value is stored unchanged --
    matching :func:`_plausibility_curve`'s own minimum at POOL time."""

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
