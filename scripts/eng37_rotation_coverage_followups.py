"""ENG-37 (ROADMAP.md Phase 13): apply the two rotation-registry owner
decisions surfaced by ENG-27 on 2026-09-04.

(1) 54 CFB weak-signal families were given ``declared_for_coverage`` rotation
    stubs by ``nfl-ats rotation declare-coverage --apply`` even though the
    rotation registry governs NFL confirmation looks only (rule 8,
    docs/rotation_registry.md). Family declarations are append-only --
    ``rotation.py`` has no delete API by design (rule 1;
    ``declare_family``/``declare_coverage_stub`` both refuse an existing
    name rather than replacing it) -- so the 54 stub ``Family`` entries are
    KEPT (they still reserve their name, harmlessly). Each is instead given
    an explicit ``no_rotation_needed`` record with reason
    ``"cfb_out_of_scope"`` (``rotation.NO_ROTATION_FIXED_REASONS``, added
    alongside this script), keyed by the weak-signal family name
    (``Family.coverage_weak_signal_family``, not the rotation-family name,
    which can differ on a name-collision suffix), so a reader never mistakes
    one of these 54 for a stub still awaiting an NFL confirmation window.
    ``registry_explorer.coverage_plan`` now also routes every FUTURE CFB
    weak-signal family straight to a ``no_rotation_needed`` record with this
    same reason instead of reserving a new stub
    (``rotation.classify_no_rotation_reason`` checks ``league`` first).

(2) ``pbp_drive_bundle`` holds a CONTIGUOUS ``[2013, 2017]`` window (5
    seasons), assigned 2026-08-13 -- three weeks before
    ``rotation.validate_registry``'s ``window_width_out_of_range`` check
    existed (2026-09-04, ENG-27). This script appends a dated grandfather
    note to that window's own ``notes`` field, matching the new
    ``rotation.GRANDFATHERED_WIDTH_VIOLATIONS`` entry that
    ``validate_registry`` uses to downgrade this ONE window from error to
    warning, without weakening the check for anything assigned on or after
    ``rotation.VALIDATOR_INTRODUCED_AT``.

Uses only ``rotation.load_registry`` / ``rotation.save_registry`` and the
dataclasses they return -- never hand-edits the tracked JSON directly.
Idempotent in the sense that matters: ``record_no_rotation_needed`` is
append-only and raises if a weak-signal family already has a record, so
re-running this script against an already-migrated registry fails loudly
(via ``rotation.RegistryError``) rather than silently duplicating or
re-appending the grandfather note.

Not a ``READ_ONLY_SCRIPT``: it writes ``registry/rotation_registry.json``
through ``rotation.save_registry`` (the project's own governed writer, which
validates before writing). It never touches ``registry/weak_signals.json``.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nfl_ats import rotation  # noqa: E402

TODAY = "2026-09-05"
GRANDFATHER_FAMILY = "pbp_drive_bundle"

GRANDFATHER_NOTE = (
    f"\n\nGRANDFATHERED {TODAY} (ROADMAP.md ENG-37): this window's width (5 "
    "seasons, [2013, 2017]) predates rotation.py's window_width_out_of_range "
    f"validator, added {rotation.VALIDATOR_INTRODUCED_AT} (ENG-27) -- "
    "assign_window's [2, 4]-season limit did not exist when this window was "
    "assigned 2026-08-13. rotation.validate_registry now reports this "
    "specific window as a warning, not an error, via "
    "rotation.GRANDFATHERED_WIDTH_VIOLATIONS; the 2-4 rule itself is NOT "
    "widened, and any window assigned on or after "
    f"{rotation.VALIDATOR_INTRODUCED_AT} still errors."
)


def _cfb_out_of_scope_notes(rotation_family_name: str) -> str:
    return (
        f"ENG-37 ({TODAY}): rotation family {rotation_family_name!r} was declared as a "
        "declared_for_coverage stub on 2026-09-04, before this scope decision. The "
        "stub Family entry is KEPT -- rotation-family declarations are append-only "
        "(docs/rotation_registry.md rule 1; there is no delete API) -- but this "
        "weak-signal family is out of the NFL-only rotation registry's scope (rule 8) "
        "and needs no NFL confirmation window."
    )


def main() -> int:
    path = rotation.default_registry_path()
    registry = rotation.load_registry(path)

    # --- (1) CFB declared_for_coverage stubs -> no_rotation_needed ---------
    cfb_stub_families = sorted(
        (
            family
            for family in registry.families.values()
            if family.status == rotation.COVERAGE_STUB_STATUS and family.coverage_league == "cfb"
        ),
        key=lambda f: f.name,
    )
    print(f"CFB declared_for_coverage stubs found: {len(cfb_stub_families)}")
    for family in cfb_stub_families:
        weak_signal_family = family.coverage_weak_signal_family
        assert weak_signal_family is not None, family.name
        registry = rotation.record_no_rotation_needed(
            registry,
            weak_signal_family,
            league=family.coverage_league,
            reason="cfb_out_of_scope",
            effect_units=family.coverage_effect_units,
            notes=_cfb_out_of_scope_notes(family.name),
        )
    print(f"no_rotation_needed records added: {len(cfb_stub_families)}")

    # --- (2) pbp_drive_bundle: grandfather the pre-validator width ---------
    pbp = registry.families[GRANDFATHER_FAMILY]
    if len(pbp.windows) != 1:
        raise SystemExit(
            f"expected exactly one window on {GRANDFATHER_FAMILY!r}, found {len(pbp.windows)}"
        )
    window = pbp.windows[0]
    if window.seasons != rotation.GRANDFATHERED_WIDTH_VIOLATIONS[GRANDFATHER_FAMILY]:
        raise SystemExit(
            f"{GRANDFATHER_FAMILY!r} window seasons {window.seasons} no longer match "
            f"rotation.GRANDFATHERED_WIDTH_VIOLATIONS; refusing to guess"
        )
    if "GRANDFATHERED" in window.notes:
        raise SystemExit(f"{GRANDFATHER_FAMILY!r} window already carries a grandfather note")
    updated_window = dataclasses.replace(window, notes=window.notes + GRANDFATHER_NOTE)
    updated_family = dataclasses.replace(pbp, windows=(updated_window,))
    families = dict(registry.families)
    families[GRANDFATHER_FAMILY] = updated_family
    registry = dataclasses.replace(registry, families=families)
    print(f"Grandfather note appended to {GRANDFATHER_FAMILY!r}'s window.")

    rotation.save_registry(registry, path)
    print(f"Saved {path}")

    issues = rotation.validate_registry(registry)
    errors = [issue for issue in issues if issue.severity == "error"]
    print(f"validate_registry: {len(issues)} issue(s), {len(errors)} error(s)")
    for issue in issues:
        print(f"  [{issue.severity}] {issue.code} {issue.family}: {issue.message}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
