"""RWB-09 experiment-provenance registry: enforcement and backfill tests.

Two concerns live here, distinct from ``tests/test_provenance.py``'s
round-trip/schema tests on the ``write_experiment_artifact()`` helper itself:

1. **Enforcement** -- a static check that every ``scripts/*.py`` file writing
   JSON into ``artifacts/`` goes through a sanctioned provenance helper, so
   the "automatic, not a discipline" property survives past this session.
2. **Backfill** -- the one-time mechanical lift from existing
   ``artifacts/**/{metadata,run}.json`` files into ``registry/experiments/``
   (``scripts/backfill_experiment_registry.py``) produces rows that actually
   parse, and correctly refuses to invent a row when nothing in an artifact
   carries recoverable provenance.

ENG-29 (2026-09-04) replaced the enforcement mechanism. Originally, EVERY
script that legitimately did not call ``write_experiment_artifact`` needed a
hand-added entry (and reason comment) in a single, only-ever-growing
``_ALLOWLISTED_UNSTAMPED_SCRIPTS`` frozenset here -- including scripts that
are trivially, mechanically read-only, whose "reason" was really just "this
scanner cannot yet see that it never writes." ``nfl_ats.script_contracts``
now answers that mechanically: ``scan_script()`` parses a script with
:mod:`ast` (never imports/executes it) and reports whether it declares
``READ_ONLY_SCRIPT = True`` and, if so, whether that claim actually holds (no
write site resolves into ``artifacts/`` or ``registry/``). A script can now
self-certify beside its own code instead of via a comment in this file.

That mechanism only covers scripts that are ACTUALLY read-only. Auditing the
88-script legacy allowlist against the new scanner (2026-09-04) found three
honest buckets, not two:

- **36 scripts genuinely are read-only** (no write site resolves into
  ``artifacts/`` or ``registry/`` -- either zero write calls at all, or every
  write destination is a caller-supplied ``--output``/``--out`` path with no
  governed default). These now carry ``READ_ONLY_SCRIPT = True`` in the
  script itself (three of them also need a small ``READ_ONLY_EXCEPTIONS``
  dict for a destination the scanner cannot statically resolve, e.g.
  ``backup_data.py``'s mirror-drive path) and have been REMOVED from the
  allowlist below.
- **2 scripts** (``snapshot_diff.py``, ``prospective_scorecard.py``) write
  real artifacts but are explicitly not experiments, and
  ``write_experiment_artifact()`` always creates a
  ``registry/experiments/<slug>/<stamp>.json`` row -- which would
  misrepresent them. Both were wired to the new
  ``nfl_ats.provenance.write_stamped_artifact()`` (stamps code
  revision/dirty state onto the payload, writes no registry row) and have
  also been REMOVED from the allowlist below.
- **The remaining 50 scripts were genuine non-experiment writers that were
  NOT read-only**: measure-only screens/evals/audits whose result tables are
  the deliverable (written under a governed ``artifacts/...`` default, not a
  caller-supplied path), plus a handful of scripts that delegate their write
  to an imported function this single-file scanner cannot see (six weekly
  ledger recorders, one lock-day wrapper). Declaring these ``READ_ONLY_SCRIPT
  = True`` would be false, so ENG-38 (2026-09-04) wired every one of them
  to a sanctioned helper instead: ``write_stamped_artifact()``/
  ``stamp_sidecar()`` at each script's own write site, or -- for the seven
  delegators -- at the write site inside the shared library function they
  call, with that function then listed in
  ``nfl_ats.script_contracts.STAMPED_LIBRARY_WRITERS`` so ``scan_script()``
  can see through the delegation (verified by
  ``tests/test_script_contracts.py::test_stamped_library_writers_really_stamp``).
  The formerly-50-entry ``_NON_EXPERIMENT_WRITER_SCRIPTS`` allowlist below is
  now empty and has been deleted: every ``scripts/*.py`` file resolves via a
  helper call, a verified ``READ_ONLY_SCRIPT`` declaration, or a delegated
  stamped-library-writer call.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from nfl_ats.provenance import (
    ExperimentRecordError,
    default_experiment_registry_root,
    experiment_command_slug,
    experiment_record_from_payload,
    load_experiment_record,
    save_experiment_record,
    verify_experiment_links,
)
from nfl_ats.script_contracts import scan_script

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"

# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------

# Trigger condition, UNCHANGED from the pre-ENG-29 grep-based gate on purpose:
# widening it (e.g. to also trigger on any mention of "registry", or on
# to_csv/to_parquet/mkdir-style writes) would sweep dozens of unrelated,
# never-audited scripts into this gate's scope as an accidental side effect
# of a ticket about HOW compliance is declared, not WHICH scripts must
# comply. ENG-29 replaces the compliance *mechanism* (allowlist entry ->
# scanner-verified declaration or helper call); it deliberately does not
# also expand the *trigger*.
_JSON_WRITE_MARKERS = ("json.dump(", "json.dumps(", "atomic_json(")
_PROVENANCE_HELPER_NAMES = ("write_experiment_artifact", "write_stamped_artifact", "stamp_sidecar")


def _writes_artifacts_json_without_helper(path: Path) -> bool:
    """A script counts as an "offender" if it writes JSON somewhere, mentions
    an `artifacts/` path at all, and never calls a provenance helper.

    Deliberately a light, static, textual check (matching this project's own
    "static grep-based test, not a runtime hook" design choice) -- it can
    over-flag a script that mentions "artifacts" in prose and writes JSON
    somewhere unrelated (harmless: see ``READ_ONLY_SCRIPT`` below), but it is
    meant to catch the real case: a NEW script copying the established
    `OUTPUT_DIR = Path("artifacts/...")` + `json.dump(...)` convention
    without a helper this registry depends on.
    """

    text = path.read_text(encoding="utf-8")
    if any(helper in text for helper in _PROVENANCE_HELPER_NAMES):
        return False
    if "artifacts" not in text:
        return False
    return any(marker in text for marker in _JSON_WRITE_MARKERS)


def test_every_script_writing_artifacts_json_uses_the_provenance_helper() -> None:
    """Every triggered script must be resolved one of four ways: it calls a
    sanctioned provenance helper (``write_experiment_artifact``,
    ``write_stamped_artifact``, or the tabular-output ``stamp_sidecar``), it is
    scanner-verified read-only (a ``READ_ONLY_SCRIPT = True`` claim the scanner
    actually confirms, honouring any ``READ_ONLY_EXCEPTIONS`` for destinations
    it cannot statically resolve), or it delegates its write to a function
    listed in ``nfl_ats.script_contracts.STAMPED_LIBRARY_WRITERS`` (ENG-38: a
    real write this single-file scanner cannot see, but one
    ``test_script_contracts.py`` independently verifies stamps at the library
    call site). ENG-38 (2026-09-04) finished wiring the last of the
    non-experiment writers, so there is no longer a fourth, allowlist-based
    escape hatch: anything that does not resolve one of these three ways is
    new, unaudited, and must be resolved before this test passes -- exactly
    the old allowlist's "fails immediately" property, now backed entirely by
    mechanical checks instead of a hand-maintained filename list.
    """

    triggered = {
        path.name
        for path in sorted(SCRIPTS_ROOT.glob("*.py"))
        if _writes_artifacts_json_without_helper(path)
    }
    unresolved = set()
    for name in triggered:
        contract = scan_script(SCRIPTS_ROOT / name)
        if (
            contract.calls_provenance_helper
            or contract.is_read_only_verified
            or contract.calls_stamped_library_writer
        ):
            continue
        unresolved.add(name)

    assert not unresolved, (
        "New/changed script(s) write JSON into artifacts/ without a sanctioned "
        f"provenance path: {sorted(unresolved)}. Either wire in a helper (see "
        "src/nfl_ats/provenance.py: write_experiment_artifact / "
        "write_stamped_artifact / stamp_sidecar), declare `READ_ONLY_SCRIPT = "
        "True` if the nfl_ats.script_contracts scanner confirms it has no "
        "artifacts/registry write sites, or add the delegated-to function to "
        "nfl_ats.script_contracts.STAMPED_LIBRARY_WRITERS if the real write "
        "lives in an imported library function that itself stamps."
    )


def test_read_only_declarations_are_scanner_verified() -> None:
    """A script that declares ``READ_ONLY_SCRIPT = True`` must actually BE
    read-only by the scanner's own static check, regardless of whether it was
    ever a provenance-gate "offender" -- the declaration is a promise other
    tooling may trust (e.g. "safe to run without side effects"), so a false
    claim is a bug in the script, not a gap in this test's scope.
    """

    false_claims = []
    for path in sorted(SCRIPTS_ROOT.glob("*.py")):
        contract = scan_script(path)
        if contract.declares_read_only and not contract.is_read_only_verified:
            false_claims.append((path.name, contract.gated_write_sites))
    assert not false_claims, (
        "Script(s) declare READ_ONLY_SCRIPT = True but the scanner finds a write "
        f"site under artifacts/ or registry/ (or an unlisted `unknown` site): "
        f"{false_claims}. Fix the script (it is not read-only) or add a "
        "READ_ONLY_EXCEPTIONS entry with a reason for a destination the scanner "
        "cannot statically resolve."
    )


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------


def _load_backfill_module() -> ModuleType:
    path = SCRIPTS_ROOT / "backfill_experiment_registry.py"
    spec = importlib.util.spec_from_file_location("backfill_experiment_registry", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_run(directory: Path, filename: str, payload: dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(json.dumps(payload), encoding="utf-8")


def test_backfill_produces_registry_rows_that_parse(tmp_path: Path) -> None:
    backfill = _load_backfill_module()
    artifacts_root = tmp_path / "artifacts"
    registry_root = tmp_path / "registry"

    # A cli.py-shaped metadata.json: a "provenance" block to lift.
    _make_run(
        artifacts_root / "demo_command" / "20260101T000000Z",
        "metadata.json",
        {
            "command": "demo-command",
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "games": 42,
            "provenance": {
                "configuration": {"command": "demo-command"},
                "configuration_sha256": "abc123",
                "feature_table": {"sha256": "feat456"},
                "code": {"revision": "deadbeef", "dirty": False},
                "uv_lock_sha256": "lock789",
            },
        },
    )
    # A bare run.json: the provenance dict IS the whole file.
    _make_run(
        artifacts_root / "other_command" / "20260102T000000Z",
        "run.json",
        {
            "configuration": {"command": "other-command"},
            "configuration_sha256": "xyz999",
            "feature_table": {"sha256": "featabc"},
            "code": {"revision": "cafef00d", "dirty": True},
            "uv_lock_sha256": "lockdef",
        },
    )
    # A dual-provenance file (the availability-ablation shape): both should
    # be noted, one used, and the row still has to parse.
    _make_run(
        artifacts_root / "dual_command" / "20260103T000000Z",
        "metadata.json",
        {
            "command": "dual-command",
            "created_at_utc": "2026-01-03T00:00:00+00:00",
            "baseline_provenance": {
                "configuration": {},
                "configuration_sha256": "base1",
                "feature_table": {"sha256": "f1"},
                "code": {"revision": "rev1", "dirty": False},
                "uv_lock_sha256": None,
            },
            "learned_provenance": {
                "configuration": {},
                "configuration_sha256": "learned1",
                "feature_table": {"sha256": "f2"},
                "code": {"revision": "rev1", "dirty": False},
                "uv_lock_sha256": None,
            },
        },
    )

    summary = backfill.run_backfill(artifacts_root, registry_root)
    assert summary["run_directories"] == 3
    assert summary["backfilled"] == 3
    assert summary["unbackfillable"] == 0

    registry_files = sorted((registry_root / "experiments").glob("*/*.json"))
    assert len(registry_files) == 3
    for path in registry_files:
        record = load_experiment_record(path)
        assert record.provenance_backfilled is True
        assert record.backfill_note

    dual_row = load_experiment_record(
        registry_root / "experiments" / "dual-command" / "20260103T000000Z.json"
    )
    assert dual_row.config_hash == "base1"  # alphabetically-first key, deterministic
    assert "learned_provenance" in (dual_row.backfill_note or "")


def test_backfill_invents_nothing_for_a_run_without_recoverable_provenance(
    tmp_path: Path,
) -> None:
    backfill = _load_backfill_module()
    artifacts_root = tmp_path / "artifacts"
    registry_root = tmp_path / "registry"

    # metadata.json with no provenance-shaped block anywhere.
    _make_run(
        artifacts_root / "bare_metadata" / "20260101T000000Z",
        "metadata.json",
        {"games": 10, "note": "no code revision recorded here"},
    )
    # A run directory with no metadata.json/run.json at all.
    (artifacts_root / "totally_bare" / "20260102T000000Z").mkdir(parents=True)

    summary = backfill.run_backfill(artifacts_root, registry_root)
    assert summary["backfilled"] == 0
    assert summary["unbackfillable"] == 2

    doc = (registry_root / "experiments" / "UNBACKFILLABLE.md").read_text(encoding="utf-8")
    assert "bare_metadata" in doc
    assert "totally_bare" in doc
    assert "no metadata.json or run.json" in doc
    assert "no artifact_provenance()-shaped block" in doc

    # No registry rows were fabricated for either directory.
    rows = list((registry_root / "experiments").glob("*/*.json"))
    assert rows == []


def test_backfill_is_idempotent(tmp_path: Path) -> None:
    """Re-running the backfill (e.g. after a new week's artifacts appear)
    must not raise or corrupt already-written rows."""

    backfill = _load_backfill_module()
    artifacts_root = tmp_path / "artifacts"
    registry_root = tmp_path / "registry"
    _make_run(
        artifacts_root / "demo_command" / "20260101T000000Z",
        "metadata.json",
        {
            "command": "demo-command",
            "provenance": {
                "configuration": {},
                "configuration_sha256": "abc123",
                "feature_table": {"sha256": "feat456"},
                "code": {"revision": "deadbeef", "dirty": False},
                "uv_lock_sha256": None,
            },
        },
    )
    first = backfill.run_backfill(artifacts_root, registry_root)
    second = backfill.run_backfill(artifacts_root, registry_root)
    assert first == second


# ---------------------------------------------------------------------------
# The real deliverable: every row already committed under
# registry/experiments/ must itself parse and be honestly labeled.
# ---------------------------------------------------------------------------


def _save_row(registry_root: Path, command: str, stamp: str, **overrides: object) -> None:
    payload: dict[str, object] = {
        "experiment_id": f"{command}/{stamp}",
        "recorded_at": "2026-01-01T00:00:00+00:00",
        "command": command,
        "artifact_directory": f"artifacts/{command}/{stamp}",
        "config_hash": "abc123",
        "schema_version": 1,
        "metrics": {},
        "source": f"nfl-ats {command}",
        "provenance_backfilled": False,
    }
    payload.update(overrides)
    record = experiment_record_from_payload(payload)
    directory = default_experiment_registry_root(registry_root) / experiment_command_slug(command)
    directory.mkdir(parents=True, exist_ok=True)
    save_experiment_record(record, directory / f"{stamp}.json")


def test_verify_experiment_links_finds_existing_and_flags_missing(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    artifacts_root = tmp_path / "artifacts"
    existing = artifacts_root / "demo-command" / "20260101T000000Z"
    existing.mkdir(parents=True)
    _save_row(registry_root, "demo-command", "20260101T000000Z")
    # A row whose artifact_directory points nowhere.
    _save_row(
        registry_root,
        "other-command",
        "20260102T000000Z",
        source="artifacts/other_command/20260102T000000Z/run.json",
    )

    results = verify_experiment_links(registry_root=registry_root, artifacts_roots=[artifacts_root])
    by_id = {r.experiment_id: r for r in results}
    assert by_id["demo-command/20260101T000000Z"].exists is True
    assert by_id["other-command/20260102T000000Z"].exists is False
    # Only the path-style source (run.json) avoids the source_not_a_path flag.
    assert "source_not_a_path" in by_id["demo-command/20260101T000000Z"].flags
    assert "source_not_a_path" not in by_id["other-command/20260102T000000Z"].flags
    # Absolute stored paths are reported, not silently re-based.
    abs_row = tmp_path / "abs_command" / "20260103T000000Z"
    abs_row.mkdir(parents=True)
    _save_row(
        registry_root,
        "abs-command",
        "20260103T000000Z",
        artifact_directory=str(abs_row),
    )
    results = verify_experiment_links(registry_root=registry_root, artifacts_roots=[artifacts_root])
    abs_result = next(r for r in results if r.experiment_id == "abs-command/20260103T000000Z")
    assert "absolute_machine_path" in abs_result.flags
    assert abs_result.exists is True
    assert abs_result.resolved_path == str(abs_row)


def test_verify_experiment_links_flags_unsafe_id(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    command_with_spaces = "weird name (v2)"
    _save_row(registry_root, command_with_spaces, "20260101T000000Z")
    results = verify_experiment_links(registry_root=registry_root, artifacts_roots=[])
    assert len(results) == 1
    flags = results[0].flags
    assert "id_not_filesystem_safe" in flags
    assert "source_not_a_path" in flags


def test_committed_registry_experiment_rows_all_parse() -> None:
    experiments_root = REPO_ROOT / "registry" / "experiments"
    if not experiments_root.is_dir():
        return  # nothing backfilled yet in this checkout
    rows = sorted(experiments_root.glob("*/*.json"))
    assert rows, "expected at least one backfilled experiment row"
    for path in rows:
        try:
            record = load_experiment_record(path)
        except ExperimentRecordError as error:  # pragma: no cover - failure path
            raise AssertionError(f"{path} does not parse as an ExperimentRecord: {error}") from None
        assert record.experiment_id
        assert record.command


def test_every_scripts_import_in_cli_puts_the_repo_root_on_the_path_first() -> None:
    """`scripts` is not part of the installed package.

    It resolves only when the repository root happens to be on ``sys.path``,
    which ``python -m nfl_ats`` provides and the ``nfl-ats`` console script does
    NOT. Because ``nfl_ats.weekly._cli_runner`` dispatches every weekly-run step
    IN-PROCESS and step 7 (``ingest-player-arrests``) is fail-closed, a bare
    ``from scripts...`` import inside a CLI handler is a lock-day abort: the
    documented Tuesday command in ``docs/week1_readiness.md`` is the console
    script, so the real 2026-09-08 run would have died before publishing
    anything. That is not hypothetical -- it was live until 2026-08-25.

    This is a static check rather than a runtime one on purpose: the import is
    lazy, so nothing executes it until lock day, and a test that only ran the
    happy path would keep passing while the command stayed broken.
    """

    source = (REPO_ROOT / "src" / "nfl_ats" / "cli.py").read_text(encoding="utf-8")
    offenders: list[str] = []
    for block in source.split("\ndef ")[1:]:
        name = block.split("(", 1)[0]
        if "from scripts." not in block and "import scripts" not in block:
            continue
        guard = block.find("_repo_root_on_path()")
        first_import = min(
            (
                index
                for index in (block.find("from scripts."), block.find("import scripts"))
                if index >= 0
            ),
            default=-1,
        )
        if guard < 0 or guard > first_import:
            offenders.append(name)
    assert not offenders, (
        "CLI handler(s) import `scripts.*` without calling _repo_root_on_path() "
        f"first: {sorted(offenders)}. Under the `nfl-ats` console script that "
        "raises ModuleNotFoundError, and weekly-run dispatches in-process, so a "
        "fail-closed step importing this way aborts the whole lock-day run."
    )
