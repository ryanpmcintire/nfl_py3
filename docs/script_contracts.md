# ENG-29: read-only script declaration for the provenance gate

## What changed

`tests/test_experiment_registry.py::test_every_script_writing_artifacts_json_uses_the_provenance_helper`
used to lean on a single, hand-maintained `_ALLOWLISTED_UNSTAMPED_SCRIPTS`
frozenset: every `scripts/*.py` file that wrote JSON into `artifacts/`
without calling `write_experiment_artifact()` needed a manually-added
filename (and a reason comment, in the test file, far from the script it
described). The list only ever grew.

`src/nfl_ats/script_contracts.py` (`scan_script()`) replaces the *judgment*
half of that pattern with a mechanical one. It parses a script with `ast`
— **never imports or executes it** — and reports:

- `declares_read_only` — a module-level `READ_ONLY_SCRIPT = True`.
- `calls_provenance_helper` — a call to `write_experiment_artifact` or
  `write_stamped_artifact` anywhere in the file.
- `write_sites` — every call the scanner recognises as a potential
  filesystem write (`open(..., "w"/"a"/"x"/...)`, `Path.write_text`/
  `write_bytes`, `atomic_json`/`atomic_text`/`atomic_csv`/`atomic_parquet`/
  `atomic_bytes`, pandas `to_parquet`/`to_csv`/`to_json`, `json.dump`,
  `shutil.copy*`/`move`, `os.replace`/`rename`, `.mkdir`), each classified as
  `artifacts`, `registry`, `stdout`, `tmp_or_arg` (a caller-supplied
  `--output`/`--out` CLI argument or a tempfile, with no governed default),
  or `unknown`.

A script may now self-certify beside its own code:

```python
READ_ONLY_SCRIPT = True
# one-line reason.

# Only needed if the scanner cannot statically resolve one destination
# (e.g. it comes from a runtime-computed path):
READ_ONLY_EXCEPTIONS: dict[int, str] = {
    205: "destination is under dest_root (DEFAULT_DESTS / --dest), the mirror drive",
}
```

`ScriptContract.is_read_only_verified` is `True` only when the script
declares `READ_ONLY_SCRIPT = True` **and** every `write_sites` entry is
classified `artifacts`/`registry`-free — `stdout` and `tmp_or_arg` sites
never count against the claim; an `unknown` site does unless its line number
has a `READ_ONLY_EXCEPTIONS` entry. `test_read_only_declarations_are_scanner_verified`
checks this for every script in the repository, not just ones already
flagged by the gate, so a false declaration is caught even if the script was
never an "offender" in the first place.

## The trigger condition did not change

The gate's *trigger* — which scripts must resolve one way or another — is
still the original, narrow, textual check: mentions `artifacts` and uses
`json.dump(`/`json.dumps(`/`atomic_json(` somewhere, and does not already
mention a provenance helper. Widening it (e.g. to also fire on any mention of
`registry`, or on `to_csv`/`to_parquet`/`mkdir`-style writes) was evaluated
and rejected: it pulls in roughly 60 additional, never-audited scripts as an
accidental side effect of a ticket about *how* compliance is declared, not
*which* scripts must comply. ENG-29 replaced the compliance mechanism, not
the trigger's scope.

## What the 2026-09-04 audit found

Running the new scanner over every script in the legacy 88-entry allowlist
produced three buckets, not the old allowlist's two ("stamped" vs. "not"):

1. **36 scripts are genuinely read-only** — either zero write calls at all,
   or every write destination is a caller-supplied `--output`/`--out` path
   with no default pointing into `artifacts/` or `registry/`. These now carry
   `READ_ONLY_SCRIPT = True` (three also needed a small
   `READ_ONLY_EXCEPTIONS` entry: `backup_data.py`'s mirror-drive destination,
   `capture_scheduler.py`'s `data/scheduler_*` paths, and
   `surface_familiarity_screen.py`'s scratchpad temp directory — none of
   which the scanner can statically resolve to a literal). Removed from the
   allowlist.
2. **2 scripts** (`snapshot_diff.py`, `prospective_scorecard.py`) write real
   artifacts but are explicitly not experiments (no hypothesis, no cell, no
   closing ground), and `write_experiment_artifact()` always creates a
   `registry/experiments/<slug>/<stamp>.json` row, which would misrepresent
   them. Both now call the new `nfl_ats.provenance.write_stamped_artifact()`
   — stamps `code_revision`/`code_dirty`/`recorded_at` onto the payload, like
   `write_experiment_artifact()` does, but never writes a registry row.
   Removed from the allowlist.
3. **50 scripts were genuine non-experiment writers that were NOT read-only.**
   Most are measure-only screens/evals/audits whose result tables are the
   deliverable, written under a governed `artifacts/...` default (a
   `--output`/`--out` flag that defaults into `artifacts/`, or a hardcoded
   `OUTPUT_DIR`). Declaring these read-only would have been false. A further
   seven (six weekly ledger recorders plus one lock-day wrapper) delegate
   their write to an imported function this single-file scanner cannot see —
   a real write by default, invisible to the AST of the entry-point script
   alone (the same single-file scope the old grep-based gate always had).
   ENG-38 (2026-09-04) migrated all 50: each script's own write site now
   calls `write_stamped_artifact()` (JSON) or the new `stamp_sidecar()`
   (CSV/Parquet — writes a `<path>.provenance.json` beside the artifact
   rather than touching the table's own bytes); the seven delegators are
   stamped inside the shared library function they call
   (`nfl_ats.bye_edge_fade_overlay.record_bye_edge_fade_challenger_decisions`
   and its five tilt-overlay siblings, plus
   `nfl_ats.scheduled_lock.execute_scheduled_lock`), with that function
   listed in `nfl_ats.script_contracts.STAMPED_LIBRARY_WRITERS` so
   `scan_script()` can see through the delegation — verified by
   `tests/test_script_contracts.py::test_stamped_library_writers_really_stamp`,
   an AST check that the listed function really calls a stamping helper.
   `_NON_EXPERIMENT_WRITER_SCRIPTS` in `tests/test_experiment_registry.py` is
   now empty and has been deleted.

## Follow-up

None outstanding for the allowlist itself. `test_every_script_writing_artifacts_json_uses_the_provenance_helper`
now only accepts three resolutions for a triggered script: a provenance
helper call, a scanner-verified `READ_ONLY_SCRIPT` declaration, or a
delegated `STAMPED_LIBRARY_WRITERS` call — there is no remaining
allowlist-based escape hatch. A new script that trips the trigger must
resolve one of those three ways.
