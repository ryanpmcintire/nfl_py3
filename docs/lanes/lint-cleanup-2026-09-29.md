# Research script lint cleanup — 2026-09-29

## Goal
Make repository Ruff checks pass with behavior-preserving Python cleanup; no experiments.

## State
Python scope complete. **Measured:** repo-wide lint **275 → 0** (final exit 0); files needing formatting **4 → 1**, plus the same one unreadable Markdown file (final exit 2). Final format scan: 1,328 already formatted. Remaining blockers are outside this packet's edit scope.

## Tried
- Changed only `scripts/lead82_unit2.py`, `scripts/lead87_unit2.py`, `scripts/lead88_unit2.py`, and this lane. The other four named scripts were already clean and remain byte-identical to their starting copies.
- **Measured:** AST review of all seven scripts permits exactly two list-unpacking changes, one `pairwise` change, and one `dict.fromkeys` change in LEAD-82. All other non-import ASTs and all report-string values are identical. Wrapped 152 long literals; preserved Unicode via escapes. No new suppressions, comments, docstrings, or tests.
- **Measured:** `.tools/uv.exe run --no-sync ruff check . --output-format json` passes. `.tools/uv.exe run --no-sync ruff format --check .` reports only the two Markdown blockers below. Both scoped Ruff commands pass for all three touched scripts.
- **Measured:** `.tools/uv.exe run --no-sync python -m py_compile scripts/lead82_unit2.py scripts/lead87_unit2.py scripts/lead88_unit2.py` and scoped `git diff --check` pass.
- **Measured:** `.tools/uv.exe run --no-sync mypy src`: 253 source files pass. `.tools/uv.exe run --no-sync pytest -q --basetemp "$LINT_LOG_ROOT/pytest-run"`: 1,645 passed, 81 warnings, 65.24 seconds. The initial pytest launch could not access its default temporary directory; the retry used a fresh directory.
- `LINT_LOG_ROOT=%TEMP%/nfl-lint-remaining-20260929-63d2465386044ef79f33eda3fe94acc7/` holds originals, logs, `semantic-review.json`, exact verification argv, and `verification-summary.json`. `UV_CACHE_DIR` used its `uv-cache` child because the default uv cache was inaccessible; test thread limits were 2.
- Earlier packet remains complete: 11 scripts, 400 lint errors / 11 unformatted files to zero; review retained in `%TEMP%/nfl-lint-cleanup-20260929-25kt_qew/`. No experiments, registry writes, publication, or Git mutations ran.

## Next
Orchestrator assigns the two Markdown fixes to their owners, reruns repository formatting, and commits the reviewed cleanup.

## Open
**Measured:** `docs/lead69_inventory.md:1` is invalid UTF-8; `docs/loso_base_artifact.md:110` needs code-block formatting. No authority to edit either file was received. Another worker's transient `board_terminal.py` formatting warning cleared without an edit here.

## Record commands
None: lint maintenance performs no fitting, scoring, or research adjudication.
