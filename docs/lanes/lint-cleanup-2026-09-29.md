# Research script lint cleanup — 2026-09-29

## Goal
Clear Ruff lint and formatting failures in the packet's scripts without changing computed results.

## State
Complete; ready for the orchestrator to commit. **Measured:** allowed scope went from 400 lint errors / 11 unformatted files to 0 / 0; all 11 scripts passed `py_compile` and scoped `git diff --check`.

Scope, all under `scripts/`: `lead69_cfb_replication.py`, `lead73_unit1.py`, `lead81_unit1.py`, `lead82_unit1.py`, `lead87_unit1.py`, `lead88_unit1.py`, `mod22_unit4.py`, `mod22_unit5.py`, `st_ratings_ats_study.py`, `tiebreaker_total_study.py`, `xlg09_unit6.py`.

## Tried
- Saved original sources; formatted and applied safe fixes while excluding C416 until reviewed. Kept five pandas groupby comprehensions with targeted C416 pragmas; removed four unused E402 pragmas. Replaced thread-limit loops with environment updates before NumPy imports, retained import paths, and made equivalent list/conditional edits.
- Wrapped report strings and escaped ambiguous Unicode without changing their runtime contents. **Measured:** 74/80 function ASTs are identical; the remaining six differ only by six list-unpacking edits, one conditional assignment, and removal of `int(len(...))`. All match after those reviewed syntax changes; startup environment values and paths match.
- Commands, with `SCOPE` expanded to the 11 paths above: `.tools/uv.exe run --no-sync ruff check SCOPE`; `.tools/uv.exe run --no-sync ruff format --check SCOPE`; `.tools/uv.exe run --no-sync python -m py_compile SCOPE`; `git diff --check -- SCOPE docs/lanes/lint-cleanup-2026-09-29.md`. All exit 0. Used temporary UV/cache directories; ran no experiments or tests.
- Logs, originals, exact verification argv, and semantic review: `%TEMP%/nfl-lint-cleanup-20260929-25kt_qew/`.

## Next
Orchestrator reviews and commits these changes, then repeats repository-wide checks after the other workers finish.

## Open
**Measured:** repository baseline was 469 errors / 12 unformatted files, versus the packet's reported 492 / 12. Latest whole-repository check has 251 errors, all in excluded `lead82_unit2.py`, `lead87_unit2.py`, and `lead88_unit2.py`; those three remain unformatted. Formatting also reports invalid UTF-8 in `docs/lead69_inventory.md`. Default formatting diagnostics crashed; `--output-format=concise` exposed these remaining failures. Concurrent counts are not attributable solely to this lane. No excluded files or generated reports were edited. No commits or publication.

## Record commands
None: lint-only work; no research outcomes were fitted, scored, or adjudicated.
