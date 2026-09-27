# SIM-04/SIM-05 Ruff repair

## Goal

Clear the existing Ruff format and check failures in the three scoped simulator scripts without changing model math, chronology, CLI behavior, report text, or output schemas.

## State

Complete. The initial scoped check found 80 issues: 57 in `scripts/sim04_loso.py`, 22 in `scripts/sim05_whatif.py`, and 1 in `scripts/sim04_backoff_diag.py`. The final scoped check found none.

## Tried

- Applied Ruff formatting to the three scripts.
- Removed one unused local, removed redundant `int(len(...))` conversions, renamed two unused unpacked values, and retained required local import ordering with E402 pragmas.
- Split 32 long string literals through implicit concatenation. Full parsed ASTs were identical before and after those splits.
- Measured verification in `.tmp/sim-ruff-final.log`: Ruff format check exit 0, Ruff check exit 0, Python compile exit 0, both supported CLI help paths exit 0, and Git diff check exit 0. Aggregate failures were 0.
- `scripts/sim04_backoff_diag.py` has no argument parser or help path and runs the study at import, so it was compiled but not invoked.

## Next

Include these files in the root release-gate review.

## Open

No scoped blockers.
