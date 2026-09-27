# Repository verification cleanup

## Goal
Restore required repository checks without changing research or served predictions.

## State
**Measured:** repository Ruff check passes and Ruff format reports 1,109 files formatted. Mypy passes for 240 source files. Logs: `.tmp/backlog-ruff-final.log`, `.tmp/backlog-format-final2.log`, `.tmp/backlog-mypy.log`.
**Measured:** the full existing suite returned 1,634 passes and 11 failures, all from one prospective-refresh fixture lacking refresh_run_id. Updating that existing fixture produced 11/11 passes in the focused rerun. Logs: `.tmp/backlog-pytest.log`, `.tmp/backlog-best-pick-pytest.log`. No test functions were added.

## Tried
Reproduced pre-existing Ruff errors against HEAD before mechanical fixes. Reviewed AST differences; format-only edits preserve ASTs. Removed a UTF-8 BOM from the restored roadmap inventory script after isolating the formatter panic; its AST is unchanged.
Commands used the locked environment: `uv run --no-sync ruff check .`, `ruff format --check .`, `mypy src`, `pytest -q` with workspace-local basetemp, then `pytest tests/test_best_pick_refresh_prospective.py -q` with a fresh workspace-local basetemp.

## Next
Keep subsequent worker changes on the same checks; do not add coverage during the test moratorium.

## Open
The full suite was not rerun after the fixture repair; the previously failing module passed its focused rerun. Simulator and positive-control workers continue separately.
