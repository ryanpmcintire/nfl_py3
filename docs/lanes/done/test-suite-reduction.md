# test-suite-reduction

## Goal
Cut the test suite roughly tenfold to durable contracts only (owner, 2026-09-23), per `docs/test_suite_reduction_proposal.md`.

## State
Done 2026-09-23. Before: 233 files, 3,872 test functions, 4,529 collected, 119 s. After: 88 files, 1,495 test functions, 1,645 collected, 49 s, all passing (measured, full `pytest -q`). Whole-file deletions landed in `f982672` (commit mislabelled as a handoff refresh); trims and restorations in the following commit.

Restored from the filename-only long tail because they are durable contracts: `test_backtest.py`, `test_active_model.py`, `test_snapshots.py`, `test_nfl_week.py`, `test_lockday_contract.py`, `test_spread_explorer.py`, and `test_clv.py` trimmed to its three arithmetic/contract tests.

## Tried
Name-based trimming via `tests/scratch/trim_tests.py FILE keep1 keep2 ...` (AST removal of every other top-level test) was the cheap method for large files.

## Next
None. A further tightening pass is possible on the four largest capture files (`test_capture_observability.py`, `test_officials_wayback_sweep.py`, `test_refresh_triggers.py`, `test_officials_archive.py`) if the owner wants it.

## Open
None.
