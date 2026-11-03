# Officials wayback test KeyError items_remaining

## Goal
Make tests/test_officials_wayback_sweep.py::test_cli_exposes_fallback_captures_and_retry_unparsed pass on HEAD.

## State
Resolved before this lane started. Production scripts/officials_wayback_sweep.py:546 emits items_remaining (added by 7164e885, 2026-11-02, used at :811). The test stub lacked it; commit 61f3ac5c (2026-11-03) added it at tests/test_officials_wayback_sweep.py:555.
Measured: single test 1 passed; whole file 14 passed. No edits made by this lane.

## Tried
Ran the single test and the whole file; checked git log for both files.

## Next
None. Move this lane to docs/lanes/done/ when the owner next tidies.

## Open
The original failure was likely a run on a tree before 61f3ac5c.
