# Independent combination validation

## Goal
Prioritize a declared future comparison of combined versus raw probabilities
after the Week 3 audit exposed historical feature-selection reuse.

## State
Implementation verified in `src/nfl_ats/independent_validation.py`, the
`independent-validation` CLI, and ten dedicated scheduler captures. Protocol:
`registry/studies/combined_vs_raw_2026.json`; design and commands:
`docs/independent_combination_validation.md`. **Measured:** enrolled 224 future
games at 2026-09-28T10:44:18Z. Commitment:
`87d0108b14e14ad1ad530017810da1a6373a2d967e40377bb3bc03f1d64eba4e`.
Local enrollment: `artifacts/prospective/independent_validation/enrollment.json`.
Frozen comparison covers 2026 Weeks 4–18; primary paired Brier, secondary
ATS/log loss/reliability; no interim result-based selection.

## Tried
**Measured:** `ruff format --check .`, `ruff check .`, `mypy src`, and
`pytest -q --basetemp .tmp/pytest-independent-final` pass in the locked uv
environment with `UV_CACHE_DIR=.tmp/uv-cache`. Updated the existing command-list
fixture. Real enrollment and score commands succeed; score withholds results.
Scheduler `--rehearse-all --only-prefix independent_validation_ --stop-on-fail`:
10/10 dry commands pass. Restarted the idle daemon hidden; health exit 0,
running PID 4264, code and schedule current at 2026-09-28T10:46:45Z. Rehearsal
uses the current Week 3 forecast and therefore captures no study games.

## Next
Inspect `nfl-ats independent-validation status` after the first Week 4 forecast;
verify eligible games appear before their pool deadlines. Keep capture coverage
complete, preserve enrollment, and use `score` only under the declared plan.

## Open
No independent result exists yet. Pinned-source or recipe drift must stop the
capture rather than mix policies. Missing captures prevent complete comparison.
One season may remain unresolved; no automatic promotion or research closure.
Proposed removal, owner decision: the existing help-order assertion
`test_registration_order_is_the_help_listing_order` duplicates parser order
without checking behavior. No tests were added or deleted in this task.
