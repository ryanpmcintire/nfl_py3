# Backlog research continuation

## Goal

Complete a substantive research checkpoint after the current-week update without
selecting a model from recent results.

## State

**Measured, complete:** corrected frozen scorecard (28–19, one pending), baseline
timing/availability and stale challenger grades; repaired pool input chronology;
completed four-cell bootstrap precision replay and three simulator rule fixes.
All numerical evidence and limitations: [checkpoint report](../../backlog_research_20260928.md).
Pool fitting remains blocked by a genuinely missing predeadline split.

## Tried

Real commands: `python scripts/prospective_scorecard_2026.py`,
`python scripts/pool_field_share_fit.py` (expected fail-closed missing input),
`python scripts/positive_control_precision_replay.py --reference
artifacts/positive_control_power/20260927T042645Z/results.json
--draw-prefixes 400,2000,20000`, fixed before/after simulator replay, and
`nfl-ats weak-signals record --batch` for both unresolved simulator loss effects.
Independent verifiers checked probabilities, 1,202,400 precision prediction rows,
20,004 simulator rows, exact historical replay, and artifact hashes.
Ruff format/check, `mypy src` (250 modules), and
`pytest -q --basetemp .tmp/pytest-backlog-research-published-20260928` passed
(1,645 tests). No new test functions/files or proposed test deletions.
Board and predictions republished. Rendered changes reviewed; Findings now links
to saved picks graded at their original lines. All 16 frozen Week 3 decisions and
the locked Best Pick remain identical. Enrolled validation sources still match.

## Next

Continue in a fresh thread from the independent Week 4 capture lane and the
active SIM-08 / pool / positive-control lanes. The latter records the next
matched-effect power work; this checkpoint does not close natural signals.

## Open

No interim Week 4–18 selection or enrollment reset. Simulator margin 3 remains
deficient and margin 17 is above its reference band. No new research parameter
or simulator distribution was promoted. Preserve unrelated owner changes to
AGENTS.md, workflow policy, the popup lane, and pre-existing September 27 records.
