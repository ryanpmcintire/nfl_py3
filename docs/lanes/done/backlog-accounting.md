# One backlog

## Goal

Use roadmap and backlog as names for the same list. Never substitute a grouped
batch count for the total unfinished items in `ROADMAP.md`.

## State

The canonical accounting rule is in `ROADMAP.md`. Corrected the seven-category
report, completed lane, and index: the implementation batch is complete while
its unresolved work remains in the same backlog. No research verdict changed.
**Measured:** the inventory now counts all 56 rows: 13 done, 43 remaining,
including 23 active/planned. The previous count omitted SIM-08's hammer symbol;
normalizing it did not add a project. Unknown statuses now raise an error.

## Tried

The real inventory and an unknown-status input both behaved as intended.
`ruff format --check .`, `ruff check .`, `mypy src`, and the full existing
`pytest -q --basetemp .tmp/pytest-backlog-accounting-retry` passed (1,645 tests).
The first run's sole failure was a subprocess using an inaccessible default uv
cache; the clean retry inherited `UV_CACHE_DIR=.tmp/uv-cache`. Commands used
`.tools/uv.exe run --no-sync`. No tests were added; code review found no test
coverage to propose removing. `git diff --check` passed.

## Next

Future backlog answers use the canonical remaining count and existing item IDs.
Any selected batch is explicitly a subset. Research next steps remain in
`ROADMAP.md`; this maintenance task does not run project jobs or publication.

## Open

Other working-tree changes are preserved, including owner policy edits,
the popup lane, September 27 records, and four generated-page greeting changes.
