# acc-u6-half-labels

## Goal
Train the base model with each half (or quarter) as its own label. Parent:
docs/lanes/accuracy-ceiling-theory.md.

## State
DONE 2026-10-01 as a design error, not a signal. With identical inputs on
every row, a stacked ridge on K sub-labels equals the base ridge at alpha/K.
Measured: max prediction difference 5e-13 (halves), 2e-12 (quarters) on the
2020 training fit (scripts/mod24_u6.py check). The multi-task arm H2 equals
H1. So per-half labels only retest ridge strength, which unit 1 alpha-only
and MOD-24 U1 already graded. The grading run was stopped by the
orchestrator. No registry cell: no new measurement exists.

## Tried
scripts/mod24_u6.py; artifacts/mod24_u6/ (selection.json, halves.parquet).

## Next
Per-period labels add information only with period-specific inputs. The
pbp_coaching_traits family (q3 adjustment) already covers the obvious one.
Not pursued.

## Open
None.
