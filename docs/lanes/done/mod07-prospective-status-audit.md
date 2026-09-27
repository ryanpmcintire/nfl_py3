# MOD-07 prospective status audit

## Goal
Reconcile the stacking backlog with the current prospective registry and ledger.

## State
The status audit is complete; the research line remains unresolved.
**Measured:** `artifacts/prospective/challenger_decisions.parquet`
contains 48 unique games, 16 each in Weeks 1, 2, and 3 of 2026. Every row uses
registered fingerprint `bc77638d47e2748c`; all three recorded prediction-source
hashes match their `recommendations.csv` files. Ledger SHA-256:
`fb15413901b262b02dca1f4f6ffe48acda7f6e2da9dbb860b5386b8603d62d98`.
**Read:** the registry marks this challenger `ACTIVE_PROSPECTIVE`.
Historical stacking results remain `unresolved_below_power`; current collection
does not promote the challenger or establish an out-of-sample edge.

## Tried
- Root verified row uniqueness, fingerprint, week counts, and source hashes with
  locked Python; result saved in `.tmp/mod07-root-coverage.json`.
- Read the recorder, weekly opt-in flow, and prospective CLI. The next capture
  uses the existing workflow, rather than a new capture implementation.
- The absent historical `scripts/mod07_weak_stack.py` is a replay gap; it does
  not block the current prediction and prospective-recording path.

## Next
At the actual Week 4 lock, run the normal weekly flow with `--record-decisions`.
Continue prospective grading without replacing previous weeks or changing the
frozen challenger fingerprint. Preserve the historical research classification.

## Open
Further prospective evidence and recovery of the historical replay script remain.
This completed lane closes the status audit only.
