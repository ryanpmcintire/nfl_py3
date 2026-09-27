# POL-10 prospective 2026 evidence

## Goal
Score frozen pregame decisions at their recorded line, retain close grades as
secondary evidence, and compare challengers on exactly paired decisive games.

## State
**Measured:** `nfl-ats prospective-score --features data/processed/game_features.parquet --start-season 2026 --bootstrap-samples 1000` exited 0 on 2026-09-27.
The current artifact is `artifacts/prospective_scoring/20260927T133936Z`.
The run reported 56 scored arms, one arm without recorded rows, one invalid arm,
two missing ledgers, and four unsupported arms. No missing arm is imputed.
**Measured:** the base model without overlays has 33 paired games against the
active model: accuracy difference -9.09 percentage points, 95% week-bootstrap
interval [-18.75, 0.00], probability_positive 0.1505. This remains unresolved;
AGENTS.md permits closure only on an admissible mechanism or power ground.
Historical records are not evidence of prospective profitability.

## Tried
- Added strict recorded-before-kickoff validation and deterministic paired-game
  identity, with matching reference, metric, and bootstrap metadata.
- Kept Tuesday and Sunday nominee arms separate; absent Sunday records remain
  absent. A malformed equal-book arm fails as a whole instead of being repaired.
- Replaced history-page comparison fallbacks with evidence from the displayed
  paired records. Historical comparison percentages are explicitly labeled.
- **Measured:** added a producer guard for missing or invalid recorded pick sides. The real
  builder accepted a valid synthetic row and rejected ten injected malformed arms;
  historical ledger rows remain unchanged.
- **Measured:** the full repository suite passed: 1,645 tests. The missing-pick guard also passed
  its isolated production-builder probe and Ruff checks.
- Verification details are saved in `.tmp/pol10-*-verification.md` and
  `.tmp/pol10-independent-final-review.md`; the full score log is
  `.tmp/pol10-hardened-canonical-score.log`.

## Next
**Measured:** publication regenerated successfully after the reader-copy fix; all 88
existing rendering tests passed. The incremental rendered-page review is clear,
handoff is current, and this batch is ready for its verified local checkpoint.
Review the default-off rookie-crew recorder before any integration or enrollment.

## Open
The equal-book source has a missing pick side; do not infer it. Two ledgers and
four scorer adapters remain unavailable. Missing probabilities are not fabricated.
No research closure or serving promotion is made here. Push remains blocked by
automatic approval review; the outstanding user approval request covers it.