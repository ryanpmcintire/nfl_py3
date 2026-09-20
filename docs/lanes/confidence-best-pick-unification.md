# Confidence and Best Pick unification

## Goal

Audit actual probability reliability and within-week ranking, then unify the
served Best Pick path without claiming consistency proves predictive power.

## State

- Owner corrected the prior session: changing the star without examining
  confidence evidence was inadequate. Plan frozen before measurement in
  `docs/confidence_ranking_audit.md`; no new fit may activate from this audit.
- Read: active calibration stores coefficients and aggregate bands but no
  prediction-level replay; loader omits active-model binding. Initial
  nomination fits an unused separate ranker and differs from Sunday rules.
- No commit/push or candidate activation performed in this lane's integration.
- Implemented: one calibrated probability and nominee selector, discrete
  non-push serving and line sweep, active-model/source binding, held-out
  predictions, and board wording. Unlocked picks ignore legacy revisions;
  locked picks retain the published side and suppress hypothetical curves.
- Explicit as-of nomination requires every kickoff deadline, excludes games
  already locked, and preserves the recorded locked nominee. Historical calls
  without an as-of time retain their prior selection behavior.
- Root reported a regenerated forecast at `2026-week-02-20260920T150905Z`;
  compatible calibration and publication remain pending. The older active
  calibration fails the strict discrete-source loader, so the live site is
  not yet operationally complete.

## Tried

- Read production fit, loader, nomination and earlier research rationale.
- Code-path audit delegated read-only; root reconstructs production evidence.
- Real final opener evaluation completed with corrected full-week cutoff;
  output `data/environment_recovery/aligned_opener_final.txt`.
- Candidate `artifacts/pick_probability/20260920T135812Z` was fitted with
  activate=False; activation.json saved beside coefficients. Corrected audit
  and prediction-level replay: `artifacts/confidence_ranking_audit/20260920_aligned`.
  Original smooth-input diagnostic: sibling `20260920`. Neither is untouched
  outer evidence. Registry recording and result interpretation still pending.
- Measured here: focused board/publication/CLI/lineage command passed 198
  tests before the final as-of contract edit; the three affected publication
  tests then passed. Direct checks exercised expired-game exclusion, locked
  and historical selection, missing-deadline failure, discrete zero-line
  mass arithmetic, and generic calibrated probability/confidence fields.
  Focused Ruff format/check and mypy passed. No new test files or functions.
- Read-only Sunday move trace: the fitted leader-median feature excludes
  Sunday observations; the no-blackout challenger uses a different input
  contract. The current DraftKings-only feed cannot supply the three leader
  books. No new coefficient or independent flip was served from that read.

## Next

- Root completes corrected audit and unresolved-signal recording, verifies
  full gates, prepares compatible calibration, and checks the regenerated
  forecast before any activation or publication decision.
- Run live refresh/ledger checks, exercise modified scheduler argv, recover
  the missed capture window, publish the board only after the active artifacts
  align, and refresh `HANDOFF.md` before any authorized push.

## Open

- Historical feature-selection reuse prevents calling this an untouched test.
- Real-site board tests still need a compatible active calibration artifact;
  do not weaken source or model validation to make the old pointer pass.
- No commit or push authorized. Prior unrelated changes remain untouched.
- Backups: `data/environment_recovery/before_probability_unification`.
