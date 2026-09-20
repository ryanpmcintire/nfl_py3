# Confidence and Best Pick unification

## Goal

Audit probability reliability and within-week ranking, then keep the served
Best Pick on the same calibrated probability as each game's side and display.
Consistency alone is not evidence that the top-ranked game wins more often.

## State

- The shared selector, discrete conditional non-push probability, line sweep,
  active-model/source binding, and board wording are published in release
  `5fb88a2` on `master` and `origin/master`. Root measured a Pages build for
  that exact SHA and a live index hash equal to committed `docs/index.html`.
- The active Week 2 forecast is
  `margin_predictions/2026-week-02-20260920T150905Z`; compatible active fit
  `pick_probability/20260920T152908Z` includes the Sunday-through-pregame
  leader-move feature. The final card and four pages were published with LA
  -7.5 as Best Pick at 65.7%; 16 paper and six revision rows reconcile with
  the served card, including the frozen Thursday result.
- Unlocked picks ignore stale legacy revisions; locked picks preserve their
  recorded side without a hypothetical new curve. Explicit as-of nomination
  excludes expired deadlines and fails closed when a kickoff is unknown.
- The ranking audit remains open. `docs/confidence_ranking_audit.md` fixed
  its plan before measurement; reuse of historical feature choices prevents
  calling its replay an untouched outer test. No ranking change is justified
  solely by the published card's internal consistency.

## Tried

- Corrected aligned opener evaluation lives in
  `data/environment_recovery/aligned_opener_final.txt`. Inactive candidate
  fit `pick_probability/20260920T135812Z` and prediction-level replay
  `confidence_ranking_audit/20260920_aligned` are research inputs, not the
  active fit or independent outer evidence.
- Focused integration checks passed 204 attribution/board tests. The release
  passed Ruff format/check, mypy, comment guard and 4,529 tests (nine skipped).
  No new test files or functions were added.
- The Sunday movement trace found the prior fitted leader feature excluded
  Sunday observations. The published fit now has the separate Sunday-through-
  pregame input contract; current private quotes can inform derived picks but
  cannot appear as public Books-now prices.

## Next

- Finish the predeclared reliability and within-week ranking audit, record
  unresolved cells under the weak-signals rules, and state what the evidence
  means for Best Pick selection before considering another ranker change.

## Open

- The confidence-ranking audit has no untouched outer test because historical
  feature selection was reused. Today's public Books-now prices remain
  unavailable under the source terms. Backups are in
  `data/environment_recovery/before_probability_unification`.