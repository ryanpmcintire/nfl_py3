# Independent combination validation

The priority is to measure whether the fitted combination improves the raw
model on games unavailable during feature selection. **Read:** the historical
selection overlap and Week 3 trace are recorded in
[the variance assessment](lanes/done/week03-variance-assessment.md).
Those historical scores remain development evidence.

The [historical selection replay](independent_combination_historical_replay.md)
is now complete. It separates selection, validation, calibration, and outer
test windows and preserves its failed primary result alongside a clearly
labeled calibration-stage diagnostic. It does not erase the earlier research
choices or replace the untouched prospective cohort below.

## Declared comparison

- Cohort: every scheduled 2026 regular-season game in Weeks 4–18, enrolled before
  any cohort game. Weeks 1–3, development seasons, and playoffs are excluded.
- Arms: frozen combined probability, raw model probability, and neutral market
  probability of 0.5. Both model arms use the same original pool grading line.
- Freeze the fitted combination coefficients, raw generation recipe, forecast
  schema, and the recorded prediction-source hashes. Pregame data and ordinary
  training updates under that fixed recipe may change. A recipe or pinned-source
  change stops capture; never silently reset enrollment or blend versions.
- Retain every capture. Select the latest eligible capture before the existing
  pool deadline for each game. The original line is fixed at first capture.
  Reject post-deadline records, observed outcomes, and changed grading lines.
- Primary contrast: mean raw Brier loss minus mean combined Brier loss on
  decisive games. Positive values favor the combination. Use the fixed
  week-block bootstrap: 20,000 draws, seed 20260928, 95% interval, and
  `probability_positive`.
- Secondary, descriptive outputs: log loss, ATS accuracy, disagreement record,
  and reliability. Report all three arms, not a winning subset. The family has
  one primary contrast, one secondary paired accuracy contrast, nine arm-level
  loss/accuracy summaries, and twelve predeclared reliability cells (three arms
  by confidence bands 50–55%, 55–60%, 60–65%, and 65–100%). Disagreement counts
  describe the paired comparison; no selected subgroup establishes an edge.
- Record discrete push mass and report pushes separately. Binary scores use
  probabilities conditional on no push. Never regrade using a closing line.
- Interim access is coverage only. Release performance only after all declared
  games have final scores. Missing captures block a complete comparison and are
  listed; no retrospective prediction backfill is permitted.

The machine-readable declaration is
[`registry/studies/combined_vs_raw_2026.json`](../registry/studies/combined_vs_raw_2026.json).
Enrollment stores its own digest and embeds the exact cohort, coefficients,
recipe, and source hashes. Capture files carry the enrollment digest and their
own content digest. The enrollment commitment is copied into the tracked lane.
Local artifacts are deliberately not committed.

## Operation

Use the locked environment to run `nfl-ats independent-validation`:

- `enroll`: create the immutable enrollment once.
- `capture` (or `capture --dry`): retain eligible paired pregame probabilities.
- `status`: inspect capture coverage and pinned-source agreement.
- `score`: join the normal feature artifact's final scores; withhold interim
  results. `--features PATH` selects an explicit final-outcome artifact.

Scheduled captures run daily at 12:10 ET, Thursday at 19:55 ET, and Sunday at
08:50 and 12:50 ET. They use the active forecast and honor the normal pool lock,
including an earlier kickoff. The daily pass supplies a pregame fallback when
an extra capture is unavailable. Inspect coverage as forecasts move to Week 4;
an out-of-period rehearsal does not verify an actual future-game capture.

Artifacts live under `artifacts/prospective/independent_validation/`: enrollment,
append-only captures, and the eventual report. This is a paper comparison, with
no automatic promotion or side-changing rule. Routine served-model updates do
not change the frozen combination coefficients used by this study.

## Interpretation

**Inferred:** one prospective season may remain too small to resolve a modest
improvement and cannot establish stability across seasons. Report the estimate,
interval, probability positive, decisive-game record, and missingness before
making a decision. Follow the research rules in `AGENTS.md`: an inconclusive
result remains `unresolved_below_power`; zero crossing does not close a signal.
Any eventual terminal verdict needs an admissible registry closing ground.
The study does not automatically choose the served card.
