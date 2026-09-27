# MOD-18 frozen opener inputs

## Goal
Make single-book and consensus opener evaluations reproducible from verified inputs.

## State
Completed input provenance support. Paired calibrated evaluation remains a separate task.
**Measured:** the real FanDuel producer wrote and validated 1,531 source rows;
DraftKings wrote 1,597 and consensus wrote 1,480. FanDuel/consensus share 1,410
raw quote keys; this is not a final scoreable or decisive evaluation population.

## Tried
- One authoritative quote selector supplies both the series and source inventory.
  The manifest binds feature bytes, raw snapshot manifests and quote bytes,
  requested selection, resolved book, output bytes, schema and row count.
- Producers validate immediately; `opener-evaluation --opener-line-manifest`
  validates against its supplied features, market root and line source before scoring.
- **Measured:** the real producer and validator exited 0. A deliberately mismatched
  feature hash was rejected. Full source mypy passed on 241 files; repository
  Ruff checks passed. Evidence: `.tmp/mod18-input-manifest-verification.md`,
  `.tmp/mod18-manifest-root-probe.log`, `.tmp/mod18-current-comparison-plan.md`.
- **Read:** existing evaluation does not implement the required completed-prior-season
  calibration and joint paired-source scoring. The recipe is saved in
  `.tmp/mod18-evaluator-recipe.md`; no new research outcome has been adjudicated.

## Next
Build the paired evaluation core, retain per-game output and fold coefficients,
then review chronology, proper-score baselines and the joint season bootstrap.

## Open
Manifests use canonical local paths; moving inputs requires regeneration.
The historical FanDuel feature artifact is not reproduced. No signal is closed
and no served probability or pick changes in this input-provenance task.
