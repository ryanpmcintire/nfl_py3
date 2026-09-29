# Shared season-held-out base artifact

## Goal
Build a reusable 2020–2025 four-term base for LEAD-83/87/89 without same-season calibration training.

## State
**Measured:** saved `artifacts/loso_base/20260929T232711656489Z/{predictions.parquet,metadata.json}`: 1,503 games, 34 pushes excluded, zero shared calibration seasons. Accuracy 57.418% [56.254,58.784], log loss 0.682774 [0.680279,0.685145]; exact served-LOSO agreement, probability_positive=0.5. Training: 57.512%, 0.681732; gaps −0.093 points, +0.001042. Protocol below preceded replay outcomes.

## Protocol
Population: every eligible 2020–2025 regular-season game in the active served fitter's frozen opener input assembly (`per_game.parquet`); preserve exclusions and coverage explicitly, with no historical pool captures or feature rebuild. Target: home covers the historical opener, conditional on no push. Terms: served model logit, move toward home, move-available flag, signed composition sum, intercept; reuse served ridge, optimizer and train-fold standardization. Six leave-one-season-out folds train only on the other five seasons; assert disjoint training and prediction seasons at runtime. No tuning or added signals. Retrospective LOSO may train on later seasons; upstream model inputs retain their original provenance.

Metrics: held-out and pooled fold-training accuracy, log loss and Brier, their gaps, same-row served `out_of_season_home_probability` evaluation, raw model and even-market baselines, six season breakdowns, and fixed home-probability reliability bins [0,.40,.45,.50,.55,.60,1]. Market 0.5 has no directional pick. Uncertainty: 10,000 paired whole-season bootstrap draws, seed 20260929, fixed predictions, 95% percentile intervals; improvement probability_positive = P(gain>0)+0.5P(gain=0). One baseline construction, six fits, no signal selection; 162 declared diagnostic looks (6 fits + 12 overall metric cells + 72 season cells + 24 reliability bins + 9 training/gap cells + 9 paired contrasts + 30 coefficient cells), including reserved market-accuracy cells reported as null. Baseline sanity checks do not close or promote signals and require no registry record.

## Tried
**Measured:** `.tools/uv.exe run --no-sync python scripts/build_loso_base.py --source-artifact artifacts/pick_probability/20260929T192747Z` exited 0; ruff and consumer loader passed. Used `UV_CACHE_DIR=tests/scratch/codex/loso_base_uv_cache`. Log: `tests/scratch/codex/build_loso_base.log`; consumer contract: `docs/loso_base_artifact.md`. No serving or registry changes.

## Next
Orchestrator: build compatible season-held-out upstream model logits, verify opener lineage/stage timing, then recalibrate and rerun affected units.

## Open
**Read:** `docs/lead89_unit1.md:65` reports 1,443/1,537 upstream cutoff violations. **Inferred:** this calibration refit cannot clear that gate or supply stage-matched/push rows. Compatible upstream season-held-out forecasts remain missing; nothing is in flight.

## Record commands
None: baseline construction and sanity comparison only, as assigned.
