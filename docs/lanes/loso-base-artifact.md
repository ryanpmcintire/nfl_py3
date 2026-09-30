# Upstream season-held-out base artifact

## Goal
Build `scripts/build_loso_upstream.py` and a reusable upstream/calibration LOSO artifact for LEAD-83/87/89; document loading in `docs/loso_base_artifact.md`.

## State
**Measured:** complete. Saved `artifacts/loso_upstream/20260929T235904133782Z/{predictions.parquet,metadata.json}`: 1,537 rows, 34 pushes, 1,503 decisive games; zero season overlaps at every refitted stage. Accuracy 865/1,503 = 57.552% [56.223,58.721]; log loss 0.681819 [0.678980,0.684150]. Served comparator: 57.418% [56.254,58.784], log loss 0.682774 [0.680279,0.685145].

## Protocol
Population: frozen active opener evaluation, regular seasons 2020–2025; predict every opener game, including pushes, and score/calibrate only decisive games. Historical opener is the frozen pool-line proxy. Target: home covers that opener. Ridge uses all completed regular-season rows through 2025 in the active hashed feature table except excluded seasons; preserve active features/profile/alpha and production residual split. Six outer folds exclude each prediction season from all fitted stages; calibration inputs also exclude their own season through inner cross-fitting. Production discrete line-conditional mapping and location correction exclude held-out seasons, with historical cutoffs retained. Refit any learned composition cut points without excluded seasons; retain point-in-time covariates and fixed served rules. Frozen features are not rebuilt.

Four terms: model logit, signed composition sum, market move, availability; served ridge 0.001/50 steps, training-only standardization, side at probability 0.5. No tuning. Diagnostics: accuracy, log loss, Brier; pooled calibration-training and held-out gap; served four-term, raw model, even-market comparators; six seasons and fixed reliability bins [0,.40,.45,.50,.55,.60,1]. Season-block bootstrap: 10,000 draws, seed 20260929, 95% intervals and probability_positive for paired gains. Four arms × (overall + six seasons + six reliability cells) = 52 descriptive looks, plus one training summary; no selection or closure. Zero crossing closes nothing. LOSO may train on later seasons; this is not a prospective or untouched outer test. Sunday movement is not a Tuesday-availability guarantee.

## Tried
**Measured:** `.tools/uv.exe run --no-sync python scripts/build_loso_upstream.py` exited 0 with cache `tests/scratch/codex/loso_upstream_uv_cache`; log `tests/scratch/codex/loso_upstream_run.log`. Both `ruff check scripts/build_loso_upstream.py` and `ruff format --check scripts/build_loso_upstream.py` via `.tools/uv.exe run --no-sync` passed. Hash/row/mass/fold readback passed; coefficient reconstruction error 1.11e-16. Used 41 margin calls (82 ridge fits), six calibrations, 642 five-bucket location fits, 1,284 protection quartile-pair calculations. Pooled training: 57.312%, log loss 0.680746; held-out gaps +0.240 points/+0.001073. Paired accuracy gain +0.133 points [-1.361,+1.563], probability_positive 0.55925. Loader, season stability, calibration table, coefficients, and limits saved in `docs/loso_base_artifact.md`.

## Next
Orchestrator: consume the pinned upstream artifact for LEAD-83/87/89, verify stage-specific timing, and fit each consumer's additional terms inside its own folds. No registry record is needed for this baseline.

## Open
**Inferred:** frozen feature-builder learned inputs and historical feature selection were not refit. Same-season pregame rolling covariates remain; Sunday movement does not establish Tuesday/Thursday availability. This retrospective LOSO does not clear those separate gates. No jobs are in flight; no source modules, tests, served-card files, commits, or pushes changed.

## Record commands
None: baseline construction and sanity comparison only, as assigned.
