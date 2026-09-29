# LEAD-79 — earlier fixture lookahead anchor

## Goal
Execute LEAD-79 unit 1: local provenance/coverage inventory; replay only if authentic sources pass the declared gate. No served changes or registry writes.

## State
**Measured:** Unit 1 complete; source gap, no research closure. 8,832 quote files / 6,189,933 rows; 5,941,819 rows fall in target-season calendar dates, but 0 authentic pool-capture files for 2020-2025 in the canonical store. Four capture files cover 2026 only. No fit or score. Protocol below was copied before execution; report: docs/lead79_inventory.md.

### Declaration (fixed before outcomes)
Mechanism: Tuesday's opener may overreact to one noisy intervening game relative to the market's earlier assessment of that exact matchup. Predeclare 2020-2025 REG games with a dated lookahead quote published before either team's intervening game, plus the authentic Tuesday pool line. One feature is lookahead-implied home margin minus Tuesday-implied home margin; fit its coefficient alongside the current model, availability terms and observed pre-deadline movement. No proxy assembled from other games' closing lines, no largest-revision screen and no automatic fade. Target opener cover probability under Protocol B's chronological LOSO, four paired baselines, IS/OOS gap, calibration and uncertainty. B=2 (anchor term, joint fit), F=6, E=0; 291 counted looks.

Rows fix population, terms, target, units and specification budget B. Inventory metadata first; require issuance/ingestion before the pool deadline. Grade authentic frozen openers. Missing archives are data gaps, never replaced by closes, realized weather or retrospective news.

Chronology-purged LOSO excludes target/later seasons from fitting; separate earlier training, selection and calibration seasons. Folds lacking training remain unavailable. Reuse certified pregame predictions; choose coefficients/penalties on earlier seasons without new grids. Freeze before prospective evaluation. One calibrated discrete-margin distribution combines information; its cover probability selects the side.

Pair combined-model, model-only, market-only and Elo baselines. Report IS/OOS opener accuracy, Brier, log loss and margin MAE, their gap, decisive-game record first, fold coefficients, season stability, five training-quantile reliability bands, season-block 95% intervals and `probability_positive`. Zero crossing closes nothing; unresolved effects remain `unresolved_below_power`.

Count reporting cells too: L = (B + 36 + 8E)(F + 1) + 25 for F scheduled folds, E extra endpoints: B candidate/four baseline specifications, four-comparator IS/OOS metric cells per fold/pooled, five bands for five models. Identical nested refits add no specification. New splits/specifications need separate preregistration. Correlated cells are not independent evidence.

## Tried
**Measured:** `UV_CACHE_DIR=.uv-cache .tools/uv.exe run --no-sync python scripts/lead79_unit1.py` ran once, exit 0; 0 read errors. Scoped `ruff check`, `ruff format --check`, and `git diff --check` passed. Identity/timestamp fields only, Arrow threads capped at 2. IS/OOS metrics and gap, 95% intervals, `probability_positive`, decisive-game record and all six fold coefficients are unavailable: 0 games scored, 0 outcome looks used (291 planned).

## Record commands
None: inventory only, with no fitted effect or research verdict to record. No registry command was run; do not invent a numerical record for missing sources.

## Next
Orchestrator: source authentic 2020-2025 Tuesday pool captures, then certify same-fixture quote issuance before both intervening games and ingestion before the deadline; only then run the predeclared paired cached replay. No commit or publication by this worker.

## Open
**Measured:** canonical target-season pool coverage is absent. **Unverified:** lookahead timing/ingestion and other unnamed custom stores; generic historical quotes are not certified lookahead quotes. Three lookahead-named files are existing screen result artifacts; no outcomes from them were read. No effect estimate, interval, or closure claim exists.
