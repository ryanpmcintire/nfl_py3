# LEAD-75 — wind-forecast innovation

## Goal
Execute the local forecast-product/issuance inventory; replay only if comparable paired forecasts pass the source gate. Owned files: `scripts/lead75_unit1.py`, `docs/lead75_*.md`, this lane.

## State
**Measured:** inventory complete; source-gated `data_gap`. Nine local archives cover 1,062 population games; Tuesday/deadline overlap is 1,045, same-product pairs 0, eligible pairs 0; ingestion fields appear in 0/9 archives. Report: `docs/lead75_inventory.md`. No outcome was loaded. Declaration below was saved first from ROADMAP.md:858 and Protocol B at docs/lanes/ideation-2026-09-29b.md:11-23.

### Declaration — frozen before outcomes
Mechanism: a wind forecast that deteriorates after Tuesday disadvantages the more passing-dependent team, while the frozen line still prices the earlier forecast. Predeclare 2020-2025 REG open-air games with Tuesday and deadline forecasts from the same forecast product for the same kickoff valid time. One term: wind-speed revision times the home-minus-away prior-game passing-exposure difference; expected home-margin sign is negative. Exposure uses only games available before Tuesday; no realized weather, roof guesses, weather bands or outcome-selected thresholds. Target opener cover probability conditional on the existing market-move term. Protocol B supplies chronology-purged LOSO, four paired baselines, IS/OOS gap, calibration and season uncertainty. B=2 (innovation term, joint fit), F=6, E=0; 291 counted looks. The new input is the matched forecast revision, with its incremental value tested after observed movement. Stop at a data gap if comparable paired forecasts are unavailable.

Rows fix population, terms, target, units and specification budget B. Inventory metadata first; require issuance/ingestion before the pool deadline. Grade authentic frozen openers. Missing archives are data gaps, never replaced by closes, realized weather or retrospective news.

Chronology-purged LOSO excludes target/later seasons from fitting; separate earlier training, selection and calibration seasons. Folds lacking training remain unavailable. Reuse certified pregame predictions; choose coefficients/penalties on earlier seasons without new grids. Freeze before prospective evaluation. One calibrated discrete-margin distribution combines information; its cover probability selects the side.

Pair combined-model, model-only, market-only and Elo baselines. Report IS/OOS opener accuracy, Brier, log loss and margin MAE, their gap, decisive-game record first, fold coefficients, season stability, five training-quantile reliability bands, season-block 95% intervals and `probability_positive`. Zero crossing closes nothing; unresolved effects remain `unresolved_below_power`.

Count reporting cells too: L = (B + 36 + 8E)(F + 1) + 25 for F scheduled folds, E extra endpoints: B candidate/four baseline specifications, four-comparator IS/OOS metric cells per fold/pooled, five bands for five models. Identical nested refits add no specification. New splits/specifications need separate preregistration. Correlated cells are not independent evidence.

## Tried
**Measured:** `.tools/uv.exe run --no-sync python scripts/lead75_unit1.py` ran once, exit 0; scoped `ruff format --check --no-cache` and `ruff check --no-cache` passed. Used offline uv with `UV_CACHE_DIR` in the temporary directory after default-cache access failed; numerical thread caps were 2, parquet reads single-threaded. **Read:** Tuesday is MEX; deadline is GFS. No external fetch, fit, score, registry write, or served change. IS/OOS metrics, gap, coefficients, decisive-game record, intervals and `probability_positive`: not estimated; scored games 0. Planned looks 291; executed outcome looks 0.

## Record commands
None: source inventory supplies no fitted/scored effect or statistical verdict to register. Do not fabricate a weak-signals or rotation command from archive counts; the orchestrator records serially after an admissible replay.

## Next
Orchestrator reviews the saved inventory; next subtask is a source acquisition decision for same-product, same-valid-time Tuesday/deadline forecasts with historical issuance/ingestion evidence, then re-audit before replay. Repository-wide checks and commits remain with the orchestrator.

## Open
**Inferred:** cross-product differences cannot isolate wind innovation. The mechanism remains untested, not closed; missing comparable forecasts and historical ingestion provenance block unit 2 under the row's source gate.
