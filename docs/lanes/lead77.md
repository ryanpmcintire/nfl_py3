# LEAD-77 — quote-freshness reliability

## Goal
Execute the declared LEAD-77 unit with local sources only; no served changes.

## State
**Measured:** inventory complete, exit 0; NFL 4,421,224 / CFB 609,952 target quote rows. All 8,104 target manifests are historical backfills; no ingestion-time fields.
**Inferred:** required pre-deadline provenance is missing, so inventory only (Protocol B:12); mechanism remains open.

## Protocol — copied before outcomes
| LEAD-77 | ⬜ | Learn quote-freshness reliability from price synchronization (batch B rank 4 of 8) | **Inferred proposal, 2026-09-29; unmeasured.** Mechanism: after news, an
asynchronous book median can lag the current information set; a recent disagreement and an old unrefreshed quote should carry different evidence against the frozen pool line.
Predeclare NFL 2023-2025 opener games and timestamped NFL/CFB quote panels from those seasons. Fit a continuous correction from quote age and its probability-distance from the
freshest observed quote to the next synchronized consensus, with training labels themselves occurring before that game's deadline. Compare NFL-only response fitting with a
predeclared partially pooled NFL+CFB response fit; the pooled correction is the ATS challenger, not the better-looking arm chosen afterward. Exclude the held-out NFL season from
both leagues; normalize units on training seasons, weight each game equally across quote rows and cluster by game/season. Snapshots raise precision of the price-response estimate,
never the independent ATS game count. Add the correction through one fitted term in the calibrated margin distribution. Protocol B, B=4 (two response fits, correction term, joint
fit), F=3, E=1 for response MAE; 217 counted looks, paired baselines and IS/OOS reporting. **Read/checked:** SKY-04 and MKT-05/15 cover staleness/leadership; XLG-09 pools final
opener errors; MOD-20 uses move availability; LEAD-70/73 concern model disagreement/history. This learns quote-age-dependent response reliability rather than another book ranking.
Units: timestamp/cadence inventory, 15-20 calls; capped panel fit and replay, 25-30. Rank rationale: reusable quote observations and a compatible cross-league mechanism, but
provider timestamps must first prove usable. |

### Protocol B — declared before any new outcomes
Rows fix population, terms, target, units and specification budget B. Inventory metadata first; require issuance/ingestion before the pool deadline. Grade authentic frozen openers.
Missing archives are data gaps, never replaced by closes, realized weather or retrospective news.

Chronology-purged LOSO excludes target/later seasons from fitting; separate earlier training, selection and calibration seasons. Folds lacking training remain unavailable. Reuse
certified pregame predictions; choose coefficients/penalties on earlier seasons without new grids. Freeze before prospective evaluation. One calibrated discrete-margin distribution
combines information; its cover probability selects the side.

Pair combined-model, model-only, market-only and Elo baselines. Report IS/OOS opener accuracy, Brier, log loss and margin MAE, their gap, decisive-game record first, fold
coefficients, season stability, five training-quantile reliability bands, season-block 95% intervals and `probability_positive`. Zero crossing closes nothing; unresolved effects
remain `unresolved_below_power`.

Count reporting cells too: L = (B + 36 + 8E)(F + 1) + 25 for F scheduled folds, E extra endpoints: B candidate/four baseline specifications, four-comparator IS/OOS metric cells per
fold/pooled, five bands for five models. Identical nested refits add no specification. New splits/specifications need separate preregistration. Correlated cells are not independent
evidence.

| LEAD-77 | B=4 / F=3 / E=1 | L=217 |

## Tried
**Measured:** ran `.tools/uv.exe run --no-sync python scripts/lead77_unit1.py` once (exit 0), scoped Ruff format/check, and a metadata-only `python -` audit. Used `UV_NO_CACHE=1`
after cache access failed. Two-thread cap; no fit, score, served changes or registry writes. Report: `docs/lead77_unit1_inventory.md`.
**Measured:** decisive record, IS/OOS metrics/gaps, all 2023/2024/2025 fold coefficients, intervals and probability_positive are unestimated. NFL cadence medians: 3,600 seconds
each year; CFB: 86,400 / 28,799 / 61,200 seconds. Declared looks 217; consumed outcome looks 0.

## Record commands
None: source inventory only, no measured effect or verdict. An effect/interval/probability_positive command would invent evidence. The orchestrator records serially after valid
scoring, with `--plain-summary` in pool-player English.

## Next
Orchestrator: verify historical ingestion provenance or authorize a prospective source protocol; certify opener/deadline joins and earlier training/selection/calibration seasons
before the capped fit/replay unit.

## Open
Quote rows exist; pre-deadline ingestion and synchronized-label eligibility remain unproven. No closure or serving decision. Global repository checks remain with the orchestrator;
no commits or publication by this worker.
