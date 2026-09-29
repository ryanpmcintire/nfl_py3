# LEAD-74 — within-spread price revisions

## Goal
Execute the frozen LEAD-74 protocol; inventory first and stop if required local sources are absent.

## State
**Measured:** inventory complete; 0 ingestion fields found; no outcomes read. **Inferred:** source-gated by Protocol B's timing rule. Declaration unchanged.

## Declared protocol (verbatim source text; wrapped)
| LEAD-74 | ⬜ | Within-spread price revisions (batch B rank 1 of 8) | **Inferred proposal, 2026-09-29; unmeasured.** Mechanism: books can absorb post-Tuesday information by changing the two
prices while keeping the spread fixed; the pool opener and a spread-point move term miss that update. Predeclare 2023-2025 REG games with timestamped Tuesday/deadline two-sided quotes from the
same book at the same spread. Feature: equal-book median change in no-vig home-cover log odds on those matched quotes; no book selection by results. Add that scalar to the existing calibrated
discrete-margin probability, targeting opener cover probability. Protocol B in `docs/lanes/ideation-2026-09-29b.md`: chronology-purged LOSO, paired combined/model-only/market/Elo baselines,
IS/OOS gap, calibration, season intervals and probability_positive. Budget B=2 specifications (pressure term, joint fit), F=3, E=0; 177 total fitting/reporting looks under the declared counting
rule. **Read/checked:** MKT-03 computes no-vig probabilities; MKT-16..19 fit spread movement; LEAD-67 changes movement units and LEAD-71 adds moneyline information. This tests revisions of prices
at an unchanged spread, not those existing inputs. Units: quote-pair/time audit, 10-15 calls; one cached replay, 20-25. Rank rationale: direct missing market information with a small scalar
addition; historical paired-price coverage must first be verified. |

Rows fix population, terms, target, units and specification budget B. Inventory metadata first; require issuance/ingestion before the pool deadline. Grade authentic frozen openers. Missing
archives are data gaps, never replaced by closes, realized weather or retrospective news.

Chronology-purged LOSO excludes target/later seasons from fitting; separate earlier training, selection and calibration seasons. Folds lacking training remain unavailable. Reuse certified pregame
predictions; choose coefficients/penalties on earlier seasons without new grids. Freeze before prospective evaluation. One calibrated discrete-margin distribution combines information; its cover
probability selects the side.

Pair combined-model, model-only, market-only and Elo baselines. Report IS/OOS opener accuracy, Brier, log loss and margin MAE, their gap, decisive-game record first, fold coefficients, season
stability, five training-quantile reliability bands, season-block 95% intervals and `probability_positive`. Zero crossing closes nothing; unresolved effects remain `unresolved_below_power`.

Count reporting cells too: L = (B + 36 + 8E)(F + 1) + 25 for F scheduled folds, E extra endpoints: B candidate/four baseline specifications, four-comparator IS/OOS metric cells per fold/pooled,
five bands for five models. Identical nested refits add no specification. New splits/specifications need separate preregistration. Correlated cells are not independent evidence.

LEAD-74: B = 2; F = 3; E = 0; L = 177.

## Tried
**Measured:** `.tools/uv.exe run --no-sync python scripts/lead74_unit1.py` ran once (exit 0).
Ruff check and format check passed. Report: `docs/lead74_unit1_inventory.md:6`.
2023/2024/2025 snapshots: 2,629/2,794/2,681; 5,031,980 quote rows; 0 read errors. All are backfills.
IS/OOS metrics, gap, intervals, probability_positive, decisive record, and fold coefficients
are unestimated; descriptive inventory counts have no sampling interval. Looks: 177 planned, 0 used.

## Record commands
None: inventory only, with no fitted effect or research verdict to record. This data gap
does not close the signal. The orchestrator alone runs future statistical record commands.

## Next
Supply verifiable predeadline ingestion provenance, then audit same-book Tuesday/deadline
quote pairs, authentic frozen openers, and prediction coverage under the unchanged declaration.

## Open
Required timing provenance is absent from the audited manifests/quote schemas; paired-price
coverage remains unaudited. Historical requested/snapshot times do not establish ingestion
(`src/nfl_ats/odds_backfill.py:258`). No source substitution, scoring, registry write, or publication.

**Root note 2026-09-29:** our own capture snapshots are directory-stamped (e.g. data/market/raw/20260929T203730Z); a follow-up unit may use the snapshot directory timestamp as ingestion provenance for self-captured quotes (not for third-party archives) and must say so in its declaration.
