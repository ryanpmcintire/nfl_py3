# LEAD-78: within-week common-opponent news

## Goal
Execute the fixed unit-one protocol without changing the served card.

## State
**Measured:** unit one complete; source gap. Twelve schedule schemas, zero recognized final/completion fields, zero 2020-2025 Splash captures/locks; 1,615 calendar games. No fit or score.

### Predeclared protocol (verbatim words; wrapped for the bounded reader)
| LEAD-78 | ⬜ | Within-week common-opponent news (batch B rank 5 of 8) | **Inferred proposal, 2026-09-29; unmeasured.** Mechanism: a completed Thursday or Saturday game changes what
earlier performances against those teams mean; Tuesday's pool line cannot contain that update for Sunday's teams. Predeclare 2020-2025 REG games with a later pick deadline, preserving
every eligible game rather than selecting surprising early results. Build one score-based common-opponent state update from games publicly final between Tuesday and the target deadline,
holding the earlier-season-trained rating architecture fixed. Feature: updated-minus-Tuesday implied home margin propagated through the already-observed opponent graph; enter it
alongside the existing market move so news already absorbed by books is controlled. No target-game plays, unfinished games or later stat corrections; retain published-final timestamps.
Protocol B supplies chronology-purged LOSO, paired baselines, IS/OOS gap, calibration and season intervals. B=3 (state-update specification, delta term, joint fit), F=6, E=0; 298
counted looks. **Read/checked:** PBP-05, RWB-01, MOD-06/10/21 and LEAD-68. Those cover opponent adjustment, season state, dynamic/graph ratings and seasonal calibration; the new
estimand is the incremental information arriving within the target week, not a replacement rating system or a rerun of closing-line ratings. Units: as-of completion/state-delta audit,
15-20 calls; cached score-only replay, 20-25. Rank rationale: no new external feed and many historical weeks; expected effect is smaller because only a few early games inform each
slate. |

Rows fix population, terms, target, units and specification budget B. Inventory metadata first; require issuance/ingestion before the pool deadline. Grade authentic frozen openers.
Missing archives are data gaps, never replaced by closes, realized weather or retrospective news.

Chronology-purged LOSO excludes target/later seasons from fitting; separate earlier training, selection and calibration seasons. Folds lacking training remain unavailable. Reuse
certified pregame predictions; choose coefficients/penalties on earlier seasons without new grids. Freeze before prospective evaluation. One calibrated discrete-margin distribution
combines information; its cover probability selects the side.

Pair combined-model, model-only, market-only and Elo baselines. Report IS/OOS opener accuracy, Brier, log loss and margin MAE, their gap, decisive-game record first, fold coefficients,
season stability, five training-quantile reliability bands, season-block 95% intervals and `probability_positive`. Zero crossing closes nothing; unresolved effects remain
`unresolved_below_power`.

Count reporting cells too: L = (B + 36 + 8E)(F + 1) + 25 for F scheduled folds, E extra endpoints: B candidate/four baseline specifications, four-comparator IS/OOS metric cells per
fold/pooled, five bands for five models. Identical nested refits add no specification. New splits/specifications need separate preregistration. Correlated cells are not independent
evidence.

LEAD-78 budget: B=3, F=6, E=0; L=298.

## Tried
**Measured:** `.tools/uv.exe run --no-sync python scripts/lead78_unit1.py` ran once, exit 0.
Scoped `ruff format --check` and `ruff check` passed. UV used a temporary cache after the user cache denied access; offline, at most two Arrow workers.
Report: `docs/lead78_unit1.md`; 298 planned looks, zero outcome looks, zero fitted folds/scored games.
IS/OOS metrics/gaps, fold coefficients, decisive W-L-P, 95% intervals and `probability_positive` are unavailable; inventory counts have no sampling interval.

## Record commands
None: metadata-only source gap, no estimate or adjudication. No registry write ran; do not fabricate an effect or closure command.

## Next
Orchestrator: recover deadline-valid historical openers/locks and published-final/ingestion timestamps, then authorize the cached score-only replay under this unchanged declaration.

## Open
**Inferred:** source gap leaves the mechanism unmeasured, not closed. Inventory covers the production schedule/Splash paths and the referenced baseline header; no claim of an exhaustive filesystem audit.
