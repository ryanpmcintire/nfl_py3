# LEAD-88 — observed total news

## Goal
Execute the declared total-response replay in `scripts/lead88_unit1.py`; research only.

## State
**Measured:** paired-total unit complete; 262 verified files / 316,530 rows → 16,720 book pairs, 1,343 games, 101/107 weekly last games. Replay is not complete. No outcomes loaded, fits or scores; 0 of 713 statistical looks executed. Verbatim declaration: `docs/lead88_protocol.md`; results: `docs/lead88_unit1.md`.

## Protocol (frozen before outcomes)
Predeclare total centre = Tuesday total + b × (Sunday-12:30 total minus Tuesday total), with one b fitted on earlier REG games under absolute loss and a separately calibrated discrete total law. Keep the margin probability, sides and Best Pick fixed; choose the integer total by the existing rounding rule. Fit on all source-complete REG games; last-game MAE is primary, all-game MAE secondary. Protocol C: F=3, B=6, K=6, **713 looks**. Total-endpoint controls are Tuesday total, deadline total, earlier-season intercept-adjusted deadline total and earlier-season unconditional median; ordinary ATS comparisons retain C's baselines.

Population: 2020–2025 archived frozen openers, row-specific source-complete subsets. Preserve pushes for distributions/nomination; exclude only from conditional cover fitting. Enforce target-week, observation, bookmaker and kickoff clocks. Historical `tue_open` captures precede noon: available anchors, not noon-capture evidence. No closing inputs.

Chronological LOSO: outer 2023/2024/2025; fit through Y−3, tune on Y−2, calibrate on Y−1. Reconstruct timestamp-matched moves for candidate/comparator in every year, replacing missing-archive zeros. Verify upstream training cutoffs. Reused archives are retrospective; freeze survivors prospectively.

One calibrated discrete-margin probability selects sides. Pair candidate, current four-term recipe, model-only, timestamp-matched market and Elo. Report decisive records first; optimistic IS/OOS and gaps, fold coefficients/stability, opener accuracy/Brier/log loss/RPS, five equal-width reliability bands, season/week-block 95% intervals and `probability_positive`. Zero crossing closes nothing; default `unresolved_below_power`. L=(27K+B+4)(F+1)+25: five arms and four contrasts × IS/OOS/gap × K, B fit/coefficient summaries, four decisive records per fold/pooled panel, plus 25 reliability cells; no unlisted variants. MAE is not evidence of a measured pool-rank gain.

## Tried
**Measured:** `.tools/uv.exe run --no-sync python scripts/lead88_unit1.py` exits 0; verified source hashes and pregame clocks. Corrected an overly conservative date-audit assumption and reran without outcomes. AST/token check: 0 comments/docstrings; recorded script hash matches. Use `UV_CACHE_DIR` in a writable temporary directory. Scratch: `tests/scratch/codex/lead88_unit1/`.

## Record commands
None: no effect estimated or adjudicated; do not invent a record or probability_positive. Future replay commands require `--plain-summary` and orchestrator execution.

## Next
Resolve rounding entry point before the outcome replay: cited study's half-up rule versus current served score lattice. Preserve all pushes, reconstruct the fixed four-term base/lineage, then fit b and the separate discrete law under the frozen folds; report every declared comparison.

## Open
**Measured:** totals exist locally; this is not an absent-total-source gate. 33 paired games lack decisive-only four-term rows; outer folds have 239/238/237 games, each with 18 last games. IS/OOS, gap, coefficients, intervals, probability_positive and decisive records remain uncomputed. Two scope/rounding questions are pending. No closure or serving claim.
