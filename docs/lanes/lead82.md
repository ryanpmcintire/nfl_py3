# LEAD-82 — Restore the post-freeze Tuesday move block

## Goal
Execute ROADMAP.md LEAD-82 using `scripts/lead82_unit1.py`; research only, owned files only.

## State
Unit 1 boundary/clock extraction complete; LEAD-82 fitted replay remains open. **Measured:** all 799 frozen-fit games in 2023–2025 have both blocks (266/266/267), with 2,349 hourly same-book boundary pairs and zero clock failures. Report: `docs/lead82_unit1.md:18-33`; outcome-free rows, source hashes and diagnostics: `tests/scratch/codex/lead82_unit1/`.

### Predeclared protocol (copied from row and Protocol C)
Predeclare the median signed change from the last quote at/before Tuesday noon Eastern to the last quote before Wednesday 00:00 Eastern, using the existing three leader books; add that early block beside the existing Wednesday-to-decision move in one calibrated probability. Target opener cover; primary Brier, with accuracy/log loss/RPS. F=1: fit 2023, calibrate 2024, outer 2025; fixed existing ridge, B=6, K=4, **261 looks**. One outer year cannot establish season consistency.

Population: 2020–2025 archived frozen openers, row-specific source-complete subsets. Preserve pushes for distributions/nomination; exclude only from conditional cover fitting. Enforce target-week, observation, bookmaker and kickoff clocks. Historical `tue_open` captures precede noon: available anchors, not noon-capture evidence. LEAD-82 uses the hourly noon boundary. No closing inputs.

Chronological LOSO: outer 2023/2024/2025; fit through Y−3, tune on Y−2, calibrate on Y−1. Reconstruct timestamp-matched moves for candidate/comparator in every year, replacing missing-archive zeros. LEAD-82 alone fits 2023, calibrates 2024, scores 2025 with existing ridge; one outer season cannot establish stability. Verify upstream training cutoffs. Reused archives are retrospective; freeze survivors prospectively.

One calibrated discrete-margin probability selects sides. Pair candidate, current four-term recipe, model-only, timestamp-matched market and Elo. Report decisive records first; optimistic IS/OOS and gaps, fold coefficients/stability, opener accuracy/Brier/log loss/RPS, five equal-width reliability bands, season/week-block 95% intervals and `probability_positive`. Zero crossing closes nothing; default `unresolved_below_power`.

Looks: L=(27K+B+4)(F+1)+25. F=outer seasons; B=candidate/nuisance specifications plus four baselines; K=endpoint/population series. Each fold/pooled panel: five arms and four contrasts × IS/OOS/gap × K, B fit/coefficient summaries, four decisive records; add 25 reliability cells. Identical refits add no specification. K=4 ordinarily; 85/89 add weekly reward/nominee Brier; 88 adds all-game/last-game total MAE. No unlisted variants.

## Tried
**Measured:** `.tools/uv.exe run --no-sync python scripts/lead82_unit1.py` exited 0; three outcome-free executions corrected an overly strict kickoff-equality check, then restricted Tuesday boundaries to the specified hourly source. `UV_CACHE_DIR` used the writable system temp directory after default-cache access failed; all compute pools capped at two. Syntax/comment/docstring/whitespace review passed. No tests added, fits, scores, registry writes, operational jobs or Git mutations.

## Record commands
None for this outcome-free extraction: no effect estimate or research verdict exists to register. Fitted-replay commands must be bash-compatible, include `--plain-summary` in pool-player English, and be run serially by the orchestrator only.

## Next
Fresh thread: reconcile the extra book pair versus the row's preflight 2,348 and audit 164 rebuilt-later-feature differences (maximum 2.5 points; `late_move_audit.json`). Verify upstream model/discrete-distribution training cutoffs, then run the declared cached fitted replay; both arms must use the rebuilt later series.

## Open
**Measured:** 0/261 outcome looks, no IS/OOS estimates/gaps, coefficients, decisive records, intervals or `probability_positive`; inventory counts have no sampling interval. All 1,537 opener rows retain pushes; no hourly Tuesday coverage in 2020–2022. Upstream cutoffs unverified; one outer year cannot establish stability. No research closure or serving decision. Launcher workaround remains owner-reported in `docs/lanes/windows-shell-popup.md`; no launcher repair claimed.
