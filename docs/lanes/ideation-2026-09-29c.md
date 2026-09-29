# Eight locally testable opener leads — 2026-09-29

## Goal
Append eight distinct, ranked, accretive Phase 12 leads using existing data only; no experiments or served changes.

## State
**Measured:** LEAD-82..89 saved at ROADMAP.md:865–872; eight unique, contiguous, four-column rows; paths and look arithmetic verified; scoped diff is eight additions and `git diff --check` passes. Protocol C precedes any new fit/score. Rankings are **inferred**, not measured gains.

### Protocol C
Population: 2020–2025 archived frozen openers, row-specific source-complete subsets. Preserve pushes for distributions/nomination; exclude only from conditional cover fitting. Enforce target-week, observation, bookmaker and kickoff clocks. Historical `tue_open` captures precede noon: available anchors, not noon-capture evidence. LEAD-82 uses the hourly noon boundary. No closing inputs.

Chronological LOSO: outer 2023/2024/2025; fit through Y−3, tune on Y−2, calibrate on Y−1. Reconstruct timestamp-matched moves for candidate/comparator in every year, replacing missing-archive zeros. LEAD-82 alone fits 2023, calibrates 2024, scores 2025 with existing ridge; one outer season cannot establish stability. Verify upstream training cutoffs. Reused archives are retrospective; freeze survivors prospectively.

One calibrated discrete-margin probability selects sides. Pair candidate, current four-term recipe, model-only, timestamp-matched market and Elo. Report decisive records first; optimistic IS/OOS and gaps, fold coefficients/stability, opener accuracy/Brier/log loss/RPS, five equal-width reliability bands, season/week-block 95% intervals and `probability_positive`. Zero crossing closes nothing; default `unresolved_below_power`.

Looks: L=(27K+B+4)(F+1)+25. F=outer seasons; B=candidate/nuisance specifications plus four baselines; K=endpoint/population series. Each fold/pooled panel: five arms and four contrasts × IS/OOS/gap × K, B fit/coefficient summaries, four decisive records; add 25 reliability cells. Identical refits add no specification. K=4 ordinarily; 85/89 add weekly reward/nominee Brier; 88 adds all-game/last-game total MAE. No unlisted variants.

## Tried
- **Measured:** metadata/clock-only `.tools/uv.exe run --no-sync python -` inventory: 447 quote files/473,148 rows. Final target-week coverage: 1,321 common-book games, 1,311 with two books, 1,321 complete multi-book Sunday panels, 1,309 paired complete markets, 101 last-game total pairs, 52 postseason pairs, 107 three-stage weeks. Early block: 799 games/2,348 leader-book pairs. Preliminary unaligned counts are superseded. Exact paths/counts are in each row; quote rows are not independent games.
- **Read:** requested sections, both earlier lanes, LEAD-66..81, cited overlaps and idea ledger. Existing movement differences within books; LEAD-82 addresses its Wednesday start.
- **Read:** `docs/lead71_unit2.md:36`: RPS +0.034858 [−0.000318,+0.085864], probability_positive 0.9741, without side gain. `docs/lanes/lead73.md:18`: accuracy −1.663 points [−3.275,−0.067], probability_positive 0.0202; positive move coefficients 6/6. Corrected MOD-22 units 4/5 add no market-conditional benefit; no QB rerun proposed.

## Record commands
None: no fit/score/verdict. Future commands require bash-compatible syntax and `--plain-summary` in pool-player English; orchestrator records serially.

## Next
Orchestrator assigns LEAD-82's clock/feature unit in a fresh thread. Units ≤30 tool calls; one job, n_jobs ≤2.

## Open
Gains unmeasured; no promotion. No external source, field-pick archive, new tests, pipeline rebuild, registry write, publication or Git mutation needed.
