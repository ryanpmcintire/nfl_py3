# LEAD-88 protocol source

Copied before inventory, fitting, or scoring from ROADMAP.md:871 and
docs/lanes/ideation-2026-09-29c.md:9-16. Whitespace wrapped only.

## Exact roadmap row

| LEAD-88 | ⬜ | Fit the tiebreaker's response to observed total news (batch C rank 7 of 8) | **Inferred mechanism:** post-Tuesday scoring news changes the likely combined score,
while a Tuesday total stays frozen; the appropriate fraction of the observed total revision need not equal a fixed shade or a rounding adjustment. Predeclare total centre = Tuesday
total + b × (Sunday-12:30 total minus Tuesday total), with one b fitted on earlier REG games under absolute loss and a separately calibrated discrete total law. Keep the margin
probability, sides and Best Pick fixed; choose the integer total by the existing rounding rule. **Measured inputs:** `data/raw/20260908T162105Z/schedules.parquet` (4,902 rows),
`data/market/raw/20200908T125500Z/quotes.parquet` (1,862), `data/market/raw/20200913T162500Z/quotes.parquet` (2,088), within 262 source files/316,530 rows; 101 weekly last games
have same-book pre-deadline total pairs. Fit on all source-complete REG games; last-game MAE is primary, all-game MAE secondary. Protocol C: F=3, B=6, K=6, **713 looks**. Total-
endpoint controls are Tuesday total, deadline total, earlier-season intercept-adjusted deadline total and earlier-season unconditional median; ordinary ATS comparisons retain C's
baselines. **Read/checked:** POL-12, MOD-17, SIM-04/08, LEAD-06/54/80 and `docs/tiebreaker_total_study.md:7-23`. That study used closing totals and static shades; this estimates
pass-through from genuinely timestamped revisions, without field guesses or conditional-tie data. Units: paired-total join, 10–15 calls; one-parameter replay, 15–20. Rank
rationale: cheap incremental tiebreaker precision; MAE is not evidence of a measured pool-rank gain. |

## Shared Protocol C

### Protocol C
Population: 2020–2025 archived frozen openers, row-specific source-complete subsets. Preserve pushes for distributions/nomination; exclude only from conditional cover fitting.
Enforce target-week, observation, bookmaker and kickoff clocks. Historical `tue_open` captures precede noon: available anchors, not noon-capture evidence. LEAD-82 uses the hourly
noon boundary. No closing inputs.

Chronological LOSO: outer 2023/2024/2025; fit through Y−3, tune on Y−2, calibrate on Y−1. Reconstruct timestamp-matched moves for candidate/comparator in every year, replacing
missing-archive zeros. LEAD-82 alone fits 2023, calibrates 2024, scores 2025 with existing ridge; one outer season cannot establish stability. Verify upstream training cutoffs.
Reused archives are retrospective; freeze survivors prospectively.

One calibrated discrete-margin probability selects sides. Pair candidate, current four-term recipe, model-only, timestamp-matched market and Elo. Report decisive records first;
optimistic IS/OOS and gaps, fold coefficients/stability, opener accuracy/Brier/log loss/RPS, five equal-width reliability bands, season/week-block 95% intervals and
`probability_positive`. Zero crossing closes nothing; default `unresolved_below_power`.

Looks: L=(27K+B+4)(F+1)+25. F=outer seasons; B=candidate/nuisance specifications plus four baselines; K=endpoint/population series. Each fold/pooled panel: five arms and four
contrasts × IS/OOS/gap × K, B fit/coefficient summaries, four decisive records; add 25 reliability cells. Identical refits add no specification. K=4 ordinarily; 85/89 add weekly
reward/nominee Brier; 88 adds all-game/last-game total MAE. No unlisted variants.
