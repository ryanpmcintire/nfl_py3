# LEAD-86 ? market training regularizer

## Goal
Execute the declared LEAD-86 unit without changing the served card.

## State
Protocol copied before outcomes or fitting; implementation pending.

### Predeclared protocol (verbatim words; wrapped)
| LEAD-86 | ⬜ | Use the observed market probability as a training regularizer (batch C rank 5 of 8) | **Inferred mechanism:** the deadline market has absorbed news unavailable to
Tuesday's pool line; a cover calibrator trained only on noisy final-game labels can underuse that information even when the observed move coefficient is stable. Predeclare the
unchanged four-term calibrated probability, adding a training penalty toward the timestamp-matched market-lattice cover probability. Candidate penalty weights are exactly 0, 0.1
and 1; choose on the earlier selection season, calibrate separately, then score the outer season. This is a soft constraint on the same probability, never a market override or a
forecast of future movement. **Measured inputs:** `artifacts/pick_probability/20260929T192747Z/per_game.parquet` (1,503 rows); `data/market/raw/20200908T125500Z/quotes.parquet`
(1,862) and `data/market/raw/20200913T162500Z/quotes.parquet` (2,088), within the 262 Tuesday/Sunday files/316,530 rows. The fixed population has 1,309 target-week games with
complete products at both captures. Protocol C: opener Brier primary, F=3, B=8 including all three penalty candidates and market construction, K=4, **505 looks**. **Read/checked:**
MKT-16..20, MOD-20/22, LEAD-66/71/73. LEAD-71 added another predictor; this tests a different loss with unchanged predictor terms. MKT-20's future-move target is not reopened, and
MOD-22's unsuccessful QB terms are not recycled. Units: cached market targets, 15–20 calls; fixed-grid replay, 20–25. Rank rationale: possible variance reduction without another
sparse feature, but the market's RPS gain does not establish a superior cover target. |

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

## Tried
Read the task packet, research rules, named row and lane. No outcomes computed.

Implementation details fixed before scoring: Bernoulli cross-entropy plus lambda times
market-target cross-entropy, ridge 0.001; choose minimum selection-season opener
Brier (ties prefer smaller lambda); do not refit after selection. Fit an intercept
and nonnegative logit slope on the separate calibration season. Use complete common
books, median within-book Tuesday-to-Sunday move, and the alphabetically first
complete Sunday book for the existing LEAD-71 discrete market-lattice construction.
Fit its historical prior only on the fit period, using Tuesday lines/totals; retain
pushes in that prior. Use the same lattice for all arms, changing only conditional
cover mass; model-only uses the archived probability, Elo uses a fitted Elo-difference
and opener logistic baseline. Optimistic IS scores each frozen fold on its fit rows;
the gap is OOS minus IS. Bootstrap 10,000 paired season-stratified week blocks, seed
20260929; five equal-width probability bands. No additional candidate or subgroup.

## Record commands
Pending measured results; orchestrator runs records serially.

## Next
Inventory the declared sources, implement scripts/lead86_unit1.py, and run once.

## Open
Source coverage and results unmeasured.

Upstream cutoff evidence (**read**): src/nfl_ats/clv.py:2184-2208 fits only completed games before the target week and substitutes the frozen opener before prediction. Archive feature selection remains retrospective.
