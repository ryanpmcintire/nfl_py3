# LEAD-94: last-game tiebreaker population

## Goal
Execute the unchanged predeclared LEAD-94 protocol. No served-card or registry changes.

## State
Protocol copied before any outcomes are computed; source and upstream-clock inventory pending.

## Protocol (verbatim declaration)
| LEAD-94 | ⬜ | Train tiebreaker totals for the last-game population (rank 5/8) | **Inferred mechanism:** the week's final primetime matchup is selected; average post-Tuesday scoring-news response across all games may misprice its total. Retain LEAD-88's total-response equation; weight training loss by a class-prior-corrected density ratio for last-game membership from Tuesday total, absolute opener and kickoff slot, capped at 10 and fitted on training seasons only. No outcome enters weights. **Measured inputs:** `data/raw/20260908T162105Z/schedules.parquet` 4,902 rows; `data/market/raw/20200908T125500Z/quotes.parquet` 1,862 and `data/market/raw/20200913T162500Z/quotes.parquet` 2,088. D: last-game MAE primary, B=7/K=6, **717 looks**. **Read/checked:** POL-12, LEAD-54/80/88, MOD-17, SIM-04/08, tiebreaker study. Changes population weights, not rounding, another pass-through term, field assumptions or ATS predictors. Rank reason: cheap replay; report effective sample size, no pool-rank claim. Units 15–20/20–25 calls. |
Population: 2020–2025 archived frozen openers, row-specific source-complete subsets. Preserve pushes for distributions/nomination; exclude only from conditional cover fitting. Enforce target-week, observation, bookmaker and kickoff clocks. Historical `tue_open` captures precede noon: available anchors, not noon-capture evidence. LEAD-82 uses the hourly noon boundary. No closing inputs.
Chronological LOSO: outer 2023/2024/2025; fit through Y−3, tune on Y−2, calibrate on Y−1. Reconstruct timestamp-matched moves for candidate/comparator in every year, replacing missing-archive zeros. LEAD-82 alone fits 2023, calibrates 2024, scores 2025 with existing ridge; one outer season cannot establish stability. Verify upstream training cutoffs. Reused archives are retrospective; freeze survivors prospectively.
One calibrated discrete-margin probability selects sides. Pair candidate, current four-term recipe, model-only, timestamp-matched market and Elo. Report decisive records first; optimistic IS/OOS and gaps, fold coefficients/stability, opener accuracy/Brier/log loss/RPS, five equal-width reliability bands, season/week-block 95% intervals and `probability_positive`. Zero crossing closes nothing; default `unresolved_below_power`.
Looks: L=(27K+B+4)(F+1)+25. F=outer seasons; B=candidate/nuisance specifications plus four baselines; K=endpoint/population series. Each fold/pooled panel: five arms and four contrasts × IS/OOS/gap × K, B fit/coefficient summaries, four decisive records; add 25 reliability cells. Identical refits add no specification. K=4 ordinarily; 85/89 add weekly reward/nominee Brier; 88 adds all-game/last-game total MAE. No unlisted variants.
For 90–97 use C's chronology, five controls/arms, reporting and look formula with F=3; historical OPENER is the frozen Splash-line proxy, never pre-2026 pool captures. Shared **measured** inputs: `artifacts/pick_probability/20260929T192747Z/per_game.parquet` (1,503 rows/107 weeks), `artifacts/opener_evaluation/20260929T192743Z/per_game.parquet` (1,537 including pushes), `data/processed/game_features_weak_stack.parquet` (4,902). Reuse the upstream cutoff/refit machinery documented in `docs/lanes/lead83.md:14–19`; a downstream LOSO label alone does not certify upstream chronology. Fixed ridge .001, no tuning grid; selection and calibration years remain separate. Each row is an independent challenger, not an eight-way winner-selection exercise. All features/quotes are pre-deadline; historical scores/plays are training labels only. One calibrated line-conditioned PMF decides sides, with the served four-term recipe as the paired base. K=4 normally; 90/91/93/95 add local RPS/worst-season loss/tail RPS/regulation log score respectively; 92 adds nominee reward/Brier; 94 adds all-game/last-game total MAE. Tiebreaker controls follow C/LEAD-88; tiebreaker changes keep ATS sides fixed. All population filters are outcome-blind, all IS/OOS gaps and diagnostic looks reported. Zero crossing closes nothing; RPS or tiebreaker gains alone cannot promote ATS sides (AGENTS.md, Margins/Promotion). Freeze survivors for subsequent normal 2026 capture collection.

## Tried
Read the named roadmap row and protocol context. No fit or score yet.

## Record commands
Pending measured replay; orchestrator runs commands serially.

## Next
Inspect LEAD-88 total equation and LEAD-83 chronology; inventory, implement, run once, lint and format.

## Open
Source availability and upstream chronology unverified. Zero crossing closes nothing; one fitted probability selects each ATS side, which this totals challenger holds fixed.
