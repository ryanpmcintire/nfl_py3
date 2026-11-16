# pool-number-price-2026-11-16 (goal-65 wave G1 slot 5)

## Goal
Market probability at the pool's exact number (FanDuel's frozen Tuesday number) at each pick deadline: free per-number sources, 2026 coverage, a forward capture, and a graded fitted term on the served card.

## State (measured 2026-11-16, UTC ~22:40)
- Sources: Bovada single-event v2 endpoint (services/sports/event/v2/events/A/description/<link>) has an "Alternate Lines" two-sided spread ladder; robots allows all; policy entry bovada_alternate_spreads added (1.0 s delay, stop on 403/429/5xx). Kalshi KXNFLSPREAD ladders ("wins by over s.5", about 25 strikes per game) with public hourly candles (policy kalshi_public_market_data). Odds Gap payloads: each book's main number only. Bovada coupon path returns {}.
- scripts/pool_number_price.py: `kalshi-backfill`, `grade`, `capture`, `slots`. Deadline = min(kickoff, Sunday 16:00 ET), checked 136/136 against the harness; sign check (result > pool number) 120/120 against replay home_covered.
- Live capture probe: data/external/alt_spreads/live/20261116T223519Z: Bovada 12 games, 285 numbers; Kalshi 13 games, 338 two-sided strikes. Proposed scheduler argv `uv run --no-sync python scripts/pool_number_price.py capture`, slots thu 19:30, sat 12:00, sun 09:15, sun 12:50, sun 15:45 (orchestrator owns scheduler edits).
- Kalshi backfill data/external/alt_spreads/kalshi_backfill/20261116T223743Z: 238 bracketing markets, 118 games, 252 requests.
- Coverage W3-10 (118 games): exact-number book quote 75; lattice-translated books 118; Kalshi 118. Translated books vs exact median mean abs diff .006 (75 games); books vs Kalshi .011.
- Grade (artifacts/pool_number_price/20261116T223931Z, declared below before scoring): P0 probability alone W3-10 66-51 v card 52-65; changed 36 picks, 25-11; accuracy +12.0 [2.5, 21.6] P+ .99; log loss P+ .999; Brier P+ .999. W7-10 32-24 v 23-33, changed 11, 10-1. F: beta 1.30 fit W3-6 (in-sample 33-28 v 29-32); scored W7-10 26-30 v 23-33, changed 3 (3-0), accuracy P+ .997 but log loss P+ .21, Brier P+ .28; beta refit on W7-10 is -0.17 (unstable). 6 cells recorded unresolved_below_power, family pool_number_price_2026; pool ran clean.

## Declaration (stamped 2026-11-16 22:35 UTC, before scoring)
M per venue = last quote at or before the deadline (Odds Gap books at own number; Kalshi bracketing strikes, two-sided hourly mid), translated to the pool number with fit_conditional_margin (default, seasons >= season-4, centred on market line, centre solved to reproduce the quote); Kalshi = median of strikes; M = median of venues. Arms: P0 M replaces card; F logit(card) + beta*logit(M), beta from W3-6, scored W7-10.

## Tried
- First backfill used a probability column as deadline; killed, fixed, rerun.

## Next
- The card itself is 52-65 on these games; compare M against the replay of the current formula (harness 85-51) and the market-move term before any serving decision; add a LOSO 2023-25 version with the Kalshi ladder history (KXNFLSPREAD from 2025).
- Orchestrator: schedule `capture` at the slots above.

## Open
- Ruff clean (ruff check passes 2026-11-16). history_rows() execs tests/scratch/mod57/u8_run.py up to boot(); the history depends on that scratch file and its u1 pickles.

## Decisive test (declared 2026-11-16 22:44 UTC, before scoring; nothing served changes)
Coverage reported first. M = the P0 market probability at the pool number (lattice translation, unchanged from the declaration above). Arms are scored as declared challengers; no side flips by rule.
- (1) 2026 W3-10, 118 games, same deadlines, graded at the pool line (result > pool number, pushes out): P0 (M, current where M missing) vs the current-formula replay p_current from artifacts/week_attribution/2026_season.csv (active coefficients 20261116T171555Z incl. counted flags, move, hook). Secondary: vs the u5 harness replay (tests/scratch/mod57/u5/score.py harness(0.0), W3-9). Records, changed picks with their record, log loss, Brier, week-block bootstrap, probability_positive. Cells family pool_number_price_2026.
- (2) History: every season the Kalshi KXNFLSPREAD ladder covers (public historical markets and hourly candles; earliest market reported). M_K = median over bracketing strikes of the last two-sided hourly mid at or before min(kickoff, Sun 16:00 ET), translated with the same lattice to the FanDuel-rebuilt pool rows of mod57 unit 8 (FD Tuesday post-noon first quote; whole FD lines split into the two half-point rows with weight 0.5; pushes kept as in the keep_pushes population). Baseline = served S: unit-8 arm S two-stage out-of-season probability (LOSO hook coefficient). P0_hist = M_K (S where missing). Weighted records, changed picks, log loss, Brier, season-block and week-block bootstrap, probability_positive. New family pool_number_price_history.
- (3) Blend B = sig((1-w) logit(base) + w logit(M)), one w in [0,1] chosen by log loss leave-one-season-out on the history in (2) only. If the history covers a single season, LOSO inside it is undefined: w is fit on the whole history and scored only out of season on 2026 W3-10 (base = p_current); inside the history a leave-one-half-season-out cross-fit (weeks <= 9 v later) is reported as the history score.
- Looks: (1) 3 metrics x 2 baselines + blend 3; (2) 3 metrics; (3) history 3. Family counts stated in the results.
### Results (1), measured 2026-11-16 22:47 UTC: `scripts/pool_number_price.py decisive`, artifacts/pool_number_price/decisive_20261116T224723Z
- Coverage: 117 graded W3-10 games (1 open/push out), M on 117, p_current on 117 (matches the attribution's own ok_current 70-47; spread_line = pool number on 117/117), u5 replay on 104 (W3-9).
- P0 v current formula W3-10: 66-51 v 70-47; changed 30 picks, P0 13-17. Accuracy -3.4 pts [-13.6, +7.9] P+ .26; log loss .6864 v .6766 (P+ .07); Brier .2466 v .2418 (P+ .07); 8 week blocks.
- P0 v u5 harness W3-9: 57-47 v 59-45; changed 20, P0 9-11. Accuracy -1.9 pts [-10.5, +5.8] P+ .29; log loss P+ .53; Brier P+ .54.
- Reading (inferred): the 66-51 v 52-65 gap was against the old played card; against the current formula the price at the pool number loses picks and calibration on the same games. Not a serving candidate on 2026.
### Results (2)-(3), measured 2026-11-16 22:53 UTC: `decisive --history`, artifacts/pool_number_price/decisive_20261116T225202Z; ladder data/external/alt_spreads/kalshi_history/20261116T224816Z (606 markets, 621 requests)
- Coverage: KXNFLSPREAD earliest market opened 2025-08-19, so 2025 is the only history season. 271 of 271 FanDuel-rebuilt 2025 games (310 rows, whole FD lines split into two rows at weight 0.5) have a Kalshi price; 2020-24 have none.
- Kalshi price v served S, 2025: 147.5-123.5 v 150-121; changed 82.5 weighted picks, Kalshi 40-42.5. Accuracy -0.9 pts [-6.6, +5.2] P+ .36; log loss .6901 v .6809, difference [+.0014, +.0173] worse, P+ .012; Brier [+.0006, +.0084] worse, P+ .013; 18 week blocks. With one season, no season-block bootstrap.
- Blend: one season, so the declared fallback applies. Half-season cross-fit weights 0.000006 and 0.000006; whole-history weight 0.000 (at the boundary). The blend equals S in 2025 and p_current in 2026 (70-47), changing no picks.
- Recorded: 9 cells unresolved_below_power in pool_number_price_2026; 4 unresolved and 2 refuted_mechanism (wrong_sign_resolved: Kalshi v S log loss and Brier, whole interval wrong side) in pool_number_price_history. pool ran clean. Looks: 2026 family 9, history family 6.
- Verdict (inferred from the measured numbers above): the parameter-free price at the pool number does not beat the current formula on 2026 (P+ .26 accuracy, .07 log loss). It is worse calibrated than served S on 2025 (wrong sign resolved), and the one-weight blend puts zero weight on it. Do not serve it as a replacement or a term. The capture argv stays useful as a forward data log only.
## Next (updated)
- Nothing here changes the card. If the capture is scheduled, re-grade at the end of the 2026 season against p_current with the 2025 result frozen.
