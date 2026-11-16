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
- Ruff style warnings remain in scripts/pool_number_price.py (open() without context, zip strict, E402).
