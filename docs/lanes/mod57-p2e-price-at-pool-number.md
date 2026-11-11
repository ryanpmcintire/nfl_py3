# MOD-57 P2-E - market price at the pool's exact number
Stamped 2026-11-11 09:37 EST (date command, same call as the record).

## Goal
Score the market's own no-vig home-cover price at the pool's exact number (median across books quoting it, repo no_vig_probabilities) on finished 2026 games, no fitted parameter. Arm M = M where it exists else current; Arm B = logit mean of M and current where M exists.

## State (measured; tests/scratch/mod57/p2e/: parse.py, run.py, hist.py, run.out, hist.out, scored.csv, payload_prices.parquet, batch.json)
- Payload markets per book (18,288 NFL book-rows, 46 payloads, 20 books): spread with two-sided juice, moneyline, totals; only extra is period_data on 216 games. Exchanges quote spread: Kalshi 641, Novig 899, Polymarket 1,048 book-rows.
- Games: Weeks 3-9, 104 of 136 harness games (Week 1-2 have no payload). Last scan before deadline: 0.97-0.99 h for early Sunday, 3.98 h for the Sunday 16:00 lock, median 27.2 h for Thursday (7 games). M exists on 66 games (>=3 books on 56). Suppliers: Fliff 55, FanDuel 48, Novig 44, Polymarket 43, Kalshi 43, DraftKings 37. Exchange supplies 44 games; only exchanges on 2.
- Sign check: 2026_03_CIN_PIT pool -3.5 (home_line +3.5 means pool negative), M .523 home, hc True; same convention as harness. Harness reproduced 85-51 (all 136).
- Results on 104 games: current 59-45 (acc .567, ll .6875, Brier .2473); M 57-47 (.548, .6903, .2487); B 59-45 (.567, .6887, .2479). Decisive: M changes 20 picks, 9-11 (current 11-9); B changes 8, 4-4. M minus current accuracy -1.95 pts [-11.3, +7.5] P+ .31; ll P+ .27; Brier P+ .28. B accuracy -0.01 [-3.9, +3.8] P+ .40; ll P+ .31; Brier P+ .31. Five M values are exactly .5 (counted home).
- Quiet (|leader median - pool| <= 0.5): 64 games, M available 59: current 36-28, M 33-31 (acc P+ .23), B 36-28. Moved: 40 games, M available 7: M changes 1 pick (1-0).
- Tuesday (scan nearest noon, lag median 4.0 h, max 29 h): price at pool number on 97 games; leans a side on 83 (home 39, away 44), leaned side 42-41; with >=3 books 34-38 on 72; differs from current on 19, 7-12.
- History 2023-25 (hist.py; FanDuel first quote after Tuesday noon, last hourly snapshot before deadline, median lag 0.57 h): 816 games; >=1 book quotes FanDuel's number 414, >=3 books 297 (FanDuel itself 307). Quiet 498 games: >=1 384, >=3 293. Moved 318: >=1 30, >=3 4. By season >=3: 101, 95, 101.
- 7 cells recorded unresolved_below_power, family mod57_price_at_pool_number, registry/weak_signals/mod57_price_at_pool_number.json.

## Tried
All arms and splits declared before results; no parameter chosen.

## Next
Nothing to serve: the price at the pool's number, on the books that still quote it, is near 50% on almost every game (it exists mostly where nothing moved) and changes picks 9-11. A leave-one-season-out unit would use the 2023-25 table (about 297 games with >=3 books, 98% quiet) and cannot reach the moved games, where the 60% limit sits; it would need a price scraped at the pool number from games that moved (not held in the archive).

## Open
Moved games have no book at the pool's number at the deadline (30 of 318 in history, 7 of 40 in 2026), so the market answer is structurally missing exactly where it would matter.
