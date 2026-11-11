# MOD-57 u7 - pool source book from raw Odds Gap payloads
Stamped 2026-11-11 09:06 EST.

## Goal
Identify the book whose line the pool freezes, weeks 3-10, using all-book response.json from 46 odds-gap-private captures. Measurement only, nothing fitted, no cover rates scored.

## State
- Parsed 17,500 book-lines, 46 snapshots, 20 books, to tests/scratch/mod57/u7_table.parquet (u7_parse.py); analysis u7_an.py (out u7_an.out), u7_an2.py.
- Sign check (measured): wk3 PHI@CHI pool -3.5, FanDuel -3.5 at pool capture; wk10 BUF/NYJ pool -7.5 = FD -7.5. Same convention.
- Tuesday coverage (measured): only 9 scans on Tuesday ET (12h x2, 14h x4, 16h x2, 18h x1). Scan nearest noon: wk3 +4.0h, wk4 +4.0, wk5 +29.0 (poor), wk6 +6.0, wk7 +2.0, wk8 +2.1, wk9 +0.01, wk10 +0.03. No scan between Sunday and noon except wk9, wk10: the last-before-noon scan is ~48h earlier (Sunday), so only wk10 has 13 games in that rule.
- Near-noon rule, 8 books on 101 common games: FanDuel 77 exact (76%), DraftKings 60, median 58, Caesars 54, BetRivers 46, Bally/BetParx 45, BetOnline/LowVig 39. Within half a point: all 86-91 of 101.
- Exchanges/Fliff (lines always .5): Fliff 68/85, Kalshi 66, Novig 66, Polymarket 65, FanDuel 62, DK 48 on 85 common games. Not preferred: half-point lines match a half-point pool trivially.
- Weeks 9-10 only (scans within 2 min of noon): FanDuel 12/12 common games (wk9 10/15, wk10 14/14), DK 9/12 (10/15, 11/14), median 7/12.
- FD vs DK where they differ (near rule): 29 games, pool equals FD 22, DK 5, neither 2. First-after rule same. Last-before rule (Sunday scans): 37 differ, FD 16, DK 6.
- Half-point step, FanDuel whole-number lines: 19 games, pool = line 0 times, pool half off 19 times. Half to underdog 12 / favourite 7; to away 13 / home 6; next FD move: none 13, away from pool 4, toward pool 2.
- Full-point-off FanDuel games: 10 (wk3 PHI@CHI, wk5 x7, wk8 x2). In 7 of 10 pool is 1 point toward FanDuel's earlier or neighbouring books or fliff/kalshi; in wk3 PHI@CHI FD later hit -3.5 = pool at the pool capture time (20:19), so timing, not a different book. Pool captures for weeks 3-8 are 15:14-20:19 ET, after the noon scan.
- Looks: 21 books x 3 time rules = 63 pooled cells (504 with weeks), plus FD-vs-DK 3, half-point 2 books.

## Tried
Nearest, last-before, first-after noon scan; common-game restriction; exchanges vs sportsbooks.

## Next
Verdict: FanDuel is the best sportsbook candidate (runner-up DraftKings, gap 17 exact on 101 games near rule, 3 games on weeks 9-10). Not a proven single source: 24 mismatches remain, mostly half-point hooks on whole numbers and a few late moves. u1c regrade used FD (55.2% held out) and DK (57.1%); FD is the closer history proxy but history keeps the 19 hook games unresolved. Wk11 noon capture (u6 jobs) would test FD exact rate with scan lag under 10 min.

## Open
Wk5 scan 29h from noon; weeks 3-8 pool lines possibly captured after later moves. Does the pool half-point a FD whole number by a rule? Counts say 12/7 fav-dog, no rule shown.
