# MOD-57 u1 - pool source book

## Goal
Find the book and time behind the pool's frozen Tuesday line (2026, 10 weeks, 150 games), and whether the 2020-25 archive reproduces it. No regrading here. Stamped 2026-11-11 08:32 EST.

## State (measured; scratch tests/scratch/mod57/u1_*.py; sign checked: pool home_spread and quote home_spread_line both positive = home favoured, flipped-sign exact rate 0-3%)
- Pool never posts a whole number where the source is whole: 51 of 51 whole-source WH games and 17 of 17 DK games differ by exactly 0.5.
- Quotes near Tuesday noon exist only Wk1, 2, 10 (first quote after noon within 0.2h; Wk3-8 first quote 2.4-6.6h late, Wk5 and 9 about 30h late). Pool file times are agent read times, not posting times.
- Wk1+2 (32 games, quote 0.1h after noon, same games): FanDuel 28 exact / 32 within 0.5; MEDIAN 20; WilliamHill 20; BetMGM 19/31; DraftKings 17; Fanatics 17; others 9-16. FanDuel beats DK 28-17 on the same games.
- FanDuel has no 2026 quotes after Wk2 (feed lost), so FD cannot be tested Wk3-10. Wk10 DK 11/14 exact (WH 8/14).
- First-quote-after-noon, all weeks (stale in Wk3-9): WH 80/150 exact (53%), within 0.5 87%; DK 33/60 (55%) where present; median 72/150 (48%), within 0.5 89%. Hard cuts Mon18/Tue09-13 are stale-quote artefacts (DK 20-29%).
- Cannot separate FD from DK as the "pool source" beyond Wk1-2 vs Wk10: evidence is FD (Wk1-2) and DK (Wk10), no book covers all weeks at noon.
- Hook (half-point side, whole-source games): DK 16 half-off: easier side = dog 7 / fav 9, home 7 / away 9; WH 50: dog 25 / fav 25, home 19 / away 31 (P of 31+ of 50 about 0.06, one look of ~6), toward later move 10 vs against 13 vs 27 no move. No rule identified.
- Pool >= 1 off first-after-noon source: DK (Wk1-2,10) 8 games e.g. SF_LA 4.5 vs 3.5, PIT_NE 5.5 vs 4.5, MIN_GB 5.5 vs 4.5, WAS_NYG 2.5 vs 1.5, BUF_NYJ -8.5 vs -7.5; pattern: pool is usually 1 closer to pick-em than DK.
- Archive (data/market/raw). 2023-25 hourly: first quote after Tue noon ET within 1h (17:55Z), per season games with DK / FD / WH: 272/270/272, 271/271/256, 272/271/256 of 272 games; >=5 books every game (median 10-16 books). 2020-22 hold only Tue ~08:55-09:55 ET snapshots (about 3.1h before noon), none after noon: DK 220/239/255, FD 204/196/255, WH 197/209/243 of 235/240/255 games.
- Median (>=5 books) minus book line, share at 0 / half / 1+ (2023-25 post-noon, DK): 75/18/3, 78/16/2, 68/27/4; FD 65/29/3, 52/37/6, 61/33/5; WH 90/6/1, 85/10/0, 88/11/1. Pre-noon snapshot 2020-22 DK: 80/19/1, 63/33/2, 76/23/1; FD 78/18/3, 75/20/4, 73/24/4.
- Looks counted: B about 72 book-cut cells, C 8 hook descriptions x 3 sources, D 3 books x 6 seasons. No cover rate scored, so no weak-signals record.

## Tried
FD vs DK vs median at hard cuts; stale-lag buckets; hook rule counts; archive coverage.

## Next
Rebuild 2020-25 pool lines: FD post-noon (2023-25) and Tue 09 ET snapshot (2020-22), pool = source with whole numbers shifted a half point (side random until identified; run both sides as declared arms), plus the observed 1-point-toward-pickem gap vs DK; regrade the served two-stage probability leave-one-season-out at those lines; report decisive-game record first. First, cheap check: Wk11 (Tue noon) capture quotes within 0.2h to separate FD from DK; FD feed must be restored.

## Open
- FD feed absent since Wk2; pool posting time unknown; 2020-22 lacks post-noon quotes.
