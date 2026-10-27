# pool-leader-romana

## Goal
Find what the FTPL pool leader (ROMANA-EMPIRE, 70-37 through Week 7) does that our card does not, and turn any mechanism into a fitted, held-out test.

## State
- 2026-10-27 the agent read Splash Standings > Picks by Week for Weeks 1-7, all 250 entries (in-page scrape). Saved locally (gitignored): data/splash/field/2026_field_pick_counts_w01-07.json (per-game field counts, scores, Romana picks) and data/splash/field/2026_romana_empire_w01-07.csv (per game: pool line, Tuesday and close market lines, our side). Script: tests/scratch/romana_analysis.py.
- Measured on 107 games: Romana 70-37 (65.4%), Best Pick 7-0. Field median 48.6%, 99th percentile 60.7%, second place 67-40. One named entrant p=0.0009, but as the best of 250 coin-flippers, P(some entry >= 70) = 0.21.
- With the market: on 83 games where the market moved off the pool line by kickoff, Romana took the side it moved toward 56 times (67%, p=0.001 vs 50%); our card 46, the field majority 43. Those picks went 40-16. 6 of 7 Best Picks had the market move 0.5-3 pts toward them.
- Contrarian: against the field majority on 67/107 (46-21); the field majority side went 45-62. Road dogs covered 40-26 this season; Romana took 39 of 66 (dog picks 42-20).
- "Follow the close, else fade the field" goes 61-46 and matches Romana on 74/107; Romana's extra comes from picks against the move (18-9).

## Tried
- Splash Activity > Pick changes lists only commissioner edits; submission times are not visible.

## Next
- Check prior work before proposing (LEAD-53 Sunday re-nomination, the 58.5% line-move archive figure, POOL-01). Candidate: a Best Pick chosen by the size of the market move off the frozen pool line, fitted leave-one-season-out on 2023-2025 hourly lines, against the served Sunday re-nomination.
- Re-scrape after each week to grow the field file (POOL-01 needs real field shares).

## Open
- None; the decision follows the held-out test.
