# pool-leader-romana

## Goal
Find what the FTPL pool leader (ROMANA-EMPIRE, 70-37 through Week 7) does that our card does not, and turn any mechanism into a fitted, held-out test.

## State
- 2026-10-27 the agent read Splash Standings > Picks by Week for Weeks 1-7, all 250 entries (in-page scrape). Saved locally (gitignored): data/splash/field/2026_field_pick_counts_w01-07.json (per-game field counts, scores, Romana picks) and data/splash/field/2026_romana_empire_w01-07.csv (per game: pool line, Tuesday and close market lines, our side). Script: tests/scratch/romana_analysis.py.
- Measured on 107 games: Romana 70-37 (65.4%), Best Pick 7-0. Field median 48.6%, 99th percentile 60.7%, second place 67-40. One named entrant p=0.0009, but as the best of 250 coin-flippers, P(some entry >= 70) = 0.21.
- With the market: on 83 games where the market moved off the pool line by kickoff, Romana took the side it moved toward 56 times (67%, p=0.001 vs 50%); our card 46, the field majority 43. Those picks went 40-16. 6 of 7 Best Picks had the market move 0.5-3 pts toward them.
- Contrarian: against the field majority on 67/107 (46-21); the field majority side went 45-62. Road dogs covered 40-26 this season; Romana took 39 of 66 (dog picks 42-20).
- "Follow the close, else fade the field" goes 61-46 and matches Romana on 74/107; Romana's extra comes from picks against the move (18-9).

- Owner rejected the first read as already known (2026-10-27). New, measured: the played card (pick-revision ledger, last pre-kick revision) followed the pool-relative move on 59/83 moved games, more than Romana's 56. Its 24 picks against the move went 8-16; Romana took the other side in 17 of them and went 14-3; on their other 31 disagreements Romana is 16-15. Of the 24: 12 had no post-Wednesday move or followed it (pool line stale or Tuesday-afternoon move) and went 3-9; 12 opposed even the Wednesday-on move and went 5-7.
- Read: served move term = leader-book summed line changes with quote_as_of >= Wednesday (sharp_book_movement_features.py:136-137), relative to the books' own prior quotes, never to the pool line. Archive grades at the Tuesday market open, so a pool line off the market never appears in the fit. Coefficients (pick_probability/20261027T221741Z): move 0.221/pt, flag_sum 0.319, model_logit 0.309; median |move| 0.5 pt. 2023-25 held-out card opposes the move on 51/556 moved games; live 24/83.

## Tried
- Splash Activity > Pick changes lists only commissioner edits; submission times are not visible.

## Next
- Unit 1 (running 2026-10-27): challenger four-term probability whose move term is measured from the graded line (Tuesday open in the archive, pool line live) to the leader-book median at the lock cutoff, including Tuesday-to-Wednesday moves; leave-one-season-out on 2020-2025 against the served fit. Script under tests/scratch/; no src edit (sharp_book_movement_features.py is pinned).
- Then: if it wins held out, wire a pool-relative move into the live path and re-enroll validation the same session.
- Re-scrape after each week to grow the field file (POOL-01 needs real field shares).

## Open
- None; the decision follows the held-out test.
