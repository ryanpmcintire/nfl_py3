# POOL-10

## Goal
Best Pick placement on half-point sides: does the field weight its single Best Pick toward the side the pool's half point helps (ROADMAP POOL-10). Context for the pool-standing panel and the standing-aware Best Pick question (revisit Week 15). Never a pick input: our Best Pick comes from our own fitted probability.

## State
Unit 3, 2026-11-21 10:02 local (Week 10 rerun), measured and recorded (7 cells replaced or added). Family `pool10_best_pick_hooks`, 18 looks this rerun. `scripts/pool10_best_pick_hooks.py --last-week 10 --subset 1-9 --subset 10-10` -> `artifacts/pool_rank_replay/pool10_best_pick_hooks_w10.json` (W1-8 and W1-9 files untouched).
- 150 games, 57 hook games (Week 10 adds 5), 2431 field Best Picks. Served multi-book Tuesday anchor.
- Outcome first: Best Picks on the half-point side won 312/591 (52.8%) v 904/1840 (49.1%), +3.7 pp, week-stratified permutation p .72; those games covered 32 of 57. Not separable.
- Favourite-adjusted model (week fixed effects, offset log picks; terms favourite, points favoured by, half-point side), half-point-side coefficient [game bootstrap within week] P+: Weeks 1-10 **0.212 [0.016, 0.382] P+ 0.998** (rate ratio about 1.24); Weeks 1-9 in the same run 0.251 [0.055, 0.429] P+ 0.9998; Week 10 alone -0.088 [-0.66, 0.52] P+ 0.328 (14 games, 5 hook). Favourite-indicator-only sensitivity 0.208 [0.014, 0.379] P+ 0.997.
- Share excess (Best Pick share minus pick share, week-cluster bootstrap): half-point side +3.01 pp [-0.30, 6.37] P+ 0.964, 7 of 10 weeks; hook games either side +1.87 [-0.16, 4.25] P+ 0.963; side the half point hurts -1.14 [-2.92, 0.90] P+ 0.131. Favourites +16.3 pp, 10 of 10 weeks.
- Implies: card unchanged; no rule moves our Best Pick. Pool panel and Week 15: after the favourite habit, a half-point side is still about a quarter more likely to carry the field's Best Pick; Week 10 alone leaned the other way on five hook games and pulled the pooled coefficient from 0.25 to 0.21. All cells `unresolved_below_power`.

## Tried
Unit 1 (2026-11-03, first-snapshot line): +2.3 pp [-1.0, 5.9] P+ 0.910. Unit 2 (2026-11-14, W1-9): coefficient 0.251 [0.049, 0.428] P+ 0.994, perm p .014; 10 cells. Unit 3 recorded 7 cells; the Week 1-8, 3-8 and Week 9 cells from Unit 2 stay as recorded.

## Next
1. Week 11+: `--last-week N --subset 1-(N-1) --subset N-N`; replace the pooled cells.
2. Matched-line contrast: half-point sides v other sides at the same pool line inside the same model (separates hook from key number).
3. Bootstrap interval for the outcome cell so it can be recorded; within-week permutation p for the Week 1-10 coefficient (check the artifact's model block).

## Open
- Not recorded: outcome cell (permutation p, no interval); favourite, underdog, hook-by-favourite and spread-size splits (resolved descriptions kept in the artifact).
- The linear points term cannot tell the half point from the pull of a short favourite at a key number (inferred). No entrant names stored.
