# POOL-10

## Goal
Best Pick placement on half-point sides: does the field weight its single Best Pick toward the side the pool's half point helps (ROADMAP POOL-10). Context for the pool-standing panel and the standing-aware Best Pick question (revisit Week 15). Never a pick input: our Best Pick comes from our own fitted probability.

## State
Unit 2, 2026-11-14 16:30 (Week 9 rerun + owed favourite-adjusted model), measured and recorded. Family `pool10_best_pick_hooks`, 20 looks. `scripts/pool10_best_pick_hooks.py --last-week 9 --subset 1-8 --subset 3-8 --subset 9-9` -> `artifacts/pool_rank_replay/pool10_best_pick_hooks_w09.json` (Week 1-8 file untouched).
- Anchor: served multi-book Tuesday line (`tuesday_market_lines`, src change 2026-11-10). 136 games, 52 hook games (per week 7,4,7,4,4,5,7,4,10), 2189 field Best Picks. Unit 1 ran under the first-snapshot line (44 hook games in Weeks 1-8, now 42), so its +2.3 is not the same construction.
- Share excess (Best Pick share minus pick share, week-cluster bootstrap): half-point side 25.3% v 21.8%, +3.56 pp [0.08, 7.07] P+ 0.977, 7 of 9 weeks. Weeks 1-8 +2.59 [-0.67, 5.96] P+ 0.942; Weeks 3-8 +1.06 [-1.92, 3.50] P+ 0.777. Hook games either side +2.17 [0.01, 4.73] P+ 0.975; side the half point hurts -1.39 [-3.30, 0.89] P+ 0.108. Favourites 74.4% v 58.5%, +15.8 [13.1, 18.8], 9 of 9 weeks.
- Owed model: Best Picks per pick on a side, week fixed effects, offset log picks (a within-week multinomial of Best Pick share), terms favourite, points favoured by, half-point side. Coefficient [95% game bootstrap within week] P+ / within-week permutation p (hook status shuffled across games, 20,000 draws):

| weeks | half-point-side coefficient | hook games |
|---|---|---|
| 1-9 | 0.251 [0.049, 0.428] 0.994 / .014 | 52 |
| 1-8 | 0.256 [0.020, 0.457] 0.983 / .026 | 42 |
| 3-8 | 0.183 [-0.081, 0.410] 0.922 / .124 | 31 |
| 9 alone | 0.310 [-0.103, 0.634] 0.942 / .143 | 10 |

- Weeks 1-9: rate ratio 1.29; leaving out one week 0.181-0.286 (9 folds); week-cluster interval [0.108, 0.423]; favourite 0.559 [0.226, 0.860]; points favoured by 0.019 [-0.005, 0.050]. Favourite-indicator-only sensitivity 0.240 [0.045, 0.409] P+ 0.9925.
- Outcome: Best Picks on the half-point side won 290/554 (52.3%) v 769/1635 (47.0%), +5.3 pp, week-stratified permutation p 0.62; those games covered 30 of 52. Not separable.
- Ours (last flagged per week): on a hook game 4 of 9 weeks, on the half-point side 3 (away side of Week 3 CAR at CLE, Week 8 CHI at SEA and Week 9 DEN at CAR, where 25 of the week's 242 field Best Picks were on our side). Unit 1 labelled Weeks 7 and 8 as NYJ and SEA; the settled record read today gives the away side in both, not reconciled.
- Implies: card unchanged; no rule moves our Best Pick. Pool panel and Week 15: after allowing for the field's favourite habit, a pick on the half-point side is about 29% more likely to be the field's Best Pick, so when our Best Pick is a favourite on that side more of the field shares the double-weighted pick and it separates us less; on the other side or an underdog it separates more. All cells `unresolved_below_power`.
- Recorded 10 cells with replace (3 share-excess windows, hook game, other side, 4 coefficient windows, sensitivity); `weak-signals pool` loads (exit 0).

## Tried
Unit 1 (2026-11-03, first-snapshot line): 121 games, half-point side +2.3 pp [-1.0, 5.9] P+ 0.910; favourites +16.1 pp. Unit 2 looks: 11 placement splits, 1 outcome, 2 model specifications, 3 subsets x (coefficient + share excess).

## Next
1. Week 10+: `--last-week N --subset N-N`.
2. Separate the half point from the key number: hook sides sit where the market is on a whole number (often 3 or 7), so compare half-point sides with other sides at the same pool line (matched-line contrast inside the same model).
3. Bootstrap interval for the outcome cell so it can be recorded.

## Open
- Not recorded, with reason: the outcome cell has a permutation p but no interval; favourite, underdog, hook-by-favourite and spread-size splits are resolved descriptions (favourite 9 of 9 weeks) that the registry's three classifications do not represent, kept in the artifact.
- The linear points term cannot tell the half point from the pull of a short favourite at a key number (inferred). Game bootstrap and week-cluster intervals differ; nine weeks is too few for the latter. No entrant names stored.
