# POOL-10

## Goal
Best Pick placement on half-point sides: does the field weight its single Best Pick toward the hook side (ROADMAP POOL-10).

## State
Added 2026-11-03 (backlog expansion 7), not started, nothing run. Mechanism: The Best Pick carries extra weight; a field that over-picks hook sides may concentrate that weight there. Fitted term: hook indicator in the Best Pick share model inside the standing-aware leverage. Data: captured field Best Pick shares if present (local).

## Tried
Novelty check only. Nearest: POOL-08, POOL-05 (slot and team ownership), confidence-best-pick-unification lane. Differs: Best Pick weight, not side share.

## Next
First unit for one agent: Count captured weeks with Best Pick shares before any test; if fewer than eight, record the count and defer.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.

## Unit 1 (2026-11-03 17:31, family pool10_best_pick_hooks, 12 looks)
scripts/pool10_best_pick_hooks.py -> artifacts/pool_rank_replay/pool10_best_pick_hooks.json. 121 games W1-8 (44 hook), 1947 field Best Picks. Excess = Best Pick share minus regular pick share, week-cluster bootstrap (4000).
- Hook games: BP 37.8% vs picks 36.4%, +1.3 pp [-0.9, 4.0] P+ 0.835. Half-point side: 22.9% vs 20.6%, +2.3 pp [-1.0, 5.9] P+ 0.910 (6/8 weeks). Other side of hook games -1.0 pp [-2.9, 1.1].
- Favourites 74.2% of BPs vs 58.1% of picks: +16.1 pp [13.0, 19.3] (8/8 weeks). Hook-side favourites +5.6 pp [2.5, 9.4]; hook-side underdogs -3.2 pp [-4.4, -1.8]. Spread size bands 0-3/3-7/7+ all within +-1.3 pp, intervals span zero.
- Outcome: BPs on hook side won 227/446 (50.9%) vs 693/1501 (46.2%) others: +4.7 pp, week-stratified permutation p 0.68. Not separable.
- Ours (last flagged per week, W1-8): hook games 3/8 (W3 CAR hook side, W7 NYJ other side, W8 SEA hook side); hook side 2/8; W9 DEN/CAR has no field capture yet. Field BPs on our game 6-15% of week total.
- Recorded 1 cell (hook-side excess, unresolved_below_power). Placement is mostly a favourite effect; hook concentration is small.
- Challenger only: if hook sides are over-owned, a shared Best Pick there carries more entries; leverage discount only, no rule changes our Best Pick.
## Next
W9+ rerun; test hook term inside favourite-adjusted field BP share model.
