# POOL-04 Field simulator calibration

## Goal
Score the pool-finish simulation behind the dashboard's first-place chance against what the 250 entries actually did (ROADMAP row POOL-04). Nothing here changes a pick.

## State (2026-11-14 15:50 ET, Weeks 1-9; measured this session unless labelled)
Weekly scores, `scripts/pool04_field_ppc.py --points 70 --out pool04_field_ppc_w09.json` (2000 sims, 129 looks):
- Season total, real results fixed: mean 66.51 observed vs 66.95 simulated; spread 5.56 vs 5.51 (ratio minus one +0.009 [-0.074, 0.106], probability_positive 0.58); leader 81 vs 82.5; entries at or above the observed top-decile total (73) 10.0% vs 15.7% [12.0, 19.6]; tied at the top 1 vs 1.3.
- By week: mean inside the 90% band 9 of 9 (observed lower every week, 0.44 in total; 216 of 34,000 picks never made are worth 0.43, inferred); spread ratio averages 1.04, outside in Week 7 (1.94 vs 1.78) and Week 9 (2.24 vs 1.81); weekly best inside 9 of 9; weekly top-decile share inside 8 of 9.
- Week-to-week correlation -0.006 vs 0.000 [-0.017, 0.018]. Split-half reliability -0.087 [-0.43, 0.16], probability_positive 0.27 (read: weekly_points_field.json); a no-skill field simulates [-0.24, 0.18]. Not above zero: confirmed.

Backtest, `scripts/pool04_forward_backtest.py --points 70 --out pool04_forward_backtest_w09.json` (same `draw_week`; real standings after Weeks 1-8 run to Week 9; 224 looks; windows overlap, one season; our entry is the model's sides, 69 points, since our row is one of 18 on 70; rank by points only):
- Inside the simulated 90% band: our points 7 of 8, leader 5 of 8, our rank 7 of 8, gap to leader 8 of 8, field mean 8 of 8, field spread 6 of 8, ties 8 of 8.
- Direction, 8 of 8 starts each: our points below the simulated mean (4.1 to 9.3 from Weeks 1-7), leader below (2.0 to 4.6), our rank worse (79th vs mean 24th to 65th).
- Components: stated cover chances overshot our points and the realised rate to date did no better; with real results fixed the leader is still 1.4 to 5.1 below and outside the band from Weeks 6-8, and field spread is 3.5-6% too wide there (entries near the top scored below the field afterwards); 15 games a week simulates 75 from Week 4 where 72 were played; ties fine. The two misses pull opposite ways on our chance. No constant added.

First-place chance, rank 65, nine weeks left (1200 paths; `pool01_forward_sim.py --our-cover`, default unchanged, stated rerun equals the panel file exactly): stated chances (55.2%) 6.0% [4.7, 7.4]; realised rate (69 of 136) 0.7% [0.3, 1.2]; difference -5.3 points [-6.8, -3.9]. The record does not pick between them: realised rate interval [42.4, 59.0], probability it is truly at or above 55.2% is 0.15, above 50% is 0.57.

## Proposed panel wording (not applied; board builders untouched)
"Our chance of finishing first is about 6 in 100 if our picks win from here as often as the model expects (about 55 of every 100), and under 1 in 100 if they keep winning at the rate they have so far (69 of 136). Replayed from each earlier week of this season, the same estimate expected a few more points than both we and the eventual leader went on to score, so read it as a rough guide."

## Tried
- Unit 1 (2026-11-04, Weeks 1-8): total spread ratio 1.036, `pool04_total_sd_ratio_fixed_cover`.
- 2026-11-14: scripts take arguments; new files pool04_field_ppc_w09.json, pool04_forward_backtest_w09.json, pool04_forward_sim_{stated,realised}_cover_w09.json. Panel files untouched: field_best_pick_model.json, weekly_points_field.json, forward_sim_real_rivals_results.json. Ruff clean.
- Recorded 8 cells in family pool04_field_calibration (names start `pool04_w01_09_`, `pool04_backtest_w01_start_`, `pool04_first_place_chance_`), all unresolved_below_power; `weak-signals pool` loads. Coverage counts stay here: overlapping windows have no independent-sample interval.

## Next
1. Root: put the wording on the panel once the board builders are free, showing both chances.
2. Per-entry per-game picks (POOL-02 read): do one entry's picks move together, do leaders pick differently; the named candidates for the Week 7 and 9 spreads and the leader's shortfall.
3. Feed the real remaining game counts to the forward run instead of 15 a week.

## Open
- Which cover rate to simulate is unresolved; an out-of-season calibration of the stated chance would settle it.
