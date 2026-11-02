# MOD-27 - get better at picks

## Goal
Owner 2026-11-02: get better at making picks. Repair the steps between the model's own read and the served card, and add information the pool's Tuesday line lacks. Every change is graded through the played card at the opener (LOSO 2020-25 plus 2026 forward) before it is served.

## State
- Evidence (week8-postmortem-2026-11-02.md): model read W3-8 50-38, card 41-47. Smooth base loses to discrete through the stack (kept). Late flips after the last feed 2-6.
- Home nudge (residual-sample median shift), measured 2026-11-02 from pick_probability/20261101T171318Z/per_game.parquet: 2020-25 flips 204; pointed away in every 2020-22 flip and home in every 2024-25 flip. Away-direction flips 80-61, home-direction 32-31, 2026 (home, +0.36..+0.45) 2-9.
- Prior work not to repeat: LEAD-67 key-number move (worse), market_move_decomposition_v1, pool_relative_move_2026_10_27 (forward log Weeks 8-18), pool_hook_term_2026_10_27 (MOD-26, P+ .997 on Brier/log loss, serving blocked by the pinned-source guard pending a bundle), MOD-23/24 base-model units, ACC-U7/U7b/U8/U9.

## Units
1. No-nudge base through the played card: lane mod27-u1-home-nudge.md. DONE (verified harness reproduces 833-670): shift by season -1.49, -1.73, -0.92, -0.06, +0.46, +0.39, +0.37 (2020-26). Through the stack, no-nudge 819-684 vs served 833-670, disagreements 24-38 (2023-25 5-14); log loss P(no-nudge better) .68; 2026 forward 57-46 vs 58-45. Not served. The orchestrator's "only 2020-22" hypothesis held at the raw read but not through the stack. A fitted home term is not new: the stack intercept already is one.
2. Clean 2026 move term and pool-relative tally through Week 8: lane mod27-u2-move-term-2026.md. DONE (verified agentU2/games.csv): clean quotes change 3 sides, served right on all 3 (clean 65-55 vs served 68-52); parser bug is not the cause. Card against the pool-relative move 6-13, with it 38-28 (P .017). Pool-relative stand-in (coef .145 from 2020-25 LOSO) vs served stack: 71-49 vs 68-52, 11 changed sides 7-4, log loss flat (P .44); not served. The forward log promised 10-27 was never built.
4. Serve the MOD-26 pool hook term (P+ .997 Brier, .996 log loss; owner row 2026-10-27), the one validated change, with validation re-enrolled as v7 for Weeks 9-18: lane mod27-u4-hook-serve.md.
5. Build the pool-relative forward log (challenger side beside the card each refresh, Weeks 9-18): lane mod27-u5-pool-move-log.md.
3. Beat-reporter capture health and 2024-26 rerun: lane mod27-u3-beat-reporters.md. DONE (verified docs/mod27_u3_beat_forward26.json): capture healthy (jobs beat_reporters_sat/sun/mon, 1,304 tagged 2026 posts). Fit 2020-25, scored 2026 Weeks 2-8 (104 games): base 61-43; +beat 59-45 (-1.92 pts, P+ .31; log loss gain +0.0103, P+ .90); +beat_lead 56-48 (-4.81, P+ .037, decisive 3-8). The 2024-25 lead result did not repeat. Not served; rerun after more weeks.

## Next
- Read each unit's verdict; any servable change goes in with the MOD-26 hook term under one validation re-enrollment (pinned-source rule), before the Tuesday weekly_lock where possible.

## Open
- None yet.
