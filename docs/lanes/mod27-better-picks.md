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
4. Serve the MOD-26 pool hook term: lane mod27-u4-hook-serve.md. SERVED 2026-11-02 (fd1d0278) as a two-stage term: served four-term fit and per-game probabilities unchanged, hook 0.805; validation v7 enrolled Weeks 9-18. A fresh review of the merged code is running (lane hook-review-2026-11-02.md).
fl_py3_wt_hook.
5. Pool-relative forward log: lane mod27-u5-pool-move-log.md. DONE: scripts/pool_relative_move_forward_log.py, 8 scheduler jobs (pool_move_log_*), MANUAL-RUN OK, Week 8 backfill card 5-8 vs challenger 6-7; scheduler restarted 09:59 ET, code current.
3. Beat-reporter capture health and 2024-26 rerun: lane mod27-u3-beat-reporters.md. DONE (verified docs/mod27_u3_beat_forward26.json): capture healthy (jobs beat_reporters_sat/sun/mon, 1,304 tagged 2026 posts). Fit 2020-25, scored 2026 Weeks 2-8 (104 games): base 61-43; +beat 59-45 (-1.92 pts, P+ .31; log loss gain +0.0103, P+ .90); +beat_lead 56-48 (-4.81, P+ .037, decisive 3-8). The 2024-25 lead result did not repeat. Not served; rerun after more weeks.

6. Pool line off the market at posting: lane mod27-u6-pool-offset.md. MEASURED: the side getting the better number versus the Tuesday-noon market went 36-22 (hook 26-18, non-hook 10-4); card against it 6-16. Generalize the hook term once the branch passes.
7. Late flips: lane late-flip-mechanism-2026-11-02.md. DONE: move-driven late flips 7-4; defect = one outlier pre-Wednesday book baseline (LV at NYJ). Guard test: lane move-baseline-guard.md (running).
8. Prediction markets (acc-u8): term worse in 2025 (Polymarket -6.47 pts, P+ .011); not served.
9. Dashboard: History shows final and Tuesday-card records per week (commit e0aad558, pushed).

10. Move baseline guard: lane move-baseline-guard.md. DONE: neither variant beats served (P .10-.12); LV at NYJ had only two leader books.
11. Generalized pool offset term: simulation coefficient 0.85-0.96, mechanical; changes no 2026 side. Bundle only with the two-stage hook.
12. Forward log now records the Tuesday-market pool offset; Week 8: better-number side 5-1, card against it 0-2.
13. Mid-season acquisitions: repeat of LEAD-23; extension 2022-25 not served (strict 835-668 vs 833-670); month-end chronology leak in the research flag. Draft-capital forward log repaired (float-noise hash gate). Tuesday 12:10 ET odds capture added.

## Next
- After the Week 9 Tuesday lock: finish unit 4 (atlas), merge, activate, enroll v7.
- Grade the pool-relative log weekly (`scripts/pool_relative_move_forward_log.py --grade`).

## Open
- None yet.
