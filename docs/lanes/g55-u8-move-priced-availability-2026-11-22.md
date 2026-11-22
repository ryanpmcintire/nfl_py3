# g55-u8-move-priced-availability-2026-11-22

## Goal
Parent: goal-55-backwards-2026-11-21.md. Learn what each piece of late who-plays news is worth from the market's own Tuesday-to-deadline line move (a label with about 1.5 points of spread, against 13 for the final margin), then serve that price from the inactives alone, with no market input at serve time. Done = one declared grade, model alone at the Tuesday opener.

## Declaration (root, 2026-11-22 06:27 EST, before any outcome read of this arm)
- Inputs: artifacts/g55_u5b/per_game.parquet (p_base, home_covered, 1,503 decisive 2020-25 games), artifacts/g55_u5b/servable_features.parquet (diff_servable_o1/o2/o3 home minus away, deadline_known; zero where not known), artifacts/opener_evaluation/20261121T171727Z/per_game.parquet (tue_open_home_spread, close_home_spread), data/processed/game_features.parquet (weekday, gametime).
- Stage 1: move = close minus Tuesday open on games whose close is at or before the pool deadline (drop Sunday kickoffs at or after 17:00 and Monday). OLS move on the three deltas plus intercept.
- Stage 2: logistic home_covered on logit(p_base) and the stage-1 predicted move.
- Nested leave-one-season-out: for held-out season s, fit stage 1 on the other seasons, predict for all games, fit stage 2 on the other seasons, score s. No market quantity enters a held-out game's prediction.
- Looks: all (primary), quiet, moved = 3, family g55_u8_move_priced_availability. Controls: base 802-701 reproduced first; matched null = the three deltas shuffled jointly within season, 50 draws, both stages refit.
- Serving candidate only if all of: at least 827 wins, null percentile at least 95, log loss no worse than base (.6968). Otherwise recorded unresolved_below_power, not retuned on these games.

## Result (measured 2026-11-22 06:40 EST, root; scratchpad g55_u8.py, artifacts/g55_u8/results.json)
- Arm 820-683 (54.56%) v base 802-701: +1.20 pts [-1.37, +3.54] P+ .806; log loss .69148 v .69678 (better); matched null wins 5/50/95% = 775/795/808, arm at 100th percentile. Quiet 199-171 (= base), moved 621-512 v 603-530 (+1.59 [-0.27, +3.69] P+ .939). Changed 322 picks, 170-152. Seasons 111/128/137/145/145/154 v base 119/130/133/132/148/140 (3 of 6 up). Stage-1 prices stable across folds (QB-out delta -4.0 to -5.1 points of move).
- Wins leg fails (820 < 827); null and log-loss legs pass. Best no-market record so far (S_ALL_v2 812). Recorded unresolved_below_power.

## G55-U8b declaration (root, 2026-11-22 06:42 EST, POST-SELECTION: both pieces chosen after their own grades)
- Same nested LOSO with S_ALL_v2's logit(p_arm) in place of logit(p_base) in stage 2. Mechanism: S_ALL_v2 gains on quiet games, U8 on moved games. Looks all/quiet/moved = 3, family g55_u8b_combined. Null: the three deltas shuffled jointly within season in stage 1 only, 50 draws. Same three serving legs; history grade is optimistic either way and only a forward read decides.

## G55-U8b result (measured 2026-11-22 06:50 EST, artifacts/g55_u8b/results.json)
- 809-694, +0.47 [-2.24, +3.06] P+ .627, log loss .69282, null 95.0 pct; quiet 191-179 (base 199-171), moved 618-515. Worse than U8 alone: the S_ALL_v2 quiet-game gain does not survive one fitted slope. Recorded unresolved_below_power (family g55_u8b_combined).

## State
U8 and U8b done and recorded (registry/weak_signals g55_u8_move_priced_availability, g55_u8b_combined). U8 820-683 is the best no-market record. Forward logging of U8 for 2026 Weeks 12-18 is BUILT inside scripts/g55_s_all_v2_forward.py (uncommitted): `freeze` also writes artifacts/g55_s_all_v2_forward/u8_frozen.json (`freeze --u8-only` refits only U8, leaving frozen.pkl alone); `pregame` writes u8_home_probability, u8_predicted_move, u8_side per game and u8_sha256 per file; `score` adds a "u8" block (last admissible file per game whose u8_sha256 matches, v frozen base, arm, card; log loss). Same argv as U7, so the Tuesday scheduler step needs no new jobs.

## Tried
- Freeze (measured 2026-11-22 06:58 EST, `freeze --u8-only`, exit 0, freeze_u8.log): u8 sha 1f338bcd4d38c35355f7c0467b4c913ea3999be45246447ffeae5a950bc508e7. Stage 1 (1,290 pre-deadline rows): int 0.0927, o1 -0.1425, o2 0.0955, o3 -4.5407. Stage 2 (1,503): int -0.0183, logit(p_base) 0.3377, predicted move 0.3312. In-sample 816-687 (log loss .69015); LOSO rebuilt from the same frame 820-683 (.69148) = the graded result; base 802-701 (.69678). frozen.pkl sha still 9d9c6638...4f36.
- Dry Week 10 (measured, exit 0, dry_run/2026_week10_20261122T115858Z.json): 14 games, U8 rows 14 = arm rows 14; base and arm max abs diff 0.0 v the U7 dry file 20261121T205156Z (player snapshot newer); U8 recomputed from the frozen file diff 0.0; sides consistent; 3 U8 v base side changes; predicted moves -0.56 to +0.41.
- `score` exit 0 (no forward rows yet); u8_report smoke on synthetic outcomes ran (scratchpad g55_u8_fwd_compare.py, no 2026 result read); `slots` exit 0; `ruff check` and `ruff format --check` clean; no comments.

## Next
1. Root: review and commit scripts/g55_s_all_v2_forward.py with the U7 script (artifacts stay untracked).
2. Tuesday 11-24: the U7 scheduler jobs (pregame after each inactives_* capture, score Tue 09:25) log U8 too; no extra argv.
3. After Week 18: one declared U8 forward look from score.json "u8" (record v frozen base at the logged opener, log loss), recorded once in family g55_u8_move_priced_availability.

## Open
- Inferred from the coefficients: with zero deltas U8 picks home when p_base > 0.4908 (fitted intercepts; logit slope 0.338 shrinks p_base), so a quiet game with p_base in (0.4908, 0.5) changes side v base.
