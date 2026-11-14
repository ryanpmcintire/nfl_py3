# POOL-08

## Goal
Half-point hook awareness: does the pool's field over- or under-pick the side the pool's half point helps (ROADMAP POOL-08). Context for the pool-standing panel and the standing-aware Best Pick question (revisit Week 15). Never a pick input: one fitted cover probability selects every side.

## State
Unit 2, 2026-11-14 (Week 9 rerun): measured and written to artifacts; NOT yet recorded in the registry (agent hit the tool cap before `record`). Family `pool08_hook_field_share`.
- Anchor drift (read, commit 940c8140, 2026-11-10): `tuesday_market_lines` in `src/nfl_ats/pool_hook.py` now returns the multi-book Tuesday line, so the same script gives different hook games than on 2026-11-03. The script now takes `--last-week`, `--subset A-B`, `--out`, `--anchor served|first_snapshot`. `--last-week 8 --anchor first_snapshot` reproduces the stored 2026-11-03 artifact key for key (measured). Under `served`, 2026_06_BUF_LV and 2026_08_PHI_WAS stop being hook games and 2026_09_GB_NE is added.
- Share on hook side minus POOL-05 line proxy (leave-one-week-out), pp [95%] P+ / within-week permutation p / hook games. `served` anchor -> `artifacts/pool_rank_replay/pool08_hook_field_share.json`; `first_snapshot` -> `..._first_snapshot.json`.

| weeks | served anchor | first-snapshot anchor |
|---|---|---|
| 1-9 pooled | +6.53 [2.23, 10.57] 0.999 / .039 / 52 | +6.19 [2.10, 10.34] 0.998 / .058 / 53 |
| 1-8 (in 1-9 fit) | +7.60 [2.96, 12.07] 0.9995 / .031 / 42 | +7.49 [3.26, 11.64] 0.9998 / .029 / 44 |
| 3-8 | +5.15 [-0.09, 10.37] 0.974 / .096 / 31 | +5.15 [0.09, 10.04] 0.976 / .090 / 33 |
| 9 alone (proxy fit on 1-8) | +2.04 [-9.47, 12.14] 0.649 / .74 / 10 | -0.21 [-11.61, 9.99] 0.501 / .97 / 9 |

- Hook games per week 1-9: served 7,4,7,4,4,5,7,4,10; first snapshot 7,4,7,4,4,6,7,5,9. Weekly means (served): 8.8, 24.6, 6.2, 6.9, 6.8, 14.7, -3.1, 2.3, 2.0; 8 of 9 positive, exact sign p .039 (inferred from the measured means). Week 9 has 252 (126) distinct within-week arrangements, so its p cannot be small.
- Owed second part, hook coefficient in the ownership proxy (home-share logit: intercept, pool line, hook side; 136 games): 0.272 [0.086, 0.457] P+ 0.999, folds 0.209-0.343, about +6.8 share points at an even split (first snapshot 0.255 [0.081, 0.434] P+ 0.998). Held-out log loss gain all games +0.0023 [-0.0027, 0.0070] P+ 0.824; hook games 0.6780 -> 0.6727 (first snapshot +0.0020 [-0.0028, 0.0064] P+ 0.803).
- Outcome: hook side 30-22 vs mean model probability 0.513: +6.4 pp [-7.1, 19.9] P+ 0.822, null p .40 (first snapshot 32-21, +9.2 [-3.2, 22.2] P+ 0.925, p .21).
- Implies: card unchanged (it picks by cover probability; the outcome look gives no ground to move any probability). Pool-standing panel: the field takes the half-point side about 6.5 share points more than the line predicts on roughly 5-6 games a week, so a pick on that side separates us from the field slightly less. Week 9 alone is smaller but its interval contains the earlier mean; the pooled estimate moved 7.7 -> 6.5. All cells `unresolved_below_power`.

## Tried
Unit 1 (2026-11-03): 44 hook games W1-8, +7.68 [3.22, 11.94] P+ 0.9995, perm p .029; coefficient 0.24-0.41 per fold. Unit 2 looks: 2 anchors x (pooled, 3 week subsets, log loss, coefficient, cover) = 14, plus one weekly sign count. Wiring the coefficient into the POOL-01 replay: `scripts/pool01_forward_sim.py:33,47-49` (read) draws each game's observed share and has no fitted ownership proxy, so the coefficient has nowhere to enter and the observed shares already carry the lean.

## Next
1. `ruff format scripts/pool08_hook_field_share.py` (one comprehension at the `cboot` line collapses to one line; `ruff check` passes; behaviour unchanged).
2. `nfl-ats weak-signals record --batch <file> --replace`: 14-cell batch is written at `C:\Users\Ryan\AppData\Local\Temp\claude\9054f05809d7697a\ea407957-0844-4cbe-8bac-9668d775076e\scratchpad\pool08_batch.json`; if gone, rebuild from the two artifacts (units rate_pp, log_loss_improvement, logit_coefficient; replace `pool08_hook_side_share_residual`, `pool08_hook_term_logloss`, `pool08_hook_side_cover_vs_model_p`; new `_w09`, `_w03_08`, `_w01_08`, `pool08_hook_coefficient`, and `_first_snapshot_anchor` twins).
3. Run `nfl-ats weak-signals pool --league nfl --effect-units accuracy_points` once and confirm it loads.
4. Rerun at Week 10+ with `--last-week N --subset N-N`.

## Open
- The replay needs a predicted-share step for unplayed weeks before the coefficient can be used there; that is the missing input for the POOL-01 wiring.
- Intervals are game-level bootstraps; nine weeks is too few for a week-block interval.
- Zero crossing closes nothing; no side rule; no entrant names stored.

- 2026-11-14 15:20: recorded 14 cells in family pool08_hook_field_share (replace; Weeks 1-9 supersedes Weeks 1-8; units rate_pp, log_loss_improvement, logit_coefficient). Nothing left unrecorded. pool loads. ruff format applied to the script; check and format --check pass. Step 1-3 of Next done.
