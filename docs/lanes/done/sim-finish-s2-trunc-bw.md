# SIM-31 S2 trunc_bw joint arm

## Goal
Run the joint arm trunc_bw (SIM10_TRUNCTILT=1 plus --h bandwidths_week_v2.json), seeds 11-13, and adjudicate under Tier B (count not lower any seed, none lost, a gate toward real beyond 2x dflt SD).

## State
- 2026-11-11 10:33 preconditions (measured): git status of sim10_drive.py, mod25_generator.py, sim10_situational.py clean (no uncommitted edits). scripts/sim10_drive.py mtime 2026-11-03 17:24; dflt_s11/s12/s13 created 11-03 14:05/13:27/13:34, so the engine file is newer than all three: rerun default seeds with label dflt first (old dirs renamed dflt_pre1124_sN). Free RAM 38.5 GB. No sim10_drive python process.
- SIM-27 lane: bandwidth-alone verdict recorded.
- PIDs started: none yet.

## Tried
Nothing run yet.

## Next
Rerun dflt s11-13, then trunc_bw s11-13, tierb_cmp.py trunc_bw 11,12,13 > artifacts/trunc_bw_cmp.log, adjudicate, write verdict here, SIM-27, SIM-21, patch queue.

## Open
None.


## In flight (2026-11-11 10:33)
Dflt rerun chain (seeds 11,12,13 sequential, label dflt, logs artifacts/dflt_S.log) running as background task b9t4uy0du; sim10_drive PIDs seen: 28420,23488,29568. Old dflt dirs renamed dflt_pre1103b_sN. Next: when artifacts/sim10/dflt_s13/gates.json exists and no sim10_drive process remains, run trunc_bw seeds 11-13 (command in the task brief: SIM10_TRUNCTILT=1, OMP_NUM_THREADS=1, --label trunc_bw --h artifacts/sim10/bandwidths_week_v2.json, logs artifacts/trunc_bw_S.log), then tierb_cmp.py trunc_bw 11,12,13 > artifacts/trunc_bw_cmp.log, adjudicate, write verdicts to this lane, SIM-27, SIM-21 and the patch queue lane.

- VERDICT trunc_bw (2026-11-11 11:18, measured, artifacts/trunc_bw_cmp.log; dflt reruns on current engine): gate counts dflt 13/14/13 vs arm 13/13/13. Lost: pts_game s12 and s13. Gained: r2_w10_18 s13. Moves: pts_game 44.856 -> 45.441 (real 44.832, +0.585, sd_dflt .164, AWAY from real, beyond 2 SD); mass_14 .0496 -> .0475 (real .0482, toward, beyond 2 SD, sd .0004); toward real inside 2 SD: mass_3, tie_share, strength_re_var (+2.81, sd 1.81), noise_re_var (-1.98), r2_w10_18; away inside 2 SD: q4_slope, mass_7, ot_share, margin_sd, home_edge. Tier B rule: count not lower on s12 (14->13) and a default-passing gate lost (pts_game): NOT ADOPTED. Default stays. Dflt reruns done because sim10_drive.py mtime 11-03 17:24 postdated dflt_s11-13; old dirs dflt_pre1103b_sN. Orchestrator owns any engine change. PIDs started: chain tasks b9t4uy0du, b9yreadzj; no sim10_drive process remains.

