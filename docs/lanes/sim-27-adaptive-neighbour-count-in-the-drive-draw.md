# SIM-27

## Goal
Adaptive neighbour count in the drive draw: replace the fixed sd bandwidth with a support-based neighbour set so thin trailing states are not smeared toward neutral (ROADMAP SIM-27).

## State
Added 2026-11-03 (backlog expansion 9), not started, nothing run. Mechanism: SIM-26 unit 2 found the kernel stage under-delivers the real TD surge for trailing offenses (real minus kernel .091, .063, .029) because a fixed score-difference bandwidth mixes deep deficits with neutral drives. Measured mechanism: neighbour count set by local support of real drives in the state space, checked leave-game-out. Fitted term: Simulator mechanism, not a probability term: the draw uses the nearest real drives with the count set by measured support; no tuned bandwidth, no offset.

## Tried
Novelty check only. Nearest: SIM-26 unit 2 (bandwidth response, running), SIM-25 (eligibility strata), SIM-21. Differs: Changes the kernel's neighbour rule by measured local support; SIM-26 measures bandwidth sensitivity and SIM-25 restricts the candidate pool.

## Next
First unit for one agent: For each trailing and leading state cell, leave-game-out compare the drawn drive distribution (TD, FG, punt, turnover, elapsed) with the real held-out drives under the fixed bandwidth and under support-based neighbour sets, and report the TD-surge gap by cell with game-cluster intervals.

## Open
Rules: measured mechanism only, never a tuned constant or offset; aggregates are checks, not targets; each stage compared with real play-by-play with game-cluster intervals; flag-off hash must equal HEAD before any flag-on run; unresolved results are reported as measured. Look count and family declared before any sign is seen. Data: Drive pool and kernel code in scripts/sim10_drive.py draw() (local).
- 2026-11-03 18:00 ET (root): premise not yet established: the 'smeared trailing states' finding came from a kernel replication with the wrong bandwidth config (see SIM-26 lane correction). Start only after SIM-26 unit 3 reports with the engine's regime bandwidths.
- 2026-11-03 18:01 ET (root): premise now measured with the engine's own regime bandwidths (SIM-26 unit 3): regime 4 (last 2 min) score-diff bandwidth 178 (flat) vs leave-game-out optimum 3.2; regime 3 18.8 vs 7.4; kernel-stage trailing-minus-leading late TD gap .108 -> .031 with measured bandwidths. Next unit: refit regime bandwidths (extended grid, engine criterion and leave-game-out MSE) into a new bandwidth file, Tier B behind --h.
- 2026-11-03 18:11 ET (root): after the bw_v2 Tier B, run a JOINT arm (SIM10_TRUNCTILT=1 plus --h artifacts/sim10/bandwidths_week_v2.json, label trunc_bw) seeds 11-13 and adjudicate with tierb_cmp.py trunc_bw 11,12,13: the truncation fix alone raised pts_game to 45.72 (SIM-21 unit 3), so the two late-game fixes are judged together.
