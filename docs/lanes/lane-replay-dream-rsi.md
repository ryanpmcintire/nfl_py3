# Lane replay (Dream-RSI) — ENG-45

## Goal
Decide whether the Dream-RSI idea (replay a recorded discovery tree offline to
score alternative exploration policies, redeploy the best) can steer this
project's lane allocation. Unit 1 asked whether a family's first cells carry
any information about its eventual resolution. Done when unit 2 either shows
a replay-chosen allocation policy beats the hand-set loop on resolutions per
cell out of time, or the row closes on an admissible ground.

## State
- 2026-09-16: ROADMAP row ENG-45 (Phase 11) carries the three-unit design.
- Unit 1 DONE. Script `scripts/lane_replay_feasibility.py` (read-only over
  the registries); artifact `artifacts/lane_replay/20260916T144228Z/`
  (`report.md`, `results.json`). Two cells recorded in family
  `lane_replay_feasibility_v1` (`weak_signals.json`, 6,580 cells now).
- The subagent's full-suite run found one pre-existing failure
  (`tests/test_board_humanised.py`, a recent registry cell without a plain
  summary: `audit_verify_nested_flag_screen_loso` from the in-flight audit
  lane); fixed by adding the summary, file re-saved through `save_registry`,
   test file passes. Registry diff is now 84 added lines, no rewrites.
- Committed in 786a569.

## Tried
- Tree from the registry: 195 named families, median 3 cells, max 1,800;
  only 3 `superseded_by` edges, so ordering is `recorded_at` only. 22 of 195
  families resolved (base rate 0.113); resolutions per cell fall from 0.027
  in 3-4-cell families to 0.003 in 10+ (descriptive).
- Early-cell predictor, 6 looks (k in 1,2,3 x 2 predictors), AUC vs eventual
  resolution with family bootstrap and a within-count-bucket permutation null:
  max |effect/SE| over first 3 cells AUC 0.688 [0.476, 0.860], P+ 0.964,
  perm p 0.046 (36 families, 12 resolved); mean |P+ - 0.5| AUC 0.586
  [0.443, 0.733], P+ 0.877, perm p 0.246 (80 families, 18 resolved). Both
  `unresolved_below_power`. Caveats: 2 of 12 resolved families carry their
  closing cell inside the first 3 (mild look-ahead); `standard_error` is
  missing on 4,441 of 6,578 cells so the stronger predictor sees only 36
  families; rotation registry adds one resolution (1 of 455 families has a
  non-unresolved verdict).

## Next (unit 2 DONE 2026-09-16, subagent-measured, parent-verified by rerun)
- Script `scripts/lane_replay_allocation.py` (imports the unit-1 tree
  builder); artifact `artifacts/lane_replay/20260916T192641Z/`.
- Spec split 2026-08-01 is unusable: 0 pre-cut families, nothing fittable.
  Operational split 2026-09-05: pre 50 fam / 3 resolved, post 148 fam /
  20 resolved; pre fit degenerate (2 scored families, all cuts tie at
  zero, frozen cut arbitrary).
- Out-of-time at matched 977-cell budget: recorded 0.00307 rpc (3),
  breadth 0.01228 (12), refine 0.00819 (8). Refine-recorded +0.00512
  [-0.00102, +0.01126], P+ 0.960, permutation p 0.164; refine-breadth
  -0.00409. Full-spend all reach 20/20: order-only differences.
- No registry cell: no valid unit for resolutions-per-cell; outcome
  lives here and in the artifact. Unresolved by default (p 0.164,
  degenerate pre-fit). Unit 3 (daemon reads a policy) not recommended
  on this evidence.

## Open
- Whether the owner wants agent-call cost recorded per lane (not recorded
  today) or accepts cells-per-family as the proxy.
- Whether unit 3 (daemon reads the policy) is wanted, given the lane count is
  set by hand today.
