# SIM-06 learned drive-outcome head vs SIM-10 kernel (2026-11-02)

## Goal
First measured SIM-06 step: gradient-trained drive-outcome head vs the SIM-10 kernel resampler's
conditional outcome distribution, same drive features and outcome classes, trained 2018-2023,
scored on held-out 2024-2025 drives. Drive-level comparison only; no served-card effect, no engine swap.

## State
- scripts/sim06_learned_head.py (imports sim10_drive read-only; SIM10_SEASONS=2018-2025,
  SIM10_OUT=artifacts/sim06/sim10 built with `sim10_drive.py build`).
- Classes: SIM-10 outcome_classes recipe, tercile/decile cut points taken from training seasons only.
- Kernel A: locked bandwidths_week.json (fit on 2009-17). Kernel B: same kernel, bandwidths
  refit by season-block CV on 2018-23 (nested, no held-out data).
- Head: hyperparameters (MLP width/L2, spline-logit C) chosen by 3-block season CV on 2018-23 only.
- Outputs: artifacts/sim06/results.json, run.log. Results in the final report of this lane.

## Tried
Run done (370 s, 1 process). Held-out 2024-25 (11,443 drives, 543 games, K=94, train 34,429):
log loss/drive learned MLP(64,64) -3.468, logit -3.473, kernel B -3.526, kernel A -3.535, marginal
-4.045. Learned minus kernel B +0.0577 [0.0456,0.0701] P+=1.00 (game bootstrap); 8-season LOSO
+0.0560 [0.0495,0.0627] season-block, every season positive. Recorded:
sim06_learned_head_vs_kernel_heldout_2024_25 (log_loss_improvement, unresolved_below_power).
Calibration: head ECE .00063 v kernel A .00033; head group totals closer on type 7-9, kernel under on type 0/1.

## Next
If learned head wins: swap would need the clock-feasibility mask (kernel uses it, head cannot),
strength tilt (DR targets) re-fit on the head, and a gate rerun of the 14 sim10 gates.

## Open
Kernel clock-feasibility mask has no head analogue; held-out has only 2 seasons so the primary
interval is a game-cluster bootstrap, the season-block interval comes from 8-season LOSO.
