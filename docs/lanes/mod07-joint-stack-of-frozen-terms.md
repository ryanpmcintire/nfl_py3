# MOD-07 joint stack of frozen forward terms

## Goal
ROADMAP MOD-07 (ensemble/stacking). Does ONE ridge logistic on logit(served out_of_season_home_probability, artifacts/pick_probability/20261103T195411Z) plus all frozen forward terms beat served out of season? Related: docs/lanes/mod07-prospective-status-audit.md (done/), docs/lanes/eng-50-online-forward-stacker.md.

## State
Done 2026-11-03T22:56Z. Script scripts/mod07_joint_stack.py (--stage features|stack); artifacts/mod07_joint_stack/{features.parquet,stack_result.json,weak_signals_batch.json}. 1503 non-push REG games 2020-25. Penalty grid 1,10,100,1e3,1e4 chosen by inner LOSO on training seasons; terms standardised on train, missing = 0.
Terms: 9 built (sky03, sky03_separated, sky04, env02, sky01, sky01snap, surface, mod35_cold_exposure, mod41_avgwin_served). Coverage: sky04 2023-25 only; sky01/sky01snap 2024-25 only; env02 retractable-roof home games (227); mod35 outdoor (741); avgwin week 7+ (944).
Result (measured): joint LOSO log-loss gain -0.00141 [-0.00433, +0.00140], P+ 0.178; decisive 77-87 of 164 (behind served); correct 819 v 829; in-sample +0.00424 (gap -0.00564). Intercept-only control -0.00005. Sum of individual gains +0.00005 vs joint -0.00141: no additivity, joint overfits (2023 fold -0.0102 with penalty 1). Individual gains: avgwin +0.00133 P+ .78 (matches MOD-49), sky03_sep +.00031, mod35 +.00026, others <=0; sky04 -0.00069 [-.00129,-.00012] (2023 sign flipped).
Sign stability (6 folds): sky03_sep +6, mod35 +6, env02 +6, sky01snap +6, avgwin -6, sky03 -6, sky04 -5, sky01 -5, surface 4/2.
Recorded: family mod07_joint_stack, 10 cells, unresolved_below_power.
Looks: 11 gain cells, 5 penalties, 6 folds.

## Tried
Single construction, no retuning. Forecast-based terms use pool-decision-cutoff forecasts (as their own histories); sky04 weights/avgwin decomposition are frozen values (fit on overlapping seasons, features not outcomes for avgwin; sky04 weights use outcomes 2023-25).

## Next
Forward version would log, per slate: each frozen term value and delta, served logit, and the ridge-stacked p with fold coefficients, never served. Candidate unit: shrink-only stack with sign-stable subset declared before looking (sky03_separated, mod35, env02, sky01snap) after the penalty grid is re-declared; rerun at Week 18 on matured forward terms.

## Open
Is the joint loss from sparse-coverage terms (sky01*, sky04) adding noise? A coverage-matched joint on 2024-25 would test it (new look, declare first). per13_locked_rest and mod41_residual_load* skipped by instruction.
