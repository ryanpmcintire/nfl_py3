# MOD-53

## Goal
Ridge penalty at the grid edge: extend the base penalty grid, report effective degrees of freedom, and test a reduced-rank base (ROADMAP MOD-53).

## State
Added 2026-11-03 (backlog expansion 9), not started, nothing run. Mechanism: MOD-23 units 16 and 17 chose the ridge penalty at the top of the grid (1000) in every fit from 2011 on, so the optimum may lie beyond the grid and the base may behave like a few-factor model. Measured mechanism: the held-out error curve over penalty and over the number of retained principal components, with effective degrees of freedom per fold. Fitted term: Base model refit with the penalty (or component count) chosen leave-one-season-out on a grid that brackets the minimum; no new input and no side rule.

## Tried
Novelty check only. Nearest: MOD-23 units 16 and 17 (chosen penalty at the grid edge, noted but not followed up), unit 9 (per-side alpha). Differs: No prior arm extends the penalty grid past 1000 or fits a reduced-rank base.

## Next
First unit for one agent: Extend the grid to 10^(-1..6) and to component counts 1..30, report the chosen penalty and effective degrees of freedom per outer fold, the held-out error curve with a bracketed minimum, and the 2011-19 and 2020-25 opener-test record, log loss and Brier against the base.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; every penalty, weight and cut point chosen leave-one-season-out; every arm counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded on the fair opener test (base 802-701 in 2020-25 and 1142-1089 in 2011-19 reproduced first, opener not a model input) on the same market and simple-model baselines. Pinned model source stays untouched: run in a scratch harness script, no src edit unless the result is adopted. Look count and family declared before any sign is seen. Data: Base 88-input table and the unit 16 and 17 scripts (local).

## Unit 1 (2026-11-03 18:03, in flight)
Script scripts/mod53_penalty_grid.py (selfcheck vs sklearn, base 802-1503 abort). Family mod53_penalty_grid_edge, declared before outcomes: arms A (ridge, grid 10^(-1+.5i) extended until slope dof < 1 from the training Gram eigenvalues; inner LOSO over training seasons), P (PCA of standardized inputs, k 1..30, inner LOSO, PCA refit inside each inner fold), I (intercept-only control) x 2 eras = 6 looks. Fewer than 3 seasons falls back to alpha 10. Launched background (task ber99i0ia); log artifacts/mod53_unit1_run.log ends "exit N"; find the process by CommandLine mod53_penalty_grid. Outputs artifacts/mod53_unit1/. On exit 0: `nfl-ats weak-signals record --batch artifacts/mod53_unit1/weak_signals_batch.json`, then fill Tried.

## Unit 1 result (measured 2026-11-03 ~18:35; artifacts/mod53_unit1/report.json, A_fits.csv, P_fits.csv; log artifacts/mod53_unit1_run.log)
Base 802-1503 reproduced; selfchecks (closed form = sklearn Ridge, inner LOSO = sklearn, dof = trace, PCA inner = SVD OLS) passed. 6 looks recorded unresolved_below_power (family mod53_penalty_grid_edge; "base model").
A (grid extended until slope dof < 1; chosen lam median / slope dof median per outer season): 2011 3162/11.1; 2012 20811/4.3 (16 of 38 fits on grid top); 2013 3162/18.7; 2014 1e4/10.6; 2015-18 31623/5.4-7.4; 2019 1e5/3.2; 2020 1e5/3.4; 2021 1e5/3.7; 2022 2.1e5/2.7; 2023 1e5/4.3; 2024 31623/10.7; 2025 31623/11.2. Top-edge fits 19 of 560; bottom 0: bracketed. Inner LOSO MSE chosen vs alpha 10 (2020-25) 173-178.6 vs 175.6-182.7; train MSE 172-178 (in/out gap ~0.5-1).
Results (arm vs base ridge a10; decisive flips W-L; P+ of d_acc season bootstrap; LL arm/base/market; Brier arm/base/market):
A 2020-25 768-735 vs 802-701 d -2.26 [-2.83,-1.63] P+ 0 flips 199-233 LL .6936/.6969/.6933 Br .2502/.2517/.2501; 2011-19 1131-1100 vs 1142-1089 d -0.49 [-1.66,.76] P+ .21 flips 305-316 LL .6956/.7195.
P (PCA k, 1..30): k median 3 (1 early, 8 in 2024-25), never at 30; 2020-25 766-737 d -2.40 [-4.14,-.76] P+ .0003 flips 197-233 LL .6945 Br .2507; 2011-19 1132-1099 d -.45 [-2.07,1.47] P+ .29 flips 356-366.
I intercept-only control: 2020-25 756-747 d -3.06 [-3.82,-2.26] P+ 0 flips 215-261 LL .6942 Br .2505; 2011-19 1135-1096 d -.31 P+ .33.
Reading (inferred): bracketing lowers held-out MSE and log loss toward market, but picks lose accuracy 2020-25; accuracy loss scales with shrinkage (a10 802 > A 768 > P 766 > I 756). Base signal sits in low-shrinkage directions that inner MSE under-rewards. Not closed: no wrong_sign_resolved designation made (pick-accuracy is one metric; MSE/LL improve).
Served-model implication (describe only): a pinned refit with the bracketed penalty would change ~15% of 2020-25 picks vs a10 (flips 432) for an LL gain ~.003 and accuracy loss ~2 points; not recommended on accuracy.

## Next
Selection criterion mismatch: choose penalty by inner-LOSO decisive-pick accuracy or log loss of the cover probability (mechanism: MSE on margin weights blowouts); or per-direction shrinkage (penalty rising on low-eigenvalue directions only).
- 2026-11-03 18:09 ET (root): the two 2020-25 accuracy cells (bracketed ridge -2.26 [-2.83,-1.63], reduced-rank PCA -2.40 [-4.14,-.77]) re-recorded refuted_mechanism / wrong_sign_resolved (whole interval below zero; same precedent as MOD-23 u16 elastic net); the 2011-19 cells had seasons recorded as 2020-2025 and were re-recorded with 2011-2019 (still unresolved). Conclusion: margin-MSE-optimal shrinkage of the base ridge lowers pick accuracy at the opener; the base's light penalty is not an accident. Unit 2 tests penalty selection by inner-LOSO pick accuracy / cover log loss.
