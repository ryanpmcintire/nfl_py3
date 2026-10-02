# mod25d-variance-source

## Goal
Make the strength generator realistic enough to produce training data: margin variance, strength-to-margin link, count-points coupling. Parent docs/lanes/mod25-synthetic-data.md. Rule: nothing targets mass at 3/7/10/14/17; shape is a held-out diagnostic.

## State
Code scripts/mod25d_variance.py (commands decomp, grid, gate, possdiag, simcache, strdiag, epadiag, and older diagnostics). Outputs artifacts/mod25d/ (run_h3.sh, run_i1.sh, gate_g6/g7, grid_crf/cri, strdiag_s1/s2/s3, epadiag_e1, possdiag_i1, conv4.py). Engine and src untouched.
Variants: crf = crp04 + cal4 (pace latent, 4th-down class calibration). New: crg = crf + cal4s (4th-down class by score x time x distance); crh = crf + chain; cri = crg + chain. chain = per-play EPA recomputed from the sim's own state path with an EP surface fitted on real train (EP_k = drive net points minus remaining EPA, binned down x dist x yardline), so sim EPA conserves its own points.

## Tried (measured 2026-10-01)
D1 late-season R2 gap is not stats convergence: corr(rolling state, leave-game-out season EPA diff) w10-18 .853 sim vs .857 real; per-game team EPA SD around season mean .181 vs .186; between-team SD .097 vs .093 (strdiag_s1). QB shock gap -.041 vs fitted -.057 (small), schedule not tested, fitted strength near-constant (offense drift small).
D2 cause: sim points follow sim EPA less tightly. Same-game margin on EPA diff: corr .874 vs .925/.939, residual variance 55 vs 31/25 (strdiag_s2); drive points minus drive EPA variance 4.44 vs 3.0/2.8 in every outcome class (epadiag_e1). Excess about 27 matches the excess non-strength variance. Cause: sampled play EPA is not path-consistent with the engine's yard-shifted state path.
D3 chain fix cri @ scale 1.0 (grid 1.0/1.25/1.5 loss 2.5/38/116, edge of grid): gate g7 R2 .094/.131/.133 all in band, autocorr .164 in band, home edge, mass17 in band. Costs: margin SD 16.2 (band 13.97-14.87), non-strength var 231 vs 178, mass3/7/10/14 .111/.074/.039/.033, le3 .192 worse. Cause: chain EPA too smooth (within SD .150 vs .186) so points per EPA 61 vs 46-49 real; one scale cannot match both EPA spread and margin SD.
D4 cal4s no effect: Q4 downs-end .156 vs .158; Q4 trailing go .485 vs .447/.519 and overall fail .479 vs .46/.51 are already in band. Excess is short-distance 4ths (dist<=2 share .63 vs .49/.55; fail there .449 vs .35/.39), i.e. 3rd-down yardage arrival and short conversion, not decision rates.

## Next
Replace the state-only EP surface with one carrying score, time and timeouts (real EPA variance), refit scale on EPA spread plus points-per-EPA slope (45-49), re-gate. Then 3rd-down yardage and short-yardage conversion by down-distance cell, then count-points coupling .17-.23 vs .05-.11.

## Open
Red-zone TD defense tilt spread unstable. Pace latent game-level only. Poss_analysis side-assign artifact affects old possession-level numbers.
