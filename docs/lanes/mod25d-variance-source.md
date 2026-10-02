# mod25d-variance-source

## Goal
Make the strength generator realistic enough to produce training data: margin variance, strength-to-margin link, count-points coupling. Parent docs/lanes/mod25-synthetic-data.md. Rule: nothing targets mass at 3/7/10/14/17; shape is a held-out diagnostic.

## State
Code scripts/mod25d_variance.py (commands decomp, grid, gate, possdiag, simcache, strdiag, epadiag, and older diagnostics). Outputs artifacts/mod25d/ (run_h3.sh, run_i1.sh, gate_g6/g7, grid_crf/cri, strdiag_s1/s2/s3, epadiag_e1, possdiag_i1, conv4.py). Engine and src untouched.
Variants: crj = crg + chain2 (see D5). crf = crp04 + cal4 (pace latent, 4th-down class calibration). New: crg = crf + cal4s (4th-down class by score x time x distance); crh = crf + chain; cri = crg + chain. chain = per-play EPA recomputed from the sim's own state path with an EP surface fitted on real train (EP_k = drive net points minus remaining EPA, binned down x dist x yardline), so sim EPA conserves its own points.

## Tried (measured 2026-10-01)
D1 late-season R2 gap is not stats convergence: corr(rolling state, leave-game-out season EPA diff) w10-18 .853 sim vs .857 real; per-game team EPA SD around season mean .181 vs .186; between-team SD .097 vs .093 (strdiag_s1). QB shock gap -.041 vs fitted -.057 (small), schedule not tested, fitted strength near-constant (offense drift small).
D2 cause: sim points follow sim EPA less tightly. Same-game margin on EPA diff: corr .874 vs .925/.939, residual variance 55 vs 31/25 (strdiag_s2); drive points minus drive EPA variance 4.44 vs 3.0/2.8 in every outcome class (epadiag_e1). Excess about 27 matches the excess non-strength variance. Cause: sampled play EPA is not path-consistent with the engine's yard-shifted state path.
D3 chain fix cri @ scale 1.0 (grid 1.0/1.25/1.5 loss 2.5/38/116, edge of grid): gate g7 R2 .094/.131/.133 all in band, autocorr .164 in band, home edge, mass17 in band. Costs: margin SD 16.2 (band 13.97-14.87), non-strength var 231 vs 178, mass3/7/10/14 .111/.074/.039/.033, le3 .192 worse. Cause: chain EPA too smooth (within SD .150 vs .186) so points per EPA 61 vs 46-49 real; one scale cannot match both EPA spread and margin SD.
D4 cal4s no effect: Q4 downs-end .156 vs .158; Q4 trailing go .485 vs .447/.519 and overall fail .479 vs .46/.51 are already in band. Excess is short-distance 4ths (dist<=2 share .63 vs .49/.55; fail there .449 vs .35/.39), i.e. 3rd-down yardage arrival and short conversion, not decision rates.
D5 (measured 2026-10-01) chain2 = nflverse-style EP: HistGradientBoosting on real 2009-17 nflverse ep (downloaded to scratchpad nv/, not in snapshots), inputs down, dist, yardline, half secs, score diff, both timeouts, half; fit 2009-15, held-out 2016-17 R2 .9795 MAE .202 (state-only down/dist/yl .9497, .300); saved artifacts/mod25d/ep_model.joblib (epfit.json). Sim EPA = EP(next state, flipped on turnover) - EP(now), points on scores (TD 7), 0 at half/game end; timeouts captured by wrapping ns DECIDE. Scale grid .6/.75/.9/1.0/1.15 (grid2_j1.json) interior optimum .9. At crj@.9 (strdiag_s4): within SD .186 (real .186), between .092 (.093), pts per EPA 49.9 (46-49), margin resid var on EPA diff 26.9 (25-31), EPA diff SD .302 (.276/.295): all consistency targets met. Gate g8: R2 w1-4/w5-9 and autocorr, home edge, mass17 in band; r2_w10_18 .112 (band .1315-.166, grid at .9 gave .130, seed noise), margin SD 15.89 (band 13.97-14.87), non-strength var 227 vs 178, mass3/7/10/14 .116/.073/.043/.034 below band, le3 .202 below. Inferred: total SD 13.98 matches (13.85-13.87) but margin SD/total SD 1.14 vs 1.06 real, so within-game team-score covariance is about -14.5 sim vs -3.3 real: the excess is anticorrelated team scoring, not EPA scale or strength.

D6 (measured 2026-10-01, covdecomp_k1.json, downdiag_k3) real shared conditions, cross-fitted by season, 2009-17 (2304 games; real home/away score cov -7.7, margin SD 14.88, total SD 13.81): explained cov(home,away) in points^2: dome .21, wind .46, precip .02, turf -.30, temp -.11, crew penalty rate .007, all env together -.09; game pace (plays, endogenous) 2.81. Env is not a mechanism for the gap (each below 7% of it). Neutral sim (no strength, drive-level) cov -1.7 vs real -9.3/-5.1: the extra anticorrelation comes from the strength layer, not possession: sim latent margin var 62 (slope 52.5, SD .150) vs real strength var about 17-32 (real reg margin var 223/208 minus neutral sim 191). Possession: plays cov A,B -47 sim vs -36 real, plays SD of diff 16.9 vs 15.5, drive corr .86 vs .90 (time of possession unusable: real clock_elapsed outliers).
D7 downdiag: sim 3rd/4th arrival too short (dist mean d3 6.16 vs 7.2/7.0, d4 6.2 vs 7.7; 2nd 7.18 vs 8.0), 3rd&1 conv .54 vs .66/.69, 4th&1 .54 vs .64/.67; 4th dist<=2 share .265 vs .19. Finer cache rounding (crm) has no effect (rejected). crn/cro (neighbour dist scale 2.5/1.25, SCALE_YDSTOGO, downdiag_k3): 3rd&1 conv .545/.566 vs crj .540 (real .66/.69), 4th&1 .565/.568 vs .542 (real .64/.67), arrival distances and dist<=2 share (.268/.265 vs .19) unchanged: a small conversion gain, not the arrival cause. The 4th-down excess is arrival (sim has fewer long 3rd downs: share 10+ .158 vs .19) not confirmed fixed.

## Next
In flight: artifacts/mod25d/run_k2.sh (covsim crj@.9 -> covsim_k1.json with sim final-score cov, pace-explained cov, plays trade-off; then gate crj,cro @.9 -> gate_k4.txt); done marker run_k2_done.txt. Read both before deciding. Then: (1) strength layer is the anticorrelation source: reduce latent-to-margin slope/size by fitting strength scale to EPA-diff SD (.276/.295) and between SD, not margin; (2) arrival of long 3rd downs (penalty/sack yardage tail) before short-yardage conversion; env mechanisms are not worth implementing (explain <0.5 of ~7). No mass targeting.

## Open
Red-zone TD defense tilt spread unstable. Pace latent game-level only. Poss_analysis side-assign artifact affects old possession-level numbers. EP model lacks nflfastR roof/spread inputs (R2 .98 ceiling here).

## Orchestrator caution (MOD-25k)
The "real strength-driven margin variance 17-32" is inferred by subtracting the
neutral-sim variance, which assumes sim noise equals real noise and so is
circular. Use the random-effects decomposition (strength-difference variance
52/58, noise 153/165, measured in MOD-25d). Sim strength variance of about 62 is
then near real; the excess is in the neutral sim's noise (about 191 vs 153-165).
Before shrinking strength, check this.

## Gate k4 (measured, crj and cro at .9, held out 2018-2025)
Plateau across the last four units. Margin SD is 15.5-15.6 (band 13.97-14.87),
non-strength variance 216-220 (real 178), and late-season R-squared .104-.107
(band .132-.166). Mass at 3 is .113-.115 and share decided by 3 or fewer
.200-.203. Passing: autocorrelation, early and mid R-squared, home edge, mass
at 17. Orchestrator decision: the distillation redesign (MOD-25 pipeline v2)
learns from noiseless latent labels, so margin noise and shape don't enter its
labels. What it needs is a realistic link between features and strength, which
matches (rolling-state vs season-EPA correlation .853 sim vs .857 real; EPA
consistency targets match). Run v2 on crj now. Keep the noise hunt open at
lower priority (3rd-down arrival, short-yardage conversion, residual noise).
