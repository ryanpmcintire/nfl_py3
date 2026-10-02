# mod25d-variance-source

## Goal
Make the neutral/strength generator realistic enough to produce training data: margin variance, strength-to-margin link, count-points coupling. Parent docs/lanes/mod25-synthetic-data.md. Rule: nothing targets mass at 3/7/10/14/17; shape is a held-out diagnostic.

## State
Code scripts/mod25d_variance.py (commands decomp, tilt, grid, gate, ipwcheck, corrdiag, pacediag, ratediag, tiltcoef). Outputs artifacts/mod25d/ (run_*.sh, gate_*.json/txt, grid_*.txt, tilt_coef.json, diag/). Engine and src untouched. Sim cache in scratchpad.
Variants: crt_no* ablations (late, fourth, pat, fine, ipw, clean); crp04 = crt + per-game pace latent (sigma .04 on clock_elapsed, own target). Earlier: c = MOD-25c; cr = c + shrunk team-season EPA effects in kernel; cre = cr + epa residualized; crw = cre + inverse-propensity pool weights; crk = crw + clock artifacts cleaned (E1); crt = crk + outcome tilts (E2).

## Tried
Real decomposition: margin var 221/205 (train/eval), random-effects noise 165/153, strength var 58/52. Variants c..crt built D1-E2 (selection-leak fix via BLUP residualisation and propensity weights; clock-artifact cleaning; outcome tilts). crt gate scale .75: SD 15.16, nonstrength 208 vs 178, R2 w10-18 .094 vs .150.
F1 (this round, measured): the "over-strong score-state reversion" was a MEASUREMENT ARTIFACT. poss_analysis side assignment (assign_sides) mislabels sides in about 17 percent of sim games by 300 s (logged points disagree with running score; |disc| 3.2 at 300 s, real .36). Tracked sides from score continuity (track_sides, SIDE_TRK) give sim crt slopes at 2700/1800/900/300 s of -.137/-.113/-.127/-.087 vs strength-removed real (leave-one-game-out BLUP) train -.082/-.105/-.110/-.077, eval -.125/-.116/-.125/-.091 (SE .01-.04). Ablating late, 4th-down, 2pt, kernel, ipw, clean each leaves -.06..-.15: none is a source, no fix needed (artifacts/mod25d/ablate_h2.json, ablate_real.json). Play-level lead/trail points, plays, seconds per play and pass rates by lead bin match real (revdiag.json). cov_total with tracked sides -96 (sim) vs real -39/-66 raw, strength adds about +55 so strength-removed real is about -95/-120 (inferred): matches.
F2: pace latent. Lognormal sigma on clock_elapsed per game (crp04): quarter play covariance -.06 -> 2.03 (real 2.06/1.47), plays sd 8.0 -> 9.95 (10.1/9.5); sigma .08 overshoots (8.8). Side effect: corr(plays, points) .197 -> .232 (real .05). Grid crp04 loss .6 20.6, .75 11.4, .9 12.2 (scale stays .75). Gate g5 crp04 vs crt: indistinguishable (SD 15.14 vs 15.16, nonstrength 208.8 vs 208.5, R2 .054/.109/.096 vs .076/.109/.094, mass3 .119 vs .120, le3 .208 vs .213).

## Next
Verdict: not ready. Fails: margin SD 15.1 (13.97-14.87), nonstrength variance 209 vs 177.5, R2 w10-18 .096 vs .150, mass 3/7/10/14 and share le3 (held-out, not targeted). Passes: autocorr, home edge, R2 w1-4/5-9, mass17. Reversion and pace are closed. Next mechanism: drive-level scoring variance (points per possession by start field position and drive length; sim possession-count and points-per-possession variance vs real decomp), plus the count-points coupling .23 vs .05 which pace did not fix. Fit to those own targets, then re-run grid and gate.

## Open
Red-zone TD defense tilt spread unstable. Poss_analysis side-assign artifact also affects any old possession-level sim numbers (cov_total, lag covariances): re-read old decomps with SIDE_TRK. Pace latent is game-level only; team-season pace not tested.
