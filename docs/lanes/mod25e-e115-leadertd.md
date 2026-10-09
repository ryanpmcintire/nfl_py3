# mod25e E115 second-half leader TD excess

## Goal
Is the H2 non-trailing TD excess (sim v real) strength persistence, in-game over-performance (ADJ/form), or uniform; name the post-draw wrapper that scales with lead. Base crHpqokgndecsmfwtjo2as2ypw2xfdcdtbkhghkeku s11-13 v real 2009-17 REG. Read-only; no sim run. Scripts and outputs in tests/scratch/e115/ (lat, build, an, an2, gtg, loso, yl, clip, geo; .txt).

## State (2026-10-09, measured; game bootstrap 90%, 300 reps; ~150 looks, one family: TD-rate decomposition)
- Sim latent recovered without running the sim (mod25_generator gen_world_latents/make_schedule from the seed; schedules match 100%). Strength diff (oh-dh)-(oa-da), 54 pts per EPA unit, SD 7.67 pts, corr .49-.51 with margin; real spread SD 5.89, slope 1.09, corr .43.
- TD/drive sim-real by half x state: H1 trail +.005, tied +.002, lead1-8 +.000, lead9+ +.005 (all intervals span 0). H2 trail +.003 [-.003,+.009], tied +.028 [+.015,+.043], lead1-8 +.020 [+.011,+.029], lead9+ +.014 [+.005,+.021]. Excess is non-trailing H2, Q3 largest.
- Strength: with exact latent the H2 leader excess looks favourite-heavy (dog +.008, mid +.011, fav +.029). Exactness inflates it: with a market-like proxy (0.641 x latent + N(0,3.2^2), matched to real SD 5.89 and slope 1.09) it is uniform: dog +.017 [+.004,+.031], mid +.018 [+.007,+.028], fav +.024 [+.015,+.033]. LPM TD/drive on edge, H2 leaders: slope real .0051 sim .0056, diff +.0005 [-.0005,+.0014], P .79; intercept diff +.0118 [+.0066,+.0185]. Level shift, not slope.
- In-game over-performance: surplus (lead minus edge x elapsed) coefficient real +.0003 sim +.00045, diff +.00015 [-.0007,+.0009]; surplus terciles +.030/+.007/+.015 (not monotone). ADJ tilts by opponent cumulative residual EPA, delta .00034 per unit (artifacts/mod25e3/adj/fit.json, mod25e_adj.py:311-316), pushes a leader down. STF acts only on next_yardline after kicks/scores (mod25e_stf.py:230-246). TDC is qtr 2/4 clock only (mod25e_tdc.py:134).
- Zone split H2 (TD/game sim-real): open field (yl>20) leaders +.086, trailers +.109; RZ non-goal-to-go leaders -.026, trailers -.063; goal-to-go leaders +.043 [+.020,+.065], trailers -.009. Open-field TD/play .0150 v .0113 (+.0037 [+.0031,+.0044]) in every state and half. Sim TD plays from yl>20: 31.5% v real 25.8%.
- Goal-to-go: sim run TD/play flat .333-.365 across states; real leader .279 (+.054 [+.040,+.072]). Real TD given (down, play type, yardline) has no state effect: LOSO state terms gain +.00014 nats/play, 6/9 folds, |beta| <= .13 logit. So conditioning the score-free GZ re-pick (mod25e_gz.py:72-117; cells median 20 plays under K_STATE 200, sim04_engine.py:32) would not move TD/play.
- Goal-line pile-up: goal-to-go run share at yl 1 real .24-.33 v sim .38-.48 (share yl<=3 +.125 [+.106,+.144], same in H1, H2 leaders, H2 trailers). Sim non-scoring run/pass from yl 2-5 carry pool yards >= yl in 15.6% (H1) / 20.2% (H2), real 0; yl 2-10 by quarter Q1 .10, Q2 .12, Q3 .15, Q4 .14, flat across states (lead9+ +.01-.02). Sim `yards` is the pool row's, so TD flag and yardage come from neighbour rows at a different yardline than the query (inferred).

## Tried
Strength terciles and fixed-point bins, proxy market, surplus terciles, LPM, zone/state/play-type splits, state-conditioned GTG likelihood. Not examined: QBC centring, F2PR, EGT/EGH, SEL/FD4 effects on H2 leaders.

## Next
Hook (specify, not wired) GEO: after the kernel draw, enforce the goal-line identity in real (TD iff yards >= yl, 0.2% exceptions): re-pick from neighbour rows restricted to the query yardline band (width chosen LOSO by held-out log-lik of {TD, next yardline bin} at yl <= 40, 9 seasons), for all downs, not only goal-to-go. Score on TD share by yl bin, yl-1 share, H2 non-trailing TD/drive; outputs, not targets.

## Open
Leader-specific part of the H2 excess is not located to a wrapper; geometry mismatch is state-independent and grows Q1 to Q3. Pace (E111b) may interact. unresolved_below_power for strength and ADJ; no closure.

## GEO hook (2026-10-09, scripts/mod25e_geo.py; fit artifacts/mod25e3/geo/fit.json, check.json; scratch tests/scratch/e115g/)
- Defect read: sim04_engine.py:996-1012 takes TD flag (points_off), yards (+yard_shift) and next_* from the neighbour row idx; :1121 clamps new_yardline = max(yardline - yards, 1) so yards >= yl on a non-scoring play piles at yl 1; TD rows' arrays yards_gained (:644) are the next drive's start minus yl (garbage), so no yards-vs-yl test exists anywhere. GZ (mod25e_gz.py:72-117) fixes only distance >= yardline via exact-yl pools.
- GEO=1 re-picks violators only; GEO=2 re-picks every run/pass from consistent rows at the exact yardline (band 0, chosen LOSO nested in all 9 folds; grid 0,1,2,3,5,8,13,21 is 8 looks x 3 variants x 9 folds). Held-out log-lik vs engine-like K=200 draw (yl<=40, 91k plays): GEO=1 +.0052 nats/play 9/9 folds, game-bootstrap p05..p95 [+.0036,+.0065], probability_positive 1.0; GEO=2 +.0085 9/9. GEO=1 biases TD low (stub yl6-10 .140 v real .198); GEO=2 stub .197 v .198, yl21-40 .0304 v .0309, violations 0.
- Engine-like violation share of draws: yl2-5 22.7%, yl6-10 13.4%, yl11-40 2.7%. Real TD with yards < yl: 0.21% (tests/scratch/e115/geo.txt).
- Not run in sim. Wiring: crH after cy.install_cky() block, before install_log_sync(dv); label "go".
