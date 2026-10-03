# mod25e-generator-fidelity

## Goal
Keep improving the MOD-25 generator (owner 2026-10-02). Gate failures to
attack: margin SD (band 13.97-14.87), non-strength variance (real 178),
late-season R2 (band .132-.166), held-out shape (mass 3 real .144). Parent
docs/lanes/mod25-synthetic-data.md; prior docs/lanes/mod25d-variance-source.md.
Rules: nothing targets mass at 3/7/10/14/17; aggregates (pts/g, margin SD,
variance) are checks only. Owner 2026-10-02: NO compensating constants; every
parameter is a named mechanism fitted to its own real behaviour.

## State (measured 2026-10-02, gate @ .9 held out 2018-25)
| variant | SD | nonstr var | r2 w10-18 | mass3 | pts/g |
|---|---|---|---|---|---|
| crk06 = no bias + 1-yd dist matching (E1) | 15.41 | 213 | .122 | .114 | 41.9 |
| crf4 = crk06 + 4th-down bug fix (E4) | 15.16 | 205 | .111 | .118 | 42.5 |
| crf4m = crf4 + gain 6.5 + scale 1.0 (E7, 3 seeds @1.0) | 16.16 | 216 | .197 | .108 | 42.5 |
| crf4 3 seeds (E5) | 15.16 +-.12 | 207 +-2.4 | .112 +-.005 | n/a | 42.0 |
Code: scripts/mod25d_variance.py (DV variants, gate, downdiag), mod25e_budget.py
(variance budget), mod25e_scorestate.py (css/css2), mod25e_deficit.py.
Artifacts: artifacts/mod25d/gate_e1*, gate_e4a.json, downdiag_e1*/e4a;
artifacts/mod25e/budget.txt, deficit_e4*; artifacts/mod25e3/.

## Tried E1-E14 (measured; details in artifacts/mod25e, mod25e3, mod25d)
- E1-E4: yard_bias removed, 1-yd dist matching (3rd&1 run share .66 vs .72), 4th-down engine bug fixed (0.93 spurious/g). Budget: excess margin var was cross-quarter cov (+16 vs -12) and drives/g; per-drive variance matches; play choice/tempo match; gap is drive outcomes (leader TD .269 vs .230, punts .352 vs .405).
- E5-E7b: cf4s merge improves xq cov but worsens strength RE/r2; gain 6.5 with scale 1.0 double counts (SD 16.2); derived yard_gain 0, scale 1.0 (crf4 equivalent).
- E9-E11: late-R2 gap not feature/drift; E10 joint P,T fit shows no conversion-gain gap (ratio 1.57 vs 1.35 is regressor noise); garbage-time/RZ-TD T gaps are era drift; RZ fp resolution (bin 5 yd vs exact) no effect.
- E12: margin resid vs EPA: pool 2009-17 sim 26.4 vs real 28.4 (no excess); 2018-25 excess sits in return TDs (sim over-produces) and sim lacks kickoffs; EPA credit equal.
- E13 dual-era gate (pool 2009-17 + held 2018-25) in mod25e_era.py. crzhk 3 seeds vs pool: SD 15.13 vs 14.63, nonstr 208 vs 189, r2 w10-18 .104 vs .146, mass3 .120 vs .141, strength RE 47.6 vs 58.0, noise 181 vs 165, xq cov +4.1 vs -6.4. Margin var and drives/g matched (era artifacts). Era drift 2009-17 to 2018-25: 4th-go .124 to .191, RZ TD .569 to .605. Design: leave-season-out pool, era-weighted by held-out play likelihood; MOD-25 distilled challenger must be rebuilt (old results void).
- E14: no persistent margin component outside scrimmage EPA; ST channels tiny; sim off/def latent corr .095 vs real .213 (fixed in E15).

## E15 (joint off/def latent; scripts/mod25e_cov.py fit|sim|e5|simfeat|channels, artifacts/mod25e3/cov, era_cov/era.txt)
- Read: generator drew levels, carry and weekly drift with ONE shared corr (season-mean corr of off, dfn, -.124 raw, noise-diluted); fixed here by cross-lag covariance (lags 1-6, 2009-17, season bootstrap): corr -.228 in dfn units = +.228 in scrim_def orientation [-.49,-.009] pp neg .98, matches split-half .213.
- Variant crzhc = crzhk + joint cov (levels, carry, stationary drift cov); SIM_FAST byte-equal (1 world x 1 season). 4x8 sim true rho .304 [.19,.43] (old .095), net var 40.7 (real 37.6, old 35.1), V_M 27.9 vs 28.8.
- e5 2 seeds vs pool 2009-17 (crzhk 3 seeds): strength RE 55.0 vs 47.6 (real 58.0; held 52.3), noise 178 vs 181 (165), late r2 .127 vs .104 (.146), xq cov +8.2 vs +4.1 (-6.4; held -12.0), SD 15.26 vs 15.13, mass3 .117 vs .120.
- Not fixed: noise +13 and xq cov +14 vs pool; rho overshoots slightly; lagged season-mean cov model/obs .56/.39 (carry untested).

## E16 (personnel mediation; scripts/mod25e_revert.py, artifacts/mod25e3/revert/revert.txt, revert.json)
- Measured, 2016-25 Q3/Q4 pass/run plays (161.7k, 2600 games), residual EPA vs leave-game-out team strength (off+def), starter index = mean prior-game snap share of the 11 on field (dev from team-season mean), wp-decile x quarter FE, game bootstrap 400.
- Personnel effect: off starter share +.387 EPA per unit [.30,.46] pp 1.00; opposing defence share -.280 [-.36,-.20] pp 0.00 (backups are worse).
- Leader offence reverts: wp>.98 r -.104 [-.123,-.085], .90-.98 -.047, .78-.90 -.019 (pp .03). Personnel explains offence -.024 (23%), but trailer-defence backups give back +.011: net 13%. Unexplained -.091.
- Leader defence does NOT revert (wp<.01 offence r -.032: leader D better, backups +.023 offset); trailer urgency rise absent (wp .01-.46 r +.003..+.012, pp .6-.8).
- Mechanism not personnel: effort/clock-management/prevent on leader offence. Next: condition leader-offence kernel on lead, check run-clock play choice.

## E17 (leader reversion in the sim; scripts/mod25e_revert_sim.py sim|an|drawn, artifacts/mod25e3/revert_sim/revert_sim.txt, drawn.txt; crzhc 2 worlds x 6 kept seasons, 3264 games, SIM_FAST=1)
- Measured, Q3/Q4 residual EPA vs leave-game-out strength by fitted-wp decile (state-only wp from real pool, same for all three): top bin wp>.99 pool -.095 [-.117,-.071], 2016-25 -.100, sim -.088 (sim-pool +.007, pp .69; vs latent -.094); .90-.99 pool -.051, sim -.031 (+.020 pp .90). The sim DOES revert at play level, so the E16 gap is not a missing play-EPA mechanism (hypothesis of a kernel erasing it: not supported).
- Drawn rows (pool 2009-17, pregame rating = engine team_row): residual vs pregame rating -.090 (>.99), -.048 (.90-.99), same as vs LGO; strong teams are not disproportionately conservative.
- New gap: strength passthrough (slope of EPA on strength) in decided states real .86 vs sim .56 at wp .90-.99 (pp .01; 2016-25 .87 vs .56 pp .00) and trailing .004-.075 .85 vs .50 (pp .01); other bins match (10 bins x 2 refs = 20 looks).
- Inferred mechanism: state neighbours (K 40) are sparse in decided states, so the team kernel cannot reweight; fix = state-dependent K or h chosen by held-out play likelihood (no build). xq cov +8 vs -6 is then drive/scoring-sequence, not play EPA.

## E19 (cross-quarter cov by drive-sequence component; scripts/mod25e_xq.py, artifacts/mod25e3/xq/xq.txt, xq_diag.txt, xq_common.txt; crzhc 3264 games vs real pool 2304, 2018-25 2127)
- Measured exact sum (drive points = mean + LGO strength + start-fp bin + lead-bin + rest, drive-quarter): xq total real pool -10.5 / 2018-25 -12.9 vs sim +8.3; gap +19.7 [8.6,33.0] pp .99 (rows sum over other components).
- Gap by component (sim-pool): count -0.2..+0.2 pp .62 (possessions by lead/quarter match: leader share .457/.456), strength +0.6 pp .59, start-fp +3.4 pp .99 (sim leader starts ~3 yd worse field position, no kickoff), lead-state +4.6 pp 1.0 (0 vs 2018-25), rest +11.0 [0,22.9] pp .95.
- Rest decomposes: same-team cross-quarter +18.6 real vs +17.7 sim (equal); opposite-team +13.1 real (+20.5 late) vs +4.4 sim, gap -8.7 [-17.9,-1.2] pp .97 (-16.0 late pp 1.0). Real teams' residual scoring is shared within a game (corr home/away mean residual +.064 pool, +.066 late vs sim -.012; game total residual var 236/233 vs 203).
- Inferred: missing component is a game-level scoring environment common to both teams (weather, pace, game-script, total), not persistent strength and not possessions; it cancels in the margin, so its absence leaves margin cross-quarter cov too positive. Fix: fit a per-game common drive-outcome shift with its cross-quarter persistence from real residuals; state-dependent kernel (E17) is second. Next: fit and sim it, check SD/xq/noise together.

## E20 (shared scoring environment; scripts/mod25e_env.py an|show|fit|sim|e5|xq, artifacts/mod25e3/env/env.txt, env_fit.json)
- Measured, real home/away offence drive-residual mean cov (c_rest of E19), pool +.0514 [.024,.079] (corr .064), 2018-25 +.061, sim -.010; pool minus sim +.062 [.017,.109] pp .99. Cross-fit by season shares of it: wind .03 (.09 late), dome/turf/temp .01-.03 each, crew ~0 pool (.06 late), pace/market total ~0 or negative, own/opp snap load negative (endogenous), field position +.28 (endogenous); all env together negative; unexplained remainder 1.3x the total. Crew and weather are not the source.
- Quarter structure (H_q x A_r): real same-quarter mean +.001, cross-quarter +.052 (late +.088), sim -.058/+.016; phi unresolved (boot 0-1), point 1.0, sigma2 .039. Cells with Q4 are largest (+.15..+.19) in real and sim alike.
- Pace latent already exists in crz (crp04, sigma .04, one multiplier per game shared by both teams).
- Built (not yet validated): game-level shift of both offences' rating in the tilt draw, sd = sqrt(real-sim cov)/(slope b 1.094 EPA per latent [1.05,1.13] x 5.75 plays/drive) = .0396 (envsd in env_fit.json), constant across quarters (phi point est), seeded per task. Variant crzhe.
- In flight/next: 1x1 SIM_FAST checks in artifacts/mod25e3/env/{orig,zero,on} (zero must equal orig; on must differ), then `sim --worlds 2 --seasons 8 --workers 3` + `xq --sim-dir`, then `e5 --seed 11/12` + mod25e_era.py with ERA_VARIANTS incl crzhe.
- E20 update (measured): wrapper moved to per-game ratings shift in run_one_game (pick wrapping broke sim09_half frame lookup). SIM_FAST 1x1: envsd 0 equals crzhc (games and plays byte-equal), envsd .0396 differs. 2x8 sim: .0396 gave cov(H,A) +.016 (sim before -.010; target +.051): transfer gain .66 of nominal (open-loop gain inversion, not a target tune), so envsd .0602 gave cov +.042, corr .046 vs real .064 (artifacts/mod25e3/env/full, full2). e5 seeds 11,12 (envsd .0602) + era.py (ERA_OUT=env_era, crzhc vs crzhe) in background job bmqwyhdgo; read artifacts/mod25e3/env/era.log.
