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

## E18 (state-dependent K/h; scripts/mod25e_kstate.py search|sim|e5, artifacts/mod25e3/kstate/{search.txt,search_jointkey.txt,kstate.json,an/revert_sim.txt})
- Measured, real 2016-17 queries (fit 2009-15, K 200, h 0.5 sd): kernel ESS is ~26 in EVERY wp decile (p10 ~10); kth-neighbour distance only 1.5 (mid) to 2.1-2.3 (decided). Sparsity of decided states is NOT the cause (E17 inference refuted for ESS).
- Search (300 looks; K 50-800 x h 0.25-8 sd-mult x 10 wp bins; tune 2009-13, val 2014-15; objective pass LL + yards CRPS + first-down LL, strength term in weights): chosen K 200, h x8 (x4, K 100 in top bins) = strength term off. Test 2016-17: strength term value negative in all 10 bins (CRPS -.17..-.22 [..] pp 0.00; pass LL -.014..-.02); chosen beats current by +.17..+.22 CRPS pp 1.00. Joint-key objective chose K 800 h x4, same direction.
- Built crzhs on crzhc (wrapper byte-equal with null tables, 1 world x 1 season, SIM_FAST=1; earlier mismatch was seed default 31 vs 21). Passthrough 2x8 sim (an/revert_sim.txt): .56 -> .15 at wp .90-.99, .50 -> .26 at .004-.075 (real .86): wrong direction, as chosen h inflates. e5 not run (mechanism fails passthrough check).
- Inferred: pregame-rating kernel has no play-level likelihood power; passthrough must come from a different term (latent strength fed to the play draw, not neighbour reweighting). Next: tilt/regression term on EPA|strength by wp bin, scored by the same held-out play objective.
- E20 result (measured, e5 2 seeds, era_out env_era/era.txt, crzhc vs crzhe vs pool): SD 15.26 vs 15.27 (14.63), nonstr 207 vs 212 (189), late r2 .127 vs .110 (.146), xq cov +8.2 vs +7.6 (-6.4), strength RE 55.0 vs 48.5 (58.0), noise 178 vs 185 (165), mass3 .117 vs .115. The common shift moved the home/away corr toward real (.046 vs .064) but did not move xq cov, and worsened noise/strength split: shared component as a constant additive latent is not the xq fix; quarter-resolved (Q4-concentrated, state-linked) structure is the next lead. Not a keeper.

## E22 (kickoff mechanism; scripts/mod25e_kick.py an|sim|e5|val, artifacts/mod25e3/kick/{an.txt,val_full.txt}, kick_era/era.txt; crzhq = crzhc + state-kNN real kickoff draws, SIM_FAST zero-mode byte-equal 1x1)
- Measured: crzhc ALREADY matches real kicks (transition rows carry post-score starts): retain .029 vs .030, start-fp sd 9.1 vs 9.2 (E14 variance claim is not this). Only return TDs missing (0 vs .0042 pool, .0024 2018-25).
- Real Q4 trailer-scores sequence: retained 2.7% (1.4% 2018-25) vs sim 3.3%; leader scores next .274/.315 vs sim .242; last 5 min .157/.183 vs sim .098 (gap, unfixed: crzhq .071).
- cov(H,A) pair sum real +.036/+.054 vs sim -.017; adjacent pairs after a score carry -.003/+.002 (sim +.002): sequences carry ~0 of the gap; after non-score adjacency +.012 vs -.005 (about 30%, punt/turnover handoffs).
- crzhq (2 seeds) vs crzhc: SD 15.35 vs 15.26, nonstr 210.5 vs 207.0, late r2 .118 vs .127, xq cov +8.9 vs +8.2, noise 181 vs 178, mass3 .117 both. No gain; hypothesis refuted as mechanism of shared Q4 cov. u4g/f2/f3 not composed.

## E23 (late leader drive anatomy; scripts/mod25e_late.py an|state|fourth|clock|fgdown, artifacts/mod25e3/late/*.txt; crzhc vs real)
- Measured: the late-leader gap (.098 vs .157/.183) sits ONLY where the old leader is now behind/tied (n 273/293): .143 sim vs .278/.345 real; leader still ahead matches (.061 vs .047/.037); start field position, plays, run share, FG make rate match. Not prevent, not short field (sim at real start mix .100/.117).
- Same shape generically: last-5-min drives tied or trailing <=8 score .22/.21 sim vs .27/.34 and .43/.48 tied real (fga .17 vs .32/.37); 4th-down go/punt/fg shares by state match real, so not the 4th-down policy. Drives starting <60s: end-with-no-outcome .665 sim vs .42/.40 real, td .026 vs .14/.12; <2:00 FG on downs 2-3 in last 15s .35/.55 sim vs .70/.70-.77 real.
- (b) refuted as a handoff mechanism: P(score) after punt/turnover at fixed start-fp equals any-drive P(score) in each source; punt net 51.0 vs 51.1, sd 15.1 vs 15.7 (short own-territory punts -1 to -2.5 yd); gap is the general per-drive scoring level (.319 vs .349/.384).
- No variant built (budget). Next: fit urgency-state drive yardage/FG-on-early-downs (qtr 4 gsr<=120, tied/trail<=8) from real plays; classifier poll (HGB, mod25_mechanisms.fit_decisions) underfits the final-15s cell.

## E24 (two-minute drill anatomy + clock redraw; scripts/mod25e_drill.py an|sim|e5|val, artifacts/mod25e3/drill/{an.txt,val_*/,drill_era/}; variant drl = f2 + dr)
- Measured (an.txt, Q4 gsr<=300 offence tied/trail<=8): snaps/game match (5.47 real vs 5.23 sim), but clock use in the last 45 s does not: mean el [0,8) 3.3 real vs 9.3 sim, [8,15) 6.2 vs 9.4, [15,45) 8.5 vs 10.0; share of snaps whose el exceeds time left 0 real vs .69/.28/.07 (kNN time axis is 15 s coarse vs the state; neighbours carry el from earlier clock). Kneel [0,8) .046 vs .109, spike [0,8) .098 vs .068. OOB, pass depth, sideline are NOT in the pbp snapshot (56 columns): unmeasurable.
- Built drl: in the 2-min window (Q4 <=120, Q2 (1800,1920]) redraw el from real 2009-17 rows of same half/code/stop/timeout-used class, kernel on gsr, score, off/def timeouts>0 (sd-standardised, sqrt(n) nearest), el=min(el, time left). dr=0 byte-equal to f2 1x1 SIM_FAST (41282 rows both); dr=1 differs.
- Chain done (val_drl, val_f2, e5 s11/s12, artifacts/mod25e3/drill_era/era.txt). drl el [0,8) 0.8 s (real 3.3; f2 8.8), over-time share .172/.114/.028 for [0,8)/[8,15)/[15,45) vs real 0 (f2 .672/.291/.077); kneel [0,8) .107 vs .046, spike .103 vs .098; drives <60s end-share .391 vs .418 (f2 .641), td .044 vs .142; FG gsr<15 downs 2-3 .572/.468 vs .705/.703 real: decision gap remains. Era (2 seeds): drl pts_game 43.5 vs crzhc 42.5, f2 43.0 (pool 45.2); strength_re 46.6 vs 55.0/47.5 (pool 58.0); xq_cov -1.1 vs 8.2/5.0 (pool -6.4); q4_slope -.056 (pool -.060).

## E21 (state-dependent passthrough; scripts/mod25e_pass.py real|an|fit|sim|e5, artifacts/mod25e3/pass/{real.txt,pass.json,pv/pass_an.txt}; crzhc + pick-weight tilt multiplier m, null byte-equal 1x1)
- Measured, Q3/Q4 slope(z)=b0+b1 z (z=|logit wp|/6), game bootstrap: pool b0 .776 [.675,.877], b1 -.206 [-.384,-.032]; 2016-25 b0 .632, b1 +.071 [-.133,.244] (pp .76): wp dependence is era-unstable, so E17's two-bin gap is a pool-era, 20-look read. By down real pool .45/.67/1.03/1.76 vs crzhc .39/.59/.90/1.73 (uniformly ~13% low); sim b0 .648 [.570,.739].
- Probes (2x8, same seed): m0=2 gives b0 .700, b1 -.030; (m0 1,m1 1) b0 .673; wp response too small, two-value solve ill-conditioned (m0 6.3, m1 -6), so ONE value fitted: m0 3.46 [1.5,5.4] (tilt exponent multiplier, all states).
- Validation (m0 3.46): b0 .796 vs pool .776, but b1 -.03 vs -.21 and 3rd down 1.35 vs 1.03 (overshoot; tilt channel loads on 3rd/4th). Decided bins mixed (.90-.99 .73 vs .86, >.99 .71 vs .51). e5 seeds not completed (killed at budget); next: down-specific m or EPA-level tilt, then e5.
- E21b (in flight, built not validated): EPA-level tilt, PASS_MODE=epa in mod25e_pass.py (install_epa_tilt: weights exp(m*(off+def dev)*row_epa), one value m). Probe m=.1 2x8 running: artifacts/mod25e3/pass/chain_epa.sh -> ea/, ea_an.log (down table vs real .45/.67/1.03/1.76). Next: linear-fit m from ea vs base .39/.59/.90/1.73, validate sim at fitted m, then e5 seeds 11,12 (PASS_MODE=epa) + era.py.
- E21b measured (2x8, seed 31): EPA tilt m=.1 overshoots 3rd (1.35, 4th 1.92); m=.03 gives downs .457/.564/1.163/1.208 vs real .45/.67/1.03/1.76, all inside real intervals, b0 .675 (real .776). ONE value m=.03 (inferred range ~.01-.07, not bootstrapped). e5 s11,s12 + era (chain_e5.sh, PASS_MODE=epa, ERA_OUT=pass_era crzhc vs crzhp) running; read pass/era.log, pass_era/era.txt.
- E21b result (e5 2 seeds, pass_era/era.txt, crzhc vs crzhp m=.03; pool real): strength RE 55.0 vs 56.3 (58.0), noise 178 vs 185 (165), late r2 .127 vs .123 (.146), xq cov +8.2 vs +13.6 (-6.4), pts/g 42.48 vs 42.58 (45.2). Passthrough fixed, aggregates not: noise and xq worse. Not a keeper for xq.

## E25 (endgame; scripts/mod25e_endgame.py el|fit|td|sim|e5|val, artifacts/mod25e3/endgame/; variants egl = el fix, egd = el fix + decision model)
- Measured: the drl el recording is not the clock the engine used. The frame logs el at the innermost policy, so drl (cap before base policy) logged the cap while base noise may have moved the real clock; the old over-time read is partly a log artifact. egl/egd redraw after base, integer-second remainder, and overwrite the logged el (lg[-1][10]).
- Measured (fit.txt, train 2009-15, test 2016-17 window): late classifier logloss 0.498 exact / 0.527 grid vs ext HGB (half seconds, half, timeout counts, previous-snap stop) 0.456. egd samples it in window for downs 1-3 (timeout class not modelled).
- TD gap: pass yards/inc/sack in last 120 s match real; flips and scores in last 30 s are half real; clock remainder artifact is the lead suspect.
- Measured egd (6x8 sim): el by gsr band [0,8)/[8,15)/[15,45)/[45,120) 3.37/6.76/9.54/12.22 vs real 3.42/6.21/8.47/12.20, over-time 0 (drl .17/.11/.03); kneel [0,8) .069 vs .046 (drl .107), spike .063 vs .098; last-30 s score plays .042 vs .069 real, flips .099 vs .275 (gap remains, mechanism open).
- e5 2 seeds, eg_era/era.txt (pool; crzhc / drl / egd): pts/g 45.2; 42.5 / 43.5 / 42.7; strength RE 58.0; 55.0 / 46.6 / 47.6; noise 165; 178 / 181 / 182; xq cov -6.4; +8.2 / -1.1 / +3.7; q4 slope -.060; -.037 / -.056 / -.039. egd fixes the clock conditionals but loses drl's xq/slope gain: that gain came from the pre-base cap (unrecorded clock effect), not from fidelity. Next: decide which clock (egd real fidelity vs drl) is the base; the late-lead carry needs a mechanism other than el.

## E26 (rule-piece strength cost; scripts/mod25e_rulestr.py real|sim|cnt|e5, artifacts/mod25e3/rulestr/{real.txt,cnt.txt,chain.sh})
- Measured real 2009-17 (288 team-seasons, slope per SD of strength): timeouts used prior-season strength +.02 [-.04,.08] pp .76 (same-season -.10 is game-script); accepted off penalties per 100 snaps -.12 [-.22,-.02] (mean 4.6, 2.6%/SD); no-play rate +.03/-.01 pp .45-.70. Penalty snaps carry 5.5% of EPA strength slope on 8.2% of snaps. Hypothesis (a) refuted: pieces' rates do not depend on strength, and independent latents could only add team variance.
- Measured crG sim (1x1, 48k draws): f2 replaces 8.8% of run/pass draws with a uniform state-only neighbour (pick in sim09_f2.py ignores team kernel, IPW, EPA tilt): hypothesis (b) for f2. f3 wipes 6.4% (accepted no-play, outcome carries no strength) and counts 1.6% of draws, rows removed from the pool (IPW*~pen).
- Built crW = crG with f2 replacement weighted by the engine kernel x IPW x PASS_W (RS_W=1, WPICK reads off_sim/def_sim from the frame; running in 11+ min of e5 showed no crash). e5 s11,s12 NOT finished (killed at return): rerun bash artifacts/mod25e3/rulestr/chain.sh (~100 min, 2 workers), then read rulestr_era/era.txt (crzhc,crG,crW).
- f3 and drl (-.9, 2 seeds, ~1.8 SD) not fixed: f3 loss is mix of wipe dilution and scoring scale (pts -1% => var -2%); drives +.8 from clock redraw not strength.

## I3 (crG composition; scripts/mod25e_crG.py sim|e5|ref|cmp --off f2,dr,f3,epa; artifacts/mod25e3/crG/, crG_era/era.txt)
- crG = crzhc generator + u4g + f2 + dr + f3 (EP accept) + EPA tilt m=.03; order drl chain -> pass.run_init -> f3_install. Null byte-equal 1x1 SIM_FAST seed 31: f2-only vs f2 ref, f3-only vs crzf3 ref, epa-only vs pass crzhp: all equal.
- f3 cannot use u3 play frames (assert kept==plays), so sim for val/an runs crG minus f3 (play_nof3, 2x8); e5 includes f3.
- e5 3 seeds vs crzhc/drl/pool: SD 15.14/15.26/15.08 (14.63); strength RE 50.4/55.0/46.6 (58.0); noise 178.6/178.1/181.1 (165.0); late r2 .119/.127/.103 (.146); xq +0.19 (vs +8.2/-1.1, pool -6.4); q4 slope -.056/-.037/-.056 (-.060); drives 23.6/22.8/23.6 (22.7); pts 43.4/42.5/43.5 (45.2); mass3 .117/.117/.115 (.141).
- Drill (nof3): el [0,8) 0.8 s (real 3.3), el>gsr .173/.129/.026; passthrough b0 .774 b1 -.207 (pool .776/-.206), downs .43/.64/1.13/1.29 vs .45/.67/1.03/1.76.
