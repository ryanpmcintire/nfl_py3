# mod25e-generator-fidelity

## Goal
Keep improving the MOD-25 generator (owner 2026-10-02). Gate failures to
attack: margin SD (band 13.97-14.87), non-strength variance (real 178),
late-season R2 (band .132-.166), held-out shape (mass 3 real .144). Parent
docs/lanes/mod25-synthetic-data.md; prior docs/lanes/mod25d-variance-source.md.
Rules: nothing targets mass at 3/7/10/14/17; aggregates (pts/g, margin SD,
variance) are checks only. Owner 2026-10-02: NO compensating constants; every
parameter is a named mechanism fitted to its own real behaviour.

## State (measured; pool 2009-17 e5 targets in brackets)
crH (scripts/mod25e_crH.py, I4) is the base: SD 15.31 [14.63], strength RE 55.9 [58.0], noise 178.6 [165], late r2 .131 [.146], xq cov +10.2 [-6.4], Q4 slope -.032 [-.060], mass3 .116 [.141], pts 42.6 [45.2]. Older: crf4 SD 15.16 nonstr 205; crzhc strength RE 55.0 xq +8.2. Code: mod25d_variance.py, mod25e_budget.py, mod25e_scorestate.py, mod25e_era.py (dual-era gate). Artifacts: artifacts/mod25e3/ (crH/ latest), mod25d/, mod25e/.

## Tried E1-E14 (measured; artifacts/mod25e, mod25e3, mod25d)
- E1-E4: yard_bias removed, 1-yd dist matching, 4th-down engine bug fixed. Excess margin var was cross-quarter cov and drives/g; play choice/tempo match; gap is drive outcomes.
- E5-E7b: gain 6.5 with scale 1.0 double counts; derived yard_gain 0, scale 1.0 kept.
- E9-E11: late-R2 gap not feature/drift; no conversion-gain gap; garbage-time/RZ-TD gaps are era drift (4th-go .124 to .191, RZ TD .569 to .605).
- E12-E14: no excess margin resid vs EPA in pool era; return TDs over-produced 2018-25; dual-era gate built (pool + held 2018-25); no persistent margin component outside scrimmage EPA.

## E15-E18 (cov, personnel, kernel; scripts mod25e_cov/revert/revert_sim/kstate)
- E15: joint off/def latent (cross-lag cov corr +.228 scrim_def orientation [-.49,-.009]) lifted strength RE 47.6 to 55.0 (crzhc); noise/xq remain.
- E16: starters explain 13% of leader-offence reversion (off share +.387 EPA/unit); remainder effort/clock not personnel.
## E19-E22 (xq decomposition, env, passthrough, kickoff)
- E19 (mod25e_xq.py): xq gap sim-pool +19.7 [8.6,33.0] pp .99; start-fp +3.4, lead-state +4.6, rest +11.0 [0,22.9]; rest = opposite-team cov (-8.7 [-17.9,-1.2] pp .97): real home/away drive residuals share a positive Q4-heavy comp
- E20 (mod25e_env.py): constant game-level shift fixed corr (.046) but not xq (+8.2 to +7.6), worsened noise/strength: not a keeper; weather/crew/pace/total are not the source.
## E23-E26 (late scoring, clock, rule pieces; compact)
- E23: late-leader gap only where old leader now behind/tied; tied/trailing last-5-min offences score too little; not 4th-down policy, handoff, start fp, prevent.
- E24/E25: clock el redraw; drl xq gain was a log artifact; egd el by gsr band matches real; kneel/spike OK.
## E27 (crH vs real: xq decomposition, quarter matrix, Q2-Q4 anatomy; scripts/mod25e_xq2.py decomp|late|late2|matrix|anatomy, artifacts/mod25e3/crH/{play6,xq2})
- Measured: crH 6x8 nof3 plays (9792 games). E19 comps sim minus pool: strength +9.2 pp 1.00 (new, E19 was +.6), start-fp +2.6, lead-state +3.1, rest +9.4 [-.6,19.6] pp .93, total +23.6 [13.1,35.8]; opposite-team rest cov +4.3 vs 
- Quarter matrix (strength-adjusted, 2x cov, crH minus pool): 24 +8.8 [4.1,13.7] pp 1.00 (vs 2018-25 +11.4 pp 1.00); 23 +3.8 pp .94, 14 +2.6 pp .85, 34 +1.9, 12 +1.2, 13 -.7; total +17.5 [8.4,27.2].
- Q2 to Q4 anatomy (offence drives in Q4 by Q2 gain): real Q4 trailing-state offences score more (adj pts/drive trail 1.92 pool, 2.17 late vs crH 1.80; tied 1.55 vs 1.38); leader drives match (1.26 vs 1.25). Gap is in lost big/gai
## E28-E28c (compact; crH play6 vs real 2009-17, measured)
- Downs is not an engine defect (share .040 vs .039). Q4 trail/tied last-5 offences score less (td .144 vs .169, fg .051 vs .081); play level matches (pass share, ypp, c3, el/play, 4th go rate).
## Results-folder note (orchestrator 2026-10-03)
mod25e_crH.py e5 wrote e5.json/sim_games/play files under e5_crH_s<seed> for every variant (only events went to the label dir); fixed (label used for variant and dir). e5_crHt_s11-12 now hold the copied EGT results (08:34/08:37 ru
## Era scoring 2026-10-03 (artifacts/mod25e3/late_era/era.txt; crH 3 seeds, crHh/crHp 2, seed 13 running)
crH / crHh / crHp vs pool: noise 178.2/176.0/175.2 (165); xq +8.6/+5.2/+5.4 (-6.4); margin var 235/229/229 (222); strength 56.3/53.8/54.2 (58.0); late R2 .132/.126/.129 (.146); pts 42.6/42.6/42.6 (45.2). crHt column INVALID (its d
ALL-FIXES 3-seed (artifacts/mod25e3/all_era/era.txt): crHpqokg (env EGT=1 EGH=1 F2PR=1 QBC=1 GZ=1 OTY=2009-2017 KICK=1) is the NEW BASE: pts/g 44.38 (pool 45.21), strength 56.4 (58.0), noise 179.0 (165), margin SD 15.34 (14.63), xq +5.2 (-6.4), Q4 slope -.047 
## E28d
Orchestrator 2026-10-03: crH baseline s11/s12 re-run (EGT unset) into artifacts/mod25e3/e5_crH_s11-12; the overwritten EGT runs are copied to e5_crHt_s11-12. (window timeout fix; EGT=1 -> eg mode 3; scripts/mod25e_to.py, artifacts
## E29 (trailer last-5-min decomposition; scripts/mod25e_trail.py, artifacts/mod25e3/crH/trail/trail.txt; measured)
- Real drive points in mod25e_late.real_play_drives included OT scoring on the last regulation play (final margin incl. OT); fixed via regulation=True (OT kickoff row margin). E23/E27/E28 late real levels carried this; corrected g
## E30 (hurry-up progress per clock second; scripts/mod25e_hurry.py el|cov|band|fine|val, artifacts/mod25e3/hurry; measured)
- Per-play components match (comp share .441/.432, yards/comp 11.37 both, el by outcome, run/pass mix); real_fit.parquet drops last plays without successor (use raw pbp near 0:00). Cause: sim stop-after ignores out-of-bounds (psto
## Next / Open
Next: see E43 (top candidate mechanism). Open: owner decision which clock; pool-era 20-look reads need leave-season-out confirmation.
## E31 (last-30 s FG state composition; scripts/mod25e_fg30.py, cmd_end; measured on q4 log2/log3 down 1-3, Q4 trail 0-3, yl<=40, hs<=30)
- FG share real .315 vs sim .17-.19; egd clf on real states .321, on sim states .183-.184, sim empirical .173-.187: composition -.137/-.139, decision -.009..+.002 (decision is faithful).
## E32 (timeout spending 30-120 s, Q4 trail/tied offence; scripts/mod25e_tospend.py, snaps code 0/1 down<=4 with a prior play in Q4; real 2009-17 vs crH play6, crHh play_h; measured)
- crH play6 predates gating (phantom calls at 0 left .04-.05); crHt e5 play files lack idx/yards/otu so cannot be scored (logger needs idx, s_otu, s_dtu, yards, oto, dto, prev-stop).
## E33 (f2 timeout model with previous-play clock state; scripts/mod25e_f2pr.py fit; F2PR=1 flag, variant label crHp; measured)
- Input prun = previous play in same half left the clock running (not stop_after incl. el<=cut; cut 14.0 derived on window rows); same function real and sim (sim reads log[-1] and PLAYLOG[-1]). Model artifacts/sim09/f2/policy_pr.j
## E34 (points/game decomposition; scripts/mod25e_pts.py [second], artifacts/mod25e3/pts.log; crHp s11-13 vs real_fit 2009-17, regulation play rows; measured)
- Regulation drives: real 44.23 vs sim 42.43 pts/g (gap -1.80 [-2.37,-1.18]; the other ~0.8 of era -2.6 is OT/kickoff-return points absent from play rows). Exact: start mix +0.54 [+.23,+.86]; drive-outcome probs -2.31 [-3.04,-1.72
- Deficit spans every start bin (deepest own starts -0.45/-0.59). Real-V ladder: down 2 -0.87, 3 -0.73, 1 -0.52, 4 -0.28; run -1.24, pass -1.14; own -1.01, rz -0.84. Conversion by down x dist matches (3rd .568/.403/.225 vs .562/.3
## E35 (draw-weight chain at real 1st&10 states; scripts/mod25e_draw.py, artifacts/mod25e3/draw.log; crH config, 2009-17, K=200; measured)
- Chain: kNN K200 (fp sd 2.8-3.8 yd) x team kernel with same-home gate (lambda 1e6) x IPW x rate tilt x dist kernel x EPA tilt m=.03. Stage effects on drawn mean yards, matched real states, each <=.09 and mixed sign (team +.05..+.
## E36 (state mix vs draw vs post-draw; scripts/mod25e_mix.py, artifacts/mod25e3/mix.log; crHp s11-13, run/pass 1st&10 yl>20 and 2nd down; measured)
- State mix carries nothing: real reweighted to sim state (sd, gsr, yl, timeouts [, dist]) moves 1st&10 mean -0.004 [-.018,+.009] pp .30 (2nd -0.004); sim-real stays -0.078 [-.141,-.004] (2nd -0.135 [-.211,-.053]); trailing share 
## E37 (pick logger; scripts/mod25e_pick.py run/analyze, artifacts/mod25e3/pick.log + pick/, 160k engine picks, crHp 2 worlds x 2 seasons incl. burn seasons; measured)
- Class policy faithful: run share of run/pass 1st&10 .510 vs real .506, 2nd .431 vs .417; cache rounding not it (exact-state vs cache neighbours +.009 yd, cell-first fp +.01).
## E38 (offence-offset decomposition and QBC flag; measured, scripts/mod25_generator.py season_ratings, artifacts/mod25e3/pick_qbc*)
- Generator latents are zero-mean (off +.0014, def -.0016, se .0013, 40 worlds x 6 seasons, old and joint-cov alike; no drift, carry-over or cov offset). Only systematic offset is the backup-QB shock: chain mean P(out) .1595 x -.0
## E39 (crHpq pts split; tests/scratch/e39a.py, e39b.py running; measured)
- Rows 2009-17 real 44.48 vs sim 43.30 pts/g (-1.17): Q1 +.04, Q2 -.66, Q3 +.20, Q4 -.74, OT -.02. By code: pass TD pts -1.26, run +.49, FG -.40, def scores equal. Real games table 45.21 exceeds rows by .73: kickoff-return TDs .28
## E41 (Q2/Q4 deficit and pass vs run TD; scripts/mod25e_q24.py, artifacts/mod25e3/q24.log; crHpq s11-13 vs real 2009-17; measured, game bootstrap)
- Q2 last 2 min -.67 [-.82,-.53] P+ 0; Q4 last 2 min -.35 [-.50,-.21]; Q4 rest-of-quarter -.38 [-.61,-.15] (trailing -.26); Q2 2-5 min +.19, Q2 rest -.18 [-.42,+.03]. Q2 deficit is almost all end-of-half (both lead and trail ~-.29
## E42 (goal-zone spot; scripts/mod25e_gz.py, GZ=1 label suffix g, GZA=<tag> audit; 2x2 audits artifacts/mod25e3/gz/audit_{base,fix}_*.json; measured)
- Null smoke: GZA only (GZ unset) 1x1 sim_games and events equal n0 exactly. GZ=1 now draws with team kernel x home gate x IPW x PASS_W(EPA tilt) over the K nearest goal-to-go rows; PLAYLOG shift patched (idx kept) so the epa asse
## E40 (OT by era rules + kickoff-return flag on crHpq; scripts/mod25e_ot.py, mod25e_crH.py; artifacts/mod25e3/e40; measured, 2 worlds x 2 seasons seed 31)
- OTY=lo-hi (season sidx cycles lo..hi) applies rules: period 15 min before 2017 else 10; pure sudden death through 2011, modified from 2012 (source hook on run_one_game, OT_SUDDEN). OTY=2018-2018 and KICK=0 are byte-equal to base

## E43 (variance budget on base crHpqokg, 3 seeds vs pool 2009-17, one estimator; scripts/mod25e_noise2.py budget|xq|drv, artifacts/mod25e3/noise2/{budget,xq,drv}.txt; measured, season/game bootstrap, E1-E42 compacted into this file's older sections, full copy noise2/lane_before_E43.md)
- Budget: margin var +13.8 [-4.3,+32.3] pp .90; noise RE +14.0 [-0.8,+29.4] pp .96; strength -1.6 pp .35. iid-drive margin var +1.7 [-2.8,+5.9] pp .76, serial excess +11.9 [-6.8,+30.2] pp .88: the excess is cross-drive, not per-drive. home/away var +1.5/+3.5, home-away cov -12.1 vs -7.7 (-4.4 [-9.0,+.1] pp .03 = +8.8 margin var): teams too anti-correlated.
- Per quarter: Q3 inc var +2.9 [+.06,+5.5] pp .98, cov with final +5.5 [+1.3,+9.7] pp 1.00, Q3 drives +.17 pp 1.00 (Q1 +.15 too); Q1/Q2/Q4 inc var flat; Q4 ppp -.06 pp 1.00 vs real, Q4 signed var -2.1 pp .99.
- xq (2x cov, sim minus real): total +17.6 [+6.0,+31.1] pp 1.00; rest +8.7 [-1.5,+20.6] pp .95, leadstate +4.0 [+2.3,+5.7] pp 1.00, startfp +2.6 [+.3,+5.1] pp .98, strength +2.6 pp .87. Pairs 24 +5.9 pp .99, 23 +4.3 pp .97, 34 +3.5, 12 +3.3, 14 +2.3, 13 -.9 (total +18.4 [+8.4,+29.7]).
- Drive level (sum x^2 per game, 22 looks fp 6 / qtr 4 / lead 5 / end 7, plus 42 fp x end): total -1.0 [-5.3,+2.9] pp .31 (drives/g 23.5 vs 23.2); no drive class carries excess (td -.4, fg -1.3 pp 0, defsc +.75 pp 1.00 with var(x) +1.6 per drive, Q3 +2.3 pp .98, fp x end cells all CI crossing 0); TD span and TD share match. Var(points | start fp) within +-.3 except fp2 -.29 pp .01.
- Inferred: excess is a missing shared positive team-coupling plus lead-state feedback, felt through Q2-Q3/Q4 and Q3; per-drive outcome variance is already right. Top candidate for E44: after a lead change or margin shift how both teams score in the next quarter (Q3 second-half start: kickoff-receipt, drive count, lead-state scoring rate), fitted from real plays; E20 constant shift and E19 opposite-team cov (corr .064 vs -.012) remain the target.

## E44 (cov(home,away) decomposition and Q3 drives; scripts/mod25e_couple.py p1|p2|p3|p4, artifacts/mod25e3/couple/, couple.log, couple4.log; 3 seeds vs 2009-17, game bootstrap; measured. Real plays drop penalty rows, so plays/drive and plays blocks are NOT comparable; drive counts and secs are)
- cov(H,A) sim -12.1 vs real -7.7 (diff -4.4 [-8.8,+.2] pp .03). Sequential orthogonal blocks (leave-game-out strength, drive counts, time): strength -0.9 [-2.9,+1.0] pp .13, drives +0.6 pp .96, time -0.3 [-2.1,+1.6] pp .34, residual -3.8 [-7.2,-0.5] pp .01. Possession accounting explains none of it; the gap is the residual shared positive component real games have (real +8.1 vs sim +4.3).
- Alternation too rigid: drive counts corr .907 vs .828 (pp 1.00), var(nh-na) .49 vs 1.02; sim total drive clock sd 22 vs real 86 s (corr -.996 vs -.947): sim clock is near conserved. Same-team-again drives sim 0 outside Q2 (real ~2%, partly my filter): no onside/retained possession. Consecutive same-team drive-length corr sim -.003 vs real +.010 (pp .01).
- Q3: drives starting in Q3 +.068 [+.004,+.126] pp .97 but Q2 -.20 pp 0 and Q4 -.16 pp 0 (real has extra end-of-half drives); Q3 first drive +11.6 s [6.9,16.0] pp 1.00, start fp +.27 pp .90; secs/drive +7 to +10 in every quarter, so not Q3-specific.
- Fix for a later unit (fitted): end-of-half possession model (extra Q2/Q4 drives) and a shared game-state term fitted to real cross-team residual cov; no constant.

## E45 (pace/style mediation of cov(H,A); scripts/mod25e_pace.py, artifacts/mod25e3/pace/pace.txt, pace.log; 3 seeds vs 2009-17, game bootstrap; measured. Sim incompletion = pass with 0 yards, approx; no-huddle is not in the data)
- Hypothesis refuted as the carrier: shared pace/style terms (total drives, total pass+run plays, penalties, state-adjusted pass rate, incompletion rate, scrim and non-play seconds) explain cov +7.7 real vs +7.65 sim (diff -0.06 [-1.6,+1.6] pp .52); the sim already has the shared positive pace component at real size. Residual after them -9.2 real vs -12.5 sim (diff -3.3 [-7.1,+0.8] pp .05): the missing piece is not pace.
- Style correlations match: teams' state-adjusted pass rate corr -.033 vs -.055, incompletion corr +.023 vs +.011 (pp .15/.27). Drive counts corr .828 vs .904 (pp 1.00); var total drives 10.9 vs 10.0 (pp 0), total plays 67.1 vs 61.5 (pp 0), penalty rows var 17.3 vs 10.1, 12.7 vs 8.95 per game.
- Rigid clock: real splits 3600 s into scrimmage 3433 s (sd 112) and non-play time (penalty no-plays sd 176, kickoffs 77, other 141); sim scrim sd 55, no-play sd 52, scrim/no-play corr -.95 vs -.55. Real non-play time varies about 3x more (penalties, timeouts, scoring-driven kickoffs). Pace latent (sigma .04): one multiplier on all elapsed, shared by both teams, fitted from a play-class split-half estimate; same cell estimator gives sim .036 vs real .021 (half-corr .24 vs .05), so it is not too small.
- Next: lead-state coupling (E43 xq leadstate +4.0 pp 1.00) is the remaining candidate; a non-play-time/penalty-count game latent would fix drive/clock rigidity but carries no covariance.

## E46 (drive-level response to margin; scripts/mod25e_resp.py, artifacts/mod25e3/resp/resp.txt; 3 seeds vs 2009-17 regulation, OLS per time band with yl0, yl0^2, leave-game-out off/def rating, game bootstrap 200; measured)
- Sim has the response qualitatively but it is weaker where real is strongest: Q4 last 5 trailers (pts/drive per 7 pts behind) real +.154 [+.09,+.22] vs sim +.080, diff -.074 [-.145,-.013] pp .01 (TD share -.012 pp 0); leaders ease off more in real: all-band lead slope -.176 vs -.137, diff +.040 [+.012,+.068] pp 1.00, Q4 early +.085 [+.035,+.145]. Q1-Q3 and trailer all-band diffs unresolved (pp .3-.9). 7 bands x 5 outcomes x 2 slopes = 70 looks.
- Phase (Q4 last 5, trail): real converts 3rd downs more (+.029 vs +.015, pp .03) and reaches red zone more (+.044 vs +.028, pp 0); leaders punt less in real (+.044 vs +.074) and attempt fewer FGs (-.052 vs -.030); sim explosive response larger. After pass/run mix control the real trail slope is -.007 vs sim -.041: the response runs through play mix and conversion, not through yardage tails. Sim turnover flag (flip) is not comparable (mean .56 vs .26): turnover and later mediation stages invalid.
- Counterfactual (first order, sim states held, slopes conditional): removing the sim-minus-real response by band accounts for 16.9 of the 19.6 raw cross-quarter gap and 4.5 of the -8.1 regulation cov(H,A) gap (-12.9 vs -4.8 real); inferred upper read, point slopes only, mechanically negative-biased.
- Next: fit the sim's drive-end/conversion draws on real (margin band x offence strength stratum) state-matched, test with the same estimator; fix sim turnover definition first.

## E47 (play level, Q3-Q4, measured; scripts/mod25e_resp.py 200 med|play, artifacts/mod25e3/resp/resp_med.txt, resp_play.txt; game bootstrap 200, shared draws; ~100 looks)
- Turnover flag fixed (INT/lost fumble downs 1-3, both sides): .117 real vs .112 sim. Q4 last-5 trailer gap -.074; after pass/run mix -.034 [-.086,+.027] pp .14. Sim TD-play yards were not the gain (median -48); both sides now use the start yardline.
- Mix shift is NOT the gap: pass-share slope per 7 pts behind real +.066 vs sim +.058 (Q4 last5; Q4 all +.062 vs +.058), mix-shift yards identical (+.139 vs +.136); same in every strength tercile. Rates of INT, zero-yard, loss, explosive per pass match.
- Gap is yards at a given deficit (Q4 all trailers): run yards/carry slope real +.286 vs sim +.136 (diff -.150 [-.232,-.080] pp 0), pass yards/att -.135 vs -.313 (-.178 [-.254,-.101] pp 0); ypp slope +.100 vs -.100 (-.200 pp 0). Last 5: ypc +.345 vs -.065, ypp +.122 vs -.199. In every tercile (st0-st2 ypa -.22/-.19/-.08, ypc -.20/-.14/-.13). Inferred: real defences give trailing offences runs and short passes (soft shells), the sim's yard draw has no margin term.
- Leaders: 4th-down decision per 7 pts of lead matches for punt (+.044 vs +.040) and go; sim kicks FGs less lightly (-.012 vs -.022 real, +.010 pp 1.00), Q4 last5 -.036 vs -.020. Punt gap sits in the drive: Q4 last5 reach-4th slope real -.016 vs sim +.017 (+.033 pp 1.00), drive ends other (clock/kneel) +.083 vs +.025 (-.058 pp 0); real leaders run out the clock before 4th down.
- Fix (fitted, no constants): add margin (trail, lead, late phase) to the yard draw's kernel/tilt (run and pass classes separately, same state kNN) and to the leader's play clock (drive end by clock); fit on real plays, test with this script (target: ypc/ypa slope diffs CI over 0).

## E48 (Q4 run yards definitions; scripts/mod25e_q4run.py, artifacts/mod25e3/q4run1-3.log, q4run_clean.log; crHpqokg s11-13, s>=2; measured)
- E47 run-yard level and slope gaps are a logging artifact: pool yards_gained = yardline minus next yardline, so pool flip/TD rows hold -47 to -66; 1.5% of sim runs (9922 of 664k; 99% pool-flip, 70% pool-TD) carry such a row with sim flip=0, po<6 (mean -53.8), missed by E47's po>=6 cap. Kneel/spike/no_play are separate codes, scrambles inside run both sides: not the cause.
- Pool-clean runs: sim ypc Q1-Q4 4.21/4.35/4.30/4.06 vs real clean 4.20/4.33/4.28/4.00. Level gap zero. Q4 trailer slope per 7 behind: real clean +.189, sim pool-clean +.263 (sim-minus-real +.011 [-.075,+.101] pp .62 on all-clean rows); last5 +.109 vs +.180. Run slope not lost; no kernel/tilt fix needed for runs.
- Open: recheck pass ypa slope (-.135 vs -.313) on pool-clean pass rows (INT/sack flips likely share the artifact); why sim logs flip=0/po<6 on pool flip/TD rows. Pick replay not run. No e5 run.

## E49 (engine vs logger on pool terminal rows; scripts/mod25e_e49.py run|analyze, mod25e_e49pass.py, artifacts/mod25e3/e49; crHpqokg flags, 2 worlds x 4 seasons, 330k draws; measured)
- Engine is NOT defective: terminal rows apply flip/points_off/points_def and ignore yards_gained; a non-terminal row's yards move the ball. Final-row logging (after f3/GZ/tfix layers) shows 14.7% pool-terminal vs 14.5% final-terminal. The -53 yards are the play-frame yards column (pool attrs at idx), which e5 play files log; flip/po there are the final row's. Logging only: read yards from the final drawn row or drop rows whose pool row at idx is terminal.
- 3669 pool-terminal draws end non-terminal (yards then replaced, mean +.3, not -53): 2911 inside yl<=25 (GZ redraw keeps idx; reverse 2978 non-terminal to TD, terminals 14572 vs 14639), 758 outside (2.2% of terminals; layer not attributed, f3 penalty overlay or tfix likely).
- Open defect candidate: 1221 final non-terminal draws (.37%) carry yards_gained < -20 (median -34, codes mostly 4/1/0, 95% pool-equal yards; ball moves back to about the 75). Source rows not inspected.
- Pass ypa clean (Q4 trailers, per 7 behind): real -.156 [se .038], sim pool-clean -.196 [se .009], diff -.041 CI [-.13,+.02] pp .12; last5 -.181 vs -.196. E47 pass gap was the same artifact; sim all-rows ypa 2.9 in Q4 vs 6.0 clean.

## E50 (big-loss non-terminal draws; scripts/mod25e_neg.py, NEG=1 label suffix n in mod25e_crH.py; artifacts/mod25e3/neg1; measured)
- Source: tfix rows (game-last, half-last) are non-terminal in the pool but keep yards = yardline minus a fabricated next spot (75 after game end, kickoff spot after half); 1175 of 1221 are Q4 (kneel 633, pass/run) at nyl 75 and Q2 at nyl about 80. Applied as plays whenever the sim clock does not expire; auto_first also true on non-flip half-last rows.
- Fix NEG=1 (default off, guard only): edge non-terminal rows take the pbp play yards (fillna 0), auto_first/repeat_down False. Run 2x2 crHpqokg+NEG: yg<-20 share .37% to .036% (59 of 166k). Drive checks sim base/NEG/real: start yl 71.51/71.62/71.62, punt share .456/.452/.416, pts per drive 1.865/1.887/1.820 (approximate drive def).
- 758 outside-GZ terminal-to-non-terminal: all code 6 = f3 penalty overlay (negates the drawn play with replay down, accepted by a gain-conditioned model); rule-consistent, no change. e5: NEG=1 EGT=1 EGH=1 F2PR=1 QBC=1 OTY=2009-2017 KICK=1 GZ=1 (label crHpqokgn).

## E51 (punt-share decomposition; scripts/mod25e_punt.py 100, artifacts/mod25e3/punt/punt.txt; crHpqokg s11-13 base (neg1 holds no play files) vs real 2009-17, all quarters, game bootstrap 100; 199 looks; measured)
- Drive punt share real .418 vs sim .426 (+.008 [+.004,+.012] pp 1.00) = reach-4th .645 vs .637 (-.008, pp 0) x punt|4th .648 vs .669 (+.021 [+.016,+.027] pp 1.00). Gap is at the 4th down, not in stalling: 1st/3rd-down conversion by dist tercile matches (3rd conv diffs +.003 each, pp .75-.90).
- Of punt|4th +.021: state mix +.021 [+.017,+.026] pp 1.00 (real cell rates on sim 4ths: .660 vs .639), decision given state +.003 [+.002,+.005]. Sim 4ths are longer (4th dist top tercile +.019, short -.020) and further from FG range (zone0 -.020, zone3 +.014); 3rd-down mix shows same (short -.018, long +.015).
- Clock is not the cause here: sim has MORE pre-4th ends by clock (.060 vs .041, Q2 +.043, Q4 +.020, pp 1.00; caveat E47 last-5 slice differs, definitions unchecked); sim turnovers pre-4th slightly fewer (-.008).
- Mechanism (inferred): 1st/2nd-down yardage leaves too few short 3rd downs, so 4ths arrive long and punt; 4th go conversion off too (dist0 -.056, dist2 +.087, pp 0/1). Fix for a sim unit, fitted: refit 1st and 2nd-down gain distribution shape (kNN/tilt, short-gain mass) so 3rd/4th distance mix matches; refit 4th-down go success by distance on real.

## E52 (source distance vs state distance; scripts/mod25e_dist.py 100, artifacts/mod25e3/dist/dist.txt; crHpqokg s11-13 vs pool rows, nonterminal, yards>-20, sim game bootstrap 100 + real binomial; 780 looks; measured)
- 1st&10 faithful (draw dist==10 .988; conv .194 vs .195, short13 .116 vs .115). 2nd and later downs are not: drawn source distance equals state distance only .70 (2nd&1), .40 (2-3), .32 (4-6), .25 (7-9), .56 (10); 2nd&1 draws .30 longer (mean +.97), 2nd&11+ .71 shorter (-2.46). Conv credited at state sticks vs source own sticks: 2nd&1 .604 vs .529.
- Sim 2nd-down conv too high at short: &1 +.031 [pp 1.00], &2-3 +.046, &4-6 +.029 (&7-9 +.003, &10+ +.001); mass 1-3 short of sticks -.033/-.041/-.022 (pp 0); arrival at 3rd -.057/-.056/-.038, 3rd&<=3 -.061/-.056/-.026 (pp 0). Incompletion mass and negatives match (zero ±.003 except &1 -.031). This is the short-3rd deficit.
- 4th go (cuts 1,5): source dist offset +.93 / +.29 / -4.91; success fdnb credit real .636/.470/.311 vs sim .574/.493/.363 (-.062 pp 0, +.023 pp .94, +.052 pp 1.00); state-sticks only -.065/-.040/-.066 (pp 0/.01/0): real lies between, same cause.
- Kernel width (9 chrono folds, held-out log loss P(first down|dist), h=c*sqrt(d)): c=.3 beats engine .6 in 9/9 folds on 2nd (.57660 vs .57721) and 3rd (.61855 vs .61929); 4th flat (6/9, .65363 vs .65360); c 1.2 and 2.4 worse 0/9. Gain small, sign consistent.
- Fix (inferred, untested): dkern c=.3 or exact-distance rows for 2nd/3rd/4th, then rerun this script: need P(sd==dist) up and 2nd&1-6 conv at real. Caveat: K=200 kNN preselection may lack exact-distance rows at 2nd&1.

## E53 (DKF=1 distance fix built, untested in a run; scripts/mod25e_dkern.py, label suffix d; read/measured)
- scripts/mod25e_dkern.py fit writes artifacts/mod25e3/dkern/fit.json: pooled 2nd-4th held-out LL over 9 folds, grid .15/.3/.6/1.2/2.4, picks c=.3 (.59444 vs .59508 at .6). install_dk sets ns DKH from it and wraps every nn tree in ExactTree: neighbours come from rows of the same rounded distance whenever that group holds >= K_STATE rows, else the original tree; team weighting untouched.
- Null (measured): flag off 1x1 sim_games.parquet and latents.npz identical to HEAD crH.py (artifacts/mod25e3/dkern/null_new vs null_old).
- Small e5 run (2x4, 3 workers, DKF=1, label crHpqokgnd) was denied by the permission classifier; not run. Next: orchestrator runs it, then mod25e_dist.py with LABEL crHpqokgnd (LABEL is hardcoded at line 13).
E53 status (orchestrator 2026-10-03): the small validation run (DKF=1 base env, scripts/mod25e_crH.py e5 --worlds 2 --seasons 4 --workers 3 --seed 11, label crHpqokgnd) was denied to the subagent by the permission check ("Interfere With Workloads"); not re-run by the orchestrator, because a denied action is not re-run on an agent's request. Pending owner approval; then set LABEL in scripts/mod25e_dist.py:13 to crHpqokgnd and rerun it.
NEG 3-seed (artifacts/mod25e3/neg_era/era.txt): crHpqokgn vs crHpqokg: pts/g 44.64 vs 44.38 (45.21), strength 55.9 vs 56.4, noise 181.6 vs 179.0, xq +6.1 vs +5.2; within seed noise apart from points. Bug fix kept by construction (fabricated -34 yd plays removed). Base for next units: crHpqokgn (+NEG=1).
