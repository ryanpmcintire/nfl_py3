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
- E17: sim reverts at play level (top bin -.088 vs -.095); gap is strength passthrough in decided states (.86 vs .56), a 20-look pool-era read.
- E18: kernel ESS ~26 in every wp decile; state-kernel sparsity not the cause; chosen K200 h x8 failed passthrough (.56 to .15). Pregame-rating kernel has no play-level likelihood power.

## E19-E22 (xq decomposition, env, passthrough, kickoff)
- E19 (mod25e_xq.py): xq gap sim-pool +19.7 [8.6,33.0] pp .99; start-fp +3.4, lead-state +4.6, rest +11.0 [0,22.9]; rest = opposite-team cov (-8.7 [-17.9,-1.2] pp .97): real home/away drive residuals share a positive Q4-heavy component (corr .064 vs -.012).
- E20 (mod25e_env.py): constant game-level shift fixed corr (.046) but not xq (+8.2 to +7.6), worsened noise/strength: not a keeper; weather/crew/pace/total are not the source.
- E21/E21b (mod25e_pass.py): state passthrough is era-unstable (pool b1 -.206, 2016-25 +.071); EPA tilt m=.03 matches downs, xq worse (+13.6): kept inside crH.
- E22 (mod25e_kick.py): kickoffs already match; only return TDs missing; kick draws no gain.

## E23-E26 (late scoring, clock, rule pieces; compact)
- E23: late-leader gap only where old leader now behind/tied; tied/trailing last-5-min offences score too little; not 4th-down policy, handoff, start fp, prevent.
- E24/E25: clock el redraw; drl xq gain was a log artifact; egd el by gsr band matches real; kneel/spike OK.
- E26: f2 replaced draws bypass strength, fixed by crW weights (kept in crH). I3/I4: crH = crzhc + u4g + crW f2 + f3 + egd + EPA m=.03; play analyses use `--off f3`.

## E27 (crH vs real: xq decomposition, quarter matrix, Q2-Q4 anatomy; scripts/mod25e_xq2.py decomp|late|late2|matrix|anatomy, artifacts/mod25e3/crH/{play6,xq2})
- Measured: crH 6x8 nof3 plays (9792 games). E19 comps sim minus pool: strength +9.2 pp 1.00 (new, E19 was +.6), start-fp +2.6, lead-state +3.1, rest +9.4 [-.6,19.6] pp .93, total +23.6 [13.1,35.8]; opposite-team rest cov +4.3 vs +13.1 (corr -.015 vs +.064: UNCLOSED). Late era total +25.7.
- Quarter matrix (strength-adjusted, 2x cov, crH minus pool): 24 +8.8 [4.1,13.7] pp 1.00 (vs 2018-25 +11.4 pp 1.00); 23 +3.8 pp .94, 14 +2.6 pp .85, 34 +1.9, 12 +1.2, 13 -.7; total +17.5 [8.4,27.2].
- Q2 to Q4 anatomy (offence drives in Q4 by Q2 gain): real Q4 trailing-state offences score more (adj pts/drive trail 1.92 pool, 2.17 late vs crH 1.80; tied 1.55 vs 1.38); leader drives match (1.26 vs 1.25). Gap is in lost big/gained big spread (.31/.38 real vs .12), contrib -6.6/-8.6 vs -2.5. Punt shares equal, TD higher real (lost big .229/.265 vs .214).
- E23 on crH not closed: leader drive after late trailer score .091 vs .157/.183 (behind/tied cell .136 vs .278/.345); last-5-min tied .187 vs .429, trail1-8 .224 vs .273, trail>8 .173 vs .226; trailing 0-3 under 2:00 td .045 vs .133, fga .137 vs .298, no-outcome end .461 vs .155; FG on downs 1-3 in last 15 s .34-.39 vs .60-.70. Sim never logs turnover on downs (.002 vs .06-.09; coded as turnover).
- Inferred mechanism: Q4 trailing/tied offence under-scores (urgency state: late drive decisions and conversion), a Q4 state-conditional score level shared by both teams; not strength, start fp, handoff, clock el. Next: fit trailing/tied Q4 drive completion (4th-down go, FG timing, TD yardage by gsr) from real plays, then rerun mod25e_xq2.py matrix; separately explain strength comp +9.2 (LGO team mean may carry sim sampling difference).

## E28-E28c (compact; crH play6 vs real 2009-17, measured)
- Downs is not an engine defect (share .040 vs .039). Q4 trail/tied last-5 offences score less (td .144 vs .169, fg .051 vs .081); play level matches (pass share, ypp, c3, el/play, 4th go rate).
- egd HGB is calibrated on real states (E28c); FG only in last 15 s (real .56 vs sim .31-.40); sim trailers reach it with more timeouts/seconds; egto (f2 pick on egd row) did not move FG; reverted. Scripts mod25e_q4.py, mod25e_q4b.py.

## Results-folder note (orchestrator 2026-10-03)
mod25e_crH.py e5 wrote e5.json/sim_games/play files under e5_crH_s<seed> for every variant (only events went to the label dir); fixed (label used for variant and dir). e5_crHt_s11-12 now hold the copied EGT results (08:34/08:37 runs). crHh runs were stopped by the orchestrator (owner: machine overloaded); e5_crH_s11-12 still hold the EGT results (duplicated into e5_crHt). To do, ONE run at a time with 3 workers: re-run crH baseline s11-12, then crHh s11-12 (script now names dirs by label), then crHp s11-12 (EGT=1 EGH=1 F2PR=1; compare E32 timeouts left 60/30/15 s, E31 last-30-s FG share, era table ERA_VARIANTS=crH,crHh,crHp).

## Era scoring 2026-10-03 (artifacts/mod25e3/late_era/era.txt; crH 3 seeds, crHh/crHp 2, seed 13 running)
crH / crHh / crHp vs pool: noise 178.2/176.0/175.2 (165); xq +8.6/+5.2/+5.4 (-6.4); margin var 235/229/229 (222); strength 56.3/53.8/54.2 (58.0); late R2 .132/.126/.129 (.146); pts 42.6/42.6/42.6 (45.2). crHt column INVALID (its dirs mix files copied from different runs: xq -43, q4 slope -.17); ignore it.

3-seed update (artifacts/mod25e3/late_era3/era.txt, mean +- seed SD): strength crH 56.3+-1.7 / crHh 54.3+-0.8 / crHp 55.1+-2.3; noise 178.2+-1.3 / 177.4+-2.9 / 177.6+-4.6; xq 8.6+-0.5 / 6.4+-2.7 / 8.3+-5.3; q4 slope -.040 all. The late-game fixes are mechanism-correct but indistinguishable on whole-game checks: the end game is not where the noise, lead-carry or -2.6 pts/g gaps live. Next: the scoring deficit (pts/g 42.6 vs 45.2) outside the end game.

## E28d
Orchestrator 2026-10-03: crH baseline s11/s12 re-run (EGT unset) into artifacts/mod25e3/e5_crH_s11-12; the overwritten EGT runs are copied to e5_crHt_s11-12. (window timeout fix; EGT=1 -> eg mode 3; scripts/mod25e_to.py, artifacts/mod25e3/to; measured)
- Cause: egd window rows (downs 1-3) carry the drawn real row's timeout flag with no availability gate and no sim-state conditioning (phantom off calls with 0 left: .05-.13/snap; trailer timeouts left at 15 s .94 vs .41 real; Q2 final-60 off calls .15 vs .22).
- Fix (mode 3, tofix in mod25e_endgame.py): f2 now exposes dv._G["f2h"] (ctdraw = gated 4-class draw); run/pass window rows get ct from it, other codes only availability gating; replacement is a window row of same down/code/stop and class via rerow_ct. Null (EGT unset, f2 refactor) equal True vs crH c_eg.
- After: phantom .001; Q2 trailer left at 15 s 1.37 vs 1.32 real; Q4 trailer .70 vs .41 (still high), Q4 leader calls over (.03-.07 vs .01-.02). FG downs 1/2/3 .048/.034/.059 vs real .073/.125/.139: UNCHANGED, timeouts were not the FG cause. e5 s11,s12 (crHt; dirs e5_crH_s11/12 were overwritten, crH s13 intact): margin_sd 15.21 (crH3 15.31), r2 w1-4 .056 (.073), w10-18 .132 (.131), xq 8.3 (10.2), q4 slope -.037 (-.032), mass_3 .117 (.116), pts 42.57 (42.63): neutral to slightly better.
- Next: Q4 leader over-calling and FG composition remain; both likely upstream of timeouts (leader drive/kneel state).

## E29 (trailer last-5-min decomposition; scripts/mod25e_trail.py, artifacts/mod25e3/crH/trail/trail.txt; measured)
- Real drive points in mod25e_late.real_play_drives included OT scoring on the last regulation play (final margin incl. OT); fixed via regulation=True (OT kickoff row margin). E23/E27/E28 late real levels carried this; corrected gap 0.267 vs 0.290 pts/drive (real 1.280, sim 1.013, trail/tied start in last 5 min).
- Decomposition (time, score, field bins; game bootstrap): composition +.012..+.025 [-.008,+.057], within-state +.243..+.256 [+.14,+.36]; P(score) within +.056 [.039,.070]. Start mix, prior-drive burn (154 vs 152 s), plays/burn per drive all match: not composition.
- Within state: under 2:00 sim scores far less (sc [30,60) .069 vs .140, fg .039 vs .100; [60,120) fg .050 vs .100; [120,180) td .172 vs .235, clock-expiry .158 vs .069). Per-play ypp, explosives, TD per play by yardline, el per play (60-300 s) match; sim snaps in last 60 s sit at 60-80 yd (.299 vs .219) rather than inside 30 (.229 vs .298): sim late drives advance less far per second.
- Not fixed: no single named mechanism; next probe sideline/stop-clock sequencing (incomplete/OOB stops, timeout use) and drive progress per clock second in last 2 min.
## E30 (hurry-up progress per clock second; scripts/mod25e_hurry.py el|cov|band|fine|val, artifacts/mod25e3/hurry; measured)
- Per-play components match (comp share .441/.432, yards/comp 11.37 both, el by outcome, run/pass mix); real_fit.parquet drops last plays without successor (use raw pbp near 0:00). Cause: sim stop-after ignores out-of-bounds (pstop = incomplete/score/flip/timeout only), so spikes fire after OOB (sim .073 vs real .010 at 15-30 s) and too rarely after running clock w/o timeouts (.153 vs .204; .066 vs .106 at 30-60).
- Fix EGH=1 with EGT=1 (eg mode 4, variant crHh): derived stop cut = el valley between incomplete and completion modes (12 s, hurry_cut in mod25e_endgame.py); stop_after adds el<=cut for run/pass in HGB training pstop and sim pstop. Validate play_h (3x8): spike run1/to0 .119/.097 vs real .204/.106, p/drive last 30 s .077 vs .094 (crH .034), 30-60 .260 vs .444 (.229), 60-120 .739 vs .882 (.650); all trail 1.040 vs 1.280 (crH 1.013). FG ending 0-30 .019 vs .044 unchanged (FG decision, not spikes).
- e5 crHh s11,s12 into artifacts/mod25e3/e5_crHh_s11-12; score with scripts/mod25e_era.py (ERA_VARIANTS=crHh).

## Next / Open
Next: E27 inferred mechanism build (urgency-state Q4 drive model), strength-comp check; rerun e5 era gate. Open: owner decision which clock; pool-era 20-look reads need leave-season-out confirmation.
- E29 note: crHt e5 s11/s12 had no e5.json/sim_games (event dirs only); rerun with --workers 2 was killed unfinished at ~45 min (cap); era scoring of crHt still pending, crH 3-seed era in crH/era.log and era_crHt/era.txt (crHt n=0).

## E31 (last-30 s FG state composition; scripts/mod25e_fg30.py, cmd_end; measured on q4 log2/log3 down 1-3, Q4 trail 0-3, yl<=40, hs<=30)
- FG share real .315 vs sim .17-.19; egd clf on real states .321, on sim states .183-.184, sim empirical .173-.187: composition -.137/-.139, decision -.009..+.002 (decision is faithful).
- Composition: snaps in last 6 s share .29 vs .13-.14; offence timeouts held 1.09-1.28 vs .59 (.55-.73 sd), hs mean 16.6 vs 13.9; with offence timeouts FG .19 real, no timeouts .42 real (sim .23-.27). Sim last window snap gsr 12.9 vs 9.1 real.
- FG execution near 0:00 is fine: sim gsr<=2 FG .74 vs .82 real, (2,6] .71 vs .85; no clock-out or half-end blocking (el never exceeds gsr).
- Fix for later unit: sim trailers carry unspent offence timeouts into the last 30 s (E28d .70 vs .41): trace timeout use in the 30-120 s hurry-up (post-completion/OOB stops), not the FG decision.

## E32 (timeout spending 30-120 s, Q4 trail/tied offence; scripts/mod25e_tospend.py, snaps code 0/1 down<=4 with a prior play in Q4; real 2009-17 vs crH play6, crHh play_h; measured)
- crH play6 predates gating (phantom calls at 0 left .04-.05); crHt e5 play files lack idx/yards/otu so cannot be scored (logger needs idx, s_otu, s_dtu, yards, oto, dto, prev-stop).
- Per-snap call rate real .084 vs crHh .079 overall; by clock: (30,45] .252 vs .266, (45,60] .181 vs .191, (60,90] .142 vs .118, (90,120] .070 vs .055; prev play clock running (el<=13 incl.), oto>0: .150 vs .120.
- Timeouts left at 60/30/15 s (first trailing Q4 snap): real .84/.57/.41, crH 1.07/.99/.93, crHh .93/.75/.64.
- f2 reads stop of the drawn row's OWN outcome (feats col stop, sim09_f2.py:196; stop array excludes el<=cut in crHh), not the previous play's running clock; sim hurry-up stop logic never reaches f2. Cause: calls after running clock in 60-120 s under-spent ~20%, compounding. Fix: refit f2 with previous-play clock-running (derived cut) fed from sim state, leave-season-out.

## E33 (f2 timeout model with previous-play clock state; scripts/mod25e_f2pr.py fit; F2PR=1 flag, variant label crHp; measured)
- Input prun = previous play in same half left the clock running (not stop_after incl. el<=cut; cut 14.0 derived on window rows); same function real and sim (sim reads log[-1] and PLAYLOG[-1]). Model artifacts/sim09/f2/policy_pr.joblib (+json cut); old policy.joblib untouched.
- LOSO 2009-17 4-class log loss .17438 -> .17406 (9 of 9 seasons lower). Q4 30-120 s, oto>0, trail/tied, prev running: real .1416, old .1214, new .1286 (n 1963); prev stopped .1205/.1203/.1158. Closes about a third of the gap.
- Logger (mod25e_scorestate e5): adds idx, yards, lo_otu, lo_dtu, oto, dto, pstop. 1x1 smoke: flag off equals crHh baseline (events, sim_games True); flag on runs and differs; e5 1x1 wrote play file with new columns.
- Full check: EGT=1 EGH=1 F2PR=1 SIM_FAST=1 mod25e_crH.py e5 --seed 11 / 12 (dirs e5_crHp_s11-12); compare E32 left at 60/30/15 (real .84/.57/.41), E31 FG share last 30 s (.315), era table.

## E34 (points/game decomposition; scripts/mod25e_pts.py [second], artifacts/mod25e3/pts.log; crHp s11-13 vs real_fit 2009-17, regulation play rows; measured)
- Regulation drives: real 44.23 vs sim 42.43 pts/g (gap -1.80 [-2.37,-1.18]; the other ~0.8 of era -2.6 is OT/kickoff-return points absent from play rows). Exact: start mix +0.54 [+.23,+.86]; drive-outcome probs -2.31 [-3.04,-1.72] (TD -1.76 [-2.45,-1.14], FG -0.41 [-.62,-.22], def score -.14); PAT/2pt per TD -0.12 [-.145,-.096] (.942 vs .969 pts over 6); all pp 0.00 for negatives.
- Deficit spans every start bin (deepest own starts -0.45/-0.59). Real-V ladder: down 2 -0.87, 3 -0.73, 1 -0.52, 4 -0.28; run -1.24, pass -1.14; own -1.01, rz -0.84. Conversion by down x dist matches (3rd .568/.403/.225 vs .562/.396/.228) except 4th go (.565 vs .640 short); FG make matches; rz arrival .261 vs .278, TD|arrival .514 vs .531.
- Top: 1st&10 yards per play -0.082 [-.152,-.018] pp .01, concentrated at yl 63-99 (-.12, -.26 per play; 20+ share .072 vs .076); drawn source rows are the cause: source yards 5.10 vs pool 5.21 at 1st&10, and 6.40 vs real 6.69 in yl 77-99 at sim dist 10 (sim minus source +0.02). Not distance-neighbour mismatch (src dist 10.02).
- Fix for later unit: fit the draw weights (state kernel plus EPA tilt) so the drawn-row yards mean equals pool mean within down x yl band; check tilt asymmetry; refit PAT/2pt success from real; 4th-go conversion lower.

## E35 (draw-weight chain at real 1st&10 states; scripts/mod25e_draw.py, artifacts/mod25e3/draw.log; crH config, 2009-17, K=200; measured)
- Chain: kNN K200 (fp sd 2.8-3.8 yd) x team kernel with same-home gate (lambda 1e6) x IPW x rate tilt x dist kernel x EPA tilt m=.03. Stage effects on drawn mean yards, matched real states, each <=.09 and mixed sign (team +.05..+.08, IPW <=.015, tilt <=.015, dist <=.02, EPA .002); K 25..400 does not move it. Weights do NOT bias yards low; E34 fix premise (reweight) is refuted at real states. Drawn minus real is class-filter dependent (run/pass queries -.14..-.5, all-codes +.17..+.30), so the E34 5.10 vs 5.21 gap is not reproduced here.
- PAT/2pt: fit_pat takes XP make .941 from seasons >=2015 (rule change; real .982-.997 in 2009-14, .940 in 2015-17) and a 2pt-attempt HGB on score/time; real_fit 2009-17 mix is .975, so the -.12 pts/TD is the era mix, not a defect (vs 2015-17 sim .942 = real .939-.942).
- Next: score logged sim 1st&10 snaps (crHp play files carry idx) against the same stage chain at the sim-visited state (state-mix vs draw faithfulness), and compare E34 only to 2015-17 for PAT.

## E36 (state mix vs draw vs post-draw; scripts/mod25e_mix.py, artifacts/mod25e3/mix.log; crHp s11-13, run/pass 1st&10 yl>20 and 2nd down; measured)
- State mix carries nothing: real reweighted to sim state (sd, gsr, yl, timeouts [, dist]) moves 1st&10 mean -0.004 [-.018,+.009] pp .30 (2nd -0.004); sim-real stays -0.078 [-.141,-.004] (2nd -0.135 [-.211,-.053]); trailing share .476 vs .474; source-row strength sim-real -0.007 epa units = -0.03 yd.
- Post-draw is not it: logged minus source yards +0.022 (2nd +0.027), 1.4% rows changed, pp 1.00 positive. Source rows themselves are low: 5.104 vs real 5.208 (2nd 5.019 vs 5.184).
- Draw: state-only chain (kNN K200 x IPW x dist kernel, neutral strength; 20000 snaps) expects run/pass 5.221 vs realized source 5.121 (-0.10 [-.04,+.25 expected minus realized] pp .91); 2nd 5.191 vs 5.008 (gap .18 [+.04,+.34] pp .99); gap sits in yl bands 2-3 (-.24,-.17). Expected matches real (5.208). Source yl is nearer goal than chain expects, so not a position shift; cell-centre query adds +.03 (not the cause).
- TD per drive: -0.011 [-.015,-.007] pp 0 all bins (-.010,-.014,-.008,-.010); start mix -0.0002.
- Next: log off_sim/def_sim and replay the engine's actual pick (team kernel, tilt, EPA) at the logged states to attribute the residual -0.07/-0.14 (strength accounts -0.03); then fit the draw to the pool mean.

## E37 (pick logger; scripts/mod25e_pick.py run/analyze, artifacts/mod25e3/pick.log + pick/, 160k engine picks, crHp 2 worlds x 2 seasons incl. burn seasons; measured)
- Class policy faithful: run share of run/pass 1st&10 .510 vs real .506, 2nd .431 vs .417; cache rounding not it (exact-state vs cache neighbours +.009 yd, cell-first fp +.01).
- Mechanism: weights at the sim's own strengths expect 4.988 vs neutral 5.096 (1st&10; 2nd 4.949 vs 5.043); per-snap delta -.113 / -.096, = sim offense mean offset -.0188 x slope 6.07 yd per epa unit. Cause read in mod25_generator.season_ratings: off = league_off + weekly + backup-QB shock (-.0567) with P(out) ~.17 never re-centred, so sim offense sits -.0097 below pool (pool +.001); world latent means are ~0 (12-season mean +.0025). Def sd sim .063 vs pool rows .037. Also f3 zeroes IPW on penalty rows (pool rp pen share .015, mean yd 10.3: drawn -.08) while overlay returns +.02.
- Fix not built: centre the shock term by its fitted mean (mean of shock x qb_out over the generated latents); null smoke = flag off byte-equal.

## E38 (offence-offset decomposition and QBC flag; measured, scripts/mod25_generator.py season_ratings, artifacts/mod25e3/pick_qbc*)
- Generator latents are zero-mean (off +.0014, def -.0016, se .0013, 40 worlds x 6 seasons, old and joint-cov alike; no drift, carry-over or cov offset). Only systematic offset is the backup-QB shock: chain mean P(out) .1595 x -.0567 = -.0090 (stationary .181 not used; chain starts at 0).
- Fix: env QBC=1 centres the shock by its exact chain-marginal expectation per week; flag off hashes identical to HEAD (smoke), flag on mean off rating -.0023 vs -.0113.
- Small run QBC=1 (2x2, seed 51): per-snap expected-yards actual-minus-neutral +.010 (1st&10), +.036 (2nd) vs -.113/-.096 (seed 41, flag off); shock explains only ~.055 of it, rest is seed (different worlds), so unattributed.
- f3 penalty rows: real run/pass rows with accepted fouls average yg 6.98 at 1.5% share, excluding them costs -.022 yd/snap (not -.08; that mixed in penalty yards); overlay fouls are no_play rows. Def sd .063/.055 sim is true latent, pool .037 is a rolling estimate: units differ, not shown a defect.
