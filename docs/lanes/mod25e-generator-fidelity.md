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

## E23-E26 (late scoring, clock, rule pieces)
- E23: late-leader gap only where old leader is now behind/tied (.143 sim vs .278/.345 real); tied/trailing last-5-min offences score too little (.22 vs .43); not 4th-down policy, handoff, start fp, prevent. Handoff after non-scores +.012 real vs -.005 (E22).
- E24/E25: clock el redraw; drl's xq/slope gain was a log artifact of the pre-base cap; egd el by gsr band matches real, kneel/spike OK, last-30 s flips .099 vs .275 real; egd loses xq gain.
- E26: strength not tied to rule-piece rates (timeouts, penalties); f2 replaced draws bypass strength, fixed by crW weights (kept in crH).
- I3/I4: crG then crH = crzhc + u4g + crW f2 + f3 + egd + EPA m=.03; null modes byte-equal 1x1 SIM_FAST; f3 cannot use u3 play frames, so play analyses use `--off f3` (nof3).

## E27 (crH vs real: xq decomposition, quarter matrix, Q2-Q4 anatomy; scripts/mod25e_xq2.py decomp|late|late2|matrix|anatomy, artifacts/mod25e3/crH/{play6,xq2})
- Measured: crH 6x8 nof3 plays (9792 games). E19 comps sim minus pool: strength +9.2 pp 1.00 (new, E19 was +.6), start-fp +2.6, lead-state +3.1, rest +9.4 [-.6,19.6] pp .93, total +23.6 [13.1,35.8]; opposite-team rest cov +4.3 vs +13.1 (corr -.015 vs +.064: UNCLOSED). Late era total +25.7.
- Quarter matrix (strength-adjusted, 2x cov, crH minus pool): 24 +8.8 [4.1,13.7] pp 1.00 (vs 2018-25 +11.4 pp 1.00); 23 +3.8 pp .94, 14 +2.6 pp .85, 34 +1.9, 12 +1.2, 13 -.7; total +17.5 [8.4,27.2].
- Q2 to Q4 anatomy (offence drives in Q4 by Q2 gain): real Q4 trailing-state offences score more (adj pts/drive trail 1.92 pool, 2.17 late vs crH 1.80; tied 1.55 vs 1.38); leader drives match (1.26 vs 1.25). Gap is in lost big/gained big spread (.31/.38 real vs .12), contrib -6.6/-8.6 vs -2.5. Punt shares equal, TD higher real (lost big .229/.265 vs .214).
- E23 on crH not closed: leader drive after late trailer score .091 vs .157/.183 (behind/tied cell .136 vs .278/.345); last-5-min tied .187 vs .429, trail1-8 .224 vs .273, trail>8 .173 vs .226; trailing 0-3 under 2:00 td .045 vs .133, fga .137 vs .298, no-outcome end .461 vs .155; FG on downs 1-3 in last 15 s .34-.39 vs .60-.70. Sim never logs turnover on downs (.002 vs .06-.09; coded as turnover).
- Inferred mechanism: Q4 trailing/tied offence under-scores (urgency state: late drive decisions and conversion), a Q4 state-conditional score level shared by both teams; not strength, start fp, handoff, clock el. Next: fit trailing/tied Q4 drive completion (4th-down go, FG timing, TD yardage by gsr) from real plays, then rerun mod25e_xq2.py matrix; separately explain strength comp +9.2 (LGO team mean may carry sim sampling difference).

## E28 (downs check, Q4 urgency anatomy; crH play6 vs real 2009-17, measured)
- Downs is NOT an engine defect: all-drive downs share .040 vs .039 real, Q4 trail/tied last-5 .186 vs .177; failed-go drive ends labelled downs 7686 of 9147 (84%) vs 1833 of 2148 (85%). E27 .002 was an analysis-cell artifact (the E27 sim frames never labelled it that way); no engine change.
- Q4 trail/tied last-5-min offences: pts/drive 1.163 vs 1.446; td .144 vs .169, fg .051 vs .081, end_half .29 vs .233. Play level matches (pass .81/.83, ypp 5.64/5.61, c3 .370/.379, el/play 14.5/14.3, 4th go rate .632/.625); 4th-down conversion given dist lower in sim (6.5-10.5 yd .305 vs .352, 10.5+ .225 vs .292).
- Located gap: FG attempts on downs 1-3 in last 2:00, trailing 0-3, yl<=40: real .075/.125/.139 vs sim .042/.036/.043 (spike down 1 .106 vs .065, 4th-down FG .905 vs .832). The late-decision model in mod25_mechanisms.make_decide (poll) is calibrated (model fg .084/.135/.132 on the same real plays), so the loss is downstream of it in the crH chain (f2/eg/f3/tfix or redraw cpick fallback). `crH sim --off eg|f3` writes events not play frames (no idx): the q4dbg runs under artifacts/mod25e3/crH/q4dbg/ are unusable for this.
- Next: trace why decide's FG class is lost (log idx pre/post DECIDE for down<=3, Q4 gsr<=120 in the u3 logger); fix at the named point, then scripts/mod25e_q4.py variant, 2 e5 seeds both eras, xq2 matrix.

## E28b (FG/spike layer trace; scripts/mod25e_q4.py, no result yet)
- Read: egd decide (mod25e_endgame.install_eg mode 2) wraps the f2 chain and, for down 1-3 inside the last 2:00 (Q4) or Q2 two-minute window, never calls f2/base DECIDE: it samples a class from its own HGB and rerow() draws a real late row by (down, class, qtr) nearest neighbour; so f2 timeout pick is bypassed there and the base late model is not the FG source in that window. f3 Overlay.apply only touches codes 0/1; u4g tw changes clock only.
- mod25e_q4.py (written, untested to completion) logs per DECIDE snap with down<=3, qtr 2/4, hs<=120: c0, c_base, c_f2, c_eg, c_f3o; `sim --out-dir D` then `read --out-dir D` prints FG/spike share per stage vs real. f3 pol uses sys._getframe(1), so never wrap dv._G["pol"]. Launch with run_in_background and no trailing `&`; run takes minutes (1x2, 2 workers).
- Next: run it, find the stage where FG share drops (suspect egd class draw or rerow fallback), fix there, then 2 e5 seeds both eras + xq2.

## E28c (FG share by full state, egd model audit; scripts/mod25e_q4b.py, q4/log2, log3; measured)
- egd HGB (classes run/pass/FG/kneel/spike, late rows 2009-17 incl. FG, spike, pstop) is calibrated on real states (down 1/2/3 pred .075/.132/.139 vs real .078/.132/.149); rerow returns class c always (None falls back to idx). Not the defect. egps logging added in install_eg (no behaviour change).
- FG occurs only in the last 15 s (real .56, sim .31 -> .40 on log2). There, sim states differ: stopped-clock states 74 vs real 158 (n 126 vs 238), timeouts left oto .91 vs .37, hs 8.8 vs 6.6; FG given state is lower because the model sees more timeouts and more clock. Composition explains ~.03 of .13, policy-given-state .10 via those features.
- Cause is upstream: in the window egd bypasses f2 timeout class pick. Tested egto (apply f2 pick to egd row): oto at hs<=15 .96 -> 1.16, FG unchanged (.041/.048/.063); reverted. Next: why trailing sim teams keep timeouts into the last 15 s (timeout-use calibration at hs<=60); no e5 run (no fix kept).

## E28d (window timeout fix; EGT=1 -> eg mode 3; scripts/mod25e_to.py, artifacts/mod25e3/to; measured)
- Cause: egd window rows (downs 1-3) carry the drawn real row's timeout flag with no availability gate and no sim-state conditioning (phantom off calls with 0 left: .05-.13/snap; trailer timeouts left at 15 s .94 vs .41 real; Q2 final-60 off calls .15 vs .22).
- Fix (mode 3, tofix in mod25e_endgame.py): f2 now exposes dv._G["f2h"] (ctdraw = gated 4-class draw); run/pass window rows get ct from it, other codes only availability gating; replacement is a window row of same down/code/stop and class via rerow_ct. Null (EGT unset, f2 refactor) equal True vs crH c_eg.
- After: phantom .001; Q2 trailer left at 15 s 1.37 vs 1.32 real; Q4 trailer .70 vs .41 (still high), Q4 leader calls over (.03-.07 vs .01-.02). FG downs 1/2/3 .048/.034/.059 vs real .073/.125/.139: UNCHANGED, timeouts were not the FG cause. e5 s11,s12 (crHt; dirs e5_crH_s11/12 were overwritten, crH s13 intact): margin_sd 15.21 (crH3 15.31), r2 w1-4 .056 (.073), w10-18 .132 (.131), xq 8.3 (10.2), q4 slope -.037 (-.032), mass_3 .117 (.116), pts 42.57 (42.63): neutral to slightly better.
- Next: Q4 leader over-calling and FG composition remain; both likely upstream of timeouts (leader drive/kneel state).

## Next / Open
Next: E27 inferred mechanism build (urgency-state Q4 drive model), strength-comp check; rerun e5 era gate. Open: owner decision which clock; pool-era 20-look reads need leave-season-out confirmation.
