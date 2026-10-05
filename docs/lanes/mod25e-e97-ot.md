# mod25e-e97-ot

## Goal
Name the mechanism that makes sim overtime end on a TD too often (P(final |m|=3 | reg tie) real .682 vs sim .432; E96). Parent: docs/lanes/mod25e-generator-fidelity.md.

## State
2026-10-05 E97 done, analysis only. scripts/mod25e_e97.py -> artifacts/mod25e3/e97/e97.txt (184 looks, one family; game bootstrap 300; P(s>r) = probability_positive of sim minus real). Modified-rule sample: real 2012-17 97 OT games, sim s11-13 years 2012-16 1833.
Measured:
- Drive 0 matches: P(TD) real .165 [.098,.237] sim .179 [.162,.197]; P(FG at idx 0 ends game) 0 in both (rule hook works; all-years sim idx0 FG end .029 vs real .018 = 2011 sudden death).
- Drive idx>=1 (sudden death) carries it: tied idx 2+: P(TD) real .066 [.013,.126] sim .157, P(s>r) 1.00; P(FG) real .368 sim .175, P(s>r) .00; idx 1 tied P(TD) .066 vs .184, P(FG) .393 vs .218. Game ends FG idx1 real .247 sim .142, idx 2+ .289 vs .163 (P(s>r) 0.00); ends TD idx1 .062 vs .159, idx 2+ .052 vs .146 (P(s>r) 1.00).
- FG attempt share on downs 2-3 inside yl<=40: idx>=1 real .15-.19 vs sim .02-.04 (P(s>r) .00); idx 0 real 0 vs sim .02-.03.
- Trailing by 3 at idx 1 (n real 18): real punt 0, downs .39; sim punt .36, downs .06.
- 4th down yl<=40 FG/go share matches (P .07-.9, n small); FG made at yl 11-20 real 17/17 sim .885 [.84,.93].
Mechanism (read): sim04_engine.py:925-962 draws OT plays by KD neighbour (K_STATE=200, line 32) on dist, fp, score, time, timeouts; no possession-index or sudden-death term. Pool phase 4 = OT rows + Q4 last 5 min (255-258); measured 45% of OT draws are OT rows, 55% Q4-last-5 rows (FG share down 2-3 .024-.026). Even OT-row draws give FG .04-.055 at down 2-3 vs pool OT rows 2012-19 .089-.106 (blends idx0 where FG is 0). FD4 skips OT (mod25e_fourth.py:420), ADJ skips phase>=4 (mod25e_adj.py:295,361), KICK/RISK/STF/YLM skip qtr>4; endgame mode>=2 skips in_ot (mod25e_endgame.py:279). Nothing models that a FG ends the game, so the offense reaches the end zone instead; TD then ends the game. Possession index (engine 1048-1063, 1106) and OT kickoff (opening_pool, 1085-1096) read correct.
Fitted fix form (not installed): at in_ot and ot_possession_index>=1, downs 1-3, yl<=45, FG attempt hazard logit = 0.351 - 0.1804 yl + 1.667 [d2] + 2.234 [d3] (real OT 354 plays, 37 FG; LOSO log loss .2222 vs constant .3370, down-only .3326; in-sample .2139). Make rate from existing FG model.

WFG wrapper written 2026-10-05: scripts/mod25e_wfg.py (enabled() env WFG=1, cmd_fit writes artifacts/mod25e3/wfg/fit.json from real REG OT plays 2012-2019, idx>=1, downs 1-3, yl<=45, offense not trailing; install_wfg wraps dv._G["pol"], draws attempt ~ Bernoulli(logit hazard), then replaces row with a pool FG-attempt row (play_type_code 3) at nearest populated yardline). Trailing-by-3 excluded (FG only ties). Hooked in mod25e_crH.py run_with_flags after install_stf; label suffix w2. Fit: yl -.1857 d2 +1.723 d3 +2.543 icpt +.627 (n 369, 44 FG).

## Tried
Drive tables by idx and state, FG by down, 4th down by yardline, engine/wrapper trace, pool phase composition, LOSO hazard fit.

WFG smoke (crH sim, 1 world, 4 seasons, workers 1, seed 31, full base env, ~815 s each): OT games off 76 / on 75; OT |m|=3 share 27/76 (.355) -> 45/75 (.600); |m|>=6 38 -> 25; ties 10 -> 5 (real P(|m|=3|reg tie) .682). Flag off installs nothing (not diffed vs pre-change base). Attempt counts not visible (games parquet only, pool process keeps STATS).

## Next
0. Run base env + WFG=1 e5 seeds 11-13 (label ...w2); check P(final=3|reg tie), mass3 vs .682/.141.
1. Install hazard as OT hook (scripts/mod25e_ot.py or new wrapper) with FG outcome from existing kick model; run base e5 s11-13 with OTY, check P(final=3|reg tie) and mass3 vs real .682 / .141.
2. Trailing-by-3 punt suppression at idx 1 (real 0/18 punts); fit from real, LOSO.
3. FG make rate at yl<=30 (sim .885 vs real 17/17 in OT) check outside OT.

## Open
Real OT n small (97 games); hazard coefficient stability across folds is good (yl -.16 to -.20) but 37 FG events.
