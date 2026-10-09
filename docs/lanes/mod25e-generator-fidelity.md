# mod25e-generator-fidelity

## Goal
Keep improving the MOD-25 game simulator until it passes its gates against real
2009-17 games: margin SD band 13.97-14.87, non-strength noise near real 165,
late-season R2 band .132-.166, mass at 3 near .141. Every parameter is a named
mechanism fitted to its own real behaviour (LOSO held-out likelihood); aggregates
are checks only; no compensating constants. The simulator is not served.

## State (2026-10-05, measured, 3 seeds)
Base label `crHpqokgndecsmfwtjo2as2` (now +YLM punt = `...as2yp`, see Next 4) = env `NEG=1 EGT=1 EGH=1 F2PR=1 QBC=1
OTY=2009-2017 KICK=1 GZ=1 DKF=1 DK2=1 CLK=1 SEL=1 KN=1 KNW=1 FD4=1 CLK2=1 TDC=1
KGZ=1 OKK=1 ADJ=1 STF=1`, run `scripts/mod25e_crH.py e5 --workers 3 --seed {11,12,13}`.
Scored by `scripts/mod25e_era.py` -> artifacts/mod25e3/stf_era/era.txt.
Sim vs real: margin SD 15.34 (14.63); noise 178.3 (165); strength 57.4 (58.0);
late r2 .135 (.146); xq cov +1.5 (-6.4); Q4 slope -.055 (-.060); mass3 .115 (.141);
pts/g 45.63 (45.21). Catch-up gap +13.8, persistence E +10.3 (start of 10-04: +21.2/+14.8).
Detailed log: docs/lanes/done/mod25e-generator-fidelity-log-2026-10-04.md.

## Tried (2026-10-04 units; all measured, details in the log)
- Kept: DKF exact-distance draws; DK2 nearest-distance + 4th-down go path; CLK
  fitted elapsed (kneel censoring fixed); SEL score-state selection reweight
  (slope .033 -> .014); KN/KNW kneel and defensive-timeout clock; FD4 fitted
  4th-down choice (old pol4 snapped Q2 hs<=450 to 0); CLK2 Q2/Q4 split; TDC
  TD-only elapsed (CLK had pooled TDs with turnovers so late TDs ran out the half);
  KFIT eiv kernel h .59 x std(eo), lam 2.0 (errors-in-variables corrected).
- Behavioural (owner direction): E75 play-level score response and halftime
  regression already match real; E76 gap carrier is later start field position;
  E77 sim has 3.1 pp fewer turnover/downs starts; E78 RISK (turnover tilt +
  kick mixture) barely moved turnovers (trail>8 pass lost -.0050 -> -.0041) and
  worsened checks on 3 seeds (SD 15.43, catch-up +13.7) -> not in base.
- KFIT eiv dropped: strength drop not from kernel (no-KFIT s11 54.0) and
  kernel worsened SD/noise in every comparison.
- Rejected: KFIT game/rolling fit (strength 36.5, units mismatch); GZE (yardline
  premise refuted: play-file idx is the pre-GZ row); source-game concentration (E60).
- Analysis scripts: mod25e_catchup, persist, selstate, srcconc, short, half2,
  halfend, trips, clock2 (label via CLK_LABEL/CU_LABEL/DIST_LABEL env; never
  import clock2 at module level, its main() overwrites artifacts).

## 2026-10-08 units (measured; details in each lane)
- E100 (mod25e-e100-seedboot.md): base ypw2 -> ypw2xfdcd (PATY+FGD+CDR) KEPT as base. Season bootstrap
  within seed: late r2 +.0136 [+.002,+.024] P.96, mass3 +.0030 P.90, Q4 slope toward real P.94; strength
  +4.09 [+.01,+8.39] P.95 moves AWAY from real 58.0 (seeds +5.2/+6.4/+0.6), SD +.166 P.93: open check.
  Last-5-min tie -> final 3 still .482 v real .625.
- E101 (mod25e-e101-logrecheck.md): on synced logs the E97 OT FG gap is gone (downs 2-3 .16-.21 v real
  .15-.19; old .02-.03 was the stale log). Held: punt slope/start, E95 gaps (same sign, intervals span 0).
  Still open: FG kick with <=5s left .224 v .355; Q4 last play <=5s .311 v .128; Q2 .405 v .263.
- E102 (mod25e-e102-touchback.md): STB=1 hook (scripts/mod25e_stb.py, suffix tb) draws touchbacks as their
  own event; punt logistic in fp,fp^2 (LOSO +854 ll, 9/9 seasons), kickoff constant .504; STF skips
  touchbacks. Smoke s31 launched 12:04 (artifacts/mod25e3/e5_..xfdcdtb_s31), unscored at write time.

## Next
00000. 2026-10-09 BASE NOW ...kekudwtpq2ugky (see E114b section at end). E117: TPO bound local `offense`, so WFG (and STF reset) silently read the TPO frame; E114 OT and aggregates confounded; fixed (tpo.py:221, wfg.py:59), 3-seed rerun next. Was: IN FLIGHT: 3-seed run base+CDW+TPO+CLQ=2+URG+CKY (label ...kekudwtpq2ugky, tests/scratch/e114/run_3seed.sh, outputs artifacts/mod25e3/e114_boot.log + tests/scratch/e114/tgt_3seed.txt). Smoke: final-drive score|reach .43 -> .69 (real .77). GEO (E115 geometry: TD iff yards>=yl) being built.
0000. 2026-10-09 NEW BASE crHpqokgndecsmfwtjo2as2ypw2xfdcdtbkhghkekudw (E110 CDW=1 timeout-aware runoff draw 45-420 s, KEPT 3 seeds: SD 15.38, noise 180.7, pts 45.14, Q4 slope -.055; lane done/mod25e-e110-leaderburn.md). In flight: E111 CLQ (Q1/Q3 real elapsed + quarter truncation), E112 TPO (Q4 trailer pace), E113 CKY (in-range kick-preserve clock) + URG (late contested pace); smoke tests/scratch/e114/run_tq.sh.
000. Previous: crHpqokgndecsmfwtjo2as2ypw2xfdcdtbkhghkeku (E108c: + CKU=1, a timeout stops the half ending; noise 185 v 165, late r2 .127 v .146 now exposed). Previous: crHpqokgndecsmfwtjo2as2ypw2xfdcdtbkhghke (E107: + CKS=1 score-state clock runoff; Q4 .183, Q2 .300, tied-last-5 final3 .478). Previous: crHpqokgndecsmfwtjo2as2ypw2xfdcdtbkhgh = base env + YLM=1 YLM_CLASSES=punt WFG=1 PATY=1 FGD=1 CDR=1 STB=1 GFL=2 CKH=1 (lanes e103-e106). Q2 last play <=5s closed (.270 v .263); Q4 .212 v .128; P(final3|tied last 5) .451 v .625 is the largest late gap.
00. 2026-10-08 E103 (lane mod25e-e103-lateclock.md): late-half gap is non-kneel plays rarely ending the half (Q4 run trailing/tied hs 10-20 real .41 v sim .09); candidate cause CLK pools half-ending plays as short exact elapsed (mod25e_clk.py:70-72). Per-second hazard (CKC, off) lost LOSO on run/pass; E103b KM censored draw (CKC, off): LOSO gain term +.096 9/9, pass +.004, run -.113 0/9; smoke kc s31 barely moved run half-end share (.09 v real .41), so pool censoring is not the carrier. E103c: timeouts refuted (sim below real at every count); real gap is in decided games (down 9+: real .67-.78 v sim .09); cause: transition frame marks each game's last play as a possession flip (sim04_engine ~580/640), so game-ending runs land in the term pool. GFL hook (scripts/mod25e_gfl.py) relabels; KEPT on 3 seeds 2026-10-08 (lane e103): target toward real, aggregates flat. Superseded 2026-10-08 by E104 (lane mod25e-e104-halfend.md): GFL=2 also relabels halftime last plays; KEPT. NEW BASE crHpqokgndecsmfwtjo2as2ypw2xfdcdtbgh = base env + YLM=1 YLM_CLASSES=punt WFG=1 PATY=1 FGD=1 CDR=1 STB=1 GFL=2. Q2 <=5s .363 (.263), Q4 .241 (.128); P(final3|tied last 5) .465 (.625) now the clearest late gap. Old next: offense timeout use after non-final plays (real .62-.73 v sim .24-.34) and clock-aware play choice; smokes must match base size (--worlds 1 --seasons 4).
0. 2026-10-08 DONE: STB KEPT on 3 seeds (lane e102): SD 15.50, strength 56.67 (real 57.96; the E100
   strength rise is gone, so no ablation needed), xq cov +1.6, Q4 slope -.057; late r2 .132 (.146) and
   P(final 3 | tied after reg) .546 (.682) moved away. New base ...ypw2xfdcdtb (env + STB=1).
   NEXT: the late-clock gaps (E101): FG kick with <=5s left .224 v .355, Q4 last play <=5s .311 v .128,
   Q2 .405 v .263, and last-5-min tie -> final 3 .497 v .625. Bootstrap any candidate with
   E100_BASE/E100_CAND/E100_OUT env on scripts/mod25e_e100.py.
Owner 2026-10-04: prioritise abstract human mechanisms (catch-up, effort, risk,
leader complacency, halftime adjustment) alongside concrete mechanics.
1. Persistence E +10.3 is the largest gap. Ruled out (measured): play-level effort,
   halftime regression (E75), team form variance (E85), desperation plays (E79),
   late turnovers (half-end flip artifact, E79), leader conservatism and weather (E88),
   special-teams form as carrier (E89). Real counter-force +6.7 vs sim +1.4 (E86);
   its resolved channel is a uniform within-type start-yardline shift for the
   opponent of an early over-performer (E88), mechanism unnamed.
2. Fixed today: KGZ (kick overwritten by GZ repick), OKK (halftime-crossing kicks as
   kicker-kept), ADJ (in-game matchup adjustment, small), STF (special-teams form).
3. Open: STF return-coverage kappa clipped at +1.0; mass at 3 short (.115 vs .141).
4. Persistence hunt 2026-10-05 (lanes mod25e-e91-startspot, -e92-penalties, -e95-endspot):
   penalties not the carrier (E92); flip-start coupling damped (E91/E93); A's own H2
   start carries the end-spot gap (E95). YLM (E94 yardline-window punt redraw, YLM=1
   YLM_CLASSES=punt, label suffix yp) full 3-seed run KEPT as play-level fix (measured,
   artifacts/mod25e3/yp_check, base_check): punt start slope .699 v real .692 (base
   .238), punt start 74.16 v 74.19 (base 74.82). E95 own-start gap +.036 -> +.021 yd/pt,
   end-spot +.037 -> +.022 (artifacts/mod25e3/e95_yp). E88 x -.0556 -> -.0599 (real
   -.1326). Checks worse (yp_era/era.txt): SD 15.34 -> 15.58, noise 178 -> 186, pts
   +.43, xq +1.5 -> +4.1: field-position chain now carries more variance; the missing
   counter-force is still unnamed. New base crHpqokgndecsmfwtjo2as2yp.
   Defect logged: STF shifts touchback starts off the rule spot (punt touchbacks .001
   v real .097 in base and yp); STF fit (stf/an.txt) matched real only without the
   touchback exemption, so the fix is a team-form touchback probability, not exemption.
5. E96 (docs/lanes/mod25e-e96-mass3.md): mass-3 gap -.026 is 65% overtime (reg-tie
   rate equal; P(final 3 | reg tie) real .682 v sim .432). E97 (docs/lanes/
   mod25e-e97-ot.md): sudden-death drives lack the walk-off FG (downs 2-3 FG share
   .15-.19 real v .02-.04 sim; pool phase 4 mixes Q4-last-5 rows, sim04_engine.py:255).
   WFG hook (scripts/mod25e_wfg.py, fit artifacts/mod25e3/wfg/fit.json, LOSO) KEPT,
   new base crHpqokgndecsmfwtjo2as2ypw2 (env base + YLM=1 YLM_CLASSES=punt WFG=1).
   Measured 3 seeds (w2_era/era.txt, e96/e96_ypw2.txt): mass3 .114 -> .122 (.141); OT
   mass3 gap -.0168 -> -.0081 [-.017,+.001]; other checks flat (SD 15.59, noise 184.7,
   pts 45.99). Still short: tied entering last 5 min P(final 3) .474 v .625 (regulation
   walk-off FG, same missing "a score ends it" term) -> next unit.
   E98 (docs/lanes/mod25e-e98-regkick.md): PAT rate fitted on 2015+ only and applied to
   all years (mod25_mechanisms.py:323); FG make flat in distance. Hooks PATY/FGD being
   written (smoke). E99/E99h (docs/lanes/mod25e-e99-regwalkoff.md): late-regulation
   FG deficit is state composition, not kick choice (HGB matches real given state);
   real offenses drain the clock and call the last timeout with seconds left (snap
   gsr<=5 in range .39 v .15). CDR hook (scripts/mod25e_cdr.py, CDR=1, suffix cd,
   LOSO fit +.65 loglik/row) smoke partial. NEXT: one 3-seed e5 with base + PATY=1
   FGD=1 CDR=1 once the PATY/FGD smoke reports; spike share (.50 v .28) open.
   Smoke dirs go in artifacts/mod25e3/smoke/ (era globs pick up _s31).
Stale-log audit 2026-10-06 (read-only): play logs (mod25c_noise.py:396-403) keep the pre-wrapper
row for wrappers that edit after base: KICK with KGZ=1 (mod25e_kick.py:281-307 flip, RTD points),
WFG (mod25e_wfg.py:63-85 whole row), CLK (mod25e_clk.py:203-233), KN/KNW (:424-437), TDC
(mod25e_tdc.py:132-150), CDR (mod25e_cdr.py:115-133) clock_elapsed. Patched: FGD (:90-93), endgame
(mod25e_endgame.py:330-336, clock only). Fine: GZ (before base), PATY, STF/YLM/ADJ (next_* only,
not logged). Play-log-scored results to recheck: YLM punt slope/start, E95 end-spot, E97/WFG OT FG
share, CDR snap timing, E32/E44 clock/drive-seconds.
Lessons: exclude half-last rows from every possession-change definition; every
engine wrapper walks frames to the 'offense' frame and binds locals before base.
CPU budget: about 6 of 24 cores; e5 --workers 3, one at a time; agents
single-threaded analysis, foreground only.

## Open
None.

## 2026-10-09 E114 3-seed (measured; artifacts/mod25e3/e114_boot.log, tests/scratch/e114/tgt_3seed.txt)
dw -> dw+TPO+CLQ=2+URG+CKY (label ...kekudwtpq2ugky). Targets all toward real: Q1/Q3 plays 34.74/35.08 -> 33.72/34.10 (real 33.96/34.26); Q2 first snap 18.3 -> 1.1 s (1.3); Q3 pts/team 5.12 -> 4.94 (4.74); Var Q3 change 55.8 -> 54.3 (50.8); trailer 9+ s/snap run 36.2 -> 35.9 (35.2), pass 24.2 -> 23.8 (23.0); final drive score|reach .415 -> .664 (.766), clock|reach .441 -> .150 (.022), P(score) .140 -> .207 (.340); regulation mass3 .0917 -> .0987 (.1004); P(final3|tied last 5) .471 -> .502 (.625). Aggregates flat: SD 15.38 -> 15.36, noise 180.7 -> 178.1 [-6.0,+1.0], pts 45.14 -> 45.26, mass3 .1246 -> .1251, late r2 .126 -> .127. AWAY: P(final3|tied after reg) .573 -> .458 [-.139,-.089] (real .682), OT mass3 .0329 -> .0265. No new hook acts on qtr 5; E117 diagnosing (lane mod25e-e117-otfinal3.md). Provisional base pending E117. GEO=2 smoke on top running (tests/scratch/e114/run_geo.sh -> artifacts/mod25e3/geo_smoke_s31.log).

## 2026-10-09 E114b 3-seed after E117 fix — KEPT (measured; artifacts/mod25e3/e114b_boot.log, tests/scratch/e114/tgt_3seed_b.txt)
NEW BASE crHpqokgndecsmfwtjo2as2ypw2xfdcdtbkhghkekudwtpq2ugky = env + CDW=1 TPO=1 CLQ=2 URG=1 CKY=1 (tests/scratch/e114/run_3seed.sh). v dw: SD 15.38 -> 15.38 (real 14.63), noise 180.7 -> 180.6 (165), strength 56.1 (58.0), late r2 .126 -> .128 (.146), mass3 .1246 -> .1272 [-.001,+.006] (.141), pts 45.14 -> 45.34 (45.21), Q4 slope -.055 -> -.056 (-.060). OT restored: P(final3|tied after reg) .573 -> .568 (.682). Targets: Q1/Q3 plays 33.70/34.04 (33.96/34.26); Q2 first snap 1.2 s (1.3); Q3 pts/team 4.96 (4.74); trailer 9+ s/snap run 35.8 (35.2), pass 23.9 (23.0); final-drive score|reach .671 (.766), clock|reach .170 (.022), P(score) .220 (.340); reg mass3 .0960 (.1004); P(final3|tied last 5) .495 (.625).
Open: SD/noise gap unchanged by clock work; Q3 TD/drive +.016 (GEO targets it: smoke running, artifacts/mod25e3/geo_smoke_s31.log, score with tests/scratch/e115/geo_tgt.py + tests/scratch/e114/tgt.py); final-drive reach .31 v .42 and in-range clock deaths .17 v .02 remain.
Hook rule (E117): never bind a local named `offense` or `off_sim`/`def_sim` in a wrapper; walk frames to engine-only locals.

## 2026-10-09 evening: full candidate (measured smoke s31; tests/scratch/e114/tgt_full.txt)
Candidate L = base + CDW=2 (E123, inc/term clock backoff) + URG2 (E122, replaces URG) + CKY2 (E122, replaces CKY) + GEO=2 (E115/E121, TD iff yards>=yl, engine clock kept) + RTD (E120, opening-kickoff return TDs); label crHpqokgndecsmfwtjo2as2ypw2xfdcdtbkhghkekuw2tpq2g2rty2u2. Smoke v GEO-only smoke: final-drive score|reach .653 -> .838 (real .766), clock|reach .122 -> .054 (.022), reach .320 -> .278 (.423, noisy); open-field TD/play .0113 (.0110); plays/game 147.1 -> 148.0 (like-for-like real 149.3, E123); pts/game 43.4 (real 45.2) — low because of the incompletion deficit (E124b in flight). 3-seed L v base running: tests/scratch/e114/run_3seed_full.sh -> artifacts/mod25e3/e125_boot.log, tests/scratch/e114/tgt_3seed_full.txt, tests/scratch/e115/geo_tgt_3seed_full.txt.
