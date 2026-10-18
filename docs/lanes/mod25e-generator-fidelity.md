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

## 2026-10-09 E125 full candidate 3-seed (measured; artifacts/mod25e3/e125_boot.log, tests/scratch/e114/tgt_3seed_full.txt, tests/scratch/e115/geo_tgt_3seed_full.txt) — NOT ADOPTED
base ...kekudwtpq2ugky -> L ...kekuw2tpq2g2rty2u2: margin SD 15.38 -> 14.50 [-1.04,-.73] (real 14.63, INSIDE gate 13.97-14.87); noise 180.6 -> 168.1 (165); mass3 .127 -> .131 (.141); P(final3|tied after reg) .568 -> .656 (.682); open-field TD/play .0111 (.0110); final-drive score|reach .741, clock|reach .100. AWAY: strength 56.1 -> 42.3 (58.0); late r2 .128 -> .100 (.146); xq cov +0.2 -> -9.4 (-6.4); pts/g 45.34 -> 42.82 (45.21); H1 TD/drive below real in every state (-.008 to -.024). Base pts matched real only because the open-field TD excess offset real deficits (drives, inside-20 TDs). E126 diagnosing strength (lane mod25e-e126-strength.md); ablation without GEO running (tests/scratch/e114/run_3seed_ng.sh -> artifacts/mod25e3/e127_boot.log). F3W=2 (E124, penalty overlay outcome-aware, inc share to real) built, not yet in a 3-seed.

## 2026-10-09 E127 no-GEO ablation 3-seed (measured; artifacts/mod25e3/e127_boot.log, tests/scratch/e114/tgt_3seed_ng.txt) — NOT ADOPTED
B -> B+CDW2+URG2+CKY2+RTD (...kekuw2tpq2rty2u2): SD 15.38 -> 15.42, noise 180.6 -> 182.0, strength 56.1 -> 55.9, late r2 .128 -> .129, mass3 .127 -> .126, pts 45.34 -> 44.94 [-.70,-.14]; so GEO (not the clock hooks) carried E125's strength drop (E126: missing TILT, fixed behind GZT=1). Final drive tie/trail 1-3 WORSE: reach .308 -> .253 (real .423), P(score) .220 -> .188 (.340), drives/g .129 -> .119; score|reach .671 -> .695, clock|reach .170 -> .125. E129 diagnosing (lane mod25e-e129-reach.md). Running: L2 = full + GZT=1 + F3W=2 (tests/scratch/e114/run_l2.sh: smoke s31 then 3-seed -> artifacts/mod25e3/e128_boot.log, tests/scratch/e114/tgt_3seed_l2.txt).

## 2026-10-16 E130 (URG2=2) rejected
3-seed vs B: reach(tie/trail1-3 final Q4) .262 vs .308 (real .423), P(score) .199 vs .220 (.340), finals/g .123 vs .129 (.141). Reach moved away; base stays B. Next: split URG2 elapsed class 3 (sack vs completion), test out-of-bounds stop from pbp. See mod25e-e129-reach.md.

## 2026-10-16 E132 BLS (blowout leader selection, BLS=1) 3-seed vs B: NOT ADOPTED, no effect
Measured (artifacts/mod25e3/e132_boot.log, tests/scratch/e131/{tgt_e132,an_bl,ch_bl}.txt; season bootstrap 1000, 90%): margin SD 15.379 -> 15.399 (+.02 [-.14,+.19]); noise 180.6 -> 181.1 (+.5 [-2.8,+3.9]); strength 56.1 -> 56.0; pts/g 45.34 -> 45.20 (-.14 [-.44,+.14]); Q4 slope -.0564 -> -.0562; late r2 .1276 -> .1253; mass3 .1272 -> .1288 (real .141). Var(M) 235.6 -> 236.3 (real 221.3; +15.0 [+3.6,+25.3]). Leader Q4 pts in 17+ (ch definition, real 3.16): sim 3.60 -> 3.59, gap +.43 [+.13,+.69] unchanged; Var(m4|17+) 51.1 v real 46.2; Q4 total 17+ gap +.46 [+.02,+.97]. 24+ pts/drive Q3-Q4 gap unchanged (Q4 300-600 +.155 [+.04,+.26]). Smoke s31 (1 world) likewise no move. Blowout targets did not move, so base stays B. bls_stats was never logged, so whether the hook reweighted draws is unconfirmed; inference: pass-only backup reweighting is too small a channel, or the pick bypass is inert. Not a refutation of the mechanism (backup share unmeasured in sim; unresolved_below_power).
Next (analysis only): sim04_engine SCORE_CLIP = +-21 makes 24+ states share neighbour weights with 9-21 states. Design a measured extension: widen the clip to the real lead range (cap at the 99th percentile of real |lead| in Q3-Q4, measured from 2009-17 pbp, not tuned), check neighbour counts per widened cell against the minimum used elsewhere, and score it first on 24+ pts/drive (real 1.55, TD/drive .175) before any aggregate. Also log bls_stats in the worker to learn whether BLS fired.

## 2026-10-17 E132 BLS and E133 SCLIP=35 rejected
Both leave blowout leader scoring unmoved (leader Q4 pts 3.57-3.60 v real 3.16; points per drive +.16 at equal drives/plays). E133 mass3 .1224 (away). Base stays B. Next: locate the lead-unconditioned per-play draw (docs/lanes/mod25e-e131-sdgap.md).

## 2026-10-17 E128 L2 = L + F3W=2 + GZT=1 3-seed vs B (measured; artifacts/mod25e3/e128_boot.log, tests/scratch/e114/tgt_3seed_l2.txt) - NOT ADOPTED
Label ...kekuw2tpq2g2rtf2gty2u2. Season bootstrap 90%: margin SD 15.379 -> 15.839 (+.46 [+.27,+.64]; real 14.63); noise 180.6 -> 173.5 (-7.1 [-10.0,-4.0]; real 165); strength_re_var 56.1 -> 77.7 (+21.5 [+16.8,+26.2]; real 58.0, overshoot; E125 without TILT was 42.3); xq_cov +0.2 -> +22.2 (real -6.4); Q4 slope -.056 -> -.013 (real -.060); mass3 .1272 -> .1208 (-.006 [-.011,-.003]; real .141); pts/g 45.34 -> 43.27 (-2.07 [-2.38,-1.82]; real 45.21); late r2 .128 -> .192 (real .146). Adding TILT to Geo/GZ weights overshot strength by +20 where its absence undershot by -14.
E138 (analysis, tests/scratch/e138/ess4.py, ess5.py): see lane mod25e-e126-strength.md.
E138 measured: no double tilt found; GEO=2+GZT=1 S slopes match engine (pass dTD .0844 v .0812) but goal zone (yl<=10) TD slope is +21% pass / +37% run and defense-only pass TD +49%; p(1-p) rescale refuted; GZT=2 not built. Next: in-sim per-play audit and F3W2 isolation (lane mod25e-e126-strength.md).

## 2026-10-18 in flight (orchestrator)
- E158 3-seed launched 07:27 ET: tests/scratch/e158/run_xfa_3seed.sh, out tests/scratch/e158/run_3seed.out, logs artifacts/mod25e3/e158_s{11,12,13}.log.
- E156b launched concurrently: X2 + FLIPA=1 + SCLIP=35 (label ...gx2sc35fa), tests/scratch/e156/run_xfasc_3seed.sh, out tests/scratch/e156/run_3seed.out, boot artifacts/mod25e3/e156b_boot.log, tgt tests/scratch/e156/tgt_3seed_xfasc.txt. Leader scorers (e155 build3/an3/extra with G_TAG) still to run on it after it finishes.
- Offline lanes started (no edits to existing scripts/ while runs are in flight; hook installs deferred as patches): mod25e-e159-strength-x2.md, mod25e-e160-reach-x2.md, mod25e-e161-ydist-pool.md.
- If a run dies, rerun only that script; check with process list on 'mod25e_crH'.
- 2026-10-18 offline queue for after the runs (each is a NEW default-off module; install patches are in its lane): GSTOP=1 scripts/mod25e_gstop.py (Geo.fix keeps the pre-repick row's clock_elapsed, mod25e_geo.py:250/:381; lane e160); axis-specific team-kernel bandwidth scripts/mod25e_axish.py (sim04_engine.py:524 uses one h for off and def; lane e159; fidelity repair, not the strength cure); V1 stochastic lattice rounding of the Geo yard shift (lane e161, passes d1 and reach, fails 4th d2/d6+ and TD yl1-20; E161b running on those). Running: E159d (like-for-like held-out strength on sim), E162 (late-Q4 leader play calling).

## E158 3-seed verdict (ran 2026-10-17 16:00-18:50; read 2026-10-18; measured, artifacts/mod25e3/e158_boot.log, tests/scratch/e158/*_xfa.txt) - KEPT as defect fix; XFA = X2 + FLIPA=1 is the new candidate base (label ...gx2fa)
- v B (season bootstrap 90%): SD 15.25 (-.13 [-.30,+.03]), noise 171.6 (-9.0 [-12.2,-5.7]), strength 61.2 (+5.0 [+1.3,+8.9]), Q4 slope -.036 (+.021), pts/g 43.63 (-1.71), late r2 .1435 (+.016, real .146), mass3 .1255, P(final3|tied after reg) .628 (+.060; real .682), P(final3|tied last 5) .499 (real .625).
- v X2 3-seed: SD 15.56->15.25, noise 180.8->171.6, Q4 slope -.025->-.036, strength 61.8->61.2, pts/g 44.19->43.63 (away from 45.21).
- Start defect repaired: cause-2 start 17+ v real +0.39 [-1.17,+1.80] (X2 -6.60), 24+ Q3Q4 -0.19 [-2.74,+2.22] (X2 -8.16).
- 2026-10-18 orchestrator error: re-launched run_xfa_3seed.sh without checking outputs; killed by PID after 12 world-seasons of s11 were rewritten (07:35-07:43). The engine is NOT bit-deterministic at fixed seed: the rescored tgt (tests/scratch/e158/tgt_3seed_xfa_recheck.txt) differs from the 10-17 table within noise (reach .309 v .311). s11 now mixes two draws of the identical config; 10-17 tables stand. Original e5.json kept at tests/scratch/e158/e5_s11_orig.json. Rule: check for existing outputs before launching any run.
- E159d (like-for-like held-out strength): real 55.8, B 56.2 (+0.4), X2 65.0 (+9.3 [3.0,15.6]); the excess is rating spread arising at B -> D1 (Geo/RTD/GZT); lane mod25e-e159-strength-x2.md.
