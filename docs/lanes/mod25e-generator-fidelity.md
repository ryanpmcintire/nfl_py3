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

## Next
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
Lessons: exclude half-last rows from every possession-change definition; every
engine wrapper walks frames to the 'offense' frame and binds locals before base.
CPU budget: about 6 of 24 cores; e5 --workers 3, one at a time; agents
single-threaded analysis, foreground only.

## Open
None.
