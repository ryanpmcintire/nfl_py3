# mod25e-generator-fidelity

## Goal
Keep improving the MOD-25 game simulator until it passes its gates against real
2009-17 games: margin SD band 13.97-14.87, non-strength noise near real 165,
late-season R2 band .132-.166, mass at 3 near .141. Every parameter is a named
mechanism fitted to its own real behaviour (LOSO held-out likelihood); aggregates
are checks only; no compensating constants. The simulator is not served.

## State (2026-10-05, measured, 3 seeds)
Base label `crHpqokgndecsmfwtjo2as2` = env `NEG=1 EGT=1 EGH=1 F2PR=1 QBC=1
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
Lessons: exclude half-last rows from every possession-change definition; every
engine wrapper walks frames to the 'offense' frame and binds locals before base.
CPU budget: about 6 of 24 cores; e5 --workers 3, one at a time; agents
single-threaded analysis, foreground only.

## Open
None.
