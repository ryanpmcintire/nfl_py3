# mod25e-generator-fidelity

## Goal
Keep improving the MOD-25 game simulator until it passes its gates against real
2009-17 games: margin SD band 13.97-14.87, non-strength noise near real 165,
late-season R2 band .132-.166, mass at 3 near .141. Every parameter is a named
mechanism fitted to its own real behaviour (LOSO held-out likelihood); aggregates
are checks only; no compensating constants. The simulator is not served.

## State (2026-10-04 evening, measured, 3 seeds)
Base label `crHpqokgndecsmfwt` = env `NEG=1 EGT=1 EGH=1 F2PR=1 QBC=1
OTY=2009-2017 KICK=1 GZ=1 DKF=1 DK2=1 CLK=1 SEL=1 KN=1 KNW=1 FD4=1 CLK2=1 TDC=1`
(no KFIT, no RISK), run `scripts/mod25e_crH.py e5 --workers 3 --seed {11,12,13}`.
Scored by `scripts/mod25e_era.py` -> artifacts/mod25e3/risk3_era/era.txt.
Sim vs real: margin SD 15.25 (14.63); noise 176.0 (165); strength 57.0 (58.0);
late r2 .130 (.146); xq cov -1.4 (-6.4); Q4 slope -.058 (-.060); mass3 .117 (.141);
pts/g 45.62 (45.21). Catch-up gap +11.3 (start of day +21.2); persistence E +9.7.
Detailed experiment log E1-E74: docs/lanes/done/mod25e-generator-fidelity-log-2026-10-04.md.

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
1. Persistence E +9.7 is the largest gap; play-level effort and halftime
   response already match (E75). Carrier is later start field position (E76/E77):
   real trailing teams generate more turnover/downs starts, concentrated in the
   last 7.5 min of Q4 (trailing run lost .023 real vs .013 sim: laterals and
   desperation plays, inferred). Reweighting candidate rows cannot create rows
   the pool lacks: next model end-of-game desperation as its own play class
   fitted from real (laterals, hail marys, forced throws) by score x time.
2. Strength 57.0 vs 58 now fine; mass at 3 short (.117 vs .141).
CPU budget: about 6 of 24 cores total. e5 at --workers 3, one at a time; agents
analysis-only, single-threaded, no sims while e5 runs. Every wrapper of
pick/pol/DECIDE must bind qtr/offense/off_to/def_to locals before calling base.

## Open
None.
