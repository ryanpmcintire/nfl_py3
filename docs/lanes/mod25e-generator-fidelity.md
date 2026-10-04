# mod25e-generator-fidelity

## Goal
Keep improving the MOD-25 game simulator until it passes its gates against real
2009-17 games: margin SD band 13.97-14.87, non-strength noise near real 165,
late-season R2 band .132-.166, mass at 3 near .141. Every parameter is a named
mechanism fitted to its own real behaviour (LOSO held-out likelihood); aggregates
are checks only; no compensating constants. The simulator is not served.

## State (2026-10-04, measured, 3 seeds)
Base label `crHpqokgndecsk2mfwt` = env `NEG=1 EGT=1 EGH=1 F2PR=1 QBC=1
OTY=2009-2017 KICK=1 GZ=1 DKF=1 DK2=1 CLK=1 SEL=1 KN=1 KNW=1 FD4=1 CLK2=1 KFIT=1
KFIT_SRC=eiv TDC=1`, run `scripts/mod25e_crH.py e5 --workers 3 --seed {11,12,13}`.
Scored by `scripts/mod25e_era.py` (ERA_VARIANTS/ERA_OUT) -> artifacts/mod25e3/cand_era/era.txt.
Sim vs real: margin SD 15.35 (14.63); noise 181.1 (165); strength 54.5 (58.0);
late r2 .129 (.146); xq cov -1.8 (-6.4); Q4 slope -.058 (-.060); mass3 .120 (.141);
pts/g 45.66 (45.21). Catch-up gap +12.1 (start of day +21.2); persistence E +10.2.
Detailed experiment log E1-E74: docs/lanes/done/mod25e-generator-fidelity-log-2026-10-04.md.

## Tried (2026-10-04 units; all measured, details in the log)
- Kept: DKF exact-distance draws; DK2 nearest-distance + 4th-down go path; CLK
  fitted elapsed (kneel censoring fixed); SEL score-state selection reweight
  (slope .033 -> .014); KN/KNW kneel and defensive-timeout clock; FD4 fitted
  4th-down choice (old pol4 snapped Q2 hs<=450 to 0); CLK2 Q2/Q4 split; TDC
  TD-only elapsed (CLK had pooled TDs with turnovers so late TDs ran out the half);
  KFIT eiv kernel h .59 x std(eo), lam 2.0 (errors-in-variables corrected).
- Rejected: KFIT game/rolling fit (strength 36.5, units mismatch); GZE (yardline
  premise refuted: play-file idx is the pre-GZ row); source-game concentration (E60).
- Analysis scripts: mod25e_catchup, persist, selstate, srcconc, short, half2,
  halfend, trips, clock2 (label via CLK_LABEL/CU_LABEL/DIST_LABEL env; never
  import clock2 at module level, its main() overwrites artifacts).

## Next
1. Persistence E +10.2 is the largest gap; about a third remains unexplained
   after SEL. Probe engine path dependence (per-drive state log).
2. Strength 54.5 vs 58: run this base without KFIT for one seed to isolate the kernel.
3. Mass at 3 short (.120 vs .141): field-goal share of close finishes.
CPU budget: about 6 of 24 cores total. e5 at --workers 3, one at a time; agents
analysis-only, single-threaded, no sims while e5 runs.

## Open
None.
