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
   response already match (E75); carrier is later start field position (E76/E77,
   80% within-start-type yardline). The late-turnover shortfall (E77-E79) is
   mostly a DEFINITION ARTIFACT: real half-ending plays carry flip=1, po=0 and
   count as 'lost' (Q2 real run lost .0178 -> .0100 excluding half-last plays,
   sim .0106; Q4 .0165 -> .0088, sim .0096; measured). Any lost/turnover metric
   must exclude half-last rows. RISK and DESP stay out. Next: recheck E77's start
   yardline gap with half-last rows excluded, then the within-type yardline
   mechanism (punt/kick return distance by score state).
2. Strength 57.0 vs 58 now fine; mass at 3 short (.117 vs .141).
CPU budget: about 6 of 24 cores total. e5 at --workers 3, one at a time; agents
analysis-only, single-threaded, no sims while e5 runs. Every wrapper of
pick/pol/DECIDE must bind qtr/offense/off_to/def_to locals before calling base.

## Open
None.

## E79 desperation class (measured, scripts/mod25e_desp.py, artifacts/mod25e3/desp/{chars,fit}.txt)
Class D = lateral row (807) or pass air_yards>=44 (p99 of hsr>120 passes) with hsr<=20 (143, window = widest where deep rate lower bound exceeds ordinary .0111): 950 rows, .41/game, lost .419, TD .007. Real D share at trail>8 Q4 last 450 s .0035 vs sim .0041; kNN k=200 pool already holds .0031 D (45% of states >=1). Only miss: Q4 trail1-8 hsr<=20 D rate real .163 vs sim .024.
DESP=1 fit: LOSO form 5 (half x time bin, deficit, timeouts, field) lam .1 gain 597 LL vs constant, 9/9 seasons; pool m=160 prior 4 (24 looks); 18 form looks. Replay: trail>8 Q4 last 450 lost real .1029 | sim .0752 | sim+DESP .0756; TD .0463|.0466|.0465 -> D explains ~0.0015 of the .028 gap; not a carrier, not for base.
Gap is downs 1-3 ordinary: lost|play real .0638 vs sim .0377 (pass share .854 vs .84; 4th-down share .071|.067, lost|4th .616|.601 match). Next: model 1-3 down turnovers by trailing late as forced throws by pressure state (sacks, deep depth), not extreme plays.
Hook (not applied): scripts/mod25e_crH.py after line 107 add `import mod25e_desp as dp` / `if dp.enabled(): dp.install_desp()` (outermost wrap, after install_tov); after line 164 `if os.environ.get("DESP") == "1": label = label + "z"`.
