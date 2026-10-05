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

KGZ (orchestrator, from E82b trace): kick pol (mod25e_kick.py) drew the
after-score kickoff, then passed the row to GZ which repicked goal-to-go rows and
overwrote points/flip/next_yardline (mod25e_gz.py:117-121): 42% of TD plays lost
the kick draw, 29% of final TDs had none (late lead>8: kick draw only 29%).
Fix KGZ=1 (label j): kick pol calls base first, then applies the kick to the
final row. First 3-seed run crashed (KeyError offense: the layer outside kick
lacks it); fixed by walking frames to the engine frame (gz pattern); 1x1 smoke
OK. 3-seed result (kgz_era, fpos2/fpos2_j.txt; measured): after-TD receiver start
slope Q4 -1.51 -> -2.48 (real -2.86), all Q -0.81 -> -1.18 (-1.69); Q1 still
-0.30 vs -2.57 (separate mechanism). Checks: SD 15.25 -> 15.35, noise 176.0 ->
178.2, xq -1.4 -> +2.6, catch-up +11.3 -> +13.8 (within ~2 seed-noise widths).
KEPT as a verified pipeline-defect fix: NEW BASE crHpqokgndecsmfwtj (+KGZ=1).
Next: Q1 after-TD kickoff start slope gap; within-type yardline still +4.9.

## Open
None.

## E79 desperation class (measured, scripts/mod25e_desp.py, artifacts/mod25e3/desp/{chars,fit}.txt)
Class D = lateral row (807) or pass air_yards>=44 (p99 of hsr>120 passes) with hsr<=20 (143, window = widest where deep rate lower bound exceeds ordinary .0111): 950 rows, .41/game, lost .419, TD .007. Real D share at trail>8 Q4 last 450 s .0035 vs sim .0041; kNN k=200 pool already holds .0031 D (45% of states >=1). Only miss: Q4 trail1-8 hsr<=20 D rate real .163 vs sim .024.
DESP=1 fit: LOSO form 5 (half x time bin, deficit, timeouts, field) lam .1 gain 597 LL vs constant, 9/9 seasons; pool m=160 prior 4 (24 looks); 18 form looks. Replay: trail>8 Q4 last 450 lost real .1029 | sim .0752 | sim+DESP .0756; TD .0463|.0466|.0465 -> D explains ~0.0015 of the .028 gap; not a carrier, not for base.
Gap is downs 1-3 ordinary: lost|play real .0638 vs sim .0377 (pass share .854 vs .84; 4th-down share .071|.067, lost|4th .616|.601 match). Next: model 1-3 down turnovers by trailing late as forced throws by pressure state (sacks, deep depth), not extreme plays.
Hook (not applied): scripts/mod25e_crH.py after line 107 add `import mod25e_desp as dp` / `if dp.enabled(): dp.install_desp()` (outermost wrap, after install_tov); after line 164 `if os.environ.get("DESP") == "1": label = label + "z"`.

## E80 start-position decomposition, half-end exclusion, strategy slopes (measured, scripts/mod25e_fpos2.py, artifacts/mod25e3/fpos2/fpos2.txt, base crHpqokgndecsmfwt s11-13, 100 boots, 102 looks)
- Half-last/game-last exclusion barely matters at drive level: 1522 real later drives follow End of half/game and are start type (28 fewer other starts than E77); the E77 turnover contamination was play-level. Gap F real -14.49 vs sim -8.87 (+5.62 [+3.59,+7.86] pp 1.00): mix +1.22 (22%) vs within-type +4.40 (78%); within by type: other +2.02, punt +0.90, koTD +0.82, koFG +0.66 (all pp >=.98).
- Level gap, not strategy: sim starts are worse for the receiver after every type (punt yl0 +0.81, net punt +1.0 yd longer in every lead band, other +2.0). Punt slope on punter lead matches (net -0.437 real, -0.505 sim, s-r pp .14); touchback share matches; punting is NOT the strategic carrier. Fair catch/return split unavailable (raw pbp has no kick/return columns).
- Strategic gap is kickoffs and turnover returns: after a TD receiver start yl0 slope per lead band real -1.69 vs sim -0.81 (s-r +0.88, pp 1.00; Q1 +2.27, Q4 +1.35); after FG -0.78 vs -0.47; other-drive return gain slope +1.26 vs +0.62 (pp 0.00). LOSO adding lead band to quarter mean: ko TD +3.1% SSE 9/9 seasons, FG +0.8% 7/9, turnover gain +0.8% 8/9, punt net +0.24% 9/9.
- Engine read: punts and turnovers draw a real row by kNN over dist/fp/scaled score/time/timeouts (sim04_engine.py:284-322, 398-433; features 225-236, score scale 216-222, k via tree.query line 316/429) and apply its absolute next_yardline (1003, 1112-1117), so they condition on score only through that kNN metric. Scored-drive kickoffs use mod25e_kick.py:259-301, nearest sqrt(N) kicks on (score diff after score, time) with kd/gs scaled by std. Mechanism for TD kickoffs: leading receiver gets worse-for-kicker starts (onside-type short fields and kneel/return choice) that the nearest-kick draw smooths.
- Fix form (not built): LOSO-fit the TD/FG kickoff draw kernel bandwidth on (kd, time) per quarter (replace the std scale by a held-out-likelihood bandwidth), and for turnovers a lead-band term in the draw metric chosen LOSO; a constant shift is excluded.

## E81 kickoff draw and turnover-return fits (measured, scripts/mod25e_kos.py, artifacts/mod25e3/kos/{kick_fit,tov_fit}.txt; LOSO 2009-17, no sim)
- Kick draw (18338 after-score kicks; 121 models x 5 lam = 605 looks): current sqrt(N)-nearest/std draw already reproduces the real kick-level start slope after TD (real -1.27 vs LOSO cur -1.04 per receiver lead band; all kicks -0.87 vs -0.81; band +2 start 69.3 vs 70.1). Best kernel (eng-score bw 2, raw time bw 150 s, lam 512, lam at grid edge) wins LL +0.0111/kick 9/9 (nested-lite 9/9) but flattens slopes (-0.81 TD, band +2 72.2): KOS kick draw is installed, not shown to move the sim gap; the E80 drive-level -1.69 vs -0.81 is not the kick draw.
- Turnover return (9060 flip/no-score run-pass plays, 21 models x 5 lam): turnover-only kNN k200 down-grouped wins LL +0.0888/play 9/9 (nested 9/9) over the engine's all-row k200 turnover share; score weight in the metric loses (ws0 best, ws.5 -.0007, ws>=1 worse); LOSO r slope on committing-team lead real +0.374 cur +0.411 best +0.213. Score term refuted for the draw metric; gain is pooling (more turnover rows).
- Hooks, not applied (KOS flag off = old path textually untouched; label suffix x): mod25e_crH.py replace lines 62-67 block by `import mod25e_kos as ks` / `if ks.enabled(): ks.install_kos()` / `elif rk.kick_enabled(): rk.install_kick()` / `else: mk.install_kick()`; after install_tov (line ~107) `import mod25e_kos as ks` / `if ks.enabled(): ks.install_tovret()`; after line 164 `if os.environ.get("KOS") == "1": label = label + "x"`.

## E82 after-TD kick path (measured, scripts/mod25e_kpath.py -> artifacts/mod25e3/e82/kpath.txt; replay of mod25e_kick.py:259-301 draw on s11-13 TD states, 1 in 4 = 50623)
- Definitions: E80 x = next drive offense lead band (kicker-kept onside flips sign), E81 x = receiver always; same edges. Sim 97.3% receiver-next, 2.7% kicker-kept (draw expects ~3.4%; band +2 4.7% vs draw 6.5%).
- Sim actual slope -0.80 (E81 x) / -1.06 (E80 x) vs draw expectation -1.24: draw is not reproduced. Band +2 yl 71.6 actual vs 68.9 expected; t<300 s: 67 vs 57-58; 14% of that cell's next yl lies outside the draw's near set (yl up to 93). Real late band +2: 69% start <=60, sim 33%.
- Real E80x -0.93 vs E81x -0.86 on all kicks: definitions explain little; defect is in the sim path late with receiver lead. Trace (scripts/mod25e_ktrace.py, 1x1 base sim, out artifacts/mod25e3/e82/trace_rows.csv) was running to log outermost pol in/out; unfinished. Next: read trace, compare pol output next_yardline to next play yl.

## E82b trace (measured, scripts/mod25e_ktrace.py -> artifacts/mod25e3/e82/trace_rows.csv; 1x1 base sim, 1412 TD kickoffs, 846 FG)
- DEFECT: kick pol (mod25e_kick.py:268-298) is outer to GZ and passes its drawn kick row via base() into gz pol (mod25e_gz.py:64-67, pol_inner 72-123), which repicks goal-to-go run/pass rows and overwrites points, flip, next_yardline (117-121). Kick draw honoured on 58% of engine-TD plays (49% of final TDs); 32% of kick-drawn TDs have points changed (25% become non-scoring, nyl 2); 29% of final TDs are gz-created TDs with no kick draw (raw row nyl, mean 75.8). FG unaffected (0%).
- Late (end of quarter <300 s) receiver lead >8, n=62 TDs: kick draw 29% (mean nyl 60.1), gz-replaced 34% (70.5), gz-new 37% (76.3); all rl>8 n=134: 33/34/34%. Honoured draws alone carry the draw slope; the rest wipe it. Later layers (dk/dk2/clk/kn/sel/fd4/tdc) alter 0% of nyl (final == gz out); next play yl == final nyl 92% (6% differ >0.5 yd, ordinary start rules).
- Fix form (not built): resolve the GZ repick before the kick decision (kick evaluates on the post-GZ row), or install kick inside gz; no constant.
