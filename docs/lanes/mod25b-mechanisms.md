# mod25b-mechanisms

## Goal
Find which football mechanisms the SIM-04 engine gets wrong that cost mass at 3 and 7, add learned decision policies (fit to decisions, never to margins), and check shape on held-out seasons. Rule: no parameter or repair may target key numbers; shape is a diagnostic only.

## State
Script `scripts/mod25_mechanisms.py` (diag, mech commands; wraps engine by source patch in a private namespace, engine and generator untouched). Outputs `artifacts/mod25_mechanisms/` (diag_before.json/txt, mech_<variant>.json, mech_run1.txt). Tables and policies fit on 2009-2017; real comparison 2018-2025 (eval) and 2009-2017 (train, in-sample).
**Measured, neutral engine 12k games vs real 2018-2025 / real 2009-2017:** m3 .110 vs .143/.146, m7 .071 vs .086/.090, SD 14.8 vs 14.3/14.9, pts/game 41.1 vs 45.6/44.5, punts 9.7 vs 7.9/9.6, 4th-down go rate .098 vs .191/.124.
Engine DOES condition on score, time, timeouts (kNN with those features; late-phase 4th-down classifier) but the classifier is argmax (deterministic), 4th-down policy only in late phases, no timeouts in it, score kernel blurs exact states.
Divergences: 2pt choice (sim attempt about 15 percent vs real 80 percent when down 2 after TD); tied game at 2:00 ends within 3 in .61 sim vs .82/.75 real; trailing 1-3 at 5:00 final=3 .25 vs .34/.29; FG-from-tie last score .057 vs .087/.091 of games; OT tie .16 vs .068/.049.
**Measured variants (12k neutral games, tables 2009-2017; real 2018-2025 / 2009-2017):** m3 base .1085; +4th-down policy .1077; +late clock policy .1045; +2pt policy .1183; all three .1129; all + finer score/time neighbour scale (looks fine, fine2, 2 looks declared before running) .1221 / .1244 (real .143/.146). m7 .070 -> .0805 (all) -> .077 (fine2; real .086/.090). 4th-down go rate .099 -> .133 (real train .124). 2pt at down-8: P(8) .07 -> .27-.32 (real .36/.32). Held-out generator-conditioned run (8 worlds x 8 seasons, scale 2, yard gain 2, bias .75, `artifacts/mod25_mechanisms/gen_run1.txt`): m3 .089 -> .109, m7 .0625 -> .0773 (gate .078-.098, short by .001), share le3 .190 -> .184 (no move), SD 15.6-15.7 (gate 13.97-14.87, strength scale not shape).
Not fixed: tied at 2:00 turnover rate per play .10 vs .06-.08 real; late-game clock burn per play (5:00 trailing 1-3: 19.5 s vs 21 real); OT drive length 145 s vs 162-174 and OT drive scoring .36 vs .40-.43; OT tie .15 vs .05-.07. Pattern: neighbour pool mixes desperate-trailing plays into tied/close states. Finer state resolution moves clock, OT and m3 toward real monotonically (only 2 settings tried; not tuned).

## Tried
- Real vs sim decomposition (last score type, state at start of last possession, state at t-minus 30/15/5/2 min, 4th down by score x time x field, kneel/spike, OT, 2pt).
- Policies: 4th-down go/FG/punt, late downs 1-3 run/pass/FG/kneel/spike, post-TD 2pt choice; all HistGradientBoosting on real decisions, sampled (not argmax); outcomes drawn from class-restricted nearest neighbours.

## Next
Try state-resolved ball-security and clock-burn policies (turnover rate and seconds per snap as functions of score/time/field fitted on real plays), OT possession model, timeout use. Regenerate with mod25_produce only after owner review; scripts/mod25_mechanisms.py is not wired into the producer.

## Open
Era drift (go rate .12 train vs .19 eval) is not learnable from pre-2018 data.

## Orchestrator note 2026-10-01 (inferred, from measured numbers)
Within-game noise is too large. Real: SD 14.5 with R-squared about .15, so the
non-strength variance is about 179. Sim at scale 2.5: SD 16.7 with R-squared
.148, about 237, roughly 32% too much. Matching real predictability therefore
needs inflated strength, which widens margins and cuts close games. The
measured turnover rate in late tied states is about double real (.119 vs .061),
a direct noise source. Next unit: fit ball security (turnover and fumble rates
by state), clock burn, OT possessions and timeouts on real plays. Then
re-choose the strength scale from its own targets (predictability, EPA
spread), never from the margin shape, and re-check the shape held out.
