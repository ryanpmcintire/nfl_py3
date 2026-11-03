# MOD-41 Cumulative workload

## Goal
Cumulative workload over the last three games plus overtime and short weeks, as one LOSO term at the opener (ROADMAP MOD-41).

## State
Unit 1 done 2026-11-03. Script scripts/mod41_cumulative_workload.py, artifact artifacts/mod41/cumulative_workload.json. Family mod41_cumulative_workload, 5 looks declared first (total, total placebo shifted one game, defended only, defended placebo, OT+short-week flags), each vs logit_p base and intercept-only, LOSO 2020-25, 1142 of 1503 games usable (4+ prior games, else term 0). Plays = pass/run, no kneel/spike, REG.
- load_total (plays run+defended, 3-game mean minus season-to-date norm): decisive 224-209 (51.7%); acc 55.69 vs base 54.69 (+1.00 pts [-2.17, 4.06] p+ .74); log loss 0.6874 vs 0.6924 (gain +0.0051 [-0.0005, 0.0108] p+ .96); Brier .2471 vs .2496; in-sample ll gain .0054, gap .0003. Coef -0.044/play-index, sign stable 6/6 folds (-0.040 to -0.048). Intercept-only acc 49.6, ll .6933.
- placebo: decisive 110-141, acc -2.06 (p+ .01), ll +0.0002; coef -0.015.
- defended only: decisive 142-153, acc -0.73, ll +0.0006 p+ .67; placebo ll -0.0003.
- flags: decisive 37-60, acc -1.53, ll -0.0010.
6 cells recorded unresolved_below_power.

## Tried
Home-minus-away window term only; no rest-day or permutation null yet.

## Next
Total plays equals both teams' game pace, so load_total is mostly opponent tempo schedule; test with rest_diff in the model and a within-week permutation null. Travel flag not built.

## Open
Placebo shares the sign (-0.015), so part is a persistent tempo trait, not recent load.

## Unit 2 done 2026-11-03
scripts/mod41_unit2_components.py, artifacts/mod41/unit2_components.json, 11 cells recorded (log_loss_improvement, unresolved_below_power, family mod41_cumulative_workload_u2) from unit2_weak_signals_batch.json. Base = served logit (802-701), LOSO 2020-25, 1142 usable of 1503.
- total: ll +.0050 [-.0006,.0107] p+ .96; acc +2.66 [-0.07,5.38] p+ .97; decisive 228-188 of 416; in-sample gap .0003. Permutation null (500, within week) p .002.
- own tempo: ll -.0003 p+ .36, decisive 100-130. opp tempo: +.0002 p+ .59, 125-113. residual load: +.0041 p+ .95, 209-193, null p .002, coef -.044 stable 6/6. joint abc ll +.0042 p+ .93, gap .0015.
- rest_diff alone -.0004; total given rest +.0050 unchanged.
- Unusable 361 = weeks 1-4 (88/92/92/88) plus 1 in week 5; served acc there 56.0% vs 52.5% usable, smaller spreads (4.7 v 5.3), so selection is early-season, not tempo-related.
Carrier: the residual recent load beyond own and opponent tempo; tempo parts carry nothing. Not a tempo trait; new to the served model (inferred), still unresolved.
Next: mechanism (injury or fatigue by position group), same-week placebo on residual, add as fitted term inside the served combine.
