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

## Unit 3 step 1 done 2026-11-03 (~16:00 local)
Frozen: artifacts/forward_terms/mod41_residual_load/frozen_coefficients.json via scripts/mod41_freeze_residual_load.py. Fit on all 1503 openers 2020-25 (served logit + s_resid, unpenalised logistic, unusable rows carry 0). b = -0.04386 logit per play of residual; LOSO by held-out season -0.0395 to -0.0501 (all negative). Decomposition beta (const, own, opp) = -0.1004, 0.1113, 0.3410 from usable rows. Constants: WINDOW 3 (unit 1 declared), usable = 4+ prior games (WINDOW+1). Data cutoff: 2020-25 openers; pbp snapshot 20261103T172043Z holds 2026 through week 8 only. hash a24aa4fa.
Wired as term mod41_residual_load in scripts/forward_terms_week.py (mod41_week). Parity check: week 9 2025 recomputed from team_state equals historical s_resid to 1e-15 (14 games). Dry Week 9 2026: 15 games, 15 with term, mean -2.36, sd 6.46 (historical sd 4.95). Real record NOT written; orchestrator runs it.
Step 2 in flight 2026-11-03: scripts/mod41_unit3_injury_mechanism.py (Out/Doubtful counts by 7 position groups from data/players/raw/20261101T141534Z/injuries.parquet, home minus away; first-stage slope on s_resid and s_resid ll gain given each group's count; 15 looks). Output artifacts/mod41/unit3_injury_mechanism.json when done; record cells after.
Step 2 done 2026-11-03 ~16:15: artifacts/mod41/unit3_injury_mechanism.json, 9 cells recorded (log_loss_improvement, unresolved_below_power, family mod41_residual_load_u3_injury_mechanism, batch unit3_weak_signals_batch.json). Out/Doubtful counts, home minus away, same-week report (published after opener, so mediation diagnostic only). 15 looks (9 mediation arms incl. none, 7 first-stage slopes in count units, not recorded).
- Residual load ll gain alone +.0041 [-.0012,.0094] p+ .945. Given group count: QB .0042 (.947), RB .0042 (.950), receivers .0039 (.938), OL .0041 (.946), DL .0041 (.948), LB .0039 (.934), DB .0042 (.955), all groups .0042 (.945). Residual coefficient stays -.0434 to -.0444 in every arm.
- First-stage slope of count on residual: QB -.0012 p+ .29, RB -.0021 .25, receivers +.0061 .86, OL +.0002 .55, DL -.0024 .29, LB +.0039 .83, DB -.0057 .20; all intervals straddle 0.
Reading (inferred): no sign the effect runs through Out/Doubtful designations; receivers and LB lean the expected direction, unresolved. Not closed.
Next: same-week placebo on residual; practice-status (DNP/limited) rather than game-status; add term to served combine once forward weeks score.
