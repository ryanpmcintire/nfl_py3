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

## Unit 4 done 2026-11-03 ~16:32
scripts/mod41_unit4_placebo_practice.py, artifacts/mod41/unit4_placebo_practice.json, 10 cells recorded (family mod41_residual_load_u4_placebo_practice, unit4_weak_signals_batch.json). 20 looks.
- Future-load placebo (next game's residual, 1056 rows): ll +.0012 [-.0025,.0046] p+ .74, coef -.029, decisive 166-165; real term on same rows +.0043 p+ .95. Caveat: next-game window overlaps current by 2 of 3 games (corr .75), so it is a weak placebo; the gain is a quarter of the real one.
- Same-week swap, 500 draws: mean -.0004, 2.5-97.5 [-.0017,.0010], p+ .23; observed .0041 above every percentile.
- Practice (DNP+limited, same-week report, after opener: mediation only): first stage QB +.0084 [.0017,.0152] p+ .99; receivers +.0126 p+ .91; DB -.0143 p+ .09; RB, OL, DL, LB, all straddle 0. Residual coef stays -.0432 to -.0447, ll gain .0039-.0042 in every arm. No mediation.
Next: term to served combine once forward weeks score; QB practice slope is the one lean (8 slopes, unadjusted).

## Unit 5 done 2026-11-03 ~16:50 (base-model add-one)
scripts/mod41_unit5_base_addone.py (modes true, proxy, true_control), artifacts/mod41/unit5/. Harness = mod23_unit8_span/mod24_u1 true_eval and proxy_eval (ridge on ats_margin, weak_stack FULL set, opener input). Baseline reproduced: 802-701 on 2020-25 (guard in script). Feature: home minus away residual load, beta (const, own, opp) refit each fold on usable rows of prior seasons only; pbp rebuilt from 2011; unusable rows 0.
- 2020-25 (n 1503): record 802-701 = baseline; 62 flips, decisive 31-31; acc +0.00 [-1.00,1.07] p+ .48 (week-block); ll .69550 vs .69692 gain +.00141 [.00047,.00235] p+ .998 (6 season blocks p+ 1.0); Brier .25102 vs .25172; market-only ll .6933. Coef (margin pts per play-index) -.016,-.039,-.053,-.071,-.073,-.090: negative 6/6, growing. Margin MSE gain in-sample .003-.123, out-of-sample .13-.92 (OOS above IS, no overfit gap).
- Permuted-feature control: 800-703, 12 flips 5-7, ll -.00003 p+ .35.
- Proxy 2014-19 (n 1487): 766-721 vs 769-718, decisive 9-12, acc -.20 [-.88,.47] p+ .24, ll -.00021 p+ .20; coef sign unstable (+.065 in 2015), OOS MSE gain negative in 4/6.
- Base route ll gain +.0014 vs combine route +.0041 (different baselines: ridge base ll .6969 vs served logit): base route carries a third of it, none in accuracy.
5 cells recorded (family mod41_residual_load_u5_base_addone, unit5_weak_signals_batch.json), unresolved_below_power. 5 looks.
Next: 2011-13 proxy not run (warm-up); feature scale not tuned.

## Unit 6 done 2026-11-03 ~17:30 (true served base)
scripts/mod41_unit6_served_base.py, artifacts/mod41/unit6_served_base.json. Base = logit(out_of_season_home_probability) from artifacts/pick_probability/20261103T195411Z (stored served record 829-674, ll .68670; raw model 802-701). Units 1-4 base was the raw model.
- LOSO refit base (usable-adjusted) acc .5516, ll .68692, Brier .24689. Term: ll .68288 (gain +.00404 [-.00118,.00936] p+ .940); acc +1.73 pts [-0.66,4.33] p+ .919; Brier .24488; decisive 157-131 of 288 (54.5%); in-sample gain .00434, gap .00030; intercept-only ll .69328. Coef on served logit .921 (.83-1.03), residual -.0439 stable 6/6 (-.0401 to -.0493). Within-week permutation (500) p .002, null mean -.00035.
- Raw base (unit 2): ll +.0041 p+ .95, acc +2.66 p+ .97, decisive 209-193; survives on served base with ~same ll gain, smaller acc gain.
- Frozen forward term fit on raw model logit (fit_base: opener_evaluation per_game, served_logit_coefficient .339); forward scoring applies it to a different probability scale than the card's four-term fit. Describe only; not modified.
2 cells recorded (family mod41_residual_load_u6_served_base, unit6_weak_signals_batch.json), unresolved_below_power.
Next: refit frozen term on the true served logit (new artifact, not overwriting) before forward scoring.
- 2026-11-03 17:29 ET (root decision): unit 6 survives on the true served base (ll +.00404 P+ .94, acc +1.73 P+ .92, decisive 157-131, perm p .002, coef negative 6/6). NOT served now: serving needs a pinned fifth term in pick_probability.py and a v8 enrolment days after v7, and MOD-41 has many looks across units 1-6; the forward ledger and ENG-48 monitor decide. Unit 7 re-freezes the term on the served scale as a new forward term (mod41_residual_load_served) so Week 9 is logged before Thursday kickoff; the raw-scale frozen term stays as written.

## Unit 7 done 2026-11-03 22:35 UTC (served-scale forward term)
Frozen artifacts/forward_terms/mod41_residual_load_served/frozen_coefficients.json (scripts/mod41_freeze_residual_load_served.py; base artifacts/pick_probability/20261103T195411Z/per_game.parquet out_of_season_home_probability, sha eb5427c1; 1503 openers 2020-25, cutoff 2025; hash 12bdeb20). Coef: resid -.04390 per play (LOSO -.0401 to -.0493), served logit .9210 (LOSO .834 to 1.032), intercept .0011. Raw-scale term untouched.
Wired in scripts/forward_terms_week.py (TERMS, mod41_week(term), combine: sigmoid(intercept + .921*logit(card home p) + b*resid)) and TERM_DIRS in scripts/forward_ledger.py. Base probability = pool_card.pick_probability of latest week card (base_card), equal to displayed card (max diff 0.0, 15 games).
Dry Week 9: 15 games, 15 with term, mean -2.36, sd 6.46, range -10.52 to 13.93. Real record NOT written; orchestrator runs `--week 9 --term mod41_residual_load_served` before Thu 2026-11-05 20:15 ET.
- 2026-11-03 17:32 ET (root): the first served-scale Week 9 record used pool_card.csv's pick_probability as base, which is the model-only pool card (it picks BAL 66.2%, IND, NYG), not the served card (JAX 51.1%, DAL, PHI). Voided (renamed VOID_wrong_base_...txt) and rewritten 2026_week09_20261103T223235Z.json with base = latest published card probability (artifacts/clv_ledger/published_picks.parquet displayed_score, the ENG-49 played-card source; forward_terms_week.served_card_probability). With the term, 6 of 15 Week 9 sides would differ from the card (JAX->BAL, CIN->ATL, WAS->LA, SF->LV, SEA->ARI, DAL->IND); forward-only, card unchanged. Note: every other forward term's base is pool_card.csv (raw-scale terms fitted on the raw model logit, so consistent for them).
