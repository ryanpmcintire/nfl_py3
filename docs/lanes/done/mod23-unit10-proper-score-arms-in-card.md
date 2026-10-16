# MOD-23 unit 10: proper-score arms inside the four-term card

## Goal
Test whether the MOD-23 arms that lose ~1 accuracy point model-alone but win log loss and RPS (acc-u2) give the fitted four-term card probability more usable information than the served model logit.

## State
Declared before scoring (2026-10-16). History check (read): unit 9 did this for unit 6 arm B only; acc-u2 Next ("fit one calibrated probability combining pick signal") was not done for unit 1/3/4 arms; no weak_signals family covers it.
Arms (3): (a) card with model_logit replaced by unit 1 greedy (nested-family) logit; (b) replaced by unit 3 compact_net logit; (c) served model_logit and compact_net logit both as inputs (five terms). Unit 4 arms not run (blend_compact_net identical to compact_net; others are shrink variants, unit 4 arms share u1/u3 information).
Logit = logit of stored walk-forward p (artifacts/mod23_unit1/greedy_per_game.parquet, mod23_unit3/compact_net_per_game.parquet), joined on game_id, no refit. Same LOSO 2020-25 fit as scripts/mod23_unit9.py (scripts/per_flag_weights_eval.py fit/predict). Views tuesday and full.
Looks: 3 arms x 2 views x 2 paired deltas (accuracy, Brier) = 12; log loss, reliability, changed picks, folds, seasons descriptive. Prior family looks: unit 9 four looks; unit 1/3/4 model-alone cells.
Script scripts/mod23_unit10.py -> tests/scratch/unit10_results.json.

## Tried
Done 2026-10-16: tests/scratch/unit10_results.json; 6 Brier cells recorded (family mod23_unit10_proper_score_arms_in_card, unresolved_below_power, 107 season-weeks). Served: Tue 836-667 Brier .246517 LL .686185; full 863-640 .244838 .682774.
Tuesday: a_u1 832-671 acc -0.27 P+ 0.31, Brier P(better) 0.66, LL P 0.67, changed 82-86. b_u3 840-663 acc +0.27 [-0.43,0.89] P+ 0.76, Brier -0.00030 [-0.00080,0.00025] P 0.86, LL -0.00059 P 0.85, changed 57-53 (57 of 110), seasons Brier better 4/6. c 835-668 acc -0.07, Brier P 0.59.
Full: a_u1 858-645 acc -0.33, Brier P 0.57. b_u3 860-643 acc -0.20 [-1.19,0.56] P+ 0.34, Brier -0.00019 P 0.80, LL P 0.80, changed 33-36. c 845-658 acc -1.20 P+ 0.08, Brier P 0.47 (collinear: served coef -0.0 to -0.49, u3 +0.37 to +0.96).
Fold coefs stable for a/b (u1 .27-.68, u3 .25-.54). Reliability: no arm fixes the 0.45-0.5 bin (pred .476, obs .42-.45) in any view.

## Next
Do not serve. Best is b_u3: Brier/LL gain ~3e-4/6e-4 with P 0.80-0.86, accuracy sign flips between views. Keep u3 logit as a weekly side log beside arm B; revisit if a pooled test across the u3 and B logits is declared.

## Open
Zero crossing closes nothing. Serving needs out-of-season Brier/LL gain with P >= the owner bar; none met.
