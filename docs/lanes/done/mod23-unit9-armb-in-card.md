# MOD-23 unit 9: arm B model logit inside the four-term card

## Goal
Swap the four-term probability's model-logit input for unit 6 arm B's walk-forward logit (same games, same LOSO folds, same fit) and grade paired against the served four-term in the Tuesday (move zeroed) and full views.

## State
Declared before scoring (2026-10-16). Script scripts/mod23_unit9.py reusing scripts/per_flag_weights_eval.py fit/predict/metrics; served frame artifacts/pick_probability/20261013T222004Z/per_game.parquet (1503 decisive games 2020-25); arm B logit = logit of p in artifacts/mod23_unit6/B_run.parquet joined on game_id (p checked to be home-cover probability against the served model_probability).
Arms: S served four-term; B four-term with model_logit replaced by arm B logit. Views: tuesday, full. Looks: 2 views x 2 paired deltas (accuracy, Brier) = 4; log loss, reliability, per-season, per-fold coefficients, changed-pick record, overlap correlations descriptive. Season-block bootstrap (6 blocks), 10000 draws.

## Tried
Done 2026-10-16, tests/scratch/unit9_results.json, unit9.log; 4 cells recorded (family mod23_unit9_armb_in_card, unresolved_below_power).
Tuesday: served 836-667 Brier .246517 LL .686185; B 844-659 .246506 .686166. +0.53 pts [-0.07,1.28] P+ 0.94; Brier P(better) 0.53; changed 34, B 21-13. Full: served 863-640 .244838 .682774; B 860-643 .244814 .682721. -0.20 pts [-0.59,0.20] P+ 0.13; Brier P(better) 0.59; changed 13, B 5-8.
Model coef per fold stable (B .26-.39 vs served .25-.41 Tuesday); flag coef identical.
Overlap 2020-25: B logit minus served logit corr with signed QB flag (home-away) 0.30; QB flag corr with flag_sum -0.02, move -0.15, move-available 0.01. Card availability term is move availability, not QB, so not already priced.

## Next
Do not serve: Brier/LL gain ~1e-5 (P~0.5-0.6), accuracy sign flips between views. Keep arm B as a weekly side log.

## Open
Overlap target: arm B flag (own QB-change, home minus away signed) vs flag_sum and the move/availability terms.
