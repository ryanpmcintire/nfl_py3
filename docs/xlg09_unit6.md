# XLG-09 unit 6 predeclaration (written before running scripts/xlg09_unit6.py)

Question. Unit 5's pooled term had a negative fitted beta in all 13 folds while
the CFB-only term's was positive. The pooled logit for each training season was
produced by a fit that included the outer held-out season's outcomes (stacking
leak), which mechanically yields a negative downstream loading. Unit 6 removes
that leak with nested LOSO and re-grades the same cells. It tests the
mechanism "unit 5's wrong sign is a stacking artifact"; it is not a new
serving candidate.

Population, features, ridge, seeds, base terms, bootstrap draws: identical to
unit 5 (Unit 3 four-term base for accuracy, Unit 4 Tuesday two-term base for
line movement; 2013-2025 graded, 2020-2025 subset; CFB point-in-time, seasons
< target season).

Nested protocol. For outer held-out season h: every training season s != h
gets its pooled prediction from a joint fit on CFB seasons < s plus NFL seasons
not in {s, h}; season h gets its prediction from CFB < h plus NFL != h (as in
unit 5). The fifth term is then fitted on training seasons only and scored on h.

Cells (4 looks, plus per-fold beta table): accuracy all-graded and 2020-2025
vs the four-term base; line movement toward the pick all-graded and 2020-2025
vs the Tuesday base. Metric and intervals as unit 5 (season-block bootstrap,
probability_positive reported). Decision use: if the nested beta turns
non-negative and the cells move toward zero or positive, unit 5's wrong sign
is attributed to the stacking leak and the pooled arm is reopened as
unresolved_below_power; if the beta stays negative the mechanism refutation
stands. Nothing here is served; no rule flips a side.

## Results (run once, 2026-09-29; measured from artifacts/xlg09_unit6/20260929T203851Z/summary.json)

Command: .tools/uv.exe run --no-sync python scripts/xlg09_unit6.py (52 s).
n=3,234 (2013-2025), 1,503 (2020-2025); 4 cells looked at.

- Nested pooled beta stays negative in 13 of 13 folds, accuracy arm -0.168 to
  -0.015 (mean -0.117), line-move arm -0.162 to -0.003 (mean -0.106). Unit 5
  unnested was -0.134 to -0.087. The stacking leak did not create the wrong
  sign; it only slightly exaggerated magnitude.
- Accuracy vs four-term base: all-graded +0.46 pts [-0.78,+1.78] P+ 0.749
  (records 1741-1493 vs 1726-1508); 2020-2025 -1.00 pts [-2.38,+0.38] P+
  0.079 (unit 5 unnested -1.40 [-2.30,-0.51]; interval now crosses zero).
- Line movement toward the pick vs Tuesday base: all-graded -0.088 pts
  [-0.127,-0.054] P+ 0.0; 2020-2025 -0.085 [-0.149,-0.031] P+ 0.0. Unit 5
  unnested was -0.079 and -0.069. Resolved wrong sign again.
- Verdict (inferred): the stacking-artifact explanation is refuted. Unit 5's
  line-move closure stands and is now replicated under a leak-free protocol.
  The accuracy cells remain unresolved_below_power (all-graded) and
  unresolved (2020-2025, P+ 0.079, no longer resolved wrong sign).

## Record commands (not run; orchestrator runs serially)

nfl-ats weak-signals record --name xlg09_unit6_nested_pooled_vs_base_linemove_all_graded --description "Unit 6 nested-LOSO pooled two-league term (stacking leak removed) as 3rd fitted term, line movement toward pick vs Tuesday base, 2013-2025" --source artifacts/xlg09_unit6/20260929T203851Z/summary.json --effect -0.08797155225726655 --effect-units ats_points --classification refuted_mechanism --closing-ground wrong_sign_resolved --league nfl --season-start 2013 --season-end 2025 --interval-low -0.1265123323864504 --interval-high -0.05385010217488092 --probability-positive 0.0 --sample-games 3234 --sample-blocks 13 --reliability 0.14132263925353555 --family xlg09_unit6_nested_pooled --category market --classification-evidence "Whole season-block interval below zero and nested beta negative in 13 of 13 folds; leak-free replication of unit 5" --plain-summary "Removing a stacking artifact does not rescue the pooled college-plus-NFL term; it still moves the line away from the pick."

nfl-ats weak-signals record --name xlg09_unit6_nested_pooled_vs_base_linemove_2020_2025 --description "Same as all-graded, 2020-2025 subset" --source artifacts/xlg09_unit6/20260929T203851Z/summary.json --effect -0.0854956753160346 --effect-units ats_points --classification refuted_mechanism --closing-ground wrong_sign_resolved --league nfl --season-start 2020 --season-end 2025 --interval-low -0.14940517844646606 --interval-high -0.031270572745227126 --probability-positive 0.0 --sample-games 1503 --sample-blocks 6 --reliability 0.14132263925353555 --family xlg09_unit6_nested_pooled --category market --classification-evidence "Whole season-block interval below zero; same mechanism" --plain-summary "On the last six seasons the leak-free pooled term still moves the line the wrong way."

nfl-ats weak-signals record --name xlg09_unit6_nested_pooled_vs_base_accuracy_all_graded --description "Nested pooled term as 5th fitted term, accuracy vs four-term base, 2013-2025" --source artifacts/xlg09_unit6/20260929T203851Z/summary.json --effect 0.4638218923933235 --effect-units accuracy_points --classification unresolved_below_power --league nfl --season-start 2013 --season-end 2025 --interval-low -0.7827359921758633 --interval-high 1.7759792467429907 --probability-positive 0.7485 --sample-games 3234 --sample-blocks 13 --reliability 0.14132263925353555 --family xlg09_unit6_nested_pooled --category market --plain-summary "Small positive accuracy point estimate, not resolvable at this size."

nfl-ats weak-signals record --name xlg09_unit6_nested_pooled_vs_base_accuracy_2020_2025 --description "Same, 2020-2025 subset" --source artifacts/xlg09_unit6/20260929T203851Z/summary.json --effect -0.9980039920159722 --effect-units accuracy_points --classification unresolved_below_power --league nfl --season-start 2020 --season-end 2025 --interval-low -2.3826208829712647 --interval-high 0.37523452157598447 --probability-positive 0.07925 --sample-games 1503 --sample-blocks 6 --reliability 0.14132263925353555 --family xlg09_unit6_nested_pooled --category market --plain-summary "Negative point estimate on six seasons; interval reaches above zero, not resolved."
