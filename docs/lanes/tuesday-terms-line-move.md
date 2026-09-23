# Tuesday-terms line-move yardstick

## Goal
Six terms added to the served four-term pick probability were all unresolved
on cover accuracy. Re-grade the five Tuesday-knowable ones (drop the market-move
contaminated base terms) against Tuesday-open-to-close line movement toward the
pick, a lower-variance yardstick (ENG-47, `line_move_toward_pick` in
src/nfl_ats/clv.py), paired vs a Tuesday-knowable base (model_logit +
composition_flag_sum only). LOSO 2020-2025, season-block bootstrap, plus the
accuracy companion. New script only; no src/ edits, no tests, no commits.

## State
Script written and run once for real: scripts/tuesday_terms_line_move.py
(ruff clean). Reuses feature builders from players_on_field_rating_eval.py
(diff_lineup_total, diff_divergence), lead65_gated_flag_in_fit.py
(gated_flag_sum), opener_error_transfer_unit2.py (cfb_transfer_logit),
pooled_signal_sixth_fit_new_family.py (reddit_home_comment_ratio_elevated),
and cell_stats/block_bootstrap from line_move_yardstick_paired_eval.py.
Artifact: artifacts/tuesday_terms_line_move/20260923T205836Z/results.json.
1,503-game population, all five terms paired on the full population
(no coverage loss). LOSO 2020-2025, season-block bootstrap primary,
week-block secondary, 2000 draws, seed 20260923.

Results (line_move_toward_pick primary, ats_points; accuracy companion,
fraction *100 = accuracy_points):
- diff_lineup_total: line-move -0.0100 pts, 95% CI [-0.0233, +0.0027],
  P+=0.056 (week CI [-0.0244,+0.0026] P+=0.060). Accuracy +0.13 pts,
  CI [-0.28,+0.46] P+=0.707. unresolved_below_power.
- diff_divergence: line-move -0.0206 pts, CI [-0.0586,+0.0113] P+=0.133
  (week CI [-0.0506,+0.0064] P+=0.070). Accuracy +0.13 pts,
  CI [-0.63,+1.09] P+=0.560. unresolved_below_power.
- cfb_transfer_logit: line-move +0.0143 pts, CI [-0.0249,+0.0714] P+=0.663
  (week CI [-0.0182,+0.0518] P+=0.772). Accuracy -0.07 pts,
  CI [-0.57,+0.39] P+=0.363. unresolved_below_power.
- reddit_home_comment_ratio_elevated: line-move -0.0130 pts, CI
  [-0.0261,-0.0021] P+=0.000 (week CI [-0.0250,-0.0010] P+=0.018);
  decisive 4-14. Accuracy -0.80 pts, CI [-1.23,-0.39] P+=0.000;
  decisive 6-18. WHOLE INTERVAL negative both metrics, both blockings,
  5 of 6 seasons negative -> candidate wrong_sign_resolved (needs a
  reliability check before the orchestrator terminally closes it; this
  run did not compute split-half reliability).
- week_gated_protection_flag_sum: line-move -0.0086 pts, CI
  [-0.0320,+0.0045] P+=0.343 (week CI [-0.0362,+0.0147] P+=0.248).
  Accuracy -0.27 pts, CI [-1.25,+0.53] P+=0.268. unresolved_below_power.

## Tried
- Confirmed FIT_FEATURES = (model_logit, composition_flag_sum,
  market_move_toward_home, market_move_available); base for this yardstick
  drops the two market-move terms per the CONTAMINATED finding already
  recorded in artifacts/line_move_yardstick/.
- Confirmed population from build_fit_population already carries `open_move`
  (close_home_spread - tue_open_home_spread) and `gameday`; reused directly,
  no recomputation.

## Next

- 2026-09-23 root: all five cells recorded (registry 7,011). Reddit recorded refuted_mechanism / wrong_sign_resolved for this Tuesday-base variant only; the 2011-2025 regrade (docs/lanes/extended-population-regrade.md) re-tests the term. Commands below are history. (for the root orchestrator)
Registry writes (`weak-signals record --help` confirmed: classification is
one of unresolved_below_power/refuted_mechanism/bounded_by_control;
closing-ground wrong_sign_resolved/no_split_half_reliability/
positive_control_bound is required only for a terminal classification):

```
nfl-ats weak-signals record --name diff_lineup_total_tuesday_line_move \
  --description "diff_lineup_total added to Tuesday-knowable base (model_logit+composition_flag_sum), LOSO 2020-2025, graded on close-minus-open line move toward the pick" \
  --source artifacts/tuesday_terms_line_move/20260923T205836Z/results.json \
  --effect -0.0100 --effect-units ats_points --classification unresolved_below_power \
  --league nfl --season-start 2020 --season-end 2025 \
  --interval-low -0.0233 --interval-high 0.0027 --probability-positive 0.0555 \
  --sample-games 1503 --sample-blocks 6 --category onfield \
  --plain-summary "Player-lineup-strength gap does not yet move the Tuesday-to-close line toward the pick more than the base model does."

nfl-ats weak-signals record --name diff_divergence_tuesday_line_move \
  --description "diff_divergence added to Tuesday-knowable base, LOSO 2020-2025, line-move yardstick" \
  --source artifacts/tuesday_terms_line_move/20260923T205836Z/results.json \
  --effect -0.0206 --effect-units ats_points --classification unresolved_below_power \
  --league nfl --season-start 2020 --season-end 2025 \
  --interval-low -0.0586 --interval-high 0.0113 --probability-positive 0.1325 \
  --sample-games 1503 --sample-blocks 6 --category onfield \
  --plain-summary "Lineup-rating divergence does not yet move the line toward the pick more than the base model does."

nfl-ats weak-signals record --name cfb_transfer_logit_tuesday_line_move \
  --description "College-transfer opener-error logit (opener_error_transfer_unit2.py) added to Tuesday-knowable base, LOSO 2020-2025, line-move yardstick" \
  --source artifacts/tuesday_terms_line_move/20260923T205836Z/results.json \
  --effect 0.0143 --effect-units ats_points --classification unresolved_below_power \
  --league nfl --season-start 2020 --season-end 2025 \
  --interval-low -0.0249 --interval-high 0.0714 --probability-positive 0.663 \
  --sample-games 1503 --sample-blocks 6 --category modeling \
  --plain-summary "The CFB-trained opener-error transfer term leans positive on the line-move yardstick but the interval still crosses zero."

nfl-ats weak-signals record --name week_gated_protection_flag_sum_tuesday_line_move \
  --description "Protection-mismatch flag zeroed after week 4, added to Tuesday-knowable base, LOSO 2020-2025, line-move yardstick" \
  --source artifacts/tuesday_terms_line_move/20260923T205836Z/results.json \
  --effect -0.0086 --effect-units ats_points --classification unresolved_below_power \
  --league nfl --season-start 2020 --season-end 2025 \
  --interval-low -0.0320 --interval-high 0.0045 --probability-positive 0.3425 \
  --sample-games 1503 --sample-blocks 6 --category onfield \
  --plain-summary "Gating the protection flag to weeks 1-4 does not yet move the line toward the pick more than the ungated base."
```

`reddit_home_comment_ratio_elevated` needs the orchestrator's own call, not a
mechanical one: whole 95% interval negative on BOTH the season block
[-0.0261,-0.0021] and week block [-0.0250,-0.0010], P+=0.000/0.018, 5 of 6
seasons negative, decisive games 4-14; accuracy companion also whole-negative
[-1.23,-0.39] P+=0.000, decisive 6-18. This meets the AGENTS.md bar for
`wrong_sign_resolved` on the line-move read (whole interval on the wrong
side), which would make `--classification refuted_mechanism
--closing-ground wrong_sign_resolved` admissible -- but this subagent did not
measure split-half reliability for the trait, and unit 6's own family
reasoning (pooled_signal_sixth_fit_new_family.py) already has three
independent positive-sign accuracy-based measurements for the same reddit
term. Before closing it, reconcile: this is the term's first test against a
different (line-move, Tuesday-knowable-base) yardstick, not a replication of
the same accuracy claim, so a resolved sign flip here does not by itself
contradict the earlier accuracy readings. Recommended: record it as
`unresolved_below_power` for now with `--classification-evidence` noting the
whole-interval-negative line-move read pending a reliability check, or run a
dedicated split-half check before invoking `refuted_mechanism` /
`wrong_sign_resolved`. Do not weaken the validator to force a close either
way.

## Open
Results and per-term deltas are in the final handback message and in
artifacts/tuesday_terms_line_move/<ts>/results.json.
