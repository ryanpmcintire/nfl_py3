# Hook review fd1d0278 (read-only)

## Goal
Find correctness bugs in the pool hook merge before the Week 9 lock and validation v7 capture.

## State
Reviewed pool_hook.py, pick_probability.py, pick_probability_fit.py, card_view.py, independent_validation.py diffs. Ran tuesday_market_lines for Week 8 (signs and half-point side correct). No blocking findings.

## Tried
Sign/direction algebra (x/(1-a), (x-a)/(1-a)), quote sign convention (home_spread_line = -home point), single served path (only card_view calls attach_pick_probability), fail-open paths.

## Next
Owner decides on three non-blocking items (fold leak in hook values, v7 capture uses live push table not enrolled one, hook can appear after the Tuesday card if no post-noon quote snapshot at lock time).

## Open
Same three items.

## Leak-free re-measure (agentHL)
Rule declared before running: for held-out season H, every training row's hook value uses push mass from seasons < the row's season AND != H; H rows keep prior-season-only hook values. Coefficient: ridge 1e-3 (FIT_RIDGE), Newton, no intercept, offset = fold stage-1 logit. Harness first reproduces production fold coefficients exactly.
Result: harness reproduced production exactly (stage-1 logits, per-fold coefficients, probabilities max diff 0). Leak-free fold coefficients 2020-25: .847 .743 .788 .785 .814 .779 (contaminated .807 .874 .793 .768 .812 .779). Whole-number stacked rows (1322): stage1 LL .68343 Brier .24520 726-596; contaminated .68223/.24459/729-593; leak-free .68220/.24457/727-595 (183 changed picks). Leak-free LL gain .00123 [.00032,.00224] P+ .9966; Brier gain .00063 [.00017,.00113] P+ .998; 107 season-week blocks. Leak-free minus contaminated LL .00003 [-.00003,.00008]. Evidence holds. Recorded weak-signals pool_hook_two_stage_log_loss_leak_free. Script: scratchpad agentHL/leakfree.py.
