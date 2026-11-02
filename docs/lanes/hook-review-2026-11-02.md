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
