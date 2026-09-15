# leader-median-model-confidence

## Goal

Redo, from scratch, the owner-rejected leader-median override-distance
analysis: check whether the served late-week leader-median follow rule
should be conditioned on the underlying model's confidence in the pick it
overrides. Done when the identical-61-49 objection is independently
addressed, a confidence gate is evaluated honestly out-of-sample
(leave-one-season-out, nested with the move-size threshold itself), and the
old outputs are replaced.

## State

Done, 2026-09-14. Shipped: `docs/leader_median_model_confidence.md`,
`scripts/leader_median_model_confidence_eval.py`,
`artifacts/leader_median_model_confidence/20260914T202723Z/` (summary.json,
flips.csv, record_commands.json). Registry family
`leader_median_model_confidence_v1`, 19 cells, all `unresolved_below_power`.
Old family `leader_median_override_distance_v1`'s three cells invalidated,
superseded-by pointing at the new cells. Old
`docs/leader_median_override_distance.md`,
`scripts/leader_median_override_distance_eval.py` and
`artifacts/leader_median_override_distance/` deleted.

## Tried

- Reproduced the headline full-rule-vs-card effect independently: +3.0038
  accuracy points, matches the stored `docs/sharp_weighted_follow.md` S1
  number to the point estimate; interval differs only by bootstrap seed.
- Proved the identical 61-49 is a mathematical identity: this session's
  `confidence - 0.5` and the prior session's "override distance" magnitude
  are the same number on every flip (max float diff 5.6e-17), so a median
  split on either produces the same two 110-game halves; independently
  reproduced 49-61 card / 61-49 follow in both halves.
- Measured near-zero correlation (-0.022) and flat logistic slopes between
  confidence and follow-correctness, with intervals on both sides of zero --
  no measured slope in either direction.
- Confidence-band table: only the firmest quarter of flips (>0.58, 42 flips)
  reads negative (-0.25 pts), directionally consistent with the owner's
  hypothesis but unresolved (interval crosses zero).
- Single-dimension leave-one-season-out gate search: same cut (~90th
  percentile) chosen in every fold, zero in-sample/out-of-sample gap, but
  small gain over the ungated rule (+0.25 pts, P+ 0.699, unresolved).
- Nested leave-one-season-out search over BOTH the move-size threshold
  (flat 0.5 / flat 1.0 / the served 1.0-with-0.5-big-spread mechanism rule)
  and the confidence gate: honest out-of-sample effect of the threshold
  alone is +2.128 pts [-0.505, +4.804] (positive, not at or below the
  card); adding a confidence gate makes the honest number worse (+1.877
  pts) and grows the in-sample/out-of-sample gap (1.377 vs 0.876 pts).
  **No confidence gate recommended.**

## Next

- The number quoted anywhere for the served leader-median rule is now the
  out-of-sample +2.13 [-0.51, +4.80], never the in-sample +3.00; sweep
  `docs/sharp_weighted_follow.md`, the board's season-ops note and the
  assistant glossary for the old headline.
- Owner decision, still open: the served rule's thresholds were chosen on
  the same games it is scored on. Keeping it served is a serving preference
  on a +2.13 out-of-sample read, not a finding.
- Done 2026-09-14 in the same session: the handle-follow constant removed
  from `pick_refresh.py`; ROADMAP handoff item rewritten.

## Open

- None.
