# Positive-control evaluator power

## Goal
Measure evaluator power using historically pinned inputs and production bootstrap precision.

## State
**Measured:** the entire original 16-cell curve completed at 400/2,000/20,000
draws, 200 fixed replicates per cell. Strongest-control detection at 20,000 draws
is DPI 34.5% [28.26%, 41.32%] and holding 75.0% [68.57%, 80.49%]. Null rates
are 0.5% [0.09%, 2.78%]. These are Wilson intervals, not natural-effect bounds.
All coefficients are in [the complete curve report](../positive_control_precision_curve_20260928.md).

## Tried
The historical reference remains `positive_control_power/20260927T042645Z`.
Four hash-verified cells were reused and 12 computed, with fixed outcomes, LOSO
fits, optimizer, seed and strict bootstrap prefixes. All 96 diagnostic summaries,
4,809,600 predictions and 9,600 replicate-prefix rows are preserved in
`artifacts/positive_control_precision_curve/historical_v1_checkpoint/`.
The original declaration is frozen there as `predeclaration.md`. A post-launch
summary-sorting edit was diagnosed; exact manifest-matching runner bytes were
recovered and archived with the diff. No numerical code or manifest was rewritten.
Root's independent verifier passed all 16 exact 400-draw reproductions, prediction
arithmetic, all 96 summaries and unchanged input hashes.

## Next
Any control matched to a natural signal requires a separate declaration with its
direction and effect size fixed before results. Preserve this completed family;
do not interpolate, choose a coefficient, or repeat it to seek a better result.

## Open
No natural effect is bounded or closed by this full precision curve. Zero
crossing closes nothing; power must be demonstrated at the relevant effect size.
One fitted probability continues to select every served side.

The measured natural DPI effect is -0.133 accuracy points, smaller in magnitude
than and opposite in direction to the smallest non-null DPI control (+0.200599).
The measured holding effect is -0.599, between but opposite in direction to the
+0.290419 and +0.943114 controls. The original positive-direction curve cannot
bound either natural signal, and neither interpolation nor absolute-value matching
is admissible. Any matched negative-effect control requires a separate prospective
family.
