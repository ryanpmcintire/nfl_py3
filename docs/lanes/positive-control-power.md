# Positive-control evaluator power

## Goal
Measure evaluator power using historically pinned inputs and production bootstrap precision.

## State
**Measured:** the predeclared four-cell precision replay completed; all historical
400-draw cells reproduced exactly. At coefficient 1.80, DPI detection with 20,000
draws is 34.5% [28.26%, 41.32%], versus 38.5% with 400 draws. Holding is 75.0%
[68.57%, 80.49%], versus 78.0%. These are Wilson intervals over 200 replicates.
Paired classification disagreement is 8% for DPI and 6% for holding. Null detection
remains 1/200 in both controls. This resolves numerical precision for these cells only.

## Tried
The historical reference is `positive_control_power/20260927T042645Z`, not the
earlier current-model analogue. Fixed outcomes, LOSO fits, and bootstrap seeds;
400/2,000 draws are strict prefixes of 20,000. All 24 diagnostic looks, 1,202,400
prediction rows, 2,400 replicate rows, and optimizer evidence are preserved in
`artifacts/positive_control_precision_replay/20260928T160726Z/`.
Independent arithmetic/hash checks passed. Full results and uncertainty are in
[the checkpoint report](../backlog_research_20260928.md); previous methods and
predeclaration are retained in [history](done/positive-control-power-before-20260928-checkpoint.md).

## Next
Use the measured 20,000-draw values for these controls. Predeclare matched-effect
cells before extending the curve or attempting a natural-signal power bound.

## Open
No natural effect is bounded or closed by this sparse precision replay. Zero
crossing closes nothing; power must be demonstrated at the relevant effect size.
One fitted probability continues to select every served side.
