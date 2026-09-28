# SIM-08 simulator rebuild

## Goal
Repair transition and terminal-state behavior, then diagnose late-game bias
without fitting weights or changing served probabilities from these diagnostics.

## State
**Measured:** September 28 repaired quarter boundaries, zero elapsed transitions,
and two overtime timeouts. The before/after replay completed 10,002 games per arm,
zero caps; the before margins exactly reproduce the prior checkpoint. Repaired
log-loss improvement over the naive histogram is +0.000939 [-0.011431, +0.015928],
probability_positive 0.5926, conditional on only three validation seasons.
Three of five key-number bands pass; margins 3 and 17 remain unresolved.
Both log-loss effects are registered `unresolved_below_power`; no serving change.
[Full results and all diagnostic looks](../backlog_research_20260928.md).

## Tried
- **Measured:** conversion repair reconciles 768/768 final margins. Full repair
  history: `done/sim08-transition-repair-history.md`.
- **Measured:** 12 neighbor looks and four chronological comparisons diagnose
  conditional scoring and clock behavior. Chronological one-score elapsed delta
  is -2.11 seconds [-2.88, -1.43], probability_positive 0 in 2,000 draws;
  scoring frequency is +0.17 points [-0.43, +0.77], probability_positive 0.706.
- **Measured:** three validation source-age slopes and six training rolling-era
  slopes bring that diagnostic family to 25 looks. Excluding 2009, training slopes
  are +0.830 [0.282, 1.391], +0.741 [0.118, 1.315], and +0.802 [0.334, 1.286]
  seconds/year, probability_positive 0.999, 0.989, and 0.9995. Cross-role game
  identity limits those bootstrap intervals; these are not causal rollout effects.
- **Measured:** 144 instrumented rollouts preserve all same-seed outputs, with
  zero caps. Exact matching yields 2 selected scoring transitions; the separately
  declared timeout-omission sensitivity yields 9 in 9 rollout clusters from
  6 scheduled games. Both fail declared 30-transition/10-cluster support gates.
  Neither computes an effect. Two rollout endpoint looks are recorded.
- **Measured:** the expanded 96-game/eight-seed sample yields 67 matched scoring
  transitions in 66 rollouts from 40 scheduled games. Its fixed-selection paired
  bootstrap estimates selected-minus-actual elapsed discrepancy -0.751 seconds
  [-2.728, +1.173], probability_positive 0.2175 (2,000 draws; 55 actual games).
  This fourth rollout endpoint look is recorded as `unresolved_below_power`.
- Full methods, artifacts, uncertainty and limitations are retained in
  `done/sim08-late-game-diagnostic-history.md`.

## Next
Continue the named endgame mechanisms before another parameter or endpoint look.
The completed predeclaration fixed seed 20260925, 3,334 games per 2015–2017 season,
training 2009–2014, copied features, both source engines, hashes, and all rows.
`artifacts/sim08_terminal_repair/20260928T161258Z/` retains the entire diagnostic
family and added transparent season uncertainty. No endpoint selected a weight.
Retain the earlier fourth-look bundle and its unresolved uncertainty separately.

## Open
Finite-draw probability_positive of 0 or 1 is not impossibility or certainty.
The deficient margin-3 mass and rollout calibration remain unresolved. Support
failure closes no signal; these diagnostics authorize no weight or serving change.
