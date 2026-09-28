# SIM-08 simulator rebuild

## Goal
Repair transition and terminal-state behavior, then diagnose late-game bias
without fitting weights or changing served probabilities from these diagnostics.

## State
**Measured:** September 28 repaired quarter boundaries, zero elapsed transitions,
timeouts, and historical overtime termination/conversion rules. The final fixed
replay has 10,002 games, zero caps, but only 2/5 key-number bands pass.
Log-loss improvement versus the training histogram is -0.000656
[-0.013844, +0.014832], probability_positive 0.370370. Versus the preceding
engine, improvement is -0.001595 [-0.002528, -0.000795], probability_positive 0
in 27 conditional season draws; Monte Carlo and prior selection are omitted.
All four effects remain registered unresolved; no serving change.
[Complete 147-look result, including superseded arm](../simulator_overtime_repair_20260928.md).

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
Historical rule defects are repaired; broader predictive accuracy is unresolved.
Preserve `artifacts/sim08_overtime_repair/20260928T171712Z/` with all three arms,
source/input hashes, 30,006 simulations and actual-game losses. Any new mechanism
or 2025+ rules extension needs a declaration before another endpoint is scored.
Do not patch key-number masses, fit on these diagnostics, or promote this engine.

## Open
Finite-draw probability_positive of 0 or 1 is not impossibility or certainty.
The deficient margin-3 mass and rollout calibration remain unresolved. Support
failure closes no signal; these diagnostics authorize no weight or serving change.
