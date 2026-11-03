# ENG-53

## Goal
Trait reliability table: one weekly table of split-half and season-to-season reliability for every keyed trait, with the attenuation-corrected shrink factor (ROADMAP ENG-53).

## State
Added 2026-11-03 (backlog expansion 6), not started, nothing run. Mechanism: Mechanism: SIM-17, SIM-18, SIM-19 and MOD-43 each computed reliability separately with different keys and estimators. A single table sets each trait's shrink toward zero from its own measured reliability, so fitted terms use the same measured shrinkage rather than ad hoc alphas. No model term; infrastructure feeding the ridge prior variance.

## Tried
Novelty check only. Nearest: ENG-50 (online stacker), MOD-20 (pooled signals), XLG-15 (reliability transfer); grep 'reliability table' found none.

## Next
First unit for one agent: Collect the reliability outputs already on disk into one table with a shared estimator (odd-even, season t to t+1, game-cluster bootstrap) and report disagreements between scripts' estimates. Data: Existing sim17, sim18, sim19, mod43 artifacts (local).

## Open
Spell inflation noted in SIM-19 (shared baseline) must be handled by one estimator. Zero crossing closes nothing; one fitted probability selects the side.
