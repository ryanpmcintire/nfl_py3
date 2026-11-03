# XLG-11 College-centered ridge prior

## Goal
Test the college coefficients as a prior mean for the NFL margin ridge. ROADMAP row XLG-11.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: A ridge penalises distance from zero; with 1,500 graded NFL games and 8,000 college games on shared inputs, the college coefficients are an informative prior mean. Fit NFL on the residual after offsetting by the college prediction; the shrink strength is chosen leave-one-season-out. Data: cfb_game_features.parquet (common 14 inputs) and the NFL game features, local.

## Tried
Novelty check only (grep of ROADMAP, docs/lanes incl. done, docs/*.md, registry/weak_signals). Nearest prior: XLG-09 unit 5 (league-interaction ridge on the move mechanism, closed wrong sign), mod24_pooled_levers (training-row levers), mod23-unit11 (names CFB pretraining as covered by lead69). Different: a coefficient prior center for the margin ridge on the shared inputs, not a pooled design.

## Next
Fit the CFB common-14 ridge walk-forward and build the offset arm beside the zero-centered arm. First unit: First unit: on the 14 shared inputs, walk-forward 2020-2025 compare NFL ridge toward zero versus toward the college coefficients, graded at the opener with log loss and accuracy, paired block bootstrap.
Research rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Reversal risk seen in mod24 (CFB rewards more shrinkage); LOSO alpha must be reported per fold.
