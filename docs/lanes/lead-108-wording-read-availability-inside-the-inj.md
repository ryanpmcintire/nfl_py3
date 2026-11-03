# LEAD-108

## Goal
Wording-read availability inside the injury scenario mixture: the LEAD-106 per-player play probability replaces the ordinal ladder as the mixture weight (ROADMAP LEAD-108).

## State
Added 2026-11-03 (backlog expansion 7), not started, nothing run. Mechanism: LEAD-106 measured the wording read beating the designation ladder by .065 log loss (P+ .998; versus the practice ladder .036, P+ .96). PER-10 built the scenario mixture on the ordinal ladder and recorded unresolved. Fitted term: expected-lineup margin shift, sum over listed starters of (1 - P_play) times the PER-09 rating, difference home minus away, one term in the calibrated probability. Data: LEAD-106 stored reads and forward captures, PER-09 ratings (local).

## Tried
Novelty check only. Nearest: PER-10 (ordinal mixture), LEAD-106 (availability read), PER-09 (ratings); grep 'mixture' with 'wording' found none. Differs: swaps only the weight source, keeps the closed PER-10 kernel.

## Next
First unit for one agent: Coverage first: count listed starters per game with a verified read in weeks 2-8, then rebuild the PER-10 mixture with read probabilities and compare opener-graded log loss to the ordinal mixture, leave-one-season-out.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
