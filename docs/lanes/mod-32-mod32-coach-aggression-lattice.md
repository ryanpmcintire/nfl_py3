# MOD-32 Coach aggression index as a variance and tail term in the line-conditional margin distribution

## Goal
Coach aggression index as a variance and tail term in the line-conditional margin distribution ROADMAP row MOD-32.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: SIM-14 found aggression holds within a coach-team spell. An aggressive coach (fourth-down and two-point go rate over expected) widens the margin distribution and moves mass across key numbers, which changes the cover probability at a given line without changing who is favoured. The served margin read uses one pooled shape for every team. Data: nflverse pbp fourth-down and two-point decisions 2009-2025, coach-team spells from the PER-07 staff state; discrete margin lattice from MOD-17 and LEAD-66.

## Tried
Novelty check only: Nearest: SIM-14 (latent for the simulator, not a served term), coach_fade_overlay.md and coach_speak_base_rates.md (coach fade by speech and tenure, no behavioural index), big_spread_lattice.md (spread-conditional shape pooled over teams). Differs: behavioural index enters the discrete lattice as a shape term.

## Next
First unit: Build the neutral-state expected go rate leave-one-season-out, form the shrunk per-spell index from prior games only, measure split-half reliability against a spell-shuffle null, then test whether the index scales the lattice variance in a held-out likelihood comparison before any cover grade.
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Look count and family to be declared before any sign is seen. Nothing here closes on zero crossing; record unresolved results with nfl-ats weak-signals record.
