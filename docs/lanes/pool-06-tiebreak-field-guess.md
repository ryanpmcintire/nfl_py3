# POOL-06

## Goal
Tiebreaker field distribution: choose the last-game total guess to maximise the chance of winning ties on wins (ROADMAP POOL-06).

## State
Added 2026-11-03 (backlog expansion 5), unmeasured, nothing run. Mechanism: Ties on wins are decided by the guess closest to the last game's total; the field's guesses cluster, so a guess just off the cluster wins more ties. Fitted term: tie-win probability from the field guess distribution and the discrete total distribution.

## Tried
Novelty check only. Nearest: POL-12 (tiebreaker score guess), POOL-04, LEAD-105 (language read of the total); grep found no field-guess model. Differs: The field's guess distribution, not the model's centre.

## Next
First unit: Check whether captured field files carry tiebreak guesses; if so, estimate cluster density and score the tie-win chance of the served guess versus alternatives over Weeks 1-8.

## Open
Data: Pool field files (captured) and the local totals distribution (free). Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
