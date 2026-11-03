# LEAD-110

## Goal
Market absorption of the wording read: does the Tuesday-to-deadline line move already load on the read availability difference (ROADMAP LEAD-110).

## State
Added 2026-11-03 (backlog expansion 7), not started, nothing run. Mechanism: If the market prices the wording, the read adds nothing beyond the move; if it lags, the read leads the move. Fitted term: read availability difference in a model of the move, then in the cover probability with move as a separate term. Data: LEAD-106 reads, Tuesday and deadline lines (local).

## Tried
Novelty check only. Nearest: LEAD-101 (news delta), LEAD-104 (view revision), MKT-25. Differs: tests absorption of one frozen read, not a new read.

## Next
First unit for one agent: Regress move on the read difference with designation-ladder and intercept-only controls, game-level bootstrap and a within-week permutation null, weeks 2-8.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
