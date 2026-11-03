# LEAD-105

## Goal
Language read of the tiebreak game total: a frozen forward read of the week's last game total graded against the model and market totals (ROADMAP LEAD-105).

## State
Added 2026-11-03 (backlog expansion 4), unmeasured, nothing run. Mechanism: The tiebreak decides ties at the top and the pool objective is first place. The last game total is pinned by news (weather, QB status, pace) that arrives through Sunday; a read made at the deadline may add to the model total guess, as one fitted term in the tiebreaker, never a rule. Contamination: forward games only; no historical backtest is admissible.

## Tried
Novelty check only. Nearest: POL-12 (tiebreaker guess), LEAD-80 (tie-conditional scores), LEAD-88 (total-news response), LEAD-94 (last-game training totals). Grep for an LLM or language read of totals or tiebreaks found nothing in ROADMAP.md, lanes, docs or registry. Differs: new information class on the tiebreak page.

## Next
First unit: Add a separate frozen prompt that returns a total and score for the last game of Week 10 only; write it before kickoff; after the game compare absolute error to the served guess and to the market total; accumulate to Week 18.

## Open
Changes: Changes the page the pool tiebreak is entered from. Data: LEAD-99 packet fields for the Monday game; served tiebreaker.json; closing total. Frozen prompt for the final game total; changes tiebreaker.json only if a fitted term. Zero crossing closes nothing; record unresolved results with nfl-ats weak-signals record and a closing ground. Look count and family declared before any sign is seen.
