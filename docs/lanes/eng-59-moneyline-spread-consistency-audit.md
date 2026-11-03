# ENG-59

## Goal
Moneyline and spread consistency: does the favourite implied by the moneyline equal the favourite implied by the spread sign in every graded opener row (ROADMAP ENG-59).

## State
Added 2026-11-03 (backlog expansion 9), not started, nothing run. Mechanism: A sign error or a stale spread in one source would flip a graded side without any model change. The moneyline gives an independent favourite for most games. Fitted term: None: a row-level consistency audit of the opener archive.

## Tried
Novelty check only. Nearest: ENG-51, ENG-55, ENG-56, LEAD-71. Differs: Cross-checks the spread sign against a second price instead of the final score.

## Next
First unit for one agent: For every graded game with both a spread and a moneyline, flag rows where the favourite disagrees or the implied probability is non-monotone with the spread across a week, and list each flagged row with its sources.

## Open
Rules: audit only, no research verdict; report counts, rows and the exact join; any graded result that changes is re-run and re-recorded through the registry validator, never weakened. Look count and family declared before any sign is seen. Data: Opener archive spread and moneyline columns; nflverse schedule lines (local).
