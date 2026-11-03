# ENG-54

## Goal
Outcome-label default audit: every scorer that joins players or games to outcomes and labels a miss as a negative (ROADMAP ENG-54).

## State
Added 2026-11-03 (backlog expansion 7), not started, nothing run. Mechanism: LEAD-106 labelled 33.5% of rows unmatched-as-not-played until review; the same pattern in any harness biases a grade. Fitted term: none; integrity check. Data: scripts and registry harnesses (local).

## Tried
Novelty check only. Nearest: ENG-51 (opener-input guard), ENG-52 (timestamp audit), LEAD-64 (capture integrity); grep 'name match' and 'label default' found none. Differs: join-level label defaults.

## Next
First unit for one agent: List every harness joining by name or fillna to a negative label, run each, report unmatched rate and direction of bias, and re-run any result that moves.

## Open
Rules: Audit only; findings reported as measured with counts; no tuned constant; any affected result is re-run, not patched. Look count and family declared before any sign is seen.
