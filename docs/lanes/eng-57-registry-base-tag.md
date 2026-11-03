# ENG-57

## Goal
Weak-signal registry base tag: mark which recorded cells were graded on the raw model probability (ROADMAP ENG-57).

## State
Added 2026-11-03 (backlog expansion 8), not started, nothing run. Mechanism: Cells in registry/weak_signals do not say which base they were graded on, so a pooled summary mixes raw-base and served-base results. The error is a missing field, not a number. Fitted term: none; the tag lets nfl-ats weak-signals pool separate bases before any effect is pooled.

## Tried
Novelty check only. Nearest: ENG-51 (forward guard on new harnesses), ENG-54, weak-signals pool command. Differs: labels the past recorded cells and splits the pooled summary; ENG-51 only guards new harnesses.

## Next
First unit for one agent: Read the scripts behind this week's recorded families, tag each family raw, served or market-only in a side table, and report the pooled accuracy-point summary split by base without changing any stored cell or validator.

## Open
Rules: Never weaken the record validator; no stored value edited; the tag is a side table. Look count and family declared before any sign is seen.
