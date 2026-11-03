# POOL-08

## Goal
Half-point hook awareness: does the field over- or under-pick the side the pool's half point helps (ROADMAP POOL-08).

## State
Added 2026-11-03 (backlog expansion 6), not started, nothing run. Mechanism: Mechanism: If entrants ignore which side the hook helps, the hook side is under-owned relative to its true cover chance and a pool-level edge exists; if they chase it, the edge is gone. Enters the POOL-01 standing-aware leverage as a fitted field-share covariate and the pool simulator, not as a side rule.

## Tried
Novelty check only. Nearest: POOL-05 (slot and team ownership add nothing), MOD-26 (hook fitted term), UI-31 (hook chip); grep 'hook' in POOL rows found POOL-01 only.

## Next
First unit for one agent: Join captured field shares to games with a pool half point versus none; compare share of the hook-helped side to the unhooked baseline with a permutation null at game level. Data: Captured field picks, MOD-26 pool lines (local).

## Open
Field picks known only for captured weeks; count games before testing. Zero crossing closes nothing; one fitted probability selects the side.
