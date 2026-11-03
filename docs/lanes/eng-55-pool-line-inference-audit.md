# ENG-55

## Goal
Pool-line inference audit: is the pool line always the Tuesday market plus or minus half a point (ROADMAP ENG-55).

## State
Added 2026-11-03 (backlog expansion 7), not started, nothing run. Mechanism: POOL-08 inferred the hook from market plus or minus 0.5 on 121 games; a wrongly inferred line misclassifies hook games. Fitted term: none; integrity check on the POOL-08 population. Data: data/pool_observables captures and Tuesday market (local).

## Tried
Novelty check only. Nearest: POOL-08, MOD-26, ENG-52. Differs: audits the inference.

## Next
First unit for one agent: Join captured pool lines to the Tuesday market for all captured games, count equal, plus half, minus half and other, and re-run POOL-08 on any changed hook set.

## Open
Rules: Audit only; findings reported as measured with counts; no tuned constant; any affected result is re-run, not patched. Look count and family declared before any sign is seen.
