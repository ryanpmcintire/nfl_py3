# SIM-24

## Goal
Next-drive start field by previous drive ending and score sign (ROADMAP SIM-24).

## State
Added 2026-11-03 (backlog expansion 7), not started, nothing run. Mechanism: SIM-21 start-field counterfactual: simulated trailing drives start 0.71 yd nearer goal and leading 0.69 yd farther than real, which explains 18% of the leading score gap. Measured mechanism: start yard line conditioned on how the previous drive ended and who scored. Data: drive pool (local).

## Tried
Novelty check only. Nearest: mod25e-e108 (drive start, generator), SIM-21. Differs: inside the SIM-10 resampler kernel.

## Next
First unit for one agent: Tabulate real start yards by previous ending, kicking team and sign of score difference and compare to the kernel candidate pool's start distribution.

## Open
Rules: Measured mechanisms only; no tuned constant, offset or boost; outputs are checks; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; flag-off hash must equal HEAD before any flag-on run. Look count and family declared before any sign is seen.
