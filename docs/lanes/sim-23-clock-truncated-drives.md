# SIM-23

## Goal
Clock-truncated drives: end-of-half and end-of-game drives that stop by expiry or kneel, separated from drive length and scoring rate (ROADMAP SIM-23).

## State
Added 2026-11-03 (backlog expansion 7), not started, nothing run. Mechanism: SIM-21 found leading drives 3.1 s short and scoring low; real leading drives end by kneel and expiry, which cut length and scoring. Measured mechanism: a truncation share by clock and sign of score difference, drawn from the real pool. Data: play-by-play drives and sim trace (local).

## Tried
Novelty check only. Nearest: mod25e-e103 (late clock), mod25e-e104 (half end), mod25e-e107 (q4 drive), SIM-18, SIM-21. Differs: separates truncation from the SIM-21 gap in the resampler.

## Next
First unit for one agent: Classify real drives as truncated and compute score rate and seconds by sign of score difference with and without them, sim versus real.

## Open
Rules: Measured mechanisms only; no tuned constant, offset or boost; outputs are checks; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; flag-off hash must equal HEAD before any flag-on run. Look count and family declared before any sign is seen.
