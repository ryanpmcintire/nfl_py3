# SIM-22

## Goal
Plays-per-drive logging in the resampler: draw plays with seconds from matched real drives so the split is observed (ROADMAP SIM-22).

## State
Added 2026-11-03 (backlog expansion 7), not started, nothing run. Mechanism: SIM-21 could only infer plays from seconds (71% of the trailing gap by a log-log slope); the resampler logs no plays. Measured mechanism: carry each matched real drive's play count and seconds as a joint draw. Data: SIM-19 and SIM-21 artifacts, drive pool (local).

## Tried
Novelty check only. Nearest: SIM-21 (inferred split), SIM-10 (resampler), SIM-18. Differs: observes rather than infers.

## Next
First unit for one agent: Add logging of matched play count with flag-off hash equal to HEAD, run a small seed set, and compare plays per drive to real by sign of score difference.

## Open
Rules: Measured mechanisms only; no tuned constant, offset or boost; outputs are checks; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; flag-off hash must equal HEAD before any flag-on run. Look count and family declared before any sign is seen.
