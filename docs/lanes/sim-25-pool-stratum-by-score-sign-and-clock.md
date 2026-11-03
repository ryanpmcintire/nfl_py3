# SIM-25

## Goal
Resampler candidate pool restricted by score sign and clock band: does drive length and scoring match when only like-situation drives are drawn (ROADMAP SIM-25).

## State
Added 2026-11-03 (backlog expansion 8), not started, nothing run. Mechanism: SIM-21 and SIM-23 left most of the leading and trailing scoring gap after the truncation fix. The pool is drawn by field position and tilted by drive outcome, but real leading drives are slower and the pool may not know the score sign. Measured mechanism: eligibility stratum by score sign and quarter, with strata sizes checked so none is thinner than the matched pool.

## Tried
Novelty check only. Nearest: SIM-21, SIM-23 (truncation, gap on non-truncated drives), SIM-22, SIM-20. Differs: changes who is eligible for the draw, not the tilt, truncation or start field.

## Next
First unit for one agent: Count real drives per score-sign by clock stratum, then compare the real seconds and scoring rate against the sim draw when the pool is restricted to the stratum, flag-off hash first, on 2025 only, reporting the share of the remaining gap removed with a bootstrap interval.

## Open
Rules: Measured mechanisms only; no tuned constant, offset or boost; outputs are checks; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; flag-off hash must equal HEAD before any flag-on run. Look count and family declared before any sign is seen.
