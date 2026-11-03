# SIM-29

## Goal
Simulated lineup availability from the injury wording read: draw each listed starter's play status from his read probability at game start (ROADMAP SIM-29).

## State
Added 2026-11-03 (backlog expansion 9), not started, nothing run. Mechanism: SIM-12 and SIM-16 model in-game exits and pregame attrition from base rates. LEAD-106 measured the wording read as a better play probability than the ladder. Measured mechanism: draw each listed starter's availability from his read probability, apply the measured replacement cost, and check the simulated margin against the real margin by availability bucket. Fitted term: Simulator input from the read probability; the simulated margin shift enters the calibrated probability only as a challenger term.

## Tried
Novelty check only. Nearest: LEAD-108 (read in the PER-10 mixture), SIM-12, SIM-16. Differs: Feeds the read into the simulator's lineup rather than the margin mixture.

## Next
First unit for one agent: For 2026 games with a stored read, run a small fixed set of simulated games per listed-starter draw, compare the simulated margin shift with the realised margin residual in games where a starter was out (replacement cost check), and report games per bucket before any score.

## Open
Rules: measured mechanism only, never a tuned constant or offset; aggregates are checks, not targets; each stage compared with real play-by-play with game-cluster intervals; flag-off hash must equal HEAD before any flag-on run; unresolved results are reported as measured. Look count and family declared before any sign is seen. Data: LEAD-106 reads, snap counts, SIM-12 replacement costs (local).
