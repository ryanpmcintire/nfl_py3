# SIM-28

## Goal
Timeouts remaining as a draw dimension for end-of-half drives (ROADMAP SIM-28).

## State
Added 2026-11-03 (backlog expansion 9), not started, nothing run. Mechanism: The final minutes of each half depend on timeouts: a trailing team with timeouts stops the clock and gets extra snaps, one without runs the clock. If the draw ignores timeouts, end-of-half drive length and scoring mix real states with different clock control. Measured mechanism: end-of-half drive length and points by timeouts remaining for the offense. Fitted term: Simulator state dimension measured from real drives, with the timeout count tracked by the engine.

## Tried
Novelty check only. Nearest: SIM-09 (urgency and stakes), SIM-23 (clock-truncated drives), SIM-19. Differs: Adds a measured clock-control state that none of SIM-09, SIM-19 or SIM-23 conditions on.

## Next
First unit for one agent: Tabulate real drives starting inside the last 3 minutes of each half by offense timeouts remaining (0 to 3), score sign and field position band, report plays, seconds and points per drive with game-cluster intervals, and check cell counts before proposing any kernel dimension.

## Open
Rules: measured mechanism only, never a tuned constant or offset; aggregates are checks, not targets; each stage compared with real play-by-play with game-cluster intervals; flag-off hash must equal HEAD before any flag-on run; unresolved results are reported as measured. Look count and family declared before any sign is seen. Data: Play-by-play timeouts remaining and drive table (local).
