# SIM-30

## Goal
Simulated conditional margin distribution as a challenger to the discrete empirical read, graded at the opener on the played card (ROADMAP SIM-30).

## State
Added 2026-11-03 (backlog expansion 9), not started, nothing run. Mechanism: The served cover and push probability is the discrete margin distribution conditional on the line; a simulation can supply the distribution with its own key-number mass. Measured mechanism: the simulator's margin mass at 3, 7, 10, 14 and 17 compared with the empirical lattice by spread band, now that the truncation double count is fixed behind a flag. Fitted term: Challenger only: simulated cover probability against the discrete read on the same games; it can be served only if it beats the discrete read on the opener grade through the played card.

## Tried
Novelty check only. Nearest: SIM-11, SIM-21, MOD-17 (lattice ledger), MOD-32. Differs: First comparison of simulator margin mass with the served discrete read; MOD-17 and MOD-32 compare lattices without the simulator.

## Next
First unit for one agent: Compare simulated and empirical margin mass at the five key numbers by spread band with game-cluster intervals, then grade the simulated cover probability against the discrete read by log loss on games where simulation exists.

## Open
Rules: measured mechanism only, never a tuned constant or offset; aggregates are checks, not targets; each stage compared with real play-by-play with game-cluster intervals; flag-off hash must equal HEAD before any flag-on run; unresolved results are reported as measured. Look count and family declared before any sign is seen. Data: Simulator margin output after SIM-21 (flag on), served discrete read (local).
