# SIM-21

## Goal
Drive length and scoring rate decomposition for trailing and leading offenses: plays per drive, seconds per play, and the uniform scoring-drive deficit (ROADMAP SIM-21).

## State
Added 2026-11-03 (backlog expansion 6), not started, nothing run. Mechanism: Mechanism: SIM-19 found the simulator trailing-offense scoring-drive rate 2 to 6 points low in every cell and drives 3 to 13 seconds short at deficit 14+. SIM-18 found sim drive length within 3 to 4 percent for leading drives. Decompose into plays per drive and seconds per play by deficit and clock to find which named mechanism is missing; the repair is a measured mechanism, not an offset.

## Tried
Novelty check only. Nearest: SIM-19 (gap measured, unit 1), SIM-18 (leading drive length), SIM-13 (snap load); grep 'drive length' found no decomposition.

## Next
First unit for one agent: Check whether the scoring-drive gap also exists for leading and tied offenses, then split the trailing gap into plays-per-drive and seconds-per-play shares with bootstrap intervals. Data: SIM-19 artifacts sim_vs_real_drives.json and instrument logs (local).

## Open
Instrument config of instr_s11 is unverified; confirm before attributing. Zero crossing closes nothing; one fitted probability selects the side.
