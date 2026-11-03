# MKT-27

## Goal
Total-line level as a margin-width input: the same spread at a total of 37 versus 53 is a different margin distribution (ROADMAP MKT-27).

## State
Added 2026-11-03 (backlog expansion 6), not started, nothing run. Mechanism: Mechanism: High-total games have more scoring events, so the discrete margin has less mass at 3 and more at 10 to 14; low-total games pile onto 3 and 6. The discrete cover and push probability conditional on the line should also condition on the total. Fitted term: total bin shifts the lattice weights in the served discrete read.

## Tried
Novelty check only. Nearest: MOD-18 (lattice conditional on line only), MOD-17, MKT-23 (total moves); grep 'total' found totals used for tiebreaks and moves, not as lattice width.

## Next
First unit for one agent: Tabulate lattice mass by total tercile within spread bins and run LOSO add-one of the total-conditioned discrete read against the unconditioned one on opener cover log loss. Data: Local lines archive with totals, margins 2009-2025 (free).

## Open
The existing lattice may already include the total; read the builder first. Zero crossing closes nothing; one fitted probability selects the side.
