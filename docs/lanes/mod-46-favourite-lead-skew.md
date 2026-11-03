# MOD-46

## Goal
Favourite lead-protection skew: how a favourite's OC trait changes the chance of winning by 3 versus 7 versus 14 (ROADMAP MOD-46).

## State
Added 2026-11-03 (backlog expansion 6), not started, nothing run. Mechanism: Mechanism: A favourite that protects leads burns clock and converts big leads into 7 to 10 point wins rather than 17, moving mass between key numbers without changing the mean. Fitted term: trait times spread bin in a multinomial over the discrete margin lattice conditional on the line, entering the cover and push probability served from the discrete read.

## Tried
Novelty check only. Nearest: MOD-18 (lattice), SIM-18 (trait), MOD-17 (lattice ledger); grep 'skew' found only MOD-18 lattice text.

## Next
First unit for one agent: Tabulate realised margin lattice mass at 3, 7, 10, 14, 17 for favourites by trait tercile within spread bins, with a permutation null and season-block bootstrap, then LOSO add-one on the discrete cover read. Data: Local margins 2009-2025 and SIM-18 coordinator traits (free).

## Open
Terciles are a look count of at least 15 cells; lopsided bins need an exact null. Zero crossing closes nothing; one fitted probability selects the side.
