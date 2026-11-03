# MOD-45

## Goal
Tempo pair as a spread input: coordinator-keyed pace and lead-protection traits set expected plays, which scale the line-conditional margin distribution (ROADMAP MOD-45).

## State
Added 2026-11-03 (backlog expansion 6), not started, nothing run. Mechanism: Mechanism: SIM-18 found coordinator lead-protection and seconds-per-play traits reliable (OC split-half .69) but with no side term. The same traits fix how many possessions a game has, which sets how wide the margin distribution is, not which side wins. Enters the one calibrated probability as a fitted scale term on the discrete margin distribution conditional on the line (key-number mass at 3, 7, 10 stays discrete), never a side rule.

## Tried
Novelty check only. Nearest: SIM-18 (traits, side term only), MOD-18 (lattice, done), mod25d-variance-source lane; grep 'plays per game' and 'tempo pair' found none.

## Next
First unit for one agent: Predict each game's plays from the two offensive coordinators' and head coaches' prior-season traits (leave-season-out), report out-of-season R2 against realised plays, then add-one LOSO on cover log loss through the discrete read against intercept-only. Data: Coordinator keys from SIM-18 and PER-07, nflverse play-by-play (free).

## Open
SIM-18 says no pick term helps; the scale term is untested. Pace is priced by the total line, so the add-one must use the total as a control. Zero crossing closes nothing; one fitted probability selects the side.
