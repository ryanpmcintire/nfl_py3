# PER-14

## Goal
Coordinator mover test: does an OC or DC trait travel with the person to the new team or stay with the roster (ROADMAP PER-14).

## State
Added 2026-11-03 (backlog expansion 6), not started, nothing run. Mechanism: Mechanism: SIM-17 and SIM-18 keyed traits to coordinators; if the trait is the person, a new hire's prior-team trait should predict his new team's tempo and halftime change. This decides whether the trait enters the opener for teams with a new coordinator, where the market has least history. Fitted term: portable trait for teams with a changed coordinator in the margin ridge.

## Tried
Novelty check only. Nearest: PER-07 (changes as a dummy, done), SIM-17 and SIM-18 (traits), MOD-44 (head coach tenure); grep 'mover' and 'portab' found none.

## Next
First unit for one agent: List coordinator moves with at least 8 games in both stints; compare the new team's tempo to the mover's old-team trait versus the new team's own prior season, with a bootstrap over movers. Data: PER-07 change table and nflverse play-by-play, 2009-2025 (free).

## Open
The mover sample is small (tens); likely unresolved_below_power. Zero crossing closes nothing; one fitted probability selects the side.
