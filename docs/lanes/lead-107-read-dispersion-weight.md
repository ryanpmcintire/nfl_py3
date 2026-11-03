# LEAD-107

## Goal
Read dispersion as precision: between-read disagreement of the frozen language reads sets how much weight each game's read term gets (ROADMAP LEAD-107).

## State
Added 2026-11-03 (backlog expansion 6), not started, nothing run. Mechanism: Mechanism: LEAD-100 measured between-read SD .052 logit against between-game .093; a game where repeated reads disagree carries less information. Fitted term: read term divided by (1 + disagreement) with the divisor fitted leave-one-week-out, entering the four-term probability for LEAD-101, 102 and 106 reads.

## Tried
Novelty check only. Nearest: LEAD-100 (between-read noise), LEAD-102 (names barely move reads), LEAD-106; grep 'dispersion' found none for reads.

## Next
First unit for one agent: For games with at least 3 stored reads, test whether read dispersion predicts read error versus the later line, with a game-cluster bootstrap and a within-week permutation null. Data: Stored repeat reads from LEAD-100 and LEAD-102 (local).

## Open
n is small (about 15 games in LEAD-100); expect unresolved_below_power. Zero crossing closes nothing; one fitted probability selects the side.
