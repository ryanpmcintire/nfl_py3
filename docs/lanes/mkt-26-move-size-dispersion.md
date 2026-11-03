# MKT-26

## Goal
Move size as a dispersion signal: does the size of the opener-to-deadline move predict how far the margin lands from the line, not which side (ROADMAP MKT-26).

## State
Added 2026-11-03 (backlog expansion 6), not started, nothing run. Mechanism: Mechanism: A big move reflects news that sharpens or scrambles the game; if absolute moves predict larger absolute residuals, the cover probability should be pulled toward 50 percent in those games. Fitted term: absolute move in a shrinkage coefficient on the served logit, fitted leave-one-season-out.

## Tried
Novelty check only. Nearest: MKT-23 (direction of moves, done unresolved), MKT-25 (revision size), mod25d-variance-source; grep 'heterosced' and 'residual variance' found none.

## Next
First unit for one agent: Regress absolute opener residual on absolute opener-to-deadline move with season-block bootstrap and a within-week permutation null; then LOSO add-one of the shrink coefficient on cover log loss. Data: Local lines archive 2013-2025 (free).

## Open
Opener grade only; the deadline move is a feature of the line, never the graded number. Zero crossing closes nothing; one fitted probability selects the side.
