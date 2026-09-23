# Lane: displayed confidence -> served probability (done)

## Goal
Replace the in-sample 12-cell displayed cover chance (audit finding 4) with the
served fitted probability.

## State
Shipped in `52aa98b`: `src/nfl_ats/displayed_confidence.py` lost the in-sample
lookup and its constants; card, board and publishing read the served
`four_term_pick_probability_v1` probability. Production already used the served
probability whenever a pick-probability model was active.

## Tried
Out-of-season calibration of the served probability is the MOD-20 unit 4 base
reliability read (`docs/lanes/pooled-signal-model.md`): quintile predicted
0.397/0.460/0.488/0.529/0.603 vs observed 0.432/0.460/0.449/0.510/0.631, OOS
Brier 0.2454 vs model-only 0.2517.

## Next
None. Whether strength-word cutoffs should be chosen out of season is a
separate question, not raised.

## Open
None.
