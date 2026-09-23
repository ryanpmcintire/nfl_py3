# market-updated-model

## Goal

Owner directive 2026-09-14 (paraphrased): a hard override rule that flips a
pick whenever the line moves enough is wrong on its face -- the market move
should be one more piece of evidence updating the model's own probability,
not a veto. Build and score, honestly leave-one-season-out, a combined
model+market probability (C2/C3) against the card (C0) and against the hard
rule it replaces (C1). Done when the doc, script, registry cells, ROADMAP
row and required checks are in.

## State

Done, 2026-09-14. Shipped: `docs/market_updated_model.md`,
`scripts/market_updated_model_eval.py`,
`artifacts/market_updated_model/20260914T210317Z/` (summary.json,
per_game.csv). Registry family `market_updated_model_v1`, 7 cells, all
`unresolved_below_power`. ROADMAP row `MKT-16` added. Nothing in `src/`
touched; `LATE_WEEK_FOLLOW_SERVED` was already `False` before this lane
started (same-day owner change, not made here).

## Tried

- LOSO logistic blend, 3 coefficients (intercept, a on model logit, b on
  leader-median move): out-of-sample **+2.003 pts vs card** [-2.101,
  +6.045] P+ 0.838, but **-1.001 pts vs the hard rule (C1)** [-2.390,
  +0.373] P+ 0.075 -- interval still touches zero so not a resolved wrong
  sign, but a fairly one-sided read against the combined model beating the
  rule it would replace.
- C3 (move gated at >=0.5) is bit-for-bit identical to C2: measured,
  `leader_median_net` never takes a value strictly between 0 and 0.5 in
  magnitude on this population (it is a median of three books' own
  half-point-increment moves), so the gate is a structural no-op here.
- C2 shares 205 of its 220 card-disagreements with C1's flip set, and adds
  15 new ones -- all on games where the model itself was within 0.02 of a
  coin flip and the market did not move at all; the tiny fitted intercept
  alone tips those.
- Confidence-band table: the share of games where C2 disagrees with the
  card falls monotonically with model confidence (40.8% to 15.9%), and the
  firmest band is the only one with a negative effect vs card (-0.25 pts,
  unresolved) -- descriptively consistent with the owner's "don't override
  a firm model" expectation.
- Season split: C2 reads below the card in 2023 (-1.5 pts) but above it in
  2024 and 2025 (+3.0, +4.5 pts) -- the C2-vs-C1 shortfall concentrates in
  one season.
- Positive control (foresight on C2's 220 disagreement games): +14.77 pts
  [+12.22, +17.37] P+ 1.0, the ceiling for any rule agreeing with that
  disagreement set.

## Next

- If the owner wants to close the C2-vs-C1 question further, the 2023
  season split (Result 9 in the doc) is the specific thing to look at next,
  not a broader re-run of this same family.
- Nothing here changes what is served; C1 (the hard rule) remains off the
  served card per the same-day owner change (`LATE_WEEK_FOLLOW_SERVED =
  False`), and this document does not argue for turning it back on.

## Open

- Whether to fit the optional third coefficient (spread-band interaction or
  books) is an owner call; deliberately not built here to keep the model at
  its declared minimum size.
