# POOL-03 Coin-flip-band leverage term

## Goal
Coin-flip-band leverage term: in games where the model is indistinguishable from 0.5, a fitted pool-objective term picks the less-owned side ROADMAP row POOL-03.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: With 250 entries and one paid place, ties at the top are common and sole first needs picks the field does not share. POOL-05 found that flipping 10 or 25 public sides loses, which is expected where the model has real edge. In the band where the model own hit rate is indistinguishable from 0.5 the accuracy cost is near zero and differentiation may add simulated first-place chance. Data: data/pool_observables (250 entries, W1-8), POOL-01 field machinery, market-implied public shares as the pre-lock ownership proxy.

## Tried
Novelty check only: Nearest: pool01-standing-aware-2026-11-02.md (observed replay and POL-05: contrarianism loses at fixed flip counts, Best Pick choice), pool-rank-card.md, POOL-02 (entrant skill). Differs: restricted to the measured coin-flip band and entering as a fitted term, never a rule.

## Next
First unit: Validate the proxy: correlate market-implied public shares with observed W1-8 field shares per game; then run the observed replay with the band cut point chosen leave-one-season-out and report the P(first) change and the accuracy cost together, counting each band as a look.
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Look count and family to be declared before any sign is seen. Nothing here closes on zero crossing; record unresolved results with nfl-ats weak-signals record.
