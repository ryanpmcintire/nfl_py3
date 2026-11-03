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

## Unit 1 (2026-11-03, family pool03_coinflip_ownership, 26 cells x 8 held-out weeks = 208 looks plus 1 proxy look)
Script scripts/pool03_coinflip_ownership.py -> artifacts/pool_rank_replay/pool03_coinflip_ownership.json.
- Proxy (measured, 121 games W1-8, leave-one-week-out logistic of served-side share on its line): calibration slope 1.03, intercept -0.003, held-out log loss/pick 0.6796 v coin flip 0.6931; Spearman pooled 0.53 [0.39, 0.65], within-week mean 0.50 (range 0.16-0.75); reliability quintiles within 0.03 except 0.532 observed v 0.503 predicted.
- Term: choose the underdog side when |p-0.5|<=band and weight*logit(share_pred) > logit(p); band and weight chosen on the other 7 weeks by simulated P(first) (POOL-01 forward sim, rank 19 / 65 pts, real rival totals), applied to the held-out week. Model p assumed calibrated (flip cost = full 2|p-0.5|).
- Result (measured, 300 paths): held-out P(first) served 8.58% -> term 9.38%, gain +0.79 pp [0.10, 1.46], probability_positive 0.988 (interval is simulation-path bootstrap only, not game-outcome uncertainty). Chosen bands 0 to 0.03 (weights mostly 0.25); sides changed per held-out week 0,0,1,1,2,1,0,0 (5 of 121 games). Recorded pool03_coinflip_ownership_term_heldout_week (unresolved_below_power).
- Caveats: gain rests on 5 flips; cover outcomes simulated with p taken as calibrated; proxy is line-only. Not servable.
## Next
Replay the 5 flipped games against real outcomes; fit the term weight inside the served logistic rather than by grid; widen paths to 1000.

## Unit 2 (2026-11-03, 6 weights x 8 folds = 48 looks, same family)
scripts/pool03_continuous.py -> artifacts/pool_rank_replay/pool03_continuous.json. Side score logit(p) - w*logit(share)*4p(1-p) (g = outcome variance, inferred), w grid fit leave-one-week-out on simulated P(first), 600 paths.
- Held-out gain -0.04 pp [-0.77, 0.75] bootstrapping weeks and paths, probability_positive 0.43 (weeks only [-0.38, 0.29]). Unit 1 gain does not survive the gate removal. Recorded unresolved_below_power.
- Real replay (measured): 6 sides changed (W3-7); changed side covered in 5 of 6; real points over all weeks 61 served v 65 term. Selected weight 0 to 0.25 by week.
- Verdict: simulator gain not a real objective effect; real-outcome +4 is a handful of games, luck-scale. Serving needs pinned pick_probability edit + re-enrolled validation.
## Next
Better ownership proxy (field shares are only known after lock) before further looks.
