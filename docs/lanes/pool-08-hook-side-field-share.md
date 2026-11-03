# POOL-08

## Goal
Half-point hook awareness: does the field over- or under-pick the side the pool's half point helps (ROADMAP POOL-08).

## State
Added 2026-11-03 (backlog expansion 6), not started, nothing run. Mechanism: Mechanism: If entrants ignore which side the hook helps, the hook side is under-owned relative to its true cover chance and a pool-level edge exists; if they chase it, the edge is gone. Enters the POOL-01 standing-aware leverage as a fitted field-share covariate and the pool simulator, not as a side rule.

## Tried
Novelty check only. Nearest: POOL-05 (slot and team ownership add nothing), MOD-26 (hook fitted term), UI-31 (hook chip); grep 'hook' in POOL rows found POOL-01 only.

## Next
First unit for one agent: Join captured field shares to games with a pool half point versus none; compare share of the hook-helped side to the unhooked baseline with a permutation null at game level. Data: Captured field picks, MOD-26 pool lines (local).

## Open
Field picks known only for captured weeks; count games before testing. Zero crossing closes nothing; one fitted probability selects the side.

## Unit 1 (2026-11-03 16:59, family pool08_hook_field_share)
scripts/pool08_hook_field_share.py -> artifacts/pool_rank_replay/pool08_hook_field_share.json. 121 games W1-8, 44 hook (Tuesday market whole, pool = market +-0.5; 18 favour home). Looks: 3 (share residual, hook-term log loss, cover vs model_p).
- Share on hook side minus POOL-05 M0h proxy (LOWO, pool line): +7.68 pp [3.22, 11.94] P+ 0.9995; within-week permutation (20000) two-sided p 0.029. 7 of 8 weeks positive (W7 -3.2). Field picks the half-point side MORE, not less.
- Hook term in home-share logit: coefficient 0.24-0.41 per fold; all-game log loss improvement +0.0033 [-0.0024, 0.0086] P+ 0.881.
- Outcome: hook side covered 26-18 (decisive record) vs mean model_p 0.509: +8.2 pp [-5.9, 22.1] P+ 0.875, perm p 0.29. Not separable from luck.
- Recorded 3 cells unresolved_below_power.
- Implication (challenger only): hook sides are over-owned, so hook edges are more shared; the pool-relative term should discount hook-side leverage by the fitted coefficient; no side rule.
## Next
Re-run at W9+; add hook coefficient to the pool-objective ownership proxy in POOL-01 replay.
