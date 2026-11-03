# POOL-02 Entrant-skill panel

## Goal
Test persistent entrant skill with a proper null before any entrant-weighted term. ROADMAP row POOL-02.

## State
Unit 1 measured 2026-11-03 (family pool02_entrant_skill, 5 looks declared first: 2 splits x [bootstrap, within-entrant permutation, coin/market-week nulls counted as one null family] + 1 posterior; 3 cells recorded, unresolved_below_power). 241 of 250 entries, W1-8, games 16,16,16,16,15,14,14,14 (pushes not observable). Measured (script in scripts/pool02_entrant_skill.py, 4000 draws, seed 20261103):
- odd v even Spearman-Brown 0.077, bootstrap [-0.227, 0.299], P+ 0.69. Coin-flipper null mean -0.007 [-0.29, 0.23], P(null >= obs) 0.28. Market-week-rate null (week p = field rate) -0.010, 0.27. Within-entrant week shuffle null -0.083 (biased negative since totals fixed), p 0.044 (not a clean null).
- W1-4 v W5-8: 0.029 [-0.301, 0.269], P+ 0.56; coin null p 0.41, market-week 0.41, shuffle 0.12.
- Observed var of entrant rates 0.00198 v binomial 0.00205: no excess spread, field is consistent with pure luck.
- Inferred: our 65 pts (53.7% of 121 games v field 48.3%, rank 19, 15 above, 9 tied): shrink 0.07 (hi CI 0.22); posterior edge +0.37 pp (sd 1.19), P(true skill > field mean) 0.62 (0.71 at upper-CI reliability, 0.50 at zero). Our P(first) 6.67% [5.29, 8.08] (POOL-01) already uses SB-weighted offsets, so skill priors do not move it; the rank is mostly luck.

## Tried
Split-half, nulls above. Our weekly split not used (only total known here).

## Next
Do not add an entrant-weighted term. Re-run on W9-18 for out-of-sample persistence (2027 season needed for out-of-season). Needs per-entrant per-game picks (view-only Chrome by root) to test line-following.

## Open
Nine entries unread; ties in pushes unknown; single season.
