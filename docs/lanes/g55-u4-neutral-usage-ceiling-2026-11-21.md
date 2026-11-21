# g55-u4-neutral-usage-ceiling-2026-11-21

## Goal
Positive-control diagnostic, deliberate leak, never servable (family mod57_availability_ceiling): how much of the P2-L who-plays ceiling (ALL 835-668 v base 802-701, 11-11 inputs) is the game result leaking through realised snap share versus usage a pregame forecast could know. Servable S_ALL reached 809-694 (G55-U3). Zero crossing closes nothing; one fitted probability selects every side.

## Declaration (frozen 2026-11-21 14:52, before any run)
- O2_neutral: P2-L O2 unchanged (sev2 = clip(1 - realised share / prior EWMA share, 0, 1); same PFR prior, gating, enrich path) except realised offense+defense share counts only the team's scrimmage plays (play_type pass, run, qb_kneel, qb_spike; both participation lists non-empty) with pre-snap |score_differential| <= 8. Mechanism: touchdown plus two-point conversion is the most one possession scores. Source participation 20260813T131635Z (gsis ids per play 2016-2025) joined on game_id, play_id to pbp 20261111T010519Z; team-games without neutral plays get sev2n 0.
- Arms on today's served features: served_base (must be 802-701), ALL_rerun = O1+O2+O3 (reference, not a look), ALL_neutral = O1+O2n+O3, O2_neutral. Looks 2 (two arms x all games). Season-block bootstrap; gap share = (W_ALLn - W_S_ALL)/(W_ALL_rerun - W_S_ALL).

## State (2026-11-21 15:02 DONE, all measured)
- Script scripts/g55_u4_neutral_usage_ceiling.py (modes build, grade; ruff format + check clean). Outputs artifacts/g55_u4/: build_summary.json, oracle_features_neutral_LEAK_NOT_SERVABLE.parquet, results.json, per_game.parquet, batch.json, build.log, grade.log, pool.log. Build 398 s, grade 4 x 14 s.
- Build checks: participation full-game share v PFR share corr 0.998, median ratio 1.00 every season; 65% of joined scrimmage plays within 8; team-games with no neutral plays 1,540 (2 in 2020-25, rest pre-2016); O1, O2 equal the P2-L cache (max diff 0), O3 max diff 0.63 (projected QB rebuilt today); corr(O2, O2n) 0.936.
- Base 802-701, max diff to G55-U3 base 0. ALL_rerun 830-673 (+1.86 [0.63, 3.06] P+ .998). ALL_neutral 823-680 (+1.40 [0.21, 2.48] P+ .998; log loss .69222 v .69678). O2_neutral 802-701 (0.00 [-0.75, 0.72] P+ .465). ALL_neutral v ALL_rerun -0.47 [-1.30, 0.20] P+ .098; v S_ALL +0.93 [-0.20, 1.97] P+ .940.
- Gap ALL_rerun - S_ALL = 21 wins; ALL_neutral - S_ALL = 14; share surviving 0.67 (7 of 21 wins were the score leak).
- Coefficients: ALL_rerun O1 -2.4..-3.3 against O2 +2.6..+3.3 every season (offsetting); ALL_neutral O1 -0.3..-0.9, O2n +0.2..+0.6, O3 about -0.95; O2_neutral alone -0.03..-0.24.
- Recorded 2 cells unresolved_below_power (g55_u4_ALL_neutral_all_accuracy_points, g55_u4_O2_neutral_all_accuracy_points); shard registry/weak_signals/mod57_availability_ceiling.json modified; pool exit 0.

## Tried
- First background launch failed: shell redirect into a missing artifacts/g55_u4 directory; created and relaunched.

## Next
1. Root: verify results.json and the shard, commit script + shard + lane, move lane to done/.
2. ROADMAP G55 update: one-score usage keeps 14 of the 21-win ceiling over S_ALL; about one third was score leak. Inferred: O2n still has a positive coefficient alongside O1, so in-game injuries and within-score personnel choices remain in the 14; read it as an upper bound on pregame-knowable usage.

## Open
- Pre-2016 training rows have O2n 0 (no participation), O2 nonzero from 2012; fits for 2020+ train on that mix.
