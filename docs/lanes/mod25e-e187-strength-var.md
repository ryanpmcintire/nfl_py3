# mod25e-e187-strength-var
## Goal
Explain GZIDX 3-seed strength_re_var 59.7 -> 55.1 (-4.6 [-8.3,-1.0]; real ~58.0) although the hook only fixes the CLK clock delta (mod25c_noise.py:376).
## State
- strength_re_var = mean over sim seasons of dv.random_effects(...)[strength_diff_var] fit on realised margins (mod25e_analysis.py:42-49), averaged by e100 lean() (mod25e_e100.py:45-46). It is an estimate from outcomes, not the latent.
- Pairing (read): mod25_generator.py:548-562 spawns world/game seeds from SeedSequence(seed); latents, schedule, game seeds are independent of the mechanism. Measured: base and cand sim_games have identical teams per game_id. Latent strengths are therefore identical; per-game rng is consumed sequentially so outcomes diverge after the first changed draw (only 0.2-0.3% of games have identical scores). Game-margin corr base v cand 0.278/0.290/0.301 (s11-13), same as the older paired fe run 0.286/0.281/0.300 (e171); team-season mean-margin corr 0.78 in both.
- Noise (measured, 14 near-neutral/other 3-seed boot logs): per-seed paired delta sd 2.82 (mean +0.15), so 3-seed-mean SE ~1.6 and -4.6 is ~2.8 sd; empirical sd of the 14 3-seed-mean deltas 0.95 (z -4.8; includes some real effects). Single-run seed sd 2.58 around mean 57.1. Cand 55.1 sits inside that spread; base 59.7 is a high draw (older bases: 56.1, 58.8, 57.5). Per-seed gi deltas -2.7/-6.1/-4.9 all negative (p~.125 for same sign by chance).
- Team-season mean-margin var base->cand: gi -0.9/-2.7/-1.5; fe pair -1.7/-0.5/-2.2 (reverse sign): no distinct drop beyond the neutral pair.
- Mechanism: none found; margin SD -0.19, pts/g -0.17, mean|m| flat. No latent store in e5 outputs, so strength-gap-binned GTG/possession split not measurable (play_* lacks team ids; join via sim_games g possible, not done).
## Tried
tests/scratch/e187/pair.py (game and team-season pairing stats). Boot-log delta table across e*_boot.log.
## Next
Resolve by replication, not mechanism: run GZIDX cand and base on seeds 14-16 once the sim queue frees (paired, same latents); if mean delta ~0, treat as noise. Otherwise bin games by |latent-proxy| (team-season mean margin) and compare possessions and TD rate base v cand.
## Open
-4.6 is neither clearly noise (z 2.8-4.8) nor explained. Adoption stands (defect fix); strength var now 3 below real, inside run spread. Unresolved_below_power; not recorded via weak-signals.

## Orchestrator 2026-10-19
- Queued paired replication seeds 14-16 (base ...faldrg and ...faldrggi both run; tests/scratch/e187/run_gi_rep.sh, last in tests/scratch/e178/queue.txt; boot -> artifacts/mod25e3/e187_gi_rep_boot.log). scripts/mod25e_e100.py now reads E100_SEEDS (default 11,12,13).
- GZIDX stays adopted either way (defect fix). If seeds 14-16 repeat a strength drop of similar size, find the mechanism (team-strength-gap bins via sim_games join); if not, record as divergence noise.
