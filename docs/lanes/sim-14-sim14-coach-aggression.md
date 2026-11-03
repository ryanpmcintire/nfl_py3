# SIM-14 Coach aggression latent

## Goal
Decide whether per-coach fourth-down go rate over expected is a stable trait the simulator should carry. ROADMAP SIM-14. Family sim14_coach_aggression, declared before outcomes: L1 odd/even split-half within coach-season, L2 cross-season t vs t+1, L3 two-point tries (not run).

## State
Unit 1 done 2026-11-03 (script: scripts/sim14_coach_aggression.py; data/pbp/team_style/raw_pbp_narrow.parquet 2009-2025 REG, schedules head coach). 64,091 fourth downs (pass/run/punt/FG, qtr 1-4, no kneel/spike), go rate 0.155. League go model: HistGB on yardline, to-go, score diff, game seconds, quarter; fit leave-one-season-out; LOSO log loss 0.190 vs 0.432 constant. Residual centered per season (league drift), measured.
- L1 split-half r 0.27 (570 coach-seasons, both halves >=5 plays), Spearman-Brown 0.43; permutation null (1000, within-season play shuffle) mean -0.01, p95 0.07, p<0.001.
- L2 cross-season r 0.46 (414 pairs, >=10 plays each), bootstrap 0.37-0.53, P(r>0)=1.0, coach-permutation null p95 0.08. Coach-season residual SD 0.031 go-rate.
Verdict: stable trait (not zero reliability). Caveat inferred: coach is confounded with team, QB, kicker, and play-caller; not separated.

## Unit 2 (declared before outcomes; family sim14_coach_aggression; new looks L4, L4b, L5, L5b, L3, 4 OC-split cells = 10, 7 recorded; all via scripts/sim14_coach_aggression.py, measured 2026-11-03)
- L4 coach movers (last old-team season vs first new-team season, consecutive, >=10 plays): n 9, r -0.41 [-0.86, 0.70], P+ 0.23. L4b gap<=3, >=5 plays: n 25, r -0.02 [-0.47, 0.38], P+ 0.46.
- L5 team across coach change: n 159, r 0.105 [-0.02, 0.23], P+ 0.95, perm p 0.10. L5b: n 153, r 0.11, P+ 0.96. Reference same coach same team: n 405, r 0.466 [0.37, 0.55].
- L4 minus L5 -0.51 [-0.97, 0.60], P(L4>L5) 0.20: L4 is not stronger; movers are underpowered (9-25 pairs), so no trait-travels claim.
- L3 two-point go over expected (2009-2017, 11,518 PAT/2pt tries, LOSO state model, coach odd/even split, n 65 coaches >=15 tries/half): r 0.50 [0.36, 0.66], SB 0.67, null p95 0.21.
- OC split (data/artifacts/staff_state, HC/OC person per team-season): same HC new OC r 0.56 [0.42, 0.68] n 106; same HC same OC 0.41 n 245; new HC same OC 0.22 n 11; new HC new OC 0.10 n 91. Persistence follows the HC staying, not the OC; no OC play-caller attribution possible (playcaller fields sparse).
Conclusion for the engine hook (inferred): not separable. Trait persists only when coach and team stay together (0.47) and falls to about 0.1 when either changes; mover sample too small to say which. Hook: key the latent to coach-team spell, shrink toward league on a coach change.

## Tried
Fourth downs only. Two-point tries not run: full pbp with two_point_attempt covers 2009-2017 only (narrow 2009-2025 file lacks it).

## Next
Engine owner hook (not applied). The engine resamples historical drive analogs rather than modelling 4th-down choice explicitly; the latent enters like stl/kst team latents (sim10_drive.py:1628-1642, drawn per world from wseed.spawn so default RNG stream unchanged): add a per-coach-season latent a_c (shrunk: variance = reliable share 0.46 of 0.031^2, tested LOSO). In the analog weighting just before cumulative sampling (sim10_drive.py:1315-1317, inside the tilt_weights block 1296-1313) multiply weights of analog drives ending in a fourth-down go (needs new flag array like self.punt, loaded at 1161-1165) by exp(beta*a_c), beta fit LOSO. When latent unset, skip the multiply: identical weights and RNG draws. Two-point: pat_points (sim10_drive.py:1330-1338) p2 would take the same offset on logit scale after L3 is measured on 2009-2017.

## Open
Midseason coach changes use schedule head coach. Record: weak-signals records for L1/L2 pending below.
