# POOL-07

## Goal
Standings-conditional variance policy: how far ahead or behind we sit changes which picks to take (ROADMAP POOL-07).

## State
Added 2026-11-03 (backlog expansion 5), unmeasured, nothing run. Mechanism: A leader should avoid contrarian coin flips; a trailer needs them. Fitted term: a leverage coefficient on the simulated finish-probability slope, fitted leave-one-week-out on the pool simulator.

## Tried
Novelty check only. Nearest: POOL-01, POOL-03 (coin-flip-band leverage), UI-21 (outlook panel); grep found no standings-conditional policy. Differs: Policy depends on standing, not game ownership alone.

## Next
First unit: Using POOL-04's calibrated simulator, compute the finish-probability gain of each pick as standings vary and test whether the served card would change.

## Open
Data: Pool standings and field simulator (captured). Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.

## Unit 1 result (2026-11-03T21:47:44Z)
Script scripts/pool07_standings_policy.py, artifact artifacts/pool07/pool07_standings_policy.json. 9 weeks (10-18), 20000 accepted paths from rank 19, 65 pts, 250 entries, paid places = 1 (rules observation; no prize table) so top 10 is a stated proxy. Looks: 4 field arms x 6 policies = 24 (policies: max-p, BP leverage, tilt at p-dev quartile cuts q25/q50/q75 derived from the arm's own |p-0.5|, trailing-tilt q50). Base arm: max-p P(first) 7.57% +-0.19, top10 47.2% +-0.35, rank 22.3, wins 74.38. BP leverage -0.02 pp (no change). Tilt q25 +0.51 pp +-0.19 first, top10 -1.9 pp, wins -0.38; q50 -0.81 pp first, -1.18 wins; q75 -3.0 pp, -3.13 wins. Trailing tilt == q50 tilt (we are behind in ~all paths). Sensitivity: sign of tilt flips with field model (weeks 1-4 shares: tilt hurts; weeks 5-8 shares: tilt +1.0 pp; no-skill: tilt ~0). No weak-signals record (no admissible effect unit). Challenger only; no pick change.
Next: derive the tilt cut from a fitted per-game leverage term, leave-one-week-out.
