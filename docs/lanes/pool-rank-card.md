# pool-rank-card

## Goal
POOL-01: choose the card that maximises expected pool finishing position given the served probabilities and a fitted field, replayed over past weeks.

## State
- 2026-09-16: queued by owner; ROADMAP row written.
- Unit 1 DONE 2026-09-16 (subagent-measured, parent-verified by rerun).
  Script `scripts/pool_rank_replay.py`; artifact
  `artifacts/pool_rank_replay/` (`results.json`, `weekly.csv`). No
  registry cell: no valid effect-unit exists for ranks/week (forcing one
  would poison unit-based pooling); outcome lives here and in the
  artifact.

## Tried
- 107 season-weeks replayed, one greedy rank-optimal rule vs served
  (LOSO served probabilities, 4000-sample Poisson-binomial field,
  100 entrants). Mean gain +1.47 ranks/week [-0.00, +2.84], P+ 0.974.
  Decisive weeks 64/107: 25 better / 16 tied / 23 worse (+2.46 on
  decisive). Per-season: +3.37 / +0.73 / +3.33 / +2.95 / -1.23 / -0.21.
  Served 841 vs rank-optimal 848 correct of 1,503.
- Field model is the weak leg: public splits cover 321/1,503 games,
  rest fall back to the opener favorite; pool published picks
  unavailable. Interval touches zero: unresolved_below_power by default.

## Next
- Standing-aware variant (unit 2) only with a fitted field on real pool
  picks; otherwise remeasure when public splits cover more games.

## Open
- Whether the owner wants rank-scale units added to the registry
  (feeds ENG-46 unit 2 scope).
