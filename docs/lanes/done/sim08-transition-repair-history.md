# SIM-08 simulator rebuild to catch the model

## Goal

Make the play simulator (`scripts/sim04_engine.py`) at least match the served
discrete read at the opener, then test whether it adds anything. Done = a LOSO
2020-2025 re-grade where the market-anchored sim ties or beats the served read
on cover-vs-miss and push log loss, recorded with `weak-signals record`.

## State

- SIM-04 was closed while its engine failed its own gate (2/5 key numbers);
  those registry rows refute that engine version only.
- Engine now (**measured**, validation 2015-2017,
  `artifacts/sim04_engine/20260927T025106Z`): 10,002 games in 166.6 seconds,
  4/5 key numbers, log loss +0.0009 vs naive, margin SD 15.00 (ratio 1.081),
  40.89 points and 24.11 possessions/game, with zero cap hits. Versus
  `20260927T021351Z`, points rose 0.31, SD fell 0.07, possessions rose 0.02,
  and key masses changed 3 .101→.104, 7 .077→.080, 10 .051→.055,
  14 .043→.047, 17 .034→.036. Its engine distribution gate is GO; that is
  not a research closure or promotion decision. Built from:
  OT neighbour pool (unit 1, `69fe2f1`); late boosted-tree 4th-down layer
  (unit 4, `0014146`), fitted on whatever seasons the tables use (unit 5,
  `0324ecd`); team yard shift 0.64 yd per unit of rating gap (unit 6,
  `65cd15e`: home edge +1.89, backup QB -1.37); `SCALE_FP` 2.5 stops
  goal-line plays borrowing open-field yardage (`3cc779e`); about 7x faster
  with identical single-process output (`0ed7c9c`).
- `scripts/sim04_loso.py` grades raw, old shift, spike-keeping tilt and a LOSO
  blend with the served cover probability (`a91cbfe`); `--n-reps`/`--workers`
  (`d9e42e8`). Detail for every unit: `docs/sim04_unit_log.md`.
- Early re-grades died on 2026-09-26 from memory (22 workers, then an
  unbounded per-game weight cache, fixed in `f62011e`). Workers are capped at
  min(8, free RAM less 16 GB over 3 GB per worker, CPUs less 4).
- Generator transition replay now preserves sampled outcomes at the goal line:
  a non-scoring crossing stops at the 1; scoring or possession-changing rows
  restore the sampled absolute destination; score without flip ends the drive
  without toggling offense. New drive diagnostics use the post-play clock.
- Terminal score reconstruction (**measured**, same artifact; 10 predeclared
  comparisons plus 1 integrity look) matches all 768 actual final margins. At
  |margin| 3, simulated terminal scores are 3 points 61.6% vs 63.2% actual,
  7 points 33.8% vs 24.8%, from a tie 51.1% vs 62.4%, and in the final five
  minutes 74.3% vs 87.2%; consecutive 7-point events are 19.6% vs 17.9%. At
  |margin| 14, terminal 7s are 63.3% vs 56.4%, consecutive 7s 40.6% vs 38.5%,
  and final-five-minute scores 47.2% vs 59.0% (39 actual games). Chronological
  reconstruction recovered 12 transition-proxy mismatches involving kickoff
  return TDs, safeties, defensive conversion returns, and one standard PAT;
  it does not add those mechanisms to the simulator.
- Late timing reconstruction (**measured**, `.tmp/sim08_timing_diagnostic.json`;
  10 primary and 6 mechanism looks) matched all 115,910 validation transition
  clocks exactly. For tied states, borrowed no-plays were 3.55% versus 5.41%
  actual, delta -1.86 points [-3.28, -0.38], across 1,774 rows/114 games; for
  one-score states they were 5.25% versus 6.08%, delta -0.84 [-1.62, -0.02],
  across 5,867 rows/455 games. Their conditional elapsed-time deltas were only
  -0.18 seconds [-3.12, 3.01] and -0.45 [-2.00, 1.18]. One-score borrowed
  scoring gaps were 2.15 seconds longer [1.10, 3.38], including field goals
  +1.07 [0.22, 1.82], offensive TDs +3.09 [1.31, 5.56], and return TDs +5.05
  [1.71, 8.36]. The loop records scores at the source clock and advances once
  to the next-live clock, so this diagnoses neighbour mixture rather than
  duplicate clock consumption; no engine change was made.
- State-matched late hazards (**measured**, `.tmp/sim08_late_hazard_diagnostic.json`;
  12 predeclared looks, 7,641 rows/463 games) do not support changing the
  uniform 40-neighbour draw. Tied score-frequency deltas move from -0.10 points
  [-1.15, 0.89], P+ .431 at ranks 1-10 to -0.89 [-1.88, 0.02], P+ .029 at
  ranks 21-40, but one-score deltas are +0.69 to +0.75 points with P+ .993-.995
  in every band. One-score conditional score times remain 1.47-1.78 seconds
  long, with all intervals above zero and P+ 1.00; ranks 1-10 are longest.
  The engine hash stayed unchanged.

## Tried

- Unit 2: Gaussian neighbour weighting, no movement, reverted. Named the gap:
  trailing 1-3 late, real teams kick 93-100%, sim 61-64%.
- Unit 3: late score-scale tightening (.61 to .63, reverted); logistic
  4th-down layer on all phases (3/5, 40-49 yd kicks halved, log loss +0.0116,
  reverted).
- Unit 5: k_state 50 helped the slope less than the bandwidth; not applied.

## Next

1. Re-grade DONE 2026-09-26 (`artifacts/sim04_loso/20260926T203158Z/`,
   table in `docs/sim04_unit_log.md` "SIM-08 LOSO re-grade"; five rows
   recorded under family `sim04_play_simulator`, names `sim08_*` and
   `served_cover_probability_loso_recalibration_at_open`). Goal NOT met: raw
   and shift refuted (wrong sign resolved); tilt -0.0023 P+ 0.19; blend
   +0.0043 P+ 0.93 over served, but -0.0004 [-0.0011,+0.0003] P+ 0.14 over a
   served-only LOSO recalibration, so the gain is shrinkage, not the sim.
2. Before another LOSO, separate training-to-validation drift from missing
   state variables for one-score scoring duration by comparing chronological
   actual hazards within the same outcome and state cells. Rank weighting is
   not a repair candidate: it would improve tied score frequency while leaving
   one-score frequency and timing biased. Unsupported special-team and
   defensive scores remain a separate bounded follow-up. Zero crossing does
   not close a signal, and one fitted calibrated probability selects the side.
3. Served cover probability is overconfident (LOSO slope 0.27-0.43, all six
   seasons gain from shrinking); this is MKT-17's known finding, owned by
   `docs/lanes/joint-probability-model.md`, not this lane.

## Open

- None needing the owner.
