# MOD-27 unit 6 - pool line off the market at posting

## Goal
Measure how often the pool's Tuesday line differs from the betting market when the pool freezes it, whether the side getting the better number covers more, and serve the offset through the same discrete-lattice mechanism as the MOD-26 hook (generalized beyond half-point hooks on whole numbers).

## State
Measured 2026-11-02 (scratch pool_offset2.py in session scratchpad 352ecbdc..., output pool_offset2.csv): Tuesday market = median across books of each book's first spread quote at or after Tuesday noon ET of the game week (tuesday_noon_et_cutoff_utc); offset = market minus pool home line; graded at the pool line (decisions.graded played arm results), 120 decided 2026 Week 1-8 games.
- Offset 0: 62 games; 0.5: 44; 1.0: 13; 2.5: 1.
- Side getting the better number: 36-22 (P+ .966); hook subset (whole-number market, pool 0.5 off) 26-18 (P+ .884); non-hook (pool off by a full point or more, e.g. IND 7.5 vs 8.5) 10-4 (P+ .941). Positive in 7 of 8 weeks (Week 3 4-5).
- Card with the better-number side 20-16; against it 6-16 (P+ .017). Hook games: card against 5-10; non-hook: card against 1-6.
- Recorded family pool_offset_at_posting_2026 (4 cells, unresolved_below_power).
- Mechanism (inferred): the card reads the base at the pool line with the point anchored on the pool line, so it treats the pool's number as the market; the served move term starts from Wednesday book quotes, so an offset already present at posting is invisible to every term except the hook (branch mod27-hook-serve, whole numbers only).

## Next
- After the hook branch passes its atlas chain: generalize the hook term to any pool-versus-Tuesday-market offset (value = logit cover at the pool line minus at the market line, base anchored at the market line, same fitted coefficient); test LOSO 2020-25 with stacked simulated offsets of 0.5 and 1.0 before serving.
- Keep the live tally weekly from pool_offset2.py logic.

## Open
- The first quote after noon can arrive hours later (Tuesday 18:25Z capture), so part of the offset can be a Tuesday-afternoon move.

## U6 offset-term simulation (measured 2026-11-02, scratch agentPO/p1.py p2.py)
- Stack: 1503 games x offsets {-1,-0.5,+0.5,+1} (declared first), pool 5 prior seasons via prior_pool, point solved so cover/(cover+loss) at market line = model_probability, pushes dropped; 5885 rows. Offset value = logit cond cover at pool line minus at market line.
- LOSO offset coef per fold: all 0.85-0.96; |0.5| 0.69-0.87; |1.0| 0.87-0.99; whole-number market lines only 0.18-0.38 (unexplained, inferred: push handling).
- vs four-term: log loss +0.00238 [0.00099,0.00379] P+ 1.0; Brier +0.00119 P+ 1.0; accuracy +0.92 pts [-0.64,2.38] P+ 0.869. 0.5 only: LL +0.00086 P+ 0.968; 1.0 only: LL +0.00384, acc +2.09 P+ 0.991.
- 2026 W3-8 (103 games; W1-2 feeds lack the discrete base): applying coef 0.8846 to served prob re-anchored at Tuesday line changes 0 sides (mean abs p shift 0.009). Recorded family pool_offset_term_generalized (2 cells).
## Next
- Serve only bundled with the hook (same coefficient ~0.88, same pinned-file edit). Live gain is mechanical (grading at pool line); no side flips now.
