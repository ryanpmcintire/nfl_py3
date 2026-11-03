# LEAD-110

## Goal
Market absorption of the wording read: does the Tuesday-to-deadline line move already load on the read availability difference (ROADMAP LEAD-110).

## State
Added 2026-11-03 (backlog expansion 7), not started, nothing run. Mechanism: If the market prices the wording, the read adds nothing beyond the move; if it lags, the read leads the move. Fitted term: read availability difference in a model of the move, then in the cover probability with move as a separate term. Data: LEAD-106 reads, Tuesday and deadline lines (local).

## Tried
Novelty check only. Nearest: LEAD-101 (news delta), LEAD-104 (view revision), MKT-25. Differs: tests absorption of one frozen read, not a new read.

## Next
First unit for one agent: Regress move on the read difference with designation-ladder and intercept-only controls, game-level bootstrap and a within-week permutation null, weeks 2-8.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.

## Unit 1 result (2026-11-03 5:4x PM local, agent)
Script scripts/lead110_absorption.py, output artifacts/lead110_absorption/report.json. 91 games (weeks 2-8, 294 id-verified sampled players). Position weights: OLS of spread minus ratings-implied on Out/Doubtful count differences by group, seasons 2011-2025 only (mkt25_unit2 method). Move = Tuesday noon ET to deadline consensus median, toward home. Deadline view, not opener.
Loading of move on read-minus-ladder availability: -0.97 [-8.8, 6.1] P+ .38, perm p .73 (designation-only ladder control -0.74 [-12.6, 5.6]). Unresolved. Ladder availability loads negative on move (-1.7 [-2.4,-0.6]), unexplained; suspect sampling-weight/baseline artefact, not diagnosed.
Cover on residual side: 38-48 all (mean -2.6 pts [-5.8, 0.7] P+ .06); top half 19-24. Looks: 2 loadings + 1 ladder control + 2 cover = 5. Recorded 4 cells (family lead110_absorption).
Next: read is sampled (not every player), so team sums are partial; need full-population forward reads (Week 9+) before conclusions.
