# LEAD-112

## Goal
Post-opener availability news: change in read availability between the Wednesday report and the Friday and game-day reports, against the line move (ROADMAP LEAD-112).

## State
Added 2026-11-03 (backlog expansion 9), not started, nothing run. Mechanism: The opener is posted before most injury news. The part of the game's availability difference that arrives after the opener is information the opener cannot contain; the part already in the Wednesday report is partly priced. Measured mechanism: late-arriving read availability difference with the Tuesday-to-deadline move as a separate term. Fitted term: Read availability difference at the latest report minus the same difference at the first report, one term in the calibrated probability with the move beside it.

## Tried
Novelty check only. Nearest: LEAD-64 (Friday designations), LEAD-110 (market absorption of the read), PER-03. Differs: Isolates the post-opener change in the read rather than its level.

## Next
First unit for one agent: List games with a first-report read and a later read, count players who change status after the first report, regress the cover residual on the late-arriving difference with a within-week permutation null, using the designation-ladder change as the control. Weeks with no revision history are counted and excluded, not imputed.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the served base (out_of_season_home_probability from the latest artifacts/pick_probability run, with raw-model, intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen. Data: LEAD-106 stored reads, PER-03 2026 revision history once built, Tuesday and deadline lines (local).
