# SIM-18

## Goal
Lead-protection clock burn: run rate and seconds per snap when leading in the fourth quarter, per coach, and the drive count it removes (ROADMAP SIM-18).

## State
Added 2026-11-03 (backlog expansion 5), unmeasured, nothing run. Mechanism: Leading teams burn clock, shrinking the number of remaining possessions and capping the trailing team; drive count is the mechanism that changes the margin distribution. Enters SIM-10 as a team-conditioned remaining-drive count; the pregame side gets a favorite-margin term in the line-conditional distribution.

## Tried
Novelty check only. Nearest: MOD-28 (margin compression trait), SIM-09 Tier B, SIM-10; grep 'prevent' found only a capture note. Differs: Clock and drive-count mechanism, not a margin-level trait.

## Next
First unit: Measure remaining drives and seconds per snap by lead size and coach, reliability with permutation null, compare with MOD-28 compression on the same games.

## Open
Data: Local play-by-play (free). Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener (opener as model input, with intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.
