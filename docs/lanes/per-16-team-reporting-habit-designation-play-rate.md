# PER-16

## Goal
Team reporting habit: play rate given the same designation, by team and coach, shrunk, as the availability prior (ROADMAP PER-16).

## State
Unit 1 done 2026-11-03 23:5x UTC (script scripts/per16_team_designation_habit.py: reliability / ladder / check2026; outputs artifacts/per16/*.json).
Coverage (measured): snap counts begin 2013 so 2013-2025 REG. Injuries joined by gsis_id -> weekly_rosters pfr_id -> snaps; verified rows only (pfr id and team-week in snaps): 61% overall, 34% of 2013 rising to 81% of 2025. Q 10,881 verified rows (play rate .673), Doubtful 1,299 (.012), Probable 3,002 (.962, only through 2015). Head coach from staff_state (first listed per team-season; mid-season changes ignored, 3.5% missing).
Reliability, Questionable (402 team-seasons, ~27 rows each; EB tau2 .0129, m 16.1, shrink .63 at mean n): team split-half odd/even r .421 [.335,.508], Spearman-Brown .59, permutation null p95 .093, p .001; coach split-half .420 [.331,.506]; team yoy .316 [.216,.412]; coach yoy same team .337 [.219,.447] (273 pairs), any team .327. Doubtful: no reliability (r -.06, 59 units; yoy -.07 [-.18,.06]); Probable split-half .26 [-.01,.51] but yoy -.16.
Held-out ladder (seasons 2016-2025, prior seasons only, EB m from prior seasons, cluster bootstrap team-week 2000): Q log loss gain vs league ladder: coach .0074 [.0040,.0107] P+ 1.00; team .0034 [.0002,.0065] P+ .979; Brier coach .0032 [.0018,.0047], team .0014 [.0000,.0029]. Doubtful ~0 (-.0014 team, P+ .18).
2026 LEAD-106 verified rows (294; Q 87): wording read still beats the team ladder: all .064 [.015,.125] P+ .997, Q .093 [.026,.155] P+ .999 (vs league ladder Q .066). Team ladder is not better than league on those 87 Q rows.
Looks: 3 designations x (3 reliability families + 2 arms x 2 metrics) = 21 plus 4 check cells. Recorded 5 cells, family per16_team_designation_habit, unresolved_below_power.

## Tried
Team and coach keys are near-collinear (HC changes rare); not separable here. No decay weighting, no position adjustment.

## Next
Confound checks before any availability term: (1) role/position mix and depth (a Q special-teamer plays more), add position and practice-status controls to the ladder; (2) selection from unverified rows (early seasons 34%); (3) coach-keyed ladder vs designation x practice ladder; (4) feed expected-availability difference into player availability inputs, graded on the served base with intercept-only and line-move-only controls.

## Open
Entry route if it holds: team/coach shrunk Q play rate replaces the league ladder inside the player availability input (expected availability difference), then the one calibrated probability; no side rule.
