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

Unit 2 done 2026-11-03 23:3x UTC (scripts/per16_unit2_confounds.py: confound / selection / check2026; artifacts/per16/unit2_*.json). Looks: 4 reliability + 8 ladder comparisons x 2 metrics + 2 selection + 5 arms x 2 subsets x 2 metrics (2026) = 46. Recorded 8 cells (unresolved_below_power).
Controls (measured): cells designation x practice (4) then x position group (35, EB m 18.3 and 54.2; held-out m 56-250); Q only. Controlled split-half: coach r .408 [.306,.500], team .408 [.311,.497] (uncontrolled .42); yoy coach same team .301 [.173,.417], team .279. Habit is not position or practice mix.
Held-out 2016-25 (9,575 Q rows) gains: position x practice cells vs dxp LL .0058 P+ 1.0; dxp+team vs dxp LL -.0016 P+ .21 (Brier .0005 P+ .82); dxp+coach vs dxp LL .0026 P+ .91 (Brier .0021 P+ .9995); ctrl+coach vs dxp LL .0072 [.0026,.0115] P+ .998, Brier .0046 P+ 1.0; ctrl+team vs dxp LL .0054 P+ .999; coach over ctrl alone LL .0014 [-.0028,.0051] P+ .76, Brier .0019 [.0009,.0030] P+ 1.0; team over ctrl LL -.0004 P+ .41, Brier .0005 P+ .78. So most of unit 1 gain came from position mix; coach adds a small Brier gain on top.
Selection (measured, play-by-play appearance of QB/RB/WR/TE; false appear 0, sensitivity .80-1.0): unverified Q rows appear less than verified in same season x group x practice: -19.3 pp [-23.9,-14.6] P+ 0; implied unverified play rate RB .45, WR .52, TE .38, QB .12 vs verified .64-.70, .47 QB. Verified play rates are biased UP. OL is 0.2% verified (3,009 rows unverified), so habit has no OL evidence. Team habit verified vs unverified skill rows r .035 [-.16,.23], 84 units, perm p .32 (underpowered, unresolved).
2026 (87 Q, 294 all): read vs dxp Q LL .028 P+ .85; vs ctrl Q .012 P+ .65; vs ctrl+team Q .031 P+ .88; vs ctrl+coach Q -.001 P+ .49 (all: .021 P+ .90). Read no longer clearly beats habit + position + practice; n tiny.

## Unit 3 gsis-keyed outcome, unverified rows included (2026-11-03 18:40 ET)
Artifact artifacts/per03/played_by_gsis.parquet (scripts/played_by_gsis.py build/validate/rerun [--complete]/lead106; validate.json). 70,729 injury player-weeks 2013-2026 keyed on gsis_id; played = in a participation list (offense_players/defense_players, 2016-2025) or any pbp player id (2013-2017 full pbp all roles; 2018-2026 narrow pbp passer/rusher/receiver/fumble only) or snaps>0 where crosswalk exists. Participation seasons cover OL (appear .56 like other groups). 2013-15 and 2026 are lower bounds for non-skill groups. 2026 pbp ends wk7.
Validation vs snaps (43,893 rows both): appearance agrees .971 (appear|snap .957, snap|appear .9997); 2016-25 1.000; 2013-15 .95; 2026 skill-only .53; union with snaps .9998. Crosswalk covers 29 OL rows.
Rerun Q, held-out coach LL gain over league ladder: all rows incl. lower-bound seasons (14,067 rows) .0111 but Probable rate .96 to .78 so untrusted; complete-source rows (2016-25 participation + 2026 crosswalk/skill wk1-7; held-out 2019-25, 9,784 Q rows): coach LL .0089 [.0053,.0127] P+ 1.0, Brier .0040; team LL .0066 [.0026,.0106] P+ .9995; split-half team .505 [.40,.60]; coach yoy same team .382 [.26,.51]. Habit survives.
Confounds (complete): pos x practice vs dxp LL .0047; coach over ctrl LL .0044 [-.0001,.0087] P+ .97, Brier .0032 P+ 1.0 (verified-only .0014 P+ .76); team over ctrl LL .0007 P+ .61; controlled split-half coach .501, yoy .349.
2026 (87 Q): read vs team ladder Q .094 [.023,.157] P+ .996; vs ctrl+coach Q .002 P+ .53 (no clear edge over habit+position+practice, same as before).
Registry: 11 per16 cells replaced with same ids (weak-signals record --batch --replace); the two selection cells kept.
Next: habit by position group incl. OL; full pbp 2018+ (only 1999-2017 on disk) to close 2013-15/2026 gaps; step (4) availability input on served base.

## Tried
Team and coach keys are near-collinear (HC changes rare); not separable here. No decay weighting, no position adjustment.

## Next
(Unit 2 done.) Remaining: unverified OL/ID-gap needs a gsis-keyed outcome (inactives list or all-players snaps with gsis map); weight habit by selection; then step (4) availability input graded on served base with intercept-only and line-move-only controls.
Old Next:
Confound checks before any availability term: (1) role/position mix and depth (a Q special-teamer plays more), add position and practice-status controls to the ladder; (2) selection from unverified rows (early seasons 34%); (3) coach-keyed ladder vs designation x practice ladder; (4) feed expected-availability difference into player availability inputs, graded on the served base with intercept-only and line-move-only controls.

## Open
Habit entry (no side rule): coach-keyed shrunk offset added to the position x practice Q play probability inside the player availability input, expected-availability difference then one calibrated probability.
Entry route if it holds: team/coach shrunk Q play rate replaces the league ladder inside the player availability input (expected availability difference), then the one calibrated probability; no side rule.
