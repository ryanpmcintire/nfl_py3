# Referee crew home-favouring penalty-yard tendency (2026-11-16)

## Goal
Test whether a head referee's persistent home-minus-away penalty-yard
differential shifts the home ATS margin beyond the line. Screen at the close
2011-2025; a survivor (null 95th percentile, same sign both halves) gets one
declared served look at the opener 2020-2025.

## Declaration (stamped 2026-11-16T21:41:32Z, before any outcome join)
- Game value D = accepted penalty yards charged to the away team minus those
  charged to the home team (play-by-play `penalty_team`, `penalty_yards`),
  regular season. Positive D favours home.
- Crew trait for a game = shrunk mean of (D minus league mean of prior games)
  over that referee's games in earlier seasons and earlier weeks of the same
  season (schedules `referee`). Shrinkage k = within-crew variance / between-crew
  variance by method of moments, estimated leaving the scored season out; no grid.
- Screen: OLS slope of home margin minus closing spread (pushes kept, ats_points
  per SD of trait), 2011-2025; expected sign positive. Null: 200 within-season
  shuffles of referee labels, traits rebuilt per shuffle. Halves 2011-2017,
  2018-2025. Survivor = real slope above null 95th percentile and same sign in
  both halves. Looks: 1 screen cell + 2 half cells; 1 served cell if survivor.
- Timing (read, docs/referee_assignments_capture.md:124-142): assignments post
  Tuesday evening to Wednesday midday, before the pick deadline but after the
  Tuesday opener; any served use is the late-week refresh path.
- Novelty (read, docs/officials_crew_leads.md:104-275): LEAD-32 measured only
  count-differential reliability (split-half r -0.052, P+ 0.325; season-to-season
  -0.10) and never ran an outcome test. This unit uses yards, a 2009+ referee
  history from schedules, LOSO shrinkage, and an outcome screen.

## State
Done 2026-11-16. Screen run once (200 shuffles); no survivor, so no served
opener look. Artifacts: artifacts/referee_tendency/{screen.json,
screen_games.parquet, weak_signals_batch.json, pool.log}.

## Tried
- Measured, close 2011-2025, 3,918 games (101 pushes kept), 57 referees:
  slope +0.023 ATS pts per SD of trait [-0.292, +0.320], P+ 0.555, null
  percentile 71.0 (null 95th +0.306). Halves: 2011-17 +0.132 [-0.368,+0.606]
  P+ 0.701; 2018-25 -0.099 [-0.491,+0.285] P+ 0.286 (signs differ). Seasons
  with positive slope 6/15.
- First stage is real: trait predicts the game's away-minus-home penalty yards
  +1.03 yd per SD, null percentile 98.5 (null 95th +0.57). Trait SD 1.56 yd.
  LOSO k 46-475 games (2012 fold: between variance not positive, trait 0).
- Inferred: a ~1 yd per SD persistent tilt is worth well under 0.1 pt, far
  below the ~1.5 pt add-one floor; the outcome screen cannot see it.
- Recorded 3 cells, family referee_home_penalty_yard_tendency, ats_points,
  unresolved_below_power; pool --effect-units ats_points exit 0.

## Next
None for this construct. A stronger read would need a bigger per-game lever
(e.g. crew tilt interacting with pass-heavy teams' DPI exposure), already
covered by LEAD-59 bins.

## Open
Serving path would be the late-week refresh (crews post Tue-Wed).
