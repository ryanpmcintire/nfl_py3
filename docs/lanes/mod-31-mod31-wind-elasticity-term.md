# MOD-31 Team wind elasticity as a pregame margin and total term

## Goal
Team wind elasticity as a pregame margin and total term: forecast wind times each side measured drop in passing ROADMAP row MOD-31.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: SIM-15 targets how wind changes play-calling inside a team. The pregame consequence is asymmetric: a side whose offense leans on the pass loses more of its expected points at the same forecast wind than a side that runs. The existing weather screens used a fixed pass-heavy label and actual weather, not a per-team elasticity and not the Tuesday forecast. Data: nflverse pbp pass rate and weather 2009-2025 (local) for the elasticity panel; forecast archive and Tuesday forecast from forecast_archive_build.md for the pregame term; roof placebo from ENV-02.

## Tried
Novelty check only: Nearest: weather_interactions.md (pass_heavy_visitor on actual weather, the forecast version never retried), LEAD-75 and lead75_paired_wind (wind revision times prior-game passing exposure, a market-innovation term), SIM-15 (simulator input, not a pregame term). Differs: per-team fitted elasticity, level of forecast wind rather than revision, margin and total jointly.

## Next
First unit: Reuse the SIM-15 team-fixed-effect panel to get each team-season pass-rate elasticity to wind built only from prior games, then check split-half reliability with a within-week permutation null; only if reliable, fit a leave-one-season-out term (forecast wind times own-minus-opponent elasticity) on margin and on total and grade at the opener.
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Look count and family to be declared before any sign is seen. Nothing here closes on zero crossing; record unresolved results with nfl-ats weak-signals record.
