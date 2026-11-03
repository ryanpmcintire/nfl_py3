# SIM-15 Weather-conditioned play-calling

## Goal
Measure how weather changes play-calling, as a simulator mechanism. ROADMAP row SIM-15.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: Wind and rain change what teams call, which changes variance and total as well as level; the simulator draws calls independent of weather. Estimate within-team elasticity of pass rate and explosive-pass share to wind and precipitation. Data: nflverse pbp with weather fields for outdoor games, 2009-2025 (local).

## Tried
Novelty check only (grep of ROADMAP, docs/lanes incl. done, docs/*.md, registry/weak_signals). Nearest prior: sim09_personnel_catalogue.md B5 (listed only), LEAD-75 wind forecast innovation (market pricing of wind), ENV-02 (roof), forecast_weather_screen.md (weather versus cover). None measures play-calling elasticity.

## Next
Build the within-team panel and run the roof placebo. First unit: First unit: team-fixed-effect regression of pass rate and deep pass share on wind speed and precipitation indicator with a roof placebo; no engine edit.
Research rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Weather-forecast timing versus observed conditions: use observed for mechanism, forecast for any pregame term.
