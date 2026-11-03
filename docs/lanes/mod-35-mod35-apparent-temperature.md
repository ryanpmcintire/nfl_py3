# MOD-35 Forecast apparent temperature (wind chill and heat index with dew point) times the climate-exposure gap between the two teams

## Goal
Forecast apparent temperature (wind chill and heat index with dew point) times the climate-exposure gap between the two teams ROADMAP row MOD-35.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: Temperature and wind are screened as separate flags, but players feel one combined quantity: wind chill in the cold and humidity-driven heat in September. A team history of playing in that range sets how far a kickoff is outside its norm. The visiting side with the larger exposure gap should lose more than either flag shows. Data: nflverse game and pbp weather fields (temperature, wind, humidity) 2009-2025; forecast archive for the Tuesday value; prior-season venue means for the exposure gap.

## Tried
Novelty check only: Nearest: forecast_cold_visitor_tilt (registry), sept_heat_home_on_production.json, schedule_flag_battery.md, weather_interactions.md. Differs: one physical index and an exposure gap; grep for dew point, wind chill and apparent temperature returned no hit.

## Next
First unit: Compute the index from recorded weather and the gap from prior seasons only, then run the interaction on margin residual with a roof placebo and a leave-one-season-out fit; compare to the existing separate cold and heat terms on the same opener population.
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Look count and family to be declared before any sign is seen. Nothing here closes on zero crossing; record unresolved results with nfl-ats weak-signals record.
