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

## Unit 1 declaration (2026-11-03, written before any sign was seen)
Family mod35_apparent_temp_exposure, 3 looks: (1) primary term x = |away team outdoor-home prior-season mean recorded wind chill for the week - forecast wind chill at site| times outdoor, LOSO 2011-25, logit(p_open_base) + x; (2) roof placebo, same x on dome/closed games with the outdoor forecast; (3) within-week permutation null of x. Apparent temp = NWS wind chill (T<=50F, wind>3mph) else air temp. Heat index NOT built: no humidity or dew point in the aware snapshot or any local file (inferred from column listing). Normal window half-width selected walk-forward on recorded-AT error, not on outcomes. Base: opener-evaluation per_game (2020-25), proxy-opener main_scored (2011-19).

## Unit 1 result (2026-11-03)
Script scripts/mod35_apparent_temp.py, outputs artifacts/mod35/results.json and record_batch.json; 3 cells recorded unresolved_below_power. Outdoor games with a visitor outdoor-home normal: 1985 (visitors from domes drop out); normal half-width chosen walk-forward on recorded-AT error (3 weeks nearly every season).
Primary (LOSO 2011-25, opener input): decisive (pick flipped vs recalibrated base) 376-329 for term in 705 flips; accuracy 52.29 vs 49.92 base, log loss 0.69236 vs 0.69338, Brier 0.24960 vs 0.25011 (intercept-only 50.53%, raw p0 50.98%). Week-block bootstrap: accuracy +2.38 pts prob_positive 0.957; log loss gain +0.00104 prob_positive 0.805. Coef 0.111 per sd, 15/15 positive, sd 0.012. In-sample minus OOS log-loss gain 0.0005. Permutation (200): p(ll)=0.01, p(acc)=0.035.
Placebo (776 dome/closed): coef 0.059 positive, log loss worse (-0.00118), accuracy +1.56 pts: not clean, so a venue-independent confound (visitor climate) is not excluded.
Heat index not built: no humidity or dew point in the snapshot or locally. Base is recalibrated p0 (near 50%).
## Next
Fetch humidity (needs network) for the heat side; test x split cold vs hot; add home-side gap; check overlap with flag_cold_visitor.

## Unit 2 result (2026-11-03; script scripts/mod35_unit2_controls.py, artifacts/mod35/unit2.json; 9 looks, same family)
Subset (needs home and visitor outdoor normals, 2012-25): 1778 outdoor, 133 dome. Served p0 on the subset: 2011-19 proxy 49.1%, 2020-25 52.7% (served rule 52.8%, n=716); intercept-only 50.1%. The ~50% base is real, not a weakened recalibration (the 2011-19 proxy is the weak half).
Cold-visitor flag is an unserved flip overlay, not in the served probability; added here as a covariate.
Controls (home normal, week, week x home normal, cold flag >=25F), recal base: acc +0.72 pP .69, ll +0.00146 pP .81 (uncontrolled this subset: acc +4.25, ll +0.00155 pP .85). Served-p0 offset: uncontrolled acc +1.35 pP .96, with controls +0.17 pP .57, ll +0.00182 pP .84. Decisive (term-base) with controls 261-248 of 509; offset 96-93 of 189. Coef positive 14/14 folds.
Sign: visitor colder-climate side ll +0.00165 pP .82; warmer-climate side ll -0.00028 pP .37; signed gap coef negative 0/14.
Placebo: dome x correlates +.46 with week, -.38 with forecast AT; x by week bucket 7 to 25 F; the accuracy gain is calendar-driven. Dome with home controls only 133 games (dome hosts lack outdoor normals).
Verdict: accuracy gain is mostly week/climate confound; log-loss gain +0.0015 survives controls, unresolved_below_power. Recorded 4 cells (record_batch_u2.json).

## Unit 3 (2026-11-03) forward test wired
Frozen: artifacts/forward_terms/mod35_cold_exposure/frozen_coefficients.json (script scripts/mod35_freeze_cold_exposure.py; fit 2011-25, 1919 outdoor games, joint logit on z0, controls, gap_pos, gap_neg; b_gap_neg 0.01588 logit/F; hash f2b4fffe...; created 2026-11-03T19:17Z; 2026 half-width 3 weeks).
Term in scripts/forward_terms_week.py (mod35_week): MEX MOS bulletin cached before run and kickoff, wind kt to mph, outdoor roof from schedule, normals from recorded outdoor-home weather before 2026; per-game controls stored in feature_timestamps for scoring.
Dry week 9: 7 of 15 have a value (5 not outdoor, 3 missing outdoor-home normal), all gap_neg 0 (visitors' normals warmer than forecast). Other terms' rows_sha256 unchanged vs HEAD script.
Caveat: history GFS MOS, 2026 MEX MOS (different model, bulletins every 2 weeks only).
Next: score after week settles vs opener; refit controls in the scorer.
