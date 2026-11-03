# MOD-29 Kickoff-clock rest

## Goal
Measure whether hour-level rest adds beyond integer rest days. ROADMAP row MOD-29.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: Rest is carried as integer days; a Sunday night to Sunday early game is about 6.7 days and Thursday night to Thursday night is 7.0, a short-week Thursday after a Monday game is 3.5. Hour-level rest and the kickoff-clock gap (local hour of previous versus current kickoff) enter as fitted terms beside the existing day-level rest term. Data: nflverse schedules gameday and gametime, already ingested (data/raw schedules parquet).

## Tried
Novelty check only (grep of ROADMAP, docs/lanes incl. done, docs/*.md, registry/weak_signals). Nearest prior: body_clock_screen.md and body_clock_night_screen.md (time-zone and night kickoff), bye and short-week screens in the weak-signals registry. Grep for hours of rest, hours since and kickoff-to-kickoff finds zero hits anywhere.

## Next
Compute hour-level rest and its overlap with rest_diff; declare the family and LOSO fit before any outcome look. First unit: First unit: compute hour-level rest for 2009-2025, report its correlation with the day-level rest_diff already in the features, then fit a LOSO term on the residual of the served ridge and grade at the opener.
Research rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Likely a tiny effect; its value is as a clean unresolved or refuted record, and as a free feature if it clears.

## Unit 1 (2026-11-03)
Family mod29_hour_rest, 3 looks declared before outcomes: hour_resid (hours-diff residual on rest_diff, refit per fold), clock_gap (home minus away local-clock gap, venue tz of previous vs current kickoff), both. Script scripts/mod29_hour_rest.py, output artifacts/mod29/hour_rest.json. Base: logit_p of opener cover prob, LOSO 2020-25, paired lead29 parquet. First-game-of-season rows get term 0. Results: see weak-signals registry cells and artifact.
Results (1503 games, 1415 usable): hours-diff vs rest_diff corr 0.997 (R2 0.994); residual sd 4.5h. OOS accuracy gain vs base (54.7%): hour_resid -0.86pt (P+ 0.175), clock_gap -0.60 (0.228), both -0.86 (0.188); log-loss gain ~0. Three cells recorded unresolved_below_power. Next: residual signal is tiny; no further unit unless pooled with other schedule terms.
