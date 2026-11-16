# v4 weather from Open-Meteo 24h previous runs (2026-11-16)

## Goal
Replace the thin live GFS MOS feed of the graded v4 W/QW arms with Open-Meteo, which covers any upcoming game. (1) Agreement of Open-Meteo 24h-lead temp/wind with the MOS archive on 2021-25 open-air games. (2) Regrade W/QW with Open-Meteo weather in training history. (3) Fetch Week 11 Open-Meteo forecasts with fetch time.

## State
- Script scripts/v4_weather_openmeteo.py (subcommands agree, grade, live); outputs artifacts/v4_weather_openmeteo/; live forecasts data/external/open_meteo/live/ (git-ignored).
- DECLARED 2026-11-16 21:12 UTC, before any run (looks: 2 arms, family v4_pinned_batch):
  - OW = W arm with temp/wind replaced by Open-Meteo previous_day1 (best_match, 24 h lead, F and mph at 10 m) wherever the value exists on an open-air game (archive roof outdoors/open) in 2021-25; otherwise MOS archive as in W. Coverage known before grading: temp 2021-25 (947 of 979 open-air), wind only 2023 (5), 2024, 2025; 2009-20 keep MOS entirely. 2026 rows unchanged from W (graded population is 2020-25 only).
  - OQW = same replacement on the QW table. scripts/v4_candidate.py has no switch for the QB replacement constant, so OQW uses the graded -0.15 (repair_ab), not -0.123.
  - Grading identical to v4_candidate: stage 1 served walk-forward opener evaluation, stage 2 four-term logistic LOSO on pick_probability/20261116T171555Z (1,503 graded). Compare vs S (829-674) and vs W / QW (836-667), season-block and week-block bootstrap, probability_positive, changed picks. No 2026 replay in this unit.

## Tried
- Agreement (measured, `agree`, artifacts/v4_weather_openmeteo/agreement.json; 912 open-air 2021-25 archive games; MOS tmp F, wsp knots x 1.15078 = mph, 10 m; Open-Meteo F, mph, 10 m, best_match): 24 h temp n 891 r 0.975, OM-MOS mean -0.55 F, sd 3.9, MAE 2.8; 24 h wind n 353 (2024-25 only) r 0.779, mean -2.30 mph (MOS 10.4 v OM 8.1), sd 3.1. 0-6 h hist: temp r 0.978 -0.45 F; wind r 0.796 -2.82 mph. Wind is level-shifted lower in OM (grid vs station); not corrected (no compensating constants), so serving OM wind needs the model trained on OM wind.
- Live Week 11 (measured, `live --week 11`, fetched 2026-11-16 21:14 UTC): data/external/open_meteo/live/week11_20261116T211419Z.csv + _raw.json (git-ignored). 9/9 outdoor games ok, plus 2 retractable unknown-roof (HOU, DAL) fetched; 2 domes skipped. Lead 76-172 h, not 24 h: re-fetch about 24 h before kickoff for parity with training.

- Grade DONE (measured, `grade` exit 0, artifacts/v4_weather_openmeteo/results.json; identity S 829-674, max gaps 0.0). Replaced: temp 891/912 open-air 2021-25, wind 353 (2024-25 only). Season-block [95%] P+ (week-block P+):
  - OW 833-670: vs S acc +0.27 [+0.13, +0.42] P+ .998 (.80), LL +0.00006 P+ .83; changed 16 (10-6). vs W (836-667) acc -0.20 [-0.56, +0.07] P+ .06, LL +0.00003 P+ .84; changed 9 (3-6), 2020 games change through stage-2 LOSO coefficients.
  - OQW 835-668: vs S acc +0.40 [+0.13, +0.67] P+ .997 (.87), LL -0.00001 P+ .50; changed 22 (14-8). vs QW acc -0.07 (1 change, 2025_09_SEA_WAS, 0-1), LL +0.00002 P+ .78.
- Recorded 8 cells (family v4_pinned_batch, unresolved_below_power, artifacts/v4_weather_openmeteo/cells.json); `weak-signals pool` exit 0.

## Next
1. The orchestrator decides the feed: OQW keeps nearly all of the QW gain (1 pick off) with a feed that covers every live game; OW loses 3 picks against W. To serve it, the v4 landing would train on this history and fetch Open-Meteo about 24 h before each kickoff (`live --week N`, which is the same lead as training).
2. Optional: an OQW arm with the -0.123 QB constant (v4_candidate needs a constant switch first).

## Open
- Wind runs 2.3 mph lower than MOS, and Open-Meteo has no 24 h wind for 2021-23, so those seasons keep MOS wind and history mixes two wind levels. A history built entirely from Open-Meteo hist 0-6 h (complete 2018-25) is untested, and it is not provably pre-lock.
- No 2026 W1-10 replay was run for OW/OQW.
