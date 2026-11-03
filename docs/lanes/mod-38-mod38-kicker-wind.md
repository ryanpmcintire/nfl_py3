# MOD-38 Kicker wind decay

## Goal
Per-kicker wind x distance sensitivity and its effect on margin mass at 3 (ROADMAP MOD-38).

## State
Unit 1 done 2026-11-03, family mod38_kicker_wind, 3 looks. 9279 FG tries 2009-2025, 49 outdoor kickers with 40+ windy kicks. Global model: dist, dist^2, wind, wind x dist (-0.04 per 10 mph x 10 yd), temp, indoor. Split-half of kicker slope (offset logistic): alt-kick 2-param r=-0.50 (Spearman -0.18, perm null mean -0.01 sd 0.17); slope-only alt r=-0.14, season parity r=-0.11. No positive reliability; recorded refuted_mechanism / no_split_half_reliability. Unit 2 (key-number 3 shift) not run per task rule.

## Tried
Scratch: scripts/mod38_kicker_wind.py (ignored). Forecast wind source if reopened: data/raw/forecast_archive/pool_decision_2009_2025/forecasts.parquet.

## Next
None for kicker slopes. A pooled kicker-wind mean effect (not kicker-specific) is already in global wind terms; SIM-15 covers calls.

## Open
Slopes unstable at ~25 windy kicks per half; a positive control for effect size was not built, so closure rests on zero reliability.
