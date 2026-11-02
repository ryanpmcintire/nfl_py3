# SKY-01 news-to-availability table (2026-11-02)

Script: scripts/sky01_news_availability.py. Artifacts: artifacts/sky01/ (report_to_status.parquet, lead_times.parquet/json, fitted_term.json, inactive_first_seen.parquet, spread_series.parquet, run logs).

## Table (measured)
6,765 player-week rows (2024: 1,690; 2025: 3,302; 2026: 1,773) from 18,838 tagged non-repost beat posts. Columns: first beat signal per class (out, limited, active), official report time (injury_play_outcomes decision time where present, else first nflverse snapshot flag), final status. Inactives captures exist for 2026 only (506 name-keyed first-seen rows, weeks 3-8) and are stored separately. Official practice/report timestamps: 2,983 rows via decision time, 607 via 2026 snapshot first-flag. Beat posts exist only 2024 onward.

## Lead times (measured, player-weeks with a beat out-signal, 4,868 rows)
- Beat over official designation: median 45.6 h (IQR 12.4 to 70.9), beat earlier in 79.2% of 2,679 pairs. By season: 2024 26.4 h, 2025 65.8 h, 2026 12.4 h.
- Beat to next adverse median-spread move: median 37.4 h (IQR 10.7 to 73.6), 2,120 of 4,084 rows with a line at the beat time ever moved adversely. Official to next adverse move: 27.2 h on 1,313 of 2,679. Inferred caveat: these are unconditional line drift after the timestamp, not a control-adjusted news response, so they bound lead, not causal pricing.

## Fitted term (measured)
One term: home minus away count of distinct players with a beat out-signal before Tuesday noon ET opener, as an offset-logit coefficient on the served four-term probability. LOSO over the only seasons with beat posts (2024, 2025; 533 non-push games, 203 with a nonzero term). Out of sample: Brier gain -0.000178 [-0.001029, +0.000555], probability_positive 0.3345; log loss gain -0.000365, probability_positive 0.3315; accuracy 0.00 points [-0.75, +0.75]. Decisive games (pick changes): 4, new 2-2 base. In sample: Brier +0.0000208, probability_positive 0.5455; gap in Brier gain 0.000199. Fold coefficients -0.0045 (2024), +0.0072 (2025): sign unstable. Looks: 1 term x 2 folds, 3 metrics x 2 (in/out) = 6 graded cells plus 3 lead cells = 9 recorded. Classification: unresolved_below_power (no refuted mechanism or positive control; 4 decisive games).

## Forward-only
Training seasons are 2024-2025 only, so the term is forward-scorable now on 2026 weeks 1-8 once four-term opener probabilities exist for 2026 (the 2020-2025 per_game artifact excludes 2026); live scoring starts with Week 9 openers.
