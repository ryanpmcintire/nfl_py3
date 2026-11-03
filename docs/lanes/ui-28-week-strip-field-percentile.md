# UI-28

## Goal
History week strip adds how each week ranked against the field (percentile and field median) next to our record (ROADMAP UI-28).

## State
Added 2026-11-03 (backlog expansion 4), unmeasured, nothing run. Mechanism: The reader sees 9-6 but not whether that was a good week for the pool. The weekly field distribution is captured; the strip should print the percentile and median in plain words. Fixes the existing strip in place, no new layout.

## Tried
Novelty check only. Nearest: UI-20 strip (record only), UI-24 (season luck band), UI-21 (outlook panel). Grep of dashboard rows and docs/*.html for a weekly field percentile found none. Differs: week-level field context.

## Next
First unit: Add the percentile and field median to the strip for Weeks 1-8 and render at 1440 and 390 widths.

## Open
Changes: Page: History. Data: data/splash/field/*_field_distribution.tsv and standings. Percentile and median in the existing History strip. Zero crossing closes nothing; record unresolved results with nfl-ats weak-signals record and a closing ground. Look count and family declared before any sign is seen.

## Unit 1 done 2026-11-03T20:56Z
Strip now prints "beat N% of entries; field median M wins" per chip, from data/splash/field/2026_weekly_points_w01-08.parquet (241 of 250 entries). Code: board_site_content.py _field_week_texts + SeasonSoFar.field_week_texts; board_terminal.py _week_strip_html. Built in memory (not published); 309 board tests pass. Next: view at 390 width; root publishes.
