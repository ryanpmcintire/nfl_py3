# LEAD-64: Friday final designations reach the card before Sunday

## Goal
Decide whether parsed PFT headline injury designations
(`nfl-ats injury-headlines`) should be wired into `report_status` /
`_injury_value_features` / `availability.resolve_unavailability` as a
timestamped fallback for Friday designations missing from the nflverse
injuries release.

## State
Decision: **do not wire.** Evidence is underpowered, not refuted. No `src/`
change made.

## Tried
1. Ran `nfl-ats injury-headlines --since 2026-08-19` fresh. **Measured**:
   `artifacts/injury_headline_designations/20260924T183645Z/` — 1060 rows,
   986 unparsed, parsed designation_counts out=20 doubtful=7 questionable=17
   ir=29 active=1.
2. Joined parsed 2026 Weeks 1-3 headline rows (62 rows) to
   `data/players/raw/20260924T131530Z/weekly_rosters.parquet` (name+team+week
   -> gsis_id): 53/62 matched (85%).
3. Joined matched rows to `data/players/raw/20260924T131530Z/injuries.parquet`
   (season=2026, week in 1..3) by (week, gsis_id), taking the last row by
   `date_modified` as the official final status. **Measured**: only 23/53
   (43%) of headline-flagged players appear anywhere in that week's official
   injury report at all; the rest are dominated by season-ending
   IR/surgery headlines (e.g. "out for the year", "placed on IR") that never
   generate a weekly practice-report row.
4. Of the 26 out/doubtful headline rows, only **3** matched a player-week
   with a non-null official final `report_status`; 2/3 agreed exactly.
   **n=3 is too small to claim precision is high or low** — this is
   `unresolved_below_power`, not a refuted mechanism.
5. Checked `data/players/inactives/` (the direct "out" ground truth: actual
   game-day inactive lists) as an alternative check. **Measured**: all 52
   capture directories contain an empty `inactives.parquet` (0 rows each).
   The inactives capture pipeline is not producing data — this channel is
   currently unusable for any comparison, independent of headline quality.
6. Timing check on the 23 rows that did match an official row: headline
   `first_seen_utc` was earlier than the official's first observed timestamp
   for that player-week in 23/23 cases (100%), median lead ~134 hours —
   confirms headlines arrive earlier as expected, but says nothing about
   accuracy on its own.
7. No `src/` files touched, so ruff/mypy/pytest and the Week 3 dry-run
   verification (step 3 of the task) do not apply this round.

## Next
- Fix `src/nfl_ats/inactives_capture.py` (or its scheduler job) so
  `data/players/inactives/<ts>/inactives.parquet` actually captures rows —
  separate bounded task. This is the strongest ground truth for "out" and
  is currently silently producing empty output every run.
- After inactives capture is fixed and/or more weeks accumulate, rerun this
  join (script pattern used this round: roster name+team+week -> gsis_id,
  then join to injuries.parquet last-row-by-date_modified, plus inactives
  join for "out") with a larger n before revisiting the wiring decision.
- Only wire `_injury_value_features` / `resolve_unavailability` once out/
  doubtful precision is measured on n large enough to be decision-grade
  (this round's n=3 is not).

## Open
- Why do season-ending IR headlines rarely produce a matching weekly
  practice-report row? Worth confirming against nflverse's own injuries
  schema semantics (is IR excluded from the weekly injury report by design,
  or is this a real gap) before assuming it's expected.
