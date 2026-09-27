# LEAD-64 Friday designation and inactives freshness

## Goal

Decide whether provisional player-status headlines can improve early-week coverage, and keep inactive
captures from treating demonstrated post-kickoff carryover as pregame evidence.

## State

Do not wire the headline feed into the card. The measured sample is too small to establish precision.
Inactive capture now rejects a parsed row when its capture time is at or after its mapped scheduled
kickoff. Schedule times use the repository's Eastern-time contract and are compared in UTC. Missing
kickoffs preserve rows; mixed captures retain valid rows and warn. When every parsed row is rejected,
the manifest records `stale_after_kickoff`. The parquet schema and source provenance are unchanged,
and successful fallback rows clear an earlier primary-source `empty_reason`.

## Tried

1. **Measured 2026-09-24:** `injury-headlines --since 2026-08-19` produced 1,060 rows, including
   74 parsed rows. Of 62 parsed Weeks 1-3 rows, 53 matched the weekly roster, but only 23 appeared in
   that week's official injury report. Only 3 of 26 out/doubtful rows had a non-null final official
   status; 2 of 3 agreed. The 23 official-row matches had median lead of about 134 hours. This
   establishes timing, not accuracy.
2. **Measured 2026-09-26:** captures `20260924T225037Z`, `20260926T193023Z`, and
   `20260926T225045Z` each stored the same 11 ATL/GB rows as `2026_03_ATL_GB`. The Saturday fallback
   files are byte-identical. The page names no source date, week, or game identifier, so the unchanged
   post-game rows are **inferred** stale carryover.
3. **Read:** `_parse_rotowire_grid` has only caller-provided season, week, URL, and fetch time.
   `_schedule_lookup` now carries the scheduled kickoff into `run_capture`. The persisted row schema
   remains unchanged, and a row with no mapped kickoff is retained.
4. **Measured 2026-09-26:** archived replay kept the Thursday pregame capture at 11 rows and changed
   both Saturday captures from 11 to 0 rows with `stale_after_kickoff` and an 11-row rejection warning.
   Existing focused tests passed 7/7. Scoped Ruff format and check both passed. Full output is in
   `.tmp/lead64-verification.log`.
5. **Read:** publication requires capture before the pick deadline and on the kickoff's Eastern date.
   Trigger detection may retain a post-kickoff capture only with `deadline_valid=False`, and comparison
   excludes games without a deadline-valid trigger. No current downstream path can use either named
   Saturday capture as pregame evidence.

6. **Measured:** native freshness audit of all three archived RotoWire pages found
   no publication time, week, or game identifier for the inactive list. All three
   parse to the same 11 player identities; date-bearing text belongs to navigation
   or general guidance. Raw and normalized payload hashes agree with their manifests.
   Evidence: `.tmp/lead64-source-freshness-report.txt`. **Inferred:** identical
   rosters alone cannot establish staleness; retain the measured kickoff guard.

## Next

- Gather source-native freshness evidence before imposing an earlier pregame cutoff. The kickoff guard
  cannot distinguish stale carryover captured before kickoff.
- Recheck headline precision after more weeks and trustworthy inactive ground truth accumulate.
- Confirm whether nflverse intentionally excludes season-ending IR cases from weekly injury reports.

## Open

- RotoWire supplies no source-native publication date, week, or game identifier in the archived page.
- The n=3 final-status agreement result is not adequate evidence for serving Friday designations.
