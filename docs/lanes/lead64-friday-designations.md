# LEAD-64 Friday designation and inactives freshness

## Goal

Decide whether provisional player-status headlines can improve early-week coverage, and keep inactive
captures from treating demonstrated post-kickoff carryover as pregame evidence.

## State

Local audit complete; no serving changes, fitting, outcome inspection, or registry writes.
**Measured:** 278 source captures, 611 inputs; official report/capture timestamp pairs n=0.
**Inferred:** no source demonstrates fresh final Friday designations; earlier cutoff deferred.
Report: `docs/lead64_freshness_audit.md`; reproduction: `scripts/lead64_freshness_audit.py`.
**Read:** the existing kickoff guard rejects captures at/after mapped kickoff, retains missing-kickoff
rows, and records `stale_after_kickoff` if all rows are rejected. Headline serving remains deferred.

## Protocol (declared before audit measurement)

Population: all locally archived injury/inactive snapshots and their capture scheduler logs available
at audit start; no downloads, scores, or outcome fields. Target: source-native report identity/time,
first appearance, status change, unchanged carryover, and capture minus official report time.
Terms: source, Eastern capture weekday, season/week/team/player, report status and practice status;
compare consecutive captures within source and team, separating new-week repeated statuses from
within-week repeats. First appearance means first local observation, not first source publication.
Use all captures (no fitting/folds); census counts, transition proportions with descriptive 95% Wilson
intervals, and timestamp lag median/range when an explicitly official team-report timestamp exists.
Unknown native times remain unknown; fetch times and generic page update times are not report times.
One descriptive audit, zero predictive looks. Check weekly-roster IR membership against same-week
weekly injury rows and explicit season-ending evidence; absence alone cannot establish exclusion intent.
Freeze this protocol; any unavailable fields are documented rather than replaced with proxies.

## Tried

Prior audit (**reported**): kickoff replay kept 11 pregame and rejected 11 rows in each of two
postgame captures; `.tmp/lead64-verification.log`. Previous headline agreement was only 2/3.
**Measured:** 8 Friday nflverse captures: 4 first designations, 0 changes, 747 repeats
(100%; descriptive 95% Wilson 99.5-100%). Saturday supplied 243 first designations.
All 17,608 current-season row observations lack date_modified. Last Friday current-week counts
were 8/6/12 versus latest 61/105/119; later-than-16:30 Friday coverage is unmeasured.
**Measured:** new-week same designation 40/71 (56.3%; 95% 44.8-67.3%); Week 3 RES report
inclusion 19/279 (6.8%; 95% 4.4-10.4%). Reported season-ending Jaxson Dart is RES and present.
**Measured:** audit command below and scoped Ruff format/check passed; two headline manifests
lack current.parquet. Logs and input hashes: `tests/scratch/codex/lead64-*` and
`tests/scratch/codex/lead64_freshness_audit.json`.
Run: `.tools/uv.exe run --no-sync python scripts/lead64_freshness_audit.py --as-of 2026-09-29T21:32:34Z`.

## Next

- Capture authoritative final-report identity and official publication time before evaluating a Friday
  cutoff. Proposed rule in the report: official time <= capture <= cutoff < kickoff; no fixed age yet.
- Recheck headline precision after more weeks and trustworthy inactive ground truth accumulate.
- Resolve nflverse exclusion intent from authoritative policy and timestamped IR transactions.

## Open

- RotoWire supplies no source-native publication date, week, or game identifier in the archived page.
- The n=3 final-status agreement result is not adequate evidence for serving Friday designations.
- Season-ending IR exclusion intent remains unresolved: RES is not a season-ending flag, and local
  season-ending headline examples include both missing and present weekly-report rows.

## Record commands

None: descriptive source audit only; no effect estimate, fit, score, or research closure verdict.
