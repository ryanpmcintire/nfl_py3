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

## Rerun 2026-10-18 (Weeks 1-6; join: `tests/scratch/codex/lead64_join.py`)

**Measured:** inactive ground truth exists only Weeks 3-6 (201 player-weeks, 33 team-weeks, 23 games; pregame-only,
Week 5 has 5 teams). 104 parsed headline designations (40 questionable, 39 IR, 20 out, 1 doubtful, 4 active);
23 checkable (16 team-weeks). Headline out 1/4 inactive (Wilson 5-70%); questionable 0/9 (0-30%); all 1/23
(0.8-21%). Only 1 of 201 inactives had any headline. Headline v nflverse official: 13/103 player-weeks matched;
headline earlier in 7/13, median lead 12.1 h, IQR -10.2 to +13.7 h (nflverse first capture is hourly-scale, not
publication). nflverse rows carry no date_modified, so no authoritative official publication time; rule
official time <= capture <= cutoff < kickoff is unusable. Zero 2026 weekly-report intervals recorded; no unit fits
(precision is not accuracy_points), so no weak-signals record.

## Declaration (not built)

Pregame-safe feature candidate: per team-week, count of starter-level Out/Doubtful headline designations dated strictly
before Friday cutoff and before kickoff, leak-asserted at build; test only as a fitted term in the one calibrated
probability, LOSO, never a flip. Not testable yet: needs official publication time and a historical headline archive.
Precision half-width <= 0.10 needs about 96 checkable out/doubtful headlines (now 4); at 1 per team-week-pair rate
(~1 per 4 captured team-weeks, ~8 team-weeks/week) that is roughly 12 further weeks of inactives coverage.

## Join audit 2026-10-18 (supersedes the Rerun precision numbers above; script `tests/scratch/codex/lead64_join2.py`)

**Measured:** the old join was defective. `headline_guess` is a slug fragment, so parsed "names" were junk
("breece hall adonai mitchell among jets", "mike hall questionable with an ankle injury mason graham") and 23 of 104
rows had no team. Fix: match every nflverse/inactives player name (normalised, suffixes stripped, len>=8) as a substring
of the slug; team from the player; week = that team's next kickoff after the page lastmod. Result: 98 named-player rows,
21 in captured team-weeks. Out 5/9 inactive (Wilson 27-81%), questionable 3/8 (14-69%), doubtful 0/0, all single-
designation 8/17 (26-69%); old figures 1/4, 0/9 were join artefacts. Headline-named inactives 10 of 201 (v 1).
Misses are explained, not noise: DeVonta Smith W4 (official Out; PHI W4 inactive capture holds only 3 names, truncated);
Smith W6 and Chase W5 (played; stories mislabelled "mixed"/post-game); Cooper W5 (season-ending IR, IR is not "inactive").
Headline v nflverse official: 76/76 player-weeks in file are earlier in the headline, median 14.3 h (IQR 13.5-17.8) before the
first official capture; before the inactive capture median 46.1 h (n=10). Caveat: lastmod is page modification, not
publication, so leads are lower bounds on lateness. Inactive coverage is Weeks 3-6, 23 games, some teams truncated.
Still `unresolved_below_power`; no weak-signals record (precision is not accuracy_points). Next: use official Out/Doubtful
(not inactives) as ground truth, treat IR as a separate class, and only then enter as a fitted LOSO term, never a flip.
