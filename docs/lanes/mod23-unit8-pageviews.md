# MOD-23 unit 8 - player-level Wikipedia pageviews in the base model

## Goal
Test whether abnormal attention to each team's starters (Wikipedia daily pageviews) adds to the base model by itself on the fair opener test (baseline 802-701, 53.36%, 1,503 decisive 2020-2025).

## State
- History check (read): team-level pageview attention was screened (attention_battery_*, docs/attention_followup.md; weak_signals families, 21 hits). No player-level pageview test found in ROADMAP.md, docs/, registry (ROADMAP has 0 hits). New here: player starters, abnormality vs own baseline, scored as a base-model feature.
- Declaration (before scoring): cutoff = Monday 00:00 UTC of the week's first game; days used are the 7 days before it (one game cycle) with a 28-day own-baseline just before that (four cycles); both are design units, not tuned. Starters = QB1, top 3 RB/WR/TE by offense snaps, top 3 defenders by defense snaps from the team's latest game with gameday before the cutoff. Score = log1p(window mean) - log1p(baseline mean), standardised with mean/sd of earlier seasons only (2016 gets 0). Assert: max day used < cutoff and source game < cutoff on every row.
- Arms (2, looks = 2 arms x 2 eras = 4 accuracy looks): A_qb (QB1 only) and B_all (QB + skill + defense), each one column home-minus-away added to the 88-input ridge, alpha 10, via mod24_u1 harness run_arm. Pre-2016 rows carry 0.
- Scripts: scripts/mod23_unit8_fetch.py (resolve titles, download), scripts/mod23_unit8_grade.py (build, grade). Raw: data/raw/wiki_pageviews/<run_id>/ with manifest sha256; processed: data/processed/wiki_pageviews/.

## Tried
- First UA without a contact URL got HTTP 429 from Wikimedia; fixed UA (repo URL) works. Fetch started 2026-10-16 as a background PowerShell task (python process running scripts/mod23_unit8_fetch.py, log artifacts/mod23_unit8/fetch.log; finished when data/processed/wiki_pageviews/latest_run.txt exists and the log prints 'run ... ok'). Nothing scored yet.

## Next
- Check fetch.log finished (kill any leftover mod23_unit8_fetch python by command line if stuck; it resumes from cached files), then `grade.py build`, then `grade.py grade`; record with weak-signals record.

## Open
- Title resolution by Wikipedia short description can pick a wrong namesake; unresolved players drop out (coverage reported).
