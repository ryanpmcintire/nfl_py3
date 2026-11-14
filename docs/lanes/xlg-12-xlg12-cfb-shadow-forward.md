# XLG-12 College shadow forward ledger

## Goal
Make forward evidence 4-5x faster by scoring frozen candidates on college openers each week. ROADMAP row XLG-12. Shadow only: this ledger serves nothing.

## State (2026-11-14 late, non-opener rows removed)
- Opener definition (read): the earliest line capture per game and book (`scripts/xlg12_predict.py:118-120` after the 11-03 opener fix), graded on that capture's provider opener, and now also observed no later than the end of the weekly freeze window (`xlg12_freeze_tue` Tue 12:00 ET plus its 360 minute grace, from `capture_scheduler.SCHEDULE`) that precedes the game's kickoff.
- Removed (measured): the 216 week 11 rows per version written 2026-11-14 (capture 19:51Z). Untouched copies and manifest: `artifacts/xlg12_removed_late_rows_20261114/` (v1, v2, `manifest.json`). Ledgers 609 -> 393 rows each. Deleted rather than tagged: the scorer's invalid list covers v1 only. Week 10 score unchanged, 62 games.
- Guard: `xlg12_predict.py:136-166` (`freeze_deadline`, `opener_rows`, `late_games`) and `:170-183` skip non-opener rows with a printed count and reason; `xlg12_weekly.py:~198-206` excludes them from the loud-failure count and prints the reason. Real freeze run now: skipped 201 rows, 36 games, wrote 0, exit 0.
- Week 11 is lost for the opener grade (only the 21 DraftKings rows frozen 11-03 remain). Next freeze Tue 2026-11-17 12:00 ET: expect `week 12 written N` and no `skipped_non_opener` games unless a capture is late.

## State (2026-11-14 14:55 ET, weekly jobs repaired)
Both weekly jobs reported OK while doing nothing (score 11-08, freeze 11-10). Causes, fixes and what was saved:
- Freeze cause (measured): `capture_cfbd_lines.py` reused any earlier capture of a week forever, and the wrapper captures next week one week early, when few lines are posted. Week 11 had only capture 20261103T181857Z (7 games with an opener); the 11-10 run fetched week 12 only (20261110T170111Z). The old command by hand prints `cached 11 ...20261103T181857Z`, `cached 12 ...20261110T170111Z`. The predictor saw the same 7 games, all already frozen, wrote 0. Every later week would have failed the same way.
- Score cause (read + measured): `score` ran only `xlg12_score.py`, which takes final scores from `features_2026.parquet`; only the freeze path rebuilt that file, last on 11-04 08:33 ET (scheduler log), before week 10 was played. With the file rebuilt, week 10 scores 62 games (measured). Second factor (measured): the cfbfastR schedule fetched 2026-11-14T19:51Z had 0 of 28 kicked week 11 games final (earliest kicked 11-11T00:00Z); latest final kickoff 11-08T03:30Z. So a Sunday run usually cannot score the week just played (inferred).
- Fixes: `scripts/capture_cfbd_lines.py:37-46` `--refresh` skips the cache; `scripts/xlg12_weekly.py:181` passes it. `scripts/xlg12_predict.py:118-120` keeps the earliest capture per game and book (a second capture used to cross-join lines with predictions); `:170` guards repeat keys. `scripts/xlg12_weekly.py:131-137` score now ingests schedules + pbp and rebuilds features first; `:157` freeze ingests the schedule before resolving the week; `:193` freeze also scores, because results are fresh on Tuesday.
- Loud failure (exit 3, `xlg12_weekly.py:24`): freeze `:206-215` when it wrote 0 rows while a regular-season game of the current week (regular season and week both from the schedule) has not kicked off and has no frozen row; score `:97-128` when a frozen game is final in the schedule but was not scored. Shown on scratch copies: `xlg12 freeze FAILED: wrote 0 rows for week 11 of the regular season while 33 games that had not kicked off have no frozen prediction` (exit 3); `xlg12 score FAILED: 62 frozen games are final in the schedule but were not scored` (exit 3). The last output line is always a plain summary, which is what the scheduler log shows.
- Ledgers (measured): `forward_ledger_open.jsonl` v1 and v2 393 -> 609 rows each, 0 duplicate keys. Week 10: 372 rows, 62 games (frozen 11-03). Week 11: 21 rows, 7 games, DraftKings (frozen 11-03) + 216 rows written 2026-11-14T19:51:25Z.
- Week 11 LOST (2026-11-14): 27 of 67 games kicked off unfrozen (11 on Nov 11-14 early, 16 Saturday 17:00-19:00Z). Not backfilled.
- Week 11 frozen today: 39 games not yet kicked off (33 new games on both books where posted, plus the Bovada line for 6 games frozen 11-03): Bovada 39 games, DraftKings 33. Capture 20261114T195106Z observed 19:51:06Z, earliest kickoff 20:00Z; every row written and observed before its kickoff, model input = grade line = provider opener. FLAG for root: the task allowed a partial freeze only from openers captured before today; none existed, so these use a same-day pre-kickoff capture. They are identifiable by `line_observed_at_utc` 2026-11-14T19:51:06.983186+00:00. Support (measured): spreadOpen and overUnderOpen were identical across captures for 11 of 11 game-book pairs seen twice (weeks 11-12), so a later capture returns the same opener.
- Week 10 forward score at the opener (measured, `forward_scores.csv` per version; 12 cells, one family, plus 12 variants-disagree cells of n 2-5; wins-losses-pushes, probability_positive from a flat-prior Beta): v1 Bovada base 29-32-1 (0.35), cv 34-27-1 (0.81), slope 29-32-1 (0.35); v1 DraftKings base 32-30 (0.60), cv 35-27 (0.84), slope 32-30 (0.60); v2 Bovada base 31-30-1 (0.55), cv 32-29-1 (0.65), slope 31-30-1 (0.55); v2 DraftKings 31-31 all three (0.50). Log loss 0.690-0.697 against 0.693 for a coin. Running tally, unresolved_below_power, no verdict, nothing recorded in the registry.
- ruff check and ruff format --check pass on the three edited scripts.

## Setup (unchanged since 2026-11-03)
- Frozen rules: v1 `artifacts/xlg12_cfb_forward/frozen_{base,cv,slope}.json` (2026-11-03T18:16:59Z, sbr_multibook 2013-19, 4785 rows, cv alpha 1000, slope k 0.516); v2 `v2/frozen_*.json` (18:33:18Z, every CFB season through 2025, 12500 rows, cv alpha 1000, slope k 0.620).
- Scripts: `capture_cfbd_lines.py` (CFBD /lines, observed_at_utc in the manifest; root decision 2026-11-03: authorized source), `xlg12_features_2026.py`, `xlg12_predict.py --forward-week N --open-input [--version v2]`, `xlg12_score.py`, `xlg12_weekly.py score|freeze [--dry]`. Jobs: `xlg12_freeze_tue` Tue 12:00, `xlg12_score_sun` Sun 20:00.
- The older `forward_ledger.jsonl` (observed-line basis) is not scored.

## Tried
- 2026-11-03 units 1-8: CFBD capture, 2026 features (truncation leak test clean), v1 and v2 freezes, opener-input ledger, scorer, wrapper, jobs. Retro weeks 1-9 with the opener as input (retrospective, not forward): v1 44.8-45.9% on n 425-488, v2 46.6-50.4%; families xlg12_cfb_retro2026 and _v2 recorded unresolved_below_power. An early 61.7% retro was a construction bug (closing line as input, graded at the opener).

## Next
- Sun 2026-11-15 20:00 ET score: expect week 11 reported as kicked off without a final score yet (schedule lag), exit 0. Tue 2026-11-17 12:00 ET freeze: captures weeks 12 and 13, freezes week 12, scores week 11's 40 frozen games if the schedule is final. Check both log lines.
- Root decides whether today's same-day week 11 rows stand. If not, list them in `forward_invalid.json`; the scorer applies that list to v1 only (`xlg12_score.py:24`), so v2 needs the same.
- When the forward tally is written up, record it with `nfl-ats weak-signals record` (accuracy_points); zero crossing closes nothing.

## Open
- Final scores come from a schedule file that updates about weekly; the daily pbp release could score on Sunday. Not built.
- By design a run exits 3 when it writes nothing and an unkicked game has no opener yet (possible Tue 2026-12-01 for Army-Navy, or a manual re-run after a full freeze).
- Provider opener time is unidentified and differs from the Tuesday NFL opener; mod24 measured the proxy gap as unresolved.
