# mod24-postseason-forward-2026-11-03

## Goal
Forward 2026 score of MOD-24 (postseason games as extra base-ridge rows). Parent: docs/lanes/mod24-unit-next-2026-11-03.md.

## State
Frozen rule written before scoring: artifacts/mod24_postseason_forward/frozen_rule.json (199 non-REG rows through 2025, sha256 of ids, alpha 10, 88 columns, 0 tuned parameters). scripts/mod24_postseason_forward.py (freeze | score FIRST LAST); imports forward2026_ftn_terms.locked_inputs/score, weekly past-only refit, delta = arm minus base margin mapped through the locked line sweep at the Tuesday opener.
RESULT (measured, report_forward2026_w01_08.json): Week 8 results not yet in, so Weeks 1-7, 102 decisive games. Served 60-42, arm 60-42, 10 flips, decisive 5-5 v 5-5 (decisive record first). Accuracy 0.0 pts [-5.9,5.9] P+ .445; log loss gain +0.0047 [-0.0091,0.0190] P+ .746; Brier gain +0.0023 [-0.0045,0.0094] P+ .747 (game-level bootstrap). 3 cells recorded, family mod24_postseason_rows, ids *_forward2026_w01_08, unresolved_below_power. Looks 3.

## Tried
One run, no tuning.

## Next
Job spec for root, Tuesday after each settled week, once Weeks 8+ have results (the frozen file is reused; score is cumulative from the first week):
  .venv/Scripts/python.exe scripts/mod24_postseason_forward.py score 1 LAST   (LAST = last settled week)
then record the report with a batch like artifacts/mod24_postseason_forward/batch.json, names suffixed _forward2026_w01_LAST, via `nfl-ats weak-signals record --batch`. Run after the week's lock, single core, ~2 GB. Weeks 10-18: LAST = 9..17 on successive Tuesdays; weeks 17-18 are the final regular-season games.
After the 2026 postseason, do NOT fold 2026 playoff rows in without a new frozen file.

## Open
Week 8 outcomes absent at run time, so w01_08 is Weeks 1-7. Effect is far below resolution at ~100 games. 2011-19 replication is another agent's lane.
