# Served-code correctness review — 2026-09-29

## Goal
Review the assigned served-code changes without changing code or running operational jobs.

## State
Complete through `bd3eadb`. Only this lane changed. **Inferred:** block unattended publication rollout on finding 1. **Measured:** proposed patch passes `git apply --check`; it remains unapplied.

## Tried
- **Measured:** requested scoped `git log --since=2026-09-29 --stat` returned empty; explicit midnight found 13 commits. Read every scoped `git show`, through `bd3eadb`; logs: `%TEMP%/codex-review-<hash>.txt`.
- **Measured:** `.tools/uv.exe run --no-sync python -B -` read-only probes and `python -B -m nfl_ats margin-predict --help` exited 0 using temporary `UV_CACHE_DIR` after default-cache denial. Logs: `%TEMP%/codex-review-20260929-{probes,render-probes,help}.txt`. No Python edits or fits/scores; Ruff and statistical intervals inapplicable.

### Ranked findings
1. **P1 — publication never retries. Measured:** four jobs, zero retries. **Read:** `scripts/capture_scheduler.py:1598`, `:1903`, `:2006` skip recorded failures despite catch-up. A transient Friday push/network or foreign-index failure waits until Monday's publication.
2. **P2 — unpushed commit hidden. Read:** `scripts/publish_site.py:63` returns success before push. After commit succeeds/push fails (`:87`), an identical rebuild (e.g. same-minute retry) reports `unchanged` without pushing.
3. **P2 — single-week captures disabled. Measured:** 16-game Week 4 slice returns false Friday/Sunday. **Read:** `scripts/capture_private_sunday_odds.py:52` rejects NaT cadence before checking an already-started week. Full schedule works; week-filtered `--features` fails.
4. **P3 — mislabeled sort. Read:** `src/nfl_ats/board_terminal.py:178`, `:590` make Kickoff restore pool order (`board_content.py:3672`), wrong for nonchronological captures. **Measured:** current 16-game capture has zero inversions.

### Per-file conclusions
| File (under `src/nfl_ats/` unless prefixed) | Result |
| --- | --- |
| `board_content.py:2438` | No defect found in pool mapping, fallback, missing-game placement, or sort. |
| `board_terminal.py:384`, `:597`, `:678`, `:912` | No defect found in requested features; finding 4 concerns sorting. Measured: source buckets and 45-point header render correctly. |
| `weekly.py:239`; `cli_commands/prediction.py:385` | No defect found: challenger flag bypasses activation; separate output remains UNLINKED. |
| `sharp_book_movement_features.py:119` | No defect found. Measured delayed-snapshot move 1.5 versus timely 4.0; live loader supplies snapshot timestamps. |
| `weak_signals.py:66` | No defect found: RPS unit accepted without weakening directional/closure validation. |
| `scripts/publish_site.py:63` | Finding 2; normal-index hooks and existing foreign-staged guard otherwise work. |
| `scripts/capture_private_sunday_odds.py:52` | Finding 3. Measured full-schedule Friday true, empty/offseason false, UTC/Eastern equal. |
| `scripts/capture_scheduler.py:1583` | Finding 1; this week's settlement/lock/publication ordering is safe in the serial daemon. |
| `scripts/scheduled_weekly_lock.py:96` | No defect found: 3,300-second child timeout fits the scheduler's 3,600 seconds. |
| Adjacent changed board assistant/site-content/CSS | No defect found in reviewed commit hunks, including close-line rendering and `bd3eadb` wording. |

### Minimal publication-recovery patch (findings 1–2; unapplied)
```diff
--- a/scripts/capture_scheduler.py
+++ b/scripts/capture_scheduler.py
@@ -1596,6 +1596,8 @@
             season_guarded=True,
             added_on="2026-09-29",
             catch_up=True,
+            retry_backoff_minutes=15,
+            max_retries=8,
         )
         for day, at in (("fri", "00:30"), ("mon", "00:30"), ("tue", "00:30"), ("tue", "14:30"))
     ),
--- a/scripts/publish_site.py
+++ b/scripts/publish_site.py
@@ -61,6 +61,11 @@
     tracked = [path for path in SITE_PATHS if (REPO / path).exists()]
     changed = _git("status", "--porcelain", "--", *tracked).stdout.strip()
     if not changed:
+        if not args.dry:
+            push = _git("push", "origin", "master", timeout=300)
+            if push.returncode != 0:
+                print(json.dumps({"status": "failed", "step": "push", "error": push.stderr.strip()}))
+                return 1
         print(json.dumps({"status": "unchanged"}))
         return 0
     if args.dry:
```

## Next
Orchestrator: review/apply recovery patch; verify recovery paths; decide single-week fallback and sort wording.

## Open
No operational/network/push verification. Index guard is not a concurrency lock. Later edits need review.

## Record commands
None: read-only code review; no research verdicts or registry writes.
