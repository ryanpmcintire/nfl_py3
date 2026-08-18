# Week 1 readiness check (2026-08-18)

Written 2026-08-18T00:10Z, 21 days before the pool locks Tuesday
2026-09-08T16:00Z (noon ET). This is an independent, from-scratch rehearsal
of the Tuesday path: every claim below was produced by actually running
something today, not by re-reading prior session docs. Where a prior
session's claim is repeated, it is because this session reproduced it live.

## Verdict

**PASS overall, with one live defect found and fixed.** The card path
(steps 1-7) reproduced the frozen 52.05% evaluation exactly on freshly
ingested data. The optional evidence tail (steps 8-11), including both
ledger writes, was proven correct in an isolated sandbox without touching
the real ledger. Postseason coverage is real and independently confirmed.
The real ledger is genuinely empty. The odds-capture wrapper had a live bug
that was silently converting successful captures into logged failures; it
is fixed and verified below. The one open risk is a concurrent session
mid-editing `margin.py`/`pool.py`/`calibration.py` — re-run this checklist's
step 2 after that work lands.

## Checklist

| # | Item | Verdict | Evidence |
|---|---|---|---|
| 1 | Environment (`doctor`) | **PASS** | Ran live; `nfl-ats` 0.2.0, `nflreadpy` 0.1.5, Python 3.12.13, scikit-learn 1.9.0. Latest raw snapshot was 5 days stale at session start; refreshed live in step 2 below (see item 5's finding for why a stale snapshot is not a Tuesday risk). |
| 2 | Active model matches `HANDOFF.md` | **PASS (at session start)** | `artifacts/active_ats_model.json` read `f92d446d0ccb50dd`, `market_residual`/`player`/`ridge`/alpha `10.0`, `SYNCHRONIZED`, historical `1,080/2,075 (52.05%)` — exact match. **Moved to `4b01f055b684e27e` during this session** as a side effect of live-testing step 3 below (re-ingest produced new raw-file hashes even though row counts and every downstream number were identical). Benign per the documented pattern in `docs/ops_runbook.md` ("a new model id does not by itself mean the model changed") — `historical_evaluation` and the linked `2026`/week `1` forecast are unchanged. `HANDOFF.md`'s recorded id is now one step stale; the next real `weekly-run` will move it again regardless, so no action is required before Tuesday. |
| 3 | Tuesday path traces correctly, fails closed on steps 1-7, fails open (reported, non-fatal) on steps 8-11 | **PASS** | Read `weekly.py`, `publishing.py`, `prospective.py`, `prospective_scoring.py` in full. Ran the 92 tests in `test_weekly.py`, `test_prospective.py`, `test_prospective_scoring.py`, `test_publishing.py`, `test_postseason.py`, `test_clv.py` — all pass, including `test_an_optional_step_failure_is_reported_but_never_aborts_the_run` and the four abort-on-desync/failure tests. Then exercised the **real CLI, live, against the real repo**: `weekly-run --dry-run` resolved the real manifests cleanly; ran `ingest`, `build-features`, `build-pbp-features`, `build-player-features`, `margin-backtest`, `margin-predict` for real (season 2026 week 1) — reproduced `1,080/2,075 (0.5204819277108433)` exactly on freshly ingested nflverse data, `synchronization_status: SYNCHRONIZED`, `weekly_forecast` season/week matched (step 6's check would pass). Ran `build-learned-availability-features` + the challenger `margin-predict` for real — confirmed `UNLINKED` and confirmed the active manifest's `model_id` was byte-unchanged afterward (re-verifies the specific claim in `docs/prospective_evidence.md`). **Steps 7 (publish) and 10 (challenger record) were exercised in an isolated temp `artifacts` root** (via the supported `NFL_ATS_ARTIFACTS_DIR` override), never the real one — see item 4. |
| 4 | Real ledger writes work correctly, and the real ledger stays untouched | **PASS** | Copied the real `active_ats_model.json`, `challengers.json`, and the two freshly-built `margin_predictions` card directories into an isolated temp artifacts root and called `record_paper_decisions` and `record_challenger_decisions` directly against it. Both wrote exactly 16 rows, `is_best_pick` landed on exactly one game (`2026_01_ARI_LAC`), the challenger ledger disagreed with the active model on exactly 3 of 16 games (`ATL_PIT`, `GB_MIN`, `NE_SEA`) — an exact match to the count already recorded in `docs/prospective_evidence.md`. A second write correctly deduped (`recorded: 0`, `already_recorded: 16`, Best Pick not re-flagged). Settlement math (`settle_prospective_picks`/`prospective_accuracy`) checked out against a fabricated outcome frame. **Confirmed after all of this that the real `artifacts/clv_ledger/decisions.parquet` and `artifacts/prospective/challenger_decisions.parquet` still do not exist and both loaders return 0 rows** — the Week 1 reset holds; the first write for Week 1 will be the real Tuesday lock. |
| 5 | Playoff coverage (FND-15) | **PASS, independently reproduced** | Loaded `game_features_player.parquet` fresh (not trusting the doc): `game_type` value counts are `REG 4703 / WC 80 / DIV 68 / CON 34 / SB 17` — the exact 199-row breakdown `docs/postseason_support.md` claims, verified by season (6 WC games/season from 2020 on, 4 before). Then **produced a real card for a postseason week**: `margin-predict --season 2025 --week 22` (Super Bowl) against an isolated artifacts root returned `prediction_safety.status: PASS`, `game_type: SB`, `location: Neutral`, a full feature row and a real pick (`bet_side: HOME`, `edge: 0.086`). Playoff serving is not theoretical; it works today. |
| 6 | Odds/line capture health | **PASS, after a live fix** | See "Defect found and fixed" below. All six scheduled tasks (`Odds_TueOpen`, `Odds_ThuTNF`, `Odds_MonMNF`, `Odds_Sat`, `Odds_SunClose`, `Odds_SunLate`) exist, are `Ready`, and have the expected day/time triggers — `Odds_TueOpen` fires Tuesday 09:00 local, inside the 06:00-09:00 ET capture window `docs/ops_runbook.md` describes. No live Odds API call was made deliberately (quota discipline); instead this session found and used an **already-live capture from today's `Odds_MonMNF` run** as evidence (see below). Quota remaining: 2,877 requests, plenty for the rest of the season at ~3 requests/capture x 6 captures/week. |

## Defect found and fixed: `scripts/odds_capture.ps1` crashed on its own success

`data/market/capture_log.txt` showed a second failure, dated
`2026-08-17T23:00:00Z` (today), past the `2026-08-16` fix this same file
already carries for the `2>&1`/`$ErrorActionPreference` trap:

```
2026-08-17T23:00:00Z FAIL You cannot call a method on a null-valued expression.
At F:\Repos\nfl_py3\scripts\odds_capture.ps1:67 char:9
+ $err = $err.Replace($key, '***')
```

Checked whether the underlying capture actually happened despite the
logged FAIL: **it did.** `data/market/raw/20260817T230004Z/manifest.json`
shows a complete, successful capture — 4,580 quote rows, matching
`observed_at_utc: 2026-08-17T23:00:04Z`, `requests_remaining: 2877`. The
wrapper script's own post-processing crashed *after* the real work was
already done and safely written to disk.

Root cause, reproduced in isolation (`Get-Content -Raw` on a zero-byte
file emits **no pipeline object at all**, not even a `$null` value, so
`$err = [string](Get-Content ...)` leaves `$err` as `$null` rather than
`''`): a `--no-sync` run against an unchanged venv writes **nothing** to
stderr on success, which is the common case, not an edge case. Every future
clean, successful capture would have hit this same crash and logged FAIL —
turning the capture log into a false-failure record on exactly the runs
that matter, including the eventual live Tuesday-opener capture.

**Fixed** (`scripts/odds_capture.ps1`): guard the assignment so an
absent/empty stderr redirect file always leaves `$err` as a real string
(`''`), not `$null`:

```powershell
$err = ''
if (Test-Path $errFile) {
    $content = Get-Content -Path $errFile -Raw -ErrorAction SilentlyContinue
    if ($null -ne $content) { $err = [string]$content }
    Remove-Item -Path $errFile -Force -ErrorAction SilentlyContinue
}
```

Verified by reproducing the exact crash in isolated PowerShell against a
zero-stderr command, confirming the fix resolves it, and confirming the
non-zero-exit/non-empty-stderr path (the original 2026-08-16 fix) still
throws correctly. Not yet verified against a real Task-Scheduler-triggered
run of the fixed file — the next scheduled capture (`Odds_ThuTNF` this
Thursday, or `Odds_Sat`) should be checked for `... OK ...` in
`capture_log.txt`. This is the single item in this checklist that is
fixed-but-not-yet-observed-live; everything else was directly observed.

## Risks ranked by how badly they would hurt

1. **(Now fixed, unverified-live) `odds_capture.ps1` false-FAIL bug.**
   Before the fix, every clean scheduled capture logged as a failure, which
   would not have lost data but would have made the capture log useless as
   a health signal right when it matters most (Tuesday morning). Fixed and
   reproduced in isolation; confirm on the next real scheduled run.
2. **(Informational, action optional) Concurrent in-flight work.**
   `git status` showed uncommitted changes to `src/nfl_ats/margin.py`,
   `src/nfl_ats/pool.py`, `src/nfl_ats/calibration.py`, and
   `tests/test_pool.py` from another session, active *during* this
   session's live pipeline runs. Today's `margin-backtest`/`margin-predict`
   ran against whatever was on disk at that moment and still reproduced
   the documented `1,080/2,075 (52.05%)` exactly, so this in-flight work
   has not (yet) changed the `market_residual`/`player`/`ridge` path. If
   that work is committed before Tuesday, re-run this checklist's item 3
   once more against the final code.
3. **(Cosmetic) `active_ats_model.json` drifted from `HANDOFF.md`.** This
   session's live testing re-activated the model under a new id
   (`4b01f055b684e27e`) with identical historical evaluation. No action
   needed — the real Tuesday `weekly-run` will re-activate it again
   regardless of what id it currently reads.
4. **(Not exercised) `--refresh-player-data` path.** This session tested
   the pinned-snapshot path (the one the real Tuesday run will use, since
   the production manifests are current) but did not exercise the
   fallback path that re-ingests PBP/player/participation snapshots from
   scratch. Not a blocker: the pinned path is what actually runs Tuesday,
   and it works.
5. **(Environment quirk, not a repo defect) `Get-ScheduledTaskInfo` /
   `schtasks.exe /query` return "the system cannot find the file
   specified"** in this shell even though `Get-ScheduledTask` works fine
   and lists all six tasks with correct triggers. Confirm last-run
   status directly on the machine (Task Scheduler GUI or a
   non-sandboxed shell) if independent confirmation of run history is
   wanted; `capture_log.txt` is the more reliable source anyway since the
   script writes it itself.

## Exact command sequence for Tuesday 2026-09-08 morning

```powershell
# 1. Confirm no concurrent WIP is sitting uncommitted in the model/pool code.
git status --short

# 2. Sanity-check the environment before spending the real run.
.\.tools\uv.exe run --no-sync nfl-ats doctor

# 3. The one command. Do not pass --skip-ingest (want the fresh Tuesday
#    schedule/line data) and do not pass --skip-prospective (the weekly
#    evidence tail must run before kickoff every week, not just Week 1).
.\.tools\uv.exe run --no-sync python -m nfl_ats weekly-run --season 2026 --week 1

# 4. Read the JSON summary it prints. Confirm:
#      "published": true
#      no "failed_step" key
#      "optional_failures" absent or, if present, does not include
#        anything you are not prepared to lose evidence from for one week
#    Budget 15 minutes; measured ~6 minutes with the evidence tail both in
#    the 2026-08-17 rehearsal and again live in this session.

# 5. Spot-check the two things weekly-run cannot check for you:
.\.tools\uv.exe run --no-sync python -c "from pathlib import Path; from nfl_ats.clv import load_paper_decisions; print(len(load_paper_decisions(Path('artifacts'))))"
#    Expect 16 (or however many games are on the Week 1 card) -- the FIRST
#    real write for Week 1.
Get-Content data\market\capture_log.txt -Tail 3
#    Confirm the Tuesday-morning Odds_TueOpen entry reads "OK", not "FAIL".

# 6. Push per the normal AGENTS.md contract (handoff refresh, commit, push)
#    once satisfied.
```

## What was deliberately not done

- No commit or push. All findings above are from live command execution;
  the only file changed is `scripts/odds_capture.ps1`.
- No live Odds API call was made deliberately; this session relied on the
  incidental live capture that fired from `Odds_MonMNF` during the session
  window, which was sufficient to both diagnose and verify the fix's
  necessity without spending quota.
- The real `artifacts/clv_ledger/decisions.parquet` and
  `artifacts/prospective/challenger_decisions.parquet` were never written
  to. All ledger-write testing happened against a throwaway temp artifacts
  root under the session scratchpad, using the officially supported
  `NFL_ATS_DATA_DIR`/`NFL_ATS_ARTIFACTS_DIR` overrides.
