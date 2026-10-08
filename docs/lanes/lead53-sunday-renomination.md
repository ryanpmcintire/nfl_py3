# LEAD-53 Sunday re-nomination history

## Goal

Explain the missing Week 1 Sunday pairing and Week 2 Tuesday row without reconstructing prospective
history from later information.

## State

**Measured:** The prospective ledger has the Week 1 Tuesday nomination for `2026_01_ARI_LAC`, no Week 1
Sunday pairing, no Week 2 row, and a Week 3 Tuesday row with no Sunday pairing.

**Measured:** The retained Week 2 failed-run log records the raw-card OR-union invariant error under
`best_pick_sunday_renomination`.

**Read:** `pick_refresh.original_card` calls `clv.load_paper_decisions`. That loader enforces the paper-ledger
composition union at `clv.py:1387-1445` and emits the exact error retained in the historical log.

**Inferred:** The loader validation failure prevented historical original-card construction from reaching the
recorder. The retained evidence does not identify the historical input difference that triggered it.

**Measured:** A current `load_paper_decisions(Path("artifacts"))` read exited 0 with 48 rows: 16 each for
2026 Weeks 1, 2, and 3. The historical failure does not reproduce against the current artifacts.

**Measured:** The Week 1 scheduler log records a successful Sunday refresh but retained only the last stdout
line. The exact recorder result and skip branch are unavailable.

**Read:** Tuesday and Sunday recorders retain their original settlement and validation paths. Every
recording-enabled Sunday attempt now writes its result or skip reason to an atomic latest-attempt diagnostic.
No validator was weakened and no historical decision was backfilled.

**Measured 2026-09-27:** the live Sunday refresh recorded the Week 3 Sunday pairing after
normalizing `paired_at_utc` to nanosecond precision. The prior millisecond Parquet column
rejected a microsecond recording time. All eleven existing recorder tests passed; the
live result is saved in `.tmp/resume-refresh-result.json`. No earlier week was backfilled.

**Measured 2026-10-08:** `.tmp/confidence_nominee_readiness.json` paired ledger has Week 4 Tuesday (2026_04_PIT_CLE, 2026-09-29T19:29:10Z) and Sunday (paired 2026-10-04T12:30:37Z, same nominee PIT_CLE, 16 eligible each); Week 3 pair also present; Week 5 Tuesday LV_NE captured 2026-10-06T22:37Z. Week 2 missing from the paired ledger, Week 1 Tuesday only. Scheduler log BEST-PICK-LEDGER shows the Week 4 Sunday refreshes (10-04 10:03, 11:57, 15:01 ET) all `already_recorded`; no BEST-PICK-LEDGER line exists for the Tuesday recording or the 12:30Z Sunday write.

**Read 2026-10-08 (cause):** the only BEST-PICK-LEDGER log site was `capture_scheduler.py` `notify_after_job`, which runs only for `refresh_`/`lineups_` jobs and logs only `refresh_sun*`. The Tuesday write (`publishing.py:416`, publish with record_decisions) and any non-scheduler or non-refresh_sun Sunday write never logged.

**Fix 2026-10-08:** `best_pick_refresh_prospective.py` `log_ledger_write` (line 458) is called by `record_best_pick_tuesday` (482) and `record_best_pick_refresh` (706); it appends one `BEST-PICK-LEDGER tuesday|sunday` line to `<data_root>/scheduler_log.txt` when `recorded` is truthy. The scheduler line now skips recorded results to avoid duplicates (`capture_scheduler.py:2390`). Verified on a temp data_root with patched inner recorders: recorded=1 wrote 2 lines, already_recorded wrote none; ruff clean. Running scheduler (PIDs 28736/31824) imports scheduler code at start: restart needed for the `capture_scheduler.py` change; recorders run in job subprocesses and pick up the library change immediately. Week 1/2 not backfilled.

## Tried

Read the scheduler log, Week 2 failure metadata and stderr, the prospective ledger, the original-card loader,
and the recorder and scheduler call paths. The current production ledger probe is retained at
`.tmp/lead53-current-paper-ledger.log`.

## Next

Keep collecting future Tuesday/Sunday pairs; the current Week 3 pair is recorded. Preserve each attempt diagnostic with the scheduler log.

## Open

The exact Week 1 guard result and the historical Week 2 input difference cannot be recovered from retained
evidence. Current artifacts no longer reproduce the Week 2 loader failure.

Orchestrator 2026-10-08: logging fix reviewed and committed; gates passed. Scheduler not restarted: until it is, a refresh_sun write may log twice (harmless). Check the next Tuesday 2026-10-13 write logs one BEST-PICK-LEDGER tuesday line.
