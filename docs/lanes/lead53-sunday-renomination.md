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

## Tried

Read the scheduler log, Week 2 failure metadata and stderr, the prospective ledger, the original-card loader,
and the recorder and scheduler call paths. The current production ledger probe is retained at
`.tmp/lead53-current-paper-ledger.log`.

## Next

Keep collecting future Tuesday/Sunday pairs; the current Week 3 pair is recorded. Preserve each attempt diagnostic with the scheduler log.

## Open

The exact Week 1 guard result and the historical Week 2 input difference cannot be recovered from retained
evidence. Current artifacts no longer reproduce the Week 2 loader failure.
