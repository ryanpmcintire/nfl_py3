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

## Tried

Read the scheduler log, Week 2 failure metadata and stderr, the prospective ledger, the original-card loader,
and the recorder and scheduler call paths. The current production ledger probe is retained at
`.tmp/lead53-current-paper-ledger.log`.

## Next

Observe the next eligible live Tuesday and Sunday results and preserve the diagnostic with the scheduler log.

## Open

The exact Week 1 guard result and the historical Week 2 input difference cannot be recovered from retained
evidence. Current artifacts no longer reproduce the Week 2 loader failure.
