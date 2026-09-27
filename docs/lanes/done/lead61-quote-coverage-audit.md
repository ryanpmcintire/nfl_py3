# LEAD-61 half-line quote coverage audit

## Goal
Explain missing prospective half-line matches without rewriting historical rows.

## State
**Measured:** a read-only trace of the 32 final Week 2/3 decisions found 16 events
absent from the selected capture, 15 with no common bookmaker, and one usable
match. The trace command exited 0 on 2026-09-27. This completes the coverage audit;
the research signal remains unresolved and the source-coverage work stays open.

## Tried
- **Measured:** all 16 Week 2 events exist in raw responses and parsed quotes with
  matching game IDs. DET-BUF has three common books. The other 15 games use a
  Bovada full-game line and BetRivers/FanDuel second-half lines, with no common book.
- **Measured:** all 16 Week 3 events are absent from the 14 usable local half-line
  captures. The latest capture is 20260915T160610Z-halves; its 16 requested events
  all returned in 6.34 seconds. No later usable half-line capture exists locally.
- **Read:** `src/nfl_ats/half_line_refresh_overlay.py:173` joins full-game and
  second-half lines on game ID and bookmaker. Lines 211-223 reject a selected
  half-line capture older than seven days. Historical decisions predate that guard.
- **Read:** `docs/lanes/done/odds-api-key-deactivated.md:3` records deliberate
  cancellation of the paid feed; lines 37-39 record disabling its capture jobs.
  `docs/lanes/done/lead61-second-half-channel.md:25` documents the current free
  providers' missing second-half markets. Paid jobs were not reactivated.
- **Measured:** root verification ran `.tools/uv.exe run --no-sync python
  .tmp/lead61_quote_trace.py`; output is `.tmp/lead61-root-quote-trace.log`, with
  per-game evidence in `.tmp/lead61-quote-coverage-evidence.json`.
- Ledger: `artifacts/prospective/half_line_refresh_decisions.parquet`, SHA-256
  `bd63a56d3993863adfe58a47c8ea9e933a40a5b8f74091198b8a79e871bdf8da`.
  Final 32-row subset digest:
  `e2ba3ac735fc39724cbcad57e1c659af797239f770194835bffe44a07b665000`.

## Next
Require a fresh, predeadline free-source payload containing full-game and
second-half lines with a common stable bookmaker identifier before wiring a feed.

## Open
**Inferred:** observed misses are explained by source coverage; this audit found
no parser or game-ID normalization defect. Missing prospective quotes cannot be
reconstructed as historical observations. This is an implementation diagnosis,
not a signal rejection or a research-closing ground under AGENTS.md.
