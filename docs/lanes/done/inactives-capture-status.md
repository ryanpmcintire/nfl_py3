# Inactives capture status

## Goal

Make capture metadata reflect a successful fallback when the primary feed is an empty placeholder.

## State

**Read:** `src/nfl_ats/inactives_capture.py` clears `empty_reason` when the combined capture has rows.
Primary-feed diagnostics remain available independently.

**Measured:** Replaying the archived `20260926T225045Z` primary and fallback HTML returned 11 rows,
`ok=true`, and no empty-feed reason. Every row matched the original parquet exactly.

## Tried

Archived-source `run_capture` replay exited 0; scoped Ruff lint and format checks exited 0;
all 7 existing inactives capture tests passed. Evidence is in
`.tmp/inactives-status-replay.log` and `.tmp/inactives-status-pytest.log`.

## Next

Continue the source freshness investigation in `docs/lanes/lead64-friday-designations.md`.
The status repair does not establish the fallback page's date or pregame availability.

## Open

**Measured:** The three populated snapshots contain the same 11 ATL/GB players, totaling 33 capture
rows. The fallback parser's date/week handling remains under separate review.
