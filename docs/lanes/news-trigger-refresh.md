# News-triggered refresh

## Goal

Dispatch prospective, pregame paper-pick refreshes from game-keyed news events while preserving the served model's single fitted probability and side selection.

## State

The MKT-08 bridge accepts lineup changes, posted inactives, and line moves with explicit game provenance. A durable activation watermark prevents historical backfill, and a successful child refresh writes a completion receipt used for deduplication. The exact UTC scan time reaches the pick-revision path as trigger provenance. Evidence rows and dry runs never count as successful dispatches. The Sunday scheduler entry runs at 12:40 Eastern with catch-up disabled; an already running scheduler must restart to load the module-level schedule.

## Tried

Injury-news dispatch was excluded because the retained source identifies capture time but not the affected game. Failed child refreshes remain pending for retry. A first live invocation establishes the watermark before scanning, so retained historical events are not dispatched.

## Next

Measured: the direct dry command passed without writes or dispatching historical events; 139 focused existing tests passed. The root restarted the verified idle scheduler hidden to load the new schedule. Let future scheduled scans create prospective evidence; score the fixed-clock and news-triggered arms only after real decision rows accrue.

## Open

The first live run intentionally dispatches no older rows. One Sunday scan does not cover later inactive postings. Injury news remains blocked until the source provides reliable game identity. No prospective comparison sample exists yet.
