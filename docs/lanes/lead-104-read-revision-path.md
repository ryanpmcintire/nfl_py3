# LEAD-104

## Goal
View-to-view revision path of the LEAD-99 read: does the change from the Tuesday view to the last deadline view carry information the final level does not (ROADMAP LEAD-104).

## State
Added 2026-11-03 (backlog expansion 4), unmeasured, nothing run. Mechanism: Each revision is the read's response to news. The level is anchored by the line; the revision is the news. Terms: final read, revision, line move since Tuesday, all fitted together; canonical-row-only scoring (latest before deadline) cannot see this. Contamination: forward games only; no historical backtest is admissible.

## Tried
Novelty check only. Nearest: LEAD-99 scoring plan (canonical latest row) and ENG-49 (one row per candidate). Grep found no revision-path term in ROADMAP.md, lanes or registry. Differs: uses the multiple views already being paid for.

## Next
First unit: Compute per-game revision (last view minus Tuesday view, in logit) for Weeks 7-9 and its correlation with the line move toward the pick from Tuesday to the deadline; no outcomes needed. Week 8 Tuesday is lost, so Weeks 7 and 9 only.

## Open
Changes: Uses the forward log more fully. Data: Stored runs for all views, Weeks 7-9, then Weeks 10-18. Revision = last deadline view minus Tuesday view, in logit. Zero crossing closes nothing; record unresolved results with nfl-ats weak-signals record and a closing ground. Look count and family declared before any sign is seen.
