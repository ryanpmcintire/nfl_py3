# Backlog triage and dashboard deployment — 2026-09-22

## Goal
Ship the bounded dashboard batch, then find additional implementation work that can proceed with available inputs.

## State
Measured: commit `22b62ee` is on `origin/master`; GitHub Pages [run 35786959715](https://github.com/ryanpmcintire/nfl_py3/actions/runs/35786959715) completed successfully. The dashboard batch and checks are recorded in [weekly-card-readiness.md](weekly-card-readiness.md). This triage is scoped, not a claim that the entire backlog is blocked or cleared.

## Tried
Read: `docs/timing_policy_audit.md:3-19` leaves MKT-08's prospective news-triggered comparison open. Its old audit command was removed; do not rerun it or treat its September 2 ledger counts as current measurements. Read: `docs/open_benchmark_suite.md:56-67` leaves SKY-07's dataset licensing checks, hosting, withheld-label custody, leaderboard rules and first release unresolved.

Reported, unverified: worker triage found UI-19 and the inspected UI-20 confidence legend already implemented; OPS-02 deletion/quarantine requires a separate owner decision, and OPS-06 still needs external host/access. This was scoped triage, not a claim that all remaining backlog work is blocked.

## Next
Resume [Week 3 acquisition](../week3-lines-2026-09-22.md) when authentic pool lines are available. Choose the next bounded backlog unit from ROADMAP's recommended execution order; use the specific blockers above to avoid repeating discovery. Preserve the existing prediction card, tiebreaker and six untracked experiment records.

## Open
Week 3 pool lines remain unavailable. The additional fix for historical-only injury snapshots and empty failed capture directories is tracked in [injury-current-season-capture](injury-current-season-capture.md). The broader Friday/Sunday acquisition workflow remains open separately from [card-designation presentation](lead64-card-designations.md).
