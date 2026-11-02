# ROADMAP reconcile 2026-11-02

## Goal
Reconcile CLOSE-NOW candidates from docs/lanes/backlog-all-2026-11-02.md against each row's own definition of done; sole ROADMAP.md writer this wave.

## State
Edited 13 rows (status cell + dated note, last column). Inventory after: 51/98 done, 47 remaining, declined_or_blocked=4.
- Flipped to done: LEAD-86, LEAD-98, LEAD-43, LEAD-55, MOD-19, ENG-45, MOD-26, PER-09, PER-10, MOD-18.
- Flipped to declined/blocked (❌): SKY-07 (retired), POL-08 (folded into POOL-01), LEAD-61 (free 2H source absent).
- Kept: LEAD-66 (declared 2009-2019 close-proxy second read lacks matching predictions; exact log score lacks tail support; no repair declared), SKY-05 (no owner-approval gate found on disk; rotation.py has window-size budgets only, LEAD-99 is a different scope).
- MKT-08: cell news_trigger_resolution_cover_probability -1.2235 pts, P+ 0.09425, recorded 2026-09-23 in registry/weak_signals/_no_family.json. Row stays forward-only.
- MOD-09/MOD-10: PBP-05 exists, status done, in docs/roadmap_archive.md:137. Player state is only modestly reliable (PER-09: ST split-half Spearman-Brown +0.436; RAPM offense +0.296, defense +0.268), so MOD-10 gate stays unmet; both rows kept.

## Tried
Read each full row, lane State and registry shard; applied precedent (declared measurement ran and cells recorded = done).

## Next
Root verifies, commits ROADMAP.md and this lane. Consider declaring a LEAD-66 tail-support repair and 2009-2019 prediction rebuild to close it; SKY-05 needs an approval-gate design or owner decision.

## Open
MOD-18 flip rests on its done-when sentence; its older "open next" half-point items are now carried by MOD-26/MOD-27 (inferred).

## Pass 2 (2026-11-02)
Edited 21 rows (note `reconciled (pass 2)` in last column). Inventory after: 56/98 done, 42 remaining.
- Flipped done: OPS-02 (inventory shipped; no-pruning policy, F: 1,037 GB free). Moved to in-progress: LEAD-80, LEAD-81 (collectors built, scheduler jobs after the 2026-11-03 lock).
- Kept with named remainder: SKY-01/03/04/05, SIM-06/07/08/11 (moonshot or heavy step pending), PER-07, LEAD-28, LEAD-29, POL-10, POL-11, POL-12, POOL-01, MOD-27, MOD-17, MKT-08 (forward-only or next pinned edit; earliest scoring Week 9 / Jan 2027).
- Open: SIM-07 flip depends on the control_summary.json power result.
