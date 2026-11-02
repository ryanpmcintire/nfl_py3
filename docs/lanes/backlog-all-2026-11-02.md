# Backlog: finish all 57 open rows (2026-11-02)

## Goal
Owner asked to finish every non-done ROADMAP row (57 of 98 on 2026-11-02, `scripts/roadmap_inventory.py`). Done means each row's own definition of done is met and its status reconciled in ROADMAP.md; rows that need future games keep their forward log running with the exact date they can be scored.

## State
- Triage complete: docs/lanes/backlog-triage-g1..g6-2026-11-02.md (one line per row).
- CLOSE-NOW candidates (to verify against full row scope): LEAD-66 LEAD-86 LEAD-98 LEAD-43 LEAD-55 MOD-19 ENG-45 MOD-26 PER-09 PER-10 MOD-18 SKY-07 SKY-05; POL-08 fold into POOL-01; LEAD-61 source gap.
- FORWARD-ONLY (score after games): MKT-08 POL-10 LEAD-29 LEAD-53 LEAD-64 LEAD-65 LEAD-75 LEAD-99 ENV-02 MOD-07 MOD-23/24/25 remainders, MOD-27 pool-relative log; mostly after Week 18 (Jan 2027).
- OWNER-PHYSICAL: PER-03 (Sportradar key or NFL.com consent), POL-05 (pool field size and prize table), OPS-06 (friend-server lingering + deployment approval), SKY-02 (licensed tracking feed).
- NEXT-UNIT queue: SIM-11 ST latent; LEAD-28+PER-07 coordinators; SKY-01; LEAD-76; LEAD-89; SKY-04; SIM-06; SKY-03; SIM-07 (as SIM-09 sub-unit); SIM-09 integration (heavy, after SIM-11); SIM-08 declaration; OPS-02 inventory script; POOL-01 standing-aware; POL-11 ledger check; POL-12 totals (after Week 9 lock); MOD-17 lattice ledger (pinned, bundle); MOD-27 day-level transaction dates; LEAD-59 sweep as nightly job; LEAD-80/81 forward capture specs; MOD-09/10 dependency checks; UI-20 per-session improvement.
- Wave 1 launched 2026-11-02 ~16:30 ET: ROADMAP reconcile (sole ROADMAP.md writer), SIM-11 ST latent (only heavy job, 4 workers), LEAD-28+PER-07, SKY-01, LEAD-76, LEAD-89.
- Results so far (committed ac26999b, 470093ba): reconcile flipped 10 rows done (LEAD-43 LEAD-55 LEAD-86 LEAD-98 MOD-18 MOD-19 MOD-26 ENG-45 PER-09 PER-10) and 3 blocked/folded (LEAD-61 POL-08 SKY-07); inventory 51/98 done, 47 remaining. LEAD-66 kept (close-proxy read, exact log score open); SKY-05 kept (no approval gate on disk). LEAD-76 unit 2 recorded unresolved (+1.1e-5 log loss, P+ 0.979, proxy 2015-2021 openers): rerun on artifacts/opener_evaluation/20261101T151614Z/per_game.parquet (2020-2025) queued for a free core. SKY-01 table + fitted term recorded unresolved (Brier -0.000178, P+ 0.33); next drift control, snap weighting, Week 9 forward scoring. LEAD-28: overlay reads current snapshot and fires (Wk8 BUF JAX LAC WAS; Wk9 NO NYG HOU); no fix needed.
- Running: SIM-11 ST latent (4 workers), PER-07 fitted term (script running), LEAD-89, SIM-08 declaration, OPS-02 inventory, POL-11 ledger check.
- Root owns commits, pushes, publication. Week 9 lock Tuesday 2026-11-03 12:20 ET outranks this lane.

## Tried
- 2026-11-02 16:20-16:22 ET: six pool_move_log_* jobs MANUAL-DRY-RUN OK (no games before deadline; nothing written).
- Coordinator snapshots are current (coordinators_tue caught up 2026-10-31 -> data/raw/coordinators/20261031T112746206891Z).

## Next
1. On each return: verify gating claims, update this State, refill the slot from the queue.
2. Commit verified lanes and ROADMAP edits at a clear point; push.

## Open
- None needing the owner beyond the four OWNER-PHYSICAL items.
