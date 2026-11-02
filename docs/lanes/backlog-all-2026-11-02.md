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
- SIM-08 unit 3 declared (docs/sim08_market_anchored_declaration.md, 9a0a951d). LEAD-89 unit 3 blocked: scripts/build_loso_upstream.py pin is model b578fbea, active is 9b7b4166 (re-activated 2026-11-02 12:10 ET).
- Done since (pushed to 0f021240): OPS-02 inventory (64.2 GiB; 33.6 GB unreferenced stamped runs, largest artifacts/mod25e3 17.2 GiB). Decision: no pruning; F: has 1,037 GB free of 1,863 GB (measured 2026-11-02), so retention stays inventory-only and nothing is deleted. POL-11 forward-only: Weeks 2-8 refreshed sides 20-12 v Tuesday 12-20 (Tuesday-line grade); single-book baseline guard sits in pinned sharp_book_movement_features.py:136-139, bundle with the next pinned edit. SIM-11 FG-make latent not adopted (10/14 v 11/14 seed 11); next lever per lane: transmission fix or weeks 10-18 x-builder.
- Uncommitted generated output (CURRENT_PREDICTIONS.md, tiebreaker.json, docs/*.html) is another session's 2026-11-02 17:17 UTC Week 8 republish; left for the Tuesday lock publication to supersede.
- Done since (pushed 98ed8360): registry None-string bug fixed (a287ea9c; load and batch kept null as text 'None', 31 strays cleared, none on terminal rows). MOD-27 day-level deadline dates: leak 3 games; repaired term -0.40 pts P+ 0.029, unresolved, not served. PER-07 fitted term -0.53 pts P+ 0.008 OOS but in-sample coef ~0 (P+ 0.49): fitting noise, unresolved; no 2026 in-season OC/DC changes yet. SKY-04: same-hour diffusion 72%, provider lag 17 s, term unresolved; tick data needed for finer lag (no free source). LEAD-89: LOSO upstream re-pinned (loso_upstream/20261102T213642884071Z), stage features carry last known report.
- Running: SIM-08 anchored build + smoke, POOL-01 standing-aware, LEAD-89 stage runs + scoring, LEAD-76 declared 2020-2025 rerun, SIM-11 next transmission mechanism (3 workers).
- Done since: LEAD-76 closed ✅ (e4a48a05; inventory 52/98). SKY-03 matchup interactions reliability r .07-.34, fitted term Brier -0.00043 P+ 0.03, decisive 14-24, unresolved (32e08ab7). SIM-11 levers inadmissible (da02856b); frontier data check running. POOL-01 standing-aware unresolved (15de7a26); Week 9 board read now captures pool observables.
- 2026-11-02 ~17:00 ET: killed orphaned `find / -name game_features_weak_stack*.parquet` (PID 12832, started 16:29 by the first LEAD-89 agent, parent gone). Total CPU stayed ~67% with the Windows System process at ~10 cores and zero disk activity while research pythons used 1 core each; launch no new compute until System load subsides. Packets must forbid filesystem-wide `find /` (search named directories only).
- Still queued: SIM-08 full run (after SIM-11 run), SIM-06 head, SKY-03 matchup, SIM-07 personnel (SIM-09 sub-unit), SIM-09 integration (heavy), SKY-05 approval gate, LEAD-66 open parts, LEAD-59 officials sweep as a nightly job, LEAD-80/81 forward capture specs, MOD-17 lattice ledger (pinned, bundle), MOD-10 gate (player-state reliability .27-.44).
- Queue when a core frees: (a) LEAD-89 re-pin: rebuild LOSO upstream on the active model (new artifact, new look budget), and fix the Tuesday arm, which zeroes all injury columns; it must use what was known Tuesday noon (prior-week final designations, IR/season-ending status, QB starter news timestamped before Tuesday), since zeroing biases the contrast toward waiting. (b) LEAD-76 declared rerun on artifacts/opener_evaluation/20261101T151614Z/per_game.parquet (2020-2025). (c) SIM-08 full run after SIM-11 finishes.
- Root owns commits, pushes, publication. Week 9 lock Tuesday 2026-11-03 12:20 ET outranks this lane.

## Tried
- 2026-11-02 16:20-16:22 ET: six pool_move_log_* jobs MANUAL-DRY-RUN OK (no games before deadline; nothing written).
- Coordinator snapshots are current (coordinators_tue caught up 2026-10-31 -> data/raw/coordinators/20261031T112746206891Z).

## Next
1. On each return: verify gating claims, update this State, refill the slot from the queue.
2. Commit verified lanes and ROADMAP edits at a clear point; push.

## Open
- None needing the owner beyond the four OWNER-PHYSICAL items.
