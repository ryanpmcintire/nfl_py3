# Roadmap reconcile 2026-11-14, group 3 (LEAD-100..112, ENG-48..60 less ENG-56)

## Goal
Transcribe the recorded state of 25 rows that ROADMAP.md still marks not started, so the orchestrator can update the roadmap. Read-only: no analysis, no new research verdict, no ROADMAP or registry edit.

## State
Done 2026-11-14 14:42 ET. Verdicts: `tests/scratch/reconcile_2026_11_14/g3.json` (25 objects, parses, ASCII, notes at most 45 words).
- in_progress 17: LEAD-100, 101, 102, 103, 104, 106, 109, 110; ENG-48, 49, 50, 51, 53, 54, 55, 58, 60.
- done 4: LEAD-111, ENG-52, ENG-57, ENG-59.
- not_started 4 (stub lanes): LEAD-105, 107, 108, 112.
- forward_only 0, blocked 0.
No unresolved result is described as settled; every research cell in these lanes is unresolved_below_power except 2 LEAD-66 RPS cells in the ENG-51 rerun (wrong sign, whole interval on that side, as the lane recorded).

## Tried
- Read all 25 lanes, the 25 row texts (head and tail), and State and Next of lead99-llm-read.md.
- Registry (grep, measured): cells exist in registry/weak_signals for eng50_online_stacker, eng51_open_pool_rerun, lead104_revision_path, lead106, lead109_position_value, lead110_absorption, lead111_phrase_ablation. None for LEAD-100, 101, 102, 103.
- Scheduler (grep, measured): capture_scheduler.py has lead101_delta_read_* (line 793), lead106_forward_* (695), schedule_snapshot_* (772); scheduler_log shows OK runs 11-05, 11-07, 11-08, 11-12. No job for forward_ledger, forward_monitor or the stacker; their artifacts were last written 2026-11-03 16:09.
- Guards (grep, measured): assert_opener_input in scripts/opener_input_guard.py and scripts/mod24_more_games.py; pool --base and by_base in src/nfl_ats/cli_commands/registry.py and weak_signals.py.
- Forward files on disk: artifacts/lead101/runs (Week 9 Thursday, Sunday; Week 10 Thursday), artifacts/lead106_injury_language/forward (same three), artifacts/per03/revisions_2026.parquet, daily data/raw_schedules through 11-14.

## Next
In-progress rows, most valuable for the weekly picks first (commands in g3.json next_unit):
1. ENG-49 then ENG-48 then ENG-50: refresh the matched forward ledger with Week 9, rerun the monitor and the stacker (two prior weeks now), add the weekly Job.
2. LEAD-106: score the Week 9 forward wording reads against snaps; this unlocks LEAD-109 and LEAD-110 reruns and the LEAD-108 first unit.
3. LEAD-101: write the scorer for the logged Week 9 delta reads, with persistence and market-excluded baselines.
4. LEAD-104: one rerun of scripts/lead104_revision_path.py now that Week 9 has Tuesday and deadline views.
5. LEAD-102 record the unit 1 look; LEAD-103 repeat blocks to 3 reads; LEAD-100 more repeat reads before kickoff.
6. ENG-55 unit 3 (rendered hook badges), ENG-58 slot and rest count for 34 moved games, ENG-60 2026 neutral check, ENG-54 unit 2, ENG-53 MOD-43 rows, ENG-51 unit 2b (root deferred to the next pinned bundle).
Time-critical stub: LEAD-105 first read must be logged before the Week 10 last game (Monday 2026-11-16, inferred).

## Open
- Lane vs row differences: ENG-48 built a betting e-process, not the mixture martingale, replayed Weeks 7-8 not 4-9, bootstrap comparison not done. ENG-53 omits MOD-43 and has no weekly refresh. LEAD-102 and LEAD-103 used the Week 9 Tuesday packet, not the Thursday deadline view. LEAD-104 had Week 7 only. LEAD-109 fit leave-one-week-out (one season of reads). LEAD-111 declared 17 looks, recorded 15 cells.
- ENG-60 lane still lists the BUF-SF and JAX-CHI sign question as open; the ENG-59 lane answers it (line moves, opener correct).
- ENG-58 lane Next still holds the first-unit text; ENG-50 unit 1 numbers before the 16:12 correction are void.
- The 25 source lanes were last written 2026-11-03 and were not edited here.
