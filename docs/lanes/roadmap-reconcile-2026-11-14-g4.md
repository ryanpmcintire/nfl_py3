# roadmap-reconcile-2026-11-14-g4

## Goal
Transcribe the recorded state of 26 ROADMAP rows still marked not started (UI-21 to UI-37, POOL-02 to POOL-10) from their 2026-11-03 lanes into a verdict file, so the orchestrator can update ROADMAP.md. Read-only; no analysis run and no new research verdict issued.

## State (2026-11-14 14:41 ET)
- Verdict file: `tests/scratch/reconcile_2026_11_14/g4.json` (26 objects, parses, plain ASCII, longest note 43 words).
- Counts: done 13, in_progress 12, blocked 1, forward_only 0, not_started 0.
- UI done (12, each phrase found in the generated site by Grep): 21, 23, 24, 25, 26, 27, 28, 31, 32, 34, 35, 36.
- UI in_progress (4, live but a declared piece is unbuilt): 22, 29, 30, 37. UI blocked (1, not live): 33.
- POOL done (1): 03. POOL in_progress (8): 02, 04, 05, 06, 07, 08, 09, 10.
- Registry families confirmed present under `registry/weak_signals` (measured, Grep): pool02_entrant_skill, pool03_coinflip_ownership, pool04_field_calibration, pool05_ownership_v2, pool08_hook_field_share, pool09_hook_value_slope, pool10_best_pick_hooks. POOL-06 and POOL-07 have no record (as their lanes say).
- Every recorded POOL result is unresolved; none is called settled here.

## Tried
- Row text from ROADMAP.md, all 26 lanes, `ui20-2026-11-03.md`, `pool-standing-refresh.md`.
- Live check by Grep with matched text only against docs/index.html, history.html, findings.html (files are single-line; line counts are useless). Builders confirmed in src/nfl_ats/board_*.py.
- Week 9 field files landed 2026-11-12 (measured, file times in data/splash/field), so every lane step worded as a rerun once Week 9 lands can run now. That is why no row is forward_only.

## Next
UI rows not fully live, by how much a pool player would use them:
1. UI-33: not live. Needs the injury-scenario producer stored for the active model first (`docs/lanes/done/injury-scenario-producer.md`).
2. UI-30: add lines for a number that moved without a side change (cover-chance change beyond quote_step) to `_changed_strip_html`.
3. UI-22: the news-found sentence has never rendered; exercise it on one real moved game.
4. UI-37: add second road game in a row and time zones crossed to `board_schedule_facts.py`.
5. UI-29: capture the top entry's picks after Week 8 and add net points each way.

POOL rows with a next unit (detail in the JSON `next_unit`):
1. POOL-08: rerun with Week 9, add the fitted hook coefficient to the ownership proxy in the POOL-01 replay.
2. POOL-09: run the row's declared unit (lattice push mass as covariate), add Week 9.
3. POOL-05: rerun with Week 9, team effect on the line-proxy residual.
4. POOL-10: rerun with Week 9, hook term in a favourite-adjusted Best Pick share model.
5. POOL-04: week-level mean misses against pick-distribution shares.
6. POOL-07: fitted per-game leverage term, then a registry record in an accepted effect unit.
7. POOL-06, POOL-02: Chrome reads by root (tiebreak guesses after lock; per-entrant per-game picks).

## Open
- Judgement calls the orchestrator may override: an unrecorded phone-width look (UI-31, UI-36) and the unrecorded zero-mismatch check on swing lines (UI-23) were not counted as owed units; UI-24 shows no market-favourite band though the row title names one and no lane lists it.
- UI-33 is `blocked` because the scenario artifact does not exist; building the producer is agent work, not an owner action.
- POOL-02 score date 2027-01-12 is inferred from the weekly cadence, not read from a schedule file.
- POOL-07 has no registry record; it cannot be written up as settled until it has one.
- UI-35 built three bands (row says four); UI-32 covers 3 and 7 only.
- Pool script commands are not recorded in the pool lanes; the invocation pattern is read from `pool-standing-refresh.md`.
