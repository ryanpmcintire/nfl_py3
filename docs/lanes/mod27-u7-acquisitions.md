# MOD-27 U7 mid-season acquisitions

## Goal
Decide whether mid-season acquisitions re-weighting (acc-u7 survivor 3) is novel, and if so measure it on the model-alone fair opener test (802-701, 53.36%, 1,503 decisive 2020-25 games) and as a fitted card term.

## State
- Novelty check DONE 2026-11-02: REPEAT. No measurement run, no outcomes read, no code written.
- Evidence (read): `src/nfl_ats/transaction_flag_features.py` lines 356-476 already parse PFR slugs for confirmed acquisitions (Sep-Dec, `ACQUISITION_RE`, speculative and pick trades excluded), find the acquired player in `snap_counts`, take the trailing snap share before the acquiring week, and flag the team's first `DEADLINE_INTEGRATION_GAMES` = 3 games. That is the snap-weighted acquired-player input this survivor proposed.
- Evidence (read): `registry/weak_signals/deadline_integration_drag_on_production.json` (LEAD-23), unresolved_below_power, +0.877 accuracy points, interval [+0.216, +1.766], 466 paired rows, 2020-2021 window only; ROADMAP lines 410-412: 70 flagged games, 25 acquisitions, P+ 0.883 in the earlier read.
- Only structured source is PFR URL slugs (headline text, direction inferred from slug). No player-level roster-transaction table is ingested; `waiver_claim` is 406 slug rows (docs/transaction_wire_battery.md line 124).

## Tried
- Grep ROADMAP.md, docs/*.md, docs/lanes, registry/weak_signals for acquisition/waiver/trade/transaction; read the code and registry entry above.

## Next
- Not a new family. The open item is to extend LEAD-23 forward, not rebuild it: rescore the existing flag on 2022-2025 (the registry window is 2020-2021 only) as one fitted term inside the four-term card probability, leave-one-season-out, plus on the model-alone test. Population is about 25 events per two seasons, so expect unresolved_below_power. Record under the existing family, not mod27_acquisitions.
- A genuinely different input would need a new source (nflverse weekly rosters by team, to catch waiver and street pickups the slug parser misses). Check whether that is ingested before declaring.

## Open
- Whether the owner wants the 2022-2025 extension of LEAD-23 or the roster-source build.
- Nothing recorded with `weak-signals record` (no new cell was measured).

## Declaration 2026-11-02 (written before any outcome read; agent L23)
- Flag: derive_deadline_integration_drag_features unchanged (DEADLINE_INTEGRATION_GAMES=3, HIGH_SNAP_SHARE_THRESHOLD=0.5, window months 9-12, signed +1 away/-1 home). Fifth fitted term added to the four-term ridge (FIT_RIDGE) by the production fit/design code, LOSO on 2020-25, opener graded.
- Primary cell (1 look): five-term vs four-term LOSO accuracy diff, all 2020-25, season-week block bootstrap. Secondary looks (declared): 2022-25 subset, 2020-21 subset, strict-chronology sensitivity (drop flags whose month-end report date is not before the game's Tuesday opener), 2026 wk1-8 frozen apply. Total 5 looks; coefficient per fold reported.
- Leak concern declared up front: report_end is month-end, so a game within days after month-end can have an opener posted before the report date; assert and report violations.
- Serve bar: beats the four-term served probability on opener grade through the played card, with the 2026 forward apply as a check only.

## Results 2026-11-02 (agent L23; scratch C:/Users/Ryan/AppData/Local/Temp/claude/F--Repos-nfl-py3/352ecbdc-246f-44c0-9f81-c287a6ef4436/scratchpad/agentL23/: flag.py flag26.py score.py results.json)
- Flag unchanged. Flagged decisive games: 2020 2, 2021 0, 2022 7, 2023 10, 2024 12, 2025 21 (52 total); 2026 wk1-8 none (both acquisitions are Oct, first flagged game is wk9: HOU, SF). Registry 2020-21 events (25) do not reproduce on the current transactions index (2020: 1, 2021: 0).
- Chronology assertion FAILS for 15 of 81 team-game flags (month-end report date not before the Tuesday opener); 2020-25 strict keeps 43 flagged games.
- Four-term baseline reproduced 833-670. Fifth term, LOSO: unchanged flag 831-672, diff -0.13 pts [-0.53, +0.20] P+ 0.185, 12 changed sides 5-7 vs 7-5, LL 0.68746 vs 0.68719; 2022-25 -0.19 [-0.75, +0.29] P+ 0.191. Strict: 835-668, +0.13 [-0.40, +0.67] P+ 0.647, 22 changed 12-10 vs 10-12, LL 0.68718. Coef per fold all positive (0.05-0.34 unchanged; 0.14-0.37 strict). In-sample acc 0.5536 vs OOS 0.5529.
- Recorded 3 cells under deadline_integration_drag_on_production (total_signals 8008). Reading: not served; does not beat the four-term served probability on the opener grade.
- Next: none for this flag unless the transaction index is rebuilt to give day-level dates (month-end is the leak) and 2020-21 events restored.

## Orchestrator check (2026-11-02)
- The deadline-integration flag is not in the served model: it appears only in challenger feature sets (constants.py football_weak_stack_deadline_drag and siblings; deadline_drag_challenger.py); the active model uses football_weak_stack. The month-end report-date leak (15 of 81 flags at or after the opener) therefore affects research results only, including the 2020-21 LEAD-23 cell (+0.877 pts), which should be read as possibly leak-inflated until rebuilt with day-level dates.
- Not served: strict variant 835-668 vs 833-670 (P+ .647), unchanged flag 831-672; about 50 flagged games in six seasons.
