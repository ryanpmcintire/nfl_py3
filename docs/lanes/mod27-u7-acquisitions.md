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
