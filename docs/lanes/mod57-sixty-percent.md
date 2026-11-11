# MOD-57 - 60% against the pool line, held out

## Goal
Owner 2026-11-11: reach 60% pick accuracy without overfitting. Yardstick: every game, graded at the pool's Tuesday line, picks taken at each game's deadline. A number counts only if every parameter was chosen out of season (2020-25 leave-one-season-out) and it repeats on the enrolled 2026 forward cohort (validation v9, Weeks 10-18). Done = the served probability's held-out all-games record reaches 60%, or every unit is recorded with what it moved.

## State (2026-11-11, all measured from artifacts/pick_probability/20261111T011859Z; scratch look2.py in the session scratchpad)
- Held-out served probability, 1,503 games 2020-25: 829-674, 55.2%; mean stated confidence 54.5%. Model alone 802-701, 53.4%. Seasons with a line-move input (2023-25): 54.9, 56.8, 54.7.
- At simulated pool lines with the hook term (whole-number Tuesday lines, 1,322 rows): 57.1%; 2023-25 59.7, 56.2, 59.7.
- Where the wins are, 2023-25 (799 games), by how far the market moved after Tuesday: no move or a half point is 59% of games and the served side is 50.4% there; a full point or more is 41% of games and the served side is 62.7% (the move side alone covers the Tuesday line 61.2%).
- Arithmetic (inferred): with 41% of games at 63%, the other 59% must reach 58% for 60% overall. The binding limit is the games where the market barely moves.
- By stated confidence, held out: top 10% of games 62.7%; 0.58 and above (16% of games) 61.5%.
- 2026 real pool lines sit off the market more than simulated history: replay of the current structure Weeks 1-9 85-51, card 70-66 (fix-picks-2026-11-10.md); that replay was tuned on those games and is not held-out evidence.

## Units
1. Pool source book: lane mod57-u1-pool-source-book.md. Find the book, time and hook rule that reproduce the 2026 pool lines, rebuild 2020-25 pool lines from that book, regrade the served two-stage probability on real offsets.
   DONE (measured, unit 1): no single book reproduces the pool all season. FanDuel just after noon matched 28 of 32 in Weeks 1-2 (DraftKings 17 on the same games); FanDuel has no 2026 quotes after Week 2; DraftKings matched 11 of 14 in Week 10. A whole-number source is always moved a half point, side split 25/25 with no rule. 2023-25 archive has both books within an hour after noon on nearly every game; 2020-22 only a 09 ET snapshot. Unit 1b (lane mod57-u1b-real-offset-regrade.md) is VOID: it took games from per_game.parquet, which has the 34 opener pushes dropped, so the half-point value it measured was removed (margin on the whole number in 0.8-1.5% of its whole-number games; check tests/scratch/mod57/root_check_u1b.py). Its 55.4% level, its coefficients and its 12 registry cells are not usable. Unit 1c DONE (lane mod57-u1c-regrade-with-pushes.md; measured, tests/scratch/mod57/u1c_run.out; the 12 unit-1b cells invalidated and superseded, 12 new cells unresolved, family mod57_pool_source_book), 1,537 games with pushes kept. Held-out all-games accuracy at rebuilt pool lines, 2023-25: DraftKings lines two-stage 57.1% v stage 1 55.1% (+2.02 pts [+0.85, +3.23], P+ 1.00; by season 56.8/56.1/58.5 v 54.8/56.3/54.2; fold coefficients 0.68-0.97, all-seasons 0.805); FanDuel lines 55.2% v 55.3% (-0.12 [-1.35, +1.11], P+ .40; coefficients 0.40-0.84). 2020-22 (09 ET snapshot, no move input) two-stage is worse on both (DraftKings -2.35 [-4.24, -0.56]). Better-number side, 2023-25: mechanical half point .531/.524, book half point against a whole median .500/.524, a full point or more .592/.547. Level is 55-57%, 3 to 5 points short of 60. Served two-stage unchanged.
2. Deadline staleness: lane mod57-u2-deadline-staleness.md. Per 2026 game, age of the last quote behind the served side against its pick deadline, the move missed, and refresh slots against the remaining kickoff slots.
   DONE (unit 2, corrected by the orchestrator, see its lane): early Sunday games were already decided on quotes about 35 minutes before kickoff. Real gaps were Thursday night (Wednesday 18:00 quotes), the Sunday 16:00 lock (12:26 quotes, no later pass) and holiday, Saturday and London slots. Fixed 2026-11-11 in scripts/capture_scheduler.py: nine pre-deadline captures and refresh_last_call_sun_1535; dry run OK, daemon restarted, health current. Unit 2's own swap of deadline lines changed 1 side in 136 games, so this is fidelity, not a source of points.
3. Deadline fair line: lane mod57-u3-deadline-fair-line.md. Spread price plus moneyline at the deadline as a finer-than-half-point move, fitted through the stack, declared for the small-move games.

   DONE (measured, unit 3; scripts/mod57_u3_deadline_fair_line.py, 12 cells recorded unresolved, family mod57_deadline_fair_line): on the 837 small-move games the four-term is 430-407, plus juice 437-400 (+0.84 pts [-1.67, +3.34], P+ .73; coefficient positive in 6 of 6 folds), plus moneyline divergence 434-403 (P+ .70); on all 1,503 games both are level or worse (juice 820-683 v 829-674, P+ .23). Model alone is 441-396 on the small-move games. Not served. Live captures hold three books and no moneylines.

## Tried
- Read 2026-11-11 (done/market-move-decomposition.md): the historical move term covers 2023-25 only (799 games), three leader books, Wednesday to min(kickoff, Sunday 12:45 PM ET); 2020-22 have no move input.
- Prior work not to repeat: MOD-23/24 base-model units (team data is priced), MOD-26 hook, MOD-27 units, MOD-55 move anchor, MOD-56 book depth, LEAD-74 price change at a fixed spread, LEAD-98 Tuesday price skew, ml_spread_divergence_on_production, fix-picks u5 single lattice term, synthetic stacked offsets (mod27-u6).

## Next
- Capture FanDuel and DraftKings within minutes of Tuesday noon in Week 11 to separate the pool's source (FanDuel has no 2026 quotes after Week 2).
- Why the offset term loses in 2020-22 on both arms (pre-noon snapshot, no move input) is unexplained; measure before any refit of the 0.805 coefficient.
- Refill with the next unit that attacks the small-move games (model alone 441-396 there, four-term 430-407, juice +0.84 pts P+ .73).
- Add the Christmas-week Friday capture and refresh pairs before 2026-12-25 (mod57-u2 lane).
- Thursday before 20:15 ET: Week 10 refresh check (fix-picks-2026-11-10.md, Next 1).

## Open
- None.
