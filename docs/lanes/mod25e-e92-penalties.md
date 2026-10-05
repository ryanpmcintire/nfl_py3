# mod25e-e92-penalties

## Goal
Test a human mechanism for the persistence gap: discipline. Does a team that
over-performs early commit more (or its opponent fewer) penalties later, moving
field position toward the outplayed side? Parent lane
docs/lanes/mod25e-generator-fidelity.md (E88: B later start-yardline slope on A
H1 residual real -.133 vs sim -.056 yd/pt).

## State
2026-10-05 E92 ran. scripts/mod25e_e92.py (patches adj2.sim_drives and
late.real_play_drives to add penalty yards by unit; sim rows mapped via idx to
pool penalty fields). Output artifacts/mod25e3/e92/e92.txt. 15 looks.

## Tried
Real 2009-17 vs sim crHpqokgndecsmfwtjo2as2 s11-13, H2 penalty yards toward A
on A H1 residual, quarter x A-lead-band cells demeaned, game bootstrap (300):
- Real net +0.085 yd/pt [-0.021,+0.182] ppos .93; A_def +0.079 [+0.027,+0.135];
  B_def +0.054 [+0.007,+0.111]; A_ret +0.023 [+0.009,+0.037]; A_off/B_off
  negative, intervals include 0. Direction: penalties favour the over-performer
  (amplify persistence), opposite to pull-back.
- Sim carries almost no scrimmage penalty yards (off .10, def .15 per H2 drive vs
  real 1.99, 2.04); kick units match (cov .12/.12, ret .28/.29). Sim net +0.017.
- Start-yardline channel (prior A drive penalty displacement of B start): real
  slope -0.0029 [-0.012,+0.005] ppos .28 of total -0.1326; sim +0.0027; gap through
  penalties +0.0055 of +0.0771 (7%, interval spans 0).
- E92b (measured, scripts/mod25e_e92b.py, artifacts/mod25e3/e92/e92b.txt): the
  .10/.15 vs 1.99/2.04 gap is a MEASUREMENT ARTIFACT, not an engine defect. Pool
  (346552 rows) aligns exactly with E92 rebuilt frame (gid/pid mismatches 0,
  penalty flag mismatches 0, yards_gained diff 0). No_play rows are 6.17% of pool,
  98% penalty, mean abs yards 7.64. But sim play-file idx points at penalty rows
  only 1.0% of plays (pool 8.06%), while sim code==6 plays are 5.94% of plays with
  mean abs yardline move 7.3 (pool 7.64): the sim does carry penalty yards (about
  0.435 vs 0.471 abs yd/play, -8%); idx on code-6 rows is not the pool row that
  supplied the yards (F3 replay path), so the idx join drops them. E92 sim
  scrimmage penalty channel (and the 7% start-yardline gap share) is void;
  kick-unit match stands. Redo needs penalty yards from sim yl movement on code-6
  rows by side, not idx. Not done.
Verdict: not a carrier of the gap (wrong direction, ~2% of slope);
unresolved_below_power, not refuted. Not recorded via weak-signals yet.

## Next
Orchestrator: weak-signals record of E92. Open follow-up: why sim scrimmage
penalty yards are ~5% of real (no_play rows drawn but yards likely not applied) is
a fidelity defect independent of the persistence gap; real kickoff penalties not
measured (kickoffs excluded from drive frame).

Orchestrator 2026-10-05: no E92 redo. The real-side penalty channel is ~2% of the
real slope and points the wrong way, so a corrected sim number cannot move the gap
materially. Lesson: play-file idx is unreliable on code-6 (no_play) rows; E91 2b
filtered to pool-consistent flip rows (97%), unaffected.

## Open
Fitted fix form if wanted: none supported for persistence.
