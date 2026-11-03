# MOD-48

## Goal
MOD-41 serving bar and forward-weeks plan: a pre-declared decision rule and the week it can first be met (ROADMAP MOD-48).

## State
Added 2026-11-03 (backlog expansion 8), not started, nothing run. Mechanism: MOD-41 residual load is frozen (coefficient -0.0439) and scored forward weekly, but no rule says what matured result serves it. Fitted term: the frozen term enters the one served probability only if the pre-declared bar is met; the bar is a serving rule, not a closing rule, and a miss closes nothing.

## Tried
Novelty check only. Nearest: ENG-48 (generic anytime-valid monitor), ENG-50 (online stacker), MOD-41 lane unit 3. Differs: a MOD-41-specific pre-declared serving rule with a power timeline, not the generic monitor or the combination.

## Next
First unit for one agent: Before more weeks score, write the bar: paired forward log-loss gain against the served probability with probability positive and decisive record, then simulate from the 2020-25 per-week gain variance how many forward games and which week the bar is reachable under the LOSO-sized effect (.004) and under half of it, and file the bar with the freeze hash.

## Open
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the served base (out_of_season_home_probability from the latest artifacts/pick_probability run, with raw-model, intercept-only and line-move-only controls) on the same market and simple-model baselines. Look count and family declared before any sign is seen.

## Unit 1 done 2026-11-03 17:38 local (plan only, nothing served, nothing recorded)
Script scripts/mod48_power.py, artifact artifacts/mod48/power.json (2000 draws, seed u2.PERM_SEED, 1142 usable 2020-25 games on the served base, 13.76 games/week, 137.6 games left in 2026 weeks 9-18).

Declared serving bar (a serving rule, never a closing rule; a miss or zero crossing closes nothing):
- Statistic: per-game paired log-loss gain of mod41_residual_load_served against the played card (ledger served_p vs cand_p, home_cover), the ENG-48 diff.
- Evidence: scripts/forward_monitor.py e-process, alpha .05, threshold e >= 20 (log e >= ln 20), truncation .5 (module constants), candidate mod41_residual_load_served added to its CANDIDATES. Anytime valid, so peeking weekly costs nothing.
- Minimum games: none beyond the monitor; evaluated at completed weeks only. Serving also requires forward mean gain > 0 at the crossing, with decisive record, probability positive and the one-game gain interval reported first.
- Stopping: serve at the first completed week with e >= 20; continue scoring otherwise with no end date. Season boundary does not reset the process.
- Retire (closing grounds only): resolved wrong sign (whole forward interval below zero) or zero split-half reliability, per AGENTS.md; a non-crossing, or an interval containing zero, retires nothing. Unresolved goes through weak-signals record.

Power (measured, resampling usable 2020-25 per-game gains; the term was shrunk toward the base for the half arm since gain is not linear in the term):
- Full (gain .0054/usable game = .0040 over all 1503): per-game sd .107; analytic growth rate .00213/game, 2345 games = 170 weeks. Simulated: 12.8% cross within 1142 games, 5.1% within 2026's remaining 10 weeks; crossers only (lucky paths) median 246 games (17.9 wk), 10-90% 16-907 games (1.2-66 wk).
- Half (gain .0027): sd .030; analytic 777 games = 56 weeks. 36.4% cross within 1142 games, 9.4% within 2026; crossers median 401 games (29 wk), 10-90% 28-1001.
- Null false crossing: boundary null (zero mean gain, term scaled 1.95x) 3.4% within 1142 games, 2.65% within 2026; permuted-term null (mean -.0059) 1.75%. All below alpha .05 (inferred: monitor valid).
Reading: the bar is not reachable in 2026 in most worlds (5-9%); expect 2027-28. The e-process is poor at a signal this small relative to per-game sd; the forward gain is also checked as a plain weekly interval, but that is not a serving gate.

Serving steps (not done; pinned-source guard applies): 1) scripts-side refit of the fifth term on the served logit LOSO (already frozen: resid -.0439, served logit .921, hash 12bdeb20); 2) add a fifth term s_resid in src pick_probability.py (pinned, owner-gated) with LOSO coefficient and the 4-term base unchanged for other terms; 3) rerun pick-probability to get a new artifact and reproduce 829-674 base plus the fifth-term LOSO result; 4) re-enrol independent validation as v8 days after v7 same session, per pinned-source rule; 5) update served-card builder so Week N card, board, alert use the same ledger (one ledger, every surface); 6) publish-board with active model id, regenerate, review diff, push.
- 2026-11-03 17:38 ET (root decision on MOD-48): the ENG-48 anytime-valid bar is mismatched to this effect size (about 170 weeks to cross at full effect; 5% chance within 2026), so it cannot be the serving gate. Serving will be decided on the out-of-season served-base evidence (unit 6) plus robustness: MOD-49 construction family (running), a Week 9 range check (Week 9 term values sd 6.46 v historical 4.95, range -10.5..13.9; they drive 6 of 15 side differences; bye weeks suspected), and MOD-52 stability by season/week block/spread. Not served for Week 9.
