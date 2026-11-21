# LEAD-110

## Goal
Market absorption of the wording read (ROADMAP LEAD-110): does the read's home-minus-away availability difference predict the line move from the read to the pick deadline, and does it say anything about the cover at the frozen Tuesday pool line beyond the served probability, which already holds the move term. Research only; one fitted probability selects every side, nothing here changes a pick.

## Declaration (written 2026-11-14 before any join of moves or outcomes)
Family `lead110_market_absorption`, 8 looks per run. x_read = sum of group weight x (1 - read P(plays)), home minus away, spread points, positive = home healthier; weights from `artifacts/lead110_absorption/report.json` (OLS 2011-2025, 3680 games). x_ladder = same with the designation ladder; x_extra = x_read - x_ladder. Read time = forward file `recorded_utc`. Line at read = each book's last quote at or before it; deadline line = same book's last quote after the read and at or before min(kickoff, Sunday 4 PM ET); medians over books quoted at both; store `data/market/raw/*/quotes.parquet`. Pool line `data/splash/2026_weekNN_*.json`. Served probability = last `clv_ledger/published_picks.parquet` row at or before the deadline. Looks: a1 post-read move on x_read; a2 on x_extra, ladder held; a3 ladder control; b1 x_read as one added term to the served logit; b0 intercept-only control; b2 ladder-term control; c1 pool-line-to-deadline gap on x_read; c2 pool-line-to-read gap on x_read. Game bootstrap 2000, permutation of x within week. Cells have fixed names and are cumulative: re-record with `--replace` each run.

## State (2026-11-21 11:16 local, Weeks 9-10; numbers measured this session)
Command `scripts/lead110_market_absorption.py` (unchanged) -> `artifacts/lead110/{report.json,games.parquet,games.csv,registry_batch.json}`, exit 0. It read the Week 10 scored parquet written by LEAD-106 this session (ladder now weeks 1-9).
- Games: 121 with reads; used 29 (Week 9 15, Week 10 14). Excluded 91 Weeks 2-8 (read after the deadline) and Week 11 IND@HOU (no scored ladder yet). Move looks 27 (CIN@ATL, JAX@BAL Week 9 have no quote after the read); cover looks 29, leave-one-week-out (2 folds).
- Week 10 timing: read Sat 21:30Z, line at read 145 min old (WAS@NYG Thursday 160), deadline quote 33-466 min before the deadline, 4 books (NE@DET 3). 9 of 14 lines moved after the read (5 of 13 in Week 9).
- (a) Move per point of read gap: a1 +0.018 [-0.093, +0.225] P+ .61, permutation p two-sided .84; a2 read minus ladder +0.014 [-0.78, +2.95] P+ .54, p .93; a3 ladder control +0.018 [-0.097, +0.257] P+ .58, p .85.
- (c) c1 pool-line-to-deadline gap -0.210 [-0.574, +0.339] P+ .22, p .21; c2 pool-line-to-read gap -0.229 [-0.600, +0.313] P+ .20, p .135. Week 9 alone was +0.31 and +0.34; the sign reversed with Week 10.
- (b) Served card at the pool line 12-17, log loss .718 vs coin .693. b1 read term: decisive 2-2, record unchanged 12-17, held-out gain -0.003 [-0.029, +0.020] P+ .42, matched-null 74th percentile (one-sided p .26), in-sample +0.004 (gap .006), coefficient +0.101 logit/pt (folds -0.002 to +0.154); detectable gain at this size about .035 (2.8 SE, inferred). b2 ladder control: decisive 3-0, 15-14, +0.001 [-0.014, +0.017] P+ .58, p .96. b0 intercept control: decisive 13-3, 22-7, +0.142 [-0.057, +0.329] P+ .92, in-sample +0.146 (gap .004), coefficient -1.18 (folds -1.31 to -1.06): away sides covered 22 of 29.
- Served side changed after the read in 1 of 29 (Week 10 CAR@NO, line moved 3.5 toward the away side, served P(home) .551 to .496).
- Registry: 8 cells `lead110_fwd_*` re-recorded with `--replace`, all unresolved_below_power. `weak-signals pool --league nfl --effect-units accuracy_points` exit 0, no coherence problems.
- Implication (inferred): over two weeks the line does not move toward the read after Saturday, the read adds nothing measurable to the served probability at the pool line, and the Week 9 hint that the read was already priced did not repeat. The read stays out of the card.
- ROADMAP LEAD-110 update: rerun through Week 10 (29 games, leave-one-week-out): post-read move slope +0.018 P+ .61, read as an added cover term -0.003 log loss P+ .42 (decisive 2-2), pool-line gap slope -0.21 P+ .22; 8 cells unresolved_below_power. Next: rerun after Week 11 is scored.

## Tried
- Unit 1 (11-03): `scripts/lead110_absorption.py`, 91 games Weeks 2-8 (reads recorded after the games); family `lead110_absorption`. Sign repair 11-14 (store lines positive when home favoured; `:166` and `:209` fixed), 4 cells re-recorded with `--replace`; ladder loading +1.71 P+ .995.
- Unit 2 (11-14): Week 9 only, 15 games, leave-one-game-out: a1 -0.030 P+ .35, c1 +0.310 P+ .94, b1 -0.112 P+ .028, b2 -0.181 P+ .0005, b0 +0.036.

## Next
1. Rerun after each scored forward week (after LEAD-106 scores Week 11): `scripts/lead110_market_absorption.py`, then `record --batch artifacts/lead110/registry_batch.json --replace`, pool once.
2. Scheduler owner: a line capture at the forward read time and inside each deadline window, and a midweek forward view so a read precedes the Tuesday-to-Saturday move.
3. Keep position and ladder probability for unverified forward rows so team sums are a census (Week 10 used 298 of 345).

## Open
- For the orchestrator, not adjudicated here: b0 intercept-only control +0.142 P+ .92 with a .004 in/out gap (22 of 29 away covers, served card 12-17 at the pool line) is a home/away tilt in the served card on these games, not a read effect; Week 9's b1/b2 whole-interval-below-zero readings did not persist (now P+ .42 and .58).
- Weights are spread points per absent player, so slope 1.0 is only a rough mark for full absorption; the Tuesday pool line already holds injuries known on Tuesday.
- Rules: zero crossing closes nothing; one fitted probability selects the side; parameters out of season; every cell a look; verdicts through `weak-signals record`.
