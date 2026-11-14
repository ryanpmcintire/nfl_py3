# Lane: SKY-01 news availability

## Goal
Timestamped report-to-status table; lead time of each source over official designation and line moves; test earliest-source signal as one fitted term on the four-term probability.

## State
Done 2026-11-02: scripts/sky01_news_availability.py, docs/sky01_news_availability.md, artifacts/sky01/. 6,765 rows (2024-2026). Beat leads official by median 45.6 h; beat to adverse line drift 37.4 h. Term OOS Brier gain -0.000178, p+ 0.3345, 4 decisive (2-2). 9 cells recorded, family sky01_news_availability, unresolved_below_power.

## Unit 2 (2026-11-02, measured)
scripts/sky01_drift_weighted.py -> artifacts/sky01/drift_weighted.json. Drift control (503 events, same-week no-signal games, same time-to-kickoff and role): net adverse share -5.7 pp [-10.9,-0.9] P+ 0.01; net median lead -6.1 h [-12.6,-1.7]. The 37 h market lead is ordinary drift. Snap-share weighted term (prior share as-of, 28% of out players unmatched weight 0): OOS Brier +0.000063 [-0.00149,+0.00142] P+ 0.55, 16 decisive 8-8, folds +0.018/+0.038; gap +0.000045. Inactives 2026: 64 joined, beat precedes inactive in 63, median 46.7 h. 5 cells recorded. 2026 forward score: first possible after Week 9 games (Nov 10); ~3% decisive rate means informative only near Week 18.

## Tried
Beat pre-opener out-count differential, LOSO 2024/2025 only. Inactives captures: 2026 weeks 3-8 only, not joined to line moves.

## Next
Score the term on 2026 once four-term opener probabilities exist for 2026 (Week 9 onward). Score 2026 forward after Week 9. Fix 28% unmatched snap weights (name join for rookies). Snapshot first-flag join not done.

## Open
ROADMAP SKY-01 row not updated (orchestrator owns). Drift-controlled lead is negative (not a closure: term sign stable in weighted fit).

## Unit 3 (2026-11-03, read-only audit)
Row definition (table with provenance) met 2026-11-02; fitted term unresolved_below_power, 14 cells recorded. Not DONE: no forward 2026 score exists. Earliest forward score Nov 10 (after Week 9 games); informative only near Week 18 (~3% decisive). Remaining 28% unmatched snap weights are players with no prior-season snap share (rookies), a data gap not a join bug (inferred, not re-measured). No code changed.

## Unit 4 (2026-11-03, measured)
scripts/sky01_snap_repair.py -> artifacts/sky01/snap_repair.json. Unmatched 287 of 1,028 scored out-rows: 171 veterans with earlier-season snaps but none last season or earlier this season (injury returns), 74 rookies/first snaps, 26 pfr id never in snaps (practice squad), 16 no player-table id (about 3 rows name-recoverable). Not mainly rookies, not a join bug. Repair: LOSO ridge-free OLS expectation (last observed share, gap, draft round, position group). Re-run: OOS Brier -0.000097 [-0.00122,+0.00087] P+ 0.46, log loss -0.000237 P+ 0.44, decisive 10 (6-4), in-sample +0.0000055, gap +0.000102 (declared: +0.000063, P+ 0.55, 8-8). Recorded as one look. Not refrozen. Recommend keep frozen variant; repaired adds no signal.

2026-11-03 pooling: SKY-01 forward appends and decision rule are declared in docs/forward_pooling_declaration_2026-11-03.md (frozen arm, estimator, power season, invalidation).

## Unit 5 (2026-11-14 18:21 ET, Get-Date; ENG-54 crosswalk repair): forward term sky01snap_idjoin
Why (measured): prior_share mapped snaps to players through the weekly-roster crosswalk, which matches 64.75% of regular-season snap rows 2013-25 (109,454 of 310,476 dropped; 77.7% matched from 2020). Lost rows are starters as often as kept ones (46.6% v 48.6% at half the snaps or more); tackles, guards and centres are 42.5% of lost rows v 15.0% of all. Players-table PFR id: 99.93% (218 dropped). Id plus roster name fallback (nfl_ats.players.attach_snap_player_ids with the players table, the SIM-16 key): 99.98%; 58 rows, 10 players, have no players-table PFR id and no roster name match. This supersedes the not-a-join-bug reading in Units 3 and 4: out-rows with no weight fall from 490 to 250 of 1,726.
Declaration: sky01snap_idjoin is sky01snap's procedure (snap-share weighted home minus away pre-opener beat out-signal, one logit term on the base card probability) with snap rows mapped by attach_snap_player_ids. Frozen 2026-11-14T23:12:54Z (18:12:54 ET) in artifacts/forward_terms/sky01snap_idjoin/frozen_coefficients.json: b +0.011744 logit per unit of weighted share, pooled 2024-25 fit (beat posts start in 2024; neither term has a 2020-23 history), source artifacts/sky01/drift_weighted_idjoin.json, coefficient_hash e35131133f30, no 2026 outcome used. First scored week: Week 11 (SKY01_IDJOIN_FIRST_WEEK in scripts/forward_terms_week.py); a Week 10 call answered before the declared first week and wrote nothing. Registered in forward_terms_week.py TERMS and forward_ledger.py TERM_DIRS. sky01snap keeps its name, coefficient (+0.019904), records and history. No scheduler edit: capture_scheduler.py:951 already runs forward_terms_week.py with every term.
History, LOSO 2024-25, n 533, opener grade, 1 look (scripts/sky01_drift_weighted.py --join players; its roster arm reproduces drift_weighted.json exactly):
- sky01snap (roster crosswalk): 69.2% of out-rows weighted; 188 nonzero games; decisive 16, term 8, base 8; log loss +0.000016 [-0.00309, +0.00290] P+ .517; Brier +0.000063 P+ .549; accuracy 0.00 pts [-1.50, +1.50] P+ .459; folds +0.0182, +0.0381; in-sample log loss +0.000169.
- sky01snap_idjoin: 82.6% weighted; 203 nonzero games; decisive 21, term 12, base 9; log loss -0.001002 [-0.00670, +0.00399] P+ .381; Brier -0.000317 [-0.00295, +0.00199] P+ .432; accuracy +0.56 pts [-1.13, +2.25] P+ .726; folds +0.0090, +0.0505; in-sample log loss +0.000097; Brier gap .000378.
Term value changes on 73 of 533 games (2024: 20, 2025: 53). unresolved_below_power; closes nothing; not served. Recorded as new cells in family sky01_news_availability: sky01_weighted_term_idjoin_logloss_oos, _brier_oos, _accuracy_oos (batch tests/scratch/eng54/sky01_idjoin_history_batch.json). Week 11 dry run today: 13 games, term values equal under both joins (mean 0.0934); only the coefficient differs so far.
Verified: the six sky01 and sky01snap forward files carry write times of 2026-11-03 and 2026-11-10; nothing else under artifacts/forward_terms was written this session except the new frozen file and another agent's xlg09_cfb_opener_served. 39 files hash-identical from 18:13:05 to the end of the unit; the 18:06 hash list was overwritten by another agent in the shared scratchpad, so hashes do not cover the first six minutes and the write times stand in for them. Ledger rebuilt with and without the new term: frame 136 x 87 and score table equal, and equal to artifacts/forward_ledger_allweeks/ledger.parquet. forward_weekly.py exit 0; pool loads; ruff: no new findings (C416, RUF059, SIM115 in sky01_drift_weighted.py were there before).
Next: Tue 2026-11-17 the existing job writes Week 11 for both terms; read both lines from forward_weekly once Week 11 settles. Open: the snap file the runner reads ends at 2025, so every 2026 weight for both terms is a prior-season mean (2026 snaps sit in data/players/forward_snaps, unread); using them would be another new term.