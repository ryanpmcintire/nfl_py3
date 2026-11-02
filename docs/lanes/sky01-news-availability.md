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
