# recent-weeks-audit-2026-10-19

## Goal
Owner reports poor pick results the last few weeks (Week 6 6-7 before MNF WAS +7.5). Measure 2026 published-card record per week against the card's own cover chances, and find any broken input on Weeks 4-6 (the Week 3 precedent was a silently empty pass-protection feed, docs/2026_three_week_diagnosis.md). A broken input changes picks; a run of ordinary losses does not justify a rule change (AGENTS.md research invariants).

## State
- Measured (as-played picks = artifacts/clv_ledger/published_picks.parquet via frozen_picks; scores = data/raw/20261013T220620Z/schedules.parquet; line sign: market_spread is home spread, home covers hm - s). W1 9-7 exp 8.78; W2 14-2 exp 8.91; W3 6-10 exp 8.81 (CHI/PHI Best Pick now graded won; diagnosis had it pending); W4 9-7 exp 8.63; W5 8-7 exp 8.41; W6 6-7 (owner count; local schedule snapshot stops after Wk5) exp 7.13 over 13 decided, P(<=6)=0.361. W4-6 23-21 vs 24.17 expected, P(<=23)=0.418. Season 52-40 vs 50.67, P(<=52)=0.649. Best Pick: W1 MIA lost, W2 NO won, W3 CHI won, W4 PIT lost, W5 NE won, W6 CLE pending.
- Pressure feed repaired: pbp snapshots from 20260928 carry 2026 (W4 snapshot 20260929 has W1-3 complete; 20261006 W1-4; 20261013 W1-5). Flags have 0% missing windows W2-W6 (flagged games W4 4... see groups). Replay of W4-6 cards: 45/45 games, 0 side or probability mismatches.
- Raw features W4-6: no new missingness vs W3; temp and wind 100% missing in live cards (35% in 2024-25 history), same as W3, unchanged.
- Groups W4-5 decided (31): raw agrees 12-11 (exp 12.57); raw disagrees 5-3 (exp 4.47). Dominant term: raw_model 5-3, situational 7-8, movement 5-2, intercept 0-1.
- Orchestrator re-ran grade.py 2026-10-19: W1-W5 records and expected wins reproduce exactly (measured).
- Unit 2 (weather skew) started: lane docs/lanes/live-weather-features.md.
## Tried
Scratch scripts in tests/scratch/recent_weeks/ (grade.py, replay.py, groups_w46.py, pbp.py, feat.py, tail.py). Week 6 not graded locally.
## Next
Re-grade W6 after schedule snapshot refresh; no pick/weight change implied.
## Open
Live temp/wind absent from cards (pre-existing).
