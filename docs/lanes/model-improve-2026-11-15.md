# model-improve-2026-11-15

## Goal
Owner 2026-11-15: keep improving the model, no overfitting. Work only units with a named mechanism, out-of-sample or forward grading, every look counted and recorded.

## State
- 2026-11-15 18:26 started. Inventory read: base-model levers exhausted (mod23-unit11 inventory, mod57 UPDATE); add-one tests on 1,503 games cannot see under ~1.5 pts. The one new information source beating its baseline forward is the LEAD-106 injury-wording read.
- Launched two units in parallel: LEAD-108 (read-weighted expected-lineup term, lane lead-108-...md) and LEAD-112 (post-opener availability news, lane lead-112-...md). Results land in those lanes.

- LEAD-108 done (measured, artifacts/lead108_wording_mixture/report.json, 24 looks, 4 cells unresolved_below_power): read vs tag ladder +.0003 [-.0018,+.0024] P+ .62; read term vs served -.0021 P+ .14; ladder term vs served -.0024 P+ .08; slopes change sign by week. Served base was the published card probability, because artifacts/pick_probability per_game has no 2026 rows (verified: 2023-25 only). Weeks 2-8 read coverage 8-40%; Week 9 full. Nothing served.

- LEAD-112 done (measured, tests/scratch/lead112/result.json, 16 looks, 12 cells unresolved_below_power): 105 games in 7 week blocks; every arm worse than the card out of sample (move only -.0095, designation late change -.0200, x practice -.0217 log loss), permutation p .62-.75, so the loss is fitting cost on 7 folds, not a sign. Root adjudication: unresolved is correct (in-sample coefficient expected sign, SE ~.2; fold signs flip). Defect: the Wednesday report carries no game designations, so the "late" term was nearly the whole designation and the post-opener mechanism was not isolated. RUF046 fixed by root.

- 2026-11-16: per add-one-noise-floor memory, no NFL one-term add-ons. Launched XLG-14 unit 4 (college-first screen of a new non-luck family, declared before outcomes; matched null; survivor gets one declared NFL look, not run). Result lands in docs/lanes/xlg-14-xlg14-cfb-first-discovery.md "Unit 4".
- Unit 4 done (measured, artifacts/xlg14_family_u4/): scoreboard-vs-EPA family, 4 candidates, 12 cells unresolved. Survivor `season_epa_gap` (season points margin minus EPA-predicted margin): +.079/SD, log-loss 100th pct of null, 13/13 folds, sign OPPOSITE the over-reaction hypothesis (teams outscoring their EPA keep covering). About 19% of null families pass one of 4 reads. NFL confirmation ran once 2026-11-16 (measured): +.034/SD [-.062,+.137] P+ .76, 5/6 folds positive; log loss 47.5th pct of null; 66 changed picks 30-36; not confirmed, unresolved, 3 cells. Correlation with served logit +.010. Nothing served. Transfer tally college/screen positive -> served: 0 of 9.
- Week 10 still has 5 open games (incl. MNF 11-16), so forward_weekly.py cannot take Week 10 until Tuesday morning.

## Tried
See mod23-unit11-2026-11-03.md inventory; do not repeat.

## Next
1. Both units are done and neither serves anything. Candidate follow-ups, each needing a declaration first: LEAD-112 Wednesday-to-Friday practice-status change only; LEAD-108 forward scoring from Week 10 (full read coverage).
2. After MNF settles (Tue 11-17 morning): Week 10 per-game grade by term (picks-getting-worse lane Next 1); rerun scripts/forward_weekly.py so the ENG-50 stacker and LEAD-99 forward log take Week 10.

## Open
- Nothing served changes from this lane before 2027 unless a forward record meets the ENG-48 monitor bar.
