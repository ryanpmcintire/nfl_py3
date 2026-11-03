# MOD-07 prospective status audit

## Goal
Reconcile the stacking backlog with the current prospective registry and ledger.

## State
The status audit is complete; the research line remains unresolved.
**Measured:** `artifacts/prospective/challenger_decisions.parquet`
contains 48 unique games, 16 each in Weeks 1, 2, and 3 of 2026. Every row uses
registered fingerprint `bc77638d47e2748c`; all three recorded prediction-source
hashes match their `recommendations.csv` files. Ledger SHA-256:
`fb15413901b262b02dca1f4f6ffe48acda7f6e2da9dbb860b5386b8603d62d98`.
**Read:** the registry marks this challenger `ACTIVE_PROSPECTIVE`.
Historical stacking results remain `unresolved_below_power`; current collection
does not promote the challenger or establish an out-of-sample edge.

## Tried
- Root verified row uniqueness, fingerprint, week counts, and source hashes with
  locked Python; result saved in `.tmp/mod07-root-coverage.json`.
- Read the recorder, weekly opt-in flow, and prospective CLI. The next capture
  uses the existing workflow, rather than a new capture implementation.
- The absent historical `scripts/mod07_weak_stack.py` is a replay gap; it does
  not block the current prediction and prospective-recording path.

## Next
At the actual Week 4 lock, run the normal weekly flow with `--record-decisions`.
Continue prospective grading without replacing previous weeks or changing the
frozen challenger fingerprint. Preserve the historical research classification.

## Open
Further prospective evidence and recovery of the historical replay script remain.
This completed lane closes the status audit only.

## Audit 2026-11-03 (forward-only check)
- Historical: nothing unrun. Stack replays are spent: [2020,2021] 53.29 v 51.32 (P+ 0.8745), 1,537-game 52.83 v 52.50, v3/v4/v5 all recorded unresolved_below_power (registry weak_stack_v4_forecast_weather, weak_stack_v5_fluview_away, post_discovery_2026_stack_terms). Retuning is inadmissible; the absent replay script is a gap, not a new look.
- Forward (measured, artifacts/prospective_scoring/20261103T153531Z): stack 63/121 (P+ 0.681 v coin) v active 66/121. Paired contrast: 45 games where stack and active picks differ in Weeks 1-8; active 24, stack 21 (about 5.6 differing games a week).
- Expected by Week 18 at that rate: about 100 differing games (inferred). Paired SD of accuracy diff is about sqrt(100)/270 = 3.7 points; the historical effect is +0.33, so the sample cannot reach power for it (would need thousands of games). Read stays a descriptive forward record, scored each Tuesday; score of record after Week 18 (Jan 2027). No registry record added (no new historical look).
