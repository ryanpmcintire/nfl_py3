# LEAD-109

## Goal
Position-value weighting of the wording read: which absences the read prices, by position group and trailing snap share (ROADMAP LEAD-109).

## State (2026-11-21 11:24, unit 2 done, numbers measured this session)
Mechanism: an availability gain matters through the player's marginal value; QB and OL absences move margin more than depth. Values |spread pts per Out| from LEAD-110 OLS 2011-2025 (prior seasons): QB 2.68, ST .81, OL .38, RB .34, RECV .28, LB .26, DL .22, DB .16. Nothing served.
Unit 2 command: `scripts/lead109_position_weights.py --forward-weeks 9 10` (new forward mode, stored reads, no model calls) -> `artifacts/lead109_position_weights/forward_report_weeks_9_10.json`; registry batch `forward_registry_batch_weeks_9_10.json`. Population: 607 id-verified forward rows (every listed player incl. 27 QB, 24 OL), 29 games, 58 team-week clusters, ladders from prior weeks. 30 looks declared before signs, family `lead109_position_value_forward`. Gain = log loss read better than ladder; bootstrap 2000; detectable = 2.8 x bootstrap SE.
- Decisive rows (read and ladder on opposite sides of .5): vs tag ladder read 11-6 (exact p .33); vs tag+practice 8-0 (p .008); QB/OL 1-0 (one QB row).
- Value-weighted gain vs tag .085 [.015, .186] P+ .996 (unweighted .052; detectable .127); vs tag+practice .037 [-.007, .099] P+ .944 (unweighted .023).
- Declared comparison, weighted minus unweighted: vs tag +.033 [-.018, .105] P+ .85, shuffle null p .165 (detectable .090); vs tag+practice +.014 [-.018, .060] P+ .72, null p .239 (detectable .057).
- Filed-status games only (560 rows): value-weighted .036 [-.004, .084] P+ .957 / .011 [-.011, .033] P+ .82; difference +.011 P+ .71 / -.003 P+ .37.
- QB 27 rows: .139 [-.033, .411] P+ .88 / .056 [-.040, .223] P+ .76 (detectable .33/.20), carried by Wk9 MIN Murray (game read before status filed, +2.90 nats) and Wk9 ARI Brissett (+1.37). OL 24 rows: -.009 [-.011, -.007] P+ 0 both ladders; all 24 OL rows were Out and sat, so the cell is the read's floor (.01-.02) against the ladder's Out rate (.003), not a position-value test.
- Group LOWO (2 weeks) worse than grand mean: -.014 [-.027, -.003] / -.005 [-.011, -.001] P+ 0; Spearman group gain vs value .00 / -.06. Halves: value-weighted Wk9 .146 vs Wk10 .017 (tag), .061 vs .009 (tag+practice), not stable, as in unit 1.
Recorded 10 cells unresolved_below_power (`registry/weak_signals/lead109_position_value_forward.json`); `weak-signals pool --league nfl --effect-units accuracy_points` exit 0.
ROADMAP LEAD-109 update: Weeks 9-10 forward rerun (607 rows, all QB/OL, 30 looks): value-weighted read gain vs tag ladder .085 [.015, .186] P+ .996, weighted minus unweighted +.033 P+ .85 (null p .17), below detectable size .090; QB .139 P+ .88 on 27 rows carried by two players; OL rows all Out, uninformative; group weights unstable across weeks; 10 cells unresolved_below_power. Next: rerun with each forward week; enrich uncertain QB/OL rows.

## Tried
Unit 1 (2026-11-03, 294 sampled weeks 2-8 reads, 24 looks): value-weighted vs tag .080 [-.011, .196] P+ .947, difference +.015 P+ .57; vs tag+practice .100 P+ .86, difference +.064 P+ .66; group LOWO worse than grand mean; recorded 2 cells in `lead109_position_value`. Unit 2 above.

## Next
Rerun `--forward-weeks 9 10 11 ...` as each LEAD-106 forward week is scored (separate report per week set). Power on high-value groups needs uncertain QB/OL rows (Questionable or practice-limited); Weeks 9-10 hold 2 Questionable QBs and 0 non-Out OL. Model entry stays as declared: V = sum value_g x (1 - P_read) x expected snap share, value_g fitted leave-one-season-out on the multi-season archive, one availability term, never a side rule.

## Open
- Orchestrator adjudication: OL cells have the whole interval below zero; recorded unresolved because the cell holds only Out rows that sat (read-floor calibration, same on Out rows of every group), not a resolved wrong sign of position value. If judged a refuted mechanism, re-record with `replace` and `--closing-ground wrong_sign_resolved`.
- The read's Out-row floor (.01-.02 vs ladder .003) costs every group a little; a LEAD-106 calibration question, not LEAD-109.
- Rules: zero crossing closes nothing; one fitted probability selects the side; weights chosen out of season; every cell a look.

## Root decision (2026-11-21 11:25 EST)
- The OL cell (whole interval below zero) stays unresolved_below_power: all 24 OL rows were Out and sat, so it measures the read's floor against the ladder's Out rate, a ceiling on the read rather than a test of position value; no refuted mechanism.
