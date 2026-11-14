# MOD-36 Add-one noise ensemble

## Goal
Say what a no-information term produces in the add-one test on the served pick probability, how big a real effect must be to show, and which grade shows it soonest (ROADMAP MOD-36). An instrument; it closes nothing.

## State
Harness (**measured**, units 2 and 3 done 2026-11-14): base `out_of_season_home_probability`, pick_probability 20261114T184614Z, opener 2020-25, leave-one-season-out, 2,000-draw game bootstrap, seed 36, 200 draws per cell. Unit 2 `scripts/mod36_addone_noise_floor.py` -> `artifacts/mod36/`. Unit 3 `scripts/mod36_metric_power.py` -> `artifacts/mod36/metric_power/` (run.log, summary.json, detectable.csv, draws.parquet; 765 s, one thread). The harness reproduces the three real units (11-13, 17-26, 36-41); unit 3 reproduces unit 2's null to the last digit.

### Unit 2: null and cover-flag control (15:24 ET)
Null = real column shuffled within season. Mean [5th, 95th]:

| Template (n; fit vs reference) | Accuracy pts | Log loss | Changed 5/50/95; record; losing | P+ acc <=.10 | P+ LL <=.10 |
|---|---|---|---|---|---|
| Short-week side, 3 values (1,503; vs served) | -0.17 [-0.74,+0.47] | -.00038 [-.00128,+.00079] | 14/37/74; 16-20; 67% | 18% | 38% |
| Rest difference, 15 values (1,503; vs served) | -0.21 [-1.00,+0.60] | -.00047 [-.00159,+.00061] | 18/50/117; 22-29; 68% | 21% | 38% |
| Questionable habit, continuous (1,503; vs served) | -0.27 [-1.13,+0.54] | -.00048 [-.00157,+.00058] | 32/76/187; 36-42; 68% | 17.5% | 41% |
| Playoff leverage, weeks 10-16 (576; vs intercept-only) | -0.90 [-2.78,+1.56] | -.00110 [-.00358,+.00158] | 23/50/103; 22-28; 78% | 29% | 44.5% |

Mean log loss is the parameter cost (1/(2 x training games) = .00040, .00104; **inferred**). Control (column tilted by the realised cover, logit per SD): smallest size with P+ >= .90 in 80% of draws is .265 / .262 / .242 / .393 on log loss (+1.57 / +1.64 / +1.36 / +3.62 true accuracy pts) and .265 / .315 / .339 / .472 on accuracy. At the sizes units fitted (.019-.146) detection is 0-12% through .082 and 8-41% at .146.
Today's three results in the matched null (percentile accuracy / log loss / record): short-week -0.13, 11-13: 56/46/56; habit -0.60, 17-26: 27/60/27; leverage -0.87, 36-41: 56/65/56. All inside the 5-95 band: no evidence of a wrong sign, none of absence; all `unresolved_below_power`.

### Unit 3: metric power (declared 15:35 ET before any run; done 15:52 ET)
Prior lanes (**read**): acc-u2-proper-scores regraded 17 real arms on log loss, Brier and lattice RPS with no control; accuracy-ceiling-theory states the 1.3-point standard error; positive-control-power measured the retrain evaluator on accuracy only; every-metric made runners emit four metrics; unit 2 compared accuracy, log loss and Brier under a cover-flag plant. None compared grades under a margin plant here.
Declared: 5 grades of one fitted cover probability on the same bootstrap weights: accuracy; log loss; Brier; mean margin against the opener of the picked side; whole-margin ranked probability score of the served discrete line-conditional lattice (`mass_preserving_lattice` prior band pool per week, tilted to reproduce the cover probability as in acc-u2, push mass kept). Plant: column shuffled within season, each final margin moved by delta points per SD of the column, rounded to a whole point at random; games landing on the line are pushes (out of cover fits and scores, kept in RPS). Grid = unit-fitted sizes and 2/8 to 12/8 of unit 2's accuracy-detectable size, converted at 5.78 points per logit (4 x .0432 games per point at the line): 17 planted cells per template. Looks: 40 detectable sizes, 20 null rows.

Smallest size with P+ >= .90 in 80% of draws, points per SD, **measured** (in brackets: interpolated crossing; true accuracy pts there):

| Grade | Short-week | Rest | Habit | Leverage (576) | Null P+ >= .90 | Null mean [5th, 95th], rest |
|---|---|---|---|---|---|---|
| Accuracy | 1.53 (1.52; +1.2) | 2.05 (1.85; +1.9) | 2.20 (2.18; +2.6) | 2.77 (2.77; +4.1) | 1.5-4% | -0.21 [-1.00, +0.60] |
| Log loss | 1.73 (1.56; +1.3) | 1.59 (1.57; +1.4) | 1.71 (1.58; +1.6) | 2.77 (2.55; +3.5) | 0-0.5% | -.00047 [-.00159, +.00061] |
| Brier | 1.53 (1.53; +1.2) | 1.59 (1.54; +1.4) | 1.71 (1.53; +1.5) | 2.77 (2.50; +3.4) | 0-0.5% | -.00023 [-.00079, +.00028] |
| Picked-side margin | 1.34 (1.32; +0.9) | 1.82 (1.60; +1.5) | 1.96 (1.78; +2.0) | 2.43 (2.27; +2.9) | 5-6.5% (576: 0.5%) | -0.035 [-0.234, +0.165] pts |
| Margin RPS | 1.53 (1.53; +1.2) | 1.59 (1.58; +1.4) | 1.71 (1.64; +1.7) | 2.77 (2.53; +3.4) | 0-0.5% | -.0051 [-.0142, +.0032] |

One point per SD fits as about .14 logit; log loss crosses at .22-.23 logit, as in unit 2. Paired planted draws flagged by RPS only vs log loss only: 172-176, 192-145, 101-164, 122-124 of 3,400.
Post hoc, 20 further looks: each grade's bar set at its own null 95th percentile of P+ (5% false alarms, in-sample). Points per SD (true accuracy): accuracy 1.40 / 1.80 / 1.94 / 2.31 (+1.0 / +1.8 / +2.3 / +3.0); log loss 1.25 / 1.18 / 1.11 / 2.04 (+0.9 / +0.9 / +0.9 / +2.4); Brier 1.22 / 1.15 / 1.12 / 2.04; picked-side margin 1.30 / 1.65 / 1.88 / 2.12; RPS 1.18 / 0.97 / 1.13 / 1.84 (+0.8 / +0.7 / +0.9 / +1.9).
Recommendation:
1. Primary grade for one-term tests: log loss of the cover probability (Brier is a dead heat), read as the percentile in the matched null, not against P+ .90. At the fixed bar it beats accuracy only on many-valued terms (15% and 27% fewer points per SD on rest and habit, none on short-week). Against the matched null it detects 1.1-1.25 points per SD (.16-.17 logit, +0.9 accuracy pts) at 1,503 games, 11-43% below accuracy read the same way.
2. No margin grade earns primary status: RPS ties log loss at the fixed bar and runs from 2% worse to 18% better at the matched bar; picked-side margin beats accuracy by 13-18% at the fixed bar only with 5-6.5% null false alarms, 3-8% once matched. **Inferred** cause: the coefficient is fitted on the cover flag and moves the lattice through one tilt, so extra thresholds add the same game noise.
3. Limits: every unit-fitted size (at most 0.84 points per SD) is still below the best detectable size, so nothing is resolved or closed; a log-loss or margin gain is not a cover gain (acc-u2: five arms passed both while losing 1.1 accuracy pts); one fitted cover probability still selects the pick.

Recorded: 12 cells family `mod36_addone_noise_floor`, 8 cells family `mod36_metric_power` (picked-side margin and RPS nulls, category control); `weak-signals pool --league nfl --effect-units accuracy_points` exit 0 after each.

## Use in future packets
1. Declare log loss as the primary grade and report the candidate's percentile in its matched null (add one dict to TEMPLATES), never its raw sign: P+ <= .10 occurs in 18-29% of no-information terms on accuracy and 38-45% on log loss; a losing changed-pick record in 67-78%.
2. Compare log loss against the parameter cost (-.0004 per parameter at 1,503 games, -.0011 at 576), not zero.
3. State power: at P+ .90 only about .25 logit per SD (+1.4 to +1.6 pts) is seen 80% of the time at 1,503 games; against the matched null about .17 (+0.9). Smaller effects stay unresolved; neither null nor control closes them.

## Tried
- Unit 1 (2026-11-03): 50 N(0,1) draws per K on the 88-input ridge refit, family mod36_noise_floor; still the read for refit-harness candidates.
- positive-control-power measured the retrain evaluator on two flag controls; this is the offset add-one harness with nulls matched to candidate marginals.
- Planted cells unrecorded: the pool takes every unresolved cell whatever its category (**read** src/nfl_ats/weak_signals.py:1583); detection shares have no unit.
- The harness script runs on import; unit 3 loads its source without the final call rather than editing it.

## Next
Pin each grade's matched bar out of sample (about 1,000 null draws, fresh planted draws); fit the term on the margin through the lattice tilt instead of the cover flag, the one route by which a finer outcome could add power; a declared negative-direction control; detectable size for a pooled family of small terms.

## Open
- 200 draws: detection shares resolve to about 5.5 points, sizes to one grid step (.19-.35 points per SD); bracketed crossings are linear interpolation. The matched bars are post hoc and in-sample.
- The plant puts 2-15 games per draw on the line while the frame holds no real pushes; the lattice pool uses closing lines where no opener is stored.
- Controls are positive-direction only. The 20 null cells (and unit 1's) sit in the pools as control category; excluding controls from pooled estimates is a src decision not made here.
