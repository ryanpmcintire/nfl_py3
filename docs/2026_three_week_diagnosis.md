# Diagnosis of the first three published weeks

**Measured:** there is a concrete input failure: the served pass-protection signal lost all current-card coverage in Week 3. The local play-by-play snapshots stop at 2025. The first two 2026 games entered the four-game lookback without pressure observations, leaving only two usable games; the signal requires three. All 32 team windows therefore became missing and all 16 matchup flags became zero. The source-health artifact nevertheless reported complete.

**Inferred:** the process was not fully healthy. Repair current-season play-by-play ingestion and add an input-coverage check that distinguishes missing pressure data from a genuine zero signal. The evidence does not justify changing the calibrated weights or adding a rule to reverse losing picks. This audit identifies the repair; it does not implement a feed or serving change.

**Measured:** the pressure signal changed two Week 2 picks into winners: Detroit to Buffalo and the Chargers to Las Vegas. With only that fitted term removed, the Week 2 card would have been 12-4 rather than 14-2. This is a same-input ablation, not a newly fitted model. No saved snapshot contains 2026 pressure data, so a corrected Week 3 card cannot be reconstructed from the available pregame captures. The missing input cannot honestly be assigned a number of Week 3 losses.

## Results versus stated expectations

**Measured:** the published cards were 9-7, 14-2 and 5-10 through Sunday, with Monday pending. Week 1 used the older historical reliability display; the current fitted combination first appears September 14. Its displayed scores are not treated as verified probabilities for the main comparison.

| Week | Published record | Expected wins | Relevant tail probability | Central 95% win range |
| --- | --- | --- | --- | --- |
| 2 | 14-2 | 8.91 | 0.75% (at least 14) | 5-13 |
| 3 | 5-10 | 8.19 | 8.15% (at most 5) | 4-12 |

**Inferred:** the bad week is plausible under the stated probabilities, and the good week was unusually favorable. It would be wrong to call the entire swing ordinary: the joint Week 2-3 dispersion tail is 0.770% under the independent calibrated-probability model. The 34 adjacent two-week windows in the comparable 2024-2025 historical replay contain none as dispersed. Across both live weeks, however, 19 wins versus 17.10 expected is unexceptional (30.79% chance of at least 19). These are conditional sampling probabilities, not probabilities that the model is sound or that luck caused the results.

**Measured:** in the same-recipe 2024-2025 historical replay, 2 of 36 weeks finished at or below one-third wins. The broader four-season reference has 4 of 72 such weeks. Three Week 3 losses missed the original pool line by half a point: Dallas, the Chargers and the Jets. They remain losses; their close margins are diagnostic context, not a grading change.

## What the losing decisions were based on

**Measured:** all 32 Week 2-3 published probabilities and sides were reconstructed from raw forecasts, contemporaneous fitted coefficients, archived schedules and arrest inputs, cached pregame weather, and captured market movement. There were zero side mismatches and zero probability mismatches at tolerance 1e-7. All lines match the original pool lines. The reconstruction uses no network data.

Week 3 raw/served disagreements were:

| Game | Raw pick | Served pick | Mechanism selecting the served side |
| --- | --- | --- | --- |
| Baltimore-Dallas | Baltimore | Dallas | First-year-coach term favored Dallas |
| Cincinnati-Pittsburgh | Pittsburgh | Cincinnati | First-year-coach term favored Cincinnati |
| Seattle-Washington | Washington | Seattle | Captured market movement favored Seattle |
| Tennessee-Giants | Giants | Tennessee | Fitted intercept and movement-availability term tipped a near-even probability |

**Measured:** the market-dominated Week 3 picks went 4-3; the four completed situational-signal-dominated picks went 0-4. The three intercept-dominated picks went 0-3 with probabilities close to 50%. These groups were examined together with every declared group below, not selected as standalone evidence of a defect. The two losing coach reversals and the movement reversal reproduce the stored inputs and sign conventions.

**Measured:** all 90 archived raw-model feature columns were present in both Sunday forecasts. No column acquired additional missing values in Week 3; temperature and wind were missing in both. The base team-stat snapshots contain 2026 Week 1 and Week 2 results. The confirmed missing-current-season dependency is play-by-play pressure, not the base score/team-stat feed. This scoped audit does not prove every source fact or feature is correct.

**Read:** `pbp08_matchup_flags.py:138` computes a one-game-shifted four-game rolling mean requiring three observations; `pbp08_matchup_flags.py:181` turns unavailable quartile assignments into false flags. `pbp08_protection_mismatch_tilt_overlay.py:68` loads the latest snapshot and returns the flags without rejecting all-missing pressure windows. `weekly.py:40` builds weak-stack features but does not itself ingest play-by-play.

## Historical check of the proposed explanations

All results below are **measured**, chronologically fitted 2024-2025 predictions. These seasons informed earlier research, so this is diagnostic historical evidence, not independent proof of an edge. Earlier 2022-2023 folds had zero movement coefficients and are retained separately as a four-season reference.

| Arm | Record | Brier | Log loss |
| --- | --- | --- | --- |
| combined | 305-228 | 0.245340 | 0.683647 |
| raw | 282-251 | 0.250898 | 0.695016 |
| neutral_market | 271-262 | 0.250000 | 0.693147 |

The combined-versus-raw Brier gain is 0.005558, week-block 95% interval [-0.001999, +0.013554], probability_positive 0.92350. On the 193 decisive disagreements, combined picks went 108-85; exact fair-coin null p=0.11305. This evidence supports keeping the question open; it does not settle the edge or justify a Week 3-driven reversal.

| Declared group | Games | Combined wins | Raw wins | Brier gain [95% interval] | probability_positive |
| --- | --- | --- | --- | --- | --- |
| all | 533 | 305 | 282 | +0.00556 [-0.00200, +0.01355] | 0.92350 |
| raw_agree | 340 | 197 | 197 | +0.00235 [-0.00447, +0.00884] | 0.74985 |
| raw_disagree | 193 | 108 | 85 | +0.01121 [-0.00589, +0.02938] | 0.89325 |
| home | 295 | 169 | 164 | +0.00689 [-0.00242, +0.01646] | 0.92125 |
| away | 238 | 136 | 118 | +0.00391 [-0.00681, +0.01400] | 0.77270 |
| favorite | 270 | 163 | 147 | +0.00861 [-0.00211, +0.01899] | 0.94065 |
| underdog | 257 | 139 | 130 | +0.00263 [-0.00920, +0.01411] | 0.67950 |
| pickem | 6 | 3 | 5 | -0.00601 [-0.13650, +0.09201] | 0.49234 |
| confidence_below_55 | 255 | 140 | 138 | -0.00010 [-0.00785, +0.00739] | 0.49650 |
| confidence_55_to_60 | 154 | 89 | 78 | +0.00758 [-0.00558, +0.02098] | 0.87025 |
| confidence_at_least_60 | 124 | 76 | 66 | +0.01469 [-0.01276, +0.04063] | 0.85005 |
| spread_below_3 | 141 | 76 | 74 | -0.00112 [-0.01728, +0.01556] | 0.44740 |
| spread_3_through_7 | 290 | 166 | 146 | +0.00776 [-0.00253, +0.01884] | 0.92570 |
| spread_above_7 | 102 | 63 | 62 | +0.00853 [-0.01046, +0.02568] | 0.81525 |
| weeks_1_to_3 | 92 | 53 | 43 | +0.00605 [-0.00370, +0.01624] | 0.89739 |
| later | 441 | 252 | 239 | +0.00546 [-0.00347, +0.01499] | 0.87965 |
| flag_coach | 84 | 48 | 46 | -0.00308 [-0.01738, +0.01125] | 0.32090 |
| flag_division | 95 | 56 | 48 | +0.02032 [-0.01524, +0.04723] | 0.87570 |
| flag_arrests | 10 | 7 | 3 | +0.07676 [+0.03319, +0.11574] | 0.99905 |
| flag_bye | 53 | 29 | 31 | -0.00359 [-0.03006, +0.02735] | 0.39115 |
| flag_cold_visitor | 53 | 28 | 27 | +0.00221 [-0.03454, +0.03108] | 0.56570 |
| flag_protection | 93 | 59 | 42 | +0.01279 [-0.00246, +0.02863] | 0.95035 |
| flag_tank_zone | 16 | 9 | 12 | +0.01938 [-0.06195, +0.08154] | 0.70396 |
| dominant_raw_model | 92 | 52 | 53 | -0.00200 [-0.01156, +0.00760] | 0.34205 |
| dominant_situational_flags | 242 | 135 | 128 | +0.00179 [-0.00792, +0.01125] | 0.64075 |
| dominant_market_movement | 191 | 113 | 97 | +0.01422 [-0.00043, +0.02932] | 0.97215 |
| dominant_intercept_and_availability | 8 | 5 | 4 | -0.00037 [-0.00235, +0.00179] | 0.34543 |

**Inferred:** there is no demonstrated repeatable wrong-sign mechanism in the losing coach or market groups. Pressure-flagged historical games favor the combined picks, but that subgroup is not an isolated estimate of the pressure term. Its absence is an input-integrity failure regardless of the uncertain performance effect. Under `AGENTS.md`, zero crossing cannot close a signal, and one fitted probability must select the served side. Every paired diagnostic contrast is retained as unresolved; none is closed or promoted.

## Reliability and coefficients

| Arm | Home-probability band | Games | Mean forecast | Observed home covers |
| --- | --- | --- | --- | --- |
| p | 0.0-0.4 | 40 | 0.3555 | 0.4000 |
| p | 0.4-0.5 | 198 | 0.4618 | 0.4343 |
| p | 0.5-0.6 | 211 | 0.5451 | 0.5545 |
| p | 0.6-1.0 | 84 | 0.6505 | 0.6190 |
| raw_p | 0.0-0.4 | 29 | 0.3802 | 0.5172 |
| raw_p | 0.4-0.5 | 206 | 0.4583 | 0.4709 |
| raw_p | 0.5-0.6 | 254 | 0.5417 | 0.5354 |
| raw_p | 0.6-1.0 | 44 | 0.6237 | 0.5227 |

| Held-out season | Intercept | Raw logit | Flag sum | Movement | Movement available |
| --- | --- | --- | --- | --- | --- |
| 2024 | -0.059191 | 0.274508 | 0.308818 | 0.210276 | 0.067970 |
| 2025 | -0.054099 | 0.292328 | 0.295997 | 0.188739 | 0.047462 |

**Measured:** coefficient signs agree across these two folds. Per-season performance, all group intervals, exact disagreement nulls, calibration diagnostics and the broader four-season results remain in the JSON artifacts. No diagnostic coefficient reaches the serving path.

## Reproduction, limits and handoff

Primary commands, using `.tools/uv.exe --cache-dir .tmp/uv-cache run --no-sync --offline`:

- `python scripts/audit_three_week_input_replay.py`
- `python scripts/audit_three_week_probabilities.py`
- `nfl-ats weak-signals record --batch artifacts/diagnostics/2026-three-weeks/weak_signals_batch.json`
- `ruff check scripts/audit_three_week_probabilities.py scripts/audit_three_week_input_replay.py`

Artifacts are under `artifacts/diagnostics/2026-three-weeks/`: published grades, source scores, replay rows, input coverage, all saved PBP manifests, historical predictions/weeks/windows, group results, source hashes and registry batch. The declaration and amendment are in `registry/studies/2026_three_week_diagnosis.json`.

This was declared after observing the three live records and the four losing Week 3 disagreements. There are 27 groups including the aggregate, 81 group intervals per historical scope, 8 reliability cells and 2 descriptive calibration fits per scope, 54 live group rows, 6 weekly tails (Week 1 sensitivity-only), 2 joint-dispersion checks, historical window comparisons and one Week 2 pressure-term ablation. Both historical scopes are retained. The registry batch contains 108 paired group metrics. Overlapping groups are not independent evidence.

The live protection lookup has no historical snapshot selector; its reconstructed values were accepted only after they reproduced every stored probability. Missing archived 2026 play-by-play prevents a corrected Week 3 replay. Week 1 score semantics prevent treating all three cards as one unchanged calibrated model. Data and prediction outputs remain untracked.

Next bounded repair: ingest a complete historical-plus-current-season PBP snapshot; check completed-game coverage rather than snapshot date alone; have prediction input checks report and reject missing required pressure windows; verify pressure coverage on the real upcoming card. Preserve original locked cards and the prospective study commitment. Before any source edit, check the enrolled source hashes and record a new study version if a pinned recipe file changes. No forecast, dashboard, scheduler, or prospective enrollment was changed by this diagnosis.
