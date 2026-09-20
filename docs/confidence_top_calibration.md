# Sunday nominee calibration

## Protocol fixed before the first run, 2026-09-20

Mechanism: maximizing noisy probabilities within a week can overstate the
selected game's chance. Fit a positive temperature to the combined home-cover
logit, either on all Sunday-eligible calibration games or on that season's
weekly nominees. Apply the single fitted transformation to every candidate.
It preserves sides and ordering; it cannot establish a better ranking.

Use the frozen full opener population and Sunday-through-12:45 inputs from
`scripts/confidence_best_pick_sunday_matched.py`, including eventual pushes
when nominating. Exclude started games before nomination; exclude pushes only
from conditional nonpush probability scoring and fitting. Break ties by game ID.

For outer seasons 2023, 2024 and 2025, reserve the preceding season solely for
temperature calibration. The chronological combined-model fit uses seasons
before that calibration season, requiring at least two. A matched diagnostic
uses all available seasons except calibration and outer seasons (leave two
seasons out); label its future-season use explicitly. Keep those three roles
disjoint in each fold. Report the natural combined-model coefficients and
positive temperature coefficients per fold. Minimize calibration log loss,
with an inverse-temperature lower bound of 0.000001 and no fitted cut points.

This is a new outer-layer diagnostic on previously examined archives, not an
untouched validation of upstream model or situational-feature selection. The
frozen discrete model inputs and selected feature definitions are not refit
inside these folds. No result automatically changes the served card.

Compare two repairs with the identical uncalibrated combined fit, frozen
model-only probabilities, and a neutral 50% market-at-opener baseline. Report
same-game Brier and log loss, nonpush accuracy, fixed reliability bands
50-55, 55-60, 60-65 and 65-100%, calibration-to-outer gaps, season stability,
and nominee-versus-rest ranking slopes with week-block intervals. Report
nominee changes and decisive outcomes before headline effects; calibrated
arms should have none. Model-only nomination is a separate ranking baseline.

Primary estimand: nominee Brier improvement over the uncalibrated combined
fit in the chronological protocol. The family has two repair arms, two
protocols and two populations (all eligible games and weekly nominees): eight
paired Brier cells, with eight log-loss and eight calibration-gap diagnostics.
The five arms, four reliability bands, three seasons and ranking diagnostics
are declared reporting cuts, not additional candidate searches. Bootstrap
10,000 season-week blocks with seed 20260920; report probability_positive.
Record every paired repair result as unresolved unless an admissible closure
ground is demonstrated. Do not interpret crossing zero as rejection.

## Results

**Decision:** keep the served probability unchanged. Measured with
`uv run --no-sync python scripts/confidence_top_calibration.py`; the complete
prediction-level output and bootstrap summaries are in
`artifacts/confidence_top_calibration/20260920_fixed/`.

The chronological nominee-calibrated arm improved nominee Brier by 0.001040
[-0.027412, 0.030943], probability_positive 0.5227. Its all-game Brier change
was -0.002490 [-0.006082, 0.001022], probability_positive 0.0853.
The fitted nominee inverse temperature ranged from 0.000001 to 1.083538.
Inferred: this does not yet support a stable replacement for the served
probability. Both repairs remain unresolved_below_power; no admissible closing
ground was demonstrated. Intervals crossing zero are not grounds for rejection.

### Decisive games and population

Both protocols contain 697 resolved eligible games and 17 pushes over 54
outer weeks, with 54 resolved nominees. The positive transformation changed
zero sides, zero nominees and zero decisive games against its uncalibrated arm.
The combined and model-only nominations differed in 48 chronological weeks:
27 were decisive, with 17 combined wins and 10 model wins; exact two-sided
null p = 0.247789. In the future-season diagnostic, 51 nominations differed:
24 decisive, 15 combined wins and 9 model wins; p = 0.307456.

### Paired probability scores

Positive improvement means lower loss. Entries give estimate, 95% week-block
bootstrap interval and probability_positive. Future-season rows are diagnostics,
not deployable chronological validation.

| Protocol | Population | Calibration | Brier improvement | Log-loss improvement |
| --- | --- | --- | --- | --- |
| Chronological | all | All-game calibration | -0.000121 [-0.001143, 0.000875]; P+ 0.4177 | -0.000350 [-0.002603, 0.001835]; P+ 0.3914 |
| Chronological | all | Nominee calibration | -0.002490 [-0.006082, 0.001022]; P+ 0.0853 | -0.005191 [-0.012793, 0.002213]; P+ 0.0880 |
| Chronological | nominees | All-game calibration | -0.002348 [-0.010688, 0.005963]; P+ 0.2889 | -0.006042 [-0.025234, 0.012890]; P+ 0.2637 |
| Chronological | nominees | Nominee calibration | 0.001040 [-0.027412, 0.030943]; P+ 0.5227 | 0.001103 [-0.059299, 0.064678]; P+ 0.5097 |
| Future-season diagnostic | all | All-game calibration | -0.000347 [-0.002138, 0.001329]; P+ 0.3525 | -0.001293 [-0.005798, 0.002884]; P+ 0.2817 |
| Future-season diagnostic | all | Nominee calibration | -0.001069 [-0.004312, 0.002077]; P+ 0.2690 | -0.002372 [-0.009567, 0.004573]; P+ 0.2660 |
| Future-season diagnostic | nominees | All-game calibration | -0.000489 [-0.016092, 0.013942]; P+ 0.4812 | -0.004443 [-0.047629, 0.034284]; P+ 0.4260 |
| Future-season diagnostic | nominees | Nominee calibration | 0.012262 [-0.013252, 0.038225]; P+ 0.8270 | 0.024575 [-0.032239, 0.082704]; P+ 0.8001 |

### Chronological nominee baselines

All rows score the same combined nominees. Model-only nomination is a separate
ranking comparison: 52 resolved games and two pushes, 38.46% accuracy, Brier
0.296179 and log loss 0.788711. The 50% market baseline has no directional pick.
Calibration gap is observed accuracy minus mean confidence, in percentage points.

| Probability | Accuracy | Mean confidence | Brier | Log loss | Calibration gap, interval and P+ |
| --- | --- | --- | --- | --- | --- |
| Combined | 51.85% | 64.78% | 0.261144 | 0.715639 | -12.925577 [-25.932646, 0.341716]; P+ 0.0278 |
| All-game calibration | 51.85% | 64.26% | 0.263492 | 0.721681 | -12.403857 [-25.458369, 0.841933]; P+ 0.0340 |
| Nominee calibration | 51.85% | 56.05% | 0.260104 | 0.714536 | -4.193449 [-17.474671, 9.334165]; P+ 0.2746 |
| Model only | 48.15% | 56.97% | 0.268053 | 0.731116 | -8.824155 [-22.122122, 4.490368]; P+ 0.1016 |
| Market 50% | n/a | 50.00% | 0.250000 | 0.693147 | n/a |

### Chronological nominee reliability

Only occupied bands are shown. The final band includes probabilities of 65% or
higher. These were fixed reporting bands; none was searched for a rule or cutoff.

| Probability | Band | Games | Predicted | Observed | Gap, interval and P+ |
| --- | --- | --- | --- | --- | --- |
| Combined | 55.00%?60.00% | 9 | 58.49% | 55.56% | -2.932418 [-36.411284, 30.427836]; P+ 0.3777 |
| Combined | 60.00%?65.00% | 19 | 62.01% | 36.84% | -25.163312 [-46.184317, -3.809577]; P+ 0.0205 |
| Combined | 65.00%?100.00% | 26 | 68.98% | 61.54% | -7.441786 [-26.293225, 10.857890]; P+ 0.2269 |
| All-game calibration | 55.00%?60.00% | 12 | 58.70% | 50.00% | -8.700722 [-34.070985, 16.621253]; P+ 0.2015 |
| All-game calibration | 60.00%?65.00% | 21 | 62.44% | 52.38% | -10.061211 [-29.592836, 9.288856]; P+ 0.1404 |
| All-game calibration | 65.00%?100.00% | 21 | 69.24% | 52.38% | -16.862580 [-36.980018, 3.126416]; P+ 0.0609 |
| Nominee calibration | 50.00%?55.00% | 31 | 51.66% | 54.84% | 3.177236 [-15.683243, 19.712261]; P+ 0.6336 |
| Nominee calibration | 55.00%?60.00% | 11 | 57.99% | 54.55% | -3.440581 [-30.968240, 23.964070]; P+ 0.3878 |
| Nominee calibration | 60.00%?65.00% | 5 | 62.83% | 40.00% | -22.826222 [-62.501563, 16.849496]; P+ 0.0878 |
| Nominee calibration | 65.00%?100.00% | 7 | 67.57% | 42.86% | -24.709011 [-54.951554, 17.246139]; P+ 0.1196 |
| Model only | 50.00%?55.00% | 21 | 52.43% | 52.38% | -0.049415 [-19.658932, 19.815914]; P+ 0.4922 |
| Model only | 55.00%?60.00% | 20 | 57.00% | 55.00% | -1.999050 [-22.314059, 18.251366]; P+ 0.4130 |
| Model only | 60.00%?65.00% | 10 | 62.60% | 40.00% | -22.600784 [-52.123651, 7.065247]; P+ 0.0525 |
| Model only | 65.00%?100.00% | 3 | 69.83% | 0.00% | -69.825946 [-71.597704, -68.831872]; P+ 0.0000 |
| Market 50% | 50.00%?55.00% | 54 | 50.00% | n/a | n/a |

### Coefficients by fold

The combined home-cover logit has the intercept and four natural feature terms
below. Each positive inverse temperature multiplies that fitted logit. The lower
bound of 0.000001 makes the 2024 chronological nominee fit nearly neutral.

| Protocol | Outer | Calibration | Training seasons | Intercept | Model logit | Composition | Move | Move available | All-game inverse temperature | Nominee inverse temperature |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chronological | 2023 | 2022 | 2020, 2021 | -0.033936 | 0.365478 | 0.273936 | 0.000000 | 0.000000 | 1.312113 | 1.083538 |
| chronological | 2024 | 2023 | 2020, 2021, 2022 | -0.040963 | 0.368354 | 0.303338 | 0.000000 | 0.000000 | 0.825623 | 0.000001 |
| chronological | 2025 | 2024 | 2020, 2021, 2022, 2023 | -0.059185 | 0.274537 | 0.308818 | 0.210276 | 0.067964 | 0.827278 | 0.242607 |
| leave_two_seasons_out | 2023 | 2022 | 2020, 2021, 2024, 2025 | -0.034666 | 0.325017 | 0.215576 | 0.231680 | 0.001473 | 1.576443 | 1.276282 |
| leave_two_seasons_out | 2024 | 2023 | 2020, 2021, 2022, 2025 | -0.040030 | 0.330104 | 0.245033 | 0.294701 | -0.019105 | 0.775114 | 0.502072 |
| leave_two_seasons_out | 2025 | 2024 | 2020, 2021, 2022, 2023 | -0.059185 | 0.274537 | 0.308818 | 0.210276 | 0.067964 | 0.827278 | 0.242607 |

### Calibration-to-outer nominee gaps

The calibration-season score is in sample for each fitted temperature; the outer
season is held out. Positive outer-minus-calibration gaps mean worse outer loss.

| Protocol | Outer | Probability | Calibration Brier | Outer Brier | Brier gap | Calibration log loss | Outer log loss | Log-loss gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chronological | 2023 | Combined | 0.228786 | 0.282885 | 0.054099 | 0.650421 | 0.761758 | 0.111338 |
| chronological | 2023 | All-game calibration | 0.229180 | 0.298251 | 0.069071 | 0.651957 | 0.796841 | 0.144884 |
| chronological | 2023 | Nominee calibration | 0.228617 | 0.286830 | 0.058213 | 0.650178 | 0.770547 | 0.120370 |
| chronological | 2024 | Combined | 0.272063 | 0.243661 | -0.028402 | 0.740281 | 0.682414 | -0.057868 |
| chronological | 2024 | All-game calibration | 0.265798 | 0.242258 | -0.023540 | 0.726260 | 0.678593 | -0.047667 |
| chronological | 2024 | Nominee calibration | 0.250000 | 0.250000 | -0.000000 | 0.693147 | 0.693147 | -0.000000 |
| chronological | 2025 | Combined | 0.265853 | 0.256886 | -0.008967 | 0.729676 | 0.702744 | -0.026932 |
| chronological | 2025 | All-game calibration | 0.259012 | 0.249967 | -0.009045 | 0.713424 | 0.689609 | -0.023815 |
| chronological | 2025 | Nominee calibration | 0.247804 | 0.243483 | -0.004321 | 0.688761 | 0.679912 | -0.008848 |
| leave_two_seasons_out | 2023 | Combined | 0.231018 | 0.245685 | 0.014667 | 0.654927 | 0.684392 | 0.029465 |
| leave_two_seasons_out | 2023 | All-game calibration | 0.230684 | 0.271069 | 0.040385 | 0.655185 | 0.750528 | 0.095343 |
| leave_two_seasons_out | 2023 | Nominee calibration | 0.230031 | 0.256605 | 0.026575 | 0.653121 | 0.710733 | 0.057612 |
| leave_two_seasons_out | 2024 | Combined | 0.246399 | 0.301977 | 0.055578 | 0.688521 | 0.806450 | 0.117929 |
| leave_two_seasons_out | 2024 | All-game calibration | 0.237720 | 0.284978 | 0.047258 | 0.667710 | 0.766777 | 0.099067 |
| leave_two_seasons_out | 2024 | Nominee calibration | 0.233117 | 0.267675 | 0.034558 | 0.658276 | 0.729218 | 0.070942 |
| leave_two_seasons_out | 2025 | Combined | 0.265853 | 0.256886 | -0.008967 | 0.729676 | 0.702744 | -0.026932 |
| leave_two_seasons_out | 2025 | All-game calibration | 0.259012 | 0.249967 | -0.009045 | 0.713424 | 0.689609 | -0.023815 |
| leave_two_seasons_out | 2025 | Nominee calibration | 0.247804 | 0.243483 | -0.004321 | 0.688761 | 0.679912 | -0.008848 |

### Ranking and limitations

| Protocol | Diagnostic | Estimate, 95% interval and P+ |
| --- | --- | --- |
| chronological | Within-week accuracy points per 10 confidence points | 3.488215 [-4.890814, 11.292617]; P+ 0.8052 |
| chronological | Nominee minus rest accuracy points | -4.914550 [-18.965312, 9.107224]; P+ 0.2568 |
| leave_two_seasons_out | Within-week accuracy points per 10 confidence points | 3.524477 [-2.894562, 9.462283]; P+ 0.8619 |
| leave_two_seasons_out | Nominee minus rest accuracy points | -7.946035 [-21.788573, 6.092635]; P+ 0.1304 |

Measured family: eight paired Brier, eight paired log-loss and eight calibration-
gap cells, plus four ranking diagnostics: 28 recorded unresolved entries.
The five arms, four bands and three outer seasons are declared descriptive cuts;
no best cell was promoted. Records retain distinct populations and units and are
not pooled as independent evidence.

The upstream feature definitions and model archive had already been examined.
These outer-layer folds do not undo that selection. Leave-two-seasons-out rows
also use future training seasons. A monotone temperature cannot repair ranking:
it only changes confidence. Next: acquire additional timestamped weekly nominees
and test a declared calibration mechanism without adding post-hoc cut points.

Registry command: `nfl-ats weak-signals record --batch
artifacts/confidence_top_calibration/20260920_fixed/registry_batch.json --replace`.
All results and intervals above are measured from the saved summary artifact.

## Execution verification

Measured in this session: `python scripts/confidence_top_calibration.py` and
the registry batch command above exited successfully. `ruff format --check .`
passed for 1,114 files; `ruff check .` passed; `mypy src` passed for 235 source
files. `pytest -q --basetemp=.pytest-confidence-calibration` passed 4,529 tests
with nine skips. Commands used `.\.tools\uv.exe run --no-sync` with a writable
temporary UV cache. Temporary pytest output was removed after the run.

Measured: `nfl-ats publish-board` regenerated the four local pages. The rendered
diff updates the existing Sunday lock state from provisional New Orleans to
locked Denver, removes expired flip-line displays, and adds plain-language
research summaries to Findings. Git reports three changed lines in Findings
and two each in the other three generated pages. The unlocked Best Pick gap
rendered as 0.4 percentage points from the saved Sunday 12:35 PM ET inputs;
the current locked card omits the gap because its lock-time eligible pool is
not available. This research changed no served probability or pick policy.

Read/reviewed: the code diff adds no tests, fixture assumptions, or research-only
runtime assertions. The dashboard guards require valid eligible probabilities
and complete kickoff data. No existing test files were removed or expanded.
The implementation, registry, report and generated pages are ready for release
under the owner's standing instruction to commit and push at clear stopping points.
