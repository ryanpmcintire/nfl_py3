# LEAD-88 unit 2: fitted response to total news

**Measured:** one replay of the protocol saved before outcome access in `docs/lanes/lead88.md`.
`UV_CACHE_DIR=tests/scratch/codex/lead88_unit2/uv-cache .tools/uv.exe run --no-sync python scripts/lead88_unit2.py`.
No serving change or registry command was executed.

## Primary result and exact closer comparison

**Measured:** on 101 last games, candidate is closer 17 times,
farther 15 times, and tied 69 times.
Unconditional closer share: 16.831683 [10.121839, 25.575785]% (exact 95% interval).
Among unequal errors: 53.125000 [34.743681, 70.906018]%;
exact two-sided binomial null p=0.860050, null closer probability 0.5.
The exact null conditions on unequal errors; ties remain in the unconditional share and MAE.

## Held-out MAE against actual combined score

**Measured:** 1333/1343 paired games and 101/101 last games have a defined served guess.
The production guard declines 10 historical guesses; all rows remain in the artifact,
including all 33 missing probability rows (33 scoreable). No outcome-based exclusion was made.
All-game MAE below means every paired game with a defined production baseline. The declared full-population
contrast is not identifiable where that baseline declines a guess; this coverage deviation is not a protocol retune.
**Measured diagnostic:** 26/255 zero-move games changed integer guesses
(3 last games) because the declared candidate recentres the lattice on the served integer total.
**Inferred:** the measured effect combines total-news response with score reprojection; it does not isolate
the news mechanism. No alternative mapping was scored after this discovery. A new declaration must
preserve the served guess when the fitted adjustment is zero before any further outcome comparison.
**Measured:** lower MAE is better; paired week-block 95% percentile intervals.
| Population | Games | Arm | MAE [95% interval] |
| --- | --- | --- | --- |
| last | 101 | candidate | 10.336634 [8.851485, 11.950495] |
| last | 101 | served | 10.405941 [8.930693, 12.009901] |
| last | 101 | tuesday_half_up | 10.445545 [8.960396, 12.049505] |
| last | 101 | deadline_half_up | 10.237624 [8.752475, 11.841832] |
| all | 1333 | candidate | 10.360090 [10.025190, 10.696684] |
| all | 1333 | served | 10.368342 [10.029432, 10.704374] |
| all | 1333 | tuesday_half_up | 10.357089 [10.020105, 10.696715] |
| all | 1333 | deadline_half_up | 10.271568 [9.936692, 10.607121] |

| Population | Served MAE minus candidate MAE [95% interval] | probability_positive |
| --- | --- | --- |
| last | 0.069307 [-0.089109, 0.237624] | 0.800550 |
| all | 0.008252 [-0.045902, 0.061013] | 0.629300 |

**Inferred:** proposed status `unresolved_below_power`; no pool-rank gain or serving change is established.
AGENTS.md requires a refuted mechanism or a powered positive control to close a signal; neither was
established here. The registry commands await orchestrator execution; this is not a settled verdict.

## Training, folds and stability

**Measured:** full-data optimistic IS coefficient b=0.800000000.
| Held season | Train games | Held games | b | Last MAE gain [95% interval] | Last probability_positive | All MAE gain [95% interval] |
| --- | --- | --- | --- | --- | --- | --- |
| 2020 | 1136 | 197 | 0.666667 | -0.153846 [-0.538462, 0.153846] | 0.211750 | 0.121827 [-0.010311, 0.257282] |
| 2021 | 1117 | 216 | 0.666667 | 0.187500 [-0.125000, 0.562500] | 0.846100 | 0.000000 [-0.133035, 0.125000] |
| 2022 | 1118 | 215 | 0.800000 | 0.111111 [-0.166667, 0.444444] | 0.756100 | -0.013953 [-0.129468, 0.109093] |
| 2023 | 1099 | 234 | 0.500000 | 0.222222 [-0.166667, 0.666667] | 0.860350 | 0.072650 [-0.025424, 0.162996] |
| 2024 | 1095 | 238 | 1.000000 | 0.222222 [-0.222222, 0.666667] | 0.830800 | -0.058824 [-0.195021, 0.078841] |
| 2025 | 1100 | 233 | 1.000000 | -0.222222 [-0.666667, 0.222222] | 0.150500 | -0.055794 [-0.208696, 0.090129] |

| Population | Candidate IS MAE [95% interval] | IS gain [95% interval] | OOS gain | OOS minus IS gain [95% interval] |
| --- | --- | --- | --- | --- |
| last | 10.247525 [8.772277, 11.861386] | 0.158416 [-0.009901, 0.326733] | 0.069307 | -0.089109 [-0.178218, -0.009901] |
| all | 10.324831 [9.989186, 10.661919] | 0.043511 [-0.009652, 0.094951] | 0.008252 | -0.035259 [-0.062039, -0.008160] |

## Baseline, retained games and lineage

**Read:** `artifacts/margin_predictions/2026-week-04-20260929T192403Z/tiebreaker.json:9`
records integer guesses; lines 15 and 24-27 name the lattice, joint total method and one-point shade.
`src/nfl_ats/tiebreaker.py:385` and `src/nfl_ats/score_lattice.py:278` define the serving recipe.
**Measured:** this replay reconstructs that recipe historically; these are not archived served cards.
Tuesday total + 0.1 x joint total residual - 1 is the baseline continuous total. Production
`challenger_centre`, `score_lattice` and `pick_consistent_top_score` select the integer score.
Margin uses the served opener residual; side uses the frozen four-term LOSO probability.
Candidate centre is served integer total + b x observed total move, under the same margin/side guards.
Joint hyperparameters and shading are inherited from serving, not selected on replay outcomes.
Joint models fit only earlier-week finals before Tuesday; target spread/total inputs use opener/Tuesday
quotes. Lattice histories also end before Tuesday. No pool captures before 2026 are required.
**Measured:** 1343 paired games / 101 last games; all 33 missing
four-term rows retained (33 opener pushes). Frozen probability reproduction
maximum error 1.11e-16; reconstructed composition flags match cached rows.
Margin training calendar-day cutoffs precede Tuesday for 1300 games and
deadline for all 1343. Joint baseline reconstruction fits: 107.
**Inferred limitations:** upstream probability/margin/features are retrospective archives.
A prior training date does not prove feature-vintage or completion-time availability. Four-term and
response LOSO fits include future seasons for earlier holdouts; this is not a prospective rolling test.
Baseline histories for later games can contain another response-fold season. This is conditional
evaluation of a fixed served recipe, not independently nested validation of its upstream training.
The inherited low-side shade is not revalidated here. Total-quote clocks are verified; upstream
forecast-feature clocks are not newly certified. No new ATS scoring or probability calibration is claimed.

## Frozen protocol and look accounting

(frozen before outcomes; unit 2 amendment)
**Reported (owner):** baseline is served `guess_home + guess_away`, using the score lattice
and current served-total/side-consistency recipe, not market half-up rounding.
Half-up Tuesday and deadline market totals are secondary comparators only.
Population: all 1,343 paired 2020â€“2025 REG games, retaining the 33 absent from the
 decisive-only four-term artifact. Historical opener is the frozen pool-line proxy.
Target: actual home + away score. Last-game MAE is primary; all-game MAE secondary.
Terms: candidate continuous centre = served integer guess + b Ã— (deadline âˆ’ Tuesday total).
Fit one no-intercept b by absolute loss on all training games; no clipping or tuning.
Choose the integer candidate with the same score-lattice rule, fixed margin and side.
Six retrospective LOSO folds (hold out each season), plus one full-population optimistic IS
fit. This owner-requested bounded replay supersedes Protocol C's three outer years for
the response only; no new ATS fit or separate discrete-law challenger in this unit.
Reconstruct historical served guesses with pregame archived margin/features and prior
finals; use the frozen four-term probability for sides, reconstruct missing rows with
the same fitted recipe. Verify clocks and record any upstream retrospective limitation.
Compare candidate, served lattice, half-up Tuesday, half-up deadline. Report all/last MAE,
candidate-versus-served paired improvement, fold b, IS/OOS improvement gap, last-game
closer/worse/ties and exact two-sided binomial null conditional on unequal errors.
Report unconditional closer share and Clopperâ€“Pearson 95% interval as well.
10,000 paired season-stratified week-block bootstrap draws, seed 88; 95% percentile
intervals; probability_positive=P(improvement>0)+0.5P(improvement=0), fixed predictions.
Look budget: 7 coefficient fits + 4 arms Ã— 2 populations Ã— 7 pooled/fold panels = 56 MAEs,
14 candidate/served effects + 2 IS effects + 2 gaps + 1 closer comparison = 82 looks.
Parent Protocol C remains 713 looks; these 82 are its bounded amended total-response
reporting family, not an extra search. No post-outcome protocol changes.
One fitted calibrated probability fixes the side; zero crossing never closes a signal.

**Measured:** 82 reporting looks, one response specification, seven b fits. Fixed baseline reconstruction
fits are disclosed separately; they are not tuned arms. IS MAE and gain express the same two IS comparisons.
10,000 paired season-stratified week-block draws, seed 88, fixed predictions; exact zero draws get half weight
in probability_positive. Intervals omit training/model-selection uncertainty; no multiplicity adjustment.

## Season MAE panels

| Season | Population | Arm | MAE [95% interval] |
| --- | --- | --- | --- |
| 2020 | last | candidate | 10.307692 [5.615385, 17.307692] |
| 2020 | last | served | 10.153846 [5.384615, 17.230769] |
| 2020 | last | tuesday_half_up | 9.923077 [5.076923, 16.846154] |
| 2020 | last | deadline_half_up | 10.153846 [5.461538, 16.923077] |
| 2020 | all | candidate | 10.101523 [9.525497, 10.656599] |
| 2020 | all | served | 10.223350 [9.688103, 10.740014] |
| 2020 | all | tuesday_half_up | 10.187817 [9.631559, 10.681608] |
| 2020 | all | deadline_half_up | 9.913706 [9.296473, 10.507118] |
| 2021 | last | candidate | 11.187500 [8.625000, 13.687500] |
| 2021 | last | served | 11.375000 [8.687500, 14.000000] |
| 2021 | last | tuesday_half_up | 11.312500 [8.437500, 14.126562] |
| 2021 | last | deadline_half_up | 11.062500 [8.312500, 13.687500] |
| 2021 | all | candidate | 10.916667 [9.971959, 11.878523] |
| 2021 | all | served | 10.916667 [9.969153, 11.908665] |
| 2021 | all | tuesday_half_up | 10.902778 [9.967879, 11.907866] |
| 2021 | all | deadline_half_up | 10.837963 [9.906092, 11.812221] |
| 2022 | last | candidate | 8.000000 [5.444444, 10.833333] |
| 2022 | last | served | 8.111111 [5.555556, 10.944444] |
| 2022 | last | tuesday_half_up | 8.555556 [6.055556, 11.500000] |
| 2022 | last | deadline_half_up | 8.000000 [5.388889, 10.888889] |
| 2022 | all | candidate | 10.427907 [9.489970, 11.282556] |
| 2022 | all | served | 10.413953 [9.432319, 11.321105] |
| 2022 | all | tuesday_half_up | 10.483721 [9.463947, 11.421277] |
| 2022 | all | deadline_half_up | 10.474419 [9.584813, 11.286380] |
| 2023 | last | candidate | 10.000000 [7.055556, 13.333333] |
| 2023 | last | served | 10.222222 [7.333333, 13.444444] |
| 2023 | last | tuesday_half_up | 10.833333 [7.944444, 14.055556] |
| 2023 | last | deadline_half_up | 10.333333 [7.444444, 13.555556] |
| 2023 | all | candidate | 10.235043 [9.545064, 10.974476] |
| 2023 | all | served | 10.307692 [9.623966, 11.050006] |
| 2023 | all | tuesday_half_up | 10.452991 [9.897407, 11.054882] |
| 2023 | all | deadline_half_up | 10.239316 [9.584345, 10.904358] |
| 2024 | last | candidate | 11.888889 [7.666667, 16.500000] |
| 2024 | last | served | 12.111111 [8.000000, 16.500000] |
| 2024 | last | tuesday_half_up | 11.777778 [7.833333, 16.000000] |
| 2024 | last | deadline_half_up | 11.666667 [7.611111, 16.000000] |
| 2024 | all | candidate | 9.831933 [9.085708, 10.564116] |
| 2024 | all | served | 9.773109 [9.072019, 10.469040] |
| 2024 | all | tuesday_half_up | 9.605042 [8.922425, 10.292576] |
| 2024 | all | deadline_half_up | 9.651261 [8.945799, 10.351132] |
| 2025 | last | candidate | 10.722222 [6.888889, 14.834722] |
| 2025 | last | served | 10.500000 [6.611111, 14.666667] |
| 2025 | last | tuesday_half_up | 10.222222 [6.222222, 14.388889] |
| 2025 | last | deadline_half_up | 10.277778 [6.166667, 14.611111] |
| 2025 | all | candidate | 10.665236 [9.717486, 11.627143] |
| 2025 | all | served | 10.609442 [9.659443, 11.598347] |
| 2025 | all | tuesday_half_up | 10.549356 [9.566477, 11.557457] |
| 2025 | all | deadline_half_up | 10.527897 [9.537769, 11.544710] |

## Saved evidence

`tests/scratch/codex/lead88_unit2/predictions.parquet`: every game, guesses, outcomes and response fold.
`summary.json`: coefficients, panels, exact null and hashes; `protocol.md`: pre-outcome declaration.
Record commands are in the lane only and require serial orchestrator execution.
**Measured:** initial command failed in baseline construction before any response fit or score. The sole
response replay then retained unavailable-baseline rows as unscorable instead of fabricating served guesses.
