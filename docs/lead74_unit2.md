# LEAD-74 unit 2: price changes at unchanged spreads

**Measured:** one replay; historical opener is the frozen pool-line proxy;
all findings are conditional on provider observed_at_utc being availability.
**Measured, decisive games first:** combined 8-5 on 13 disagreements;
accuracy 61.54 [33.33, 91.67]%; probability_positive versus 50% = 0.7966500000000001.

## Frozen declaration

Historical frozen-line standard: use the historical OPENER as the frozen pool-line
proxy, never pre-2026 Splash captures. Freeze the 2020-2025 REG population in
`artifacts/pick_probability/20260929T192747Z/per_game.parquet`; restrict to its
2023-2025 subset with admissible quote pairs; exclude opener pushes in every arm.
Target: home covers that opener, conditional on no push. Four-term base inputs and
discrete model probability remain frozen; refit calibration only on earlier seasons.
For each game/book take the latest complete Tuesday quote and latest complete quote
after Tuesday but strictly before min(kickoff, Sunday 12:45 Eastern). Require both
sides, opposite lines, finite American prices, valid snapshot/update clocks and
identical spreads across endpoints. Tuesday means local midnight through midnight.
No choosing an older deadline quote to manufacture an unchanged line. Feature is
the equal-book median change in no-vig home-cover log odds; use every matched book.
Exclude games with no matched book; no outcome-dependent source/book selection.
One joint logistic fit adds this scalar to intercept, model logit, signed market
move, move availability and composition sum. Same-population four-term refit is
the primary comparator; frozen discrete model, neutral 0.5 opener market and cached
pregame Elo are diagnostics (mark unavailable rather than reconstruct missing Elo).
Chronology-purged LOSO: scheduled folds 2023/2024/2025 train only earlier covered
seasons; unavailable when no earlier pressure variation. Ridge 0.001 fixed from
the served fitter; train-fold standardization; no selection grid or second-stage
calibration on the held-out season. Calibration is the joint fitted probability;
upstream model training/selection remain frozen. One full-population IS diagnostic.
Report opener accuracy, log loss, Brier, paired gains, decisive side-disagreement
record, per-fold natural coefficients, IS/OOS gaps, season stability and five
training-quantile reliability bands. Retain margin MAE only if cached predictions
exist; a cover-only recalibration supplies no new margin prediction. No side flip rule.
10,000 paired season-stratified week-block bootstraps, seed 20260929, fixed
predictions; 95% percentile intervals and probability_positive=P(gain>0)+0.5P(tie).
Reduce B=2 to B=1 (joint fit only), F=3, E=0: original accounting caps reporting
looks at (1+36)*(3+1)+25=173, down from 177; count unavailable cells and diagnostics.
One primary accuracy contrast; no new subgroup, tuning or outcome-driven amendment.
An interval crossing zero never closes the signal; no registry commands run here.

## Coverage and source audit

| Season | Frozen non-push games | Paired games | Matched books | Nonzero pressure |
|---|---:|---:|---:|---:|
| 2023 | 266 | 167 | 1141 | 108 |
| 2024 | 266 | 179 | 844 | 135 |
| 2025 | 267 | 182 | 901 | 139 |

**Measured:** 7428 snapshots passed hash, identity and clock checks;
9816 same-book endpoint pairs, 2975 unchanged-spread pairs,
540 source games; 0 duplicate-side rows excluded.
OOS population: 361 games and 36 season/week blocks.
Unavailable folds: {"2023": "No earlier covered season with identifiable pressure and both targets"}.
**Read:** the frozen probability artifact has no cached Elo probability; Elo cells remain unavailable.
**Measured:** no games or books were selected by outcome. Snapshot availability_proven remains false
under the strict receipt validator; this unit uses asof_consistent under the root's conditional rule.

## Paired chronological out-of-season results

**Measured:** 95% percentile intervals from 10,000 paired week-block resamples within season.
Accuracy is percent; accuracy gains are percentage points. Positive gains always favor combined.
Market 0.5 ties choose home consistently; its accuracy is a tie-convention diagnostic.

| Arm | Accuracy [95% interval] | Log loss [95% interval] | Brier [95% interval] |
|---|---:|---:|---:|
| combined | 53.19 [49.31, 56.76] | 0.7023 [0.6884, 0.7164] | 0.2540 [0.2473, 0.2608] |
| four_term | 52.35 [47.55, 56.90] | 0.7006 [0.6857, 0.7153] | 0.2532 [0.2460, 0.2604] |
| model | 53.74 [48.98, 58.60] | 0.6953 [0.6851, 0.7061] | 0.2511 [0.2460, 0.2564] |
| market | 50.14 [45.35, 54.84] | 0.6931 [0.6931, 0.6931] | 0.2500 [0.2500, 0.2500] |

| Comparator | Metric gain | Estimate [95% interval] | probability_positive |
|---|---|---:|---:|
| four_term | accuracy_points | 0.8310 [-1.3124, 2.8490] | 0.7967 |
| four_term | log_loss | -0.0017 [-0.0064, 0.0026] | 0.2294 |
| four_term | brier | -0.0007 [-0.0029, 0.0013] | 0.2614 |
| model | accuracy_points | -0.5540 [-6.1281, 4.9275] | 0.4158 |
| model | log_loss | -0.0070 [-0.0222, 0.0083] | 0.1797 |
| model | brier | -0.0029 [-0.0104, 0.0045] | 0.2180 |
| market | accuracy_points | 3.0471 [-4.4693, 10.3451] | 0.7872 |
| market | log_loss | -0.0092 [-0.0233, 0.0047] | 0.1003 |
| market | brier | -0.0040 [-0.0108, 0.0027] | 0.1251 |

**Measured:** unchanged cached model margin MAE 10.0087 [9.2833, 10.7689] points.
Cover recalibration supplies no new expected margin; no challenger margin gain is claimed.

## In-sample/out-of-season gaps

**Measured:** full-population IS diagnostics below use the same OOS rows.
Gap is OOS minus IS. All-season IS fits never select held-out picks.

| Arm | Metric | IS | OOS minus IS [95% interval] |
|---|---|---:|---:|
| combined | accuracy_points | 55.955679 | -2.770083 [-6.896552, 1.373626] |
| combined | log_loss | 0.688748 | 0.013585 [0.005298, 0.022179] |
| combined | brier | 0.247812 | 0.006141 [0.002114, 0.010291] |
| four_term | accuracy_points | 56.786704 | -4.432133 [-9.550869, 0.282486] |
| four_term | log_loss | 0.689804 | 0.010804 [0.002892, 0.019416] |
| four_term | brier | 0.248319 | 0.004922 [0.001052, 0.009137] |
| model | accuracy_points | 53.739612 | 0.000000 [0.000000, 0.000000] |
| model | log_loss | 0.695339 | 0.000000 [0.000000, 0.000000] |
| model | brier | 0.251054 | 0.000000 [0.000000, 0.000000] |
| market | accuracy_points | 50.138504 | 0.000000 [0.000000, 0.000000] |
| market | log_loss | 0.693147 | 0.000000 [0.000000, 0.000000] |
| market | brier | 0.250000 | 0.000000 [0.000000, 0.000000] |

Each fold's training IS and held-out OOS metrics:

| Fold | Arm | Metric | Train/test | IS | OOS | Gap |
|---|---|---|---:|---:|---:|---:|
| 2024 | combined | accuracy_points | 167/179 | 59.281437 | 56.424581 | -2.856856 |
| 2024 | combined | log_loss | 167/179 | 0.669020 | 0.702422 | 0.033402 |
| 2024 | combined | brier | 167/179 | 0.237955 | 0.253522 | 0.015566 |
| 2024 | four_term | accuracy_points | 167/179 | 61.077844 | 54.748603 | -6.329241 |
| 2024 | four_term | log_loss | 167/179 | 0.670162 | 0.698570 | 0.028408 |
| 2024 | four_term | brier | 167/179 | 0.238391 | 0.251902 | 0.013511 |
| 2024 | model | accuracy_points | 167/179 | 49.101796 | 53.631285 | 4.529489 |
| 2024 | model | log_loss | 167/179 | 0.691098 | 0.696933 | 0.005835 |
| 2024 | model | brier | 167/179 | 0.248965 | 0.251811 | 0.002846 |
| 2024 | market | accuracy_points | 167/179 | 46.706587 | 48.044693 | 1.338106 |
| 2024 | market | log_loss | 167/179 | 0.693147 | 0.693147 | 0.000000 |
| 2024 | market | brier | 167/179 | 0.250000 | 0.250000 | 0.000000 |
| 2025 | combined | accuracy_points | 346/182 | 57.803468 | 50.000000 | -7.803468 |
| 2025 | combined | log_loss | 346/182 | 0.680785 | 0.702245 | 0.021461 |
| 2025 | combined | brier | 346/182 | 0.243725 | 0.254377 | 0.010652 |
| 2025 | four_term | accuracy_points | 346/182 | 58.381503 | 50.000000 | -8.381503 |
| 2025 | four_term | log_loss | 346/182 | 0.680798 | 0.702612 | 0.021814 |
| 2025 | four_term | brier | 346/182 | 0.243732 | 0.254560 | 0.010828 |
| 2025 | model | accuracy_points | 346/182 | 51.445087 | 53.846154 | 2.401067 |
| 2025 | model | log_loss | 346/182 | 0.694117 | 0.693771 | -0.000346 |
| 2025 | model | brier | 346/182 | 0.250437 | 0.250309 | -0.000128 |
| 2025 | market | accuracy_points | 346/182 | 47.398844 | 52.197802 | 4.798958 |
| 2025 | market | log_loss | 346/182 | 0.693147 | 0.693147 | -0.000000 |
| 2025 | market | brier | 346/182 | 0.250000 | 0.250000 | 0.000000 |

## Season stability

| Season | Games | Metric gain versus four-term | Estimate [95% interval] | probability_positive |
|---|---:|---|---:|---:|
| 2024 | 179 | accuracy_points | 1.6760 [-2.2731, 5.3576] | 0.8134 |
| 2024 | 179 | log_loss | -0.0039 [-0.0134, 0.0048] | 0.2036 |
| 2024 | 179 | brier | -0.0016 [-0.0062, 0.0025] | 0.2332 |
| 2025 | 182 | accuracy_points | 0.0000 [-1.6484, 1.6393] | 0.5002 |
| 2025 | 182 | log_loss | 0.0004 [-0.0004, 0.0011] | 0.8222 |
| 2025 | 182 | brier | 0.0002 [-0.0002, 0.0006] | 0.8245 |

## Natural coefficients

**Measured:** slopes use original feature units; absent pressure is the base arm.

| Fold | Arm | Train/test | Intercept | Model logit | Composition | Move | Availability | Pressure |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2024 | combined | 167/179 | -0.196488 | 0.387552 | 0.479893 | -0.269486 | 0.000000 | 2.286876 |
| 2024 | four_term | 167/179 | -0.190697 | 0.466968 | 0.477187 | -0.125294 | 0.000000 | 0.000000 |
| 2025 | combined | 346/182 | -0.129739 | 0.384318 | 0.330117 | 0.137021 | 0.000000 | -0.237145 |
| 2025 | four_term | 346/182 | -0.130241 | 0.378647 | 0.330385 | 0.121689 | 0.000000 | 0.000000 |
| IS | combined | 528/528 | -0.067700 | 0.458546 | 0.210324 | 0.230083 | 0.000000 | -1.103360 |
| IS | four_term | 528/528 | -0.071173 | 0.430512 | 0.208189 | 0.161536 | 0.000000 | 0.000000 |

## Reliability

**Measured:** bands use earlier-season training prediction quintiles for each arm.
Tied cut points can leave empty bands; held-out outcomes never set cut points.

| Arm | Band | Games | Mean home probability | Home-cover fraction |
|---|---:|---:|---:|---:|
| combined | 1 | 76 | 0.3499 | 0.5132 |
| combined | 2 | 68 | 0.4326 | 0.4118 |
| combined | 3 | 75 | 0.4655 | 0.4400 |
| combined | 4 | 70 | 0.5141 | 0.5714 |
| combined | 5 | 72 | 0.6005 | 0.5694 |
| four_term | 1 | 77 | 0.3517 | 0.4935 |
| four_term | 2 | 60 | 0.4340 | 0.4167 |
| four_term | 3 | 87 | 0.4651 | 0.4828 |
| four_term | 4 | 64 | 0.5139 | 0.5312 |
| four_term | 5 | 73 | 0.5991 | 0.5753 |
| model | 1 | 61 | 0.4073 | 0.5246 |
| model | 2 | 56 | 0.4588 | 0.3929 |
| model | 3 | 66 | 0.4917 | 0.4394 |
| model | 4 | 73 | 0.5268 | 0.6164 |
| model | 5 | 105 | 0.5836 | 0.5048 |
| market | 1 | 0 | unavailable | unavailable |
| market | 2 | 0 | unavailable | unavailable |
| market | 3 | 0 | unavailable | unavailable |
| market | 4 | 0 | unavailable | unavailable |
| market | 5 | 361 | 0.5000 | 0.5014 |

## Interpretation and limits

**Inferred:** retain unresolved_below_power pending the orchestrator's serial registry write;
no positive-control bound or split-half reliability test was run. AGENTS.md permits no closure
merely from an interval crossing zero. Research closure and card promotion remain separate.
This conditional historical comparison does not authorize changing the served card.
**Measured:** B=1, F=3, E=0; 173 reserved fitting/reporting looks under the original formula,
including unavailable 2023/Elo cells; one primary accuracy contrast, no outcome-driven tuning.
Coefficients, calibration cells and correlated comparators are not independent discoveries.
**Inferred:** only two held-out seasons limit stability assessment. Intervals condition on the
frozen upstream model and predictions; they omit refitting and archive-correction uncertainty.
Earlier-season fitting here does not re-certify upstream model selection or feature construction.
Unchanged-spread book prices can be at a different line from the opener; their pressure is
a fitted opener predictor, never an alternative-line probability or automatic side flip.

## Reproduction

```bash
UV_NO_CACHE=1 .tools/uv.exe run --no-sync python scripts/lead74_unit2.py
```

**Measured:** predictions, quote endpoints, metrics, hashes and declaration are preserved
in tests/scratch/codex/lead74_unit2/. Record commands are in the lane; not executed here.
