# LEAD-69 unit 1

**Measured:** final verification of the protocol saved before outcomes in
`lead69_protocol.md` and the lane.
Source population: `artifacts/pick_probability/20260929T192747Z/per_game.parquet`; quote source: `artifacts/sharp_book_weighted_movement/spread_quotes.parquet`.
The source artifact was active when inspected; it is pinned for this experiment.

## Decisive games first

| Mode | Arm vs base | Wins | Losses | Different picks |
| --- | --- | --- | --- | --- |
| is | horizon | 5 | 7 | 12 |
| is | slots | 10 | 11 | 21 |
| oos | horizon | 8 | 6 | 14 |
| oos | slots | 23 | 19 | 42 |

## Source inventory

```json
{
  "population": "artifacts/pick_probability/20260929T192747Z/per_game.parquet",
  "quotes": "artifacts/sharp_book_weighted_movement/spread_quotes.parquet",
  "starting_games": 799,
  "starting_by_season": {
    "2023": 266,
    "2024": 266,
    "2025": 267
  },
  "leader_capture_rows": 966036,
  "invalid_or_postkick_capture_rows": 1568,
  "missing_kickoffs": [],
  "structural_snf_mnf_exclusions": 114,
  "structural_exclusions_by_season": {
    "2023": 37,
    "2024": 38,
    "2025": 39
  },
  "eligible_games": 685,
  "eligible_by_season": {
    "2023": 229,
    "2024": 228,
    "2025": 228
  },
  "slot_counts": {
    "friday": 5,
    "other": 2,
    "saturday": 21,
    "sunday_early": 423,
    "sunday_late": 175,
    "thursday": 59
  },
  "hours_range": [
    25.0,
    125.0
  ],
  "postdeadline_capture_rows_removed": 66,
  "kickoff_revision_games": 456,
  "kickoff_revision_examples": [
    "2023_01_DET_KC",
    "2023_02_SEA_DET",
    "2023_02_KC_JAX",
    "2023_09_MIA_KC",
    "2023_02_MIN_PHI"
  ],
  "maximum_deadline_contraction_minutes": 180.0,
  "slot_labels_updated_after_deadline_contraction": 20,
  "conservative_deadline_additional_removals": 48,
  "missing_safe_move_games": [],
  "rebuilt_move_differs_from_stored": 2,
  "largest_move_rebuild_difference": 0.5
}
```

The starting 799 is the source population; SNF/MNF exclusions are structural,
before outcomes.
The comparator is the served four-term specification refitted on identical
eligible rows and rebuilt
deadline-safe moves. This is not a replay of the historical served card.
The production move begins Wednesday; the horizon begins at the Tuesday-noon freeze.
Snapshot and observation timestamps precede the deadline; book updates precede
observation.

## Aggregate performance and in-sample/out-of-sample gap

Accuracy is a fraction; line movement is points. Intervals are exact season-
cluster bootstrap 95%
percentiles across all 27 ordered resamples. Only three seasons are available.

| Arm | Metric | IS [95% interval] | OOS [95% interval] | IS minus OOS |
| --- | --- | --- | --- | --- |
| base | accuracy | 0.573723 [0.559284, 0.584760] | 0.560584 [0.546711, 0.575651] | 0.013139 |
| base | log_loss | 0.678350 [0.677077, 0.679338] | 0.680301 [0.678285, 0.682271] | -0.001951 |
| base | brier | 0.242779 [0.241968, 0.243477] | 0.243705 [0.242537, 0.244874] | -0.000926 |
| base | line_move | 0.595985 [0.561056, 0.637876] | 0.586496 [0.533936, 0.654497] | 0.009489 |
| horizon | accuracy | 0.570803 [0.556852, 0.584315] | 0.563504 [0.549196, 0.581551] | 0.007299 |
| horizon | log_loss | 0.677854 [0.676686, 0.679429] | 0.681281 [0.678522, 0.684424] | -0.003427 |
| horizon | brier | 0.242535 [0.241778, 0.243526] | 0.244140 [0.242651, 0.245827] | -0.001605 |
| horizon | line_move | 0.571168 [0.531670, 0.612017] | 0.595255 [0.539273, 0.666662] | -0.024088 |
| slots | accuracy | 0.572263 [0.559336, 0.586799] | 0.566423 [0.555482, 0.577546] | 0.005839 |
| slots | log_loss | 0.671684 [0.664542, 0.677648] | 0.689719 [0.676426, 0.703523] | -0.018036 |
| slots | brier | 0.240257 [0.237350, 0.242602] | 0.245899 [0.242298, 0.248944] | -0.005642 |
| slots | line_move | 0.552920 [0.533973, 0.571517] | 0.539781 [0.516795, 0.557823] | 0.013139 |
| model_only | accuracy | 0.516788 [0.497990, 0.539185] | 0.516788 [0.497990, 0.539185] | 0.000000 |
| model_only | log_loss | 0.698687 [0.694864, 0.702146] | 0.698687 [0.694864, 0.702146] | 0.000000 |
| model_only | brier | 0.252642 [0.250751, 0.254388] | 0.252642 [0.250751, 0.254388] | 0.000000 |
| model_only | line_move | 0.113504 [0.084161, 0.138229] | 0.113504 [0.084161, 0.138229] | 0.000000 |
| market | accuracy | 0.513869 [0.507237, 0.519196] | 0.513869 [0.507237, 0.519196] | 0.000000 |
| market | log_loss | 0.693147 [0.693147, 0.693147] | 0.693147 [0.693147, 0.693147] | 0.000000 |
| market | brier | 0.250000 [0.250000, 0.250000] | 0.250000 [0.250000, 0.250000] | 0.000000 |
| market | line_move | 0.139781 [0.020815, 0.260508] | 0.139781 [0.020815, 0.260508] | 0.000000 |

## Paired gains versus four-term base

Positive means improvement for every metric.

| Mode | Arm | Metric | Gain [95% interval] | probability_positive |
| --- | --- | --- | --- | --- |
| is | horizon | accuracy | -0.002920 [-0.010270, 0.003436] | 0.277778 |
| is | horizon | log_loss | 0.000495 [-0.000674, 0.001910] | 0.703704 |
| is | horizon | brier | 0.000243 [-0.000336, 0.000943] | 0.703704 |
| is | horizon | line_move | -0.024818 [-0.034106, -0.013508] | 0.000000 |
| is | slots | accuracy | -0.001460 [-0.012746, 0.007822] | 0.351852 |
| is | slots | log_loss | 0.006666 [-0.000526, 0.014680] | 0.962963 |
| is | slots | brier | 0.002522 [-0.000532, 0.005859] | 0.851852 |
| is | slots | line_move | -0.043066 [-0.067781, -0.023355] | 0.000000 |
| oos | horizon | accuracy | 0.002920 [-0.002485, 0.007794] | 0.796296 |
| oos | horizon | log_loss | -0.000980 [-0.002285, 0.000110] | 0.148148 |
| oos | horizon | brier | -0.000436 [-0.001017, 0.000055] | 0.148148 |
| oos | horizon | line_move | 0.008759 [0.005336, 0.012165] | 1.000000 |
| oos | slots | accuracy | 0.005839 [-0.003433, 0.017164] | 0.722222 |
| oos | slots | log_loss | -0.009418 [-0.024737, 0.004531] | 0.148148 |
| oos | slots | brier | -0.002194 [-0.006087, 0.001737] | 0.148148 |
| oos | slots | line_move | -0.046715 [-0.106860, 0.009576] | 0.148148 |

## Season stability

| Season | Arm | N | Record | accuracy | log_loss | brier | line_move |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | base | 229 | 133-96 | 0.580786 | 0.677687 | 0.242212 | 0.685590 |
| 2023 | horizon | 229 | 135-94 | 0.589520 | 0.678056 | 0.242391 | 0.698690 |
| 2023 | slots | 229 | 133-96 | 0.580786 | 0.707767 | 0.249353 | 0.558952 |
| 2023 | model_only | 229 | 116-113 | 0.506550 | 0.699722 | 0.253057 | 0.126638 |
| 2023 | market | 229 | 119-110 | 0.519651 | 0.693147 | 0.250000 | -0.010917 |
| 2024 | base | 228 | 124-104 | 0.543860 | 0.680455 | 0.243714 | 0.541667 |
| 2024 | horizon | 228 | 125-103 | 0.548246 | 0.680212 | 0.243594 | 0.550439 |
| 2024 | slots | 228 | 129-99 | 0.565789 | 0.673191 | 0.240871 | 0.506579 |
| 2024 | model_only | 228 | 125-103 | 0.548246 | 0.693516 | 0.250110 | 0.072368 |
| 2024 | market | 228 | 118-110 | 0.517544 | 0.693147 | 0.250000 | 0.135965 |
| 2025 | base | 228 | 127-101 | 0.557018 | 0.682773 | 0.245195 | 0.531798 |
| 2025 | horizon | 228 | 126-102 | 0.552632 | 0.685589 | 0.246444 | 0.536184 |
| 2025 | slots | 228 | 126-102 | 0.552632 | 0.688121 | 0.247458 | 0.553728 |
| 2025 | model_only | 228 | 113-115 | 0.495614 | 0.702820 | 0.254758 | 0.141447 |
| 2025 | market | 228 | 115-113 | 0.504386 | 0.693147 | 0.250000 | 0.294956 |

## Natural-unit coefficients

No held-out row affects its fold means, scales, or coefficients.

### base

| Held season / IS | Training N | intercept | model_logit | composition_flag_sum | market_move_toward_home | market_move_available |
| --- | --- | --- | --- | --- | --- | --- |
| in_sample | 685 | 0.007085425 | 0.093010593 | 0.257237282 | 0.222241174 | 0.000000000 |
| 2023 | 456 | -0.023947510 | 0.101588106 | 0.228033915 | 0.241441772 | 0.000000000 |
| 2024 | 457 | -0.004530759 | -0.064601668 | 0.243977197 | 0.232904349 | 0.000000000 |
| 2025 | 457 | 0.048437895 | 0.271898062 | 0.297421444 | 0.200650110 | 0.000000000 |

### horizon

| Held season / IS | Training N | intercept | model_logit | composition_flag_sum | market_move_toward_home | market_move_available | move_log_hours |
| --- | --- | --- | --- | --- | --- | --- | --- |
| in_sample | 685 | 0.005769159 | 0.096832003 | 0.259500409 | 1.761497294 | 0.000000000 | -0.323625264 |
| 2023 | 456 | -0.023269621 | 0.112085990 | 0.230785383 | 2.307983973 | 0.000000000 | -0.434823142 |
| 2024 | 457 | -0.004782751 | -0.064835876 | 0.244300617 | 0.371261192 | 0.000000000 | -0.029133083 |
| 2025 | 457 | 0.045407918 | 0.284572096 | 0.300002518 | 3.316705751 | 0.000000000 | -0.653789708 |

### slots

| Held season / IS | Training N | intercept | model_logit | composition_flag_sum | market_move_toward_home | market_move_available | move_slot_thursday | move_slot_friday | move_slot_saturday | move_slot_sunday_late | move_slot_other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| in_sample | 685 | 0.016719280 | 0.173205997 | 0.251368889 | 0.164059069 | 0.000000000 | 0.287156818 | -17.483695221 | 0.728509443 | 0.107783061 | 10.958896621 |
| 2023 | 456 | -0.008330510 | 0.190442623 | 0.207288744 | 0.127760484 | 0.000000000 | 0.354778270 | -9.734077113 | 2.979105965 | 0.190633905 | 10.603193776 |
| 2024 | 457 | -0.001206918 | -0.035575199 | 0.249866264 | 0.172042620 | 0.000000000 | 0.097283356 | -21.490777485 | 0.189363519 | 0.196675082 | 0.000000000 |
| 2025 | 457 | 0.059220966 | 0.376729764 | 0.295198597 | 0.194328190 | 0.000000000 | 0.641515907 | -16.887986710 | 0.421128563 | -0.059708990 | 10.664876049 |

### Fold coefficient stability

| Arm | Term | Minimum | Maximum | Positive folds | Negative folds |
| --- | --- | --- | --- | --- | --- |
| base | intercept | -0.023947510 | 0.048437895 | 1 | 2 |
| base | model_logit | -0.064601668 | 0.271898062 | 2 | 1 |
| base | composition_flag_sum | 0.228033915 | 0.297421444 | 3 | 0 |
| base | market_move_toward_home | 0.200650110 | 0.241441772 | 3 | 0 |
| base | market_move_available | 0.000000000 | 0.000000000 | 0 | 0 |
| horizon | intercept | -0.023269621 | 0.045407918 | 1 | 2 |
| horizon | model_logit | -0.064835876 | 0.284572096 | 2 | 1 |
| horizon | composition_flag_sum | 0.230785383 | 0.300002518 | 3 | 0 |
| horizon | market_move_toward_home | 0.371261192 | 3.316705751 | 3 | 0 |
| horizon | market_move_available | 0.000000000 | 0.000000000 | 0 | 0 |
| horizon | move_log_hours | -0.653789708 | -0.029133083 | 0 | 3 |
| slots | intercept | -0.008330510 | 0.059220966 | 1 | 2 |
| slots | model_logit | -0.035575199 | 0.376729764 | 2 | 1 |
| slots | composition_flag_sum | 0.207288744 | 0.295198597 | 3 | 0 |
| slots | market_move_toward_home | 0.127760484 | 0.194328190 | 3 | 0 |
| slots | market_move_available | 0.000000000 | 0.000000000 | 0 | 0 |
| slots | move_slot_thursday | 0.097283356 | 0.641515907 | 3 | 0 |
| slots | move_slot_friday | -21.490777485 | -9.734077113 | 0 | 3 |
| slots | move_slot_saturday | 0.189363519 | 2.979105965 | 3 | 0 |
| slots | move_slot_sunday_late | -0.059708990 | 0.196675082 | 2 | 1 |
| slots | move_slot_other | 0.000000000 | 10.664876049 | 2 | 0 |

## OOS reliability

Fixed equal-width home-probability bins; upper edge 1.0 is included.

| Arm | Bin | N | Mean forecast | Observed home cover |
| --- | --- | --- | --- | --- |
| base | 0.0-0.2 | 0 | NA | NA |
| base | 0.2-0.4 | 54 | 0.344201 | 0.425926 |
| base | 0.4-0.6 | 533 | 0.505078 | 0.491557 |
| base | 0.6-0.8 | 95 | 0.648942 | 0.673684 |
| base | 0.8-1.0 | 3 | 0.805411 | 1.000000 |
| horizon | 0.0-0.2 | 0 | NA | NA |
| horizon | 0.2-0.4 | 59 | 0.347408 | 0.440678 |
| horizon | 0.4-0.6 | 527 | 0.505206 | 0.493359 |
| horizon | 0.6-0.8 | 97 | 0.655819 | 0.670103 |
| horizon | 0.8-1.0 | 2 | 0.866634 | 0.500000 |
| slots | 0.0-0.2 | 7 | 0.056828 | 0.428571 |
| slots | 0.2-0.4 | 57 | 0.352914 | 0.473684 |
| slots | 0.4-0.6 | 515 | 0.506740 | 0.491262 |
| slots | 0.6-0.8 | 99 | 0.653150 | 0.646465 |
| slots | 0.8-1.0 | 7 | 0.871339 | 0.714286 |
| model_only | 0.0-0.2 | 0 | NA | NA |
| model_only | 0.2-0.4 | 43 | 0.373810 | 0.534884 |
| model_only | 0.4-0.6 | 594 | 0.499818 | 0.511785 |
| model_only | 0.6-0.8 | 48 | 0.621592 | 0.520833 |
| model_only | 0.8-1.0 | 0 | NA | NA |
| market | 0.0-0.2 | 0 | NA | NA |
| market | 0.2-0.4 | 0 | NA | NA |
| market | 0.4-0.6 | 685 | 0.500000 | 0.513869 |
| market | 0.6-0.8 | 0 | NA | NA |
| market | 0.8-1.0 | 0 | NA | NA |

## Interpretation, looks, and limitations

**Measured:** 2 predeclared research looks: one-parameter horizon and descriptive
slot sensitivity.
Diagnostic accounting: 5 probability arms (3 fitted, 2 fixed); 12 numerical fits
(3 arms x 4 folds/IS);
25 fixed calibration cells; 40 aggregate metric cells; 60 season-metric cells; 16
paired gain cells;
4 decisive-record cells. Disclosed looks are not independent confirmations or
selection searches.
**Inferred:** classification remains `unresolved_below_power`; a zero crossing
cannot close a signal.
Only three seasons and previously selected base features limit inference. LOSO is
the declared season
holdout design, not an untouched chronological outer test. The descriptive arm is
not promotable here.
Line-move grading shares market information with the fitted move term and is descriptive.
Move availability is constant here: its coefficient is unidentifiable separately
from the intercept;
standardized ridge returns zero. No deployment or registry mutation occurred.
Registry commands await the orchestrator. See `lead69_predictions.md` for every
scored row.

## Verification and execution history

**Measured:** the requested single execution expanded to one inventory halt, an initial fit pass,
and one corrective fit pass. The first halt read no outcomes. The last pass corrected 20 stale slot
labels after conservative deadline contraction; categories and model terms were unchanged.
All four primary horizon effects, intervals, and probability_positive values are exactly unchanged.
The final artifact contains 12 numerical fits; 24 fits were executed across both scoring passes.
There were 2 declared model comparisons; no parameter or population selection followed outcomes.

**Measured:** 685 unique saved prediction rows; zero capture/snapshot deadline violations and zero
late-slot labels with early deadlines. Independent arithmetic from saved predictions reproduces
all four primary summaries within 1e-10. Scoped Ruff format/check pass; no new tests were added.

**Read:** weak-signals CLI accepts a JSON batch with shared fields and cells.
`lead69_registry_payload.md` is JSON under the packet-permitted filename, containing eight correlated
metric records (two comparisons times four metrics); the exact command is in the lane and was not run.
**Measured:** all eight payload cells pass the production registry's in-memory validator; zero registry writes.

**Inferred:** revised kickoff clocks, including conservative contractions up to 180 minutes, limit
the timing interpretation. A future source audit should reconcile published schedules against capture
clock revisions before any serving decision. This is not evidence that the horizon mechanism is refuted.
