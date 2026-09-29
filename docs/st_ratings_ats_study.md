# Special-teams ratings ATS study (declaration)

**Status:** declared 2026-09-29 BEFORE any special-teams rating was joined to a
game or any outcome was scored. Sections 1-8 are frozen; section 9 (Results) is
appended after the single run and nothing above it is edited afterwards.

Parent: `docs/st_player_ratings.md` section 10-11 (season-lagged ratings,
`artifacts/st_player_ratings/season_lagged_2019_2024/ratings.parquet`, unserved).
Script: `scripts/st_ratings_ats_study.py`. Lane: `docs/lanes/st-ratings-ats-study.md`.

## 1. Research rules in force

An interval crossing zero never closes this signal; `probability_positive` is
reported instead. Only a resolved wrong sign (whole interval on the wrong side)
or a positive control proven able to detect this size closes it; everything else
is `unresolved_below_power`. One fitted calibrated probability selects every
side; the ST term enters that probability as a fitted coefficient and no rule
flips a pick. The ratings stay unserved under every outcome.

## 2. Population

Regular-season games 2020-2025 with an opener line in the served opener
population (`jpm.load_opener_population`, pushes removed, exactly the population
of `scripts/four_term_probability_eval.py`). Target: home cover versus the
opener (`home_covered`). Grading is at the opener.

## 3. Pregame ST rating per team-game (leakage-safe)

For team T in game g of season s: take T's most recent earlier regular-season
game in season s (week 1 has none, so the rating is missing). The participants
are the players on T's side of every special-teams play in that earlier game
(`classify_special_teams_unit` not null; offense_players belong to
possession_team, defense_players to the other side). Team rating = mean of the
season-s ratings (`target_season == s`) of those participants; a participant
without a rating row counts 0 (neutral prior). Ratings exist for targets
2022-2025 only. Games in 2020-2021 therefore carry no rating.
Term: `st_diff` = home rating minus away rating; missing on either side = 0.
Only game-day-earlier data is used; the ratings for target s never use season s.

## 4. Model and folds

Baseline arm `m5`: `x0_model_logit, flag_sum, move, move_available` (the served
four-term inputs) fitted by `jpm.loso_arm` (fixed L2 1e-3, standardized).
Candidate arm `m5_st`: the same four plus `st_diff` as one additional fitted
term. LOSO folds by season (2020-2025, six folds); the held-out season is scored
out of sample. In-sample arms are fitted and scored on all games; gap =
in-sample minus out-of-sample.

## 5. Metrics and uncertainty

Log loss, Brier, opener-graded accuracy (pick = p >= 0.5) for both arms, OOS and
in-sample, on (a) all population games and (b) the covered subset (both teams
have >= 50% of prior-game ST participants rated). Paired improvement (baseline
minus candidate loss; accuracy points for accuracy) with week-blocked bootstrap
(`week_blocked_bootstrap`, 20000 samples, seed 20260914), 95% interval and
`probability_positive`. Fold coefficients of `st_diff` and their sign stability
are reported per fold.

## 6. Coverage accounting

Report by season: games, games with a prior ST game, mean fraction of
participants rated, games in the covered subset, share with nonzero `st_diff`.
Early windows (2022, 2023 have 141 and 131 source plays) are expected sparse;
they are retained, not dropped, and reported separately in coverage.

## 7. Looks

One family `st_rating_ats_study`. Declared looks: 3 metrics x 2 populations = 6
paired OOS improvements; the primary read is log loss on all games. Six
per-season fold coefficients are diagnostics, not extra decision looks.
Decisive-game record: the count of games where the two arms pick different
sides and each arm's record there is reported before the headline.

## 8. Decision rule

Nothing is served or promoted. Record the primary read as
`unresolved_below_power` unless the whole interval is on the wrong side
(`wrong_sign_resolved`). No parameter is revised after the run.

### Implementation audit saved before the resumed run

The stopped worker left sections 1-8 above and an unexecuted partial script.
They remain unchanged. Missing on either side makes the differential zero;
coverage averages both sides and distinguishes missing prior participation,
missing season ratings, zero rated players, and the declared 50% coverage floor.
The production rating validator and strictly earlier game dates enforce timing.
This is a retrospective LOSO study graded at the opener, not a simulation of
information available when that opener was first posted. Later-season training
folds and subsequently collected historical inputs limit prospective claims.

The required descriptive comparisons use market probability 0.5 (home on a
tie) and the unchanged model-only opener probability on these same games.
Calibration uses five fixed equal-width probability bins for each of market,
model-only, baseline, and candidate. Every metric reports in-sample minus OOS
for both fitted arms; all six coefficients, including the intercept, are saved
per candidate fold, with baseline coefficients alongside. Per-season metrics,
decisive records, and coefficient signs describe stability without selecting a
season. Bootstrap draws condition on the fitted predictions, not refitted models.

Expanded look accounting (no extra selection): six primary-family OOS cells,
six corresponding in-sample cells, 12 season-by-metric descriptive cells per
metric (36 total), four scored arms, 14 logistic fits (12 LOSO, two pooled),
and 20 calibration bins = **86 looks**. The six OOS comparisons in section 7
remain the declared effect family; no diagnostic becomes a new decision rule.
No alternative feature, threshold, penalty, or population is searched.

The command writes its complete JSON result, including prediction-level rows,
to the temporary run log; only this document, the script, and the lane are edited.
Registry commands are handed to the orchestrator and are never executed here.

## 9. Results (single resumed run, 2026-09-29)

**Measured:** the real command completed with exit 0 and empty stderr:

```powershell
$env:UV_CACHE_DIR='.tmp/uv-cache'; $env:PYTHONDONTWRITEBYTECODE='1'
.tools/uv.exe run --no-sync python scripts/st_ratings_ats_study.py > .tmp/st-ratings-ats-study.log 2> .tmp/st-ratings-ats-study.err
```

All result tables below are **measured** from that log: 1,503 nonpush games,
107 season-week blocks, six held-out seasons, 20,000 bootstrap draws per
population, two numerical threads. The covered subset has 632 games / 54 blocks.
The run retains all 1,503 prediction rows in the same log; registry writes are pending.

### Decisive games, then the primary result

**Measured:** on 69 differing picks, baseline 40-29 and candidate 29-40.
Within the covered subset: 67 differing picks, baseline 39-28 and candidate 28-39.
These are historical forced picks, not per-game probabilities or profit estimates.

| Population | Metric | OOS improvement | 95% week-block interval | probability_positive |
|---|---|---:|---|---:|
| all | log_loss | -0.00298055 | [-0.00662267, +0.00015283] | 0.030750 |
| all | brier | -0.00140499 | [-0.00314813, +0.00010285] | 0.032700 |
| all | accuracy | -0.73186959 | [-1.64273410, +0.19841598] | 0.058925 |
| covered | log_loss | -0.00712085 | [-0.01529401, +0.00040326] | 0.032200 |
| covered | brier | -0.00335787 | [-0.00727229, +0.00025694] | 0.035500 |
| covered | accuracy | -1.74050633 | [-3.81100914, +0.34784124] | 0.053925 |

Positive improvement means lower loss or higher accuracy; accuracy improvements
are percentage points. **Inferred:** the primary all-game log-loss result
(-0.00298055, [-0.00662267, +0.00015283], probability_positive 0.03075) leaves
the declared signal `unresolved_below_power`. AGENTS.md research closure rules
and section 8 require a resolved wrong sign or an adequately powered positive
control to close it. Neither closing ground was established. This run supplies
no promotion or serving decision; the orchestrator must record the six cells.

### In-sample, out-of-sample, and gap

Accuracy here is a fraction; gap is in-sample minus OOS, exactly as declared.

| Population | Arm | Metric | In-sample | OOS | Gap |
|---|---|---|---:|---:|---:|
| all | base | log_loss | 0.68272120 | 0.68403019 | -0.00130899 |
| all | base | brier | 0.24481829 | 0.24543604 | -0.00061775 |
| all | base | accuracy | 0.56952761 | 0.56353959 | +0.00598802 |
| all | cand | log_loss | 0.68250836 | 0.68701075 | -0.00450238 |
| all | cand | brier | 0.24471638 | 0.24684103 | -0.00212465 |
| all | cand | accuracy | 0.56753160 | 0.55622089 | +0.01131071 |
| covered | base | log_loss | 0.68419543 | 0.68613665 | -0.00194122 |
| covered | base | brier | 0.24552954 | 0.24640833 | -0.00087879 |
| covered | base | accuracy | 0.56170886 | 0.56012658 | +0.00158228 |
| covered | cand | log_loss | 0.68370481 | 0.69325750 | -0.00955269 |
| covered | cand | brier | 0.24529540 | 0.24976621 | -0.00447081 |
| covered | cand | accuracy | 0.55696203 | 0.54272152 | +0.01424051 |

| Population | Metric | In-sample improvement | 95% interval | probability_positive | IS minus OOS effect |
|---|---|---:|---|---:|---:|
| all | log_loss | +0.00021284 | [-0.00066795, +0.00106908] | 0.688850 | +0.00319339 |
| all | brier | +0.00010191 | [-0.00033087, +0.00052301] | 0.684550 | +0.00150689 |
| all | accuracy | -0.19960080 | [-0.99469496, +0.59446161] | 0.312725 | +0.53226880 |
| covered | log_loss | +0.00049063 | [-0.00162304, +0.00253754] | 0.675350 | +0.00761147 |
| covered | brier | +0.00023414 | [-0.00080235, +0.00123999] | 0.670800 | +0.00359202 |
| covered | accuracy | -0.47468354 | [-2.36593060, +1.39751553] | 0.307575 | +1.26582278 |

### Same-population market and model-only baselines

The market reference is the unadjusted 0.5 home-cover probability; its tied
accuracy mechanically selects home. The model-only reference uses the archived
raw opener cover probability. The opener artifact matches the active model's
feature-table hash (**measured**); this is the loader's matching criterion.

| Population | Reference | Log loss | Brier | Accuracy |
|---|---|---:|---:|---:|
| all | market | 0.69314718 | 0.25000000 | 0.49634065 |
| all | model_only | 0.69701134 | 0.25177326 | 0.53825682 |
| covered | market | 0.69314718 | 0.25000000 | 0.50474684 |
| covered | model_only | 0.69566916 | 0.25119315 | 0.52848101 |

### Coverage

Missing prior participation and missing season ratings overlap; do not sum them.
Mean rated share averages both teams over all games, with absent sides assigned
zero. Partial coverage remains neutral-imputed in the all-game population.

| Season | Games | Both prior ST | Missing prior ST | Missing rating season | Both any rated | Mean rated share | At least 50% each | Below floor with prior | Nonzero differential |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 220 | 5 | 215 | 220 | 0 | 0.000% | 0 | 5 | 0 (0.00%) |
| 2021 | 236 | 5 | 231 | 236 | 0 | 0.000% | 0 | 5 | 0 (0.00%) |
| 2022 | 248 | 4 | 244 | 0 | 4 | 8.753% | 3 | 1 | 4 (1.61%) |
| 2023 | 266 | 250 | 16 | 0 | 250 | 51.199% | 124 | 126 | 250 (93.98%) |
| 2024 | 266 | 253 | 13 | 0 | 253 | 74.872% | 253 | 0 | 253 (95.11%) |
| 2025 | 267 | 252 | 15 | 0 | 252 | 77.098% | 252 | 0 | 252 (94.38%) |

**Measured:** 456 games lack an available target-season rating partition;
734 do not have participant sets for both prior ST games. Only 632/1,503 (42.05%) meet the
predeclared coverage floor. **Inferred:** sparse early personnel classifications
and source ratings limit what this particular pooled ST feature can establish.
Missing seasons were retained, and no threshold was changed after these counts.

### Fold coefficients

Coefficients are on original input units, including the intercept. The script
uses training-fold standardization and fixed L2=0.001. Full coefficient standard
errors and approximate Wald intervals are retained in the log.

| Arm | Held season | Train | Test | Intercept | Model logit | Flag sum | Move | Move available | ST differential |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 2020 | 1283 | 220 | -0.051424 | +0.316272 | +0.243374 | +0.209216 | +0.030961 | n/a |
| base | 2021 | 1267 | 236 | -0.079581 | +0.092657 | +0.278920 | +0.213509 | +0.061988 | n/a |
| base | 2022 | 1255 | 248 | -0.066419 | +0.242619 | +0.244781 | +0.210831 | +0.047851 | n/a |
| base | 2023 | 1237 | 266 | -0.060026 | +0.249640 | +0.248111 | +0.224825 | +0.029959 | n/a |
| base | 2024 | 1237 | 266 | -0.078006 | +0.176075 | +0.278211 | +0.249901 | +0.058927 | n/a |
| base | 2025 | 1236 | 267 | -0.075298 | +0.209232 | +0.299989 | +0.158927 | +0.059090 | n/a |
| cand | 2020 | 1283 | 220 | -0.051611 | +0.316918 | +0.245539 | +0.210812 | +0.029495 | -120.589156 |
| cand | 2021 | 1267 | 236 | -0.079890 | +0.093111 | +0.281295 | +0.215162 | +0.060586 | -123.412502 |
| cand | 2022 | 1255 | 248 | -0.066637 | +0.243436 | +0.247092 | +0.212412 | +0.046402 | -119.715264 |
| cand | 2023 | 1237 | 266 | -0.060263 | +0.250346 | +0.250447 | +0.227436 | +0.027693 | -121.765590 |
| cand | 2024 | 1237 | 266 | -0.077991 | +0.176136 | +0.278187 | +0.249770 | +0.059014 | +5.461086 |
| cand | 2025 | 1236 | 267 | -0.075125 | +0.216845 | +0.307675 | +0.156106 | +0.058568 | -513.593692 |

| Held season | ST coefficient | Approximate 95% Wald interval |
|---|---:|---|
| 2020 | -120.589156 | [-420.682382, +179.504070] |
| 2021 | -123.412502 | [-423.619286, +176.794282] |
| 2022 | -119.715264 | [-419.692551, +180.262023] |
| 2023 | -121.765590 | [-422.170317, +178.639137] |
| 2024 | +5.461086 | [-344.880669, +355.802841] |
| 2025 | -513.593692 | [-1106.065324, +78.877940] |

**Measured:** the ST coefficient is positive in 1/6 folds; its original-unit
scale spans -513.59 to +5.46. **Inferred:** the signs and magnitudes do not
establish stable weighting. The large original-unit values reflect tiny shrunk
ratings; magnitude alone is not a probability or a reason for closure.

### Season stability

These are descriptive differences, not selected seasons or new decision rules.
Accuracy differences are percentage points; loss differences favor the candidate
when positive. Covered 2020/2021 cells are empty and remain reserved looks.

| Population | Season | Games | Log-loss improvement | Brier improvement | Accuracy points | Differing picks | Baseline record | Candidate record |
|---|---:|---:|---:|---:|---:|---:|---|---|
| all | 2020 | 220 | +0.00004802 | +0.00002294 | +0.45455 | 1 | 0-1 | 1-0 |
| all | 2021 | 236 | -0.00002702 | -0.00001344 | +0.00000 | 0 | 0-0 | 0-0 |
| all | 2022 | 248 | +0.00006294 | +0.00003117 | +0.00000 | 0 | 0-0 | 0-0 |
| all | 2023 | 266 | +0.00000635 | +0.00001254 | -0.37594 | 1 | 1-0 | 0-1 |
| all | 2024 | 266 | -0.00011041 | -0.00005099 | -0.75188 | 2 | 2-0 | 0-2 |
| all | 2025 | 267 | -0.01674864 | -0.00790663 | -3.37079 | 65 | 37-28 | 28-37 |
| covered | 2022 | 3 | +0.00055898 | +0.00029026 | +0.00000 | 0 | 0-0 | 0-0 |
| covered | 2023 | 124 | -0.00005944 | -0.00001301 | +0.00000 | 0 | 0-0 | 0-0 |
| covered | 2024 | 253 | -0.00011723 | -0.00005418 | -0.79051 | 2 | 2-0 | 0-2 |
| covered | 2025 | 252 | -0.01771834 | -0.00836400 | -3.57143 | 65 | 37-28 | 28-37 |

### Reliability table

Five fixed bins per arm; upper edge belongs to the next bin, except 1.0.
Empty bins are shown and counted. These observed rates are descriptive.

| Arm | Probability bin | Games | Mean probability | Observed home-cover rate |
|---|---|---:|---:|---:|
| market | 0.0-0.2 | 0 | n/a | n/a |
| market | 0.2-0.4 | 0 | n/a | n/a |
| market | 0.4-0.6 | 1503 | 0.500000 | 0.496341 |
| market | 0.6-0.8 | 0 | n/a | n/a |
| market | 0.8-1.0 | 0 | n/a | n/a |
| model | 0.0-0.2 | 0 | n/a | n/a |
| model | 0.2-0.4 | 176 | 0.367219 | 0.500000 |
| model | 0.4-0.6 | 1272 | 0.492751 | 0.490566 |
| model | 0.6-0.8 | 55 | 0.626278 | 0.618182 |
| model | 0.8-1.0 | 0 | n/a | n/a |
| base | 0.0-0.2 | 0 | n/a | n/a |
| base | 0.2-0.4 | 104 | 0.362340 | 0.413462 |
| base | 0.4-0.6 | 1274 | 0.493407 | 0.486656 |
| base | 0.6-0.8 | 125 | 0.637384 | 0.664000 |
| base | 0.8-1.0 | 0 | n/a | n/a |
| cand | 0.0-0.2 | 0 | n/a | n/a |
| cand | 0.2-0.4 | 133 | 0.359462 | 0.421053 |
| cand | 0.4-0.6 | 1217 | 0.492920 | 0.487264 |
| cand | 0.6-0.8 | 153 | 0.648406 | 0.633987 |
| cand | 0.8-1.0 | 0 | n/a | n/a |

### Reproduction and limits

**Measured:** 86 declared looks were reserved: six OOS effect cells, six paired
in-sample cells, 36 season/metric cells, four scored arms, 14 logistic fits,
and 20 calibration bins. Only the six OOS cells form the recorded effect family.
Bootstrap uncertainty is conditional on the fitted predictions and uses joint
season-week resampling; it does not include model refitting or source-rating
estimation uncertainty. A prospective chronological validation remains separate.

**Measured:** syntax parsing passed; source has zero comments and zero docstrings.
The saved pre-run declaration hash and executed script hash were verified before
appending these results. The log retains partition hashes, coefficients, source
provenance, calibration cells, and all prediction-level rows. No tests were added.
The prediction rows reproduce aggregate metrics within 2.6e-12; assigned-file
whitespace and the six unexecuted registry commands' PowerShell syntax passed.

Pre-run declaration SHA-256: 8101ee53e1216331f614fa803e1ac2a4ec1151d7c628cd1373a9790e3c7b1b0d.
Executed script SHA-256: fa56b8104e4e9651d297b1997eeb8de7af4469e83aae4f9069bb40de133129aa.
Ratings SHA-256: 623130dd430cbd1c9a74871e2083c7bbaf2089ac6ed1f1750b831b47823913c1.
Run-log SHA-256: 232765ffd0255cadd3cad2a4c981c4352c5d6269dd2786c5fec47c9d6e3955d4.

Exact unexecuted registry commands are in `docs/lanes/st-ratings-ats-study.md`.
