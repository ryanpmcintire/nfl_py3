# SIM08 late-game diagnostic history

## State-matched neighbor hazards

**Measured:** The predeclared 12-look family used 7,641 late-game validation rows from 463 games,
including 1,774 tied rows from 114 games and 5,867 one-score rows from 455 games.
The cohort is Q4 with at most 300 seconds remaining, or overtime, with absolute pre-play
margin at most eight. Each interval uses 2,000 validation-game-cluster bootstrap replicates.

Frequency differences are percentage points; elapsed differences are seconds. Conditional
elapsed comparisons use actual-scoring query states with scoring candidates in that rank band.

| State | Ranks | Metric | Difference | 95% interval | probability_positive | Query rows / games |
|---|---|---|---:|---|---:|---|
| tied | 1_to_10 | Scoring frequency (pp) | -0.096 | [-1.153, +0.889] | 0.4305 | 1774 / 114 |
| tied | 1_to_10 | Conditional scoring elapsed (s) | +2.603 | [-5.398, +11.468] | 0.7510 | 100 / 93 |
| tied | 11_to_20 | Scoring frequency (pp) | -0.778 | [-1.771, +0.199] | 0.0550 | 1774 / 114 |
| tied | 11_to_20 | Conditional scoring elapsed (s) | -0.870 | [-8.560, +4.737] | 0.4400 | 96 / 91 |
| tied | 21_to_40 | Scoring frequency (pp) | -0.893 | [-1.876, +0.018] | 0.0290 | 1774 / 114 |
| tied | 21_to_40 | Conditional scoring elapsed (s) | -1.286 | [-7.553, +2.594] | 0.3790 | 107 / 99 |
| one_score | 1_to_10 | Scoring frequency (pp) | +0.692 | [+0.129, +1.275] | 0.9925 | 5867 / 455 |
| one_score | 1_to_10 | Conditional scoring elapsed (s) | +1.777 | [+1.124, +2.350] | 1.0000 | 268 / 230 |
| one_score | 11_to_20 | Scoring frequency (pp) | +0.750 | [+0.175, +1.345] | 0.9940 | 5867 / 455 |
| one_score | 11_to_20 | Conditional scoring elapsed (s) | +1.702 | [+1.048, +2.289] | 1.0000 | 268 / 229 |
| one_score | 21_to_40 | Scoring frequency (pp) | +0.715 | [+0.139, +1.289] | 0.9950 | 5867 / 455 |
| one_score | 21_to_40 | Conditional scoring elapsed (s) | +1.470 | [+0.878, +2.013] | 1.0000 | 288 / 248 |

**Measured:** All declared rank bands were present and the engine hash was unchanged.
Evidence: `.tmp/sim08_late_hazard_diagnostic.json`, declaration and helper of the same prefix.
**Inferred:** The one-score candidate pool warrants a temporal diagnostic. These comparisons do
not establish rollout bias or justify changing neighbor weights. Tied-state results remain
unresolved. No signal is closed or promoted; AGENTS.md requires a separate admissible closing ground.

## Chronological same-cell comparison

# SIM08 chronological hazard drift

## Declaration

Four comparisons were declared before signs were read: tied and one-score late-game states, each split into
scoring-transition frequency and elapsed time conditional on a scoring transition. The cohort is regulation Q4
with at most 300 seconds remaining plus overtime, with absolute home margin at most eight.
**Read:** Training is 2009-2014 and validation is 2015-2017, matching `scripts/sim04_engine.py:22-23`. State
cells use down, phase, distance,
field-position, score, time-bucket, timeout, and sampler-route fields derived through `build_transition_frame`
(`scripts/sim04_engine.py:550`) and the training-fitted fourth-down policy (`scripts/sim04_engine.py:329`).

Each estimate is the validation-minus-training difference over common cells, weighted by the harmonic overlap
in the two eras. Intervals and `probability_positive` come from 2,000 independent game-cluster bootstrap draws
by era. An interval crossing zero does not close a signal, and this diagnostic does not select or change any
served side. One fitted calibrated probability remains the side-selection contract.

## Measured

**Measured:** The four validation-minus-training estimates are:

- Tied scoring frequency: -0.95 pp, 95% [-1.94, +0.27] pp, P(positive) 0.065. Common-cell population:
  train 2,873 rows/192 games; validation 1,724/114.
- Tied conditional scoring elapsed: -4.42 s, 95% [-25.86, +9.71] s, P(positive) 0.351. Population:
  train 134 scores/124 games; validation 88/79.
- One-score scoring frequency: +0.17 pp, 95% [-0.43, +0.77] pp, P(positive) 0.706. Population:
  train 11,052 rows/866 games; validation 5,805/455.
- One-score conditional scoring elapsed: -2.11 s, 95% [-2.88, -1.43] s, P(positive) 0.000.
  Population: train 497 scores/427 games; validation 298/247.

**Measured:** The full eligible tied populations were train 3,038 rows/192 games/209 scoring transitions and
validation 1,774 rows/114 games/125 scoring transitions. One-score populations were train 11,354/866/661 and
validation 5,867/455/353.

**Measured:** This family has four looks. The prior candidate-pool diagnostic declared 12 looks, so the linked
diagnostics contain 16 comparisons in total. The prior one-score candidates scored 0.69-0.75 pp more often
than validation actual states but took 1.47-1.78 seconds longer when scoring. The +0.17 pp chronological
frequency difference has the wrong direction to explain that score-rate excess. The -2.11-second conditional
time difference has the direction and scale needed to explain the timing excess. Tied-state frequency drift
also points opposite the prior candidate-minus-actual difference, while tied conditional timing remains
imprecise.

## Interpretation and next action

**Inferred:** The one-score result supports chronological population drift as a plausible source of the
candidate pool's longer scoring-transition time. It does not establish causal drift, final rollout bias, or a
repair. **Measured:** The frequency evidence does not support drift as the source of the candidate-pool
scoring-rate excess. **Inferred:** Conditioning on scores, rounded state cells, common-cell restriction, and
the absence of trajectory-level matching limit the timing comparison.

Next, measure conditional scoring elapsed by candidate source season within the same one-score cells and
neighbor rank bands. A monotone recency pattern would identify whether source-season weighting is a justified
sampler repair; an absent pattern would redirect diagnosis to neighbor distance or unmodeled trajectory state.
This is one bounded artifact pass and requires no engine or full-simulation change.

Artifacts: `.tmp/sim08_chronological_hazard_declaration.json`,
`.tmp/sim08_chronological_hazard_diagnostic.py`, and `.tmp/sim08_chronological_hazard_diagnostic.json`.

**Measured:** The locked helper completed in 4.98 seconds and wrote the diagnostic artifact.
PowerShell returned status 1 because the redirected loky physical-core warning was promoted from stderr; the
saved log contains no traceback. The helper passes locked Ruff check and format check. The engine SHA-256
remained
`6085591667e45f4de3910e4e71ff48b17fb73879c3391ab21257ed21867c8873`.


## Source-season comparison within candidate state cells

# SIM08 source-season late-hazard diagnosis

## Declaration

**Measured** The declaration was saved before result inspection in
`.tmp/sim08_source_season_declaration.json`. The family contains three looks: the exact-state-cell adjusted
slope of conditional scoring-transition elapsed time on candidate source season for neighbor ranks 1-10,
11-20, and 21-40. Uncertainty uses 2,000 independent two-way cluster bootstrap draws over validation query
games and candidate source games. Probability positive is the share of finite draws with a slope above zero.

The population is actual-scoring validation transitions in regulation Q4 with at most 300 seconds remaining or
overtime and an absolute pre-play home margin of 1-8. Candidates must have positive points and are compared
within the same rounded sampler state cell and rank band. Score frequency is outside this conditional-timing
family.

## Result

| Rank band | Slope, seconds/source season | 95% cluster interval | Probability positive | Scoring exposures | Validation queries / games | Source games |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-10 | -0.968 | [-2.327, -0.267] | 0.0010 | 1,500 | 268 / 230 | 496 |
| 11-20 | -0.524 | [-1.010, -0.183] | 0.0005 | 1,514 | 272 / 229 | 515 |
| 21-40 | -0.600 | [-1.045, -0.292] | 0.0000 | 2,970 | 289 / 248 | 649 |

**Measured** All 2,000 draws were finite in every look. Total candidate exposures were 3,530, 3,530, and
7,060. Informative scoring exposures after exact-cell demeaning were 1,450, 1,457, and 2,922 across 221, 219,
and 250 cells. All six training seasons, 2009-2014, were present. The validation cohort before rank-band
availability filters contained 353 scoring transitions from 283 games in 307 rounded state cells.

**Measured** Within-cell centered elapsed means were largest for source season 2009 in every band: +4.16,
+2.03, and +2.23 seconds. Source years 2011-2014 were generally below their cell means. This shape cautions
against treating the linear slope as a smooth year-by-year law.

**Inferred** Older source seasons carrying longer scoring transitions within the same state cells and throughout
the neighbor ranks is a concrete temporal/population-drift mechanism for the candidate-pool timing excess found
in the prior diagnostic. This is a diagnosis of pools reached by actual validation trajectories. It does not
measure final simulated rollout bias: simulated query states, score-event frequency, repeated transition draws,
and terminal-state formation can change the rollout effect. It also does not authorize source-season weights
chosen from held-out outcomes.

## Next action and verification

The next bounded action is a predeclared training-only rolling-era replay within 2009-2014. For each query
season, candidates should come only from earlier seasons, and the same state-cell adjusted source-age timing
association should be estimated. Replication without validation outcomes would justify specifying an
out-of-season source-age challenger inside each fold before one full validation rollout comparison.

**Measured** The locked helper completed in 10.4 seconds and wrote
`.tmp/sim08_source_season_diagnostic.json`. Ruff formatting and checking passed. The engine SHA-256 before and
after the helper was `6085591667e45f4de3910e4e71ff48b17fb73879c3391ab21257ed21867c8873`; array alignment,
declared-look count, and source-season coverage checks all passed. No inference here closes a signal because an
interval crosses zero, and no side selection was performed; one fitted calibrated probability remains the only
admissible side selector.

## Training-only rolling-era replay

The declaration fixed six looks before result inspection: three neighbor-rank bands (1–10, 11–20, and 21–40) for all strictly earlier sources, plus the same three after excluding source season 2009. Each query season was handled separately, and candidates retained the engine's route and exact rounded sampler-state conditioning. The population was actual scoring transitions with an absolute pre-play home margin of 1–8 in the final five minutes of regulation or overtime. Uncertainty used 2,000 two-way bootstrap draws over query and source games.

**Measured** — `UV_CACHE_DIR=.tmp/uv-cache ./.tools/uv.exe run --no-sync python .tmp/sim08_rolling_era_diagnostic.py` completed once in 43.2 seconds. Every look had 2,000/2,000 finite bootstrap estimates.

| Sources | Rank | Seconds per source-age year | 95% interval | Probability positive |
| --- | ---: | ---: | ---: | ---: |
| All earlier | 1–10 | +1.474 | [+0.713, +3.001] | 1.0000 |
| All earlier | 11–20 | +1.429 | [+0.558, +2.894] | 1.0000 |
| All earlier | 21–40 | +1.379 | [+0.797, +2.233] | 1.0000 |
| Exclude 2009 | 1–10 | +0.830 | [+0.282, +1.391] | 0.9990 |
| Exclude 2009 | 11–20 | +0.741 | [+0.118, +1.315] | 0.9890 |
| Exclude 2009 | 21–40 | +0.802 | [+0.334, +1.286] | 0.9995 |

**Measured** — Query cohorts contained 561 actual scoring transitions from 454 season-summed games: 106/88 in 2010, 112/90 in 2011, 115/91 in 2012, 126/100 in 2013, and 102/85 in 2014.

| Sources | Rank | Queries / games | Candidate / scoring exposures | Source games / unique scoring transitions | Informative cells / scoring exposures |
| --- | ---: | ---: | ---: | ---: | ---: |
| All earlier | 1–10 | 443 / 384 | 5,610 / 2,395 | 450 / 554 | 281 / 1,877 |
| All earlier | 11–20 | 425 / 364 | 5,610 / 2,356 | 470 / 562 | 280 / 1,870 |
| All earlier | 21–40 | 480 / 402 | 11,220 / 4,657 | 573 / 736 | 336 / 3,838 |
| Exclude 2009 | 1–10 | 340 / 296 | 3,076 / 1,317 | 331 / 405 | 199 / 1,017 |
| Exclude 2009 | 11–20 | 329 / 285 | 3,053 / 1,325 | 345 / 408 | 194 / 1,002 |
| Exclude 2009 | 21–40 | 369 / 309 | 6,266 / 2,704 | 436 / 551 | 234 / 2,183 |

**Measured** — The scoring-exposure shares were 42.69%, 42.00%, and 41.51% for all earlier sources and 42.82%, 43.40%, and 43.15% after excluding 2009. This study separates conditional scoring elapsed from score frequency; it does not estimate an unconditional scoring hazard.

**Read** — `build_tables` and the prior diagnostic's candidate-pool path supplied the exact route and rounded-state matching used here. The independently checked SHA-256 for `scripts/sim04_engine.py` remained `6085591667e45f4de3910e4e71ff48b17fb73879c3391ab21257ed21867c8873`, equal to the declaration and artifact hashes.

**Measured** — Support is triangular. Season 2009 cannot be a query because it has no strictly earlier allowed source. The 2010 all-source cohort has only 2009 as a source and therefore no within-cell source-age variation. After excluding 2009, 2010 has no candidates and 2011 has only 2010, so the sensitivity slopes are identified by 2012–2014. The artifact confirms every candidate source season is strictly earlier than its query season.

**Inferred** — Longer conditional scoring elapsed for older sources repeats in chronological training-only pools across all three declared rank bands. The reduced but still positive exclude-2009 slopes show that 2009 amplifies the pattern and is not its sole cause. This actual-trajectory candidate-pool result does not establish final rollout bias, choose source-season weights, or select a served side. The six probability-positive estimates range from 0.989 to 1.000; finite bootstrap estimates do not prove certainty or authorize a serving change.

The next bounded action is a fixed-seed paired rollout diagnostic that logs selected source season and elapsed time under the unchanged engine, with one predeclared late-score timing endpoint. It should test whether the measured pool-age pattern reaches simulated terminal sequences before any sampling-weight proposal is considered.

**Review limitation:** Query and source games are resampled independently by role. A game can be a query in one season and a source for later seasons, so the intervals do not preserve that cross-role identity. The result remains a candidate-pool diagnostic, not a causal or rollout estimate.

## Rollout tracing and declared timeout sensitivity — 2026-09-27

**Measured:** `.tmp/sim08_rollout_trace.json` traces 18 validation games with
8 fixed seeds each. All 144 traced rollouts exactly match their uninstrumented
same-seed outputs; cap hits are zero. The 2,597 late one-score selections contain
154 scoring transitions. Exact state and route matching supplies only 2 selected
scoring transitions, 2 game-seed clusters, 2 actual scoring transitions from
2 games, and 2 state cells. The declared minimum of 30 transitions and 10 rollout
clusters fails. No effect, interval, or probability_positive was computed.

**Measured:** the separately declared timeout-omission sensitivity
(`.tmp/sim08_rollout_timeout_support.json`) supplies 9 selected scoring transitions,
9 game-seed clusters from 6 scheduled games, 8 actual transitions from 8 games,
and 7 cells. Both gates still fail; the elapsed estimate remains null. This is
one additional look, two cumulative rollout elapsed-endpoint looks. Timeout
composition differs: total variation distance 0.43056; largest cell share gap
is +0.22222 for one timeout on each side. This diagnostic does not justify
ignoring timeout state in the simulator.

Both artifacts preserve engine SHA-256
`6085591667e45f4de3910e4e71ff48b17fb73879c3391ab21257ed21867c8873`.
The real trace/support commands and Ruff checks exited 0. Helpers and declarations
are `.tmp/sim08_rollout_trace.py`, `.tmp/sim08_rollout_trace_declaration.json`,
`.tmp/sim08_rollout_timeout_support.py`, and
`.tmp/sim08_rollout_timeout_support_declaration.json`.

**Read:** the support helper refuses effect computation. Any later estimator must
resample scheduled games before seeds, cluster actual reference transitions by
game, and preserve shared game identities across both roles. The current sample
has one shared identity, `2016_01_CAR_DEN`. The independently sampled game-seed
proposal was never used to calculate an effect.

The next declared sample uses 96 games, 32 per validation season, and 8 seeds per
game without further matching relaxation. Its support report must include unique
scheduled games, timeout imbalance, and cells supported by only one reference
game before an effect is released. These support failures close no research
signal and authorize no engine weighting or served probability change.

## Fourth rollout endpoint: stabilized reference, 2026-09-27

**Measured:** the fixed 96-game, eight-seed expansion produced 768 rollouts with
zero caps and 67 matched selected scoring transitions in 66 rollouts from 40
scheduled games. The actual validation reference has 56 transitions in 55 games.
Selected-minus-actual elapsed discrepancy is **-0.750553 seconds**, with a paired
2,000-draw interval **[-2.727589, +1.173367]** and **probability_positive 0.2175**.
The raw exact-cell discrepancy is +1.940299 seconds; it is reported as a sensitivity,
not substituted for the stabilized estimator. This is the fourth rollout endpoint
look, separate from the earlier 25 neighbor/chronological/source-age looks.

Training-only alpha selection chose 32 from 0.5, 1, 2, 4, 8, 16, 32 using rolling
2010-2014 reference error; the final prior uses 589 games from 2009-2014. Validation
is 2015-2017. Hierarchical reference means use route/down/phase, then score/time,
then distance/field position. The bootstrap resamples shared game identities jointly
across roles within season, then seeds within scheduled games; reference priors are
recomputed. Neighbors, source choices, matching and alpha remain fixed, so intervals
omit selection uncertainty. Timeout matching remains omitted: total variation
0.287580 and maximum bucket gap 0.074360. Sparse exact reference cells remain a
limitation; these diagnostics are neither causal effects nor served ATS gains.

**Measured:** `nfl-ats weak-signals record` exited 0 and stored
`sim08_team_conditioned_late_clock_reference_bias_v1` as `unresolved_below_power`,
using nondirectional `elapsed_seconds_bias` units; `favours_candidate` is null.
Neither admissible closure nor a positive-control bound has been established.
AGENTS.md's research-closure rule therefore keeps the line open. No fitted weight,
calibration, prediction or served side changed.

The durable ignored replay bundle is
`artifacts/sim08_reference_bootstrap/20260927T055500Z/manifest.json`, SHA256
`f69145aa9e1dc994498994013a50fbc305ada6b3d8076da7433c11c9196a2ae3`.
All 17 bundled file hashes and 10 external input hashes were verified. It includes
the frozen declarations, support/reference scripts, effect JSON, engine, environment
lock and command logs. Replay requires the recorded input hashes. The shell wrapper
reported exit 1 after a joblib physical-core warning, while the completed effect JSON
records all 2,000 draws; this execution caveat is preserved with the raw logs.