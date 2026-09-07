# weak_stack_v5: away-market FluView only

Predeclared 2026-09-07 before scoring.

Read: docs/weak_stack_v3.md reports the situational arm probability_positive
0.3415; docs/weak_stack_v4.md:159 and registry entry
weak_stack_v4_forecast_weather_opener_probability_rule report -1.0645 accuracy
points, 95% [-2.6882, +0.5953], probability_positive 0.0956 (prior measurements,
unverified by rerun). Read: registry/weak_signals.json entries
fluview_away_market_elevated and fluview_home_market_elevated both carry
reliability 0.9814475666016571; this is trait reliability, not probability_positive.
Read: fluview_away_market_elevated_on_production already records a CLOSE-graded
stack (0.000 points, [-1.1562,+1.1606], probability_positive 0.403). Thus
"never built" is superseded. The existing elevated feature uses thresholds
computed over the full historical panel (src/nfl_ats/fluview_production_feature.py:93).

## Frozen construction

Exactly one new column: `fluview_away_market_ili_asof`, continuous reported ILI
percentage in the visiting team's state. No home feature, elevated threshold,
weather, situational flag, tuning, or outcome-selected transformation.
`weak_stack_v5` = existing `weak_stack` plus that column, both targets.
Builder: additive `attach_fluview_away_asof_features` in
src/nfl_ats/fluview_production_feature.py, called by the v5 evaluation script.
The continuous form avoids using future predictor distributions to choose an
historical elevated threshold; it tests the same illness family, not the exact
previous elevated-cell construction (inferred design rationale).

Source: data/raw/fluview/20260820T003258Z/fluview_raw.parquet, state-region,
epiweek, issue/lag and release_date vintages. Reuse build_checkpoint_tables and
asof_lookup in scripts/fluview_battery_screen.py. STATE_BY_TEAM comes from
scripts/fluview_battery_ingest.py. Join via game_id and game date, not a naive NFL
week equals CDC epiweek join. Decision timestamp is Tuesday 00:00 of each game's
calendar week (prior Tuesday for Monday games), earlier than the pool deadline.
Only releases STRICTLY BEFORE that date are admitted: date-only releases become
available on the following midnight. Unknown release dates never qualify.
Latest known epiweek wins; late revisions to older epiweeks cannot replace it.
Missing state coverage and non-Home locations stay null. No global imputation:
existing chronological model training folds handle missingness. Future releases,
including revisions and future extremes, must leave prior-game features unchanged.

## Population and estimands

Pinned incumbent: artifacts/opener_evaluation/20260907T122841Z/per_game.parquet;
metadata and active manifest must agree in model id and feature-table digest.
Recompute incumbent from the digest-matched production table and assert all
saved opener probabilities/picks agree before comparing. Use current Gaussian
probability method explicitly (v4's omitted method defaulted to ECDF).
Candidate calls nfl_ats.clv.opener_pick_evaluation unchanged, weekly chronological
refits, same alpha/regressor/calibration/minimum training count as pinned baseline.
Read: clv.py:2027 inherits close-era non-spread inputs; this approximation remains
disclosed and shared by both arms.

Primary population: the 1,537 saved opener archive games with non-null v5 feature;
report eligible count/1537, nonpush denominator, weeks, seasons and per-season
coverage. Grade forced probability picks at the saved opener, exclude pushes
identically via existing correctness nulls. Paired accuracy points = 100 times
candidate-minus-incumbent mean correctness on identical nonpush rows.
Report sign rule and close grade only as diagnostics. Full archive policy read
falls back to incumbent on missing-feature games, reported separately.

Uncertainty: existing clv.week_blocked_bootstrap, 20,000 draws, seed 20260817,
95% percentile interval, probability_positive = fraction of draws > 0.
Within-week correlation is ZERO; no correlation estimate, design effect, or
padding. Season bootstrap is secondary. Per-season deltas disclose stability.
Frozen-pick null: 200 seeded within-week permutations of realized home-cover
labels, holding both arms' fitted probabilities/picks fixed. Report null mean,
95% interval and observed percentile; never refit or retune against null draws.

## Rotation and decision

Measured: rotation declare/assign for weak_stack_v5_fluview_away inherits
fluview_elevated_on_production, acknowledges mined 2018-2025, assigned 2020-2021.
Record that window's eligible paired result separately through rotation record
(the installed CLI spelling of record-look). Full 2020-2025 primary is an explicitly
reused descriptive look, NOT an untouched confirmation window. Prior elevated-away
close result and home sibling opener looks on 2020-2023 compound multiplicity;
no quantified posterior correction is invented. No independent vote or simple
pooling with these correlated measurements. Record full and assigned-window results
through weak-signals record with shared family and overlap notes.
The forced-pick EV rule favours v5 if primary probability_positive > 0.5;
no 0.90/95% promotion gate. Do not activate, publish, or change the card.

## Binding taxonomy (verbatim from fleet brief)

An interval or CI that contains zero is NEVER grounds to reject, fail, or close an experiment. At this
evaluator's ~2-point resolution, "contains zero" is the EXPECTED outcome for a real small signal. Only two
grounds ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (2) bounded by a positive control proven able to
detect an effect that size. Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the binary "contains zero". The
registry code hard-rejects inadmissible closures; if a record command errors, the verdict is wrong, not
the validator. Never use 95%, 0.90, or any threshold as a DECISION bar; decide on expected value
(`probability_positive` above 0.5 favours playing it), thresholds only govern what docs may CLAIM.
Grade at the OPENER (the pool's grade); a close-graded number may never veto a play. Within-week game
correlation is ZERO by owner mandate: never estimate or pad it. Never say something "needs N more
games": the data is fixed and the project is model-limited.


## Results (added after scoring and registry recording)

Measured: `artifacts/weak_stack_v5_opener_eval/20260907T140503Z/opener_summary.json`
from `.\.tools\uv.exe run --no-sync python scripts/weak_stack_v5_opener_eval.py`:
**Keep the incumbent on the predeclared full-archive decision read**: candidate
52.037% versus incumbent 53.109%, delta **-1.072194 accuracy points**, week-blocked
95% **[-2.374339, +0.217395]**, **probability_positive 0.0492**. There are 1,431
feature-covered games / 1,537 archive games (**93.1034%**), 1,399 nonpush games,
107 weeks. This supports the incumbent EV decision, not a mechanism closure.

Measured in the same artifact: full-archive fallback-to-incumbent policy scores
52.3619% versus 53.3599%, delta -0.998004 points, 95% [-2.210352,+0.201748],
probability_positive 0.0492, 1,503 nonpush games. This policy is a diagnostic only;
no active model, played profile, forecast or card was changed.

Measured in the same artifact: assigned 2020-2021 window is directionally different:
+0.490196 points, 95% [-1.492537,+2.500157], probability_positive 0.64455;
416 eligible / 466 archived games, 408 nonpush, 35 weeks. It favours v5 within
that window, but it is not substituted for the predeclared full-archive decision.
Both windows and primary result are disclosed; no post-result season selection.

Measured: 85 nonpush picks flip in the primary comparison; frozen-pick null
mean +0.090779 points, 95% [-1.072194,+1.358113], observed percentile 0.035.
Season-blocked primary interval [-2.330363,0.000000], probability_positive
0.02165 is secondary. Sign-rule opener delta -0.071480 points and close
probability-rule delta -1.783167 points are descriptive only.

Measured per-season eligible/archive counts and accuracy-point deltas from the
same artifact: 2020 196/227, 0.0000; 2021 220/239, +0.9217; 2022 241/255,
-3.8462; 2023 259/272, -1.1858; 2024 258/272, -0.7937; 2025 257/272,
-1.1905. These are descriptive season magnitudes, not separate confirmations.

Measured: `weak-signals record` accepted
`weak_stack_v5_fluview_away_opener` and
`weak_stack_v5_fluview_away_opener_2020_2021`, both `unresolved_below_power`;
`rotation record` accepted the latter assigned-window measurement with verdict
`unresolved` under family `weak_stack_v5_fluview_away`. The CLI command is named
`record`, not `record-look`; no validator changes were needed. Registry notes
explicitly disclose overlap and prohibit treating these as independent votes.

Measured verification: touched-file Ruff format/check clean; `mypy src` clean
(246 files); `pytest -q -n 2 -p no:cacheprovider` with a lane-specific scratch
basetemp on tests/test_weak_stack_v5.py and tests/test_margin.py: 21 passed.
The baseline replay asserted all saved opener probabilities and picks match.

Inferred caveats: continuous ILI is a different construction than the previous
full-panel elevated indicator, so this result cannot settle every illness
representation. Repeated archive use and prior family selection discount claims
of independent evidence. Read: clv.py:2027 retains close-era non-spread inputs
for both arms. The historical state mapping is inherited; a future live use
would require contemporaneous source capture and a prediction-time merge.
