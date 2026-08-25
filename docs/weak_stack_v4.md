# weak_stack_v4 — forecast-weather feature arm (predeclaration)

Written **before** `scripts/weak_stack_v4_opener_eval.py` scored anything.
Only population/coverage counts were examined pre-freeze (**measured** this
session: the kickoff-nearest forecast archive holds 4,431 REG rows for
2009-2025 with 4,379 non-null forecast values, 98.8%). No accuracy, delta,
interval or `probability_positive` for the candidate profile was computed
beforehand.

## Binding closing-grounds taxonomy (verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) refuted mechanism — a RESOLVED wrong sign (whole interval on
the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". The registry code
hard-rejects inadmissible closures; if a record command errors, the verdict is
wrong, not the validator.

## Why this exists, and why now

`weak_stack_v3` was refused at the opener on EV (**reported**, prior session,
unverified by me: 53.03% vs 53.36%, delta −0.333, `probability_positive` 0.34).
Its own post-mortem recorded that the two STRONGEST families were never in it:
forecast weather and FluView were "deferred for lack of merge surface." v3
therefore tested fifteen situational-flag columns while the families with the
best registered evidence sat out.

**The forecast merge surface now exists.** `data/raw/forecast_archive/
kickoff_nearest_2009_2025/forecasts.parquet` is complete: one row per REG game,
4,431/4,431, keyed on `game_id`, built by
`scripts/ingest_forecast_archive.py --cutoff-mode kickoff_nearest`. That is
exactly the join v3 lacked.

`forecast_weather_kn_warm_team_cold_late_full` is the strongest registered
member (**read** this session, `registry/weak_signals.json`): +0.1686 accuracy
points, week-blocked 95% [+0.0091, +0.3169], `probability_positive` 0.9800,
season-blocked 0.9671 — but on a 1.51% slate fraction (65 flagged team-games).
A hand-coded cell that narrow is a poor production lever. The question this arm
asks is different and, I think, better: **given the continuous forecast
variables as model inputs, does ridge find more than the hand-coded cells did?**

## Leak safety

The `kickoff_nearest` cutoff selects the forecast issuance NEAREST the kickoff,
and every row's `issuance_runtime_utc` precedes its `kickoff_utc`. This is
pregame information by construction.

It is also *playable* information under this pool's rules: picks stay editable
until each game's own kickoff and only the LINES freeze on Tuesday
(`docs/late_week_refresh.md`, and the pick-deadline logic in
`nfl_ats.pick_refresh.pick_deadline`). A kickoff-nearest forecast is therefore
available at decision time, unlike a Tuesday-noon forecast which would be the
conservative-but-weaker choice.

**Production consequence, stated up front:** a promoted v4 needs a LIVE
forecast at predict time, not the archive. That path already exists and is
already fail-open —
`nfl_ats.forecast_weather_kn_warm_team_cold_late_tilt_overlay.
fetch_shared_kickoff_nearest_forecasts_fail_open`, shared by two live
challengers. If this arm wins, wiring that fetch into the card path is required
work, not an afterthought, and a week whose fetch fails must fall back to the
incumbent profile rather than to zero-filled weather.

## Declared feature family (frozen before scoring)

Continuous and structural only — no thresholds, no hand-picked cells, because
the cells are what v3 already tried:

| column | source | note |
|---|---|---|
| `forecast_temp_f` | archive | forecast temperature at the kickoff-nearest issuance |
| `forecast_wind_mph` | archive | forecast wind |
| `forecast_precip_prob_pct` | archive | forecast precipitation probability |
| `forecast_is_outdoors` | archive `roof` | 1 when `roof == "outdoors"`, else 0 |
| `forecast_temp_f_outdoor` | derived | `forecast_temp_f` masked to outdoor games, else the outdoor median |
| `forecast_wind_mph_outdoor` | derived | `forecast_wind_mph` masked the same way |

The two `_outdoor` interactions are included because a dome game's forecast
temperature is not a football input at all, and leaving it unmasked forces the
model to learn the interaction through `forecast_is_outdoors` alone. Masking to
the **outdoor median** rather than to zero keeps a dome game from reading as an
extreme cold game.

**Missing values** (52 rows, 1.2%, all rows whose `icao_station` never mapped)
are filled with the column median computed on the TRAINING rows only, matching
how every other family in this table handles absence. A missing forecast is not
evidence of mild weather.

Total: **6 new columns**, versus v3's 15.

`weak_stack_v4` = `weak_stack` (production) + this family. Deliberately NOT
built on `weak_stack_v3` or `weak_stack_surface`: the question is whether
forecast weather adds to PRODUCTION, and stacking it on a profile already
refused at the opener would confound the answer.

## FluView is deferred, and this is not a rejection

FluView is not in this arm. **The reason is a missing join, not a weak
interval**, and the distinction matters because this file's own taxonomy
forbids the other kind of deferral.

**Measured** this session: `data/raw/fluview/20260820T003258Z/fluview_raw.parquet`
holds 809,716 rows keyed on `(region, epiweek, issue, lag)` — genuinely
point-in-time capable, which is why it is worth building. But joining it to
games needs three things that do not exist in this repository yet: a
team → state/region map, a kickoff-date → epiweek map, and an issue-aware
selection that takes only the release available BEFORE each kickoff. Its
coverage also starts at epiweek 201040, so it cannot span 2009.

That is a build, not a judgement. The six registered `fluview_*` signals stay
`unresolved_below_power` and this arm says nothing about them.

## Endpoints (frozen, in the project's declared priority order)

Identical to `scripts/weak_stack_v3_opener_eval.py`, which this adapts
line-for-line; only `CANDIDATE_PROFILE` and the features table differ. Both
arms hold `ridge_alpha=10.0` and `regressor="ridge"` fixed at the incumbent's
own values, so only `feature_profile` varies.

1. **PRIMARY** — paired forced-pick accuracy delta at the OPENER grade under
   the PRODUCTION probability rule, paired Tuesday-opener population,
   week-blocked bootstrap, `probability_positive`.
2. Opener grade under the sign rule, for comparability with prior runs.
3. Close-graded delta on the same paired games (secondary; per AGENTS.md a
   close-graded number may never veto a play).
4. Each arm's absolute opener/close accuracy (sanity anchor: baseline should
   land near the tracked 53.36% production-rule opener figure).
5. Brier / log-loss direction, paired, week-blocked.
6. Pick-flip count and where flips concentrate by |opener line| bucket.
7. Coverage and distribution of the six new columns on the paired opener
   population.

## Decision rule (frozen)

This is a FORCED-PICK pool: 285 cards get submitted either way, so the decision
is expected value, not a threshold. `probability_positive` above 0.5 favours
playing the candidate; below 0.5 favours keeping the incumbent. The predeclared
thresholds elsewhere in this project govern what the docs may CLAIM, never
which card is played.

The grade that decides is the **opener**. A close-graded number is reported and
never gates.

Whatever the result, it is recorded via `nfl-ats weak-signals record` with
`probability_positive` reported. An interval containing zero closes nothing.

---

## Results (post-scoring addendum, 2026-08-25)

Run: `artifacts/weak_stack_v4_opener_eval/20260825T223935Z/opener_summary.json`
(**measured** this session). Population: **1,537 paired Tuesday-opener games,
107 weeks, seasons 2020-2025.** Both arms `ridge_alpha=10.0`,
`regressor="ridge"`; only `feature_profile` differs.

### Sanity anchor first

The baseline arm scores **53.36%** at the opener under the production
probability rule — matching the tracked production figure exactly. The
instrument is measuring the right thing.

### Headline

| endpoint | baseline | candidate | delta (pts) | week-blocked 95% | P+ |
|---|---|---|---|---|---|
| **opener, probability rule (PRIMARY)** | 53.36% | 52.30% | **−1.065** | [−2.688, +0.595] | **0.0956** |
| opener, sign rule | 52.83% | 50.90% | −1.929 | [−3.347, −0.594] | 0.00255 |
| close, probability rule | 52.09% | 52.09% | 0.000 | [−1.678, +1.702] | 0.4754 |
| close, sign rule | 51.56% | 50.76% | −0.796 | [−2.288, +0.673] | 0.1372 |

Season-blocked primary: [−3.439, +0.938], P+ 0.1629.

Paired probability quality at the opener moved the same way and both intervals
exclude zero: Brier −0.001574 [−0.002839, −0.000336] P+ 0.0064; log-loss
−0.003192 [−0.005787, −0.000648] P+ 0.0071.

### The decision

**Keep the incumbent `weak_stack` profile. `weak_stack_v4` is not promoted.**

Stated as the decision first, per AGENTS.md: at P+ 0.0956 on the primary
endpoint, playing the candidate would be taking the ~10 side of a 90/10 bet.
That is an expected-value call, not a threshold call, and it does not depend on
any interval containing or excluding zero.

The candidate's coverage was never the problem: the six columns cover 98.9% of
the paired opener population, `forecast_is_outdoors` 100%, with real spread
(temp 1-108°F, wind 0-31 mph, precip 0-100%).

### Where it went wrong, measured

142 picks flipped under the probability rule. The baseline was right on
**55.8%** of exactly those games; the candidate on **44.2%**. The flips were
actively harmful rather than a wash, and they concentrated in the ordinary
short-line buckets ([0,3): 39 flips, baseline 59.0% vs candidate 41.0%;
[3,7): 72 flips, 55.7% vs 44.3%). Only the [10,inf) bucket favoured the
candidate, on 12 flips.

*Inferred, not measured:* the market prices the same public forecast, so six
continuous weather columns mostly add variance to a market-residual model whose
market term is already near-sufficient — the same shape as this project's own
`team quality is already priced` result. Nothing here measures that mechanism;
it is a hypothesis for anyone who revisits the family.

### Classification, and what is NOT closed

Recorded 2026-08-25 (`registry/weak_signals.json`, now 450 signals):

* `weak_stack_v4_forecast_weather_opener_probability_rule` —
  **`unresolved_below_power`**. Both blockings' intervals contain zero, so no
  admissible closing ground is met on the PRIMARY endpoint and none is claimed.
  It leans hard against the candidate, and that lean is what decides the play.
* `weak_stack_v4_forecast_weather_opener_sign_rule` — **`refuted_mechanism`**,
  closing ground `wrong_sign_resolved`. The whole interval sits below zero on
  both blockings ([−3.347, −0.594] and [−3.272, −0.829]). That is a resolved
  wrong sign, the one admissible ground, not an interval-contains-zero
  rejection.

**This closes nothing about forecast weather as a phenomenon.** The scope of
that closure is exactly this construction: six continuous columns bolted onto a
~275-column ridge at alpha 10, scored under the sign rule.

In particular it says **nothing against the live pick-level forecast
challengers**. `forecast_weather_kn_warm_team_cold_late` (+0.1686 accuracy
points, [+0.0091, +0.3169], P+ 0.9800, both windows excluding zero) is a
different construction — a narrow post-prediction tilt on ~1.5% of the slate —
and it, `forecast_weather_kn_precip_high_total_tilt` and
`forecast_cold_visitor_tilt` all remain ACTIVE_PROSPECTIVE challengers and all
recorded normally in this session's lock-day rehearsal. The finding here is
narrower and more useful than "weather does not work": **the cells beat the raw
variables.** Handing ridge the continuous inputs did not recover what the
hand-coded conditions capture, and it cost accuracy to try.

### What this does not answer

* **FluView is still unbuilt**, for the join reasons stated above, and this arm
  says nothing about it. Its six registered signals stay
  `unresolved_below_power`.
* **A leaner arm was not tried.** Two columns (`forecast_wind_mph_outdoor`,
  `forecast_precip_prob_pct`) rather than six, or a groupwise ridge penalty that
  penalises the weather block harder than the market block
  (`nfl_ats.margin`'s existing group-wise machinery), are both untested and both
  cheap. Neither is predeclared here, and neither should reuse this window
  without saying so.

### Owner hypothesis tested, 2026-08-25: "the 5-day-out forecast isn't accurate enough"

Reasonable, and measurably not what happened in this arm. The archive used here
is `kickoff_nearest`, not a long-range forecast.

**Measured** from `data/raw/forecast_archive/kickoff_nearest_2009_2025/forecasts.parquet`:

* Lead time from forecast issuance to kickoff: **median 6.0 hours**, mean 5.56,
  95th percentile 9.4 hours, max 11.5. Not days.
* Temperature accuracy vs the observed value (n=2,975 outdoor games with
  observations): **r = 0.9642, MAE 3.25°F**. The temperature input is
  effectively the actual temperature.
* Wind accuracy: **r = 0.6486, MAE 3.16 mph.**

So this arm already handed the model near-perfect temperature and it still lost
about a point. Better temperature forecasting has essentially no headroom left
on this path — the ceiling is r ≈ 0.96 and we are at it.

**Wind is the exception and the hypothesis holds there.** r = 0.649 at a
six-hour lead is genuinely noisy, and wind is the channel with the most
plausible football mechanism. A wind-only arm using a better wind source (or
the observed value as a positive control, to bound how much a perfect wind
forecast could possibly be worth) is untested, cheap, and not predeclared here.

Note also that these lead times are compatible with the pool's own deadline:
picks lock at each game's kickoff (capped at Sunday 16:00 ET), so a forecast
issued a median six hours before kickoff is available at decision time for the
Sunday-afternoon slate.
