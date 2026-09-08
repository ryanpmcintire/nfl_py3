# Home-side offset promotion (MOD-18 lane S, 2026-09-07 evening)

## The decision

Decision (read: coordinator, 2026-09-07 evening, on the expected-value rule in
AGENTS.md "A promotion bar is not a decision bar"; precedent the same morning's
`gaussian_median` promotion at `probability_positive` 0.597 through the card):
serve lane S's arm S2, a walk-forward HOME-SIDE OFFSET by spread bucket added
to the served point prediction OUTSIDE the ridge, before the Tuesday
2026-09-08 09:15 ET lock. The mapping (`gaussian_median`), the ridge, the
residual sample and the three-member overlay union are unchanged.

Measured (lane S, `docs/home_side_location.md`; coordinator verified 88
registry rows under `mod18_home_side_location_v1` and re-ran the lane's
tests): on the active model `a4c757efd2525da6`'s matched opener archive,
1,503 non-push games 2020-2025, week-blocked bootstrap 20,000 draws seed
20260817, within-week correlation zero:

| read | delta | 95% | `probability_positive` |
|---|---|---|---|
| S2 standalone at the opener | -0.20 pts | [-1.60, +1.20] | 0.368 |
| S2 THROUGH the played three-member card | **+0.33 pts** (55.56% vs 55.22%) | [-0.80, +1.46] | **0.695** |

Stated plainly, against it: the two S2 reads differ by 8 games of 1,503;
Brier and log loss lean worse (`probability_positive` 0.10-0.11); 2024 reads
-2.26 (P+ 0.03) and 2025 +3.00 (P+ 0.93); the archive is mined. For it: it is
the first candidate that moves the diagnosed error (the 10.5+ home point
error shrinks from +2.91 to +1.97; 7.5-10 accuracy 48.5% to 51.5%; 10.5+
44.3% to 47.3%), and the decision read the family predeclared is the one
through the played card. Every cell remains `unresolved_below_power`; the
2026 paired rows settle it at no window cost.

## What is served

Read (`src/nfl_ats/home_side_location.py`, production section;
`src/nfl_ats/cli_commands/prediction.py`):

- `margin-predict` fits the offsets for the target week from the archived
  out-of-time opener stream (`tue_open_home_spread + residual_at_open`
  versus `result`, per lane-J spread bucket, shrunken mean with the declared
  100-game prior toward zero, five trailing seasons, whole target week and
  every later week excluded, only rows with a result). History precedence,
  all read-only: (a) the newest opener evaluation matched to the active
  model; (b) otherwise the newest evaluation of any model id, with the
  mismatch recorded as a warning -- the normal lock-day path, because the
  Tuesday chain activates the refit model BEFORE its own evaluation runs;
  (c) otherwise zero offsets with a warning. This layer never blocks the lock.
- The offset is added to the served `market_residual` point BEFORE every
  downstream quantity (`MarginModel.predict(center_offset=...)`), so the
  point, fair spread, market residual, cover probability, push split and the
  flip-line scan (`line_sweep`) all move together. The market, fair-margin
  and straight-up companion methods are untouched.
- The served policy is named `home_side_offset_by_bucket_v1`
  (`HOME_SIDE_OFFSET_POLICY`); `HOME_SIDE_OFFSET_SERVED = False` turns it off.
- `home_side_offset.json` beside the card carries both reads per game, the
  fitted offsets, prior counts, the history source path and model id, and
  warnings; `metadata.json` carries the same summary under
  `home_side_offset` (provenance, never reader text).
- The late-week refresh (`pick_refresh.plan_refresh`) applies the identical
  per-game offset from that sidecar when it refits at the frozen Tuesday
  line, so a late switch can only come from new information, never from
  silently dropping the correction. A pre-promotion card (no sidecar) refits
  exactly as before.

## The paired challenger

`home_side_offset_off_incumbent` (`artifacts/prospective/challengers.json`,
`src/nfl_ats/home_side_offset_incumbent_overlay.py`) records the UNCORRECTED
read's forced picks from the sidecar verbatim -- never a refit -- beside the
published-card decision recorder in `publish-predictions --record-decisions`,
fail-open, and in the lock-day rehearsal. It refuses a card without a served
sidecar and a configuration-fingerprint drift, exactly like
`gaussian_mean_mapping_incumbent`.

## Known gap, disclosed

The headline archive score on the board is read from the active model's
opener evaluation composed through the overlay union; that evaluation scores
the raw model and does not apply the walk-forward offset, so after this
promotion the published archive number is the pre-correction read
(55.2% for the current model) rather than lane S's composed 55.6%. It is
still read from an artifact keyed to the active model, not a constant. The
follow-up is to apply the same walk-forward offset inside
`opener-evaluation` so the headline and the served card use one policy;
queued in ROADMAP MOD-18.

## Proof and activation

Measured entries are appended below by the coordinator as they are run: the
isolated chain proof (weekly-run + publish in a scratch artifacts root), the
Week 1 side diff, the offsets used, and the real refresh.

### Isolated chain proof (measured, coordinator, 2026-09-07 21:05 ET)

Scratch artifacts root seeded with the active manifest, its linked backtest
and forecast, the matching opener evaluation `20260907T152026Z` and the
challenger registry; `NFL_ATS_ARTIFACTS_DIR` / `NFL_ATS_REGISTRY_DIR` pointed
at it; weekly-run's own step-5 argv run verbatim:

```text
nfl-ats margin-predict --season 2026 --week 1 --features data\processed\game_features_weak_stack.parquet --feature-profile weak_stack --probability-method gaussian_median
```

- Result: `synchronization_status: SYNCHRONIZED`, `active_model_id`
  `a4c757efd2525da6` (unchanged), forecast `2026-week-01-20260908T010555Z`.
- History source used: `opener_evaluation/20260907T152026Z` (precedence a,
  matched to the active model; no warnings).
- Offsets served (points; prior games): 0-3 +0.032 (504); 3.5-6.5 +0.385
  (468); 7 -0.171 (65); 7.5-10 +0.784 (160); 10.5+ +1.916 (113). These equal
  lane S's `week1.json` `s2_offsets_2026` to four decimals.
- Pool card versus the live card `2026-week-01-20260907T232720Z`: exactly one
  side changes, `2026_01_NE_SEA` NE +3.5 -> SEA -3.5 (home cover 0.4976 ->
  0.5098); ARI at LAC moves 0.334 -> 0.390 without changing side; the other
  14 games move less.
- The sidecar's uncorrected probabilities reproduce the live card's
  probabilities to 5.6e-17 on all 16 games, so the paired challenger records
  the pre-promotion read exactly.
- `scripts/lockday_rehearsal.py`: PASS, 42 active paths (33 publish), 0
  errors, with the new recorder in the table.

### Real refresh and publish (measured, coordinator, 2026-09-07 21:21-21:55 ET)

`scripts/refresh_lineup_forecast.py` ran the real chain: weekly-run refit
(model id unchanged, `a4c757efd2525da6`, SYNCHRONIZED), forecast
`2026-week-01-20260908T011739Z` with `home_side_offset.json` (same offsets and
source as the isolated proof, no warnings), `CURRENT_PREDICTIONS.md`
republished with SEA -3.5 (51.0%) as the only changed side. weekly-run then
aborted at its final `publish-board` step: the spread explorer refits the
model and requires its probabilities to reproduce the card, and the served
offset broke that reproduction. Fixed the same hour: every module that refits
the active recipe to reproduce or extend the served card now adds the
per-game served offset before comparing (`spread_explorer`, both compute
functions and their callers; the `ecdf_mapping_incumbent`,
`gaussian_mean_mapping_incumbent` and `smooth_cdf_mapping` recorders via
`center_offsets_from_metadata`, rebuilt from the forecast's own metadata; the
late-week refresh via the sidecar). `publish-board` then regenerated all four
pages; the lock-day rehearsal passes with 42 active paths and 0 errors.

Not yet aligned (follow-up, disclosed): challengers that refit the recipe for
their OWN arm without a reproduce-the-card check (`expected_lineup_loss`,
`deadline_drag`, `qb_revenge_deadline_drag_stack`, `inactives_refresh`,
`best_pick_nomination` dispersion) still refit the uncorrected centre; they
keep recording, paired against a base that differs from the served card by
the offset. And `opener-evaluation` still scores the raw model, so the board
headline is the pre-correction archive number.

### Known gap closed: the headline evaluation applies the served offset (measured, 2026-09-08 07:02 ET)

`opener-evaluation` now applies the SAME walk-forward home-side offset the
card serves (`nfl_ats.clv.opener_pick_evaluation`, `home_side_offset`
argument, default = `HOME_SIDE_OFFSET_SERVED`): for each scored week the
per-bucket offsets are fitted with `fit_home_side_offsets` on the RAW
out-of-time opener points of the weeks already scored (`prior_rows_before`,
five trailing seasons, whole target week excluded) and added to the point
through `MarginModel.predict(center_offset=...)`. `residual_at_open` /
`residual_at_close` stay the RAW residual -- they are the archive stream
`fit_production_home_side_offsets` reads on lock day, so correcting them in
place would compound the correction (pinned:
`test_opener_pick_evaluation_serves_the_walk_forward_home_side_offset`
reproduces the evaluation's own per-week offset from its artifact with
production's fit function). The served read lives in
`home_cover_probability_at_*` and the `*_probability_rule` pick columns
(what the board composes); `*_raw` twins and `home_side_offset_at_open`
sit beside them; `metadata.json` carries a `home_side_offset` block
(policy, served, games with a non-zero shift, picks changed).
`--no-home-side-offset` scores the raw model as a comparison run that never
identifies itself as the active model.

Measured on the active model `a4c757efd2525da6`, feature table
`457aafb7...`, artifact `artifacts/opener_evaluation/20260908T110201Z`
(1,537 games 2020-2025; 1,521 carried a non-zero shift, mean |shift| 0.47
points, 109 opener picks changed):

| read | with the offset | without (raw twin) |
|---|---|---|
| model alone, opener, probability rule | **53.76%** | 53.96% |
| model alone, close, probability rule | 53.15% | 53.09% |
| through the played three-member card (`overlay_subset_composition/20260908T110450468551Z`) | **55.56%** (+1.80 pts over its own baseline, week-blocked [-0.20, +3.81], P+ 0.958, 269 flips) | 55.22% on the previous evaluation (+1.26, [-0.79, +3.34], P+ 0.880) |

Per season, model alone with / without: 2020 51.8 / 51.8; 2021 54.7 / 53.8;
2022 53.6 / 54.4; 2023 53.8 / 55.6; 2024 52.3 / 54.5; 2025 56.2 / 53.2.
These are lane S's numbers reproduced by the production path (standalone
-0.20 pts, +0.33 through the card); the board headline now reads 55.6% and
the Model page carries "The home-side push" (this week's offsets by spread
size beside the archive record, both reads shown). Every cell remains
`unresolved_below_power`; the 2026 paired rows settle it.

### Own-arm refit challengers aligned (2026-09-08, Codex lane B; measured 111 tests)

`nfl_ats.card_refit.load_card_refit(metadata, card, forecast_dir)` replays the
served correction for any recorder that refits the active recipe on top of the
card: the per-game offsets come from the card's `metadata.json`
(`center_offsets_from_metadata`), else its sidecar (`served_center_offsets`),
never from a new fit, and the probability mapping is the card's recorded one.
A pre-promotion card (no block, no sidecar) keeps the caller's historical
mapping and the uncorrected centre, with the warning carried in the recorder
result. Wired into `expected_lineup_loss_challenger`, `deadline_drag_challenger`,
`qb_revenge_deadline_drag_stack_challenger` and
`era_weighted_half_life_8_overlay` (its reproduction check no longer hard-codes
the Gaussian read). Not wired, on purpose: `best_pick_nomination` (the alpha-2000
nomination refit is a served decision of its own and stays as measured) and
`pool-card-at-lines` (a standalone command with explicit configuration).
Tests: `tests/_card_refit_test_kit.py` replays the served card in legacy /
metadata / sidecar modes for each recorder.

### S3 played: the push is served only on spreads of seven points or more (measured, 2026-09-08 08:20 ET)

Predeclared by Codex lane E (read-only, gpt-6-astra low effort) before computing:
S3 = S2 with the offset served ONLY in the "7", "7.5-10" and "10.5+" buckets
(zero in "0-3" and "3.5-6.5"), everything else unchanged. Mechanism: the
home-side location error was diagnosed on big spreads (lanes L/P/Q/S); the
small buckets showed none, and S2's correction there ran negative through
2023 and cost -1.99 pts (P+ 0.028). Disclosed plainly: a post-hoc restriction
on the same mined 1,537-game archive S2 was selected on.

Measured on `artifacts/opener_evaluation/20260908T110201Z` (1,503 non-push
games 2020-2025, week-blocked bootstrap 20,000 draws, seed 20260817,
within-week correlation zero; lane E's standalone read re-run by the
coordinator, the through-card read computed by the coordinator with lane S's
`composed_picks`/`comparison` helpers):

| read | S3 | S2 | delta | 95% | P+ |
|---|---|---|---|---|---|
| model alone, opener | 54.56% (820/1503) | 53.76% | +0.80 pts | [-0.20, +1.81] | 0.934 |
| model alone vs raw | 54.56% | 53.96% (raw) | +0.60 | [-0.27, +1.51] | 0.896 |
| THROUGH the played three-member card | **55.89%** (840/1503) | 55.56% | **+0.33** | [-0.60, +1.29] | **0.736** |
| Brier (improvement) | | | +0.00072 | [+0.00005, +0.00144] | 0.982 |
| log loss (improvement) | | | +0.00156 | [+0.00014, +0.00306] | 0.985 |

Per season through the card, S3 minus S2: 2020 +0.45, 2021 -0.42, 2022 +0.40,
2023 +1.50, 2024 0.00, 2025 0.00 (S3 and S2 pick identically in the small
buckets' flips there). Per bucket standalone: 0-3 +1.99 (P+ 0.959), 3.5-6.5
+0.18 (P+ 0.549), the big buckets identical to S2 by construction. Week 1:
exactly one served side changes back, NE +3.5 at SEA (S2 had flipped it to
SEA -3.5; S3 leaves the 3.5-6.5 bucket uncorrected, home cover 0.498).

Decision (coordinator, EV rule, AGENTS.md "A promotion bar is not a decision
bar"): S3 is PLAYED from the 2026-09-08 Week 1 lock (12:20 ET, after the pool's noon spread lock). Served as
`HOME_SIDE_OFFSET_BUCKETS` in `home_side_location.fit_home_side_offsets`
(fitted values and prior counts are still reported for every bucket; the
served value is zero outside the three big buckets, so the sidecar, the
card's metadata block, the late-week refresh and every refit recorder carry
the same zeros), policy id `home_side_offset_big_spreads_v2`. The headline
evaluation `artifacts/opener_evaluation/20260908T115957Z` (401 games with a
non-zero shift, 43 picks changed vs raw; model alone 54.56% vs 53.96% raw)
composes to **55.89%** through the played card
(`overlay_subset_composition/20260908T120013552183Z`, +1.33 over its own
baseline, [-0.73, +3.40], P+ 0.890). Every cell recorded under
`mod18_home_side_location_v1_s3_*` as `unresolved_below_power`; the paired
challenger `home_side_offset_off_incumbent` keeps recording the uncorrected
read, so the 2026 rows settle S3 the same way they settle S2.
