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
