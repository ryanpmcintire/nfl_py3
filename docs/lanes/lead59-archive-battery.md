# lead59-archive-battery

## Goal

LEAD-59: let the 2009-2014 officials archive reach the crew traits, then
re-run the officials archive battery on production with the archive on and
report every arm with `probability_positive`. Done when the battery stages
run clean on the fixed builders and the numbers are recorded through
`nfl-ats weak-signals record` by the coordinator.

## State

Src fix done, uncommitted (opencode lane D, `big-pickle`, verified by the
coordinator 20:15 ET): `src/nfl_ats/officials_flag_features.py` (crew
traits from crew rows, penalty left-join with NaN, rookie floor from the
population), `src/nfl_ats/crew_tilt_refresh_overlay.py` (drop referee-games
absent from the penalty-type snapshot). Measured no-op with the archive
off: crew-tilt trait tables and all five builders bit-identical.
`scripts/officials_archive_battery_eval.py` restored from `b7ed31d~1`
(comments stripped). Battery re-run is opencode lane G (`mimo-v2.5-free`,
relaunched 20:32 ET after nemotron quit with no output): brief
`tests/scratch/lanes/laneG_brief.md`, report `laneG_report.md`; stages
traits, opener, served-proxy, era, composition, card; never `record`.

## Tried

- A literal inner join in `_referee_name_season` (what the doc asked)
  would drop 287 modern referee-games whose zero holding count is genuine
  and shift the holding trait sum 461.65 -> 508.70; the lane implemented the
  narrower filter instead.

Lane G done 20:44 ET; the coordinator regenerated the record commands
(`--stage record`), ran all 23 with `--replace` (measured: 23 `recorded`
lines), and wrote the results into ROADMAP LEAD-59 and the tail of
`docs/officials_archive_battery.md`. Uncommitted.

## Next (superseded, see the dated Next section below)

Commit. The penalty-type trait binning with the archive on now has its NaN
filter decided and measured -- see "Type-trait binning measured 2026-09-23"
below, whose own "## Next" section has the exact record commands.

## 2009-2014 sweep probe 2026-09-17 (measured)

Bounded probe of `scripts/officials_wayback_sweep.py` (subagent, defaults
untouched, first 20 2009 REG games as the predeclared set, killed by its
25-minute cap during game 8): 7 attempted, 42 rows, 6 officials-block hits.
Throttle present but passable: 5 of 7 games drew CDX backoff (900 s total,
60/120/240 s schedules), no final 403. Rate ~21-23 games/h, extrapolating to
~51-102 h wall for all 1,536 REG 2009-2014 games. Implication: the full sweep
is not a foreground task; it needs a background scheduler job spread over
weeks, or a sampled archive. Probe wrote only under ignored `data/raw/`
(`officials_pfr_wayback/probe20260917Tb20/`, no parquet — killed before final
write); no tracked file touched.

## Type-trait binning predeclaration 2026-09-23 (before measuring)

Open item resolved by this unit: whether `_build_referee_type_trait_data`
(`src/nfl_ats/experiment_runner.py:1372`) needs a NaN filter with the archive
on. Read first: `src/nfl_ats/crew_tilt_refresh_overlay.py:173`
`_referee_name_season` already carries the fix for the CREW-level holding
trait (drop referee-games whose `game_id` is absent from the latest
`game_penalty_types.parquet` snapshot entirely, keep the type-specific
left-join + `fillna(0.0)` only for games the snapshot does cover, since a
missing (game_id, penalty_type) row there means zero of that type occurred,
not missing coverage -- confirmed by measurement: 0.0% of existing
`penalties_total` rows equal 0, i.e. the table is sparse-positive-only).
That function is hardcoded `include_archive=False` via
`load_officials_for_prospective_channel` (serving-safety guard, not a bug) so
it cannot be reused to test archive-on directly. `src/nfl_ats/` is off limits
this unit (registry/src frozen), so `scripts/lead59_type_trait_bins.py`
reimplements the same narrower-filter pattern standalone, forcing
`include_archive=True` on `officials_archive.load_officials`, for both
tracked penalty types (`Defensive Pass Interference`, `Offensive Holding`).

**Predeclared bins (both mirror the two live prospective-challenger flags,
so each carries the same named mechanism, not a new one):**

1. `dpi_tilt_pass_heavy_favorite` -- home team favored (`tue_open_home_spread`
   > 0) AND home prior-rolling pass rate top quartile
   (`game_features_pbp.parquet` `home_pbp_off_pass_rate`) AND the game
   referee's prior-season Defensive Pass Interference count in the top
   lagged quartile (archive-on, narrower-filter population). Mechanism:
   a DPI-heavy referee plus a pass-heavy favored home offense draws more
   flags against the home team, denting its cover odds.
2. `holding_tilt_run_heavy` -- home prior-rolling pass rate BOTTOM quartile
   (run-heavy) AND the game referee's prior-season Offensive Holding count in
   the top lagged quartile (archive-on, narrower-filter population).
   Mechanism: a holding-heavy referee plus a run-heavy home offense draws
   more holding flags on the home line, denting its cover odds.

**Predeclared measurement (before running):** each bin enters the served
four-term fit (`FIT_FEATURES` in `pick_probability_fit.py` --
`model_logit`, `flag_sum`, `move`, `move_available`) as ONE added fitted
term (its own 0/1 flag), refit standardized ridge logit
(`_fit_logit`/`FIT_RIDGE`), LOSO by season over `build_fit_population`'s
2020-2025 coverage. Paired vs the base four-term fit via
`nfl_ats.signal_atlas._cell` (season-then-week block bootstrap, 20,000
draws, seed 20260923, 95% interval, reliability edges [0.0, 0.5, 1.0]).
Exactly 2 looks, no others; family = `lead59_type_trait_archive_bins_fit_v1`.
Also reported as a diagnostic, not a look: flag agreement between the
archive-on narrower-filter build and the production archive-off build
(`experiment_runner._build_referee_type_trait_data`), to state plainly
whether the archive changes anything here at all given
`game_penalty_types.parquet` covers only 2015-2025 (measured 2026-09-23:
seasons 2015-2025 present, 2009-2014 absent) and the narrower filter drops
any game absent from that snapshot -- so archive-added 2009-2014 rows are
structurally unreachable by this consumer and a 0-flip diagnostic is the
expected honest outcome, not a target.

## Type-trait binning measured 2026-09-23 (real run, uncommitted)

`scripts/lead59_type_trait_bins.py` run once for real:
`artifacts/lead59_type_trait_bins/20260923T205414Z/` (`summary.json`,
`per_game.parquet`). Ruff clean, no `src/` edit, no registry write.

**Diagnostic (not a look):** archive-on narrower-filter build vs the live
production archive-off `_build_referee_type_trait_data` build -- **0 of 2,892
shared games changed** for both DPI and holding (`n_changed_on_shared: 0`,
identical game/official coverage). Confirms the predeclared expectation:
`game_penalty_types.parquet` covers seasons 2015-2025 only (measured), the
narrower filter drops every game absent from it, and every archive-added
2009-2014 row is absent from it, so this consumer cannot be reached by the
archive at all even after the correct fix -- structurally, not by omission.
1,534 of 4,426 archive-on referee-games were dropped as snapshot-uncovered
(all pre-2015).

**Look 1, `dpi_tilt_pass_heavy_favorite`** (referee prior-season DPI lag
quartile == 4 AND home favored AND home prior-rolling pass rate top
quartile) as one added fitted term: 38/1503 games flagged, 18 decisive.
Accuracy -0.133 pts, 95% CI [-0.669, +0.330], **P+ 0.2986**. Brier
improvement -0.00177, CI [-0.00615, +0.00010], P+ 0.0857. Trait reliability
-0.066 (158 referee-season pairs). Both intervals cross zero ->
`unresolved_below_power`.

**Look 2, `holding_tilt_run_heavy`** (referee prior-season holding lag
quartile == 4 AND home prior-rolling pass rate bottom quartile) as one added
fitted term: 54/1503 games flagged, 21 decisive. Accuracy -0.599 pts, 95% CI
[-1.793, +0.346], **P+ 0.1422**. Brier improvement -0.00055, CI [-0.00138,
+0.00002], P+ 0.0323. Trait reliability 0.323 (158 pairs). Both intervals
cross zero (brier upper edge barely positive) -> `unresolved_below_power`,
not `wrong_sign_resolved` (whole interval must sit on the wrong side; it
does not).

Both looks LOSO 2020-2025 (6 folds), season-then-week block bootstrap 20,000
draws seed 20260923, paired against the served four-term fit
(`model_logit`, `composition_flag_sum`, `market_move_toward_home`,
`market_move_available`) via `signal_atlas._cell`. Family
`lead59_type_trait_archive_bins_fit_v1`, 2 looks total, both predeclared
before this run.

## Next

2026-09-25: verified both type-trait cells already in
`registry/weak_signals.json` (`lead59_dpi_tilt_pass_heavy_favorite_fit_term`,
`lead59_holding_tilt_run_heavy_fit_term`, family
`lead59_type_trait_archive_bins_fit_v1`, both `unresolved_below_power`).
Nothing further to run.

Root: record both cells (source is the summary above; neither is served,
neither changes a pick; both close the open item honestly as
`unresolved_below_power`):

```
nfl-ats weak-signals record --name lead59_dpi_tilt_pass_heavy_favorite_fit_term \
  --description "DPI-heavy referee crew (archive-on lagged top quartile, coverage-filtered) crossed with a pass-heavy home favorite, added as one fitted term to the served four-term pick probability, LOSO by season 2020-2025" \
  --source artifacts/lead59_type_trait_bins/20260923T205414Z/summary.json \
  --effect -0.133 --effect-units accuracy_points --classification unresolved_below_power \
  --league nfl --season-start 2020 --season-end 2025 \
  --interval-low -0.669 --interval-high 0.330 --probability-positive 0.2986 \
  --sample-games 1503 --sample-blocks 107 --reliability -0.0663 \
  --family lead59_type_trait_archive_bins_fit_v1 \
  --classification-evidence "Accuracy CI [-0.669,+0.330] and Brier CI [-0.00615,+0.00010] both cross zero on 18 decisive games; trait year-over-year reliability -0.066 (158 pairs) is near zero. No AGENTS.md closing ground applies." \
  --plain-summary "When a DPI-heavy referee crew works a pass-heavy home favorite, the home side did not measurably do better or worse in a small sample; not ready to serve." \
  --category onfield

nfl-ats weak-signals record --name lead59_holding_tilt_run_heavy_fit_term \
  --description "Holding-heavy referee crew (archive-on lagged top quartile, coverage-filtered) crossed with a run-heavy home offense, added as one fitted term to the served four-term pick probability, LOSO by season 2020-2025" \
  --source artifacts/lead59_type_trait_bins/20260923T205414Z/summary.json \
  --effect -0.599 --effect-units accuracy_points --classification unresolved_below_power \
  --league nfl --season-start 2020 --season-end 2025 \
  --interval-low -1.793 --interval-high 0.346 --probability-positive 0.1422 \
  --sample-games 1503 --sample-blocks 107 --reliability 0.3226 \
  --family lead59_type_trait_archive_bins_fit_v1 \
  --classification-evidence "Accuracy CI [-1.793,+0.346] and Brier CI [-0.00138,+0.00002] both cross zero (barely, on the Brier upper edge) on 21 decisive games; whole interval is not on the wrong side, so wrong_sign_resolved does not apply." \
  --plain-summary "When a holding-heavy referee crew works a run-heavy home offense, the home side trended worse in a small sample, but it is too small to call; not ready to serve." \
  --category onfield
```

Then commit `scripts/lead59_type_trait_bins.py`,
`artifacts/lead59_type_trait_bins/20260923T205414Z/`, and this lane, together
with the still-uncommitted LEAD-59 fixes noted above. Add the two cells to
`docs/officials_archive_battery.md`'s tail and ROADMAP LEAD-59.

## Open

Whether the `n_censored_2015_debut` key should be renamed now that the
floor is population-derived (cosmetic).
