# Opener-grade error mining: where the active model misses, and what to do about it

Written 2026-08-20. **This is a mined battery: the 12 cell families below were
fixed before any hit rate was computed (see "Predeclared cell list"), but no
multiplicity correction is claimed anywhere in this document.** Every cell is
one look at an already-collected archive, not a new experiment with its own
rotation window. Per AGENTS.md's binding rule, an interval crossing zero is
never treated here as a rejection of anything -- cells are reported with
their effect, week-blocked interval, and `probability_positive`, and every
recorded entry is `unresolved_below_power` unless otherwise noted.

## What this measures

The active model (`weak_stack`/`market_residual`/ridge alpha 10, Gaussian
mapping, model_id `3083f6cbc5e45acb`) graded at the Tuesday opener under the
**production pick rule** (`home_cover_probability_at_open >= 0.5`), which
scores **53.36%** (802/1503) on the frozen 1,537-paired-game 2020-2025
archive after excluding 34 opener-line pushes -- read from
`docs/opener_evaluation.md`'s 2026-08-19 addendum and
`artifacts/opener_evaluation/20260819T174244Z/metadata.json`'s
`opener_accuracy_probability_rule: 0.5335994677312043`. This document
stratifies that same graded population into cells and asks where the 53.36%
comes apart -- some cells score well above it, some well below, all
consistent with a single frozen model's noise once you account for `n`.

**Provenance of the per-game table (measured):** `per_game.parquet` from
`artifacts/opener_evaluation/20260819T174244Z/` (the most recent tracked
opener-evaluation run, reproducing the production-rule numbers exactly per
the addendum) joined on `game_id` to `data/processed/game_features_weak_stack.parquet`
for context columns (`div_game`, `rest_diff`, `weekday`, `gametime`,
`total_line`, `neutral_site`). Join is one-to-one, verified (measured: zero
unmatched rows). **34 opener-line pushes were dropped** (rows where
`correct_at_open_probability_rule` is NaN because `result == tue_open_home_spread`
exactly, i.e. `margin_vs_open == 0`, measured on all 34) before any cell was
computed, leaving **n=1,503**, matching the addendum's own push-excluded
population (measured: `0.5335994677312043 * 1503 == 802.0` exactly, i.e. the
frozen baseline IS the mean of this exact 1,503-row population, not an
approximation of it).

## Predeclared cell list (fixed before any hit rate was computed)

1. Opener spread magnitude: 0-2.5 / 3-6.5 / 7-9.5 / 10+
2. Favorite side: home favorite / road favorite / pick'em
3. Division game: yes / no
4. Rest differential sign (home_rest - away_rest): home more rested / away
   more rested / even
5. Slate: primetime (Thu/Mon any kickoff, or Sun/Sat kickoff >= 20:00) vs the
   rest of the Sunday slate
6. Week-of-season third: weeks 1-6 / 7-12 / 13-18
7. Total (o/u) bucket: quartiles of `total_line` over this population
   (method predeclared; edges are data-dependent and reported below)
8. Pick agrees with opener-to-close movement: `open_move = close_home_spread
   - tue_open_home_spread`; "agrees" if the spread moved toward the picked
   side, "disagrees" if away, "flat" if `open_move == 0`
9. Pick = favorite vs dog (pick'em games get their own bucket)
10. Pick = home vs away
11. Season (2020-2025)
12. Model confidence bucket, `|home_cover_probability_at_open - 0.5|`:
    <0.02 / 0.02-0.05 / 0.05-0.10 / >0.10 -- the calibration-in-the-small check

## Method

For each cell: `n`, hit rate (mean of `correct_at_open_probability_rule`),
effect in accuracy points vs the 53.36% baseline, a week-blocked 95%
interval, and `probability_positive` (P+, fraction of week-blocked bootstrap
draws of the cell's own hit rate that exceed 53.36%). Bootstrap: resample
(season, week) blocks with replacement **within the cell's own games**
(same block-draw mechanics as `nfl_ats.clv.week_blocked_bootstrap` /
`scripts/nfl_bias_battery_screen.py::block_bootstrap_two_group`), 2,000
samples, seed 20260820. Within-week game correlation is treated as zero per
owner mandate; the week is the resampling unit.

`total_line` quartile edges on this population (measured):
`(28.499, 42.0]`, `(42.0, 45.0]`, `(45.0, 48.0]`, `(48.0, 58.0]`.

## Full results table

| Family | Cell | n | Hit rate | Effect vs 53.36% (pts) | Week-blocked 95% (pts) | P+ |
|---|---|---:|---:|---:|---|---:|
| opener_spread_magnitude | 0-2.5 | 370 | 55.68% | +2.32 | [-2.98, +7.44] | 0.815 |
| opener_spread_magnitude | 3-6.5 | 734 | 55.31% | +1.95 | [-2.17, +5.92] | 0.822 |
| opener_spread_magnitude | 7-9.5 | 245 | 48.98% | -4.38 | [-10.88, +2.20] | 0.090 |
| opener_spread_magnitude | 10+ | 154 | 45.45% | **-7.91** | **[-15.14, -0.53]** | 0.020 |
| favorite_side | home_favorite | 607 | 51.07% | -2.29 | [-6.47, +1.90] | 0.142 |
| favorite_side | road_favorite | 884 | 54.98% | +1.62 | [-1.62, +4.78] | 0.839 |
| favorite_side | pick_em | 12 | 50.00% | -3.36 | [-28.36, +21.64] | 0.387 |
| division_game | division | 542 | 52.03% | -1.33 | [-5.63, +3.23] | 0.283 |
| division_game | non_division | 961 | 54.11% | +0.75 | [-2.76, +4.20] | 0.666 |
| rest_diff_sign | home_more_rested | 252 | 51.59% | -1.77 | [-7.24, +4.07] | 0.275 |
| rest_diff_sign | away_more_rested | 265 | 49.81% | -3.55 | [-9.79, +2.61] | 0.114 |
| rest_diff_sign | even | 986 | 54.77% | +1.41 | [-1.99, +4.94] | 0.803 |
| slate | primetime | 338 | 56.51% | +3.15 | [-2.17, +8.00] | 0.874 |
| slate | sunday_slate | 1165 | 52.45% | -0.91 | [-4.05, +2.21] | 0.280 |
| week_third | early_1-6 | 524 | 55.15% | +1.79 | [-2.69, +6.48] | 0.793 |
| week_third | mid_7-12 | 483 | 53.21% | -0.15 | [-5.03, +4.73] | 0.497 |
| week_third | late_13-18 | 496 | 51.61% | -1.75 | [-6.32, +3.08] | 0.227 |
| total_bucket | (28.5, 42.0] | 397 | 55.67% | +2.31 | [-2.55, +6.98] | 0.826 |
| total_bucket | (42.0, 45.0] | 393 | 54.20% | +0.84 | [-4.50, +6.07] | 0.649 |
| total_bucket | (45.0, 48.0] | 349 | 49.00% | -4.36 | [-9.61, +1.24] | 0.065 |
| total_bucket | (48.0, 58.0] | 364 | 54.12% | +0.76 | [-4.54, +5.74] | 0.615 |
| movement_agreement | agrees ⚠️ | 494 | 47.37% | **-5.99** | **[-10.35, -1.66]** | 0.004 |
| movement_agreement | disagrees ⚠️ | 639 | 56.96% | +3.60 | [-0.21, +7.49] | 0.969 |
| movement_agreement | flat | 370 | 55.14% | +1.78 | [-3.94, +7.17] | 0.740 |

⚠️ **Label bug, corrected 2026-08-20** (see "Reconciliation with
`docs/observed_movement_channel.md`" below): the `agrees`/`disagrees`
labels on these two rows are SWAPPED. The n=494 row (-5.99 pts) is the true
`disagrees` cell; the n=639 row (+3.60 pts) is the true `agrees` cell. The
numbers themselves (n, hit rate, effect, CI, P+) are correct and unchanged
-- only the two labels need to be read swapped. Left as originally measured
rather than silently edited, per the project's "annotate, don't erase"
convention; the registry carries both the annotated original entries and
new `_corrected` entries under the right labels.
| pick_side | favorite | 699 | 54.36% | +1.00 | [-2.41, +4.43] | 0.714 |
| pick_side | dog | 792 | 52.53% | -0.83 | [-4.44, +2.87] | 0.313 |
| pick_side | pick_em | 12 | 50.00% | -3.36 | [-28.36, +21.64] | 0.387 |
| pick_home_away | home | 623 | 53.61% | +0.25 | [-3.58, +4.04] | 0.554 |
| pick_home_away | away | 880 | 53.18% | -0.18 | [-3.36, +3.14] | 0.452 |
| season | 2020 | 220 | 52.27% | -1.09 | [-9.07, +5.44] | 0.403 |
| season | 2021 | 236 | 55.08% | +1.72 | [-6.15, +10.40] | 0.648 |
| season | 2022 | 248 | 53.23% | -0.13 | [-5.88, +6.41] | 0.480 |
| season | 2023 | 266 | 54.14% | +0.78 | [-4.31, +6.26] | 0.595 |
| season | 2024 | 266 | 54.89% | +1.53 | [-4.28, +7.36] | 0.714 |
| season | 2025 | 267 | 50.56% | -2.80 | [-9.73, +4.10] | 0.216 |
| confidence_bucket | <0.02 | 327 | 52.91% | -0.45 | [-5.96, +5.23] | 0.440 |
| confidence_bucket | 0.02-0.05 | 316 | 54.75% | +1.39 | [-3.51, +6.40] | 0.721 |
| confidence_bucket | 0.05-0.10 | 519 | 51.64% | -1.72 | [-6.18, +2.64] | 0.202 |
| confidence_bucket | >0.10 | 341 | 55.13% | +1.77 | [-3.94, +7.09] | 0.736 |

Bold: the two cells whose week-blocked 95% interval sits entirely below
zero (`movement_agreement=agrees` **[mislabeled; this is actually the
`disagrees` cell -- see the reconciliation section]**,
`opener_spread_magnitude=10+`). Per AGENTS.md's taxonomy this would be an
admissible `wrong_sign_resolved` closing ground for a candidate
hypothesized to *beat* the baseline in that cell -- both are recorded as
`unresolved_below_power` anyway (not `refuted_mechanism`) because the point
of this document is to keep them open as dampener leads, not to close a
line of work. (The `movement_agreement` closing-ground reasoning is now
moot either way: this cell describes where the *active model itself*
underperforms, not a new candidate hypothesized to beat the baseline, so
`wrong_sign_resolved` was never actually applicable to it -- noted here
because the original text's phrasing invited the confusion.)

## Top 10 cells by |effect| x sqrt(n)

| Rank | Cell | n | Effect (pts) | \|effect\|·sqrt(n) |
|---|---|---:|---:|---:|
| 1 | movement_agreement = agrees ⚠️ (actually `disagrees`) | 494 | -5.99 | 133.2 |
| 2 | opener_spread_magnitude = 10+ | 154 | -7.91 | 98.1 |
| 3 | movement_agreement = disagrees ⚠️ (actually `agrees`) | 639 | +3.60 | 91.1 |
| 4 | total_bucket = (45.0, 48.0] | 349 | -4.36 | 81.5 |
| 5 | opener_spread_magnitude = 7-9.5 | 245 | -4.38 | 68.6 |
| 6 | slate = primetime | 338 | +3.15 | 57.9 |
| 7 | rest_diff_sign = away_more_rested | 265 | -3.55 | 57.8 |
| 8 | favorite_side = home_favorite | 607 | -2.29 | 56.4 |
| 9 | opener_spread_magnitude = 3-6.5 | 734 | +1.95 | 52.9 |
| 10 | favorite_side = road_favorite | 884 | +1.62 | 48.1 |

## Confidence-bucket monotonicity (calibration-in-the-small)

**Measured** (this run): hit rate by increasing `|p-0.5|` bucket is
**-0.45, +1.39, -1.72, +1.77 points** vs baseline -- not monotone (it dips
between the 2nd and 3rd buckets), and every one of the four week-blocked
intervals crosses zero (P+ between 0.20 and 0.74, none near either extreme).
**Answer: on this stratification, the model's own confidence magnitude does
not show a resolvable, monotonic relationship with opener hit rate.** This
is `unresolved_below_power`, not a refutation of calibration-in-the-small --
four coarse buckets on ~1,500 games is a low-power test for a signal this
size, and the true relationship could easily be present but invisible at
this resolution (consistent with the project's ~2-point evaluator
resolution). It does mean this document does **not** independently support
"lean harder on high-confidence picks" as a validated policy yet; a smoother
reliability-diagram approach (rolling/spline calibration across `p`, more
than 4 bins) would have more power to resolve this than repeating the same
4-bucket cut on more seasons.

## Registry entries recorded

24 cells recorded to `registry/weak_signals.json` via
`nfl-ats weak-signals record`, all `effect_units=accuracy_points`,
`classification=unresolved_below_power`, `closing_ground=null`, league `nfl`,
seasons `2020-2025` (except the single-season cell). Verified present via
`nfl-ats weak-signals status` after recording (registry total went 239 ->
263). Names (prefix `opener_error_mining_`): `movement_agreement_agrees`,
`movement_agreement_disagrees`, `movement_agreement_flat`,
`spread_magnitude_10plus`, `spread_magnitude_7_9p5`,
`spread_magnitude_3_6p5`, `spread_magnitude_0_2p5`, `total_bucket_45_48`,
`total_bucket_below_42`, `slate_primetime`, `rest_diff_away_more_rested`,
`rest_diff_even`, `favorite_side_home_favorite`,
`favorite_side_road_favorite`, `season_2025`, `week_third_early`,
`week_third_late`, `confidence_bucket_lt0p02`,
`confidence_bucket_0p02_0p05`, `confidence_bucket_0p05_0p10`,
`confidence_bucket_gt0p10`, `division_game_yes`, `pick_side_favorite`,
`pick_home_away_home`. The remaining ~16 computed-but-unrecorded cells (the
complements/small-n pick'em cells, `total_bucket`'s two middling buckets,
`season` 2020-2024, `division_game_non_division`, `pick_side_dog`,
`pick_home_away_away`) are fully reported in the table above; they were not
separately recorded because they are the algebraic complement or a
lower-|effect|/lower-n version of an already-recorded cell in the same
family, not because they were judged uninformative on their own.

## Ranked leads

Ordered by |effect|·sqrt(n) (the same ranking as the top-10 table above),
one sentence each on the exploitable policy or feature. Every effect below
is **measured** on this run; every proposed mechanism is **inferred** (my
reasoning, not evidence) and labeled as such.

1. ~~**Movement-agreement fade (measured -5.99 pts agrees / +3.60 pts
   disagrees, n=494/639, largest \|effect\|·sqrt(n) in the battery).** When
   the line's observed move from Tuesday open to close later confirms our
   forced pick's side, the model does *worse* at the opener grade than when
   the market moves against it. Since picks stay editable until kickoff and
   observed movement is a legal late-week signal (already the subject of
   `docs/observed_movement_channel.md`), this is a candidate contrarian
   overlay: **treat late confirmation from the market as a reason to trust
   the pick *less*, not more** -- inferred mechanism: this looks like a
   reverse-line-movement/public-money pattern (public steam pushing a
   number toward the side that then fails to cover), but this document does
   not test that mechanism directly.~~ **CORRECTED 2026-08-20, see
   "Reconciliation with `docs/observed_movement_channel.md`" below: the
   `agrees`/`disagrees` labels above were swapped (a construction bug, not
   a data error).** The n=494 cell (-5.99 pts) is actually `disagrees`; the
   n=639 cell (+3.60 pts) is actually `agrees`. Read correctly, the effect
   *reverses sign*: the model does *better* when movement confirms its
   pick and *worse* when movement contradicts it -- the opposite of a
   contrarian-fade mechanism, and fully consistent with (not independent
   of) `docs/observed_movement_channel.md`'s already-wired
   movement-following overlay. **This needs reconciliation against
   the already-recorded `movement_oracle` and `movement_direction_tilt_*`
   families before being treated as a new independent lead** -- both touch
   the same observed-movement ground from a different angle (oracle: pick
   the side movement favors, regardless of the model; this cell: does the
   model's *existing* pick agree with movement) and could be describing the
   same underlying effect. That reconciliation is now done; see below.
2. **Big-spread dampener (measured -7.91 pts at 10+, -4.38 pts at 7-9.5,
   n=154/245; the 10+ cell's whole CI sits below zero).** The model loses
   accuracy on the largest opener spreads while the two smallest buckets
   (0-2.5, 3-6.5) are both positive. A magnitude-based shrinkage --
   regressing `home_cover_probability` toward 0.5 as `|opener spread|`
   crosses roughly 7, or excluding 10+ spreads from Best Pick eligibility
   -- is a direct, cheap policy change to try.
3. **Home-favorite vs road-favorite asymmetry (measured -2.29 pts home
   favorite vs +1.62 pts road favorite, n=607/884).** Combined with lead 2
   (home favorites skew toward the large-spread bucket), this suggests the
   weakness is specifically pricing *home* favorites, not favorites in
   general -- worth a home-favorite-specific interaction term rather than a
   magnitude-only dampener.
4. **Total-line middle-high band (measured -4.36 pts in (45, 48], n=349,
   the only negative of the four total quartiles).** Games with totals in
   roughly the 45-48 range underperform while the other three quartiles are
   all positive. Inferred: this band may capture higher-variance,
   pass-heavy games where a margin-residual model is intrinsically noisier;
   worth checking whether `total_line` or its interaction with pace/EPA
   features belongs in the weak_stack profile directly.
5. **Primetime edge (measured +3.15 pts primetime vs -0.91 pts Sunday
   slate, n=338/1165).** The model does better in nationally-televised
   games. Inferred: possibly better data coverage/injury reporting on
   primetime games, or primetime lines carry more public-money distortion
   that the model already fades well. Actionable as a small primetime
   confidence boost, or at minimum as grounds not to discount primetime
   picks.
6. **Away-team-rested weakness (measured -3.55 pts when the away team has
   more rest, n=265, vs +1.41 pts when rest is even).** The model
   specifically underperforms when the road team is better rested than
   home -- worth auditing `rest_diff`'s away-side calibration in the
   `weak_stack` feature set rather than assuming the feature works
   symmetrically.
7. **Late-season fade (measured -1.75 pts weeks 13-18 vs +1.79 pts weeks
   1-6, n=496/524).** Early season is the model's best stretch and late
   season its worst, among week-thirds. Inferred: late-season games include
   more motivation-mismatch/tanking situations the current feature set may
   not fully capture; a week-of-season interaction term or tighter Best
   Pick screening in weeks 13-18 is the natural next step.
8. **Division-game dampener (measured -1.33 pts division vs +0.75 pts
   non-division, n=542/961).** Consistent with well-known market lore that
   division games run closer to a coin flip; supports a division-game
   eligibility discount for Best Pick specifically, on top of whatever the
   model's raw probability already says.
9. **Confidence-bucket sizing: not yet supported.** The calibration check
   above found no monotonic relationship between `|p-0.5|` and hit rate
   at this resolution -- **do not** lean harder on raw
   `home_cover_probability` magnitude for Best Pick or refresh-pass sizing
   based on this document; if that policy is wanted, it needs its own
   higher-power test (smoothed reliability curve, more seasons, or a
   pooled look), not a read of these 4 buckets.
10. **Pick-side and home/away: negative controls, not leads (measured
    pick_home_away +0.25/-0.18 pts, pick_side +1.00/-0.83 pts, n in the
    600-900s, all P+ near 0.5).** No gross home/away or favorite/dog bias
    is visible in the current model at the opener grade -- useful for
    ruling out two obvious confounds before chasing subtler ones, and
    consistent with the model not simply re-deriving "always take the
    favorite" or "always take home."

## Caveats

- Mined battery: 12 predeclared families, ~40 cells, no multiplicity
  correction. Read every P+ and interval as exactly what it is, not as a
  significance test.
- `total_bucket` edges are quartiles of this exact population -- a
  data-dependent choice made once, not re-tuned after seeing hit rates.
- `slate` (primetime definition) is a stated approximation: Sunday/Saturday
  kickoffs are flagged primetime only at `gametime >= 20:00`; early/mid
  Saturday games (rare, only recorded in this archive) fall into
  `sunday_slate`.
- Week-blocked intervals resample only weeks that contain at least one game
  in the cell; thin cells (`favorite_side=pick_em`, n=12) have very few
  blocks and correspondingly enormous intervals -- reported for
  completeness, not as evidence of anything.
- This entire analysis approximates "the active model if the market had
  stopped at Tuesday's opener" the same way `docs/opener_evaluation.md`
  does: only `spread_line` is swapped to the opener value; `total_line` and
  every other feature are close-era values.

## Reconciliation with `docs/observed_movement_channel.md` (added 2026-08-20)

**Verdict: (ii) construction bug.** The `movement_agreement=agrees` and
`movement_agreement=disagrees` cell labels in this document's Predeclared
Cell 8 (and every place they are referenced above) were **swapped**. The
underlying numbers -- game counts, hit rates, effects, week-blocked
intervals, `P+` -- are all correct; only the two labels were exchanged when
the cell was written up. Once relabeled, this document and
`docs/observed_movement_channel.md` (the ESTABLISHED, already-production
movement-refresh policy) measure the same effect and agree completely: they
are not in tension.

### How the bug was found

`scripts/observed_movement_channel.py`'s own Arm 1 diagnostic reports
`model_agrees_already_fraction_all_nonzero_movement = 0.5586206896551724`
on the pre-push-exclusion 1160 nonzero-movement games (out of the tracked
1537), i.e. **648 games where the movement direction already matches the
production pick, 512 where it does not**
(`artifacts/observed_movement_channel/20260820T093426Z/metadata.json`,
`cells_summary.csv` row `1_oracle`, `n_flip_changed=512`). This uses the
*identical* `open_move`/`pick_home_at_open_probability_rule` columns this
document's Cell 8 uses -- same `opener_pick_evaluation` output, same
`open_move = close_home_spread - tue_open_home_spread` formula (this
document's own stated definition, matching `nfl_ats/clv.py` line 1982
exactly).

This document's push-excluded population (n=1503, 34 opener pushes
dropped, 27 of them on nonzero-movement games and 7 on zero-movement
games) can only ever be a strict *subset* of that full 1160/648/512 split
-- push exclusion can only remove games, never add them. But this
document's `agrees` cell reports n=494 and its `disagrees` cell reports
n=639. **639 exceeds 512**, the full pre-push-exclusion count of games
where movement disagrees with the pick -- mathematically impossible for a
subset built by removing pushes, unless the `disagrees` label is actually
being applied to the *agrees* population. Checking the arithmetic both
ways confirms it exactly: `648 - 9 = 639` and `512 - 18 = 494`, and
`9 + 18 = 27`, precisely the number of nonzero-movement opener pushes in
the archive. The two cells are the `agrees`/`disagrees` split with the
labels exchanged, not a different or buggy computation.

Independently, this session recomputed both cells from scratch on
`artifacts/observed_movement_channel/20260820T093426Z/per_game_tue_close.parquet`
(push-excluded to n=1503, matching this document's own population) using
the CORRECT convention -- `agrees` iff `(open_move > 0) == pick_home_at_open_probability_rule`
-- and `nfl_ats.clv.week_blocked_bootstrap` with this document's own spec
(week blocks, 2000 samples, seed 20260820): it reproduces `n=639` /
`hit_rate=56.96%` / `effect=+3.60 pts` / `CI=[-0.22,+7.60]` / `P+=0.965`
for the TRUE `agrees` cell, and `n=494` / `hit_rate=47.37%` /
`effect=-5.99 pts` / `CI=[-10.42,-1.93]` / `P+=0.002` for the TRUE
`disagrees` cell -- matching this document's originally-reported numbers
to within ordinary bootstrap resampling noise, under the swapped labels.

### Recomputed table (measured 2026-08-20, `scratchpad/recompute_movement_agreement.py`)

All rows: opener grade, `pick_home_at_open_probability_rule`, push-excluded
(n=1503 for the unfiltered population), week-blocked 95% CI (2000 samples,
seed 20260820). "Market pick" is the `docs/observed_movement_channel.md`
policy pick (movement side when the line moved, falling back to the
production pick on an exact tie) -- identical to that document's Arm 1/2
`oracle_pick_home` construction.

| Population | Cell | n | Model hit rate | Model effect (pts) | Model CI | Model P+ | Market-pick hit rate | Market effect (pts) | Paired delta (market−model, pts) | Paired CI | Paired P+ |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|---:|
| Unfiltered (any nonzero move) | agrees (true) | 639 | 56.96% | +3.60 | [-0.22, +7.60] | 0.965 | 56.96% | +3.60 | +0.00 | -- | -- |
| Unfiltered | disagrees (true) | 494 | 47.37% | -5.99 | [-10.42, -1.93] | 0.002 | 52.63% | -0.73 | **+5.26** | [-2.86, +14.12] | 0.880 |
| Unfiltered | flat | 370 | 55.14% | +1.78 | [-3.98, +7.32] | 0.732 | 55.14% | +1.78 | +0.00 | -- | -- |
| \|open_move\| >= 1.0 (B's eligibility filter) | agrees | 363 | 59.78% | +6.42 | [+1.21, +11.44] | 0.993 | 59.78% | +6.42 | +0.00 | -- | -- |
| \|open_move\| >= 1.0 | disagrees | 290 | 45.17% | -8.19 | [-14.23, -2.02] | 0.008 | 54.83% | +1.47 | **+9.66** | [-2.68, +21.74] | 0.935 |

Two consistency cross-checks against the already-recorded
`observed_movement_*` family, both exact: (1) diluting the unfiltered
`disagrees` paired delta by its population share, `0.329 * 5.26 = 1.73`
pts, matches `observed_movement_oracle_full_slate`'s registered
`paired_delta_point = 1.7299` pts almost exactly; (2) diluting the
threshold-filtered `disagrees` paired delta the same way,
`0.193 * 9.66 = 1.86` pts, matches `observed_movement_threshold_1_0`'s
registered `paired_delta_point = 1.8629` pts to three decimal places. Where
the market and the model disagree is *exactly* where
`docs/observed_movement_channel.md`'s threshold overlay earns its
population-wide edge; this document's mined cell was measuring the same
effect from the model's side, mislabeled.

### What this means for Lead 1 (movement-agreement fade)

The original Lead 1 above proposed a **contrarian** overlay: fade the
market when it confirms the pick. That does not survive -- it was built on
swapped labels, and the corrected sign is exactly the opposite. Read
correctly: **the model's own opener pick is more likely to be right when
observed Tuesday-to-close movement already agrees with it (56.96%, +3.60
pts), and less likely to be right when movement disagrees with it (47.37%,
-5.99 pts, whole week-blocked CI below zero)**. That is not a new,
independent contrarian lead; it is this document's own confirmation, from
the model's side of the ledger, of the mechanism `docs/observed_movement_channel.md`
already measured and already wired into production: when the market and
the model disagree by enough (`|open_move| >= 1.0`), flip to the market.

What IS new here, and does survive, is a **sharper characterization of
where that policy's edge concentrates**: within the disagreement
population specifically, the market-side pick beats the model by +5.26 pts
unfiltered (n=494, P+=0.880) and +9.66 pts once restricted to
`|open_move| >= 1.0` (n=290, P+=0.935) -- both `unresolved_below_power`
per AGENTS.md (their week-blocked intervals cross zero, the expected shape
at this evaluator's resolution, not a rejection), but both point the same
direction as, and are numerically consistent with, the already-promoted
`observed_movement_threshold_1_0`/`observed_movement_oracle_sunday_1600_realism`
policy. This does not license a NEW overlay beyond what is already wired
(the production Sunday-realism threshold-1.0 rule already targets almost
exactly this same disagreement-and-large-move population); it is
independent evidence, from a different mined battery, that the existing
policy is pointed at the right games.

### Registry corrections

Two original entries were annotated **in place** (not deleted, per
AGENTS.md's "annotate, don't erase" instruction) to flag the swapped label
and point to the corrected entry:
`opener_error_mining_movement_agreement_agrees`,
`opener_error_mining_movement_agreement_disagrees`. Four new entries were
recorded: `opener_error_mining_movement_agreement_agrees_corrected` (n=639,
+3.60 pts), `opener_error_mining_movement_agreement_disagrees_corrected`
(n=494, -5.99 pts), and two new paired-delta findings,
`opener_error_mining_movement_agreement_disagrees_overlay_paired_delta`
(n=494, +5.26 pts) and
`..._disagrees_overlay_paired_delta_move_ge_1_0` (n=290, +9.66 pts). All
six remain `unresolved_below_power`; none of this reconciliation produces
an admissible `refuted_mechanism` or `bounded_by_control` closure under
AGENTS.md's taxonomy -- the last two are explicitly flagged in the
registry as correlated decompositions of the already-recorded
`observed_movement_oracle_full_slate`/`observed_movement_threshold_1_0`
entries (same archive, same movement definition, a subpopulation cut, not
an independent sample) and must not be pooled additively with them.
Verified present via `nfl-ats weak-signals status` after recording
(registry total 271 -> 275: two annotated in place, four new).
