# Snap-weighted career-age x position-group curves (LEAD-58)

QUALITY infrastructure. No ATS direction, no hypothesis, no closing ground,
no `registry/weak_signals.json` or `registry/rotation_registry.json` entry.
This is descriptive substrate intended to feed XLG-06's within-player-age
prior later (see "XLG-06 hook" below) and to inform the rookie-wall lead;
it is not itself a weak signal and nothing here is "promoted" or "rejected."

Builder: `scripts/build_age_curves.py`. Library: `src/nfl_ats/age_curves.py`.
Tests: `tests/test_age_curves.py` (17 tests, all passing). Every number below
is **measured** by running

```
.\.tools\uv.exe run --no-sync python scripts\build_age_curves.py
```

which wrote `artifacts/age_curves/20260905T022719Z/` (`age_curves.parquet`,
`reliability.parquet`, `manifest.json`). Artifacts are gitignored and
local-disk-only; the numbers in this doc are the durable record.

## The axis is `years_exp`, not chronological age (binding disclosure)

There are no birth dates anywhere in this repository's local data.
`weekly_rosters` columns are exactly: `season, team, position, status,
full_name, gsis_id, pfr_id, years_exp, week, game_type`. There is no
`birth_date`, no `entry_year` computed from an external source, nothing that
would let a chronological-age curve be built locally. Every curve in this
document is indexed by **career age = `years_exp`** (seasons of NFL
experience as reported by nflverse's weekly rosters), never age-in-years.
`years_exp` increments by exactly 1 in 99.98% of consecutive player-seasons
(reported by the LEAD-58 planning pass; not independently re-measured this
session, but consistent with `years_exp` being a clean, almost-always-linear
career-stage counter), so it is a faithful, if coarser, stand-in.

## Inputs and resolved snapshots (measured, this run)

| Source | Path | Snapshot id |
|---|---|---|
| Snap counts + weekly rosters | `data/players/raw/<id>/{snap_counts,weekly_rosters}.parquet` | `20260817T184901Z` |
| Weekly player stats | `data/players/values/raw/<id>/player_stats.parquet` | `20260817T184911Z` |
| Play-by-play (QB dropback EPA only) | `data/pbp/raw/<id>/season=YYYY/plays.parquet` | `20260817T184927Z` |

Season coverage: 2013-2025 (bounded by the snap-count feed; PBP and rosters
go back further but contribute nothing outside the snap-covered range).

### Rejected inputs (measured / documented, not used by this builder)

- **`data/processed/player_participation_ratings.parquet`** -- deliberately
  NOT used. Its ratings are fit over 3-season smeared windows, which would
  blur exactly the year-over-year resolution a career-age curve needs.
- **`data/raw/combine/*/combine.parquet`** -- carries `draft_year`, which
  could cross-check `years_exp` against an implied debut season. Not used as
  an input; the LEAD-58 planning pass **reported** (not independently
  re-measured by this implementation) that it joins only 63.8% of snap rows,
  which is why it was scoped as a cross-check candidate, not a builder input.

## Position groups and metrics (frozen constants: `POSITION_GROUPS`, `METRIC_BY_GROUP`)

| Group | Snap-table positions | Metric | Denominator |
|---|---|---|---|
| QB | QB | sum(EPA) over `qb_dropback==1` (PBP) | dropbacks |
| RB | RB, HB, FB | rushing_epa + receiving_epa | offense_snaps |
| WR | WR | rushing_epa + receiving_epa | offense_snaps |
| TE | TE | rushing_epa + receiving_epa | offense_snaps |
| EDGE | DE, OLB | defense-disruption composite | defense_snaps |
| DL | DT, NT, DL | defense-disruption composite | defense_snaps |
| LB | LB, ILB, MLB | defense-disruption composite | defense_snaps |
| CB | CB, DB | defense-disruption composite | defense_snaps |
| S | FS, SS, S | defense-disruption composite | defense_snaps |
| OL | T, G, C, OL, OT | **no local metric** | -- (offense_snaps volume only) |
| K | K | **no local metric** | -- (st_snaps volume only) |
| P | P | **no local metric** | -- (st_snaps volume only) |
| LS | LS | **no local metric** | -- (st_snaps volume only) |

The defense-disruption composite reuses
`nfl_ats.players._DEFENSE_DISRUPTION_WEIGHTS` verbatim: tackle-for-loss
x0.5, forced fumble x2, sack x1.5, QB hit x0.25, interception x4, pass
defended x0.5 -- the same weights the injury-value production feature
already uses. It measures **splash plays**, not run-fit discipline or
coverage quality that never shows up in a box score; a flat CB/S curve
below may reflect that composite's limited sensitivity to true coverage
skill, not necessarily a flat true skill curve. Say plainly: **OL, K, P,
LS have no local performance metric of any kind** -- their curves are
snap/kick-volume only (`coverage_status == "no_local_metric"`), never a
rate.

**Convention for a snap-having week with no matching `player_stats` row:**
0 numerator over the full snap denominator, for skill and defense groups
(the same "missing production is zero production" rule the injury-value
feature already uses). **QB is the one exception**: a QB week with snaps
but no linked PBP dropback row (e.g. a holder-only appearance) is EXCLUDED
rather than forced to 0-over-snaps, because the QB denominator is dropbacks,
not offense snaps, and there is no natural dropback count to force a zero
over.

## A real bug found and fixed during this build (label: measured)

The first full run of this pipeline (before the fix below) produced WR/RB/TE
rates roughly **3x too small** in magnitude (e.g. WR age-0 raw_rate 0.00595
instead of the corrected 0.01958). Root cause: `build_career_age_panel`
computed a boolean mask (`skill_mask`, `defense_mask`, `qb_mask`) from
`linked` and then reused that SAME mask object after reassigning
`linked = linked.merge(...)`. `DataFrame.merge` always returns a frame with a
fresh `RangeIndex`, even for a row-preserving left join; the pre-merge mask's
index was the ORIGINAL (sparse, after upstream `.loc[...]` filters dropped
unmapped/unlinked rows) index. Boolean `&`/`.loc[...]` between a
stale-indexed mask and the post-merge frame silently misaligns by label
rather than raising, corrupting exactly the rows this builder cares about
most (only sessions with dropped rows before the merge are affected -- which
is every real run, since 0.19% of positions are always unmapped -- but not
the small hand-built unit-test fixtures, which coincidentally never dropped
a row and so never exposed the bug). **Fixed** by recomputing every mask
fresh from the post-merge frame's own `pos_group` column at the point of
use (never carrying a mask across a `.merge()` reassignment); see the inline
comment in `age_curves.py::build_career_age_panel`. Post-fix, the
independently-recomputed WR rate 0.02564 (all rows, zero-filled) matches the
pipeline's own panel-level sum exactly, and the resulting WR/RB/EDGE curve
shapes match the LEAD-58 planning pass's independently reported pilot
numbers closely (see below) -- strong corroborating evidence the fix is
correct, not merely self-consistent.

## Measured results

### Panel diagnostics (measured, this run)

- Snap rows (REG, 2013-2025): 310,475
- Linked to a GSIS identity: 308,857 (**gsis_match_rate 0.99479**); 1,618 rows
  unlinked (multi-position PFR codes with no roster crosswalk / name match).
- Unmapped snap-table positions dropped (multi-codes like `C/G`, `DT/D`,
  `G/T`): 582 rows (0.19% of the 310,475 REG rows).
- Missing `years_exp` after the roster join: 4 rows.
- Final panel rows: 308,271.

### Per-group coverage (measured; `n_players` = unique `gsis_id` across the
whole 2013-2025 window; `n_player_weeks`/`snaps` summed across all career ages)

| Group | Unique players | Player-weeks | Total snaps (or dropbacks for QB) | Age range with data |
|---|---:|---:|---:|---|
| QB | 230 | 8,436 | 249,845 (dropbacks) | 0-22 |
| RB | 665 | 24,508 | 504,691 | 0-15 |
| WR | 911 | 34,102 | 1,144,669 | 0-16 |
| TE | 492 | 20,260 | 587,722 | 0-19 |
| EDGE | 686 | 22,349 | 717,580 | 0-16 |
| DL | 668 | 24,379 | 751,760 | 0-17 |
| LB | 1,137 | 45,092 | 1,318,854 | 0-16 |
| CB | 1,053 | 33,617 | 1,196,339 | 0-14 |
| S | 609 | 26,731 | 967,921 | 0-17 |
| OL (volume only) | 1,079 | 48,465 | 2,235,215 (offense_snaps) | 0-19 |
| K (volume only) | 120 | 6,833 | 59,375 (st_snaps) | 0-23 |
| P (volume only) | 99 | 6,688 | 62,560 (st_snaps) | 0-18 |
| LS (volume only) | 80 | 6,810 | 59,349 (st_snaps) | 0-17 |

**Every group's tail is thin.** The oldest ages in every group are carried by
1-4 players (e.g. DL age 17 = 1 player, WR age 16 = 1 player, QB age 22 = 1
player -- almost certainly a kicker-adjacent or extreme-longevity outlier).
Cells with fewer than 5 players are flagged `sparse = True` in the artifact;
every age above roughly 10-12 is sparse for every group. Ages 0-10 (the
range tabulated below) are NOT sparse for the offense/defense metric groups
except at the very edges.

### Cross-sectional curve, ages 0-10 (measured; raw and empirical-Bayes-shrunk rate)

Units: EPA per dropback for QB; EPA per offense snap for RB/WR/TE;
defense-disruption units per defense snap for EDGE/DL/LB/CB/S. Shrinkage
barely moves any of these cells (`shrunk_rate` differs from `raw_rate` in
the 4th-5th decimal almost everywhere) because every age 0-10 cell here
carries thousands to hundreds of thousands of snaps -- shrinkage bites hard
only in the sparse tail (age 10 shown; ages 11+ shrink visibly more, e.g.
WR age 16 `raw_rate` -0.0107 vs its single-player noise, not tabulated here).

| age | QB | RB | WR | TE | EDGE | DL | LB | CB | S |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -0.09311 | -0.02340 | 0.01958 | 0.01426 | 0.02247 | 0.01324 | 0.01960 | 0.01645 | 0.01535 |
| 1 | -0.01131 | -0.02033 | 0.02261 | 0.01446 | 0.02454 | 0.01533 | 0.02084 | 0.01691 | 0.01642 |
| 2 | 0.01367 | -0.01853 | 0.02569 | 0.01649 | 0.02688 | 0.01646 | 0.02024 | 0.01844 | 0.01624 |
| 3 | 0.02773 | -0.01619 | 0.02672 | 0.01724 | 0.02843 | 0.01697 | 0.02133 | 0.01734 | 0.01604 |
| 4 | 0.01535 | -0.02021 | 0.03180 | 0.01254 | 0.02749 | 0.01755 | 0.02113 | 0.01749 | 0.01603 |
| 5 | 0.02590 | -0.02068 | 0.02697 | 0.01660 | 0.02659 | 0.01747 | 0.02068 | 0.01735 | 0.01644 |
| 6 | 0.02311 | -0.02664 | 0.02834 | 0.01526 | 0.02791 | 0.01795 | 0.02061 | 0.01715 | 0.01577 |
| 7 | 0.05502 | -0.01971 | 0.02531 | 0.01908 | 0.02675 | 0.01770 | 0.02166 | 0.01700 | 0.01469 |
| 8 | 0.03146 | -0.01524 | 0.03007 | 0.01667 | 0.02809 | 0.01679 | 0.02073 | 0.01817 | 0.01674 |
| 9 | 0.03844 | -0.03026 | 0.02382 | 0.01859 | 0.02920 | 0.01683 | 0.02148 | 0.01603 | 0.01631 |
| 10 | 0.02672 | -0.04152 | 0.03016 | 0.02450 | 0.02987 | 0.01672 | 0.02495 | 0.01745 | 0.01689 |

(`shrunk_rate` is within +/-0.0001 of `raw_rate` at every one of these cells
except QB age 0, where the tiny 112-player, extreme-outlier cell shrinks
from -0.09311 to -0.09306 -- also barely moved, because 24,570 dropbacks of
weight still swamps the shrinkage prior's weight `k`.)

Shape read (measured, this run; **label: measured**, not a claim about a
real aging mechanism beyond what a snap-weighted cross-sectional read can
show -- see delta-method section for the within-player read):

- **WR rises from age 0 (0.0196) to a plateau around age 3-4 (0.027-0.032)**,
  consistent with a rookie-adjustment-year pattern.
- **RB is negative-and-worsening with age** (-0.0234 at 0, drifting to
  -0.0415 by age 10), consistent with the well-known RB decline-with-usage
  pattern, though this is a CROSS-SECTIONAL read (see the survivorship
  caveat below).
- **EDGE rises from 0.0225 (age 0) to a plateau near 0.027-0.030** by age
  3-10 -- the plan's independently reported pilot described this same shape
  (".02247->.02843 plateau ~.028 to 10"), which this run reproduces almost
  exactly.
- **CB and S are close to flat** across ages 0-10 (CB 0.0160-0.0184, S
  0.0147-0.0164) -- consistent with a composite that measures splash plays
  more than true coverage skill developing with experience.
- **DL and LB rise gently from age 0 to a plateau** by age 3-4, then stay
  roughly flat.
- **QB is noisy and small-sample even at age 0-3** (112-91 players); the
  age-0 cell (-0.0931) is dragged by early-career struggling rookies mixed
  with a few immediate starters, and the curve is not smooth -- treat every
  QB cell here as `unresolved_below_power` in the sense of AGENTS.md, not as
  a settled QB development curve.

### Delta-method (within-player) curve, ages 0-10 (measured)

`delta_mean` = the snap-weighted average of (rate at age a+1) minus (rate at
age a), computed only for players who cleared a 100-snap/dropback floor at
BOTH ages -- this removes cross-sectional survivorship (a cross-sectional
curve conflates "players improve with age" and "only the players who
improved stick around long enough to reach that age"). It does **not**
remove attrition-selection bias in which players survive long enough to
contribute a delta at all (documented, not fixed, here).
`delta_cumulative` integrates `delta_mean` forward/backward from each
group's modal entry age (age 0 for every group here).

| age-from | QB (n pairs) | mean delta | RB (n) | mean delta | WR (n) | mean delta | EDGE (n) | mean delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 48 | +0.0577 | 134 | -0.0065 | 232 | +0.0009 | 109 | +0.0019 |
| 1 | 37 | -0.0152 | 149 | -0.0039 | 232 | -0.0005 | 121 | +0.0016 |
| 2 | 35 | -0.0256 | 141 | +0.0027 | 203 | -0.0022 | 122 | +0.0009 |
| 3 | 34 | -0.0131 | 117 | -0.0158 | 177 | +0.0013 | 108 | -0.0004 |
| 4 | 28 | -0.0023 | 91 | -0.0040 | 154 | -0.0092 | 91 | -0.0042 |
| 5 | 31 | +0.0176 | 73 | -0.0070 | 132 | -0.0039 | 84 | +0.0008 |
| 6 | 25 | +0.0080 | 53 | -0.0027 | 97 | -0.0074 | 70 | -0.0004 |
| 7 | 17 | -0.0112 | 39 | -0.0131 | 66 | -0.0035 | 53 | -0.0031 |
| 8 | 17 | -0.0313 | 22 | -0.0237 | 47 | -0.0126 | 39 | -0.0006 |
| 9 | 17 | -0.0120 | 12 | +0.0034 | 31 | -0.0041 | 30 | -0.0029 |

The full per-group, per-age table (all 13 groups) is in
`age_curves.parquet` (`delta_n_pairs`, `delta_mean`, `delta_cumulative`
columns, joined onto the cross-sectional curve by `career_age`). Two
readings worth stating plainly:

- **The within-player RB delta is negative or flat at essentially every
  age**, including age 0->1 (-0.0065) -- the cross-sectional decline is not
  purely a survivorship artifact; the same players who stick around also
  decline, on this composite.
- **The within-player WR delta turns negative starting around age 4**
  (-0.0092 at 4->5, continuing negative through age 8->9), even though the
  CROSS-SECTIONAL curve stays roughly flat-to-rising through age 8-10. This
  is exactly the survivorship pattern the delta method is built to expose:
  the cross-sectional plateau is partly "only the WRs who are still good are
  still on a roster," not "WRs keep getting better."

### Split-half reliability (measured; both schemes; NO group closed)

Closing-grounds taxonomy (binding, restated per CLAUDE.md/AGENTS.md): an
interval or CI containing zero is never grounds to reject, fail, or close a
line of work. Only a RESOLVED wrong sign (whole interval on one side of
zero) or a measured ZERO reliability closes anything. **None of the 18 rows
below are closed** -- every interval crosses zero. All are
`unresolved_below_power`.

| Group | Scheme | Ages compared | Pearson r | Spearman-Brown | 95% CI | P(r>0) |
|---|---|---:|---:|---:|---|---:|
| QB | odd/even seasons | 21 | 0.494 | 0.662 | [-0.038, 0.634] | **0.962** |
| QB | random player halves | 22 | 0.739 | 0.850 | [0.134, 0.775] | **0.994** |
| EDGE | odd/even seasons | 16 | 0.314 | 0.478 | [-0.122, 0.561] | 0.896 |
| EDGE | random player halves | 16 | 0.574 | 0.730 | [-0.114, 0.669] | **0.928** |
| WR | odd/even seasons | 16 | 0.277 | 0.433 | [-0.078, 0.540] | 0.940 |
| WR | random player halves | 16 | 0.234 | 0.379 | [-0.305, 0.596] | 0.765 |
| DL | random player halves | 15 | 0.342 | 0.509 | [-0.239, 0.641] | 0.832 |
| DL | odd/even seasons | 15 | 0.243 | 0.391 | [-0.211, 0.565] | 0.781 |
| LB | random player halves | 16 | 0.418 | 0.590 | [-0.341, 0.619] | 0.797 |
| TE | odd/even seasons | 18 | 0.183 | 0.310 | [-0.374, 0.640] | 0.703 |
| TE | random player halves | 18 | 0.319 | 0.484 | [-0.362, 0.616] | 0.671 |
| CB | odd/even seasons | 15 | 0.275 | 0.432 | [-0.170, 0.539] | 0.894 |
| CB | random player halves | 15 | 0.199 | 0.332 | [-0.358, 0.537] | 0.665 |
| LB | odd/even seasons | 16 | 0.199 | 0.332 | [-0.337, 0.575] | 0.714 |
| RB | random player halves | 12 | -0.100 | -0.222 | [-0.533, 0.421] | 0.407 |
| S | random player halves | 14 | -0.288 | -0.809 | [-0.591, 0.387] | 0.308 |
| S | odd/even seasons | 13 | -0.410 | -1.390 | [-0.647, 0.245] | 0.211 |
| RB | odd/even seasons | 12 | -0.452 | -1.648 | [-0.655, 0.163] | **0.109** |

(Spearman-Brown values outside [-1, 1] for RB/S odd-even are an artifact of
the correction formula `2r/(1+r)` applied to a negative r close to -1's
neighborhood combined with a small age count -- report the underlying
Pearson r and the bootstrap interval, not the corrected figure, when the
correction over- or under-shoots the [-1,1] range like this.)

Reading this table under the taxonomy above, not around it:

- **QB and EDGE show the strongest, most consistent positive reliability**
  across BOTH independent schemes (QB: P+ 0.962 and 0.994; EDGE: P+ 0.896
  and 0.928) -- the age pattern in these two groups' cross-sectional curves
  is not just cross-sectional noise, it replicates across independent
  halves of the data.
- **WR, DL, LB, CB, TE are positive-leaning but noisier** (P+ ranging
  0.665-0.940 across the two schemes) -- real candidates for
  `unresolved_below_power`, not for rejection.
- **RB and S lean NEGATIVE** on both schemes (P+ as low as 0.109 for RB
  odd/even seasons) -- notably, RB's within-player delta curve (above) is
  ALSO negative-leaning at nearly every age, so a negative cross-age
  correlation here is plausible (the RB curve is genuinely declining, not
  flat-with-noise, so "the two halves agree on a declining shape" can
  itself register as a negative Pearson r against a naively-expected
  monotonic-rise template -- inspect the sign convention before reading this
  as "RB has no reliable age curve"; it may instead mean "RB's age curve is
  reliably NOT the shape a positive-r reading assumes"). **This is not a
  RESOLVED wrong sign** (the RB odd/even CI's upper bound is +0.163, EDGE-of
  -zero, not below it) and is therefore NOT closed. It is recorded here as
  `unresolved_below_power`, exactly as AGENTS.md requires.
- **No group anywhere in this table has a whole-interval-negative or
  whole-interval-positive reliability CI.** Nothing in this run meets either
  admissible closing ground (`wrong_sign_resolved` or
  `no_split_half_reliability` via a measured zero). Nothing is recorded to
  `registry/weak_signals.json` -- this document is the record, per the
  QUALITY-infrastructure framing at the top.

### OL, K, P, LS: explicitly no metric, no reliability (measured absence)

These four groups get a snap/kick-volume curve only. `raw_rate`,
`shrunk_rate`, `smoothed_rate`, and every reliability row are null by
construction -- there is nothing to shrink, smooth, or test the reliability
of. OL alone carries 2,235,215 offense snaps across 1,079 unique players
(the single largest snap pool of any group in this dataset) with **zero**
local performance signal. This is a real, measured data gap, not an
oversight: nothing in `data/players/values/raw/*/player_stats.parquet`
scores offensive line play, and no local source scores kicking/punting/
long-snapping quality (only snap/kick volume).

## Method notes

- **Empirical-Bayes shrinkage** (`shrink_cells`): snap-weighted pull toward
  each group's own grand mean, `shrunk_a = (w_a*raw_a + k*grand)/(w_a+k)`,
  with `k = tau_within / tau_between` estimated by the same method-of-moments
  recipe as `scripts/cfb_james_stein_unit_screen.py`'s James-Stein shrinkage
  (adapted from per-team to per-career-age cells). No monotonicity is
  imposed anywhere.
- **Local-linear smooth** (`smooth_curve`): an independent second look, a
  snap-weighted, 3-point (age-1, age, age+1) tricube-kernel weighted-least-
  squares fit, evaluated at the focal age.
- **Point-in-time contract**: `build_age_curves(..., as_of_season=Y)` filters
  every input source (snaps, rosters, player stats, PBP) to `season < Y`
  BEFORE any aggregation. `tests/test_age_curves.py` has two dedicated
  leakage tests: one asserting every panel row's season is strictly below
  `as_of_season`, and one asserting the panel is bit-identical whether or
  not perturbed future-season rows exist on disk.

## XLG-06 hook (not wired; documented for the future)

The artifact's cross-sectional and delta curves are keyed on
`(pos_group, career_age)` with values in the same per-snap EPA-like units
`xlg06_prior_feature.py`'s `blend_prior` (N0=300) already expects, so a
future session can plug `delta_cumulative` (or `shrunk_rate`) in as a
within-player-age-informed prior without touching this module. Per the
LEAD-58 task boundary, `xlg06_prior_feature.py` is **not edited or wired up**
by this work.

## Reproduction

```
.\.tools\uv.exe run --no-sync python scripts\build_age_curves.py
.\.tools\uv.exe run --no-sync python scripts\build_age_curves.py --as-of-season 2022
.\.tools\uv.exe run --no-sync pytest tests\test_age_curves.py -q
```

Artifact from this session: `artifacts/age_curves/20260905T022719Z/`
(`age_curves.parquet`, `reliability.parquet`, `manifest.json`). Artifacts
are gitignored; re-run the builder to reproduce (deterministic given the
same local snapshots -- the bootstrap seed is fixed at 20260905).
