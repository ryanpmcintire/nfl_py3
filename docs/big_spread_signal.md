# Does anything predict the cover at 10.5+? (MOD-18 lane AK)

Follow-on to `docs/spread_hole_diagnosis.md` (lane F),
`docs/spread_hole_group_penalty.md` (lane M) and `docs/spread_hole_target.md`
(lane N). Those three lanes located a favourite-ward lean on big spreads,
removed it (T1), and measured that removing it does not buy accuracy. None of
them asked the prior question this lane asks.

Closing-grounds taxonomy, verbatim, because this document reports intervals:
an interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (a) refuted mechanism — a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (b) bounded by a positive
control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never "contains zero".

Within-week correlation is ZERO by owner mandate; every bootstrap here is
week-blocked (season-blocked reported alongside where the harness carries it).
Football margins are discrete and multimodal, never Gaussian. **No arm in this
lane is a rule, a flip, or a candidate for the played card.** This lane is a
DIAGNOSIS. It measures information content and reports it; any change to served
behaviour is a separate, later, predeclared lane.

## Design, frozen 2026-09-10T01:14:53Z before any candidate was scored

Family `mod18_big_spread_signal_v1`.

### The question

On spreads of 10.5 or more the served model is 47.3% right as served (read:
`docs/spread_hole_diagnosis.md:28`), and 10.5+ has **no** long-run market
underdog tilt to be on the wrong side of — 48.85% dog cover rate over 2009-2025
(read: `docs/spread_hole_diagnosis.md:192`). Three lanes have tried to fix the
model's parameterisation there and every fix lost through the card. The
question none of them asked:

> At 10.5+, does ANY input we hold carry out-of-sample information about the
> cover, or is the model adding noise to a coin flip?

If nothing does, that is a **named mechanism** — "no pregame input we hold has
signal at 10.5+" — and the honest served behaviour at 10.5+ is to stop letting
the residual decide there. That is a modelling change for a later lane to
predeclare, not a flip to bolt on today.

### Population and lines

- Feature table: `data/processed/game_features_weak_stack.parquet`, sha256
  `fadeed326dc277416519e75914154c0d7b822b5ef28ba6d19efaa6469cb64e21`, which is
  the ACTIVE model's own table (`artifacts/active_ats_model.json`
  `feature_table_sha256`, `feature_profile` `weak_stack`).
- Regular season only, `nfl_ats.modeling.regular_season_rows`, completed games.
- **Two line surfaces, both reported.**
  - `close_proxy` — the schedule table's own `spread_line`, available
    2009-2025. This is the only surface that reaches the first two eras.
  - `opener` — the Tuesday opener from the market archive
    `artifacts/opener_evaluation/20260910T005854Z` (1,537 games, 2020-2025),
    paired exactly as `nfl_ats.clv.opener_pick_evaluation` pairs it. The
    opener is the project's primary grade, so it is the primary surface
    wherever it exists.
- **Big-spread cut**: `abs(spread_line) >= 10.5`, evaluated on the surface's
  own line. n by era is reported in the results, not asserted here.

### Grading and arms

Every model arm is a walk-forward: one weekly refit on completed regular-season
games strictly before that week's first kickoff, minimum 500 training games,
`Ridge(alpha=10)`, target `market_residual`, mapping `gaussian_median` — the
served configuration. **Every arm is scored RAW, with no S3 home-side offset.**
That is a predeclared choice and it is stated as a handicap where it matters:
the S3 offset is worth +3.1 accuracy points to the incumbent at 10.5+ (read:
`docs/spread_hole_diagnosis.md:34-36`), but it is a served correction fitted
from an arm's own out-of-time stream, and refitting it per family would confound
"does this family carry information" with "does the offset rescue it". The
served incumbent's offset-carrying numbers at 10.5+ are reported separately from
the archive as the reference point.

The 500-game floor means the first scored week is the first week at which 500
completed games exist. Era 1 for the model arms is therefore the part of
2009-2013 that clears that floor; the market and margin facts (parts 3, 4, 5)
use the full 2009-2013 because they need no training set. The exact first
scored season is reported, not assumed.

#### Part 1 — one arm per feature FAMILY

The served `weak_stack` profile's 90 columns carry 12 declared families
(`nfl_ats.margin.FEATURE_FAMILIES`, surfaced by `margin_feature_groups`):
`offense` (21), `defense` (18), `player_continuity` (9), `bias` (9), `context`
(7), `player_injuries` (7), `results` (6), `player_qb` (5), `market` (2),
`elo` (2), `experience` (2), `player_values` (2). Each family becomes one arm:
a runtime feature profile holding only that family's columns, fitted and scored
by exactly the machinery above. Nothing in `src/` is edited; the registration is
a dict insertion made by this lane's script and the resulting column lists are
written to the artifact JSON so they can be audited.

Reported per family, per surface, per era and pooled, restricted to 10.5+:

- **sign accuracy vs the market** — the forced pick is home when the family's
  predicted market residual is positive, graded against the surface's own line,
  pushes excluded. The effect is reported in `accuracy_points` **above the coin
  flip**, i.e. `(accuracy - 0.5) * 100`, with a week-blocked bootstrap (20,000
  samples, seed 20260821) and its `probability_positive`. The market is the
  50% reference because the pool is forced picks against the line.
- **Brier score** of the family model's `home_cover_probability` against the
  realised home cover, pushes excluded, alongside the Brier of a constant 0.5.
- **split-half reliability of the family's signal at 10.5+**: inside each
  walk-forward week the training rows are split by a seeded permutation into
  two disjoint halves, the family model is fitted separately on each half, and
  both are scored on that week's 10.5+ rows. Pooled over every week, the
  reliability is the Pearson correlation between the two prediction streams,
  reported raw and Spearman-Brown corrected (`2r / (1 + r)`). This measures
  whether the family produces a STABLE signal at all, independently of whether
  that signal predicts the cover — which is the construct the closing-grounds
  taxonomy names ("the trait has no split-half reliability"). A reliability
  indistinguishable from zero is the only reading in this lane that could close
  a family, and it is reported with its own bootstrap interval.

#### Part 2 — the full served model, specialised versus served

- **F-served**: the full 90-column `weak_stack` model, trained on ALL games,
  scored at 10.5+. This is the served behaviour restricted to the bucket.
- **F-spec105** (primary specialist): identical, except each week's training
  rows are restricted to `abs(spread_line) >= 10.5`. Minimum 150 training rows,
  predeclared; weeks below the floor are not scored for this arm and the count
  is reported.
- **F-spec75** (secondary specialist, predeclared here before scoring):
  identical with the training restriction at `abs(spread_line) >= 7.5`, because
  the 10.5+-only training set is thin by construction and a wider big-spread
  training set is the obvious second question. Same 150-row floor.

Primary quantity for part 2: the paired accuracy-point difference
`F-spec105 - F-served` on the common scored 10.5+ games, week-blocked, with its
`probability_positive`. Secondary: `F-spec75 - F-served`, and each arm's own
accuracy above the coin flip.

#### Part 3 — the margin-distribution facts at 10.5+

Computed on the full 2009-2025 completed regular season, no model needed:

1. the integer histogram of `result - spread_line` (favourite-ward signed as
   the football recorded it) for `abs(spread_line) >= 10.5`, pooled and by era;
2. the exact-push rate at 10.5+ overall, and specifically for games whose line
   is exactly 14 and exactly 17, plus every other integer line at 10.5+ with at
   least 20 games;
3. whether the served `gaussian_median` read misplaces mass at 10.5+ in a way
   that **flips SIDES**. Lane F showed it cannot, for a symmetric read about the
   same centre (read: `docs/spread_hole_diagnosis.md:246-255`); this lane states
   the number by re-reading the same weekly residual samples under the project's
   own `discrete_residual` method (which rounds `centre + residual` to the
   integer lattice, `src/nfl_ats/calibration.py:444-447`) and counting the
   10.5+ games whose side changes. The Brier of both reads at 10.5+ is reported
   alongside, because a mass-correct probability can be worth having even when
   it changes no side.

#### Part 4 — the market's own behaviour at 10.5+

1. **Opener-to-close move.** On the opener archive, restricted to
   `abs(opener spread) >= 10.5`: the accuracy of following the move (pick the
   side the line moved toward, `close - open`), graded at the opener, on
   2023-2025 (the late-week follow rule's declared exposure) and on 2020-2025
   for the wider n. Games with zero move are excluded and counted.
2. **Heavy handle.** The action-network archive
   `data/raw/public_betting/20260820T111148Z/actionnetwork/index.parquet`,
   matched to the schedule exactly as `scripts/public_betting_battery_screen.py`
   matches it (team-pair merge, kickoff within 72 hours, latest pregame capture
   per game). Restricted to 10.5+ and to games where `spread_home_money_pct` is
   present, the accuracy of taking the side holding the larger share of money,
   graded at the surface's own line. Money share exists only for 2023 onward in
   that archive, so this is a small-n read by construction and is reported as
   such.

#### Part 5 — era stratification

Eras 2009-2013 / 2014-2019 / 2020-2025, applied to every cell in parts 1, 3 and
4 that reaches them (part 2's paired arms and part 4.1's opener arm exist only
where their inputs do). Per the owner's rule, era readings differ in MAGNITUDE
and a weaker era is never absence.

### Recording

Every cell is recorded with `nfl-ats weak-signals record`, units
`accuracy_points`, league `nfl`, family `mod18_big_spread_signal_v1`, named
`big_spread_signal_<family|arm>_<window>`, with each family arm's split-half
reliability carried in the `--reliability` field so the cell can be adjudicated
later. The exact argv lists are saved as `record_commands.json` in the artifact
directory and are run one at a time.

**Terminal classification rules, declared here before any cell was scored.** A
cell is recorded `unresolved_below_power` unless one of exactly two conditions
holds:

- `wrong_sign_resolved` — BOTH the week-blocked and the season-blocked 95%
  interval sit entirely below zero. For an accuracy-above-the-coin-flip cell
  that means the arm is resolvedly worse than a coin flip at 10.5+, which
  refutes the stated mechanism ("this input predicts the cover") in its stated
  direction. Nothing else, and in particular no interval that merely contains
  zero, may close a cell.
- `no_split_half_reliability` — the family's 95% week-blocked reliability
  interval lies entirely inside [-0.05, +0.05], i.e. the reliability is
  resolved AT zero rather than merely unmeasured. A wide interval that happens
  to include zero is not this condition and closes nothing.

If a record command errors, the verdict is wrong, not the validator, and the
cell is reclassified `unresolved_below_power`.

### Reuse discount, stated up front

The 2020-2025 opener window has been used by lanes F, M and N on this same
question. That is a stated discount, not a ban (AGENTS.md: windows retire
per-family, and a reused window carries a stated discount). The
2009-2019 close-proxy surface is NEW to this question and carries no such
discount.

### What this lane may and may not do

It may name a mechanism and it may name the exact arm a follow-up lane would
run. It may **not** change any served behaviour, touch any ledger, manifest,
forecast or board, or propose a threshold flip. The owner's ban on unexplained
threshold flips is binding and a rule that fires only inside a spread bucket is
exactly what that ban names.

## Results

Every number in this section is **measured this session** by the commands at the
bottom; the tables live in `artifacts/big_spread_signal/20260910T011453Z/`.

### The replay is faithful

The `F_served` arm reproduces the archive it is meant to reproduce. Joined game
for game against `artifacts/opener_evaluation/20260910T005854Z/per_game.parquet`
on all **1,537** archived games: maximum line gap **0.0**, maximum residual gap
**0.0173** points, maximum probability gap **0.0025**, and the sign-rule
accuracy matches the archive's own `correct_at_open` exactly — **50.77%** at
10.5+ and **52.96%** over all lines, on both sides of the join. (The small
residual gap is the training cutoff: this lane takes each week's first kickoff
over ALL of that week's games, the archive over the paired games only, so a
handful of weeks train on one or two extra rows.)

The first week that clears the 500-game training floor is **2011 week 1**, so
era 1 for every model arm is **2011-2013**, not 2009-2013. The market and margin
facts below use the full 2009-2025.

### The scale of the question

At `|line| >= 10.5` the lane scores **130** non-push games at the opener
(2020-2025) and **373** at the close proxy (2011-2025). The whole population
2009-2025 is **443** games including pushes. A 2-point effect is far below
resolution in any of those cells, which is the situation the closing-grounds
taxonomy exists for.

### (1) Which inputs carry signal at 10.5+

Forced-pick accuracy above the coin flip, graded at the **opener** (2020-2025,
130 non-push games at 10.5+), week-blocked bootstrap, 20,000 samples, seed
20260821. Split-half reliability is the correlation between two half-training
fits of the same arm, raw and Spearman-Brown corrected, with its own
week-blocked interval.

| arm | accuracy | pts vs coin flip | 95% week | week P+ | season P+ | Brier | split-half r / SB [95%] |
|---|---:|---:|---|---:|---:|---:|---|
| **player_injuries** | **58.46%** | **+8.46** | [+0.00, +16.42] | **0.9764** | 0.9371 | 0.25087 | +0.006 / +0.013 [-0.177, +0.198] |
| **elo** | **56.15%** | **+6.15** | [-1.64, +13.64] | **0.9385** | 0.8320 | 0.25400 | **+0.505 / +0.671 [+0.357, +0.645]** |
| **results** | **55.38%** | **+5.38** | [-2.59, +13.28] | **0.9084** | 0.8799 | 0.25310 | **+0.625 / +0.770 [+0.508, +0.728]** |
| F_spec105 | 53.85% | +3.85 | [-5.17, +12.41] | 0.8000 | 0.8573 | 0.27420 | n/a |
| player_qb | 53.85% | +3.85 | [-5.15, +12.90] | 0.7957 | 0.8391 | 0.25125 | -0.402 / -1.343 [-0.648, -0.015] |
| market | 53.08% | +3.08 | [-4.81, +10.80] | 0.7804 | 0.6755 | 0.25403 | +0.200 / +0.333 [+0.019, +0.371] |
| defense | 53.08% | +3.08 | [-5.19, +11.54] | 0.7674 | 0.6809 | 0.25558 | -0.215 / -0.547 [-0.392, -0.032] |
| **F_served (all 90 columns)** | **50.77%** | **+0.77** | [-7.63, +8.78] | **0.5688** | 0.5523 | 0.26294 | -0.077 / -0.167 [-0.261, +0.110] |
| player_values | 50.00% | +0.00 | [-8.68, +8.93] | 0.4978 | 0.5079 | 0.25377 | -0.325 / -0.961 [-0.477, -0.167] |
| player_continuity | 49.23% | -0.77 | [-9.86, +8.47] | 0.4364 | 0.4541 | 0.25624 | +0.544 / +0.705 [+0.289, +0.690] |
| experience | 46.92% | -3.08 | [-11.36, +4.68] | 0.2225 | 0.3174 | 0.25476 | +0.071 / +0.132 [-0.122, +0.266] |
| offense | 46.92% | -3.08 | [-11.94, +5.47] | 0.2414 | 0.2521 | 0.25985 | +0.097 / +0.176 [-0.089, +0.283] |
| F_spec75 | 46.15% | -3.85 | [-12.79, +4.89] | 0.1967 | 0.1758 | 0.26906 | n/a |
| context | 44.62% | -5.38 | [-13.64, +2.48] | 0.0905 | 0.0810 | 0.25759 | +0.083 / +0.153 [-0.101, +0.257] |
| bias | 41.54% | -8.46 | [-16.10, -0.69] | 0.0172 | 0.0283 | 0.25453 | -0.134 / -0.311 [-0.802, +0.272] |

The same table on the wider **close proxy** surface (2011-2025, 373 non-push
games at 10.5+), which reaches all three eras:

| arm | accuracy | pts vs coin flip | 95% week | week P+ | season P+ | split-half r / SB [95%] |
|---|---:|---:|---|---:|---:|---|
| **elo** | **53.35%** | **+3.35** | [-1.85, +8.42] | **0.9012** | 0.9310 | **+0.219 / +0.359 [+0.082, +0.364]** |
| **player_injuries** | **53.08%** | **+3.08** | [-1.90, +8.12] | **0.8852** | 0.8882 | +0.545 / +0.706 [-0.145, +0.776] |
| **results** | **52.82%** | **+2.82** | [-2.02, +7.52] | **0.8764** | 0.9445 | **+0.380 / +0.551 [+0.262, +0.494]** |
| player_qb | 52.55% | +2.55 | [-2.25, +7.34] | 0.8501 | 0.8638 | -0.181 / -0.443 [-0.350, -0.010] |
| market | 51.74% | +1.74 | [-3.41, +6.84] | 0.7528 | 0.7766 | +0.236 / +0.382 [+0.120, +0.352] |
| F_spec75 | 51.49% | +1.49 | [-3.89, +6.70] | 0.7084 | 0.6923 | n/a |
| **F_served (all 90 columns)** | **51.47%** | **+1.47** | [-3.33, +6.10] | **0.7265** | 0.7219 | +0.009 / +0.018 [-0.290, +0.273] |
| defense | 51.21% | +1.21 | [-3.94, +6.27] | 0.6764 | 0.6788 | -0.111 / -0.250 [-0.277, +0.052] |
| offense | 50.40% | +0.40 | [-4.83, +5.50] | 0.5608 | 0.5418 | +0.038 / +0.074 [-0.066, +0.152] |
| F_spec105 | 50.17% | +0.17 | [-6.04, +6.25] | 0.5202 | 0.5353 | n/a |
| context | 49.33% | -0.67 | [-6.09, +4.73] | 0.3970 | 0.4128 | +0.100 / +0.182 [-0.068, +0.242] |
| experience | 48.53% | -1.47 | [-7.10, +3.92] | 0.2945 | 0.3537 | -0.385 / -1.250 [-0.522, -0.227] |
| player_values | 46.38% | -3.62 | [-8.99, +1.75] | 0.0924 | 0.1274 | -0.166 / -0.397 [-0.450, +0.257] |
| player_continuity | 44.77% | -5.23 | [-10.10, -0.37] | 0.0185 | 0.0120 | +0.016 / +0.031 [-0.118, +0.347] |
| bias | 44.24% | -5.76 | [-10.85, -0.79] | 0.0117 | 0.0287 | -0.162 / -0.386 [-0.605, +0.249] |

**The answer to the lane's question is no — the premise was wrong.** Something
does carry information at 10.5+, and the same three blocks carry it on both
surfaces: `results` (season-to-date scoring margin and ATS residual), `elo`, and
`player_injuries`. On the opener they read 55.4% / 56.2% / 58.5% with
`probability_positive` 0.908 / 0.939 / 0.976; on the close proxy 52.8% / 53.4% /
53.1% at 0.876 / 0.901 / 0.885. **The 90-column model that contains all three
reads 50.77% and 51.47%** — below every one of them, at
`probability_positive` 0.569 and 0.727.

Two guards on that reading, stated before the interpretation:

- **These are 130- and 373-game cells and the leaders are the maximum of 15
  correlated arms**, so the accuracy figures at the top of each table are
  inflated by selection. Every one of them is recorded
  `unresolved_below_power`. What is NOT selection-inflated is the *ordering*
  being the same on two different line surfaces and the reliability column.
- **Split-half reliability separates the three.** `results` (+0.625 opener /
  +0.380 close, both intervals entirely above zero) and `elo` (+0.505 / +0.219,
  same) produce a signal that reproduces when the training set is cut in half.
  `player_injuries` does not on the opener (+0.006, [-0.177, +0.198]) — its
  accuracy is the best in the table and its signal is the least stable, which
  is exactly the pattern a lucky 130-game cell makes. Its reliability interval
  is far too wide to be resolved at zero, so it closes nothing and stays open.

Era stratification of the three, close proxy (per the owner's rule these are
magnitudes, and a weaker era is never absence — but a sign change is reported
as a sign change):

| family | 2011-2013 (n=75) | 2014-2019 (n=136) | 2020-2025 (n=162) |
|---|---|---|---|
| results | 49.33%, -0.67, P+ 0.443 | 54.41%, +4.41, P+ 0.857 | 53.09%, +3.09, P+ 0.802 |
| elo | 45.33%, -4.67, P+ 0.200 | 55.15%, +5.15, P+ 0.898 | 55.56%, +5.56, P+ 0.913 |
| player_injuries | 42.67%, -7.33, P+ 0.083 | 55.88%, +5.88, P+ 0.919 | 55.56%, +5.56, P+ 0.919 |
| F_served | 57.33%, +7.33, P+ 0.890 | 50.00%, +0.00, P+ 0.501 | 50.00%, +0.00, P+ 0.499 |

All three lean negative in 2011-2013 and positive in the two later eras. I think
the most likely cause is that 2011-2013 is where the walk-forward is at its
thinnest — the first scored week trains on 512 games and the era never exceeds
about 1,300 — so the per-family fits there are the noisiest in the study; that
is **inferred**, not measured, and the sign change is reported rather than
explained away.

### (2) Does specialising the model on big spreads help

Paired, on the games both arms scored, week-blocked:

| arm | surface | window | n | trained on all | specialist | picks changed | delta | 95% week | week P+ | season P+ |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| F_spec105 | opener | 2020-2025 | 130 | 50.77% | 53.85% | 48 | **+3.08** | [-7.14, +13.45] | **0.7188** | 0.6832 |
| F_spec105 | close proxy | 2011-2025 | 289 | 50.87% | 50.17% | 106 | -0.69 | [-6.57, +5.12] | 0.4024 | 0.3809 |
| F_spec105 | close proxy | 2014-2019 | 127 | 51.97% | 44.09% | 46 | -7.87 | [-15.45, +0.00] | 0.0221 | 0.0000 |
| F_spec105 | close proxy | 2020-2025 | 162 | 50.00% | 54.94% | 60 | **+4.94** | [-3.55, +13.29] | **0.8752** | 0.9303 |
| F_spec75 | opener | 2020-2025 | 130 | 50.77% | 46.15% | 44 | -4.62 | [-14.62, +5.60] | 0.1873 | 0.2468 |
| F_spec75 | close proxy | 2011-2025 | 369 | 51.76% | 51.49% | 133 | -0.27 | [-6.02, +5.54] | 0.4638 | 0.4562 |
| F_spec75 | close proxy | 2011-2013 | 71 | 59.15% | 69.01% | 19 | **+9.86** | [-1.47, +20.25] | **0.9603** | 0.9039 |
| F_spec75 | close proxy | 2014-2019 | 136 | 50.00% | 47.06% | 58 | -2.94 | [-13.28, +7.63] | 0.2916 | 0.2800 |
| F_spec75 | close proxy | 2020-2025 | 162 | 50.00% | 47.53% | 56 | -2.47 | [-10.60, +5.68] | 0.2730 | 0.1644 |

Specialising is **unresolved in both directions and it changes a third of the
picks** (48 of 130 at the opener, 106 of 289 on the close proxy). It leans
positive where the data is thickest and most recent — +3.08 at the opener
(P+ 0.719) and +4.94 on the 2020-2025 close proxy (P+ 0.875) — and negative in
2014-2019 (-7.87, P+ 0.022). The one thing it clearly does is damage the
probability: `F_spec105`'s Brier at 10.5+ is **0.27420** (opener) and **0.27867**
(close proxy) against the all-games model's 0.26294 and 0.25170, because its
out-of-time residual sample is a few hundred rows instead of a few thousand. A
specialist is not free; it buys a possible side gain with a measurably worse
number on the card.

### (3) Where the margin actually lands at 10.5+

2009-2025 regular season, **443** completed games with `|spread_line| >= 10.5`.

The favourite's final margin is emphatically lumpy: **30.25%** of all big-spread
games land on one of the ten key numbers (+/-3, 7, 10, 14, 17), and the top of
the histogram is exactly where football says it should be —

| favourite's final margin | games | share |
|---|---:|---:|
| +3 | 30 | 6.77% |
| +7 | 26 | 5.87% |
| +14 | 26 | 5.87% |
| +17 | 17 | 3.84% |
| -3 | 14 | 3.16% |
| +10 | 13 | 2.93% |
| -7 | 3 | 0.68% |
| -10 | 3 | 0.68% |

Pushes at 10.5+ are rare because most big lines are half-points: **1.58%** of all
443 games push, but only **206** of them carry an integer line and among those the
push rate is **3.40%**. At the two lines the brief names:

| line | games | pushes | push rate |
|---|---:|---:|---:|
| exactly 11 | 47 | 1 | 2.13% |
| exactly 13 | 45 | 0 | 0.00% |
| **exactly 14** | **68** | **3** | **4.41%** |
| **exactly 17** | **13** | **1** | **7.69%** |

By era, at 10.5+: key-number share **33.33%** (2009-2013, n=141) / **36.03%**
(2014-2019, n=136) / **22.89%** (2020-2025, n=166); push rate 2.13% / 0.00% /
2.41%; underdog cover rate **53.62% / 44.85% / 48.15%**, which reproduces
`docs/spread_hole_diagnosis.md:189-192` on the current feature table.

**And the mapping still cannot rescue the bucket.** Re-reading each week's own
residual sample under the project's own `discrete_residual` method — which
rounds `centre + residual` to the integer lattice
(`src/nfl_ats/calibration.py:444-447`) — instead of `gaussian_median`:

| surface | scope | n | side flips | Brier gaussian_median | Brier discrete | accuracy gaussian_median | accuracy discrete |
|---|---|---:|---:|---:|---:|---:|---:|
| opener | 10.5+ | 130 | **2 (1.54%)** | 0.26294 | 0.26447 | 44.62% | 44.62% |
| opener | all lines | 1,503 | 61 (4.06%) | 0.25174 | 0.25301 | 53.96% | 53.36% |
| close proxy | 10.5+ | 373 | 17 (4.56%) | 0.25170 | 0.25333 | 51.21% | 51.47% |
| close proxy | all lines | 3,818 | 143 (3.75%) | 0.25649 | 0.25821 | 51.47% | 51.44% |

That is the number lane F's argument implied and did not state: at 10.5+ the
discrete read moves **2 picks out of 130** at the opener and leaves the bucket's
accuracy identical to four decimal places. The owner's binding premise about the
margin distribution is confirmed by the histogram above — it is not Gaussian and
not close — and that premise is about the push, flip-line and alternative-line
answers, not about which side a big spread is picked on. The side is decided by
the point.

### (4) The market's own behaviour at 10.5+, which is where the signal is

Following the opener-to-close line move — pick the side the line moved toward —
graded at the opener, zero-move games excluded:

| scope | window | n | zero-move excluded | accuracy | pts vs coin flip | 95% week | week P+ |
|---|---|---:|---:|---:|---:|---|---:|
| all lines | 2020-2025 | 1,133 | 377 | 55.08% | +5.08 | [+2.26, +7.89] | 0.9999 |
| **10.5+** | **2020-2025** | **108** | 22 | **62.96%** | **+12.96** | [+3.70, +22.17] | **0.9967** |
| all lines | 2023-2025 | 587 | 216 | 55.71% | +5.71 | [+1.83, +9.65] | 0.9982 |
| **10.5+** | **2023-2025** | **50** | 8 | **70.00%** | **+20.00** | [+6.86, +32.93] | **0.9985** |

Per season at 10.5+: 52.63% (2020, n=19), 62.50% (2021, n=24), 53.33% (2022,
n=15), 58.82% (2023, n=17), 72.73% (2024, n=11), 77.27% (2025, n=22) — above the
coin flip in all six, mean absolute move 1.33 points.

**The sharpest number in this lane is the interaction.** On the 108 big-spread
games that moved, the served model and the market's move agree on **70.37%** of
them, and:

| | n | served model accuracy at 10.5+ |
|---|---:|---:|
| model agrees with the line's move | 76 | **59.21%** |
| model disagrees with the line's move | 32 | **28.13%** |

The whole 10.5+ hole lives in 32 games. Where the model is on the same side as
the market's own repricing it is 59.2% right at 10.5+ — better than its
all-lines average. Where it fights the repricing it is 28.1% right.

Two honest caveats. First, the closing line is what the repo's own harness calls
an **oracle** (`src/nfl_ats/clv.py:2262-2263`): it is fully known only at
kickoff, so this is an upper bound on what a Tuesday-to-Sunday follow channel can
reach. It is not a fantasy bound — the pick deadline is min(kickoff, Sunday 16:00
ET) while the pool's line freezes Tuesday, and a late-week follow rule is already
served (`docs/follow_threshold_live_card.md`) — but the served rule reads a
late-week snapshot, not the close, and it is not tuned by line size. Second, the
2020-2025 opener window is reused from lanes F, M and N, which is the stated
discount declared above.

**Heavy handle**: the action-network archive carries a money share for 2023-2025
only, and after matching to the schedule it holds **148** games, of which **8**
are at 10.5+. That cell reads 62.50%, +12.50 accuracy points, 95% week-blocked
[-16.67, +37.50], `probability_positive` 0.7870; restricted to a >=70% money
share, 7 games at 57.14%, [-30.00, +35.71], 0.6770. The all-lines cell is 146
games at 51.37%, [-8.33, +8.43], 0.6133. Eight games cannot answer anything; the
cell is recorded and left open.

### (5) A defect this lane found on the way past

The served probability rule and the plain sign of the same residual disagree at
10.5+, and the sign is better there. On the identical 130 opener games:

| rule | all lines (1,503) | 10.5+ (130) |
|---|---:|---:|
| sign of the residual | 52.96% | **50.77%** |
| served probability rule, raw | 53.96% | **44.62%** |
| served probability rule, S3-served | 54.56% | 47.69% |

(The 47.69% is this lane's count on the 130 non-push games with
`|opener| >= 10.5`. `docs/spread_hole_diagnosis.md:28` reports 47.33% for the
same quantity on 131 games; the extra game is **2021_06_HOU_IND**, whose opener
is **10.25** — a bucketing that tests `x <= 10` for the 7.5-10 bucket drops a
10.25-point line out of it and into 10.5+. Measured this session on the current
archive: 133 rows and 130 non-push at `>= 10.5`, versus 134 and 131 under that
bucketing. It is one game and it moves the cell by 0.36 points.)

`gaussian_median` puts `P(home cover) > 0.5` exactly when
`residual + median(residuals) > 0` (read: `src/nfl_ats/calibration.py:291-292`),
so the
served rule shifts every game's decision boundary by one global constant fitted
on all lines. That constant is worth **+1.00 accuracy points overall and -6.15 at
10.5+**. It is a single number applied identically to a 1-point line and a
17-point line, and at 10.5+ it is the difference between a coin flip and a
6-point loss. This is a modelling defect to fix, not a threshold to flip, and it
is reported here as a diagnosis.

### Decision

**Which inputs carry signal at 10.5+:** `results` and `elo` do, with
`probability_positive` 0.908 / 0.939 at the opener and 0.876 / 0.901 on the
close proxy, and — the part that is not selection-inflated — split-half
reliability resolved above zero on both surfaces (`results` +0.625 / +0.380,
`elo` +0.505 / +0.219). `player_injuries` scores highest of all at the opener
(58.46%, P+ 0.976) on a signal with no measurable stability (r +0.006), so it is
recorded and left open rather than believed. Every other block, and the assembled
90-column model itself, is unresolved. And the market's opener-to-close move
carries far more than any of them: **62.96%** over 108 games, P+ 0.9967.

**What the served model should do at 10.5+, in one sentence:** stop letting its
own assembled residual settle the side there and let the market's own late-week
repricing settle it instead — because the residual as assembled is 50.77% while
two of its own input blocks are 55-56% and the line's own move is 62.96%, and
because the model's entire big-spread deficit sits in the 32 games out of 108
where it takes the opposite side to that move (28.13% right) rather than the same
side (59.21% right).

**The exact arm a follow-up lane would predeclare.** One primary, one secondary,
both model-level and continuous in the line, neither a bucket rule and neither
run here:

- **AK-1 (primary), line-size shrinkage of the served residual.** The served
  point becomes `line + k(|line|) * residual`, where `k` is a monotone
  non-increasing function of `|line|` fitted **walk-forward on training rows
  only** each week (isotonic regression of the realised `ats_margin` on the
  fitted residual within line-size bins of the training set, no outcome from the
  scored week). No threshold, no bucket, no flip: `k` is continuous, applies at
  every line, and the mechanism it encodes is the one measured here — the
  assembled residual's information about the cover falls with line size while the
  blocks that feed it do not. Predicted from this lane, and stated as a
  prediction so it can be scored: `k` near 1 below 7, and near 0 above 10.5.
- **AK-2 (secondary), line-size exposure for the late-week follow.** Make the
  served follow rule's threshold a fitted function of `|line|` rather than the
  single constant it is today, because the move is worth +5.08 accuracy points
  across all lines and +12.96 at 10.5+. Grade it the way
  `docs/follow_threshold_live_card.md` grades the incumbent — through the played
  card, at the opener.

**What either arm changes on 2026 Week 1: nothing that is at 10.5+, because no
Week 1 game is.** Measured this session by fitting the served `weak_stack` ridge
on the 4,431 completed regular-season games before the 2026-09-09 cutoff and
predicting all 16 Week 1 games: the largest line on the card is **ARI at LAC at
9.5** and the next is **CLE at JAX at 8.5**, both inside 7.5-10 and neither in
this lane's evidence domain. Under AK-1's shrinkage the two of them behave very
differently, and the reason is this week's residual median of **+0.3691**:

| game | line | residual | served P(home) | served pick | pick under k=0.75 | k=0.5 | k=0.25 | k=0 |
|---|---:|---:|---:|---|---|---|---|---|
| ARI at LAC | 9.5 | -5.7554 | 0.3351 | ARI +9.5 | ARI | ARI | ARI | LAC |
| CLE at JAX | 8.5 | -0.1171 | 0.5079 | JAX -8.5 | JAX | JAX | JAX | JAX |

ARI at LAC survives any shrinkage down to `k = 0.064` — its residual is 5.8
points, the largest on the card, so the pick is not fragile. **CLE at JAX is the
opposite and it is the one to watch:** its residual is **-0.117 points**, i.e.
the model's own point says CLE by a hair, and the served pick is JAX only because
the +0.3691 location constant carries it over the line. Shrinking the residual
makes that pick *more* JAX, not less. Across the whole 16-game card a shrinkage
would change 0 picks at `k = 1.0` (the identity check passes), 1 pick at `k =
0.75` and `k = 0.5` (NE at SEA, a 3.5-point line, already played), 2 at
`k = 0.25` and 8 at `k = 0`.

**Nothing in this lane changes any served behaviour.** No ledger, manifest,
forecast, board or active-model file was touched, and no arm here is proposed for
the card. The owner's ban on unexplained threshold flips is why AK-1 is written
as a continuous fitted shrinkage rather than a rule that fires above 10.5.

### Registry

All **89** cells are recorded under family `mod18_big_spread_signal_v1`, named
`big_spread_signal_<family|arm>_<surface>_<window>`, with each family arm's
split-half reliability in the `--reliability` field. The exact argv lists are in
`record_commands.json` in the artifact directory and were run one at a time
against the main checkout's registry (`registry/weak_signals.json`), which now
holds **4,924** signals of which **89** carry the `big_spread_signal_` prefix.
(The registry gained more than 89 rows while this lane ran, because other
sessions write to the same file; the 89 figure is a direct count of this lane's
prefix, not a difference of two totals.) **Two** are terminal under the rule
frozen above, and both name
`wrong_sign_resolved` because both their week-blocked and their season-blocked
intervals sit entirely below zero:

- `big_spread_signal_offense_close_proxy_2011_2013` — -10.00 accuracy points,
  week [-19.12, -1.25], season [-15.52, -2.17], n=75.
- `big_spread_signal_player_continuity_close_proxy_2011_2025` — -5.23 accuracy
  points, week [-10.10, -0.37], season [-9.56, -0.75], n=373.

**A caveat that belongs next to the first of those, stated rather than buried:**
the `offense` block reads **+9.56** accuracy points in 2014-2019 with a
week-blocked interval of [+1.13, +17.81] that sits entirely *above* zero. So the
terminal cell closes the 2011-2013 reading of that block at 10.5+, not the block
itself. That is what the predeclared per-cell rule does when a family's sign
varies by era, and it is reported as a bound on one 75-game era cell, never as
"the offense inputs are refuted".

The other **87** cells are `unresolved_below_power` and report
`probability_positive`, including every favourable cell —
`big_spread_signal_player_injuries_opener_2020_2025` at +8.46 [0.00, +16.42],
P+ 0.9764; the two follow-move cells at P+ 0.9967 and 0.9985; and
`big_spread_signal_f_spec75_close_proxy_2011_2013` at +19.01 with a week-blocked
interval of [+6.72, +30.26] that sits **entirely above** zero at P+ 0.9977. A
favourable interval is not a closing ground either, so none of them is closed.

## Commands run

```
python scripts/big_spread_signal.py \
  --features data/processed/game_features_weak_stack.parquet \
  --market-root data/market/raw \
  --archive artifacts/opener_evaluation/20260910T005854Z \
  --public-betting data/raw/public_betting/20260820T111148Z/actionnetwork/index.parquet \
  --out artifacts/big_spread_signal/20260910T011453Z --min-train-games 500

nfl-ats weak-signals record ...   (89 cells, mod18_big_spread_signal_v1;
the exact argv lists are saved as record_commands.json in the artifact
directory and were run one at a time)
```
