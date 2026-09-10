# Let only the blocks that carry signal choose the side on big spreads (MOD-18 lane BE)

Acts on the diagnosis in `docs/big_spread_signal.md` and on the two lanes that
followed it (`docs/line_size_shrinkage.md`, `docs/spread_hole_target.md`,
`docs/home_push_on_served_card.md`). Those lanes shrank the whole residual
toward zero, deleted the line column, and re-tuned the home push. None of them
did the thing the diagnosis most directly implies.

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
week-blocked with the season-blocked reading alongside. Football margins are
discrete and multimodal, never Gaussian. Decide on expected value; composition
is not the signal.

## Design, frozen 20260910T041850Z before any candidate was scored

Family `mod18_big_spread_reduced_model_v1`.

### The defect being acted on, all read, none re-derived here

1. At `|line| >= 10.5` the assembled 90-column ridge is **50.77%** right at the
   Tuesday opener while its own `results` block alone is **55.38%** and `elo`
   alone **56.15%** (read: `docs/big_spread_signal.md:262-269`). Those are the
   only two blocks whose **split-half reliability interval sits entirely above
   zero on both line surfaces** — `results` +0.625 opener / +0.380 close proxy,
   `elo` +0.505 / +0.219 (read: `docs/big_spread_signal.md:262-264`,
   `:283-285`). `player_injuries` scores higher still at the opener (58.46%) on
   a signal with no measurable stability (r +0.006), so it is not in the
   declared set.
2. Shrinking the WHOLE residual toward zero at big lines (AK-1) improves Brier
   and log loss and **loses accuracy through the played card** — best arm
   -0.333 accuracy points at `probability_positive` 0.3214 (read:
   `docs/line_size_shrinkage.md:311-314`). Its own guard says why: as `k` falls
   the residual stops deciding and the location constant plus the S3 offset
   decide instead (read: `docs/line_size_shrinkage.md:338-346`).
3. Deleting the line column (T1) removes the favourite-ward lean and still
   loses through the card, -0.665 at P+ 0.0709 (read:
   `docs/spread_hole_target.md:281`).
4. The served home push is worth **+4.58** accuracy points at 10.5+ on the
   played card and captures **43%** of a **+5.344**-point perfect-foresight
   ceiling on its own flip set (read: `docs/home_push_on_served_card.md:220`,
   `:246-249`).

**The mechanism this lane encodes.** At big lines the assembled residual is
worse than two of its own inputs, so the problem is not the residual's SIZE
(AK-1's axis) and not the line column (T1's axis) — it is the 82 other columns
diluting the two blocks that carry stable signal. The intervention is
therefore: at big lines, **let only those blocks choose the side**.

**This is not a threshold flip and not a bucket rule in the banned sense.** The
owner's ban names "a rule that flips the model's pick" whose only justification
is "an accuracy dip located at a spread threshold". Every arm here is a
**fitted, walk-forward re-estimation of the model itself** on a declared column
subset, applied on a line region, with the mechanism named in advance and
measured (the reliability column of `docs/big_spread_signal.md`). It names its
mechanism, it changes the model rather than post-processing its output, and its
coefficients are reported per season so the mechanism is visible. It is
nonetheless a discontinuity in the line, and that is stated as the honest cost
of the design rather than hidden: R3 exists precisely to measure how the answer
moves when the region moves.

### The served decision statistic, stated exactly

For one game with opener line `L_signed`, that week's fitted ridge residual `r`
from model `M`, that week's S3 home-side offset `o` for the game's bucket
(`nfl_ats.home_side_location.fit_home_side_offsets`, fitted on the arm's OWN
prior out-of-time raw stream `L_signed + r`, zeroed outside the buckets `7`,
`7.5-10`, `10.5+`), and `M`'s own out-of-time residual sample with median `m`
and standard deviation `s` (ddof=1):

```
P(home cover) = Phi( (r + o + m) / s )
```

which is `gaussian_median` written out (read:
`src/nfl_ats/calibration.py:291-292`). The forced pick is home iff that
probability is >= 0.5. Every arm below differs ONLY in which model `M` supplies
`r`, `m` and `s` for a given game; the offset fitter, the mapping and the
threshold are the served ones, unchanged.

### Arms

**R0 — served.** The 90-column `weak_stack` market-residual ridge, alpha 10,
`gaussian_median`, S3 offset from its own stream. Replayed exactly by this
lane's own walk-forward; the identity check against
`artifacts/opener_evaluation/20260910T005854Z/per_game.parquet` is reported
before any arm is read and the target is a maximum residual gap of 0.0.

**R1 (primary) — the reduced model at 10.5+.** For every game with
`abs(opener line) >= 10.5`, `r`, `m` and `s` come from a walk-forward ridge on
the `results` + `elo` families ONLY — the **eight** columns

```
results:  home_point_diff, away_point_diff, diff_point_diff,
          home_ats_residual, away_ats_residual, diff_ats_residual
elo:      elo_diff, elo_home_win_prob
```

(`nfl_ats.margin.FEATURE_FAMILIES`, recomputed by the script, not retyped).
Same weekly windows as the served model (all completed regular-season games
strictly before that week's first kickoff, minimum 500 training rows), same
`ridge_alpha=10.0`, same `target="market_residual"`, same
`distribution_fraction`, so `m` and `s` are the REDUCED model's own
out-of-time residual sample. **The reduced model is trained on all games**, not
only big ones — the reduction is in columns, not in rows, because
`docs/big_spread_signal.md`'s `F_spec105` already measured row restriction and
it costs Brier (read: `docs/big_spread_signal.md:362-366`). Every game below
10.5 keeps the served model's `r`, `m`, `s` exactly. The S3 offset is refitted
from R1's OWN raw stream, so R1's offsets differ from R0's at 10.5+ by
construction, exactly as every candidate arm in
`scripts/spread_hole_arms.py` refits its own.

**R2 (secondary) — the family set chosen walk-forward.** R1 with the reduced
model's column set chosen each week from the prior weeks' single-family sign
accuracy at 10.5+, rather than declared in advance. Each week, one runtime
profile per declared family (12 of them) is fitted on the same training rows
and scored on that week's games; the accumulated prior stream then gives each
family a shrunken sign accuracy at 10.5+

```
a_f = (correct_f + 0.5 * 20) / (n_f + 20)
```

on non-push 10.5+ games only, 20 pseudo-observations toward the coin flip. The
**top two** families by `a_f` supply the reduced model's columns for that week.
Ties are broken by larger `n_f`, then alphabetically by family name — declared,
not tuned. **Warm-up, declared:** until the accumulated prior stream holds at
least 20 non-push games at `|line| >= 10.5`, R2 uses R1's family set
(`results` + `elo`), because with no prior 10.5+ games every family's `a_f` is
exactly 0.5 and the ranking would be alphabetical noise. The chosen set is
written out per week so the selection is auditable.

**R3 (mechanism check) — R1 applied from 7.5+.** Identical to R1 with the
region `abs(opener line) >= 7.5` instead of `>= 10.5`. The 7.5-10 bucket also
loses on the served card (51.55% before the home push, read:
`docs/spread_hole_target.md:281-285`) but the market's underdog tilt there is
different in kind — 53.92% dog cover over 2009-2025 at 7.5-10 against 48.85%
at 10.5+ (read: `docs/spread_hole_diagnosis.md:189-192` as quoted in
`docs/spread_hole_target.md:163` and `docs/big_spread_signal.md:34-35`). R3 is
declared as a mechanism check on region sensitivity, **not** as a search over
thresholds: exactly one alternative region is scored, it is named here before
any number is read, and no third region will be added afterwards.

No other arm, no other family set, no other region, and no re-tuning after
signs are seen.

### Population and grade

`artifacts/opener_evaluation/20260910T005854Z` (active model
`2e8c616b476dd0d2`, feature table
`fadeed326dc277416519e75914154c0d7b822b5ef28ba6d19efaa6469cb64e21`,
weak_stack market-residual ridge alpha 10, `gaussian_median`), **1,537
archived games 2020-2025, 1,503 non-push at the opener**. Graded at the
**Tuesday opener** by the production probability rule
(`home_cover_probability >= 0.5`), which is the pool's own grade.

Two surfaces, both reported:

- **standalone** — the arm's own opener pick, no card.
- **card** — the served nine-member OR union
  (`nfl_ats.unserved_tilt_marginals.served_card_flip_set`, i.e.
  `scripts/spread_hole_arms.py --card served` semantics), **recomputed on the
  candidate**, never inherited: every member conditions on the model's own
  probability.

### What gets reported

Paired accuracy-point deltas (candidate minus R0) on the games both arms score,
**week-blocked and season-blocked** bootstraps, 20,000 samples, seed 20260821,
via `nfl_ats.overlay_composition.blocked_bootstrap_matrix` — the project's own
resampler. Reported **overall** and by bucket `0-6.5`, `7`, `7.5-10`, `10.5+`
(`nfl_ats.spread_regime.spread_bucket`, coarsened as
`scripts/spread_hole_arms.py` coarsens it). Alongside every cell:
**favourite-pick share**, **home-pick share**, **Brier**, **log loss**, and
**picks changed**. Plus the **per-season fitted coefficients of the reduced
model** (standardised ridge coefficients on the eight declared columns, mean
over that season's weekly refits) so the mechanism is visible rather than
asserted.

### Positive control

Two, both named before scoring:

1. **Perfect foresight at 10.5+** — on each arm's own changed-pick set at
   10.5+, take the winning side. This is the ceiling any repair confined to
   that arm's flips could reach. The comparable published number is
   **+5.344** accuracy points at 10.5+ on the home push's flip set, week
   [+1.709, +9.375], `probability_positive` 0.9997 (read:
   `docs/home_push_on_served_card.md:246-249`), so the instrument demonstrably
   resolves an effect of that size on a flip set in this bucket.
2. **The market-move oracle** — following the opener-to-close move at 10.5+ is
   **62.96%** on the 108 big-spread games that moved, +12.96 accuracy points,
   week [+3.70, +22.17], `probability_positive` 0.9967 (read:
   `docs/big_spread_signal.md:431`, reproduced in
   `docs/line_size_shrinkage.md:378`). Recomputed in this lane's own replay and
   reported.

### Terminal classification, declared before any cell was scored

A cell is `unresolved_below_power` unless exactly one of:

- **`wrong_sign_resolved`** — BOTH the week-blocked and the season-blocked 95%
  interval sit entirely below zero.
- **`positive_control_bound`** — a 10.5+ cell whose week-blocked AND
  season-blocked 95% intervals both lie entirely **below +5.3** accuracy points
  (control 1's resolved magnitude) **and** whose point estimate is at most
  zero. The second condition is a deliberate restriction in the keep-open
  direction: a cell that leans positive is not closed on a magnitude bound,
  because bounding a size is not evidence that a small effect is absent.

Nothing else closes a cell. An interval containing zero closes nothing, and a
favourable interval is not a closing ground either. If a record command errors,
the verdict is wrong, not the validator, and the cell is reclassified
`unresolved_below_power`.

### Reuse discount, stated up front

The 2020-2025 opener window has been used by lanes F, M, N, AK, AO and AW on
adjacent questions, and the reliability column that selects `results` + `elo`
was measured on the opener surface of that same window. That is a stated
discount, not a ban (AGENTS.md: windows retire per-family, and a reused window
carries a stated discount). The close-proxy corroboration of the same two
blocks (2011-2025, read: `docs/big_spread_signal.md:283-285`) is the
independent part.

### Recording

Every cell is recorded with `nfl-ats weak-signals record`, units
`accuracy_points`, league `nfl`, family `mod18_big_spread_reduced_model_v1`,
named `big_spread_reduced_model_<arm>_<card|standalone>_<bucket>_2020_2025`,
one command at a time with
`NFL_ATS_REGISTRY_DIR=F:\Repos\nfl_py3\registry`, argv saved to
`record_commands.json` in the artifact directory.

### What this lane may and may not do

It may name the exact promotion path and the Week 1 picks that would change. It
may **not** promote, publish, or touch any ledger, manifest, forecast, board or
active-model file. Week 1 pick effects are written read-only to the scratchpad.

## Results

Every number below is **measured this session** by the commands at the bottom;
the tables live in `artifacts/big_spread_reduced_model/20260910T041850Z/`.
Nothing above this line was edited after scoring.

### The replay is exact

The R0 arm is the served archive, not an approximation of it. Joined game for
game against `artifacts/opener_evaluation/20260910T005854Z/per_game.parquet` on
all **1,537** archived games: maximum line gap **0.0**, maximum residual gap
**0.0**, maximum home-side-offset gap **0.0**, maximum probability gap
**1.1e-16**, pick agreement **1.000**, and the R0 replay's own accuracy is
**54.5576%**, identical to the archive's `correct_at_open_probability_rule`
(measured: `arm_results.json` `identity`). Every delta below is the reduced
model and nothing else. The lane scores **107** weeks, 2020-2025.

### The mechanism, measured before the arms are read

Split-half reliability of each model's own signal — inside every walk-forward
week the training rows are split by a seeded permutation into two disjoint
halves, the model is fitted separately on each, both are scored on that week's
games, and the two prediction streams are correlated over the pooled 1,537
games (measured: `split_half_reliability.csv`, permutation seed 20260910,
week-blocked bootstrap seed 20260821, 2,000 draws):

| model | scope | n | split-half r | Spearman-Brown | 95% week | P(r > 0) |
|---|---|---:|---:|---:|---|---:|
| **R0, all 90 columns** | all lines | 1,537 | **-0.020** | -0.040 | [-0.079, +0.041] | 0.2545 |
| **R0, all 90 columns** | 10.5+ | 133 | **-0.138** | -0.320 | [-0.276, +0.028] | 0.0505 |
| **R1, `results` + `elo`** | all lines | 1,537 | **+0.504** | +0.671 | [+0.461, +0.549] | 1.0000 |
| **R1, `results` + `elo`** | 10.5+ | 133 | **+0.590** | +0.742 | [+0.452, +0.706] | 1.0000 |
| R2, weekly-chosen pair | all lines | 1,537 | +0.143 | +0.251 | [+0.058, +0.228] | 0.9995 |
| R2, weekly-chosen pair | 10.5+ | 133 | +0.207 | +0.343 | [+0.021, +0.387] | 0.9820 |

**The assembled model's own signal does not reproduce when its training set is
halved — anywhere, not only on big spreads.** The eight-column reduced model's
does, with an interval entirely above zero on both scopes. That is the least
selection-inflated number in the lane: it is a property of the model, measured
without reference to any outcome's sign, and it reproduces
`docs/big_spread_signal.md:269`'s -0.077 for the served model at 10.5+ on an
independent rebuild. Neither R0 cell is closed on it — the rule requires the
interval to sit entirely inside [-0.05, +0.05] and neither does.

### The reduced model's coefficients, per season

Mean standardised ridge coefficient over that season's weekly refits
(measured: `reduced_model_coefficients_by_season.csv`; the six missing-value
indicators are identical to four decimals within a season and are omitted):

| season | home_point_diff | away_point_diff | diff_point_diff | home_ats_residual | away_ats_residual | diff_ats_residual | elo_diff | elo_home_win_prob |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020 | +1.0766 | -0.1980 | +0.9257 | -0.6227 | +0.4629 | -0.7677 | +0.5101 | -0.7467 |
| 2021 | +1.0508 | -0.2875 | +0.9715 | -0.6091 | +0.5490 | -0.8195 | +0.0850 | -0.3251 |
| 2022 | +0.8748 | -0.3262 | +0.8706 | -0.4851 | +0.5731 | -0.7485 | +0.5791 | -0.8209 |
| 2023 | +0.8796 | -0.2894 | +0.8470 | -0.4760 | +0.5482 | -0.7263 | +0.3243 | -0.6037 |
| 2024 | +0.8226 | -0.2831 | +0.8027 | -0.4184 | +0.5150 | -0.6624 | +0.1095 | -0.3737 |
| 2025 | +0.7914 | -0.3093 | +0.7975 | -0.4031 | +0.5281 | -0.6596 | -0.0500 | -0.1433 |

The shape is stable across six seasons and it is football-legible: the home
team's own scoring margin and the home-minus-away margin gap push the point
toward the home side, and the ATS-residual block enters with the opposite sign
— a team that has been beating the number recently is discounted, which is the
mean-reversion the market has already priced. The `elo` pair shrinks toward
zero over the window (elo_diff +0.51 in 2020 to -0.05 in 2025), i.e. the
power-rating block is being crowded out by the form block as the training set
grows. That is the mechanism visible, not asserted.

### R2's family selection, walk-forward

Which two blocks the weekly ranking picked (measured:
`r2_family_selection.csv`): 2020 all 17 weeks at the declared warm-up
(`results` + `elo`, because fewer than 20 prior big-spread games exist); 2021
splits across `player_injuries` + `player_qb` (10), `elo` + `player_injuries`
(4), `context` + `player_injuries` (2) and three singletons; 2022 `elo` +
`player_qb` (10) and `elo` + `player_injuries` (8); 2023 `elo` + `market` (10),
`elo` + `player_qb` (5), `elo` + `player_injuries` (3); **2024 and 2025 are
`elo` + `player_injuries` in all 36 weeks.** `elo` is in the chosen pair in 94
of the 107 scored weeks.

### The decision cell: through the played card, at the opener

Paired against the served card on the same **1,503** non-push games,
week-blocked and season-blocked bootstraps, 20,000 samples, seed 20260821. The
served card scores **56.89%** (measured: `arm_results.csv`).

| arm | card accuracy | picks changed | delta | 95% week | week P+ | 95% season | season P+ |
|---|---:|---:|---:|---|---:|---|---:|
| **R2** (weekly-chosen pair at 10.5+) | **57.02%** | 14 | **+0.133** | [-0.336, +0.609] | **0.7020** | [-0.190, +0.616] | 0.6947 |
| R1 (`results` + `elo` at 10.5+) | 56.69% | 15 | -0.200 | [-0.723, +0.329] | 0.2185 | [-0.535, +0.138] | 0.1419 |
| R3 (`results` + `elo` at 7.5+) | 56.15% | 35 | -0.732 | [-1.593, +0.131] | 0.0442 | [-1.387, -0.135] | 0.0057 |

Standalone, against the served model's own **54.56%**:

| arm | accuracy | picks changed | delta | 95% week | week P+ | season P+ |
|---|---:|---:|---:|---|---:|---:|
| **R2** | **55.02%** | 35 | **+0.466** | [-0.264, +1.188] | **0.8992** | 0.8962 |
| R1 | 54.82% | 38 | +0.266 | [-0.533, +1.020] | 0.7501 | 0.7087 |
| R3 | 53.96% | 87 | -0.599 | [-1.806, +0.597] | 0.1656 | 0.2859 |

### By bucket, where the lane was aimed

R1 and R2 are bit-identical to the served model below 10.5 by construction, so
those buckets are structural identities and are not recorded as measurements.
At 10.5+ (n=131, served 47.33% standalone / 51.15% on the card):

| arm | surface | accuracy | picks changed | delta | 95% week | week P+ | season P+ |
|---|---|---:|---:|---:|---|---:|---:|
| R2 | standalone | **52.67%** | 35 | **+5.344** | [-2.963, +13.287] | **0.9012** | 0.8962 |
| R1 / R3 | standalone | 50.38% | 38 | +3.053 | [-6.107, +12.030] | 0.7484 | 0.7087 |
| R2 | card | **52.67%** | 14 | **+1.527** | [-4.032, +7.031] | **0.7036** | 0.6947 |
| R1 / R3 | card | 48.85% | 15 | -2.290 | [-8.264, +3.497] | 0.2193 | 0.1419 |

R3's own region, 7.5-10 (n=194, served 51.55% standalone / 52.58% card):
standalone 44.85%, -6.701, [-13.990, +0.518], P+ 0.0347 (season 0.1186); card
48.45%, -4.124, [-8.867, +0.516], P+ 0.0401 (season 0.0288). **The region
matters and it matters in the declared direction**: the same eight columns that
lean positive at 10.5+ lean negative at 7.5-10, which is what the different
market tilt in that bucket predicted. That is the mechanism check answering,
not a threshold being searched.

### The guard that belongs next to the 10.5+ numbers

Home teams covered **58.02%** of the 131 scored 10.5+ games in this window
(measured: `arm_results.csv` `home_cover_rate`). The reduced model's home-pick
share at 10.5+ rises from the served **58.78%** to **67.94%** (R1) and
**68.70%** (R2) standalone, and its favourite-pick share from **60.31%** to
**78.63%** and **73.28%**. So R2's 52.67% at 10.5+ is reached while picking
home more often than the served model does, in a window whose home teams
covered 58.02% — simply picking home every time would have scored 58.02%. Part
of the 10.5+ gain is a partial recovery of a home tilt this window carries, and
the cell is recorded, open, and not over-read.

### What the reduced model unambiguously buys: the probability

Brier and log loss, standalone (measured: `arm_results.csv`):

| cell | served Brier | R1 | R2 | R3 |
|---|---:|---:|---:|---:|
| all lines (1,503) | 0.25158 | 0.25067 | 0.25084 | **0.25010** |
| 10.5+ (131) | 0.26130 | **0.25083** | 0.25280 | 0.25062 |
| 7.5-10 (194) | 0.26030 | identical | identical | 0.25607 |

Log loss moves the same way (all lines 0.69662 to 0.69475 for R1; 10.5+ 0.71625
to 0.69482). Every arm's stated confidence at 10.5+ gets better even where its
side gets worse — the same split the shrinkage lane found
(`docs/line_size_shrinkage.md:348-368`), reached by a different route.

### The positive controls

Both resolved on this replay (measured: `controls.csv`):

| control | scope | n | accuracy | served on the same games | effect | 95% week | week P+ |
|---|---|---:|---:|---:|---:|---|---:|
| market's opener-to-close move | all lines | 1,133 | 55.08% | 54.63% | +5.075 | [+2.265, +7.886] | 0.9999 |
| **market's opener-to-close move** | **10.5+** | **109** | **63.30%** | **49.54%** | **+13.303** | [+4.285, +22.381] | **0.9978** |
| market's opener-to-close move | 7.5-10 | 157 | 53.50% | 50.32% | +3.503 | [-4.118, +11.074] | 0.8203 |
| perfect foresight on R2's card flips | 10.5+ | 131 | 57.25% | 51.15% | +6.107 | [+2.400, +10.294] | 0.9999 |
| perfect foresight on R1's card flips | 10.5+ | 131 | 55.73% | 51.15% | +4.580 | [+1.504, +8.397] | 0.9990 |
| perfect foresight on R2's card flips | all lines | 1,503 | 57.42% | 56.89% | +0.532 | [+0.199, +0.928] | 0.9998 |

So the instrument resolves +4.6 and +6.1 on 131-game flip sets in this bucket,
bracketing the predeclared +5.3, and +13.3 for the market's own repricing.
**R2 delivers +1.527 of its own +6.107 ceiling at 10.5+ through the card, 25%
of what was available on the games it touched** — the same capture fraction it
reaches overall (+0.133 of +0.532). The served home push, for comparison,
captures 43% of its own ceiling (read:
`docs/home_push_on_served_card.md:246-249`).

### Decision

**R2 has the higher expected opener accuracy through the played card:
57.02% against the served card's 56.89%, +0.133 accuracy points,
`probability_positive` 0.7020 week-blocked and 0.6947 season-blocked.** Under
"a promotion bar is not a decision bar", declining R2 is taking the 30/70 side
of that bet, not the cautious side. R1 is the wrong side at 0.2185 and R3 at
0.0442, so of the three declared arms only the walk-forward-chosen family set
is a play.

Three things that belong in the same breath as that number, before anything is
done with it:

1. **R2's edge lives entirely in 14 games out of 1,503.** Below 10.5 it is the
   served model bit-for-bit, and after the nine-member card is recomposed on it
   only 14 of the 35 picks it changes standalone are still changed. A
   +0.133-point card edge that rests on 14 games is a lead, not a finished
   candidate.
2. **The block R2 keeps choosing is the one with no measured stability.**
   `elo` + `player_injuries` is its pick in all 36 weeks of 2024-2025, and
   `player_injuries` is exactly the block `docs/big_spread_signal.md:262`
   found scoring best at 10.5+ (58.46%) on a signal whose split-half
   correlation is +0.006. R1 — the arm built on the two blocks whose
   reliability IS resolved — is the arm that loses through the card. That
   inversion is the honest tension in this lane's result and it is not
   resolved by anything measured here.
3. **The mechanism is confirmed and is worth more than the arms.** The
   assembled 90-column model's own signal has a split-half correlation of
   -0.020 over all lines and -0.138 at 10.5+, while an eight-column subset of
   its own inputs reaches +0.504 and +0.590. The served model is not just
   diluted on big spreads; its aggregate prediction stream does not reproduce
   under a training-set split anywhere. That is a modelling defect with a
   measured size, and it points at regularisation and column selection rather
   than at any line region.

**What promoting R2 would take** (stated so a follow-up lane does not have to
rediscover it; nothing here was done):

- `src/nfl_ats/margin.py` — a real named feature profile for the reduced column
  set (it exists here only as a runtime registration inside this lane's
  script), plus a region/profile hook in the fit path so one week can serve two
  fitted models and `MarginModel.predict` can take the reduced model's own
  residual median and scale for the rows in the region. The point, the fair
  spread, the market residual and every probability must move together, as
  `center_offset`'s docstring requires.
- A walk-forward family selector beside `nfl_ats.home_side_location`, which is
  the exact precedent: fitted on the prior out-of-time stream, served per week,
  with its own artifact so the chosen pair is auditable.
- `nfl_ats.outcomes.score_outcome_week` (the sole production weekly-forecast
  entry point), then `weekly-run --record-decisions`, then the opener
  evaluation, the overlay-composition re-selection (every card member
  conditions on the model's probability), and `publish-board` /
  `publish-predictions` — AGENTS.md requires every headline number to be
  recomputed against the active model before the site is regenerated.

**Which 2026 Week 1 picks would change: none under R1 or R2, one under R3.**
Measured this session by fitting all three models on the 4,431 completed
regular-season games before the 2026-09-09 cutoff and re-deriving every Week 1
probability — the R0 rebuild reproduces the published sidecar's 16
probabilities to **5.6e-17**, so this is the served card re-derived, not
approximated. **No Week 1 game is at 10.5+**; the largest line is ARI at LAC at
9.5 and the next CLE at JAX at 8.5, so R1 and R2 are the served card exactly.
Under R3, which reaches 7.5:

| game | line | served P(home) | served pick | R3 P(home) | R3 pick |
|---|---:|---:|---|---:|---|
| **ARI at LAC** | 9.5 | 0.3580 | ARI +9.5 | **0.5109** | **LAC -9.5** |
| CLE at JAX | 8.5 | 0.5327 | JAX -8.5 | 0.5590 | JAX -8.5 |

ARI at LAC is the fragile game under every intervention this program has run —
`docs/line_size_shrinkage.md:492-494` found the same game flipping under two of
its four arms — and R3 is the arm this lane's own numbers say not to play
(-0.732 through the card, P+ 0.0442). CLE at JAX moves further onto JAX, as the
shrinkage lane predicted.

**Nothing in this lane changes any served behaviour.** No ledger, manifest,
forecast, board or active-model file was touched, and no arm here is promoted.

### Registry

All **31** scored cells are recorded under family
`mod18_big_spread_reduced_model_v1`, named
`big_spread_reduced_model_<arm>_<card|standalone>_<bucket>_2020_2025` and
`big_spread_reduced_model_control_*`, run one at a time against
`registry/weak_signals.json` (now **5,410** signals, **31** carrying this lane's
prefix). The exact argv lists are `record_commands.json` and the outcomes are
`registry_records.json`; every command returned 0.

**Two** are terminal, both `bounded_by_control` on `positive_control_bound`,
and both are the same measurement under two arm labels — R1 and R3 are
identical at 10.5+:

- `big_spread_reduced_model_r1_card_10p5_plus_2020_2025` and
  `big_spread_reduced_model_r3_card_10p5_plus_2020_2025` — -2.290 accuracy
  points, week [-8.264, +3.497], season [-7.080, +1.409], n=131. Both intervals
  exclude the predeclared +5.3 magnitude, the point estimate is negative, and
  the control on this arm's own flip set resolves +4.580 at
  `probability_positive` 0.9990. **This bounds the SIZE of what
  `results` + `elo` does at 10.5+ through the played card. It does not say the
  effect is zero, and it closes neither the arm's standalone cell (+3.053,
  P+ 0.7484, open) nor the mechanism.**

The other **29** are `unresolved_below_power` and report
`probability_positive`, including every favourable cell — R2 through the card
at +0.133 (P+ 0.7020), R2 standalone at 10.5+ at +5.344 (P+ 0.9012), and all
six control cells whose intervals sit entirely ABOVE zero. A favourable
interval is not a closing ground either, so none of them is closed. **No cell
qualified for `wrong_sign_resolved`**: R3's card cells are the most negative in
the lane and their week-blocked intervals still contain zero.

**20** further cells were not recorded and are listed in
`record_skipped.json`: they are structural identities where the arm IS the
served model by construction (every bucket below the arm's region, and the
7.5-10 foresight controls for R1 and R2, which change no pick there), so a row
there would be a definition entered as a measurement.

## Commands run

```
python stage1_streams.py          # 107 opener weeks x 15 weekly ridge refits
python stage2_arms.py             # 4 arm replays, served card recomposition, bootstraps
python stage3_control_week1.py    # both positive controls, coefficients, 2026 Week 1 read
python stage3b_reliability.py     # split-half reliability of R0/R1/R2, 107 weeks x 6 fits
python stage4_record.py           # 31 weak-signals record commands, one at a time

nfl-ats weak-signals record ...   (31 cells, mod18_big_spread_reduced_model_v1;
the exact argv lists are saved as record_commands.json in the artifact directory
and were run one at a time with NFL_ATS_REGISTRY_DIR=F:\Repos\nfl_py3\registry)
```

All five scripts are copied into
`artifacts/big_spread_reduced_model/20260910T041850Z/` beside their outputs.

