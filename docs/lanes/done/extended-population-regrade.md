# Extended population regrade (2011-2025 fit population)

## Goal
Use the new 3,734-game 2011-2025 fit population
(`artifacts/extended_fit_population/20260923T205910Z/population.parquet`,
~2.5x the served 1,503-game 2020-2025 population where every added term was
below power) to re-test whether more games recovers power for three
already-underpowered added-term candidates. Research only: no `src/` edits,
no registry writes, no commits. This session records only the predeclaration,
the run, and exact `nfl-ats weak-signals record` commands for the root to
execute.

## Predeclaration (written before any fit ran)
- **Population**: the named parquet, columns `game_id, season, week,
  home_covered, model_logit (served discrete read), composition_flag_sum,
  market_move_toward_home, market_move_available (0 before 2020),
  opener_source`. 15 seasons, 2011-2025.
- **Base**: four-term ridge-logit LOSO by season, features
  `(model_logit, composition_flag_sum, market_move_toward_home,
  market_move_available)` — identical to served `FIT_FEATURES`
  (`src/nfl_ats/pick_probability_fit.py:46`).
- **Family**: `extended_population_regrade_2011_2025`, exactly 3 predeclared
  looks (one fitted term each vs base; look 3 is one joint look over several
  terms, not 7 separate looks):
  1. `week_gated_protection_flag_sum` — replace `composition_flag_sum` with
     `composition_flag_sum - flag_protection * (week > 4)`
     (`scripts/lead65_gated_flag_in_fit.py`'s `gated_flag_sum_v1`).
     Mechanism: LEAD-65 (2020-2025) read the PBP08 protection-mismatch
     flag's edge as concentrated in weeks 1-4 (prior-season carryover still
     unpriced) and a probable drag folded flat into the sum from week 5 on;
     this re-tests that read with 2.5x the games.
  2. `reddit_home_comment_ratio_elevated` — add as a 5th term
     (`scripts/pooled_signal_sixth_fit_new_family.py`,
     `data/processed/game_features_weak_stack_reddit.parquet`). Mechanism:
     elevated home-side comment-volume ratio proxies public/local attention
     not fully priced into the market-move term; prior 2020-2025-only test
     was unresolved_below_power, this re-tests with 2.5x the games.
  3. `composition_flags_separate` — replace the single summed
     `composition_flag_sum` term with the individual signed member flags,
     entered jointly (one look, several coefficients), letting the fit
     choose each member's own weight instead of forcing equal weight inside
     the sum. Mechanism: same as look 1's premise generalized — an
     equal-weight sum can hide member-specific sign/magnitude differences.
- **Point-in-time coverage gate (checked before fitting, terms without it are
  dropped and named)**: of the 9 composition members
  (`COMPOSITION_ORDER`), 2 are always zero (`flag_interim_hc`,
  `flag_precip`, `OWNER_HELD_MEMBERS`) leaving 7 counted members
  (`COUNTED_FLAG_COLUMNS`). `flag_cold_visitor` depends on
  `data/raw/forecast_archive/full_2020_2025/forecasts.parquet` — **2020-2025
  only, no pre-2020 coverage** — dropped from look 3, leaving 6 members
  (`flag_coach, flag_division, flag_arrests, flag_bye, flag_protection,
  flag_tank_zone`). Verified pre-2020 coverage for the rest: `flag_arrests`
  (`data/raw/player_arrests/.../incidents_point_in_time.parquet` spans
  2000-2026), `flag_protection` (PBP snapshot covers season=2009+, same
  basis LEAD-65 used), `flag_coach/flag_division/flag_bye/flag_tank_zone`
  (schedule-derived, no season floor). `reddit_home_comment_ratio_elevated`
  has nonzero counts every season 2010-2025 (17-33/season) — usable back to
  2011.
- **Evaluation**: LOSO over all 15 seasons; each variant graded (a) on all
  15 held-out seasons pooled and (b) separately restricted to the 2020-2025
  held-out folds only, both vs base on the same rows. Paired season-block
  (then week-block) bootstrap via `nfl_ats.signal_atlas._cell`
  (2,000 draws, seed 20260923), reporting `probability_positive`
  (never "contains zero"), decisive-game record, accuracy-points delta and
  interval, brier/log-loss deltas, in-sample-vs-OOS gap, per-fold natural
  coefficients.
- **Classification rule**: `unresolved_below_power` unless the whole
  interval sits on the wrong side (`wrong_sign_resolved`) or a positive
  control bounds it. Crossing zero alone closes nothing.
- New script: `scripts/extended_population_regrade.py` (run once, no
  `src/` edits, no tests added). Output:
  `artifacts/extended_population_regrade/<ts>/summary.json`.

## State
Ran once for real: `python scripts/extended_population_regrade.py` ->
`artifacts/extended_population_regrade/20260923T210646Z/summary.json` (+
`population_used.parquet`). `ruff format`/`ruff check` clean on the new
script. No `src/` edits, no tests, no commit, no registry write (see Next).

**QA**: recomputed `composition_flag_sum` from the 7 individually rebuilt
member flags matches the population's stored sum on all 3,734 rows (0
mismatches) -- the recomputation is a faithful reconstruction.

**Coverage gate result**: `flag_cold_visitor` is 0 for every 2011-2019 row
(forecast archive is 2020-2025 only) -- **dropped**, as predeclared. The
other 6 counted members and `reddit_home_comment_ratio_elevated` all have
nonzero pre-2020 coverage (measured counts in the artifact's
`flag_pre_2020_coverage`/`reddit_coverage`).

**Base** (measured): in-sample accuracy 53.59%, LOSO OOS accuracy 53.40%
(n=3,734, 2011-2025).

**Per-look results** (accuracy points, variant minus base, paired
season-then-week block bootstrap, 2000 draws, seed 20260923):

1. `week_gated_protection_flag_sum` -- all 15 seasons: -0.08 pts, 95% CI
   [-0.97, 0.90], P+ 0.434, n=3,734, blocks=260, decisive 195 (96 variant /
   99 base). 2020-2025 held-out only: -1.00 pts, CI [-2.32, 0.33], P+ 0.070,
   decisive 73 (29/44). Fold coefficient on `gated_flag_sum` positive in
   15/15 LOSO folds. IS/OOS gap 0.08 pts (no overfit blowup).
   **unresolved_below_power** -- CI crosses zero both slices.
2. `reddit_home_comment_ratio_elevated` -- all 15 seasons: -0.268 pts, CI
   [-1.00, 0.40], P+ 0.228, decisive 60 (25/35). 2020-2025 held-out only:
   +0.133 pts, CI [-0.52, 0.83], P+ 0.672, decisive 14 (8/6). Coefficient
   positive in 14/15 folds. IS/OOS gap 0.32 pts. **unresolved_below_power**
   -- sign is not even stable in direction across the two grading slices.
3. `composition_flags_separate` (6 members: coach, division, arrests, bye,
   protection, tank_zone; `flag_cold_visitor` excluded) -- all 15 seasons:
   -0.214 pts, CI [-1.63, 1.18], P+ 0.378, decisive 278 (135/143). 2020-2025
   held-out only: -1.863 pts, CI [-3.96, 0.13], P+ 0.031, decisive 132
   (52/80). All 6 member coefficients positive in >=14/15 folds (stable
   sign). IS/OOS gap 0.67 pts. **unresolved_below_power** -- the
   2020-2025-only CI upper bound is +0.13, just short of the whole-interval-
   negative bar AGENTS.md sets for `wrong_sign_resolved`; extending to
   2011-2025 washes the negative read out almost entirely (-0.214 pooled),
   so the 2.5x-games population does not resolve it either way.

**Bottom line**: none of the three predeclared terms clears refutation or
control-bound; all three stay `unresolved_below_power`, same terminal state
as their 2020-2025-only reads, now measured at 2.5x the games with a
consistent, near-zero-centered result. `docs/lanes/opener-population-
backfill.md`'s own sanity look (base 4-term fit only, no added term) already
found the +2,231 pre-2020 games net roughly a wash for the base fit itself
(-0.46 pts on the 2020-2025 held-out slice); this session's three added-term
reads are consistent with that -- more games did not surface a signal these
terms did not already show.

## Tried
See State (full measured numbers). No other approach attempted.

## Next
Root (not this subagent) runs, one per look, then decides whether to also
record the 2020-2025-only cell as a sibling under `--batch` (numbers are in
State above if so):

```
nfl-ats weak-signals record --league nfl --name extended_population_week_gated_protection_flag_sum \
  --description "Week<=4-gated protection-mismatch flag sum vs served composition_flag_sum, 2011-2025 fit population (3,734 games, 15-season LOSO)" \
  --source artifacts/extended_population_regrade/20260923T210646Z/summary.json \
  --effect -0.08 --effect-units accuracy_points --classification unresolved_below_power \
  --season-start 2011 --season-end 2025 --standard-error 0.487 \
  --interval-low -0.966 --interval-high 0.899 --probability-positive 0.434 \
  --sample-games 3734 --sample-blocks 260 --family extended_population_regrade_2011_2025 \
  --classification-evidence "CI crosses zero on both the pooled 15-season grade and the 2020-2025-only held-out subgrade (-1.00 pts, CI [-2.32,0.33], P+ 0.070); 2.5x the games did not resolve LEAD-65's prior below-power read" \
  --category onfield \
  --plain-summary "Limiting the protection-mismatch flag to only count in the first month of a season, instead of all season, made no measurable difference once the model saw 15 years of games instead of 6."

nfl-ats weak-signals record --league nfl --name extended_population_reddit_home_comment_ratio \
  --description "reddit_home_comment_ratio_elevated added as a 5th fitted term, 2011-2025 fit population (3,734 games, 15-season LOSO)" \
  --source artifacts/extended_population_regrade/20260923T210646Z/summary.json \
  --effect -0.268 --effect-units accuracy_points --classification unresolved_below_power \
  --season-start 2011 --season-end 2025 --standard-error 0.348 \
  --interval-low -1.002 --interval-high 0.396 --probability-positive 0.228 \
  --sample-games 3734 --sample-blocks 260 --family extended_population_regrade_2011_2025 \
  --classification-evidence "Pooled 15-season effect is negative (P+ 0.228) while the 2020-2025-only held-out subgrade is positive (+0.133 pts, P+ 0.672) -- sign is not stable across grading slices, no resolution either way" \
  --category attention \
  --plain-summary "Adding how lopsided the home team's Reddit comment traffic is as a fifth input made no reliable difference, and it even flipped direction depending on which years were graded."

nfl-ats weak-signals record --league nfl --name extended_population_composition_flags_separate \
  --description "6 composition members (coach/division/arrests/bye/protection/tank_zone; cold_visitor dropped, no pre-2020 coverage) entered separately instead of summed, 2011-2025 fit population (3,734 games, 15-season LOSO), one joint look" \
  --source artifacts/extended_population_regrade/20260923T210646Z/summary.json \
  --effect -0.214 --effect-units accuracy_points --classification unresolved_below_power \
  --season-start 2011 --season-end 2025 --standard-error 0.707 \
  --interval-low -1.625 --interval-high 1.175 --probability-positive 0.378 \
  --sample-games 3734 --sample-blocks 260 --family extended_population_regrade_2011_2025 \
  --classification-evidence "2020-2025-only held-out subgrade reads -1.863 pts, CI [-3.955,0.129], P+ 0.031 -- close to but not fully on the negative side (upper bound +0.129), so wrong_sign_resolved's whole-interval bar is not met; pooling in the 2011-2019 games washes the negative read down to -0.214, P+ 0.378" \
  --category modeling \
  --plain-summary "Letting the model weigh each of six situational flags on its own, instead of adding them into one combined score, made no reliable difference across 15 years of games, though the newest six years alone still lean slightly worse."
```

`--reliability` (split-half reliability of the underlying trait) is omitted
from all three commands above: `summary.json`'s `cells.*.metrics.reliability`
is a calibration table (predicted-vs-observed by probability band,
`signal_atlas._metrics`), not a split-half reliability estimate -- no such
number was computed this session, so none is recorded rather than inventing
one. A future unit that wants `--reliability` populated needs a genuine
split-half construction first.

## Open
- None of the three terms should be dropped from future consideration --
  AGENTS.md bars closing on a zero-crossing interval alone; these are
  `unresolved_below_power`, not refuted.
- `composition_flags_separate`'s 2020-2025-only lean (P+ 0.031) is the
  closest of the three to a resolvable negative signal; if a future session
  wants to chase it, the natural next step is checking whether it is really
  one member (likely `flag_protection`, already implicated by LEAD-65) or a
  genuine joint effect, not re-running this same joint look again.
