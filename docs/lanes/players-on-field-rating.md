# players-on-field-rating (ROADMAP MOD-22, unit 1)

## Goal
Bottom-up team strength: a team's rating on a given day as the sum of the
players actually expected on the field (point-in-time availability x role
share x player rating), not team aggregate + injury adjustment. Unit 1: build
the point-in-time on-field rating differential per game, compare it to the
full-health depth-chart sum to isolate "who is actually playing," grade it (a)
against line movement and (b) as one added fitted term in the served
four-term pick-probability model, LOSO by season. Record cells with
`nfl-ats weak-signals record`, family `players_on_field_rating_v1`.

What is new vs PER-09 (`docs/latent_ratings_on_production.md`,
`scripts/latent_ratings_on_production.py`): PER-09 already built and cached
this exact construct (`artifacts/latent_ratings_on_production/expected_lineup_ratings.parquet`,
2,383 REG games 2017-2025: `diff_lineup_total` = point-in-time expected-active
sum differential, `diff_divergence` = availability delta from full-health,
`home_full_strength_total`/`away_full_strength_total` = full-health depth
chart) and screened it with ad hoc rule-based tilts on top of the served pick
(Screen A/B, both `unresolved_below_power`). This unit reuses that same
point-in-time construct unmodified (no rebuild) but (1) adds the close-minus-
open line-move look, never tested by PER-09, and (2) grades it as ONE fitted
term inside the served four-term calibrated logit (`build_fit_population` /
the same ridge-Newton fit as `fit_pick_probability`), LOSO by season, per
AGENTS "one calibrated probability decides every pick" -- not a rule tilt.

## State
Script `scripts/players_on_field_rating_eval.py` (owned, no comments) ran
successfully once:
`/f/Repos/nfl_py3/.tools/uv.exe run python scripts/players_on_field_rating_eval.py`
Output: `artifacts/players_on_field_rating/20260923T191937Z/summary.json` +
`per_game.parquet` (1,503 games, 100% coverage merged from the cached
PER-09 ratings table).

**Line-move look (a)**, season-block bootstrap (2000 draws, seed 20260923),
close-minus-open vs each term: `diff_lineup_total` slope 2.03, r 0.101, CI
[1.28, 2.50], **P+ 1.00**; `diff_divergence` slope 9.72, r 0.110, CI [3.25,
14.41], P+ 0.998; `diff_full_strength_total` (no availability weighting)
slope 1.34, r 0.071, CI [0.39, 2.00], P+ 0.998. The market moves toward the
point-in-time construct more than toward the naive full-health aggregate --
confirms the construct carries real information the market also reacts to,
not by itself an unpriced edge at open.

**Fitted-probability-term look (b)**, LOSO by season, base = the served
four-term model (`model_logit`, `flag_sum`, `move_toward_home`,
`move_available`), 1,503 games, decisive-game record via the repo's existing
`signal_atlas._cell` full-vs-reduced pattern:
- `diff_lineup_total` added: accuracy **-0.266 pts**, 95% CI [-0.677, 0.000],
  **P+ 0.040**; decisive games 4 (0-4, exact-null p 0.125, too few to
  adjudicate); Brier P+ 0.249, log-loss P+ 0.263 (both cross zero); IS-OOS
  accuracy gap 0.0033 vs base 0.0020 (no overfitting blowup); per-fold
  standardized coefficient sign-flips once (2020 +0.071, all other folds
  negative -0.02 to -0.31) -- not stable season to season.
- `diff_divergence` added: accuracy -0.067 pts, 95% CI [-0.914, 0.910], P+
  0.425; decisive games 35 (17-18, coin flip); Brier P+ 0.020, log-loss P+
  0.021 (both interval fully negative -- calibration degrades); coefficient
  sign-flips twice across the 6 folds (-2.54, +0.32, -0.79, -0.19, -1.96,
  +2.20) -- unstable.

Neither interval is fully on the wrong side of zero (both touch or cross
zero), so per AGENTS neither closes; both are `unresolved_below_power`.
Decision implication: do not add either term to the served model this round;
the coefficient instability across LOSO folds is itself the reason (a real
but currently unmeasurable-at-this-resolution effect, consistent with PER-09).

## Tried
Reused `build_fit_population`, `_fit_logit` (aliased) from
`src/nfl_ats/pick_probability_fit.py`; `_cell`/`_metrics` from
`src/nfl_ats/signal_atlas.py` (full-vs-reduced decisive-game pattern);
`build_pairing_table`/`close_reference_table` from `src/nfl_ats/clv.py` for
the close line. No src/ or ROADMAP.md edits made.

## Next

0. DONE 2026-09-23 (root): both accuracy cells recorded, registry 6,990; the commands below are history. Unit 1 complete.
1. **BLOCKED right now**: `nfl-ats weak-signals record` cannot run --
   `src/nfl_ats/cli.py` import chain is currently broken by an uncommitted,
   in-flight edit to `src/nfl_ats/displayed_confidence.py` (git status `M`,
   not touched by this lane): first missing name was
   `ProductionDisplayedConfidence`, retried a minute later and the missing
   name had changed to `served_strength_bands` -- confirms another
   concurrent process is actively editing that file. Do not touch it; wait
   for it to settle, then run:
   - `nfl-ats weak-signals record --name players_on_field_rating_diff_lineup_total_pick_probability_term --family players_on_field_rating_v1 --league nfl --season-start 2020 --season-end 2025 --effect -0.2661343978709248 --effect-units accuracy_points --interval-low -0.6766014385668202 --interval-high 0.0 --probability-positive 0.04025 --sample-games 1503 --sample-blocks 6 --classification unresolved_below_power --reliability 0.154 --category onfield --source artifacts/players_on_field_rating/20260923T191937Z/summary.json --classification-evidence "interval touches but is not fully on the wrong side of zero; only 4 decisive games; per-fold coefficient sign-flips once" --description "diff_lineup_total added as 5th fitted term to the served pick-probability model, LOSO by season, MOD-22 unit 1" --notes "reliability inherited from PER-09 split-half of the underlying season-lagged RAPM ratings (min 0.154 defense), not remeasured this session"`
   - same for `diff_divergence`: effect -0.0665335994677312, interval-low
     -0.9139580463669194, interval-high 0.9097749286714685,
     probability-positive 0.425, name
     `players_on_field_rating_diff_divergence_pick_probability_term`.
   - optionally a third cell for the line-move look (`diff_lineup_total`
     close-minus-open correlation 0.101, CI [1.28,2.50] slope units,
     effect-units `correlation`, classification `unresolved_below_power`
     since the registry has no "confirmed-serve" state per AGENTS
     promotion-bar rule) if the owner wants it in the registry too.
2. After recording, this unit is done; MOD-22 unit 2 (if queued) would be
   deciding whether `diff_lineup_total` or `diff_divergence` is worth a
   dedicated rotation window given both are `unresolved_below_power` with
   unstable per-fold coefficients (same open-window budget constraint PER-09
   already documented).

## Open
- Registry recording deferred solely by the concurrent `displayed_confidence.py`
  edit above; all numbers needed to record are captured in this file and in
  `artifacts/players_on_field_rating/20260923T191937Z/summary.json`.
- No ROADMAP.md MOD-22 row exists yet (this task's instructions excluded
  editing ROADMAP.md); the primary orchestrator should add one when it next
  touches ROADMAP.md, citing this lane.
