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

## Unit 2 state (2026-09-23, ran once for real)
Ran `/f/Repos/nfl_py3/.tools/uv.exe run python scripts/players_on_field_rating_unit2.py`
(exit 0) -> `artifacts/players_on_field_rating_unit2/20260923T202320Z/summary.json`
+ `per_game.parquet` (1,503 games, same population as unit 1). Both predeclared
looks add ONE term to the served four-term fit, LOSO by season.

**(a) `look_a_unpriced_residual`** (`diff_lineup_total` residualised on
`tue_open_home_spread` via OLS fit on LOSO training seasons only, applied to
the held-out season, residual is the added term): accuracy **-0.532 pts**,
95% CI [-2.041, 0.840], **P+ 0.259**; decisive games 54 (23-31, exact-null p
0.341, not significant); Brier P+ 0.203, log-loss P+ 0.195 (both cross zero);
IS-OOS accuracy gap 0.0047 vs base 0.0020 (small, no blowup); per-fold
standardised coefficient sign-flips once across 6 folds (2020 +0.351, 2021
+0.085, 2022 -0.033 near-zero, 2023 +0.480, 2024 +0.800, 2025 +0.395) --
mostly positive sign but magnitude unstable (0.03x to 0.80x).

**(b) `look_b_unseen_interaction`** (`diff_lineup_total * (1 -
market_move_available)`, added as the 5th term): accuracy **-0.333 pts**, 95%
CI [-1.141, 0.279], **P+ 0.151**; decisive games 23 (9-14, exact-null p
0.405); Brier P+ 0.194, log-loss P+ 0.201 (both cross zero); IS-OOS accuracy
gap 0.0040 vs base 0.0020; per-fold coefficient sign-flips once across 6
folds (2020 +0.637, 2021 -0.415, 2022 +0.313, 2023 +0.131, 2024 +0.147, 2025
+0.098) -- unstable magnitude, one sign flip.

Both intervals cross zero (neither fully on the wrong side), so neither is
`wrong_sign_resolved`; both classify `unresolved_below_power` per AGENTS.
Decision implication: do not add either term to the served model this round
-- same conclusion as unit 1's two terms, and consistent with them (both
unit-2 arms are derived from `diff_lineup_total`, the same construct unit 1
already found unstable across LOSO folds).

`nfl-ats weak-signals record --help` import chain checked clean this session
(`python -c "import nfl_ats.cli"` exit 0) -- the concurrent
`displayed_confidence.py` edit that blocked unit 1's recording is gone from
git status now, so unit 1's deferred commands AND the unit 2 commands below
should both be runnable by the root.

## Tried
Reused `build_fit_population`, `_fit_logit` (aliased) from
`src/nfl_ats/pick_probability_fit.py`; `_cell`/`_metrics` from
`src/nfl_ats/signal_atlas.py` (full-vs-reduced decisive-game pattern);
`build_pairing_table`/`close_reference_table` from `src/nfl_ats/clv.py` for
the close line. No src/ or ROADMAP.md edits made.

## Next

- 2026-09-23 (this session): unit 3 feasibility checked and closed as
  infeasible -- see "Unit 3 state" above. The proposed Tuesday-open-to-
  decision-time availability CHANGE construct is not measurable as a signal
  distinct from unit 1's already-recorded `diff_divergence` (1.8% of
  2020-2024 team-games have any real report by Tuesday noon; 2025's proxy
  timestamps equal the decision cutoff exactly, no sub-week resolution). No
  new record command for unit 3. With all four unit-1/2 terms
  `unresolved_below_power` and unit 3 infeasible, MOD-22 has exhausted the
  constructs reachable from the cached PER-09 point-in-time lineup table --
  root's call whether to close the family (no refuted mechanism, no powered
  control, so per AGENTS it stays `unresolved_below_power` rather than
  `closed_negative`; "closed" here would mean stop spending window budget on
  it, not reclassify the interval).
- 2026-09-23 root: unit 2 cells recorded (registry 6,997); unit 1 cells were already recorded earlier, the commands below are history. All four MOD-22 terms are unresolved_below_power, so none closes the family (AGENTS: only a refuted mechanism or a powered control closes). Unit 3, if any, needs a different construct, for example the availability change between Tuesday open and kickoff, not another transform of diff_lineup_total.

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
3. **Unit 2 recording (new, root runs once unit 1's two commands above have
   gone through)**:
   - `nfl-ats weak-signals record --name players_on_field_rating_residual_unpriced_pick_probability_term --family players_on_field_rating_v1 --league nfl --season-start 2020 --season-end 2025 --effect -0.5322687957418496 --effect-units accuracy_points --interval-low -2.0412909349786417 --interval-high 0.840472818925878 --probability-positive 0.25875 --sample-games 1503 --sample-blocks 6 --classification unresolved_below_power --reliability 0.154 --category onfield --source artifacts/players_on_field_rating_unit2/20260923T202320Z/summary.json --classification-evidence "interval crosses zero; only 54 decisive games (23-31, exact-null p 0.341); per-fold coefficient sign-flips once, 2022 near-zero -0.033, other 5 folds positive 0.085 to 0.800, magnitude unstable" --description "diff_lineup_total residualised on tue_open_home_spread (OLS fit on LOSO training seasons only, applied to held-out season) added as 5th fitted term to the served pick-probability model, LOSO by season, MOD-22 unit 2 look a" --notes "reliability inherited from PER-09 split-half of the underlying season-lagged RAPM ratings (min 0.154 defense), not remeasured this session"`
   - `nfl-ats weak-signals record --name players_on_field_rating_unseen_interaction_pick_probability_term --family players_on_field_rating_v1 --league nfl --season-start 2020 --season-end 2025 --effect -0.33266799733865604 --effect-units accuracy_points --interval-low -1.1414123645757024 --interval-high 0.2793639125747327 --probability-positive 0.1505 --sample-games 1503 --sample-blocks 6 --classification unresolved_below_power --reliability 0.154 --category onfield --source artifacts/players_on_field_rating_unit2/20260923T202320Z/summary.json --classification-evidence "interval crosses zero; only 23 decisive games (9-14, exact-null p 0.405); per-fold coefficient sign-flips once, 2021 -0.415, other 5 folds positive 0.098 to 0.637" --description "diff_lineup_total x (1 - market_move_available) interaction added as 5th fitted term to the served pick-probability model, LOSO by season, MOD-22 unit 2 look b" --notes "reliability inherited from PER-09 split-half of the underlying season-lagged RAPM ratings (min 0.154 defense), not remeasured this session"`
   - After recording both, unit 2 is done. MOD-22 (units 1+2, four terms
     tested, all `unresolved_below_power`, all derived from the same
     `diff_lineup_total`/`diff_divergence` on-field construct with unstable
     LOSO coefficients) is a candidate to close as a family review rather than
     spawn a unit 3 -- root's call, not this subagent's.

## Unit 2 predeclaration (before fitting, 2026-09-23)
Exactly two looks, both add ONE term to the served four-term fit
(`model_logit`, `flag_sum`, `move_toward_home`, `move_available`), LOSO by
season, base unchanged from unit 1:
- (a) `look_a_unpriced_residual`: residualise `diff_lineup_total` on
  `tue_open_home_spread` (OLS, fit on the LOSO training seasons only, applied
  to the held-out season -- no leakage); the residual is the added term.
- (b) `look_b_unseen_interaction`: `diff_lineup_total * (1 - move_available)`
  (the construct gated to games where the market move is not yet observed) is
  the added term.
No other arms, bands, or cuts. Script: `scripts/players_on_field_rating_unit2.py`
(new, reuses `design_matrix`/`standardisers`/`natural_coefficients`/`predict`/
`loso`/`in_sample_fit` from `players_on_field_rating_eval.py` by direct
import, plus `signal_cell`/`signal_metrics` and `build_fit_population`/
`FIT_FEATURES`/`FIT_RIDGE`/`_fit_logit` as unit 1). Artifacts under
`artifacts/players_on_field_rating_unit2/<ts>/`.

## Unit 3 state (2026-09-23, feasibility check, ran once for real)
Ran `/f/Repos/nfl_py3/.tools/uv.exe run python scripts/players_on_field_rating_unit3.py`
(exit 0) -> `artifacts/players_on_field_rating_unit3/20260923T202749Z/summary.json`,
3,230 team-games / 1,615 REG games, 2020-2025, joined to the same injury
snapshot PER-09 uses (`data/players/raw/20260910T205112Z/injuries.parquet`,
`effective_observed_at` + `observed_at_basis`).

**Feasibility result: NOT reconstructible as a distinct signal. Stopping at
Step 1 per the lane's own predeclared stop condition; no Step 2 fit run.**

Two independent reasons, both measured this session:
1. **2020-2024 (100% real `date_modified` timestamps, confirmed by
   basis-mix counts):** only **1.8% of team-games (57/3,230 overall; 0.9%-3.5%
   per season)** have any real injury revision visible by Tuesday noon ET (the
   AGENTS pick-freeze instant), vs **82.8% visible at the existing
   decision time** (kickoff-24h; 91-100% per season 2020-2024). Real injury
   reports essentially never exist yet at Tuesday open -- practice/status
   reports start Wednesday. So for 99%+ of team-games, "the Tuesday-time
   lineup" is numerically indistinguishable from the full-health lineup
   (`full_strength_total`), and a "Tuesday-to-decision-time change" term
   would almost everywhere equal `decision_time_lineup - full_strength_lineup`
   -- which is exactly `diff_divergence`, the construct unit 1 already fit
   and recorded as `unresolved_below_power` (effect -0.067 pts, P+ 0.425).
   Building "unit 3" would not be a new look; it would recompute unit 1's
   cell under a new name, inflating the look count with no new information.
2. **2025 (100% `week_proxy`, no real `date_modified` rows at all):** every
   proxy timestamp is exactly 24.0h before kickoff (min=max=mean=24.0,
   std=0.0; 0 team-weeks have more than one distinct timestamp) -- i.e. the
   proxy **is** the decision-time cutoff by construction, not a real
   sub-week observation. 0% of 2025 team-games have anything visible at
   Tuesday noon. There is no sub-week timestamp resolution to reconstruct a
   "Tuesday state" from for this part of the window; treating "Tuesday" as
   full-health and "decision time" as the proxied lineup would just
   reproduce `diff_divergence` again, and treating the proxy itself as a
   "Tuesday" value would be a leakage-shaped duplicate of the decision-time
   value (identical instant), not new information.

Decision implication: MOD-22 has no unit 3 fitted-term look to run. The
already-recorded unit 1 `diff_divergence` cell is the closest measurable
proxy for "availability change from a clean-slate baseline to decision time"
across the whole 2020-2025 window, and it is already `unresolved_below_power`.
No new `weak-signals record` command is warranted for unit 3 -- this is a
data-availability/feasibility finding, not a fitted effect with a confidence
interval, so it does not fit the `probability_positive` recording schema; the
root should treat this artifact (`summary.json`) as the citation if MOD-22 is
written up or closed as a family.

## Open
- Registry recording for unit 1 (2 commands) and unit 2 (2 commands) is all
  that is outstanding; the CLI import blocker is gone, all four numbers are
  captured in this file plus `artifacts/players_on_field_rating/20260923T191937Z/summary.json`
  and `artifacts/players_on_field_rating_unit2/20260923T202320Z/summary.json`.
  This subagent made no registry writes (shared registry, out of scope).
- No ROADMAP.md MOD-22 row exists yet (this task's instructions excluded
  editing ROADMAP.md); the primary orchestrator should add one when it next
  touches ROADMAP.md, citing this lane, and should also record the note above
  that MOD-22 (4 terms, all unresolved_below_power) is a closure-family
  candidate rather than a unit 3.
