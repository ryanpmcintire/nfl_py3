# XLG-06 Stage 4: prior wiring + ATS evaluation (predeclaration)

**Status:** predeclared 2026-09-03. Sections 0–9 are frozen. NOTHING below
has been run: no rotation family declared, no window assigned, no feature
wired, no ATS comparison scored. A run that spends a window must cite this
document unchanged (sections 0–9 byte-identical) and append results as
section 10.

**Parents:** `docs/xlg06_rookie_prior_screen.md` (Stage 1),
`docs/xlg06_stage2_crosswalk.md` (identity),
`docs/xlg06_stage2_nfl_screen.md` (Stage-2 gate, r=+0.1004, P+ 0.9275),
`docs/xlg06_stage3_prior_spec.md` (frozen prior parameters).

---

## 0. Binding closing-grounds taxonomy (AGENTS.md, pasted verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (1) refuted mechanism — a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (2) bounded by a positive
control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". The registry code
hard-rejects inadmissible closures; if a record command errors, the verdict is
wrong, not the validator.

Decisions are expected value. `probability_positive` above 0.5 favours the
candidate; predeclared thresholds govern only what a document may CLAIM, never
which card is played.

---

## 1. What this look decides, and the honesty preamble

Whether adding the frozen Stage-3 rookie prior to the production chain moves
forced-pick accuracy at the pool-relevant grade. Disclosure before numbers:
the prior is weak by measurement (Stage-3 slope 95% [-1.72, +4.01],
R² 0.0048), so the modal honest outcome of this look is
`unresolved_below_power` — a predicted near-null that must be recorded, not
treated as a closure. A positive result proposes a challenger, never a
promotion; a negative result that does not meet a closing ground changes
nothing.

## 2. Rotation family (to be declared at run time, not here)

Family `xlg06_rookie_prior_on_production`, inheriting nothing, spending one
earliest-eligible confirmation window via `nfl-ats rotation declare/assign`.
The window is spent permanently when `record` runs; declaration and
assignment alone spend nothing, and this document explicitly does NOT run
either — the runner cites the assignment output in section 10.

## 3. Arms (frozen)

- **Baseline:** production `weak_stack` / `market_residual` / ridge alpha 10,
  unchanged, full production chain fit.
- **Candidate:** baseline + frozen Stage-3 prior features (section 4), full
  production chain fit on both arms. The ONLY difference between arms is the
  prior columns; N0 stays 300 (the sensitivity appendix selected nothing,
  and this look selects nothing — re-tuning N0 here would be window shopping).

## 4. Feature wiring (frozen design, implementation at run time)

Per team-game, for each drafted skill player (WR/RB/TE) on the pregame
expected lineup with a usable recruiting rating: `blend_prior` from
`src/nfl_ats/xlg06_prior.py` at frozen Stage-3 parameters, with snaps = that
player's strictly-prior NFL offensive snaps and observed_avg = their
strictly-prior EPA/game. Team-level columns: snap-weighted mean prior
expectation for the home and away skill groups plus the differential
(3 columns). Players without a rating contribute weight zero (documented
missingness, not imputed signal). A leakage test pins that only
strictly-prior snaps and the pre-draft rating enter; the rookie-season
outcomes that fit (a, b) never enter any row.

## 5. Population, grade, metric (frozen)

Rotation-assigned window, paired non-push games, probability-rule forced
picks at the OPENER grade (the pool grade; close secondary for diagnosis
only). Metric: paired accuracy-points delta (candidate − production) with
week-blocked 95% interval and `probability_positive`. Prediction-level pairs
retained.

## 6. Controls (frozen)

- Frozen-pick null: 200 within-week permutations (same harness as prior
  on-production confirmations).
- Realized-margin positive control (leak the outcome into the prior
  columns): must score large and positive or the harness is blind and the
  look is void, not negative.

## 7. Reliability gate (frozen)

The Stage-2 outcome reliability (Spearman-Brown 0.6464) already rules out
`no_split_half_reliability` for the underlying construct. No re-measurement
here; if the candidate moves the wrong way with the whole interval below
zero, `wrong_sign_resolved` is admissible.

## 8. Decision rule (frozen)

Record through both registries under family `xlg06_rookie_prior_on_production`:
`unresolved_below_power` / `unresolved` unless an admissible closing ground
fires. No card, model, or profile change follows under any outcome; a
positive lean proposes a no-window-cost prospective challenger at most.

## 9. What this look may therefore claim

At most: whether the frozen rookie prior moves production accuracy on one
assigned window, with stated uncertainty. It may not claim an optimal N0, a
position-specific effect (the prior is pooled skill by design), or any
generalization beyond the assigned window.

## 10. Results (wiring implemented 2026-09-03; assignment and scoring NOT run)

**Rotation pool measured exhausted:** family `xlg06_rookie_prior_on_production`
declared 2026-09-03 (open, no windows); `rotation assign` refuses — opener
pool [2020, 2025] has 3/3 windows spent (12 prior uses each of 2020/2021) and
the close pool likewise has zero unspent windows. Scoring is impossible by
mechanism design until new seasons open fresh blocks. The declaration stands
as the queued vehicle.

**Wiring implemented without scoring** (`src/nfl_ats/xlg06_prior_feature.py`,
`tests/test_xlg06_prior_feature.py`, 6 tests): snap-weighted Stage-3 prior
expectations per side plus differential, every input from completed REG weeks
strictly before the game (game-time trailing-4-week activity, career sums
over visible rows only, latest-team attribution), missing sides NaN.
Validated end to end on real inputs (272-game 2024 REG slice, 8.1s):
home/away/diff coverage 1.000/0.993/0.993, means +0.83/+0.80/+0.02 EPA/game
— sane magnitudes, no ATS outcome scored, no window touched.
