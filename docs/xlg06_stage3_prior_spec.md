# XLG-06 Stage 3: frozen rookie-prior spec (predeclaration)

**Status:** predeclared 2026-09-03, BEFORE any prior parameter for this spec
was estimated. Sections 0–9 are frozen; section 10 (Results) is appended
after the fit and nothing above it is edited afterwards.

**Owning work package:** XLG-06 Stage 3. Files: this document,
`src/nfl_ats/xlg06_prior.py`, `tests/test_xlg06_prior.py`,
`artifacts/xlg06_stage3_prior/`.

**Parents:** `docs/xlg06_rookie_prior_screen.md` (Stage 1),
`docs/xlg06_stage2_crosswalk.md` (identity audit),
`docs/xlg06_stage2_nfl_screen.md` (Stage-2 gate: r=+0.1004, P+ 0.9275,
`unresolved_below_power`). Stage 2 earned "the next predeclared
prior-weight step" — this document is that step, and only that step.

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

This slice fits NO new effect: it freezes parameters for a prior. There is
therefore no registry verdict to record here, and none will be — the ATS
evaluation that could earn one needs its own predeclaration on a window this
slice does not spend (section 9).

---

## 1. What this slice builds, and what it explicitly does not

A frozen, versioned **prior spec**: a parametric map from a drafted skill
player's pre-draft recruiting rating to an expected rookie production rate,
plus a fixed exposure-decay schedule that blends the prior with observed NFL
production as snaps accumulate. Output is a parameter artifact
(`prior_spec.json`: intercept, slope, their bootstrap intervals, decay
constant, worked examples), not a feature column, not a model change, not an
ATS comparison. Wiring the spec into the game feature table and scoring it is
Stage 4 (or a challenger proposal) and is NOT run here.

## 2. Inputs (frozen)

- Per-player Stage-2 table `artifacts/xlg06_stage2_nfl/20260903T170528Z/rookie_epa.parquet`
  (n=272, `rating_num`, `rookie_epa`, `rookie_reg_weeks`, `rookie_season`).
  No new data pull, no new join, no ATS outcome is touched.
- Rate form: `rookie_epa_per_game = rookie_epa / rookie_reg_weeks`
  (REG weeks with a row; bye weeks never enter the denominator).

## 3. Prior form (frozen)

For a player with recruiting rating `r` and `s` accumulated NFL offensive
snaps before the target game:

```text
mu(r)      = a + b * r
w(s)       = N0 / (N0 + s)
prior(s)   = w(s) * mu(r) + (1 - w(s)) * observed_avg
```

- `(a, b)` are the OLS coefficients of `rookie_epa_per_game ~ rating` on the
  Stage-2 population. Fitting the map on the same population that gated it
  is disclosed, not hidden: the INDEPENDENT check is the future ATS look on
  unspent windows, which is exactly why this slice spends none.
- `N0 = 300` snaps, fixed and disclosed as a placeholder constant (roughly
  1.5 games of full-time skill snaps — a whole offseason of takes, not a
  measurement). Section 10 carries a sensitivity appendix recomputing the
  weight curve at N0 ∈ {150, 600}; the appendix selects nothing.
- `observed_avg` is the player's own EPA-per-game to date (zero snaps →
  pure prior). Snap counts come from the existing snap-count snapshot at
  wiring time, not in this slice.

## 4. Uncertainty (frozen)

Bootstrap CI on the slope `b` (10,000 cohort-blocked resamples, seed
20260905, same block definition as Stage 2). Intercept CI likewise. R² of
the OLS fit reported as description, never as a gate.

## 5. Convergent validity (frozen, weak by design)

`stars` vs `rating` rank correlation on the Stage-2 population (both from
the same recruiting source, so this checks internal consistency, not
external validity). Reported once, no gate, no verdict.

## 6. Leakage (frozen, fail-closed)

- The map's only player inputs are the pre-draft rating and (at wiring
  time) strictly-prior NFL snaps. No post-rookie information enters the
  parameters: the fit uses rookie-season outcomes only, and the artifact
  records the exact input snapshot IDs.
- Tests pin: (a) the fitter refuses rows with missing rating/year; (b) a
  synthetic post-dated predictor column can never silently substitute for
  the rating (column-allowlist contract); (c) determinism for fixed seeds.

## 7. Worked examples (frozen set, values filled after the fit)

Three fixed ratings (0.80 / 0.90 / 1.00) × three snap counts (0 / 300 /
1200): prior EPA/game from the frozen parameters. Illustrates shrinkage,
selects nothing.

## 8. Test contract (release-blocking)

`tests/test_xlg06_prior.py` covers, without network access: OLS recovery on
a synthetic line; weight-curve boundaries (s=0 → pure prior, s≫N0 →
observed); determinism; column-allowlist rejection; N0 sensitivity helper
shape. No ATS, no registry, no window in any test.

## 9. What this slice may therefore claim, and the explicit next step

At most: a frozen, reproducible map from rating to expected rookie rate
with stated uncertainty and a fixed decay — the mechanical prerequisite for
proposing a recruiting-prior feature. It may not claim the prior helps ATS
picks, has an optimal N0, or generalises beyond drafted WR/RB/TE. The next
step is a SEPARATE predeclaration (feature wiring + ATS evaluation on
unspent windows, Stage 4); writing it is queued, not run, here.

## 10. Results (added after the fit, 2026-09-03)

Estimated by `fit_prior.py` (scratch, same formulas as `src/nfl_ats/xlg06_prior.py`;
artifact `artifacts/xlg06_stage3_prior/20260903T191431Z/prior_spec.json`):

- Frozen map (n = 272): `mu(r) = -0.9686 + 1.3868 * r` EPA/game, R² = 0.0048.
  Slope 95% [-1.7174, +4.0144], intercept 95% [-3.2203, +1.7201] (10,000
  cohort-blocked resamples, seed 20260905). The slope interval comfortably
  contains zero — the map is a weak, high-uncertainty prior, exactly what a
  Stage-2 r of +0.10 implies. This is a parameter with an interval, not a
  verdict, and §0's taxonomy is not invoked.
- Worked examples (observed average 0.0): a 0.90 recruit opens at +0.28
  EPA/game, halved at 300 snaps (+0.14), down to +0.06 at 1,200 snaps; a
  1.00 recruit opens at +0.42, a 0.80 at +0.14. Small magnitudes throughout —
  the prior whispers, as designed.
- Convergent validity: recruiting `stars` vs `rating` Spearman rho = +0.9191
  (n = 397 linked skill rows) — internally consistent, same-source, no
  external-validity claim.
- N0 sensitivity appendix: at 300 snaps the prior weight is 0.33 (N0=150),
  0.50 (N0=300), 0.67 (N0=600); at 1,200 snaps 0.11 / 0.20 / 0.33. Curves
  reported, nothing selected.

What this implies, before what is wrong with it: the mechanical prerequisite
for a recruiting-prior feature now exists as a frozen, reproducible artifact
with stated uncertainty. What is wrong with it: the slope is unresolved, R²
is under half a percent, the total (not per-snap) construct conflates
opportunity with efficiency, N0 is a placeholder, and the map is fit on the
same population that gated it. The independent check — wiring plus an ATS
evaluation on unspent windows — is Stage 4, queued separately. No registry
entry was written (a fitted parameter is not an effect).
