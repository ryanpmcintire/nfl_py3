# Hierarchical APM: unit partial pooling vs flat (predeclaration)

**Status:** predeclared 2026-09-03, BEFORE any hierarchical-vs-flat
comparison number was computed. Sections 0–8 are frozen; section 9
(Results) is appended after the run and nothing above it is edited
afterwards.

**Owning work package:** PER-09 (latent player ratings: hierarchy remains
after the special-teams and unit slices). Files: this document,
`scripts/hierarchical_apm_screen.py`,
`tests/test_hierarchical_apm_screen.py`,
`artifacts/hierarchical_apm/`.

**Parents:** `src/nfl_ats/participation.py` (pooled APM recipe),
`docs/unit_apm_ratings.md` (unit fits), `docs/st_player_ratings.md`.

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

This slice compares EPA-prediction error, not ATS picks: no rotation window
is spent or needed. A verdict here concerns the estimator only and can never
promote a feature.

---

## 1. What this slice asks, and what it does not

Whether partially pooling player APM coefficients toward their roster-unit
means predicts held-out play EPA better than the flat pooled fit — the
"hierarchy" half of PER-09's remainder. It does NOT score ATS, build
season-lagged ratings, or wire anything.

## 2. Population and task (frozen)

Same valid competitive 11-on-11 scrimmage table as the pooled screen
(`build_participation_play_table`), seasons 2019–2024. Task: predict
held-out play EPA (clipped ±5.0, the frozen clip).

## 3. Arms (frozen)

- **Flat:** the pooled recipe verbatim — Ridge alpha 1000, team effects at
  scale 11.0, one coefficient per participant (offense +1 / defense −1).
- **Hierarchical:** the flat fit followed by one fixed empirical-Bayes
  step — each player's coefficient shrunk toward their unit's player-mean
  with weight `n / (n + k)`, `k = 500` plays (the frozen reliability-prior
  scale, NOT tuned here), unit from the frozen
  `scripts/unit_apm_screen.py` roster mapping (same modal rule). Team
  effects and intercept pass through untouched.

Expanding-origin evaluation: for each held-out season Y in {2022, 2023,
2024}, fit both arms on seasons [Y−3, Y−1] and score mean squared error on
Y's plays. Metric: pooled MSE difference (flat − hierarchical) over the
three held-out seasons, positive favors hierarchy, with a play-level
paired bootstrap (2,000 samples, seed 20260903) for the interval.

## 4. Reliability context (frozen, no re-measurement)

Unit traits are already gated real (SB 0.24–0.44); this slice does not
re-litigate their existence. A null MSE result means hierarchy adds no
predictive value at k=500, not that units are noise.

## 5. Positive control (frozen, diagnostic only)

A deliberately-leaky variant (fit includes the held-out season) must beat
both arms decisively or the harness is blind and the run is void, not
negative.

## 6. Leakage (frozen)

Fits see only strictly-prior seasons (the [Y−3, Y−1] window is enforced by
construction, asserted fail-closed). No pregame application exists in this
slice. Tests pin the window enforcement and the unit-mean shrinkage algebra
on synthetic frames.

## 7. Test contract (release-blocking)

`tests/test_hierarchical_apm_screen.py` covers, without network access:
shrinkage algebra (k=0 → flat, k→∞ → unit mean, missing unit → flat
 passthrough), window enforcement (a future-season row can never enter a
fit), determinism, and the paired-MSE comparison helper.

## 8. Decision rule (frozen)

This slice records NOTHING to the weak-signal registry: none of the CLI's
effect units (`ats_points`, `accuracy_points`, `brier`, `log_loss`, `mae`,
`correlation`, and the three `*_improvement` variants) honestly contains an
MSE delta, and forcing one in would corrupt the pool (the same unit-misuse
rule Stage 1's doc states for correlations). The measurement lives in §9,
the artifact, and the row, with units stated plainly as MSE points. A
future ATS look on open windows carries its own record under its own units.

## 9. Results (added after the run, 2026-09-03)

Measured by `scripts/hierarchical_apm_screen.py` in one run (artifact
`artifacts/hierarchical_apm/20260903T202953Z/results.json`):

| Held-out | Train/test plays | MSE flat | MSE hierarchical | Delta | 95% CI | P(hier better) |
|---|---|---|---|---|---|---|
| 2022 | 80,810 / 29,054 | 1.704976 | 1.703416 | +0.001559 | [+0.000727, +0.002439] | 1.0000 |
| 2023 | 84,685 / 28,670 | 1.727162 | 1.725249 | +0.001913 | [+0.001022, +0.002845] | 1.0000 |
| 2024 | 85,972 / 28,241 | 1.713056 | 1.711150 | +0.001906 | [+0.001077, +0.002748] | 1.0000 |

Pooled delta +0.001793. The leaky control (1.674975 / 1.698864 / 1.683728)
beats both arms decisively every season, so the harness is valid and the
run stands.

What this implies for the decision, before what is wrong with it: unit
partial pooling at the frozen k=500 predicts held-out play EPA better than
the flat fit in all three seasons with intervals excluding zero — small in
absolute terms (~0.1% of MSE), but consistent in direction, which is what
the predeclaration asked. The hierarchical estimator is adopted for the
next season-lagged builder. What is wrong with it: k was fixed, not
selected (any k-tuning now would be window shopping on these same seasons);
the gain is small enough that its ATS consequence is unproven and untested
(rotation pools exhausted). Per frozen §8: no registry entry (no honest
unit for an MSE delta), no window, no wiring.
