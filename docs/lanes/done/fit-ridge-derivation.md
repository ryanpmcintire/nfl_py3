# fit-ridge-derivation

## Goal
Audit every numeric constant in the served four-term pick-probability fit
(`src/nfl_ats/pick_probability_fit.py`, `src/nfl_ats/pick_probability.py`) for
provenance, and derive `FIT_RIDGE` out-of-sample via nested leave-one-season-out
(LOSO) selection, graded against the served constant on held-out data. Read-only:
no `src/` edits, no registry writes, no commits.

## State (complete, read-only)
Constant provenance table (all `read` from `src/nfl_ats/pick_probability*.py`;
git history via `git log --diff-filter=A -S"<name>"`):

| constant | value | file | first added | derivation evidence |
|---|---|---|---|---|
| `FIT_RIDGE` | `1e-3` | pick_probability_fit.py:47 | commit `6131b6d` (2026-09-15, "Week 1 is 9-7 everywhere...", bundles MKT-16/17/18/19) | none found — bundled default, no grid search or note in commit body or docs/lanes |
| `FIT_ITERATIONS` | `50` | pick_probability_fit.py:48 | same commit `6131b6d` | none found — fixed Newton-step count, no convergence study |
| `PROBABILITY_EPSILON` | `1e-6` | pick_probability.py:93 | same commit `6131b6d` | none found — standard numerical clip, not fit to data |
| `CONFIDENCE_BAND_EDGES` | `(0.50,0.52,0.55,0.58,0.62,1.0)` | pick_probability.py:91 | same commit `6131b6d` | none found — no data-derived quantile note |
| `STRENGTH_BAND_QUANTILES` | `(1/3, 2/3)` | pick_probability.py:89 | same commit `6131b6d` | tertile split by construction, not a fitted quantity |
| `STRENGTH_ROUNDING_PLACES` | `3` | pick_probability.py:90 | same commit `6131b6d` | display rounding, not a modeling choice |
| composition-flag weights | equal (each flag is $\{-1,0,1\}$, unweighted sum, `pick_probability.py:621-623`) | pick_probability.py | same commit | **confirmed equal-weighted** by reading `signed_composition_flags`: `flags[FLAG_SUM_COLUMN] = flags[COUNTED_FLAG_COLUMNS].sum(axis=1)`, no per-flag coefficient |

All of the above landed in one bundled commit with no logged sweep; **all are
underived constants** per AGENTS.md ("a constant derived from games it is
scored on never reaches `src/`" — these were never derived from games at all,
in-sample or out).

### FIT_RIDGE nested LOSO derivation — measured this session
Script: `scripts/fit_ridge_derivation.py` (ruff-clean). Predeclared grid
`RIDGE_GRID = np.logspace(-4, 2, 13)` (13 points, 1e-4..1e2). For each outer
LOSO season fold, ridge is chosen by inner LOSO over training seasons only
(pooled log-loss criterion), never touching the outer test season. Graded vs
the served fixed `FIT_RIDGE=1e-3` fit on the identical outer folds.

Artifacts: `artifacts/fit_ridge_derivation/20260923T210622Z/` (`report.json`,
`served_population_predictions.parquet`, `extended_population_predictions.parquet`).

**Dataset A — served population, `build_fit_population()`, 2020-2025, 1,503 games, 6 seasons** (real production call, not a fixture):
- chosen ridge per fold: 2020→31.6, 2021→10.0, 2022→31.6, 2023→31.6, 2024→31.6, 2025→10.0 (all far above 1e-3)
- nested: record 859-644, accuracy 0.57152, Brier 0.245407, log loss 0.683948
- served (1e-3): record 859-644 (**identical picks**), accuracy 0.57152, Brier 0.245388, log loss 0.683952
- gap: log loss/Brier differ by ~2e-5; served is marginally better

**Dataset B — extended population, `artifacts/extended_fit_population/20260923T205910Z/population.parquet`, 2011-2025, 3,734 games, 15 seasons**:
- chosen ridge per fold: 100.0 (grid ceiling) in every one of 15 folds — inner CV always prefers maximal shrinkage
- nested: record 1990-1744, accuracy 0.532941, Brier 0.248581, log loss 0.690386
- served (1e-3): record 1994-1740, accuracy 0.534012, Brier 0.248598, log loss 0.690453
- **measured** paired effect (nested minus served accuracy, season-block bootstrap, 20,000 reps, 15 blocks): −0.001071 accuracy points, SE 0.001095, 95% CI [−0.003415, 0.000829], probability_positive (nested beats served) = 0.134

**Conclusion (stated plainly):** `FIT_RIDGE=1e-3` is underived (no logged
search) but empirically indistinguishable from — and on the larger 2011-2025
population, slightly better than — the nested-LOSO-selected ridge. The
interval on the nested-vs-served gap crosses zero: `unresolved_below_power`,
not evidence to change the served constant. The nested inner CV pushing to
the grid ceiling (heavy shrinkage) with zero practical effect on picks is
itself informative: this 4-feature standardized logit is not ridge-sensitive
in this regime, so the underived default is doing no measurable harm. No `src/`
change is implied by this evidence.

## Tried
- `git log --diff-filter=A -S"<const>"` for all six named constants — all trace to one bundled commit, no derivation notes.
- Attempted `build_fit_population(artifacts_root, data_root)` live (real repo data, no mocks) — succeeded, 1,503 graded 2020-2025 games.
- Loaded `extended_fit_population/20260923T205910Z/population.parquet` (3,734 games 2011-2025) as the second, better-powered grading population.
- Nested LOSO ridge selection implemented by reusing served fit primitives (`_fit_logit`, `_design`, `_standardisers`, `_predict`, `_confidence_bands`) imported from `nfl_ats.pick_probability_fit` — no reimplementation drift from the served math.
- `ruff check --fix scripts/fit_ridge_derivation.py` — clean.

## Next (for the root orchestrator only — not run this session)
Record the FIT_RIDGE-sensitivity result as an unresolved modeling signal:

```
uv run nfl-ats weak-signals record \
  --name fit_ridge_nested_loso_vs_served \
  --description "Nested leave-one-season-out ridge selection for the four-term pick-probability fit vs the served fixed FIT_RIDGE=1e-3, graded on 2011-2025 extended population (3734 games) and served 2020-2025 population (1503 games)" \
  --source artifacts/fit_ridge_derivation/20260923T210622Z/report.json \
  --effect -0.001071 \
  --effect-units accuracy_points \
  --classification unresolved_below_power \
  --classification-evidence "95% season-block bootstrap CI [-0.003415, 0.000829] crosses zero; probability_positive(nested beats served)=0.134; N=3734 games, 15 season blocks" \
  --league nfl \
  --season-start 2011 --season-end 2025 \
  --standard-error 0.001095 \
  --interval-low -0.003415 --interval-high 0.000829 \
  --probability-positive 0.134 \
  --sample-games 3734 --sample-blocks 15 \
  --category modeling \
  --plain-summary "We checked whether picking the ridge penalty by held-out seasons instead of using the fixed value would sharpen the model's picks; it does not — the current fixed value performs the same or slightly better." \
  --notes "FIT_RIDGE=1e-3 and its sibling constants (FIT_ITERATIONS, PROBABILITY_EPSILON, CONFIDENCE_BAND_EDGES, STRENGTH_BAND_QUANTILES, equal composition-flag weights) are all underived, all landed in commit 6131b6d with no logged search. This result does not justify changing FIT_RIDGE in src/."
```

Root should also decide whether the other five underived constants
(`FIT_ITERATIONS`, `PROBABILITY_EPSILON`, `CONFIDENCE_BAND_EDGES`,
`STRENGTH_BAND_QUANTILES`, equal flag weights) warrant their own derivation
lanes; none were touched by this session beyond provenance lookup.

## Open (Unit 1)
- `--reliability` (split-half) was not computed for the recorded signal above;
  add via the `signal_atlas` block-bootstrap tooling referenced in
  `docs/lanes/lead59-archive-battery.md` if the root wants a terminal-quality
  record instead of `unresolved_below_power`.
- Grid ceiling (100.0) was selected in every Dataset-B fold; if the root ever
  wants a genuine "does heavier ridge help" answer, extend `RIDGE_GRID` past
  1e2 — not done here since it made zero practical difference to picks.

## Unit 2 — reader-facing band constants (this session, read-only)

### Where each constant actually reaches readers — measured
- **Only `STRENGTH_BAND_QUANTILES=(1/3,2/3)` reaches the reader-facing card/board.**
  It sets `StrengthBand.minimum` for "lean"/"strong" in the active pick-probability
  artifact -> `public_board.py:740` builds `StrengthBands(lean_min=model.lean_minimum,
  strong_min=model.strong_minimum)` -> `board_content.py:3507-3508,3615,3688,3816`
  renders the Slight/Lean/Strong word and the sentence "Slight, Lean and Strong
  split the chances into thirds of the card's own record" (`board_content.py:3439-3440`)
  plus `landing_rate_sentence()` ("strong estimates won X%, slight estimates
  won Y%", `pick_probability.py:962-979`, wired via `publishing.py:140`).
- **`CONFIDENCE_BAND_EDGES=(0.50,0.52,0.55,0.58,0.62,1.0)` does not reach readers
  at all.** `grep CONFIDENCE_BAND_EDGES src/` hits only `pick_probability.py`
  (definition) and `pick_probability_fit.py:306` (`_confidence_bands`, which
  only feeds `cli_commands/prediction.py:431`, the `fit-pick-probability` CLI's
  operator-facing JSON printout). Correcting the lane's Unit-2 framing: there is
  one reader-facing band constant, not two.

### Predeclared band meaning (written before deriving, then derived)
Slight/lean/strong split picked-side held-out probability into equal-count
thirds; success = held-out cover rate monotone strong>lean>slight, and bands
distinguishable (season-block bootstrap interval on strong-minus-slight excludes
zero). Edges derived properly nested LOSO: for each held-out test season, tertile
edges come only from the pooled held-out probabilities of the *other* seasons,
never the test season itself — this differs from the currently-served procedure,
which pools all seasons' held-out probabilities together to pick one fixed edge
pair (not renested per fold).

### Measured — script `scripts/confidence_band_derivation.py` (ruff-clean)
Artifacts: `artifacts/confidence_band_derivation/20260923T211436Z/` (`report.json`,
`served_loso_predictions.parquet`, `extended_loso_predictions.parquet`). Bootstrap:
season-block, 20,000 reps.

**Dataset A — served population 2020-2025, 1,503 games, 6 seasons:**
| band | served games/rate | nested games/rate | nested 95% CI |
|---|---|---|---|
| slight | 497 / 55.5% | 495 / 55.6% | [51.0%, 59.7%] |
| lean | 501 / 55.7% | 493 / 55.8% | [53.3%, 58.6%] |
| strong | 505 / 60.2% | 515 / 60.0% | [57.0%, 63.9%] |

Nested served edges essentially match the pooled served edges (lean~0.53,
strong~0.564-0.568 across all 6 folds). strong-minus-slight: mean +4.4pp, 95%
CI **[-2.1pp, +12.4pp]** (crosses zero), probability_positive=0.865:
`unresolved_below_power`. slight vs. lean are barely separated (55.6% vs 55.8%).

**Dataset B — extended population 2011-2025, 3,734 games, 15 seasons:**
| band | served games/rate | nested games/rate | nested 95% CI |
|---|---|---|---|
| slight | 1239 / 51.9% | 1215 / 51.7% | [48.1%, 54.8%] |
| lean | 1224 / 52.4% | 1235 / 52.2% | [49.4%, 55.0%] |
| strong | 1271 / 55.9% | 1284 / 56.2% | [53.5%, 58.4%] |

Nested edges are stable across all 15 folds (lean 0.516-0.517, strong
0.534-0.536), matching the pooled served edges (0.517/0.535). Bands are
monotone non-decreasing (51.7% -> 52.2% -> 56.2%). strong-minus-slight: mean
+4.4pp, 95% CI **[+1.0pp, +8.2pp]** — interval fully positive, probability_positive=0.995.
Middle band (lean) still sits close to slight (52.2% vs 51.7%): nearly all the
separation is between "strong" and the bottom two-thirds, not evenly spread
across three bands. (Open: strong-vs-lean and lean-vs-slight diffs were not
separately bootstrapped — only strong-vs-slight was; point estimates suggest
lean adds little.)

### FIT_ITERATIONS convergence — measured
Newton's method converges (relative step norm < 1e-6) by **iteration 4** on
both datasets; iterations 46-50 show relative steps ~1e-16 (floating-point
noise only). `FIT_ITERATIONS=50` is a large overprovision with no convergence
risk — harmless but ~12x more than needed.

### PROBABILITY_EPSILON binding — measured
Dataset A (2020-2025): raw `home_cover_probability_at_open` ranges
[0.2306, 0.6905] — epsilon (1e-6) never binds (0 of 1,503 games). Dataset B
(2011-2025): 1 of 3,734 games (0.03%) has its raw base-model probability
clipped to exactly the `PROBABILITY_EPSILON` boundary (`model_logit` hits
-13.8155 = logit(1e-6) exactly) — an extreme near-certain call outside the
current 2020-2025 served window. Epsilon does its job on that one game; no
evidence it needs to change.

### CONFIDENCE_BAND_EDGES monotonicity (diagnostic-only, not reader-facing)
Neither dataset's hand-picked bins (0.50/0.52/0.55/0.58/0.62) are strictly
monotone: Dataset A accuracy by band = 57.2%(n=313), 55.2%(n=516), 55.6%(n=297),
61.5%(n=247), 60.0%(n=130) — dips twice. Dataset B = 52.5%(n=1544), 52.4%(n=1469),
56.8%(n=518), 57.7%(n=142), 62.3%(n=61) — one near-flat (0.04pp) inversion
between the first two bins, otherwise increasing. Since this never reaches a
reader, no reader-facing consequence; flagging only because the root asked to
check it.

### Conclusion (stated plainly)
The one reader-facing band constant (`STRENGTH_BAND_QUANTILES`, tertile split)
already behaves like a nested-LOSO derivation in practice — pooled served edges
and per-fold nested edges are nearly identical on both datasets — so re-deriving
it changes nothing about which games say "strong" vs "slight." What the larger,
better-powered 2011-2025 population adds is a genuine positive resolution: the
strong-vs-slight held-out cover-rate gap is real (whole 95% interval positive),
but the middle "lean" band carries little independent separation from "slight."
`CONFIDENCE_BAND_EDGES` is confirmed dead code for readers — it only feeds a
CLI diagnostic — so its non-monotonicity has no card/board consequence.
`FIT_ITERATIONS=50` and `PROBABILITY_EPSILON=1e-6` are both doing no harm
(no convergence risk; epsilon binds on a vanishingly small, out-of-window
fraction of games). No `src/` change is implied by this evidence.

## Next (Unit 2, for the root orchestrator only — not run this session)
Record the strength-band separation result as an unresolved-but-directionally-
positive modeling signal (root should confirm/adjust `--classification` against
the full `weak-signals` vocabulary; this session only reused the enum value
seen in Unit 1):

```
uv run nfl-ats weak-signals record \
  --name strength_band_tertile_nested_loso_separation \
  --description "Nested leave-one-season-out tertile band edges for the reader-facing Slight/Lean/Strong pick-probability words (STRENGTH_BAND_QUANTILES=(1/3,2/3)), graded on 2011-2025 extended population (3734 games) and served 2020-2025 population (1503 games)" \
  --source artifacts/confidence_band_derivation/20260923T211436Z/report.json \
  --effect 0.0444 \
  --effect-units accuracy_points \
  --classification unresolved_below_power \
  --classification-evidence "2011-2025 (3734 games, 15 season blocks): season-block bootstrap 95% CI on strong-minus-slight held-out cover rate [0.0103, 0.0823], fully positive, probability_positive=0.995; but 2020-2025 (1503 games, 6 blocks) CI [-0.0215, 0.1244] crosses zero, probability_positive=0.865, so the served 6-season window alone is underpowered even though the larger population resolves positive" \
  --league nfl \
  --season-start 2011 --season-end 2025 \
  --standard-error 0.0184 \
  --interval-low 0.0103 --interval-high 0.0823 \
  --probability-positive 0.995 \
  --sample-games 3734 --sample-blocks 15 \
  --category modeling \
  --plain-summary "We checked whether the board's Slight/Lean/Strong wording actually tracks how often those picks win. On the bigger 15-season history it does for the top group: Strong picks clearly win more than Slight picks. The current fixed cutoffs already match what a season-by-season re-derivation would produce, so no wording change is needed; but the middle 'Lean' group barely differs from 'Slight' in win rate." \
  --notes "CONFIDENCE_BAND_EDGES (0.50,0.52,0.55,0.58,0.62,1.0) was also checked and confirmed NOT reader-facing (CLI diagnostic only, cli_commands/prediction.py); its bands are not strictly monotone in either dataset but this has no card/board consequence. FIT_ITERATIONS=50 converges by iteration 4 (12x overprovisioned, harmless). PROBABILITY_EPSILON=1e-6 never binds in the 2020-2025 served window; binds on 1 of 3734 games (0.03%) in the larger 2011-2025 population, correctly clipping an extreme near-certain call."
```

Root should also decide whether `FIT_ITERATIONS` and `PROBABILITY_EPSILON`
warrant any `src/` comment/constant change given they were confirmed harmless
(this session recommends no change), and whether the never-reader-facing
`CONFIDENCE_BAND_EDGES` diagnostic is worth keeping, simplifying, or removing
from the CLI output given it does not gate any served decision.

## Open (Unit 2)
- strong-vs-lean and lean-vs-slight band-pair diffs were not separately
  bootstrapped (only strong-vs-slight); add if the root wants a full pairwise
  distinguishability table.
- `equal composition-flag weights` (the sixth underived constant from the
  provenance table) was not touched by either unit; still open for a future
  lane if the root wants it derived/tested.
