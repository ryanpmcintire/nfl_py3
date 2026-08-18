# Purged, embargoed cross-validation

Owner: this document, `src/nfl_ats/purged_cv.py`, `tests/test_purged_cv.py`,
and `scripts/purged_validate.py`. Empirical work is CFB-only (free,
unlimited, rotation-registry rule 8); nothing here reads or writes
`registry/rotation_registry.json`, and no NFL model, feature profile, or pick
moves as a result of this document.

## Why this exists

`walk_forward_backtest` and `cfb_walk_forward_benchmark` are the leak-safe
evaluators of record: predict week `t` using only games strictly before it,
refit, advance. That is correct and nothing here replaces it. But it extracts
exactly ONE test observation per game in exactly ONE ordering of history, and
every early season is spent purely as training with zero test credit — a
2007 week can never be scored, because there are not yet 500 prior games to
train on.

**Purged, embargoed cross-validation** (Lopez de Prado's combinatorial purged
CV) is the supplementary instrument this document adds: partition the
timeline into many blocks, let ANY block serve as a test block (using both
earlier AND later blocks as training), and delete ("purge") every training
observation whose feature window could overlap the test block, plus an
additional forward-only safety buffer ("embargo"). This is a research
instrument for detecting whether an effect exists at all and how precisely it
can be measured — it is not a new way to generate or grade a live pick.

## 1. Leakage channels, enumerated and measured

The frozen CFB base benchmark (`CFB_MODEL_FEATURE_COLUMNS`: market, context,
experience, and span-8 EWMA team-state columns only — no opponent-adjusted or
graph features) has exactly one rolling-window feature family. Every channel
below was checked against the actual source, not assumed.

| # | Channel | In frozen CFB base benchmark? | Measured contamination span |
|---|---|---|---|
| 1 | Team-form EWMA (`cfb_features.build_cfb_team_states`, span 8, NFL params verbatim) | **Yes — the only one** | 12 weeks to fall below 5% weight, 19 weeks to fall below 1% (closed-form `(1-alpha)**k`, `alpha=2/9`; cross-checked against a literal EWMA recursion in `test_ewma_retained_weight_matches_direct_recursion`) |
| 2 | Trailing residual/calibration sample (`fit_margin_model`/`fit_cfb_residual_model`'s `distribution_fraction=0.20` trailing-by-time split within a fold's training set) | Yes (used by every method) | Not a fixed window — a proportional split. Empirically shown (§3, positive control) to inject a fold-to-fold, sign-unstable location bias of **0.9–1.3 accuracy-relative points**, the same order of magnitude as the smallest effects this project tracks. Not reduced by purge width; see the raw-sign ablation below. |
| 3 | Opponent adjustment (`fit_opponent_effects`, weighted ridge, half-life 16 weeks) | **No** | 69 weeks (5%) / 106 weeks (1%) via `half_life_contamination_weeks` (`0.5**(age/half_life)`) |
| 4 | Schedule-strength graph ratings (`GraphRatingConfig`, half-life 8 weeks) | **No** | 35 weeks (5%) / 53 weeks (1%), same formula |
| 5 | Season-level / team-season effects (offseason retention 0.67; per-season league mean used only for the offseason regression) | Part of channel 1's construction | The league mean for season *S* is only applied when transitioning INTO season *S+1*, after *S* is fully complete — checked directly against `build_cfb_team_states`; not a separate leak. Already covered by channel 1's purge width for any realistic 1–2 season gap; offseason retention (0.67/season) decays further on top. |
| 6 | The market line itself (`spread_line`) | Yes (feature and target-construction input) | Not purgeable in the windowed sense: each row's `spread_line`/`ats_margin` is self-contained (no lookback), so it cannot leak a SPECIFIC other game's answer. But a training row dated after a test block was quoted with full public knowledge of that block's real outcomes — a conceptual caveat for anyone using purged CV to argue about market efficiency specifically, not a mechanical accuracy leak. |

Channels 3–4 are declared and guarded against drift
(`test_declared_half_lives_match_source`) but do not feed the defaults below,
since neither feature family is in the benchmark being validated.

## 2. The scheme and its defaults

`purged_embargoed_folds` partitions the sorted unique (season, week) pairs
into `n_blocks` contiguous, near-equal groups. Every
`test_group_size`-combination of blocks is one path's test set (`k=1` —
leave-one-block-out — is the default and what every headline number below
uses; `k>1` generates the full combinatorial family and is exercised in
`test_combinatorial_test_group_size_generates_expected_path_count`, but not
run exhaustively at full scale here — see §5). For each path: purge removes
training rows within `purge_weeks` of EITHER edge of any selected block;
embargo removes an additional `embargo_weeks` strictly AFTER a selected
block's end.

```
DEFAULT_PURGE_WEEKS   = ewma_contamination_games(span=8, threshold=0.05) = 12
DEFAULT_EMBARGO_WEEKS = ewma_contamination_games(span=8, threshold=0.01) - 12 = 7
```

Both are computed, not typed in (`test_default_purge_and_embargo_are_derived_not_hardcoded`
pins this). `purged_cv_backtest` refits the FROZEN, unmodified
`fit_cfb_residual_model`/`fit_market_baseline` on each fold and scores the
held-out block(s), never touching the primary walk-forward evaluator.

## 3. Does it leak? Three checks, all run on the full ~12,500-game CFB table

### Headline sanity check (real data)

| | n tested | accuracy | 95% week-blocked CI |
|---|---|---|---|
| Walk-forward (`cfb_walk_forward_benchmark`, reference run) | 11,780 / 12,500 (94.2%) | 51.66% | [50.76%, 52.67%] |
| Purged CV (defaults, k=1, 294 folds) | 12,283 / 12,500 (98.3%) | 51.22% | [50.42%, 52.01%] |

Purged CV's point estimate is not inflated relative to the reference
evaluator — if anything lower — on the same frozen model.

### Negative control 1 — global permutation (`permute_target`)

Permutes `ats_margin` itself (not `result`) so the row-to-outcome pairing is
destroyed while every feature, including the real team-state trajectory,
stays at its true value. **A first draft permuted `result` instead and kept
`spread_line` fixed, which bakes a deterministic `-spread_line` term into the
"permuted" target — `spread_line` is itself a model feature, so that
construction handed the model a real, exploitable relationship. A smoke run
showed ~68% forced-pick accuracy on "permuted" data, which is what caught
it** before any real experiment ran on it. Fixed version, 40 replicates per
condition, n_blocks=20:

| Condition | mean accuracy | false-positive rate (95% CI excludes 50%) |
|---|---|---|
| Measured purge/embargo (12/7 wk) | 50.11% | 10.0% (4/40) |
| **Zero purge/embargo** | 50.09% | 12.5% (5/40) |

Both correctly centered at 50%. The difference in false-positive rate
(4 vs. 5 out of 40) is noise, not a break.

### Negative control 2 — team-persistent null (`team_persistent_null`)

A global permutation destroys every team-persistent structure, so it cannot
distinguish a leaky split from a clean one even in principle: the
shared-team feature-proximity channel (§1, channel 1) can only inflate
results when the target has real persistent structure to leak. This control
draws one latent quality per (team, season) — independent of every real
feature, so true population accuracy is still exactly 50% — and derives
`ats_margin = quality[home] - quality[away] + noise`. 30 replicates per
condition, n_blocks=20:

| Condition | mean accuracy | false-positive rate |
|---|---|---|
| Measured purge/embargo (12/7 wk) | 50.20% | 16.7% (5/30) |
| **Zero purge/embargo** | 50.32% | 20.0% (6/30) |

### Leak-sensitivity verdict — reported honestly, not massaged

**Neither negative control broke at zero purge/embargo.** Both stayed
correctly centered at 50% and the false-positive-rate gap between measured
and zero purge (1 extra replicate out of 30–40 in each case) is well inside
binomial sampling noise. Per this project's own rule — if it does not break,
say so rather than declare success — that is the honest result: across two
different null constructions, one specifically engineered to carry the exact
persistent structure the identified channel could exploit, deliberately
disabling purge/embargo did not produce a reproducible failure in this
pipeline's frozen CFB base-benchmark configuration.

This does not retroactively prove no leak is possible for every feature
profile — channels 3–4 (opponent adjustment, graph ratings) were not
exercised, and a stronger team-persistence setting or a different feature
profile could behave differently. It does mean: for the ONE configuration
this document validates, purge/embargo's measured defaults are the
principled, derived choice (§1), but the specific stress tests run here could
not make the alternative (zero purge) fail differently from it. The
(unrelated, pre-existing) elevated false-positive rate — 10–20% against a
5% nominal target in every condition, purged and unpurged alike — tracks
with the reduced bootstrap resolution used for speed in the validation script
(200 samples per replicate vs. the 500–1,000 used for headline numbers), not
with purge width; it appears equally on both sides of every comparison.

### Positive control — a genuine, separate finding

`inject_synthetic_signal` plants a `{-1,+1}` feature with a KNOWN population
forced-pick accuracy (`target_accuracy = Phi(beta/noise_std)`), calibrated
with `noise_std` = the frame's OWN real `ats_margin` standard deviation
(~15.4), matching the scale `ridge_alpha=10` is actually tuned for.

| Target | Population truth (realized) | Full pipeline (smoothed probability) | Sign-only ablation (`sign(predicted_margin - spread_line)`) |
|---|---|---|---|
| 51.3% (1.3-pt, injury scale) | 51.67% | **49.33%** (CI [48.4%, 50.1%] — wrong sign) | 50.67% |
| 53.0% (3-pt, clear signal) | 53.34% | 51.14% (under-recovered) | **53.01%** (matches population truth) |

The full pipeline under-recovers, and at the smallest scale gets the sign
wrong. Diagnosis: this is **not** the purge/embargo split. The sign-only
ablation — which uses the exact same purged/embargoed training set and ridge
fit, but skips the out-of-time residual/calibration step entirely — tracks
the true planted magnitude closely at both scales. The distortion traces to
channel 2 (§1): the residual/calibration sample's median shifted by
0.9–1.3 points across individual folds in one worked example (10 folds,
n_blocks=10), unstable in direction fold to fold, driven by the ridge fit
absorbing per-fold noise across ~33 real-but-irrelevant covariates. On real
data, where the true effect (~2 points) dominates this noise, the gap between
the sign-only and full estimates is only 0.1 point (51.31% vs. 51.22%) — so
the headline number above is not meaningfully affected. **But any future
claim near the 1–1.3 point scale specifically should be cross-checked against
the sign-only estimator, or the calibration mechanism revisited — this is an
open finding, not a resolved one.**

## 4. Quantifying the prize

| | Walk-forward | Purged CV (k=1, defaults) |
|---|---|---|
| Test games (pooled, all windows) | 11,780 / 12,500 (94.2%) | 12,283 / 12,500 (98.3%) |
| Test games, thin 2006–2011 era | 2,393 | 2,904 (+21%) |
| Test games, clean core (2012–2019, 2021–2025) | 8,933 | 8,933 (identical — already fully covered) |
| SE, pooled "all" (week-blocked bootstrap) | 0.488 pt | 0.406 pt |
| SE, clean-core only (same 8,933 games both sides) | 0.494 pt | 0.475 pt |

Effective-sample multiplier from the SE ratio on the pooled window:
`(0.488 / 0.406)^2 ≈ 1.44x` — **equivalent to about 1.4x more data**, for
estimation precision, not fresh games. Within the clean-core window alone
(identical test games on both sides), the multiplier is a smaller
`(0.494/0.475)^2 ≈ 1.08x`, isolating the part of the gain that comes purely
from each fold training on both earlier and later data rather than from new
test coverage. Most of the overall 1.44x comes from finally scoring the thin
2006–2011 era, which the walk-forward's 500-game training floor excludes
entirely.

Minimum detectable effect (two-sided, 95% CI, 80% power convention,
`MDE = 2.80 * SE`):

- Walk-forward: `2.80 * 0.488 ≈ 1.37 points`
- Purged CV: `2.80 * 0.406 ≈ 1.14 points`

**Could this resolve a 1.3-point effect on data already available?**
On CFB's own 12,500-game corpus: **yes** — purged CV's directly measured MDE
(~1.14 pt) clears 1.3 points; the walk-forward's own measured MDE (~1.37 pt)
does not. Applying the SAME ~1.2x SE-improvement ratio to the NFL
evaluator's own stated ~2-point resolution (`AGENTS.md`) would put an
NFL purged-CV MDE at roughly **1.6–1.7 points — still short of 1.3**. So the
honest answer for the dataset this project actually bets on is **not yet**:
this technique would need to be run against the real, much smaller NFL
corpus (explicitly out of scope here — no NFL rotation window was spent) to
get a real answer rather than an analogy.

## 5. What this does NOT license

- **Not fresh evidence.** Purged CV reuses the same ~12,500 CFB games across
  up to hundreds of folds. It sharpens the ESTIMATE of an effect already
  present in this data; it does not create new games, and it does not count
  toward the rotation registry's fresh-data bookkeeping.
- **Does not touch the multiple-comparisons problem.** The project has run
  roughly 130–150 looks at this data across its history; purged CV is one
  more instrument on the same pile, not a correction for how many times the
  pile has been looked at.
- **Not a change to how picks are made.** `walk_forward_backtest` and
  `cfb_walk_forward_benchmark` remain the strictly causal, single-ordering
  evaluators of record. Purged CV trains on chronologically later data than
  its test block by design — correct for measuring whether an effect exists
  in a backtest, but never a description of how a live pick may be produced.
  No model, feature profile, or `artifacts/active_ats_model.json` changed as
  part of this work.
- **Does not fix the calibration-sample noise found in §3.** The
  residual/calibration-sample instability behind the positive control's
  under-recovery is unresolved by this document; any claim near the
  1–1.3 point scale specifically should be cross-checked with the sign-only
  estimator until it is.
- **Defaults are scoped to one feature family.** `DEFAULT_PURGE_WEEKS`/
  `DEFAULT_EMBARGO_WEEKS` are derived from the span-8 team-state EWMA, the
  only rolling-window feature the frozen CFB base benchmark uses. Pointing
  `purged_cv_backtest` at the `graph` or `player_*` margin feature profiles
  requires widening purge to the much larger declared spans in §1
  (69/106 weeks, 35/53 weeks) first — using the base-benchmark defaults
  there would under-purge.
- **The leak-sensitivity check is an honest non-result, not a clean proof.**
  §3 states plainly that deliberately zeroing purge/embargo did not break
  either negative control tested. Do not cite this document as having
  "proven" the split leak-safe in the way a break-then-fix result would;
  cite it as having derived a principled purge/embargo width and having
  failed to falsify it under two stress tests, which is a different and
  weaker claim.

## Reproducing this

```
./.tools/uv.exe run --no-sync python scripts/purged_validate.py
```

Reads `data/processed/cfb_game_features.parquet` (already built, 12,500
games); writes predictions/summaries/JSON results to the caller's scratch
directory only. No artifact, registry, or tracked-doc write.
