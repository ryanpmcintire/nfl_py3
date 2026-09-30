# Shared four-term LOSO base

**Measured:** the new [upstream LOSO artifact](#upstream-loso-artifact) now refits the margin model and learned input stages. The original calibration-only artifact below remains separately documented.

**Measured:** `artifacts/loso_base/20260929T232711656489Z/` contains
`predictions.parquet` and `metadata.json`, built by
`scripts/build_loso_base.py`. All 1,503 eligible regular-season opener games from
2020–2025 have a prediction from a four-term fit trained on the other five seasons.
Runtime checks found zero shared training/prediction seasons and zero missing
eligible predictions. The source opener evaluation has 1,537 games; its 34 pushes
are excluded by the served fitter's conditional non-push target.

**Read:** the served fitting implementation is
`src/nfl_ats/pick_probability_fit.py:65` (optimizer), `:81` (design), `:88`
(training standardization), and `:387` (LOSO evaluation). Its frozen input table
is `artifacts/pick_probability/20260929T192747Z/per_game.parquet`, linked to
`artifacts/opener_evaluation/20260929T192743Z`. The builder uses those exact cached
inputs, fits six models independently with ridge 0.001 and 50 optimizer steps,
and writes only a research artifact. The active model is `b578fbea1c5c706f`.

## Scope of the guarantee

**Measured:** independent refitting reproduces the served
`out_of_season_home_probability` and every fold coefficient exactly: maximum
absolute differences are both 0. This packages and audits an existing calibration
LOSO calculation; it does not create an independent statistical comparison.

**Read:** `docs/lead89_unit1.md:65` reports 1,443 upstream training-cutoff
violations among 1,537 opener games, and `:80` identifies upstream weekly model
fits that include earlier games of the prediction season. That count was not
remeasured here. **Inferred:** this artifact does **not** clear the upstream
LEAD-83/87/89 season-holdout gate. The four-term fit excludes the prediction
season; its input model logit retains upstream training provenance. A genuinely
end-to-end season-held-out base requires new upstream forecasts and verified
opener lineage before recalibration.

**Read:** `docs/lead89_unit1.md:91` also requires stage-matched features.
These inputs use Sunday pre-kick market moves and cannot represent Tuesday or
Thursday information. Historical openers are the frozen pool-line proxy; no
pre-2026 pool captures are assumed. The probability is conditional on no push at
that opener. It is not a push probability or an alternative-line distribution.
Retrospective LOSO uses later seasons when predicting earlier seasons; selected
features are not an untouched outer test. These baseline diagnostics neither
close a signal nor authorize a serving change, under AGENTS.md's research rules.

## Rebuild

From the repository root, with the existing locked Windows environment:

```bash
UV_CACHE_DIR=tests/scratch/codex/loso_base_uv_cache .tools/uv.exe run --no-sync python scripts/build_loso_base.py --source-artifact artifacts/pick_probability/20260929T192747Z
```

PowerShell equivalent: set `$env:UV_CACHE_DIR='tests/scratch/codex/loso_base_uv_cache'`
then run the same `.tools/uv.exe` command. Without `--source-artifact`, the builder
reads the active pick-probability pointer. `--output-dir` must name a new directory
under `artifacts/loso_base`. The builder caps numerical thread pools at two,
does not rebuild upstream pipelines, and fails on changed inputs, missing eligible
games, source/model mismatch, incompatible fit terms, season overlap, or failure
to reproduce the source LOSO evaluation. It leaves serving pointers untouched.

## Consumer contract

Pin a specific directory; never silently select the newest artifact. Use the
canonical `base_home_probability` as the four-term home-cover probability.
Join on `(game_id, season)` with one-to-one validation and check coverage before
scoring. Do not fill missing predictions with 0.5 or reuse full-data fitted
coefficients on a held-out season. Fit additional terms and any thresholds only
inside the research unit's declared folds; one fitted probability selects a side.
An outer-block protocol cannot consume this retrospective LOSO artifact as though
it were trained only before that block. Whole-population calibration does not
become fold-safe merely by joining this file.

| Columns | Meaning |
| --- | --- |
| `game_id`, `season`, `week`, `game_type`, `gameday`, `home_team`, `away_team` | Frozen game identity; one row per eligible game. |
| `tue_open_home_spread` | Historical opener used for target and grading. |
| `margin_vs_open`, `home_covered` | Outcome audit fields; never candidate features. `home_covered` is binary. |
| `home_cover_probability_at_open`, `model_probability`, `model_logit` | Served discrete model input, epsilon-clipped input, and its logit. |
| `composition_flag_sum`, `market_move_toward_home`, `market_move_available` | Other three exact served fit inputs; missing moves use zero and availability 0. |
| `flag_*` | Frozen signed component flags; metadata identifies counted components through the hashed source metadata. |
| `base_home_probability`, `base_pick_home`, `base_pick_probability`, `base_correct` | LOSO probability, side at 0.5 (ties home), probability of that side, opener correctness. |
| `held_out_season`, `base_training_games`, `base_training_seasons`, `same_season_training_rows` | Fold audit; training seasons are a comma-separated string and overlap count is 0. |
| `served_evaluation_home_probability`, `market_even_home_probability` | Sanity comparators; market-even is 0.5 without a directional pick. |
| `base_probability_policy` | Discrete conditional non-push input policy. |

Example loader (the caller supplies its predeclared eligible `research_rows`):

```python
import hashlib
import json
from pathlib import Path

import pandas as pd

root = Path("artifacts/loso_base/20260929T232711656489Z")
metadata = json.loads((root / "metadata.json").read_text())
path = root / "predictions.parquet"
if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["predictions_sha256"]:
    raise ValueError("Prediction artifact hash mismatch")
base = pd.read_parquet(path)
if base.duplicated(["game_id", "season"]).any():
    raise ValueError("Duplicate base keys")
for season, fold in metadata["folds"].items():
    expected = set(metadata["seasons"]) - {int(season)}
    if set(fold["training_seasons"]) != expected:
        raise ValueError("Invalid training season set")
if not base["same_season_training_rows"].eq(0).all():
    raise ValueError("Calibration season overlap")
joined = research_rows.merge(
    base[["game_id", "season", "base_home_probability"]],
    on=["game_id", "season"],
    how="left",
    validate="one_to_one",
)
if joined["base_home_probability"].isna().any():
    raise ValueError("Research population is not covered by this base")
```

Metadata includes SHA-256 hashes of the exact cached source, opener data, metadata,
coefficients, active pointers and builder/fitter code; output Parquet hash;
per-fold natural and standardized coefficients, training means/stds, training-game
and fit-input hashes, season sets and counts; coverage; and full diagnostics.
Natural coefficient order is named explicitly; standardized arrays use intercept
then `features`. Artifact files are local and ignored by Git.

## Measured sanity results

**Measured:** the real builder command exited 0. `ruff check` and formatting passed;
the consumer loader joined all 1,503 rows and verified six folds and all input/output
hashes. Uncertainty
uses 10,000 paired whole-season bootstrap draws, seed 20260929, fixed predictions,
95% percentile intervals. With only six seasons these intervals omit refit and
feature-selection uncertainty. The frozen protocol in the lane declares 162
diagnostic looks: six fits and 156 reporting cells, including null market accuracy
cells; no tuning, selection, promotion, or registry record.

| Baseline | W–L | Accuracy %, 95% interval | Log loss, 95% interval | Brier, 95% interval |
| --- | --- | --- | --- | --- |
| Shared LOSO | 863–640 | 57.418 [56.254, 58.784] | 0.682774 [0.680279, 0.685145] | 0.244838 [0.243553, 0.246046] |
| Served LOSO evaluation | 863–640 | 57.418 [56.254, 58.784] | 0.682774 [0.680279, 0.685145] | 0.244838 [0.243553, 0.246046] |
| Raw model | 802–701 | 53.360 [51.729, 55.000] | 0.696917 [0.692602, 0.702095] | 0.251715 [0.249653, 0.254163] |
| Even market | No pick | Not applicable | 0.693147 [0.693147, 0.693147] | 0.250000 [0.250000, 0.250000] |

**Measured:** paired LOSO-minus-served accuracy gain is 0.000 percentage points
[0.000, 0.000], and log-loss improvement is 0.000000 [0.000000, 0.000000];
`probability_positive=0.5` for both. Relative to the raw model, accuracy gain is
4.059 points [2.358, 5.503], log-loss improvement 0.014143 [0.008917, 0.019726];
`probability_positive=1.0` for both. Relative to even market, log-loss improvement
is 0.010373 [0.008002, 0.012868], `probability_positive=1.0`.
These are calibration baseline diagnostics on previously selected features.

**Measured:** pooled training uses 7,515 fit/game evaluations, five appearances per
game. Training accuracy is 57.512%, log loss 0.681732, Brier 0.244350. Held-out minus
training gaps are −0.093 accuracy points, +0.001042 log loss and +0.000488 Brier.

| Held season | Training games | Held games | Intercept | Model logit | Move | Available | Composition | W–L | Accuracy % | Log loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2020 | 1283 | 220 | -0.045251 | 0.341246 | 0.220730 | 0.028113 | 0.244789 | 122–98 | 55.455 | 0.685840 |
| 2021 | 1267 | 236 | -0.062332 | 0.177916 | 0.223716 | 0.044183 | 0.271867 | 133–103 | 56.356 | 0.686447 |
| 2022 | 1255 | 248 | -0.055698 | 0.249647 | 0.222903 | 0.039674 | 0.238476 | 150–98 | 60.484 | 0.679696 |
| 2023 | 1237 | 266 | -0.038064 | 0.341480 | 0.230272 | 0.002286 | 0.246114 | 150–116 | 56.391 | 0.677980 |
| 2024 | 1237 | 266 | -0.056178 | 0.256116 | 0.246066 | 0.039146 | 0.262262 | 155–111 | 58.271 | 0.683667 |
| 2025 | 1236 | 267 | -0.054099 | 0.292328 | 0.188739 | 0.047462 | 0.295997 | 153–114 | 57.303 | 0.683748 |

**Measured:** all four slopes remain positive across the six folds; intercepts
remain negative. Model-logit slopes range 0.177916–0.341480 and availability slopes
0.002286–0.047462. No coefficient or fold was selected from this comparison.

| Home probability bin | Games | Mean prediction | Observed home-cover rate |
| --- | --- | --- | --- |
| [0, .40) | 114 | 0.357291 | 0.368421 |
| [.40, .45) | 243 | 0.425372 | 0.477366 |
| [.45, .50) | 488 | 0.475198 | 0.422131 |
| [.50, .55) | 347 | 0.526717 | 0.544669 |
| [.55, .60) | 182 | 0.569783 | 0.593407 |
| [.60, 1] | 129 | 0.643683 | 0.658915 |

**Measured:** the prediction-file SHA-256 is
`6bcf21130ef03ad3c9c522b71d61167290ec0a69a181b48d688541f16dfaeb98`.
The source fit-table SHA-256 is
`0490c806caf9e9707abe28f3e3e42f85df312084ae52d9d1588d173bba5b01e5`.
The concise run log is `tests/scratch/codex/build_loso_base.log`.

## Upstream LOSO artifact

**Measured:** `artifacts/loso_upstream/20260929T235904133782Z/` contains
`predictions.parquet` and `metadata.json`, built by
`scripts/build_loso_upstream.py`. It covers all 1,537 frozen opener games,
including 34 pushes; calibration and accuracy use the 1,503 decisive games.
The feature table SHA-256 is
`a3eedb0323818f95b029047c5b80de3ce8d458a48b82ae4489c1fb70d16d5685`,
matching the active model. The opener source is
`artifacts/opener_evaluation/20260929T192743Z/per_game.parquet`.
Metadata hashes every source, the builder, and the saved predictions.

**Measured:** six outer calibration folds exclude their prediction season.
Each calibration training row comes from an inner margin fit excluding both
that row's season and the outer season. The served margin fitter uses ridge
alpha 10, the `weak_stack` features, and its chronological 80/20 residual split.
Its training pool includes all available completed regular-season feature rows
through 2025 except the excluded seasons. The line passed to each prediction
is the frozen historical opener.

**Measured:** the discrete line-conditional reader, fitted location correction,
and learned protection quartiles also exclude the relevant seasons. Location
training points use another level of cross-fitting; 41 distinct excluded-season
sets cover these dependencies. Existing fixed flags and movement inputs were
reproduced exactly before the replay; the original protection flags were also
reproduced before refitting their thresholds. Every margin and calibration
training-season check passed, as did the discrete, location, and protection
checks. Independent saved-row verification reconstructed the four-term
probabilities from fold coefficients with maximum error 1.11e-16.

**Measured:** results below use the same decisive opener rows. Intervals are
95% season-block bootstrap intervals, 10,000 draws, seed 20260929.

| Probability | Record | Accuracy, 95% interval | Log loss, 95% interval | Brier |
| --- | --- | --- | --- | --- |
| Upstream LOSO four-term | 865–638 | 57.552% [56.223, 58.721] | 0.681819 [0.678980, 0.684150] | 0.244338 |
| Served evaluation | 863–640 | 57.418% [56.254, 58.784] | 0.682774 [0.680279, 0.685145] | 0.244838 |
| Upstream model only | 785–718 | 52.229% [51.057, 53.254] | 0.694954 [0.692731, 0.696769] | 0.250823 |
| Even market | No directional pick | — | 0.693147 | 0.250000 |

**Measured:** pooled calibration-training accuracy is 57.312%, log loss
0.680746, and Brier 0.243848 over 7,515 fold-training appearances. Held-out minus
training gaps are +0.240 accuracy points, +0.001073 log loss, and +0.000490 Brier.
Against the served evaluation, the paired accuracy gain is +0.133 points
[-1.361, +1.563], probability_positive 0.55925; log-loss reduction is 0.000955
[-0.000076, +0.002103], probability_positive 0.9632. These are baseline sanity
checks and do not authorize serving or settle a research mechanism.

| Held-out season | Decisive games | Wins | Accuracy | Log loss |
| --- | --- | --- | --- | --- |
| 2020 | 220 | 120 | 54.545% | 0.682637 |
| 2021 | 236 | 134 | 56.780% | 0.685498 |
| 2022 | 248 | 143 | 57.661% | 0.680401 |
| 2023 | 266 | 158 | 59.398% | 0.675500 |
| 2024 | 266 | 158 | 59.398% | 0.683979 |
| 2025 | 267 | 152 | 56.929% | 0.683353 |

**Measured:** natural-scale coefficients below use an intercept and the four
declared terms. Exact values, standardizers, per-fold training metrics, and gaps
are saved in `calibration_folds`. All four slopes remain positive; coefficient
standard deviations are 0.0913, 0.0210, 0.0176, and 0.0208 respectively.

| Fold | Intercept | Model logit | Composition | Move | Available |
| --- | --- | --- | --- | --- | --- |
| 2020 | -0.14904 | 0.39681 | 0.25217 | 0.22271 | 0.14816 |
| 2021 | -0.11752 | 0.24557 | 0.27852 | 0.22525 | 0.08912 |
| 2022 | -0.15449 | 0.55769 | 0.24628 | 0.21665 | 0.11234 |
| 2023 | -0.13843 | 0.43128 | 0.24493 | 0.22888 | 0.08734 |
| 2024 | -0.12472 | 0.39990 | 0.26832 | 0.24818 | 0.09558 |
| 2025 | -0.13463 | 0.38260 | 0.30431 | 0.18901 | 0.11214 |

**Measured:** fixed home-probability reliability bins:

| Probability band | Games | Mean probability | Home-cover rate |
| --- | --- | --- | --- |
| [0, .40) | 144 | 35.819% | 38.194% |
| [.40, .45) | 260 | 42.821% | 45.000% |
| [.45, .50) | 447 | 47.591% | 43.400% |
| [.50, .55) | 326 | 52.561% | 53.681% |
| [.55, .60) | 187 | 57.004% | 60.963% |
| [.60, 1] | 139 | 64.428% | 65.468% |

**Measured:** the declared diagnostic family has 53 summaries/cells: four arms
across overall, six seasons, and six reliability bins, plus pooled training.
The fixed fit inventory is 41 margin-model calls (82 ridge estimator fits,
including residual-split estimators), six four-term fits, 642 location fits
with five predefined buckets each, and 1,284 protection quartile-pair
calculations. No candidate or parameter was selected from these results.

### Loading and limits

Pin this directory explicitly. `base_home_probability` and
`four_term_probability` are identical calibrated home probabilities conditional
on no push. `model_logit` is the season-held-out discrete opener input.
`push_probability` is upstream discrete push mass; the two upstream
unconditional masses are `home_cover_probability_excluding_push` and
`home_loss_probability`, whose sum with push mass equals one.
Push rows have null `home_covered` and `base_correct`.
Training-season columns and zero overlap counts accompany every row.

```python
import hashlib
import json
from pathlib import Path

import pandas as pd

root = Path("artifacts/loso_upstream/20260929T235904133782Z")
metadata = json.loads((root / "metadata.json").read_text())
path = root / "predictions.parquet"
if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["predictions_sha256"]:
    raise ValueError("Upstream prediction artifact hash mismatch")
base = pd.read_parquet(path)
if base.duplicated(["game_id", "season"]).any():
    raise ValueError("Duplicate upstream predictions")
for column in ("upstream_training_seasons", "base_training_seasons"):
    if any(str(s) in training.split(",") for s, training in zip(base.season, base[column])):
        raise ValueError("Same-season training")
joined = research_rows.merge(base, on=["game_id", "season"], how="left", validate="one_to_one")
if joined["base_home_probability"].isna().any():
    raise ValueError("Research population exceeds upstream coverage")
```

**Read:** `docs/lead89_unit1.md:91` still requires stage-matched inputs.
**Inferred:** this artifact establishes exclusion in the refitted margin,
residual, calibration, location, discrete, and protection-threshold stages.
It does not certify the frozen feature builder's internal learned inputs,
feature-selection independence, or Tuesday/Thursday availability of Sunday
movement. Earlier same-season games may supply legitimate pregame rolling
covariates. Later seasons can train earlier LOSO folds; this is not a
chronological prospective evaluation. LEAD-89's stage-availability gate and
each consumer's fold-specific fitting obligations remain.

**Measured:** the real command and both required lint checks exited 0:

```bash
UV_CACHE_DIR=tests/scratch/codex/loso_upstream_uv_cache .tools/uv.exe run --no-sync python scripts/build_loso_upstream.py
.tools/uv.exe run --no-sync ruff check scripts/build_loso_upstream.py
.tools/uv.exe run --no-sync ruff format --check scripts/build_loso_upstream.py
```

The replay caps numerical and Arrow threads at two and runs one fit at a time.
The execution log is `tests/scratch/codex/loso_upstream_run.log`. No source modules,
tests, served artifacts, registries, or Git history were changed.
