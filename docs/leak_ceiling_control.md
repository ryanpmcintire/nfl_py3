# Deliberate-leak positive control: the absolute classification ceiling

Question: if we fit the model ON THE SAME OUTCOMES it predicts (total leak),
what accuracy can ANY feature set achieve on this data? This bounds legitimate
models from above and empirically tests the 55-58% oracle-wall claim against
OUR grading structure rather than a borrowed one.

Status: measure-only positive control, first run 2026-08-22
(`artifacts/leak_ceiling/20260822T222810Z/results.json`). Nothing recorded to
the rotation registry, no pick played, no promotion decision, no wagering
implication claimed.

## LEAK DECLARATION (loud, deliberate)

Every arm's training set is the ENTIRE completed frame: all 4,431 completed REG
games 2009-2025 in `data/processed/game_features_weak_stack.parquet`,
INCLUDING the target seasons and the target games themselves (**measured**,
recorded as `training_rows` per arm). The walk-forward skeleton is retained
only to define the standard scored population; under this policy every fold's
training set is identical to the full frame, so one fit per arm is algebraically
equivalent to refitting per fold and no fold loop is executed. Any resemblance
between these numbers and a forecastable edge would be a misreading; this is an
instrument, not a model.

## Method

- Population: the standard scored instrument exactly — REG 2018-2025 with
  result and spread present on the production weak-stack table = 2,127 games,
  of which 2,075 non-push are graded (**measured**; identical count to the
  active model's close-grade evaluation artifact).
- Grading contract matches `backtest.summarize_predictions`: forced pick at
  p >= 0.5 over non-push rows; cover probabilities produced by
  `MarginModel.predict(probability_method="gaussian")`, the same read that
  graded the active model's 52.10%.
- One leak-fit path shared by all arms: `Ridge(alpha)` on `ats_margin` trained
  in-sample on the full frame; IN-SAMPLE residuals become the predictive
  distribution. Deliberately more leaky than production's out-of-time residual
  split — that is the point.

## Arms and results (all **measured**)

| Arm | Features | Accuracy (2,075) | Brier |
| --- | --- | --- | --- |
| A `market_line_leak` | spread_line + spread_line^2 | **49.40%** | 0.2502 |
| B `weak_stack_leak` | full weak_stack design (~249 cols), alpha=10 | **55.57%** | 0.2458 |
| B2 `weak_stack_leak_alpha1` | same, alpha=1 shrinkage-sensitivity | **56.05%** | 0.2457 |
| C `pbp_same_game_leak` | same-game PBP summaries (6 cols) | **84.05%** | 0.1080 |

No-leak references from the same population and grading
(`artifacts/margins/20260820T004951Z/summary.csv`, **measured**): market
baseline 50.94%, active weak_stack walk-forward market_residual 52.10%.

Season table (accuracy %, **measured**):

| Season | A line | B weak_stack | B2 alpha=1 | C same-game PBP |
| --- | --- | --- | --- | --- |
| 2018 | 45.7 | 51.8 | 51.4 | 85.8 |
| 2019 | 47.2 | 56.9 | 56.5 | 85.4 |
| 2020 | 47.7 | 57.8 | 59.0 | 81.2 |
| 2021 | 52.2 | 60.4 | 61.6 | 85.4 |
| 2022 | 46.4 | 52.5 | 51.0 | 83.5 |
| 2023 | 52.7 | 57.0 | 58.9 | 87.6 |
| 2024 | 53.0 | 56.0 | 57.1 | 82.1 |
| 2025 | 49.8 | 52.0 | 52.8 | 81.5 |

Season spread for arm B: 51.8-60.4% (B2: 51.0-61.6%); no season approaches
~100%, and the two weakest leak seasons (2018, 2022, 2025) are also where the
honest model is closest to baseline — consistent with outcome noise dominating
everywhere (**inferred**).

Arm A note: 49.40% is not a bug. The line's LEVEL carries essentially no
in-sample-readable direction for cover once the line itself defines the
settlement; the squared term was added precisely so favorite-longshot curvature
was expressible, and the fit still cannot beat chance. The market's information
about ATS outcomes is already spent defining the bet (**measured number,
inferred interpretation**).

## Ceiling implication

The expected shape was "mid-60s-to-70s, not ~100%" because outcomes are noisy
given features. The measured shape is sharper and splits the question in two:

1. For PREGAME-style feature classes — the only ones this project may legally
   use — total leak buys just **55.6% (+0.5pt at lighter shrinkage)** on our
   grading structure, barely above the honest 52.10%. This revises the
   55-58% oracle-wall claim DOWNWARD for our model class: even cheating does
   not reach 57-58% with pregame features under ridge/market-residual
   machinery. The wall is not something we are "below"; it is approximately
   where the ceiling itself sits (**measured**, single estimator family — see
   limitation below).
2. Arm C proves the instrument and grading structure CAN express much higher
   numbers when features genuinely observe the outcome-generating process:
   same-game play-level summaries hit **84.05%** in-sample. So the modest arm-B
   ceiling is NOT an artifact of the grading contract or the probability path;
   it reflects that pregame observable structure explains very little of
   ATS outcome variance (**measured**; arm C is outcome-contaminated by
   construction — deliberately, as a positive control).

Decision reading: any candidate feature family whose honest accuracy claims
imply approaching ~56%+ on this population should be treated with extreme
suspicion, because it would have to extract nearly all of the structure that
survives even when fitting the targets directly. Conversely, the gap between
the honest 52.10% and the leaked 55.6% (~3.5 points) is the MAXIMUM room any
amount of additional pregame feature engineering could possibly buy within this
model class — and realistic recoverable room is a small fraction of it
(**inferred**).

## Limitations (declared)

- "Ceiling" here bounds THIS estimator class (ridge on standardized linear
  designs at the tested alphas). Arm C shows richer feature classes exceed it;
  a nonlinear learner on pregame features could shift the number. What is
  bounded is the production recipe plus its feature surface, which is what the
  project actually deploys.
- Arm C's six summary columns are a small, deliberately simple contamination
  probe, not an exhaustive same-game design; 84.05% is a floor on that arm's
  true ceiling, not an optimum.
- Single data snapshot (`data/pbp/raw/20260817T184927Z`, weak-stack table sha
  `0a18e2d90ae4…`); numbers move with refreshes.

---

## 2026-08-25: the same control at the OPENER grade

The ceiling above is **close-graded**, and this file's own decision reading
draws the project's headline strategic conclusion from it: "the gap between the
honest 52.10% and the leaked 55.6% (~3.5 points) is the MAXIMUM room any amount
of additional pregame feature engineering could possibly buy."

The pool settles at the **opener**, not the close. Applying a close-graded
ceiling to an opener-graded record is a grade mismatch, so the ~3.5-point
figure could not be used for the thing actually played until the control was
re-run at the same grade. `scripts/leak_ceiling_opener.py` does that.

Same recipe as arm B -- the weak_stack design (90 features, ~249 design columns
after imputation indicators), ridge, in-sample fit on the ATS target, in-sample
residuals as the predictive distribution, forced pick at p >= 0.5 -- with one
substitution: the target and the grading line are the Tuesday opener
(`tue_open_home_spread`) instead of the close. The target is taken from the
archive's own `margin_vs_open` and asserted equal to
`result - tue_open_home_spread`, so a schema change cannot silently redefine
it. Population 1,503 games after pushes, seasons 2020-2025.

Run: `artifacts/leak_ceiling_opener/20260826T011832Z/results.json` (**measured**).

| arm | features | opener accuracy | close accuracy (arm B above) |
|---|---|---|---|
| market line only (leak) | 2 | 50.90% | 49.40% |
| **weak_stack (leak), alpha=10** | 90 | **59.28%** | 55.57% |
| weak_stack (leak), alpha=1 | 90 | 59.41% | 56.05% |

### The finding

**The opener ceiling is 59.28%, about 3.7 points ABOVE the close-graded
ceiling** — measured on the same estimator class and the same design. There is
materially more exploitable structure at the opener than at the close.

That is the project's own core thesis (the opener is the softer line and the
close is the market at its sharpest), and it now has a ceiling attached rather
than only a per-model comparison.

### Headroom at the grade the pool settles on

| reference (same grade, same archive) | accuracy | gap to the 59.28% ceiling |
|---|---|---|
| raw model, probability rule | 53.36% | **5.92 pts** |
| played four-member union, in-sample | 55.42% | 3.86 pts |
| played union, de-inflated planning estimate | ~54.6% | **~4.68 pts** |

The last row is the honest one: a de-inflated expectation against a leaked
ceiling. So the room is **larger** than the ~3.5 points this file previously
implied, not smaller, and it is larger precisely because the opener is the
grade being played.

### What this does NOT license

* It is still a **leak** arm — fitted on the outcomes it is scored against.
  Nothing here is achievable, and the original limitation stands unchanged:
  this bounds ridge on standardized linear designs at the tested alphas, not a
  richer model class.
* "Realistic recoverable room is a small fraction of it" remains the right
  posture. A bigger ceiling means the search is not obviously exhausted; it
  does not mean the points are sitting there.
* The suspicion rule above should be re-read at this grade: an honest
  opener-graded claim approaching ~59% deserves the same extreme scepticism
  the ~56% figure earned at the close.
* Nothing is recorded to the weak-signal registry from this: it is a
  strategic diagnostic about where to spend effort, not a signal, and no
  rotation-registry window is spent.

### Why it matters for what to work on next

The close-graded number made further pregame feature work look nearly
exhausted. At the opener it is not: roughly 4.7 points separate the played card
from its own estimator-class ceiling. That argues the feature surface is still
worth attacking — while noting that this session's two feature arms
(`weak_stack_v4` forecast weather, and its perfect-information oracle) both
came back negative, so the room being there is not evidence that any particular
family will reach it.
