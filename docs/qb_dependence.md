# CFB QB-dependence interaction feature — predeclaration and screen results

Predeclared: the mechanism, frozen feature definition, evaluation protocol,
and gate thresholds below reproduce (verbatim in substance, reorganized for
this repo's `docs/` convention) the frozen scoping spec ("SPEC-6") an
orchestrator drafted and froze before any code, test, or accuracy number for
this family existed — **read**, that spec document, verified at scoping time
against `git status --short` showing no `cfb_qb_dependence` / `qb_dependence`
paths in the working tree. This document formalizes that freeze into
`docs/`, mirroring `docs/cfb_role_features.md`'s shape: predeclaration first,
results appended below in a separate section, the predeclaration never
edited in place. It is a CFB-only screen (free per `docs/rotation_registry.md`
rule 8): no NFL rows, no NFL feature table, no rotation window, no rotation
family declared.

## Mechanism being operationalized

Direct quote, **read** `docs/pool_edge_plan.md:207-208`: *"QB-dependence
interaction (team output conditioned on QB reliance)"* — a team's offensive
output should react more to a swing in QB quality when the team's offense
actually leans on the QB (a pass-heavy scheme) than when it does not
(run-heavy, defense-first). The project's model is a linear ridge regression
over additive per-team-state features (`margin.fit_margin_model`), so this
interaction can only be captured if a product term is added explicitly — no
existing linear feature combination reproduces it.

## Substrate reused (read, not re-derived)

- CFB PBP already carries `passer_player_id` and the `pass`/`rush` play-type
  flags already used for the explosive-play indicator (`cfb_features.py`'s
  `cfb_competitive_plays` / `build_cfb_team_game_metrics`) — **read**,
  confirmed present in the ingested snapshots for every season 2004-2025
  (**measured** this session: `passer_player_id` null rate on competitive
  pass plays is 2.4% in 2006, 1.3% in 2013, 0.09% in 2020/2025 — coverage is
  not a material concern anywhere in the benchmark window).
- CFB has **no** pregame injury/availability signal of any kind
  (`docs/injury_value_lost.md` sec 5; `docs/cfb_data.md`, **reported**, not
  independently re-verified this session). **Consequence:** the CFB
  QB-quality state is the *raw* trailing EPA/dropback state only — no
  `start_probability` / replacement-EPA blend is attempted or possible. This
  is a **structural** asymmetry between this CFB screen and the eventual NFL
  feature (a separate, later, separately predeclared spec), not a shortcut.

## Frozen feature definition

Nine columns (`CFB_QB_DEPENDENCE_COLUMNS`, `src/nfl_ats/cfb_qb_dependence.py`):
`{home,away,diff}_{off_pass_rate,qb_starter_epa_per_dropback,qb_dependence_interaction}`.

- **`off_pass_rate`**: strictly-lagged span-8 EWM share of a team's
  competitive offensive plays (the same 5-95% win-probability subset
  `CFB_STATE_METRICS` already uses) that are passes. Span 8 and the 3-game
  maturity floor are **not new** — the same convention every other
  `CFB_STATE_METRICS` column already uses ("NFL parameters taken verbatim",
  `cfb_features.py`). No offseason cross-season regression is applied (see
  "Underived constants" below).
- **`qb_starter_epa_per_dropback`**: the CFB analogue of
  `quarterbacks.build_qb_game_metrics` / `build_qb_states`, restricted to the
  same competitive-play subset, gated on a minimum 5 dropbacks per game-row
  (copied from the NFL floor) and a minimum 20 career dropbacks before a
  player's state is exposed (`CFB_QB_MIN_DROPBACKS`, **inferred**, see
  below). Attached to a game via the same "most recent game's leading
  passer" identity rule the NFL production path uses
  (`players.py`'s `latest_qb_appearance` / `_latest_qb_state` — **not** the
  unwired depth-chart pipeline in `quarterbacks.py`, which feeds no
  `FEATURE_SETS` entry, per the task's own read of `constants.py`). Span 12
  EWM (`CFB_QB_STATE_SPAN`, **inferred**), no offseason regression.
- **`qb_dependence_interaction`**: `{side}_qb_starter_epa_per_dropback *
  {side}_off_pass_rate`, built **per side** and then differenced
  (`diff = home - away`) — the literal reading of "a team's own output
  conditioned on its own reliance," per the spec's decision-needing-review
  #1 (the recommended construction, adopted as this spec's own default per
  the task's binding instruction to follow the spec's stated defaults).

Kept out of `CFB_MODEL_FEATURE_COLUMNS` (the frozen XLG-03 contract), the
same way `BIAS_FEATURE_COLUMNS` is kept out of the NFL frozen sets — these
nine columns ride in the canonical CFB table as additive columns only. REG
bit-identity of every pre-existing `cfb_game_features.parquet` column is
tested (`tests/test_cfb_qb_dependence.py::test_reg_bit_identity_of_existing_cfb_columns`).
Missing values (early-season warm-up, unresolved passer identity,
insufficient competitive volume) are left `NaN`, handled by the existing
ridge pipeline's `SimpleImputer(strategy="median", add_indicator=True)` —
not hand-imputed to a neutral constant.

## NFL/CFB asymmetry (flagged prominently, per the task's binding instruction)

CFB has no pregame injury/availability signal, so `qb_starter_epa_per_dropback`
here is the **raw** trailing state only. The eventual NFL feature would have
both `qb_starter_epa_per_dropback` **and** `qb_expected_epa_per_dropback`
(`constants.PLAYER_QB_STATE_METRICS`) available to multiply by reliance —
this CFB screen can only test the raw half. A follow-up NFL spec (Step 3, not
authorized here) would need to decide which of the two NFL columns the
interaction multiplies; the CFB result below cannot answer that question,
only whether the raw-state version of the mechanism clears a CFB screen at
all.

## Underived constants flagged (per the task's binding instruction)

Per the task's hard override, this spec does **not** touch or inherit
`players.py`'s `_REPLACEMENT_QB_EPA = -0.15`, its mismatched
`qb_min_dropbacks` (20 in production vs 50 in `quarterbacks.build_qb_states`'s
own signature), or `constants.DEFAULT_OFFSEASON_RETENTION` (measured wrong at
0.67, true range ~0.35-0.45). Instead, `src/nfl_ats/cfb_qb_dependence.py`
defines its own, local, flagged constants:

- **`CFB_QB_MIN_DROPBACKS = 20`** — copied from `players.py`'s production
  default as the closest-fidelity recommendation. A judgment call, not a
  measurement (**inferred**).
- **`CFB_QB_STATE_SPAN = 12`** — copied from `players.py`'s `qb_span=12`. No
  independent CFB derivation (**inferred**).
- **`CFB_QB_MIN_GAME_DROPBACKS = 5`** — the per-game-row inclusion floor,
  copied from `quarterbacks.build_qb_game_metrics`'s identical floor. A
  data-hygiene constant, not one of the three flagged-as-wrong values.
- **No offseason regression** is applied to either new state. A plain,
  unbroken chronological EWM simply continues across the season boundary —
  mirrors `cfb_role_features.py`'s own player-trail convention (which also
  applies none), and is mathematically identical to calling
  `cfb_features.build_cfb_team_states` with `offseason_retention=1.0`, but
  implemented standalone because that function iterates a hardcoded
  module-level metric tuple and cannot be parameterized onto a new metric
  without touching `src/nfl_ats/cfb_features.py`.
- **`_REPLACEMENT_QB_EPA` is not used at all** — no `start_probability` blend
  exists on CFB, so there is nothing to blend it into.
- **Design choice (inferred, flagged): both new states are restricted to the
  same competitive-play (5-95% win-probability) subset `CFB_STATE_METRICS`
  already uses.** NFL's `quarterbacks.build_qb_game_metrics` has no
  win-probability filter, so this is not a byte-for-byte port of that
  function — only "the CFB analogue," which is what the task's spec calls
  for. Chosen to keep the interaction's own inputs internally consistent
  with the surrounding CFB feature contract it rides alongside.

## Frozen evaluation protocol

**Step 0 — split-half reliability audit, BEFORE any accuracy number**
(`cfb_qb_dependence.cfb_qb_dependence_reliability`, mirroring
`docs/injury_value_lost.md` sec 3.1 exactly, the repo's own precedent for
this order of operations, already reused twice —
`scripts/cfb_role_continuity_remeasurement.py`,
`scripts/cfb_value_weighted_continuity_screen.py`): reshape to one row per
team-game, split each team-season into odd/even weeks (>=2 observations per
half required), correlate the two halves' means (Pearson r, Spearman rho),
Spearman-Brown correct to a full-length reliability, bootstrap a 95% CI and
`probability_positive` that the correlation is positive. Run identically on
the interaction column and its two constituents separately, so a low
interaction reliability can be diagnosed as "one input is noisy" vs. "the
product itself is unstable even though both inputs are fine."

*Gate:* if the interaction column's Spearman-Brown reliability sits at or
below this repo's own worked "no split-half reliability" examples (coach ATS
reputation 0.063, play-EPA dispersion 0.014, `docs/pool_edge_plan.md:283-284`)
→ `refuted_mechanism` (admissible ground `no_split_half_reliability`), record
and stop. Anything meaningfully above that floor (this repo's own cleared
range: `injury_value_lost` 0.87-0.93, `cfb_role_continuity` 0.68-0.72)
proceeds to Step 2. No single fixed numeric bar is used; the comparison is
against this repo's own worked examples, stated explicitly in the results
section below.

**Step 2 — the two-arm accuracy screen**
(`scripts/qb_dependence_cfb_screen.py`, structured like
`scripts/residual_location_screen.py`): baseline (`CFB_MODEL_FEATURE_COLUMNS`,
the frozen XLG-03 contract, unchanged) vs candidate (baseline **plus**
`diff_qb_dependence_interaction` **and** its two constituent diff columns,
added alongside the baseline, never instead of it — a positive result must
be attributable to the *product*, not to either main effect alone). Both
arms fit with `fit_cfb_residual_model(training, feature_columns=...)`,
walk-forward over the full CFB benchmark history
(`CFB_BENCHMARK_START_SEASON=2006` to `CFB_BENCHMARK_END_SEASON=2025`),
paired on `game_id`, scored with `experiments.paired_feature_comparisons` on
the clean-core window (2012-2019, 2021-2025), week- and season-blocked,
**20,000 samples, `on_degenerate="raise"`** (BINDING per the task
instruction). Before any accuracy number is read, the script prints the
MDE80 power check (`estimation_variance.mde80`) for this screen's own
disagreement fraction `f` and sample size `n`, mirroring
`docs/cfb_role_features.md`'s own reclassification use of the identical
check.

*Gate (predeclared, mirroring SPEC-5's screen convention and MOD-07's Brier
bar — the closest existing precedents in this repo, borrowed for
consistency, not independently derived for this mechanism — flagged as a
decision needing owner review per the spec):*

- `probability_positive` >= 0.75 on accuracy **or** a resolved Brier gain
  with `probability_positive` >= 0.90 → the mechanism earns Step 3 (draft,
  do not run, the NFL interaction columns and an NFL rotation-family
  declaration — a separate, later, separately predeclared spec).
- Interval excludes zero on the wrong side **and** Step 0 already ruled out
  no-split-half-reliability **and** the MDE80 check shows the screen could
  have detected the hypothesized size → closer to `bounded_by_control` (state
  which admissible ground applies before recording `closed_negative`).
- Anything else (the likely, predeclared-default outcome for a small single
  feature) → `unresolved_below_power`. Not a negative. Record via
  `nfl-ats weak-signals record --league cfb` and stop; the result joins the
  stacker/pool, it is not deleted and is not grounds to abandon the
  mechanism.

This spec explicitly does **not** authorize declaring an NFL rotation
family, calling `rotation.assign_window` for any family, building the NFL
interaction columns in `players.py`/`constants.py`, or touching
`game_features_player*.parquet`. That is Step 3, gated on this screen's
result, deferred to a follow-up spec.

## Declared limitations

1. **Structural NFL/CFB asymmetry** (see above) — the raw-state-only
   construction cannot be extended to the blended `qb_expected_epa_per_dropback`
   form without a separate NFL-side design decision.
2. **Both new states restricted to competitive plays** — an inferred design
   choice, not literally dictated by the spec's text, made for internal
   consistency with the surrounding CFB feature contract.
3. **No offseason regression** — a team's/player's state from the end of one
   season carries into the next season's opener unchanged (continuous EWM,
   no decay toward a league mean). This is a deliberate simplification to
   avoid inheriting the flagged-wrong `DEFAULT_OFFSEASON_RETENTION`, not a
   claim that no decay is the correct model.
4. **Small-sample CFB subgroup collapse.** No stratified subgroup analysis
   (e.g. by pass-rate tercile) is run in this screen; per
   `docs/injury_value_lost.md` sec 3.3's caution, any future stratification
   on a small slice should be reported, not treated as the headline.

---

## Results (measured this session)

Full run: `.\.tools\uv.exe run --no-sync python scripts/qb_dependence_cfb_screen.py --start-season 2006 --end-season 2025`,
artifact `artifacts/qb_dependence_cfb/20260818T214601Z/` (**measured**;
runtime 64.2s — well inside the ~2-hour projected-time budget; a smaller
2018-2025 timing dry run completed in 18.8s first, per the task's contention
guard). 2,940,720 pbp rows loaded across 2006-2025; 12,500 canonical CFB
games. Column coverage of the nine new columns ranges 89.3%-98.4% non-null
(`diff_qb_dependence_interaction` lowest at 89.3%, `home_off_pass_rate`
highest at 98.4%) — the remainder is left `NaN` and handled by the existing
ridge pipeline's median imputer, per the frozen design.

### Step 0 — split-half reliability (measured, before any accuracy number)

Odd/even-week team-season split, full 2006-2025 CFB history:

| Column | `n` team-seasons | Pearson r | 95% CI | Spearman rho | Spearman-Brown reliability | `probability_positive` |
|---|---|---|---|---|---|---|
| `qb_dependence_interaction` | 2,325 | 0.9208 | [0.9106, 0.9303] | 0.9225 | **0.9588** | 1.000 |
| `qb_starter_epa_per_dropback` | 2,366 | 0.9196 | [0.9086, 0.9288] | 0.9227 | **0.9581** | 1.000 |
| `off_pass_rate` | 2,361 | 0.9846 | [0.9826, 0.9864] | 0.9730 | **0.9923** | 1.000 |

**Comparison stated explicitly, per the predeclaration's own instruction not
to use a single fixed bar:** all three reliabilities sit far above this
repo's own "no split-half reliability" worked examples (coach ATS reputation
0.063, play-EPA dispersion 0.014) and even above its "cleared" precedent
range (`injury_value_lost` 0.87-0.93, `cfb_role_continuity` 0.68-0.72) — the
interaction column's own reliability (0.9588) exceeds every previously kept
signal in this repo. **This directly rules out `refuted_mechanism` via
`no_split_half_reliability`.** The two constituents are diagnostic: neither
is the weak link, and the product (0.9588) is not meaningfully noisier than
either factor alone (0.9581, 0.9923) — no evidence of noise amplification
from multiplying two states together. **Gate cleared; Step 2 proceeds.**

### Step 2 — MDE80 power check (read before the accuracy number)

Clean-core window, full paired sample: `f` (fraction of games where the two
arms' forced picks differ) = **0.1385**, `n` = **9,093** games →
**MDE80 = 1.093 accuracy points**. This screen's own instrument could not
reliably detect an accuracy effect smaller than ~1.1 points at 80% power on
this sample.

### Step 2 — accuracy and Brier screen (clean-core, candidate minus baseline)

20,000 samples, seed 20260818, 8,933 paired games. **Unit note:**
`paired_feature_comparisons`'s raw `accuracy_improvement` estimate is a
per-game FRACTION; converted to "accuracy points" (this repo's
`accuracy_points` scale, e.g. `docs/cfb_role_features.md`'s own "-0.0067
(-0.67 pts)") by multiplying by 100. Brier and log-loss are reported in
their own raw (unconverted) units, matching that same precedent.

| Block | Metric | Estimate | 95% interval | `probability_positive` | Blocks |
|---|---|---|---|---|---|
| week | accuracy_improvement | **+0.0224 pts** (raw 0.000224) | [-0.8206, +0.8681] pts | **0.5140** | 199 |
| week | brier_improvement | -0.0001 | [-0.0005, +0.0003] | 0.3633 | 199 |
| week | log_loss_improvement | -0.0002 | [-0.0010, +0.0007] | 0.3582 | 199 |
| season | accuracy_improvement | +0.0224 pts (raw 0.000224) | [-0.8888, +0.9046] pts | 0.5186 | 13 |
| season | brier_improvement | -0.0001 | [-0.0004, +0.0003] | 0.3262 | 13 |
| season | log_loss_improvement | -0.0002 | [-0.0008, +0.0006] | 0.3185 | 13 |

No block count is degenerate (week: 199 blocks; season: 13 blocks, above
`MIN_BLOCKS_FOR_INTERVAL=10`).

### Verdict: `unresolved_below_power` (category 3) — not a negative

Applying the predeclared gate mechanically:

1. **Screen-clears check:** accuracy `probability_positive` = 0.514, far
   below the 0.75 bar; Brier shows no resolved gain (point estimate negative,
   `probability_positive` = 0.363, below the 0.90 bar even if it had been
   positive). **Does not clear.**
2. **`bounded_by_control`-adjacent check:** neither interval excludes zero on
   the wrong side — both the accuracy and Brier 95% intervals straddle zero
   comfortably, with point estimates (+0.0002 pts accuracy, -0.0001 Brier)
   an order of magnitude smaller than their own interval half-widths. This
   condition requires the interval to exclude zero on the wrong side; it
   does not. **Does not apply.**
3. **Default:** the point estimate (+0.0224 accuracy points) sits at roughly
   **2.05%** of the MDE80 floor (1.093 points) — the screen could not have
   resolved an effect this small even if the true effect were real and this
   size. Reliability (Step 0) rules out "no split-half reliability" as the
   explanation; the null reading is not a measurement artifact of an
   unreliable trait. **`unresolved_below_power` applies.**

This is the textbook shape AGENTS.md describes as the *expected* outcome for
a real-but-small single feature at this evaluator's ~2-point resolution: a
highly reliable trait (0.96 split-half) producing an accuracy signal too
small for this instrument to see, not a trait that is itself noise. The
mechanism is not refuted — it is simply not resolvable at this screen's
current power, and per AGENTS.md's binding rule, an interval containing zero
is never grounds for rejection on its own.

### Proposed `weak-signals record` command (NOT executed — proposed only, per the task's instruction)

```
nfl-ats weak-signals record \
  --name cfb_qb_dependence_interaction \
  --description "CFB QB-dependence interaction (raw qb_starter_epa_per_dropback x off_pass_rate, per side, differenced) vs the frozen XLG-03 CFB benchmark, candidate adds the interaction plus its two constituent diff columns alongside the baseline." \
  --source "artifacts/qb_dependence_cfb/20260818T214601Z/ ; scripts/qb_dependence_cfb_screen.py ; docs/qb_dependence.md" \
  --effect 0.0224 \
  --effect-units accuracy_points \
  --classification unresolved_below_power \
  --league cfb \
  --season-start 2006 \
  --season-end 2025 \
  --interval-low -0.8206 \
  --interval-high 0.8681 \
  --probability-positive 0.5140 \
  --sample-games 8933 \
  --sample-blocks 199 \
  --reliability 0.9588 \
  --classification-evidence "Week-blocked accuracy P+=0.5140 (below 0.75 clear bar), Brier P+=0.3633 with no resolved gain (below 0.90 bar); both 95% intervals straddle zero comfortably (do not exclude zero on the wrong side, so bounded_by_control does not apply). Point estimate (+0.0224 pts) sits at ~2.05% of this screen's own MDE80 floor (1.093 pts, f=0.1385, n=9093) -- below detection power, not evidence of zero effect. Step 0 split-half reliability is 0.9588 (interaction), 0.9581/0.9923 (constituents) -- far above this repo's no-split-half-reliability examples (0.063, 0.014) and even above its own cleared precedents (0.68-0.93), ruling out refuted_mechanism." \
  --notes "CFB screen only; NFL Step 3 (interaction columns using qb_starter_epa_per_dropback vs the injury-blended qb_expected_epa_per_dropback, NFL rotation-family declaration) is explicitly deferred to a separate, later, separately predeclared spec per docs/qb_dependence.md. This result does not authorize any NFL rotation window."
```

Units note: `accuracy_points` is stored as PERCENTAGE POINTS
(`registry/weak_signals.json`'s own documented scale, e.g. record 1.10 for a
1.1-point gap, not 0.011) — the raw per-game fraction this screen measured
(0.000224) converts to **0.0224 accuracy points** for the registry, matching
`docs/cfb_role_features.md`'s own precedent of reporting its raw -0.0067
estimate as "-0.67 pts." Recording the raw fraction (0.0002) instead would
understate the effect 100x and make this entry incomparable to every other
`accuracy_points` entry in the registry.
