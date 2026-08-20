# The agentless experiment pipeline

**Owner-requested, 2026-08-18.** "Put some data in and get an answer out."
One CLI entry point that takes a declarative spec and runs the whole
standardized research loop -- reliability check, screen, bootstrap,
mechanical classification, registry record, provenance stamp -- with zero
hand-transcription.

## Why this exists

This session's recorders caught three separate hand-transcription defects in
one sitting: a 100x fraction-vs-points scaling bug, a sign bug, and a
corrupted source path -- all in numbers a human copied from console output
into `registry/weak_signals.json` by hand. Every piece needed to avoid that
error class already existed as separate machinery:

- `nfl_ats.experiments.paired_feature_comparisons` -- the block-bootstrap
  engine with the D4 degeneracy guard.
- `nfl_ats.estimation_variance` -- `MIN_BLOCKS_FOR_INTERVAL`,
  `guard_block_count`, and the honest refit-correction band this module
  cites.
- `nfl_ats.weak_signals` -- `record_signal`/`validate_closure`, which IS the
  closing-ground taxonomy encoded as a validator, not prose.
- `nfl_ats.provenance.write_experiment_artifact` -- the run-provenance stamp
  every CLI command already gets as a side effect of its artifact write.
- `scripts/penalty_discipline_interval.py` / `scripts/nfl_bias_battery_screen.py`
  -- the subset-vs-complement, week-blocked joint bootstrap, full-slate-scaling
  pattern this module generalizes into a registry of named, reusable flag
  builders.

What was missing was the glue. `src/nfl_ats/experiment_runner.py` is that
glue: it computes every registry field directly from data, so there is no
point in the loop where a human retypes a number.

## Quick start

```powershell
.\.tools\uv.exe run nfl-ats experiment run registry/experiment_specs/penalty_discipline_reproduction.json --dry-run
```

Drop `--dry-run` to actually stamp an artifact under
`artifacts/experiment_runner/<UTC timestamp>/` and record the result to
`registry/weak_signals.json`. Add `--replace` to intentionally overwrite an
existing entry of the same name (matching `weak-signals record`'s own
convention -- silently overwriting is refused otherwise, so a second look at
one signal can never masquerade as new evidence).

```powershell
.\.tools\uv.exe run nfl-ats experiment run my_spec.json
```

## The spec schema

A spec is one JSON object, validated as strictly as `weak_signals.json`
payloads (unknown fields rejected, every value type-checked). Fields:

| Field | Type | Notes |
|---|---|---|
| `name` | string | Registry key. Must be unique unless `--replace` is passed. |
| `hypothesis` | string | Prose. Becomes the recorded `description`. |
| `experiment_type` | `"subset_bias"` \| `"feature_arm"` | See below -- both are implemented. |
| `population.league` | `"nfl"` \| `"cfb"` | Only `"nfl"` builders are registered today (both `subset_bias` flag builders and `feature_arm`'s `margin.MARGIN_FEATURE_PROFILES`). |
| `population.seasons` | `[start, end]` | Filters the population after the builder computes its trait (a lag needs the full local history to compute correctly; the seasons window only trims the final comparison). For `population.grade="opener"`, the paired Tuesday-opener archive itself only covers ~2020-2025; a wider `population.seasons` is silently trimmed to that intersection (see "opener" below), never an error. |
| `population.grade` | `"close"` \| `"opener"` | `subset_bias` supports both: `"close"` reads `game_features.parquet`'s own `spread_line`; `"opener"` restricts to the paired Tuesday-opener archive (`clv.build_pairing_table`/`close_reference_table`, same population `clv.opener_pick_evaluation` uses) and overwrites `spread_line`/`home_cover`/`ats_margin` to the opener line before any builder runs. `feature_arm` supports only `"close"` (`outcomes.walk_forward_outcomes` grades against `game_features.parquet`'s own spread_line, i.e. the close, across its full history). |
| `construct.flag_builder` | string | `subset_bias` only. A name registered in `FLAG_BUILDERS` (see below). No `eval()` of arbitrary code -- every construct is a Python function reviewed and committed to this module. |
| `construct.params` | object | `subset_bias` only. Builder-specific keyword arguments (e.g. `large_favorite`'s `threshold`). |
| `construct.baseline` / `construct.candidate` | object | `feature_arm` only. Each is `{"feature_profile": <name in margin.MARGIN_FEATURE_PROFILES>, "ridge_alpha": <float, default 10.0>}`. |
| `endpoints.primary` | `"accuracy"` | The project's primary bar; this is the only value accepted. |
| `endpoints.secondary` | list of `"brier"`/`"logloss"` | Must be empty for `subset_bias` (a raw cover-rate comparison has no probabilistic prediction to score Brier/log-loss against). For `feature_arm` these are computed by `experiments.paired_feature_comparisons` alongside accuracy and reported (never gate the registry `effect`, which is always the accuracy comparison). |
| `blocking.primary` | `"week"` \| `"season"` | Default `"week"`. |
| `blocking.secondary` | `"week"` \| `"season"` \| `null` | Default `"season"`. Must differ from `blocking.primary`. |
| `samples` | int >= 10 | Bootstrap resamples per blocking. Default 20,000 (see `paired_feature_comparisons`'s own docstring for why 20,000, not 2,000). |
| `seed` | int, **required, no default** | No wall-clock nondeterminism: every run must be reproducible from the spec alone. |
| `reliability_check.method` | `"split_half"` \| `"not_applicable"` | Required. Most `subset_bias` hypotheses are per-game situational conditions (home underdog, large favorite) with no persistent trait to split-half -- `"not_applicable"` is the honest, common case for those. |
| `reliability_check.reason` | string | Required and must be non-empty when `method` is `"not_applicable"`. |

See `registry/experiment_specs/penalty_discipline_reproduction.json` for a
complete, runnable example (it reproduces the `penalty_discipline` registry
entry exactly -- see "Validation anchor" below).

## `experiment_type: "subset_bias"` (fully implemented)

A pregame-safe boolean flag vs. its complement, cover rate vs. the spread.
Generalizes two existing precedents:

- `scripts/nfl_bias_battery_screen.py`'s design -- a flag over the WHOLE
  population, complement = everyone else. `fraction_of_slate = n_flag /
  n_total`. (`hc_year_one_fade` in `registry/weak_signals.json` uses the same
  scaling logic.)
- `scripts/penalty_discipline_interval.py`'s design -- a flag restricted to
  part of the population (e.g. quartile 1 vs quartile 4, excluding the
  middle two quartiles from the direct comparison), where BOTH arms are
  exploitable. `fraction_of_slate = (n_flag + n_complement) / n_total`.

Which design applies is a property of the named builder (whether it returns
an `eligible` restriction), not something the runner guesses.

### The flag-builder registry

| Name | Leagues | Trait for split-half? | Description |
|---|---|---|---|
| `penalty_rate_quartile` | nfl | yes (year-over-year rate correlation) | Prior-season team penalty-rate quartile 1 vs quartile 4. Reproduces `scripts/penalty_discipline_interval.py`. |
| `home_underdog` | nfl | no | Home team getting points, vs. everyone else. |
| `large_favorite` | nfl | no | Favored by more than `params.threshold` (default 10) points, vs. everyone else. |
| `drought_severe_grass` | nfl | no | Fresh US Drought Monitor D2+ county exposure at an outdoor grass venue, using the official Thursday 08:30 ET release cutoff; `params.d2_area_threshold` defaults to 50%. |
| `division_revenge_game` | nfl | no | 2nd meeting this season vs. same opponent; team lost the 1st meeting. Ported from `scripts/nfl_bias_battery_screen.py`. |
| `extra_rest_edge` | nfl | no | Team's rest minus opponent's rest >= 4 days. Ported from `scripts/nfl_bias_battery_screen.py`. |
| `short_week` | nfl | no | Team's own rest <= 5 days. Ported from `scripts/nfl_bias_battery_screen.py`. |
| `west_coast_early_kickoff` | nfl | no | Traveling Pacific-timezone team, non-PT opponent, kickoff before 14:00 ET. Ported from `scripts/nfl_bias_battery_screen.py`. |
| `sandwich_spot` | nfl | no | Non-division game flanked by a division game last week and next week. Ported from `scripts/nfl_bias_battery_screen.py`. |
| `backup_qb_start` | nfl | no | Starting QB differs from the team's modal QB this season (>=3 prior starts); rows with fewer than 3 prior starts are excluded from both arms via `eligible`. Ported from `scripts/nfl_bias_battery_screen.py`. |
| `motivation_mismatch` | nfl | no | Competitive team (>=40% prior win pct) facing a `bad_team_late`-shaped opponent. Ported from `scripts/nfl_bias_battery_screen.py`. |

The seven bias-battery builders above are faithful ports of
`scripts/nfl_bias_battery_screen.py`'s own flag logic (same masks, same
thresholds, same history-feature derivations), so this runner can re-screen
an already-recorded close-graded `bias_battery_*` entry at another grade
(e.g. the opener) without a second bespoke script -- see
`registry/experiment_specs/bias_battery_*_opener.json` for the 2026-08-19
opener re-screen of eight of them (`registry/weak_signals.json`'s
`bias_battery_*_opener` entries).

Adding a builder means adding one function to `FLAG_BUILDERS` in
`src/nfl_ats/experiment_runner.py` -- it receives the loaded feature table,
the requested season window, `construct.params`, and the repo root, and
returns a `SubsetBiasConstruct` (population table, boolean flag, optional
eligibility mask, sign, and reliability info). No spec can name a builder
that isn't reviewed code.

### What the runner computes, end to end

1. Load `data/processed/game_features.parquet` (or `--features`), run the
   named builder.
2. Filter to `population.seasons`.
3. Compute `raw_gap_pct` (unscaled subset-minus-complement cover-rate gap,
   sign-oriented) and `fraction_of_slate`, then `effect =
   scale_subset_effect(...)` -- the ONE place a cover-rate fraction becomes
   accuracy POINTS (the exact 100x step a hand-transcription got backwards
   this session).
4. Joint block-bootstrap the primary blocking (default week) and, if
   requested, the secondary (default season), reusing
   `estimation_variance.guard_block_count` (the D4 degeneracy guard) on each.
5. If `reliability_check.method == "split_half"`, use the builder's own
   reliability measurement (e.g. year-over-year Pearson correlation of the
   underlying trait); if the builder has no persistent trait, the runner
   refuses to run rather than silently reporting `null`.
6. Mechanically classify the primary interval (see below).
7. Build a `weak_signals.WeakSignal` with every field computed from the run
   -- `classification_evidence` is auto-generated prose citing the measured
   numbers, the seed, the spec path, and (on a real run) the artifact path.
8. Unless `--dry-run`: stamp a provenance artifact
   (`nfl_ats.provenance.write_experiment_artifact`) and record the signal to
   `registry/weak_signals.json` under a filesystem lock (see "Concurrency"
   below).

## `experiment_type: "feature_arm"` (implemented)

Two `margin.fit_margin_model` arms (baseline/candidate feature profile
and/or `ridge_alpha`), each walked forward with
`outcomes.walk_forward_outcomes` (`methods=("market_residual",)` only, the
close/`nflverse_spread` grade -- the same grade
`scripts/ridge_alpha_promotion_eval.py.run_nflverse_grade` uses), tagged with
a `feature_set` label ("baseline"/"candidate") and paired by `game_id`
through `experiments.paired_feature_comparisons` -- the ALREADY-REVIEWED
block-bootstrap engine this whole module exists to stop hand-transcribing
output from, reused rather than re-derived.

`paired_feature_comparisons` returns `accuracy_improvement`,
`brier_improvement`, and `log_loss_improvement` together; the runner always
computes accuracy (the registry's recorded `effect`, scaled *100 into
accuracy POINTS -- the same 100x step `scale_subset_effect` performs for
`subset_bias`) and additionally computes brier/log_loss only when named in
`endpoints.secondary` (recorded raw, unscaled, per
`weak_signals.EFFECT_UNITS`'s documented convention). Mechanical
classification reuses `classify_subset_bias_result` on the primary accuracy
interval, exactly as `subset_bias` does -- no separate classification logic.

`population.grade` must be `"close"`; `reliability_check.method` must be
`"not_applicable"` (a model-arm comparison has no per-entity trait to
split-half); `population.league` must be `"nfl"`.

## Mechanical classification (AGENTS.md, binding)

`AGENTS.md`'s "An interval crossing zero is NOT grounds for rejection" rule
names exactly two admissible closing grounds: a refuted mechanism (wrong
sign, or no split-half reliability) and a positive-control bound. This
runner has authority over only ONE of those, mechanically, on ONE condition:

`refuted_mechanism` / `wrong_sign_resolved` fires only when **both**:

1. the PRIMARY (week-blocked) interval sits entirely below zero, **and**
2. the inflation factor needed to widen that interval back across zero
   exceeds `HONEST_REFIT_WIDENING_UPPER_BOUND = 1.099` -- the documented
   one-sided 95% upper bound on how much an honest, refit-aware interval
   could widen a naive one for a fit-changing comparison
   (`docs/estimation_variance.md`: "...1.293x to 1.003x (one-sided 95% upper
   bound 1.099x)"). The same constant is what the registry's own reviewer
   adjudication cites verbatim on `mod06_js_shrinkage_position_prior_cfb`:
   closure was refused there because re-crossing zero needed only a 1.082x
   widening, "inside the documented 1.003-1.099x honest refit-correction
   band". `experiment_runner.widening_factor_to_recross_zero` reproduces
   that exact 1.08-1.09x figure from the entry's own recorded numbers (see
   `tests/test_experiment_runner.py`).

Every other outcome -- including a naive interval that excludes zero but
would need less than 1.099x widening to re-cross -- is recorded
`unresolved_below_power` with no `closing_ground`. **The runner never
produces `bounded_by_control`, and never produces a reliability-grounded
`no_split_half_reliability` closure; both remain human adjudications, and a
spec has no field that could request them.**

## Concurrency: the single-writer convention

`registry/weak_signals.json` has always had a documented single-writer
convention; every existing CLI writer (`weak-signals record`, `rotation
record`, ...) already assumes it and none of them lock. This runner is now
one more caller, so it enforces the convention mechanically rather than
merely inheriting the assumption: a cheap filesystem lock
(`experiment_runner._RegistryLock`, exclusive file creation via `O_CREAT |
O_EXCL`, atomic on both POSIX and Windows, no new dependency) wraps the
load-modify-save critical section. **Concurrent `nfl-ats experiment run`
invocations targeting the same registry must still not overlap with
`nfl-ats weak-signals record` or `nfl-ats rotation record` invocations --
only writers that go through this runner's lock see it.** If a run dies
mid-write and leaves a stale `.lock` file, remove it by hand; the lock times
out (30s default) and raises a clear error rather than hanging forever.

## Validation anchor: bit-for-bit reproduction of `penalty_discipline`

`tests/test_experiment_runner.py::test_penalty_discipline_reproduces_the_recorded_registry_entry`
runs the full pipeline at **full fidelity** (`samples=20000`,
`seed=20260818` -- not a reduced sample; the whole run, PBP load included,
takes about three seconds) and checks it against the recorded entry:

| Field | Recorded | Reproduced |
|---|---|---|
| `effect` | +0.3288 | +0.32880305490528466 |
| `interval` (week-blocked) | [-1.0389, +1.6849] | [-1.0388784801407385, +1.68488795453963] |
| `probability_positive` | 0.6828 | 0.68285 |
| `standard_error` | 0.6938 | 0.6938082573229161 |
| `reliability` | 0.261 | 0.26044445677002404 |
| `sample_games` | 4085 | 4085 |
| `sample_blocks` | 277 | 277 |

This is because `penalty_rate_quartile` and the runner's generic bootstrap
are a faithful port of `scripts/penalty_discipline_interval.py`'s own
construct and joint block bootstrap (same seed, same block-id derivation
order, same single `rng.multinomial` call shape) -- re-running that script
directly (`.tools/uv.exe run --no-sync python
scripts/penalty_discipline_interval.py`) reproduces the same floats to more
digits than the test asserts.

## Validation anchor: `feature_arm`, an algebraic identity (not a recorded number)

No `feature_arm`-shaped entry in `registry/weak_signals.json` is
bit-for-bit reproducible the way `penalty_discipline` is above: every
`player_family_base_vs_*` entry (the obvious profile-vs-profile candidates)
is recorded **UNCONFIRMED** in its own `notes` field -- "derived by analogy
to `participation_offense_defense_rapm`'s registered value ... not
independently recomputed for this arm. `probability_positive` DERIVED via
normal approximation ... not re-bootstrapped" -- so there is nothing
recorded to check a fresh run against.

`tests/test_experiment_runner.py::test_feature_arm_identical_arms_measure_exactly_zero_on_real_data`
anchors instead on an algebraic identity that must hold for ANY correctly
wired `feature_arm` run: `margin.fit_margin_model`'s ridge fit is fully
deterministic (closed-form solver, no bootstrap/shuffling, and the
calibration-distribution split is a deterministic ordered slice), so two
arms with an IDENTICAL `feature_profile`/`ridge_alpha` produce
BIT-IDENTICAL predictions on the same training data -- every paired
accuracy/brier/log_loss improvement must measure exactly `0.0` (estimate,
interval, and `probability_positive` all `0.0`). Runs the real pipeline
end to end (real feature load, real `walk_forward_outcomes` fits, real
`paired_feature_comparisons` bootstrap) on one season of the cheapest
(`base`) profile for speed.
`tests/test_experiment_runner.py::test_run_feature_arm_experiment_end_to_end_on_synthetic_data`
separately anchors the runner's OWN glue (feature-set tagging, the 100x
accuracy scaling, which metrics `endpoints.secondary` computes) against a
mocked `walk_forward_outcomes` with hand-computable arithmetic.

## What is deliberately out of scope this pass

- CFB flag builders and CFB `feature_arm` arms (both are NFL-only today;
  `subset_bias`'s three original builders plus the seven bias-battery ports
  all require NFL-shaped feature-table columns, and `feature_arm` requires
  `population.league == "nfl"` explicitly -- CFB support needs CFB-specific
  feature-table paths and column names, a straightforward but separate
  follow-up).
- `feature_arm` at `population.grade == "opener"` (would need
  `clv.opener_pick_evaluation`'s weekly-refit/opener-substitution machinery
  wired in separately, the pattern
  `scripts/ridge_alpha_promotion_eval.py.run_opener_grade` already
  demonstrates for one specific comparison; not implemented here).
- `backup_qb_start`-style constructs that need a schedules-snapshot merge for
  a league without an equivalent QB-name column are not addressed (NFL's
  `data/raw/*/schedules.parquet` has `home_qb_name`/`away_qb_name`; no CFB
  equivalent was checked).
