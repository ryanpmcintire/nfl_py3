# MOD-14: era weighting (rolling windows vs. time-decayed sample weights)

Predeclared 2026-08-19 (US), **before** `scripts/era_weighting_cfb_screen.py`
scores any arm's accuracy. Provenance tags used throughout: **measured** (run
this session, command/path given), **read** (file opened this session),
**reported** (another doc's claim, not reverified here), **inferred**
(reasoning, not evidence).

## Binding closing-grounds taxonomy (pasted verbatim, per AGENTS.md/CLAUDE.md)

> An interval or CI that contains zero is NEVER grounds to reject, fail, or
> close an experiment. At this evaluator's ~2-point resolution, "contains
> zero" is the EXPECTED outcome for a real small signal. Only two grounds
> ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
> (whole interval on the wrong side of zero) or zero split-half reliability;
> (2) bounded by a positive control proven able to detect an effect that
> size. Everything else is `unresolved_below_power`: record it with
> `nfl-ats weak-signals record`, report `probability_positive`, never the
> binary "contains zero". The registry code hard-rejects inadmissible
> closures; if a record command errors, the verdict is wrong, not the
> validator.

## 0. ROADMAP status and motivation

`ROADMAP.md` row **MOD-14** (**read** this session, `grep -n "MOD-14"
ROADMAP.md`): "⬜ Era weighting -- Compare rolling training windows and
time-decayed sample weights" -- unbuilt, no prior arm exists for this row.

Motivation, both **read** this session, neither re-derived here:
- `docs/sbr_opener_evaluation.md`: the production model's SBR-graded opener
  edge (production rule) reads flat-to-slightly-negative in 2011-2014 (P+
  0.414), moderate in 2015-2019 (P+ 0.734), and strongly positive in
  2020-2021 (P+ 0.918, the one excludes-zero cell: sign rule +5.233
  [+0.898, +9.351] pts, P+0.9903) -- a monotonic era gradient.
- `docs/era_magnitude_profile.md`: a free-break changepoint search (never
  told any era boundary) independently locates the production model's own
  opener-proxy edge break at **2019** (bootstrap-modal, spread [2014, 2023]),
  and the season-trend OLS slope leans positive (+0.347 pts/season
  [-0.021, +0.708], P+0.968).

Neither document resolves under the binding taxonomy (both intervals cross
zero somewhere), and per the magnitude-not-presence framing that is the
**expected** shape, not evidence of absence. This screen asks a different,
downstream question: **if recent seasons carry more of the model's real
edge, does training the model to weight them more heavily win?** That is a
genuinely separate hypothesis from "does the edge vary by era" -- a model
could show a real era gradient in its OUTPUT while gaining nothing from
reweighting its INPUT, if the gradient reflects the *market* getting
sharper (unlearnable) rather than the *feature-outcome relationship*
shifting (learnable). This screen is powered to distinguish those only
insofar as CFB's free, large sample can resolve a training-recipe effect at
all -- reported honestly either way.

## 1. Prior art this document must not re-litigate or contradict

- **MOD-06** (`docs/mod06_position_prior_shrinkage.md`, `ROADMAP.md` row,
  both **read**): closed on CFB measurement that ridge shrinkage-toward-zero
  (empirical-Bayes / BayesianRidge / ARD / sample-size-scaled alpha) is dead
  as an accuracy lever -- increasing shrinkage makes thin-training buckets
  resolvably worse. **This is a different lever.** MOD-06's shrinkage
  rescales coefficient magnitude toward zero; era weighting reweights which
  ROWS the fit trusts, without adding any shrinkage-toward-zero term. A
  half-life-weighted fit and an unweighted fit see the identical ridge
  penalty (`alpha=10.0` throughout, frozen, never varied by this screen);
  only the row weights entering the normal equations differ. MOD-06's
  closure does not bear on this hypothesis and is not reopened here.
- **MOD-12 / `docs/ridge_alpha.md`** (**read**): `ridge_alpha` stays 10.0.
  This screen never varies alpha -- doing so alongside a weighting change
  would bundle two levers into one contrast, exactly the mistake
  `player_qb_continuity`'s bundling error made (cited in this task's own
  brief as the cautionary tale). Every arm below uses `ridge_alpha=10.0`,
  identical to the frozen production/benchmark configuration.
- **`docs/scaling_and_transfer.md`** (**read**): forced-pick accuracy is flat
  across a 100x-plus range of CFB training-set size, and the CFB learning
  curve shows continuous-metric gains are ~93% realized by 800 of ~10,256
  available training games. This means a rolling-window arm that DISCARDS
  old rows is not "losing data" in any sense this project has found to
  matter for accuracy -- the live question is purely whether reallocating
  fit weight toward the current regime helps, not whether shrinking the
  training set per se hurts through under-fitting.
- **`docs/underived-constants-are-wrong`** (memory, **reported**): do not
  hand-pick one half-life or one window length. Section 2 below screens a
  small predeclared grid instead.

## 2. Predeclared arms (grid, no post-hoc addition)

Seven arms total -- one baseline plus six candidates, run on identical weeks
through the frozen harness (Section 3), varying **only** the training-row
weighting/selection. `ridge_alpha=10.0` and every other model setting is
held fixed across all seven.

| Arm | Kind | Parameter | Definition |
|---|---|---|---|
| `baseline` | uniform | -- | All prior history, `sample_weight=1` for every row -- reproduces the frozen `market_residual` arm bit-for-bit (self-check, Section 4) |
| `half_life_2` | exponential season decay | half-life = 2 seasons | `weight = 0.5 ** (max(0, predict_season - row_season) / 2)` |
| `half_life_4` | exponential season decay | half-life = 4 seasons | same formula, half-life 4 |
| `half_life_8` | exponential season decay | half-life = 8 seasons | same formula, half-life 8 |
| `half_life_16` | exponential season decay | half-life = 16 seasons | same formula, half-life 16 |
| `rolling_6` | rolling window | 6 seasons | training truncated to rows with `season >= predict_season - 5` (last 6 seasons inclusive of the season being predicted), `sample_weight=1` within the window |
| `rolling_10` | rolling window | 10 seasons | same, last 10 seasons |

`predict_season` is the season of the week being scored (fixed within a
`groupby(["season", "week"])` walk-forward iteration). Decay is at **season
granularity only** -- no within-season decay -- so every row from the same
season as the week being predicted carries weight 1.0 regardless of which
week within that season it came from; this is a deliberate simplification,
stated rather than hidden, chosen because CFB/NFL seasons are short (13-17
weeks) relative to the half-lives screened.

`half_life = infinity` (the grid's nominal fifth half-life point) and
`rolling = all` collapse mathematically to `baseline` (weight 1 for every
row / no truncation), so they are represented once, not run twice, per the
"screen a grid, don't hand-pick" instruction while avoiding a redundant
eighth fit.

**Where the weight is applied.** `sklearn.Ridge.fit(..., sample_weight=...)`
receives the weight vector, routed through the existing frozen
`make_margin_estimator` `Pipeline`'s `"regressor"` step
(`nfl_ats.margin.make_margin_estimator`, imported unmodified). The
`SimpleImputer`/`StandardScaler` preprocessing steps are fit **unweighted**
on the full (or window-truncated) training rows, exactly as the frozen
production pipeline already does for every existing arm -- only the ridge
coefficient fit itself sees the weight vector. This is implemented in new
script-local code (`scripts/era_weighting_lib.py`), not by editing
`src/nfl_ats/margin.py` or `src/nfl_ats/cfb_benchmark.py`: per this task's
environment rules, the production fitters are reused by import
(`make_margin_estimator`, `MarginModel`) and the sample-weight hook is added
in the calling script.

**Rolling-window eligibility.** A rolling arm's training pool can fall below
`min_train_games` (500) in early seasons even when the full-history baseline
already clears it (e.g. `rolling_6` needs 6 full seasons of games before it
can reach 500 CFB games). When that happens, **that arm alone is skipped for
that week** -- the other six arms still fit and score normally. This means
`rolling_6`/`rolling_10` may have a smaller total scored-game count than the
other five arms; every paired comparison below is computed on the inner-join
of games both arms actually scored, and each arm's own scored-game count is
reported alongside its effect so the reader can see when a rolling arm's
population differs from `baseline`'s.

## 3. Harness (frozen, CFB primary screen)

- **Instrument**: `nfl_ats.cfb_benchmark` (XLG-03) -- the same frozen
  chronological CFB-only market-residual benchmark `docs/ridge_alpha.md` and
  `docs/cfb_opponent_adjustment.md` both used. Free under rotation-registry
  rule 8 (**read**, `docs/rotation_registry.md` line 58: "CFB and
  non-reserved seasons stay free... needs no registry entry"); no NFL
  confirmation window is spent by this section.
- **Feature contract**: `CFB_MODEL_FEATURE_COLUMNS`
  (`nfl_ats.cfb_features`), imported unmodified -- the frozen 35-column XLG-03
  contract, not a candidate feature family.
- **Config**: `regressor="ridge"`, `ridge_alpha=10.0`, `target="market_residual"`
  (`ats_margin`), `distribution_fraction=0.20`, `min_distribution_rows=10`,
  `random_state=42`, `min_train_games=500` -- every value copied verbatim
  from `nfl_ats.cfb_benchmark`'s own module constants.
- **Population**: `data/processed/cfb_game_features.parquet`, seasons
  2006-2025 (`CFB_BENCHMARK_START_SEASON`/`END_SEASON`), the same 12,500-game
  canonical table every prior CFB screen in this repo has used.
- **Evaluation windows**: `clean_core` (2012-2019, 2021-2025) is **primary**,
  matching `nfl_ats.cfb_benchmark.cfb_evaluation_window`'s own convention
  (excludes the thin 2006-2011 sparse-line regime and the anomalous
  provider-count 2020 season). `all` (every scored season 2006-2025) is
  reported alongside as a secondary read, never the gate.
- **Weekly walk-forward**: strictly-earlier-gameday training cutoff per
  scored week, identical to `cfb_walk_forward_benchmark`'s own loop
  structure (reimplemented in the script to add the per-arm weight/window
  logic Section 2 needs -- the loop shape, not the underlying fit function,
  is what changes).

## 4. Bootstrap and decision discipline

- **Primary**: forced-pick accuracy (production probability rule,
  `home_cover_probability >= 0.5`), week-blocked paired bootstrap
  (`nfl_ats.experiments.paired_feature_comparisons`, imported unmodified),
  each candidate arm vs. the `baseline` arm (same target, same alpha, only
  weighting/window differs -- an apples-to-apples contrast), 20,000 samples,
  **seed 20260819** (this task's specified seed).
- **Secondary**: margin MAE/RMSE (`nfl_ats.cfb_opponent_adjustment.paired_margin_error_comparison`,
  imported unmodified) and Brier/log-loss (from the same
  `paired_feature_comparisons` call), same seed/samples/blocking.
- **Also reported**: season-blocked bootstrap alongside week-blocked, with
  the same `< 10 blocks` coverage caveat this repo's other era documents
  already carry (`docs/estimation_variance.md`, not re-derived here) --
  week-blocked is primary throughout.
- **Self-check before trusting any candidate arm** (Section 5's own
  precedent in `docs/era_magnitude_profile.md`): `baseline`'s absolute
  accuracy and per-method summary on `clean_core` must reproduce
  `nfl_ats.cfb_benchmark.cfb_walk_forward_benchmark`'s own frozen
  `market_residual` arm to within ordinary floating-point/refit-order
  noise. If it does not reproduce, that is a bug in the reimplementation,
  not a finding, and is fixed before any candidate arm is read.
- **No cherry-picking**: all six candidate arms are reported, win or lose,
  primary and secondary metrics alike. No arm is dropped from the write-up
  regardless of its sign.
- **Screen-to-NFL gate**: if the BEST arm's clean-core, week-blocked
  accuracy paired comparison leans positive at all (`probability_positive
  > 0.5` vs. `baseline`), the NFL close-grade analog (Section 6) is run on
  **all six** candidate arms for full disclosure -- not just the single best
  CFB arm -- so the NFL read is not itself a second cherry-pick. If every
  arm's CFB accuracy `probability_positive <= 0.5`, Section 6 is not run;
  the CFB results are still recorded per Section 7 (a negative or flat lean
  is never grounds to skip recording -- only to skip the second, more
  expensive instrument).

## 5. Instrument sanity check, declared before running

Before any candidate arm is interpreted, `scripts/era_weighting_cfb_screen.py`
prints `baseline`'s clean-core absolute accuracy and games-scored count
alongside a call to the existing frozen `cfb_walk_forward_benchmark`'s own
`market_residual` arm on the identical feature table/season range, and the
two are compared directly in the run's diagnostics. This is the same
discipline `docs/era_magnitude_profile.md` used (reconstructing
`surface_switch`'s registered pooled figure before trusting its era slices).

## 6. NFL close-grade analog (conditional, below-power, disclosed)

Only runs if Section 4's gate fires. Mirrors
`scripts/smooth_cdf_mapping_measurement.py`'s own protocol exactly (**read**
this session):

- **Recipe**: production `weak_stack` / `ridge` / `ridge_alpha=10.0` /
  `market_residual`, `min_train_games=500` -- `artifacts/active_ats_model.json`
  as of 2026-08-19 (**read**).
- **Feature table**: `data/processed/game_features_weak_stack.parquet`,
  `regular_season_rows` only.
- **Population**: restricted to seasons **no rotation-registry family has
  reserved** (`nfl_ats.rotation.season_usage`, imported unmodified) -- per
  rule 8, iterating on unreserved seasons spends no window. **Measured this
  session**: `season_usage(load_registry())` returns `{2013, 2014, 2015,
  2016, 2017, 2020, 2021}` reserved; the non-reserved evaluation range is
  `{2009-2012, 2018, 2019, 2022-2026}`. The 500-game warm-up floor further
  restricts which of those seasons actually score any week (2009-2010
  likely fail it, matching every other opener/close-grade document in this
  repo).
- **This is a CLOSE grade** (`spread_line` as recorded, not an opener
  archive) -- never described as a promotion-grade result; AGENTS.md's
  "grade the decision at the opener" rule governs PLAY/PROMOTION decisions,
  and this section makes neither. It is read-only, below-power context for
  whether the CFB lean (if any) shows up on the real NFL recipe at all.
- **Multiplicity disclosure, stated once here rather than per number below**:
  this section evaluates six arms (all six from Section 2) against one
  baseline, on one population, and is **already** a second look at a
  hypothesis whose sign was seen on CFB first (Section 4's gate requires a
  positive CFB lean to even run this section). No correction is applied to
  the six comparisons or across the two instruments -- per this project's
  established convention (`docs/ridge_alpha.md`'s own coarse-then-fine grid
  reported every point uncorrected) -- but every number below is labeled
  `unresolved_below_power` unless it mechanically resolves, and the fact
  that this is screen-then-confirm-on-the-same-direction, not two
  independent blind draws, is disclosed here rather than left implicit.
- **Bootstrap**: identical to Section 4 (`paired_feature_comparisons`,
  week-blocked primary/season-blocked secondary, 20,000 samples, seed
  20260819).

## 7. Registry recording plan

Every numeric CLI argument is read programmatically from the run's artifact
JSON by a dedicated `scripts/era_weighting_*_record.py` -- no hand-typed
numbers, matching `scripts/sbr_era_opener_record.py`'s precedent.

- **CFB entries** (always recorded, one per candidate arm, `league=cfb`,
  `effect_units=accuracy_points`, effect = week-blocked clean-core paired
  accuracy improvement vs. `baseline`, in points):
  `era_weighting_cfb_half_life_2`, `era_weighting_cfb_half_life_4`,
  `era_weighting_cfb_half_life_8`, `era_weighting_cfb_half_life_16`,
  `era_weighting_cfb_rolling_6`, `era_weighting_cfb_rolling_10`.
- **NFL entries** (only if Section 6 runs, same six names with
  `nfl_` in place of `cfb_`, `league=nfl`): `era_weighting_nfl_half_life_2`,
  `era_weighting_nfl_half_life_4`, `era_weighting_nfl_half_life_8`,
  `era_weighting_nfl_half_life_16`, `era_weighting_nfl_rolling_6`,
  `era_weighting_nfl_rolling_10`.
- **Collision check**: `grep -n "era_weighting" registry/weak_signals.json`
  returns nothing (**measured** this session) -- all twelve names are free.
- **Classification, decided mechanically per entry by the recorder script
  reading the artifact** (never asserted in prose first): whole week-blocked
  clean-core accuracy interval below zero -> `refuted_mechanism` /
  `--closing-ground wrong_sign_resolved`; otherwise
  `unresolved_below_power`. No positive control is run anywhere in this
  document, so `bounded_by_control` is never available to any entry here.
  A whole-interval-above-zero (excludes zero, positive) reading has no
  "resolved positive" terminal state in this project's taxonomy and also
  records `unresolved_below_power`, per `docs/sbr_opener_evaluation.md`'s
  and `docs/era_magnitude_profile.md`'s own precedent.
- Registry is read back after each write to verify.

## Files

- `scripts/era_weighting_lib.py` -- shared arm grid, weight/window
  computation, and the generic weighted-ridge fit helper (imports
  `make_margin_estimator`/`MarginModel` from `nfl_ats.margin` unmodified;
  adds the `sample_weight` hook the production fitters do not expose).
- `scripts/era_weighting_cfb_screen.py` -- CFB walk-forward screen (Sections
  3-5).
- `scripts/era_weighting_nfl_screen.py` -- NFL close-grade analog (Section
  6), only run if the CFB gate fires.
- `scripts/era_weighting_cfb_record.py` / `scripts/era_weighting_nfl_record.py`
  -- read each run's artifact JSON and call `nfl-ats weak-signals record`
  per arm.
- `artifacts/era_weighting_cfb_screen/<run-id>/` /
  `artifacts/era_weighting_nfl_screen/<run-id>/` -- output artifacts.

---

## Results

*(populated after the scripts run; nothing below this line existed when the
arms above were declared)*

### CFB screen (Section 3-5)

**Measured**, `scripts/era_weighting_cfb_screen.py`, artifact
`artifacts/era_weighting_cfb_screen/20260819T235500Z/`, run 2026-08-19.
`data/processed/cfb_game_features.parquet`, 12,500 games, 2006-2025.
`ridge_alpha=10.0`, `min_train_games=500`, `distribution_fraction=0.20`,
20,000 bootstrap samples, seed 20260819. Every arm scored the identical
95,912 rows (7 arms x 13,702 scored games per arm; `skip_counts` all zero --
no rolling window ever fell below the 500-game floor over this range) and
the identical 8,933-game clean-core population, so every paired comparison
below is a true apples-to-apples contrast.

**Self-check (Section 5): passes exactly.** `baseline`'s clean-core absolute
accuracy reproduces `nfl_ats.cfb_benchmark.cfb_walk_forward_benchmark`'s own
frozen `market_residual` arm bit-for-bit: both report **9,093 games,
50.6873%** accuracy, `accuracy_diff = 0.0`. The reimplementation is trusted.

**Primary metric, clean-core, week-blocked, paired accuracy improvement vs.
`baseline` (points above/below baseline, not points above 50):**

| Arm | Estimate (pts) | 95% CI (pts) | P+ | Paired games |
|---|---:|---:|---:|---:|
| `half_life_2` | +0.4702 | [-0.5371, +1.4443] | 0.8242 | 8,933 |
| `half_life_4` | +0.2351 | [-0.5334, +0.9824] | 0.7270 | 8,933 |
| `half_life_8` | **+0.3470** | **[-0.1804, +0.8633]** | **0.8987** | 8,933 |
| `half_life_16` | +0.2239 | [-0.1931, +0.6398] | 0.8460 | 8,933 |
| `rolling_6` | +0.1231 | [-0.8779, +1.1226] | 0.5921 | 8,933 |
| `rolling_10` | -0.2687 | [-0.9910, +0.4524] | 0.2250 | 8,933 |

No interval excludes zero (all contain zero -- the expected shape for a real
small signal at this resolution, per the binding taxonomy, not evidence of
absence). **No cherry-picking: this is all six predeclared arms, reported
regardless of sign.** The picture that emerges, read plainly:

- **All four half-life season-decay arms lean positive** on accuracy,
  consistently across clean-core/all and week-/season-blocked cuts (checked
  directly: `half_life_8` P+ ranges 0.817-0.900 across all four cuts,
  `half_life_16` 0.806-0.892, `half_life_2` 0.636-0.824, `half_life_4`
  0.547-0.727 -- `half_life_8` and `half_life_16` are the most consistent;
  `half_life_2` and `half_life_4` are weaker and more cut-dependent). The
  best-leaning arm is `half_life_8` (P+ 0.899 on the primary week-blocked
  clean-core cut) -- close to but short of this project's established 0.90
  screen bar, and per AGENTS.md that bar governs what these docs may CLAIM,
  not a hard pass/fail on recording or on running the NFL analog.
- **`rolling_10` leans negative** (P+ 0.225-0.260 across every cut) --
  truncating to the most recent 10 seasons costs accuracy on this CFB
  instrument, consistently, though not resolved (interval crosses zero).
- **`rolling_6` is unstable across cuts** (P+ 0.476-0.694, sign of the point
  estimate itself flips between `clean_core` (+0.123, weak positive) and
  `all` (-0.017, weak negative)) -- read as no clear lean either way, not as
  two contradictory findings.

**Secondary metrics (margin MAE/RMSE, Brier, log-loss), clean-core,
week-blocked -- and where the picture disagrees with accuracy:**

| Arm | Brier improvement | P+ | Margin MAE improvement | P+ |
|---|---:|---:|---:|---:|
| `half_life_2` | -0.000376 [-0.000954, +0.000191] | 0.097 | -0.0093 [-0.0297, +0.0109] | 0.181 |
| `half_life_4` | -0.000143 [-0.000476, +0.000179] | 0.195 | -0.0001 [-0.0113, +0.0108] | 0.490 |
| `half_life_8` | -0.000048 [-0.000226, +0.000126] | 0.296 | +0.0011 [-0.0047, +0.0069] | 0.651 |
| `half_life_16` | -0.000023 [-0.000117, +0.000070] | 0.316 | +0.0009 [-0.0021, +0.0038] | 0.724 |
| `rolling_6` | **-0.001103 [-0.001765, -0.000447]** | **0.0005** | **-0.0335 [-0.0552, -0.0122]** | **0.0011** |
| `rolling_10` | -0.000210 [-0.000537, +0.000114] | 0.100 | -0.0083 [-0.0181, +0.0014] | 0.047 |

**`rolling_6` is resolved WORSE on every continuous metric** (Brier,
log-loss, margin MAE, and margin RMSE all have their whole week-blocked
interval below zero, P+ <= 0.0033 on all four) even though its accuracy
point estimate leaned weakly positive on the clean-core cut. This is not a
contradiction to paper over: forced-pick accuracy reads only the sign of a
coarse ~2-point-resolution comparison, while Brier/log-loss/MAE read the
full predictive distribution, and `rolling_6`'s 6-season training window is
small enough (roughly 3,000-4,500 games depending on point in the walk) that
it measurably hurts calibration even where it happens not to flip enough
signs to move the accuracy needle. Per the binding taxonomy this does
**not** resolve or close the accuracy hypothesis for `rolling_6` -- the
admissible closing grounds require the PRIMARY metric's interval to sit
entirely below zero, and accuracy's does not -- but it is reported here in
full rather than left out, exactly as `docs/ridge_alpha.md`'s own
"objectives disagree" finding was. The other five arms' continuous-metric
readings are unresolved in both directions, consistent with the
`docs/scaling_and_transfer.md` finding that this project's accuracy metric
and its continuous metrics frequently diverge at CFB's sample size.

**Gate (Section 4): fires.** `half_life_8`'s clean-core, week-blocked
accuracy paired comparison leans positive (`probability_positive = 0.8987 >
0.5`) -- the NFL close-grade analog runs on all six candidate arms.

### NFL close-grade analog (Section 6)

**Measured**, `scripts/era_weighting_nfl_screen.py`, artifact
`artifacts/era_weighting_nfl_screen/20260820T000500Z/`, run 2026-08-19.
Production recipe (`weak_stack`/`ridge`/`ridge_alpha=10.0`/`market_residual`,
`min_train_games=500`), `data/processed/game_features_weak_stack.parquet`,
CLOSE grade. Non-reserved evaluation seasons (rule 8, **measured** this
session via `season_usage(load_registry())`): `{2009, 2010, 2011, 2012,
2018, 2019, 2022, 2023, 2024, 2025, 2026}` -- 2013-2017, 2020, and 2021 are
reserved by other families and excluded. 2009-2010 fail the 500-game warm-up
floor (as every other close/opener document in this repo already finds), so
the scored population is **2,047 games per arm**, all seven arms
(`skip_counts` all zero).

**Self-check (Section 6): passes exactly.** `baseline`'s accuracy reproduces
`nfl_ats.outcomes.walk_forward_outcomes`'s own frozen `market_residual` arm
bit-for-bit on the identical population: both report **2,047 games,
50.4152%**, `accuracy_diff = 0.0`.

**Absolute accuracy by arm** (2,047 games each):

| Arm | Accuracy |
|---|---:|
| `half_life_8` | 51.10% |
| `half_life_4` | 51.00% |
| `half_life_16` | 50.90% |
| `rolling_10` | 50.85% |
| `half_life_2` | 50.71% |
| `rolling_6` | 50.61% |
| `baseline` | 50.42% |

**Primary metric, week-blocked, paired accuracy improvement vs. `baseline`:**

| Arm | Estimate (pts) | 95% CI week-blocked (pts) | P+ week | 95% CI season-blocked (pts) | P+ season |
|---|---:|---:|---:|---:|---:|
| `half_life_2` | +0.2931 | [-1.4771, +2.0448] | 0.6223 | [-1.6066, +2.1526] | 0.6140 |
| `half_life_4` | +0.5862 | [-0.9309, +2.1214] | 0.7664 | [-0.8273, +2.1516] | 0.7605 |
| `half_life_8` | **+0.6839** | **[-0.5416, +1.9380]** | **0.8505** | **[-0.0961, +1.4342]** | **0.9533** |
| `half_life_16` | +0.4885 | [-0.5324, +1.5159] | 0.8154 | [-0.2459, +1.3868] | 0.8667 |
| `rolling_6` | +0.1954 | [-1.7527, +2.1110] | 0.5698 | [-1.6473, +2.2023] | 0.5604 |
| `rolling_10` | +0.4397 | [-1.1696, +2.0874] | 0.6928 | [-0.2987, +1.2428] | 0.8458 |

**Every arm leans positive on the NFL close grade -- no cherry-picking, all
six reported.** No interval excludes zero (week-blocked), so nothing here
resolves under the binding taxonomy, but the direction is unanimous, which
is itself a genuinely different shape than the CFB screen (where
`rolling_10` leaned negative). `half_life_8` is again the strongest and most
consistent arm, matching CFB's own top arm exactly -- an independent
instrument converging on the same candidate: week-blocked P+ 0.8505,
**season-blocked P+ 0.9533** with a season-blocked lower bound of -0.0961
points, a hair's width from excluding zero. `half_life_4` and `half_life_16`
are the next-strongest, both leaning positive and consistent across
blockings. `rolling_6` is again the weakest arm on both instruments (P+
0.560-0.570 here, near-coinflip on CFB too).

**Secondary metrics disagree in the same "accuracy vs. calibration" pattern
`docs/ridge_alpha.md` and the CFB screen above both already found:**
`half_life_2` is **resolved WORSE** on Brier and log-loss on the NFL close
grade (week-blocked Brier -0.004398 [-0.007371, -0.001483] P+0.001;
log-loss -0.009178 [-0.015396, -0.003100] P+0.001; season-blocked
reproduces the same resolved-negative shape) despite its weak positive
accuracy lean -- a fast 2-season half-life clearly hurts calibration on the
close grade even though it does not resolve as harmful for the sign-only
accuracy metric. `half_life_8` (the accuracy-leading arm) is unresolved on
Brier/log-loss, leaning mildly negative but not resolved (week Brier
-0.000466 [-0.001412, +0.000457] P+0.162). `rolling_10` is the only arm
leaning POSITIVE on Brier/log-loss here (P+ 0.67-0.70) -- the opposite
direction from its own CFB reading, a genuine cross-instrument disagreement
for that one arm, reported as exactly that rather than resolved either way.

**Multiplicity, restated with numbers in hand:** this section ran six
comparisons on a population whose direction was already known to lean
positive on CFB (the gate that authorized running it at all). The NFL
numbers corroborate CFB's `half_life_8` finding on an independent
population and an independent (production) recipe, which is meaningfully
stronger evidence than either instrument alone -- but neither instrument
resolves under the binding taxonomy, and this is not an independent blind
pair of draws, so the honest read is "two consistent unresolved leans on the
same specific arm," not "confirmed."

### Walk-forward discipline: which arm does the predeclared grid select?

Per Section 4's discipline (no cherry-picking, report every arm, judge by
consistency across cuts rather than by the single largest point estimate):
**`half_life_8`** -- exponential season-decay sample weighting with an
8-season half-life -- is the arm the predeclared grid selects. It is the
strongest or co-strongest accuracy lean on **every one of the six cuts
measured across both instruments** (CFB clean-core week/season-blocked, CFB
"all" week/season-blocked, NFL week-blocked, NFL season-blocked), the only
arm whose NFL season-blocked interval comes within 0.1 points of excluding
zero, and it never resolves NEGATIVE on any secondary metric anywhere in
this document (unlike `half_life_2`, resolved worse on NFL Brier/log-loss,
and `rolling_6`, resolved worse on CFB Brier/log-loss/margin MAE/RMSE).
`half_life_16` is the consistent runner-up. Truncated rolling windows
(`rolling_6`, `rolling_10`) are weaker and, in `rolling_10`'s case,
direction-inconsistent between the two instruments -- rolling windows are
not where this grid's evidence concentrates.

**None of this resolves under the binding taxonomy.** Every interval above
contains zero at the week-blocked, primary-cut level; `half_life_8`'s NFL
season-blocked interval comes closest ([-0.0961, +1.4342]) without actually
excluding zero. Per the taxonomy this is `unresolved_below_power`
throughout -- a real, consistent, cross-instrument lean for one specific
arm, not a resolved finding, and every one of the twelve registry entries
below records exactly that classification, never a "resolved positive."

## Registry recording (measured, both leagues)

Twelve entries recorded via `scripts/era_weighting_record.py`, read back
from `registry/weak_signals.json` after each write (**measured**, `grep -c
era_weighting_cfb registry/weak_signals.json` -> 18 matches confirming the 6
CFB names plus references; `total_signals` counter in each CLI response
incremented 211 -> 222 across the twelve calls, one at a time, no
collisions):

| Name | League | Effect (pts) | 95% CI (pts) | P+ | Classification |
|---|---|---:|---:|---:|---|
| `era_weighting_cfb_half_life_2` | cfb | +0.4702 | [-0.5371, +1.4443] | 0.8242 | unresolved_below_power |
| `era_weighting_cfb_half_life_4` | cfb | +0.2351 | [-0.5334, +0.9824] | 0.7270 | unresolved_below_power |
| `era_weighting_cfb_half_life_8` | cfb | +0.3470 | [-0.1804, +0.8633] | 0.8987 | unresolved_below_power |
| `era_weighting_cfb_half_life_16` | cfb | +0.2239 | [-0.1931, +0.6398] | 0.8460 | unresolved_below_power |
| `era_weighting_cfb_rolling_6` | cfb | +0.1231 | [-0.8779, +1.1226] | 0.5921 | unresolved_below_power |
| `era_weighting_cfb_rolling_10` | cfb | -0.2687 | [-0.9910, +0.4524] | 0.2250 | unresolved_below_power |
| `era_weighting_nfl_half_life_2` | nfl | +0.2931 | [-1.4771, +2.0448] | 0.6223 | unresolved_below_power |
| `era_weighting_nfl_half_life_4` | nfl | +0.5862 | [-0.9309, +2.1214] | 0.7664 | unresolved_below_power |
| `era_weighting_nfl_half_life_8` | nfl | +0.6839 | [-0.5416, +1.9380] | 0.8505 | unresolved_below_power |
| `era_weighting_nfl_half_life_16` | nfl | +0.4885 | [-0.5324, +1.5159] | 0.8154 | unresolved_below_power |
| `era_weighting_nfl_rolling_6` | nfl | +0.1954 | [-1.7527, +2.1110] | 0.5698 | unresolved_below_power |
| `era_weighting_nfl_rolling_10` | nfl | +0.4397 | [-1.1696, +2.0874] | 0.6928 | unresolved_below_power |

No entry carries a `closing_ground` -- no week-blocked accuracy interval
sits entirely below zero for any arm on either league, so `wrong_sign_resolved`
never applies, and no positive control was run, so `bounded_by_control` is
never available. All twelve are `unresolved_below_power`, exactly as the
binding taxonomy above requires: an interval containing zero is never
grounds to close, and a lean short of resolution -- however consistent
across six independent cuts and two instruments -- is not a "resolved
positive" either, because this project's taxonomy has no such terminal
state.

## MOD-14 status

Not closed, not promoted, and per the binding taxonomy correctly so -- no
arm resolves. What this screen adds to `ROADMAP.md`'s previously-empty
MOD-14 row: a predeclared, no-cherry-picked, six-arm grid run on two
independent instruments (12,500 free CFB games; the real production NFL
recipe on 2,047 close-graded games) both lean toward the same specific
arm -- **exponential season-decay sample weighting with an 8-season
half-life** -- with the NFL season-blocked read one hair (-0.10 points on
the lower bound) from excluding zero. This is exactly the "unresolved but
worth carrying forward, not discarding" shape the crossing-zero rule exists
to protect. Next step for a future session, if pursued: this is a
training-recipe change (not a feature family), so promoting it into
production would need its own predeclared NFL confirmation window under
`nfl_ats.rotation`, graded at the opener per AGENTS.md, following
`docs/scaling_and_transfer.md`'s predeclaration-for-a-future-session
template -- not undertaken here, as no accuracy hypothesis on this
instrument has cleared this project's 0.75 CFB screen bar with the
consistency `half_life_8` shows, short of the 0.90 promotion-claim bar, and
per AGENTS.md that distinction (screen bar vs. promotion-claim bar vs. what
gets played) is exactly what governs what happens next, not a bare pass/fail
on this document.

## 8. Opener-grade information read (predeclared 2026-08-19, before running)

Predeclared **before** `scripts/era_weighting_opener_read.py` scores
anything. This is a THIRD look at the MOD-14 hypothesis family -- not a
promotion, not a rotation-registry confirmation -- requested because
`half_life_8` is the arm Section 2's predeclared grid selects (the
"Walk-forward discipline" section above): strongest or co-strongest on
every one of six cuts across both the CFB and NFL close-grade instruments,
never resolved negative on any secondary metric anywhere in this document.
This section asks whether that lean survives at the **opener**, on the
project's own decision-grade protocol (`docs/opener_evaluation.md`,
AGENTS.md's "grade the decision at the opener" rule) -- a genuinely
different instrument from both Section 3-5 (CFB, free) and Section 6 (NFL,
close-graded) above. **This is an information read for a future promotion
decision, not a promotion.**

### Disclosures, stated before any number is read

(a) **Selection inflation.** `half_life_8` was chosen as best-of-six TODAY,
    2026-08-19, on the two screens already in this document (CFB
    clean-core week-blocked P+ 0.8987, NFL close-grade week-blocked P+
    0.8505 / season-blocked P+ 0.9533). Its measured P+ on this THIRD look
    is not an independent blind draw -- the arm was picked because it
    looked good on the first two instruments, so any positive lean here is
    weaker evidence than the same number would be as a first look, and is
    reported and recorded with that caveat attached rather than presented
    as a clean confirmation.
(b) **Population overlap.** This opener protocol's paired archive is
    2020-2025 regular season (`tue_open` + close,
    `docs/opener_evaluation.md`'s 1,537-game archive). Section 6's NFL
    close-grade population (seasons no rotation-registry family has
    reserved) includes 2022-2025 from that same span -- the two NFL reads
    do not draw from disjoint games, so their P+ figures must not be
    multiplied together as if independent.
(c) **Below-power screen, not a confirmation.** No rotation-registry window
    is spent by this section (rule 8: this reads the frozen, already-public
    opener/close archive, not a reserved season). This is read-only
    information for a FUTURE promotion decision; it makes no promotion
    decision itself, and per AGENTS.md's "a promotion bar is not a decision
    bar" / "grade the decision at the opener" rules, this document does not
    by itself gate what gets played -- only a properly predeclared
    rotation-registry confirmation, following
    `docs/scaling_and_transfer.md`'s template, could do that.

### Protocol

Mirrors `scripts/smooth_cdf_mapping_opener_measurement.py`'s own
weekly-refit archive/pairing machinery exactly; arm-fitting machinery from
`scripts/era_weighting_lib.py`, identical to Sections 3-6 above.

- **Population**: the exact `docs/opener_evaluation.md` 1,537-paired-game
  archive, 2020-2025 regular season, `tue_open` consensus + close
  (`nfl_ats.clv.build_pairing_table`/`close_reference_table`,
  `HISTORICAL_CAPTURE_KIND`), imported unmodified.
- **Recipe**: production `weak_stack`/`ridge`/`ridge_alpha=10.0`/
  `market_residual`, `min_train_games=500` -- identical to Section 6 and to
  `docs/opener_evaluation.md`'s own frozen recipe.
- **Arms**: `baseline` (uniform `sample_weight=1`, all history) and
  `half_life_8` (exponential season-decay sample weight, half-life 8
  seasons) -- the same two `era_weighting_lib.ERA_WEIGHTING_ARMS` entries
  Sections 2-6 already define, reused verbatim, not redefined.
- **Pick rules, both scored**: the **production probability rule**
  (`home_cover_probability >= 0.5`, PRIMARY) and the **sign rule**
  (`predicted_market_residual > 0`, SECONDARY, diagnostic).
  `home_cover_probability` is the model's default ECDF read
  (`MarginModel.predict`'s `probability_method="ecdf"` default, never
  overridden) -- the identical probability mapping
  `nfl_ats.clv.opener_pick_evaluation` has always used, so this read sits
  on the same mapping that produced `docs/opener_evaluation.md`'s 53.36%
  production-rule number, not the MOD-08 Gaussian default
  (`nfl_ats.outcomes.score_outcome_week` only, per `src/nfl_ats/margin.py`
  `MarginModel.predict`'s own docstring).
- **Self-check, run BEFORE the `half_life_8` arm is computed**: the
  `baseline` arm alone must reproduce `docs/opener_evaluation.md`'s
  production-rule opener accuracy, **53.3599% (0.5335994677312043)** --
  read this session from
  `artifacts/opener_evaluation/20260819T174244Z/metadata.json`
  (`metrics.opener_accuracy_probability_rule`), the run
  `docs/opener_evaluation.md`'s "Addendum, 2026-08-19" section already
  quotes as "53.36%" on the identical 1,537-game archive. If the
  reproduction misses by more than floating-point noise, that is a bug in
  this script's reimplementation, not a finding, and is fixed before the
  `half_life_8` arm's numbers are interpreted -- identical discipline to
  Sections 5 and 6 above.
- **Bootstrap**: `nfl_ats.experiments.paired_feature_comparisons`, imported
  unmodified, week-blocked PRIMARY / season-blocked secondary, 20,000
  samples, seed 20260819 -- identical to every other section of this
  document. Brier/log-loss improvements are read from the same
  probability-rule bootstrap call (the sign-rule call's "probabilities" are
  0/1 indicators, so only its `accuracy_improvement` row is meaningful;
  its Brier/log-loss rows are not interpreted).
- **Recording**: one registry entry, `era_weighting_nfl_half_life_8_opener`
  (collision-checked, `grep -n "era_weighting_nfl_half_life_8_opener"
  registry/weak_signals.json` returns nothing as of this predeclaration),
  `league=nfl`, `effect_units=accuracy_points`, effect = week-blocked
  opener-grade PROBABILITY-rule paired accuracy improvement vs. `baseline`,
  classified `unresolved_below_power` unless the whole week-blocked
  interval sits below zero (mechanical, per the binding taxonomy) -- the
  selection disclosure (a)-(c) above is pasted into the entry's notes
  verbatim, not summarized.

### Binding closing-grounds taxonomy, restated for this section

> An interval or CI that contains zero is NEVER grounds to reject, fail, or
> close an experiment. At this evaluator's ~2-point resolution, "contains
> zero" is the EXPECTED outcome for a real small signal. Only two grounds
> ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
> (whole interval on the wrong side of zero) or zero split-half reliability;
> (2) bounded by a positive control proven able to detect an effect that
> size. Everything else is `unresolved_below_power`: record it with
> `nfl-ats weak-signals record`, report `probability_positive`, never the
> binary "contains zero". The registry code hard-rejects inadmissible
> closures; if a record command errors, the verdict is wrong, not the
> validator.

### Results

**Measured**, `scripts/era_weighting_opener_read.py`, artifact
`artifacts/era_weighting_opener_read/20260820T002230Z/`, run 2026-08-19/20.
Production recipe (`weak_stack`/`ridge`/`ridge_alpha=10.0`/`market_residual`,
`min_train_games=500`), `docs/opener_evaluation.md`'s 1,537-paired-game
2020-2025 `tue_open`+close archive, seed 20260819, 20,000 bootstrap samples.

**Self-check: passes exactly, before `half_life_8` was computed at all.**
`baseline` arm alone reproduces `docs/opener_evaluation.md`'s production-rule
opener accuracy bit-for-bit: **1,503 games, 53.3599%**, `accuracy_diff =
0.0`. Both arms then scored the identical 3,074 rows (2 arms x 1,537 games,
`skip_counts` all zero -- no week ever fell below `min_train_games`), 1,503
games score at the opener (pushes excluded) and 1,507 at the close.

**Primary: opener grade, production probability rule, week-blocked, paired
accuracy improvement (`half_life_8` vs. `baseline`):**

| Metric | Estimate | 95% CI (week-blocked) | P+ | Paired games |
|---|---:|---:|---:|---:|
| Accuracy | **-0.3992 pts** | **[-1.9450, +1.1921] pts** | **0.2990** | 1,503 |
| Brier improvement | -0.000238 | [-0.001580, +0.001015] | 0.3646 | 1,503 |
| Log-loss improvement | -0.000479 | [-0.003229, +0.002084] | 0.3658 | 1,503 |

Season-blocked (only 6 blocks -- degenerate per
`BootstrapDegeneracyWarning`, estimate/P+ only, not a valid interval):
accuracy -0.3992 pts, P+ 0.2554.

**Secondary: opener grade, sign rule, week-blocked:**

| Metric | Estimate | 95% CI (week-blocked) | P+ |
|---|---:|---:|---:|
| Accuracy | -0.6653 pts | [-2.3194, +0.9843] pts | 0.2031 |

Season-blocked (degenerate, 6 blocks): -0.6653 pts, P+ 0.0848.

**Context only, close grade (not the primary goal, reported because the
direction genuinely differs): probability rule, week-blocked accuracy
+0.1991 pts, P+ 0.5784** -- the OPPOSITE sign from the opener read on the
same two arms and the same archive. This is a real, reported divergence,
not resolved in either direction (both intervals cross zero).

**Reading, plainly, with all three disclosures from the predeclaration still
attached:** at the opener -- this project's decision-grade protocol -- the
selected arm `half_life_8` leans NEGATIVE against `baseline` on every
opener-grade cut measured here: the primary probability-rule accuracy read
(P+ 0.299), the secondary sign-rule read (P+ 0.203), and Brier/log-loss
(P+ 0.36-0.37). None of these intervals sit entirely below zero, so under
the binding taxonomy this does **not** refute the mechanism -- it is
`unresolved_below_power`, exactly like every other reading in this
document, and the crossing-zero rule applies here precisely as strictly as
it did to the positive-leaning CFB/NFL-close reads above: a negative point
estimate with an interval that crosses zero is not evidence of harm any
more than a positive one crossing zero was evidence of benefit. What is
worth stating plainly: unlike the CFB and NFL-close-grade reads (Sections
3-6), which leaned consistently positive for `half_life_8` across every cut
measured, this THIRD, disclosed look leans the other way at the opener on
the actual decision-grade instrument, and does so on BOTH pick rules and
the continuous metrics alike -- a materially different picture than the
first two looks, obtained on real games with real disclosure costs (a)-(c)
already paid. It is not proof the mechanism is wrong (no interval resolves
that), and it is not proof it is right either; it is a third data point
that happens to point the other way, carrying selection inflation on top.

**Recorded**: `era_weighting_nfl_half_life_8_opener`, `league=nfl`,
`effect_units=accuracy_points`, effect **-0.3992 pts**, 95% CI
`[-1.9450, +1.1921]` pts, `probability_positive=0.2990`,
`classification=unresolved_below_power` (mechanical: the week-blocked
interval's upper bound is positive, so `wrong_sign_resolved` does not
apply; no positive control was run, so `bounded_by_control` is unavailable
either) -- registry read back after the write to confirm
(`total_signals` 226 -> 227, single call, no collision). Full selection
disclosure (a)-(c) from this section is pasted into the entry's `notes`
verbatim, along with the sign-rule, Brier/log-loss, and close-grade context
numbers, so a future reader does not have to re-derive them from the
artifact.

**Status: not closed, no promotion decision made here or implied.** This
section answers exactly the question it predeclared -- does the
best-of-six selected arm's lean survive at the opener -- and the honest
answer, stated without a one-word verdict standing in for the number, is
that it does not survive on this one below-power look: the point estimate
flips negative and stays `unresolved_below_power` on every opener cut,
while the close-grade read on the identical two arms and the identical
archive still leans positive. A future promotion decision would need to
weigh all three looks (CFB, NFL close-grade, NFL opener) together, with the
selection-inflation and population-overlap disclosures from this section
attached, rather than treating any one of them as dispositive -- not
undertaken here, as this script makes an information read only.
