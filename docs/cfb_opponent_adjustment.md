# CFB dimension-neutral opponent-adjustment SUBSTITUTION screen — predeclaration

Predeclared 2026-08-17 (US), written and frozen BEFORE any candidate feature
was built, any arm was fit, or any outcome was scored. Constants below are
mirrored in `src/nfl_ats/cfb_opponent_adjustment.py`. Results are appended in
a separate section and never edit this one.

Rotation registry: **not touched**. Rule 8 ("CFB and non-reserved seasons stay
free") governs; no NFL confirmation window is spent by this screen.

## The gap being closed

The active NFL model consumes raw per-play team rates. On 2009-2012 NFL data,
split-half reliability of team per-play EPA is ~0.80 (offense) but only ~0.46
(defense), so the defensive per-play columns are roughly half noise.
Opponent adjustment is the textbook fix and is already implemented in this
repo (`src/nfl_ats/opponent_adjustment.py`, PBP-05: a weekly, time-decayed
ridge offense/defense decomposition).

PBP-05 was judged "no stable improvement" — but every recorded test **added**
the adjusted columns on top of the raw ones, inflating the design from 58 to
106-142 columns on ~4,700 rows at ridge alpha 10. That confounds the signal
question with a dimension-inflation variance penalty. **The dimension-neutral
substitution — remove the raw columns, put the adjusted ones in their place,
identical column count — has never been run.** This screen runs it, on CFB,
where the clean core is 8,933 games rather than ~2,000.

## Hypothesis

If opponent adjustment measures the same football quantity as the raw
per-play rate but with less schedule-induced noise, then **substituting** the
adjusted columns for the raw ones — leaving the design width, the regressor,
the penalty, the walk-forward, and the splits untouched — should reduce
margin error. If it does not, the earlier "no stable improvement" verdict was
about the adjustment itself, not about dimension inflation, and the family
closes.

## Frozen recipe

### Baseline arm (`cfb_benchmark_v1`)

The frozen XLG-03 CFB benchmark, byte-identical: `fit_cfb_residual_model` on
`CFB_MODEL_FEATURE_COLUMNS` (**35 columns**: 2 market, 7 context, 2
experience, 24 team-state), Ridge alpha 10, median-imputed + standardized
pipeline, no calibration, trailing-20% out-of-time residual pool, weekly
refits with strictly-earlier training and a 500-game minimum, seasons
2006-2025.

### Candidate arm (`cfb_benchmark_v1_opponent_adjusted`)

The identical 35-column contract with exactly six columns swapped in place:

| Removed (raw) | Substituted (opponent-adjusted) |
|---|---|
| `home_off_epa_per_play` | `home_off_epa_per_play_adjusted` |
| `away_off_epa_per_play` | `away_off_epa_per_play_adjusted` |
| `diff_off_epa_per_play` | `diff_off_epa_per_play_adjusted` |
| `home_def_epa_per_play` | `home_def_epa_per_play_adjusted` |
| `away_def_epa_per_play` | `away_def_epa_per_play_adjusted` |
| `diff_def_epa_per_play` | `diff_def_epa_per_play_adjusted` |

**35 columns in, 35 columns out.** Every other column, the regressor, alpha,
the residual-distribution recipe, the walk-forward loop, the week set, and
the splits are identical between arms. The other 18 team-state columns
(success rate, explosive rate, pace, offense and defense) are untouched;
only the EPA/play pair has an opponent-adjusted equivalent.

### The adjustment (opponent-adjusted columns)

Fit with the SHARED estimator core extracted from
`nfl_ats.opponent_adjustment` — the same code the NFL PBP-05 path runs, not a
second copy:

- Per (season, week), a weighted Ridge decomposes each observed team-game
  offensive EPA/play into `intercept + offense(team) + defense(opponent)`.
- **Point-in-time contract, preserved exactly from the NFL implementation**:
  eligible history is rows strictly earlier in (season, week) order AND with
  `gameday` strictly before the scored week's earliest kickoff. No game from
  the scored week — including earlier days of that week — can enter its own
  fit.
- Time decay: exponential, **half-life 16 weeks** (NFL value verbatim).
- Penalty: **ridge alpha 10** (NFL value verbatim).
- Warm-up: **minimum 64 eligible team-games** before a week is fit (NFL value
  verbatim); below it the adjusted columns stay NaN for that week.
- Column values: `home_off_..._adjusted = intercept + offense(home)`,
  `home_def_..._adjusted = intercept + defense(home)`, likewise for away, and
  `diff_* = home_* - away_*`. These are the opponent-purged analogues of the
  raw team-state rates, on the same per-play EPA scale — deliberately NOT the
  NFL module's *matchup expectation* (`intercept + offense(home) +
  defense(away)`), which would make `home_off` and `away_def` the same number
  and collapse four columns to two.
- Teams are ESPN team ids; history is the canonical XLG-03 FBS-vs-FBS
  team-game table (`build_cfb_team_game_metrics`) restricted to canonical
  benchmark games.
- No CFB-specific tuning of any kind. All three parameters are the NFL
  values, taken verbatim in the repo's established CFB convention.

### Evaluation

- Walk-forward: XLG-03 protocol verbatim, three arms scored on **identical
  weeks and identical games** (`market`, `market_residual` = baseline,
  `market_residual_opponent_adjusted` = candidate).
- **Primary metric: margin error** — paired per-game **MAE improvement**
  (baseline |error| − candidate |error|, positive = candidate better) and
  **RMSE improvement**, on the `clean_core` split (2012-2019 + 2021-2025, the
  8,933-game headline). Continuous, far more power than accuracy at this n.
- Uncertainty: **week-blocked and season-blocked paired bootstrap, 2,000
  samples, seed 20260817**, reporting the estimate, the 95% interval, and
  `probability_positive` (the fraction of blocked resamples favoring the
  candidate). **Never a bare pass/fail.**
- Secondary: cover **Brier**, **log loss**, and forced-pick accuracy
  improvements via `nfl_ats.experiments.paired_feature_comparisons` on the
  same rows, same blocks, same seed.
- Reported splits: `clean_core` (headline), `thin_2006_2011`, `regime_2020`,
  `all` — the same splits the `cfb_role_continuity` and MOD-16 screens used.

## Frozen decision rule

One run. No parameter, column, target, or split retuning after seeing any
result; any variant is a NEW predeclaration, not a rerun.

- **Substitution rescues opponent adjustment** only if the clean-core
  week-blocked paired **margin MAE improvement** is positive with a 95%
  interval whose lower bound exceeds zero (equivalently
  `probability_positive` decisively above 0.975), with the season-blocked
  interval and RMSE coherent in sign.
- **Unresolved at this sample size** if the point estimate favors the
  candidate but the interval spans zero.
- **Closed** if the point estimate favors the baseline. Combined with
  PBP-05's additive negative, that closes opponent adjustment as a feature
  family for this program: neither adding it nor substituting it helps, so
  the earlier verdict was not an artifact of dimension inflation.

Secondary metrics are coherence checks and cannot overturn the primary rule
in either direction.

## Declared limitations, stated before the run

1. **Missingness differs between arms by construction.** Raw team-state
   columns need three prior games (NFL maturity rule verbatim); the adjusted
   columns exist as soon as a week clears the 64-team-game warm-up, and a
   team with no eligible history receives effect 0.0 (i.e. the league
   intercept) rather than NaN — the NFL module's behavior, preserved. The
   candidate therefore has fewer missing values. Both arms impute with the
   same median imputer, so this is a real (and legitimate, point-in-time)
   part of what substitution buys, not a leak. The per-arm missing rate and
   the effective post-imputer design width (`SimpleImputer(add_indicator=True)`
   appends one indicator per column that is missing in training, so the two
   arms can differ in *effective* width even at identical *input* width) are
   both reported.
2. **The team universe is taken from the full table**, so the design carries
   all-zero columns for teams not yet seen. Ridge sets those coefficients to
   exactly zero and leaves the rest unchanged, so this is structural, not an
   outcome leak — but it is inherited from the NFL implementation and is
   recorded here rather than silently fixed.
3. **A large win is a leak until proven otherwise.** The leakage regression
   test (release-blocking, `tests/test_cfb_opponent_adjustment.py`) is the
   gate: perturbing current-week and future plays must leave every adjusted
   column bit-identical, and shuffling input order must too.
4. This is a CFB screen. It licenses (or refuses) an NFL predeclaration; it
   is not itself an NFL result and spends no NFL window.

---

## Results (run 2026-08-17, artifacts in the session scratchpad)

Recorded once, against the frozen rule above. Nothing in the predeclaration
was edited after seeing a number. Scale: 280 scored weeks, 11,989 scored
games, 9,093 in the clean core (8,933 of them non-push, matching the XLG-03
headline sample exactly).

### Dimension neutrality actually held

| | baseline | candidate |
|---|---|---|
| input columns | 35 | 35 |
| effective design width after `SimpleImputer(add_indicator=True)` | 63 | 63 |
| clean-core missing rate, substituted columns | 0.23%-0.60% | 0.00% |
| correlation with the raw column it replaced | — | 0.90-0.94 |

The declared risk that the arms could differ in *effective* width did not
materialize: both land at 63. The adjusted columns are fully populated where
the raw ones are ~99.5% populated, as predicted.

### Primary metric — margin error, clean core, paired

| block | metric | improvement (points) | 95% interval | `probability_positive` |
|---|---|---|---|---|
| week | MAE | **-0.0003** | [-0.0069, +0.0063] | **0.463** |
| week | RMSE | **-0.0027** | [-0.0091, +0.0036] | **0.213** |
| season | MAE | -0.0003 | [-0.0053, +0.0044] | 0.467 |
| season | RMSE | -0.0027 | [-0.0070, +0.0014] | 0.103 |

Absolute levels on the clean core: market 12.2411 MAE, baseline 12.2564,
candidate 12.2567. The substitution moves margin error by three ten-thousandths
of a point on 9,093 games.

Other splits (week-blocked MAE improvement): `thin_2006_2011` -0.0180
(P=0.149), `regime_2020` -0.0206 (P=0.006), `all` -0.0047 (P=0.148). Every
split points the same direction or nowhere.

### Secondary metrics — cover probability, clean core, 8,933 paired games

Accuracy +0.00045 (P=0.548), Brier +0.000017 (P=0.562), log loss +0.000032
(P=0.559); season-blocked figures agree. All three nominally favor the
candidate and all three are indistinguishable from zero.

### Verdict: CLOSED

The frozen rule says *closed* when the primary point estimate favors the
baseline. It does, on both MAE and RMSE, under both blockings. Combined with
PBP-05's additive negative, **opponent adjustment is closed as a feature
family for this program**: adding the columns does not help, and substituting
them does not help either, so the original verdict was not an artifact of
dimension inflation. The substitution framing was worth building — it was a
real confound and it is now eliminated — but it does not rescue the family.

### The instrument was not asleep: a deliberate leak IS detected

The honesty guardrail runs in both directions. A null is only worth reporting
if the harness could have seen a win, so `--leak-probe` scored a
deliberately contaminated arm: the identical substitution, with the
adjustment fit **once on the entire 2006-2025 history**, so the columns can
see every game they score, including the future.

| arm | clean-core MAE improvement | `probability_positive` | RMSE improvement | `probability_positive` |
|---|---|---|---|---|
| honest (the screen) | -0.0003 | 0.463 | -0.0027 | 0.213 |
| deliberately leaked | **+0.0129** | **0.984** | **+0.0181** | **0.998** |

Two things follow. First, the screen has real detection power at roughly the
0.013-point scale — the same games, same bootstrap, same seed flag the leaked
arm at P=0.98-0.998 while the honest arm sits at 0.46/0.21, so the null is
measured, not underpowered. Second, and more damning for the family: **even
cheating with perfect future knowledge of team quality is worth only ~0.013
points of margin error here.** That is the ceiling on the entire family in a
market-residual design, and the honest version reaches none of it.

### Post-hoc control: which half of the substitution did what?

Identified after freezing, and stated plainly: the substitution changes *two*
things at once. It adjusts for opponent, **and** it swaps the raw columns'
span-8 exponentially weighted game window for a 16-week calendar half-life.
A null could in principle be two real effects cancelling. So a third,
clearly post-hoc arm was run through the identical harness — same decay, same
penalty, same warm-up, same cutoffs, opponent block removed — to separate
them (clean core, week-blocked):

| comparison | MAE improvement | P | RMSE improvement | P |
|---|---|---|---|---|
| time weighting only (raw EWM → 16-week decay) | -0.0008 | 0.400 | -0.0038 | 0.113 |
| opponent block only (decay → decay + opponent) | **+0.0005** | 0.679 | **+0.0012** | 0.820 |
| both together (= the frozen screen) | -0.0003 | 0.463 | -0.0027 | 0.213 |

The opponent block, isolated, does point the right way — but at +0.0005 MAE
it is operationally zero, roughly a tenth of what the maximal leak buys, and
the weighting change bundled with it in the frozen substitution costs about
as much. This is the strongest form of the negative: it is not that the
adjustment was hidden by a confound, it is that the adjustment is worth
~0.001 points of margin error against a market line.

### Why the reliability argument did not translate

The motivating fact stands (NFL defensive per-play EPA split-half reliability
~0.46 vs ~0.80 for offense). What this screen shows is that halving the noise
in a column does not help a model whose target is the *residual from the
market line*. The spread is in the design and already prices team quality; the
team-state columns are largely redundant with it either way, so making them
more reliable has almost nothing left to improve. The leak probe puts a hard
number on that redundancy: perfect knowledge of season-long team quality is
worth 0.013 points of margin error.

### Consequences

- No NFL predeclaration for opponent-adjusted features is licensed. The
  rotation registry was not touched and no confirmation window was spent
  (rule 8).
- The shared estimator core extracted for this screen
  (`fit_opponent_effects`, `eligible_opponent_history`,
  `opponent_adjustment_weeks`) stays: it is verified byte-identical to the
  previous NFL implementation on the full 4,902-game NFL table, and it means
  there is one leakage contract to audit rather than two.
- A revisit would need a different *target*, not a better adjustment — the
  ceiling measured here binds any variant of the family.
