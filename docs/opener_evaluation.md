# Opener-graded evaluation of the frozen active model — predeclaration

Predeclared: 2026-08-17 (US), before the run. This is the first measurement
under the project's clarified **primary goal (Ryan, 2026-08-17): beat the
opening line his football pool (Splash Sports) grades against.** Closing
lines and vig-beating are secondary. The frozen constants and recipe are in
`src/nfl_ats/clv.py` (section 8); results are appended below and never edit
the predeclaration.

## Question

The active model's 52.05% was graded against closing lines — the market at
its sharpest. The replicated MKT-06 sign test showed the close moves
*toward* the model's fair margin in 54.9% of games, which implies the same
model should grade **better against the Tuesday opener** than against the
close. How much better, measured honestly?

## What this is and is not

- It scores the **frozen incumbent only** (`market_residual`/`player`/
  ridge/alpha-10, weekly refits with strictly-earlier training — the exact
  recipe behind the 52.05% evaluation). Zero selection dimensions: no
  variants, no tuning, one run.
- It reads 2020–2025 outcomes, which lie inside the ledgered 2018–2025 era.
  That is admissible here precisely because nothing is being selected —
  this is a re-measurement of an already-fixed pick stream generator at a
  different settlement line, predeclared with one look. No candidate model
  gains or loses standing from this run.
- Declared approximation (inherited from the MKT-06 pilot machinery): only
  `spread_line` is swapped to the opener; all other features (including
  `total_line`) are close-era values. The result approximates "the active
  model if the market had stopped at Tuesday's opener."

## Frozen recipe

For every archived game with a paired `tue_open` consensus and a resolvable
close (2020–2025 historical snapshot archive; 227–272 games per season):
one weekly-refit model per (season, week), trained on completed games
strictly before the week's first kickoff (≥500 games), then:

- **Opener arm**: residual evaluated at the opener; forced pick = sign of
  the residual; settled against the opener (`result − tue_open`).
- **Close arm**: same model's residual evaluated at the close; forced pick
  settled against the close. (Grading convention identical to the 52.05%
  evaluation.)
- **Movement oracle** (diagnostic upper bound): pick the side the close
  eventually moved toward, settle at the opener — how much accuracy pure
  line-movement capture is worth.

Pushes are excluded per arm; the paired opener-minus-close delta uses games
non-push under both lines.

## Frozen reporting (no accept/reject gate — this is a measurement)

`opener_accuracy`, `close_accuracy`, `opener_minus_close` (paired),
`opener_vs_coin_flip`, and `movement_oracle_accuracy`, each with week- and
season-blocked bootstrap intervals AND `probability_positive` (fraction of
blocked resamples above zero — continuous evidence, not a binary verdict),
2,000 samples, seed 20260817. Per-season table reported. Whatever the
numbers say, nothing is retuned on them; they set the baseline that any
future pool-targeted candidate must beat at the opener.

## Interpretation contract, fixed in advance

- `opener_accuracy` is the project's headline **pool-relevant** number if
  Splash-style pool lines are set near the Tuesday opener and frozen; it
  overstates pool edge if pool lines post later in the week (a separate
  Splash Sports timing investigation is running).
- If `opener_minus_close` comes back ≈ 0 or negative, the line-movement
  signal does not translate into settlement advantage and the pool edge
  claim rests on `opener_vs_coin_flip` alone.

---

## Results — `player` profile (run 2026-08-17, artifact `artifacts/opener_evaluation/20260817T135624Z/`, 1,537 paired games 2020–2025)

**Superseded as the active model's number by the `weak_stack` run below
(added 2026-08-18) — kept here because it is still the correct number for
the `player` profile specifically, and other documents may cite it.** The
active model was promoted from `player` to `weak_stack` on 2026-08-17-18
(commit `68b4dc0`); the dashboard's findings page headline (52.83%/51.56%)
comes from the `weak_stack` run, not this one, and previously had no run in
this document to cite for that number.

| Metric | Estimate | Week-blocked 95% | Season-blocked 95% | P(positive) |
|---|---|---|---|---|
| **Opener accuracy (the pool number)** | **52.50%** | [49.90%, 55.08%] | [50.21%, 54.31%] | **96.8% / 98.6%** (vs coin flip) |
| Close accuracy (same games) | 51.09% | [48.32%, 53.66%] | [48.92%, 53.14%] | — |
| **Opener minus close (paired)** | **+1.35 pts** | [+0.41, +2.35] | [+0.49, +2.14] | **99.95% / 99.8%** |
| Movement oracle at opener | 55.08% | [52.26%, 57.82%] | [53.99%, 56.19%] | — |

## Results — `weak_stack` profile, the ACTIVE model (run 2026-08-18, artifact `artifacts/opener_evaluation/20260818T013115Z/metadata.json`, 1,537 paired games 2020–2025)

This is the run behind the dashboard findings page's headline 52.83%/51.56%
numbers (`src/nfl_ats/dashboard/findings_content.py`'s `HEADLINE`). Same
recipe, same predeclaration, same 1,537-game paired archive as the `player`
run above; only the active model's `feature_profile` changed (`player` →
`weak_stack`, ridge alpha 10.0 in both).

| Metric | Estimate | Week-blocked 95% | Season-blocked 95% | P(positive) |
|---|---|---|---|---|
| **Opener accuracy (the pool number)** | **52.83%** | [50.34%, 55.36%] | [50.98%, 54.83%] | **98.8% / 100%** (vs coin flip) |
| Close accuracy (same games) | 51.56% | [48.91%, 54.12%] | [49.83%, 53.49%] | — |
| **Opener minus close (paired)** | **+1.28 pts** | [+0.34, +2.29] | [+0.21, +2.14] | **99.5% / 98.6%** |
| Movement oracle at opener | 55.08% | [52.26%, 57.82%] | [53.99%, 56.19%] | — |

Per season (opener accuracy): 2020 46.8%, 2021 55.5%, 2022 51.2%,
2023 54.1%, 2024 53.8%, 2025 52.8% — positive in five of six seasons. The
exception, 2020, is the COVID season: the archive's thinnest slice (227
paired games vs 272; bookmaker coverage itself was fine at ~11 books per
consensus) and a genuine regime break — empty stadiums collapsed
home-field advantage league-wide, which a model trained on pre-2020
seasons would systematically misprice. Mean absolute
open-to-close movement and the oracle ceiling (55.1%) indicate the model
captures roughly half the value of perfectly foreseeing line movement.

### Reading, per the interpretation contract

- **The same frozen model is resolvably better against the opener than
  against the close**: the paired +1.35-point delta carries ~99.9%
  probability of being positive under both blockings. The replicated
  line-movement signal does translate into settlement advantage — grading
  at the close was systematically understating the model.
- **Against the coin flip at the opener — the primary pool question — the
  model sits at 52.5% with a ~97–99% probability of genuine positive
  skill**, and the season-blocked interval excludes 50%. This is the
  strongest goal-relevant evidence the project has produced.
- Splash Sports context (researched 2026-08-17): their legacy pool engine
  posts lines Tuesday morning, revises once Wednesday, then freezes them
  for the week at half-point numbers, identical for all entrants — very
  close to the `tue_open` label measured here. The 52.5% figure is
  therefore approximately the pool-relevant grade, subject to the
  Wednesday revision (which pulls the pool number slightly toward the
  eventual close and would land true pool accuracy between the opener and
  close grades).
- Caveats: 1,537 games over six seasons; the close-era-features
  approximation declared above; one predeclared look, nothing retuned.

### Consequence

The project's headline metric changes: pool-targeted candidates are now
graded at the opener (`opener-evaluation` is the yardstick), and 52.50%
at the opener replaces 52.05%-at-the-close as the incumbent's number for
the primary goal. Prospective 2026 scoring should record both grades —
the live Tuesday captures make the opener grade available in production.

**Addendum, 2026-08-18:** the paragraphs above (and this section) describe
the `player`-profile run and read as the current incumbent's number as of
2026-08-17. The active model was promoted to `weak_stack` on 2026-08-17/18
(commit `68b4dc0`), and the "Results — `weak_stack` profile" table above is
now the incumbent's actual number for the primary goal: **52.83% opener /
51.56% close**, season-blocked opener interval **[50.98%, 54.83%]**. The two
profiles' readings agree qualitatively (opener beats close, opener beats a
coin flip with high confidence, per-season and movement-oracle behavior
essentially unchanged) — nothing in the "Reading" section's *interpretation*
is wrong for `weak_stack`, only its specific point estimates, which belong
to `player`.

## Addendum, 2026-08-19: the instrument graded the wrong pick rule

This evaluation (and the `52.83%`/`52.50%` numbers above, artifact
`artifacts/opener_evaluation/20260818T013115Z/`) has always graded picks with
the **sign rule**: `residual_at_open > 0`. Production — `pool.py` and
`backtest.py` — has always played a different rule: the **probability
rule**, `home_cover_probability >= 0.5`. The two usually agree but do not
have to: `home_cover_probability` is the share of the model's empirical
out-of-time residual distribution landing above the line, so its 0.5
crossing tracks that distribution's *median*, while the sign rule tracks the
point prediction, i.e. the distribution's *mean* (via `MarginModel.predict`,
`src/nfl_ats/margin.py`). They coincide only when that residual distribution
is symmetric.

`src/nfl_ats/clv.py`'s `opener_pick_evaluation` and `opener_evaluation_metrics`
now compute the probability rule alongside the sign rule, additively —
`home_cover_probability_at_open`/`_at_close`,
`pick_home_at_open_probability_rule`/`_at_close_probability_rule`,
`correct_at_open_probability_rule`/`_at_close_probability_rule` on the
per-game frame, and `opener_accuracy_probability_rule`,
`close_accuracy_probability_rule`, `opener_minus_close_probability_rule`,
`opener_vs_coin_flip_probability_rule` in the metrics dict. The sign-rule
fields are untouched — same values, same names.

**Measured, 2026-08-19**, by running `nfl-ats opener-evaluation --features
data/processed/game_features_weak_stack.parquet --feature-profile weak_stack
--regressor ridge --ridge-alpha 10.0` fresh, in an isolated scratch
artifacts/registry location (the tracked
`artifacts/opener_evaluation/20260818T013115Z/` directory was not touched or
regenerated). The run reproduces that tracked artifact's sign-rule numbers
exactly — `opener_accuracy` 52.8277% (52.83%), `close_accuracy` 51.5594%
(51.56%), `opener_minus_close` +1.28pts, on the identical 1,537 paired
2020–2025 games (season split 227/239/255/272/272/272) — confirming this is
the same computation, not a drifted re-run. On that same run, **the
probability rule scores `opener_accuracy_probability_rule` 53.3599%
(53.36%) at the opener vs the sign rule's 52.83%** (`close_accuracy_probability_rule`
52.09%, `opener_minus_close_probability_rule` +1.55pts, week-blocked P+
0.9985 / season-blocked P+ 0.9995 for that probability-rule opener-minus-close
delta). This matches `docs/pool_edge_plan.md`'s close-graded read that the
probability rule beats the sign rule by +2.12 points (P+ 0.990) — production
has always played the probability rule, so 53.36% is what production would
have scored on this archive, not a new claim of edge. This is a
post-hoc instrument-fidelity note relative to the 2026-08-17 predeclared
sign-rule protocol, not a re-run or a re-selection: nothing above is
retuned, and the sign rule stays the predeclared historical record.

**Owner decision, 2026-08-19 (same day, after reading the above):** the
public site's headline leads with the production-rule grade — 53.4% at the
opener, 52.1% at the close — because it grades the rule every published pick
has actually used, with the sign-rule protocol figure (52.8%) retained
alongside as provenance on every surface that shows it. A real (tracked-tree)
`opener-evaluation` run was executed for this
(`artifacts/opener_evaluation/20260819T174244Z/`, reproducing both rules'
numbers above exactly); the site generator and the findings content module
prefer the `*_probability_rule` metrics when an artifact carries them and
fall back to sign-rule fields on older artifacts. Season-by-season under the
production rule, all six seasons finish above the coin flip (2020 52.3%,
2021 55.1%, 2022 53.2%, 2023 54.1%, 2024 54.9%, 2025 50.6%), and the
season-blocked interval [51.97%, 54.56%] excludes it.
