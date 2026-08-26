# Graph ratings v2: predeclaration

Written before this engine ever touches an ATS outcome, per the binding rule
that a design gets predeclared before it gets scored. **No ATS number
appears in this document.** Everything below is either a design decision
(edge construction, weighting, constraint, arms, split) or a **measured**,
non-ATS coherence diagnostic used only to compare structural hyperparameters
against each other on the CFB corpus.

Engine: `src/nfl_ats/graph_ratings_v2.py`. Tests:
`tests/test_graph_ratings_v2.py` (36 tests, all passing). This is a NEW
exploration lane, not a patch to `nfl_ats.graph_ratings`
(`add_schedule_strength_features`), which is untouched and keeps powering
production. Nothing in `graph_ratings_v2.py` is imported by `features.py` or
any production pick path. Another agent is separately screening which
statistics qualify as model INPUTS
(`docs/graph_input_screen.md`); this document and this module are the
ENGINE only.

## 1. Why a rebuild, not a patch

Two measured defects in the original module (**read**,
`src/nfl_ats/graph_ratings.py` lines 266-269 and its production use):

1. **Magnitude compression.**
   `performance[...] += 1.0 + min(numeric_margin, 28.0) / 14.0` squeezes
   every game's edge weight into `[1.0, 3.0]` and treats every margin above
   28 identically. A one-point win and a forty-point win are nearly the
   same edge.
2. **Raw-margin edges measure team quality.** This project has a measured
   ceiling (**read**, `docs/pool_edge_plan.md` section "The standing lesson:
   measuring teams better is bounded near zero" -- "the market... prices
   team quality -- that is the one thing it is unambiguously good at") that
   features which only measure team quality better are bounded near zero,
   because the market already prices quality well. `graph_pagerank` (the original module's PageRank column)
   was reported — by the task brief that commissioned this rebuild, not
   independently re-verified in this session — at a 0.997 split-half
   reliability, i.e. an extremely repeatable measurement of something the
   market already prices. That reported figure motivates keeping a
   raw-margin arm as a **positive control**, not a claim this module
   asserts on its own authority.

## 2. Edge construction

Both arms below share one mechanism; only the per-game **signal** differs.

- **Residual arm** (`edge_signal="residual"`, default): signal =
  `ats_margin = result - spread_line` (home perspective; positive means
  home beat its number). This already exists as a column in both league
  tables before this module runs (`features.py` calls `add_ats_outcomes`
  before `add_schedule_strength_features`; `cfb_game_features.parquet`
  carries it natively — **read**, `data/processed/cfb_game_features.parquet`
  column list). Edges therefore rank **who beats their number**, not who is
  good.
- **Raw-margin control arm** (`edge_signal="raw_margin"`): signal =
  `result` alone, the exact quantity the original module used. Same engine,
  same weighting, same constraint — the only thing that changes is which
  column feeds the edges. This isolates "does using the market residual
  change what the instrument measures" from every other design choice.
  `test_residual_and_raw_margin_arms_agree_when_the_market_is_uninformative`
  proves the two arms are mechanically identical when `spread_line == 0`
  everywhere (`ats_margin == result` by construction), so any divergence
  between the arms on real data comes from the market signal, not from a
  hidden implementation difference.

For a game between home team H and away team A with signal `s` (after the
optional injury discount, section 5):

```
W[H, A] += s
W[A, H] += -s
```

Branch-free by construction: a push (`s == 0`) contributes exactly zero to
both cells with no special case, and the sign of `s` alone determines which
team is endorsed and which is detracted — no separate tie-handling branch,
no floor. `test_exact_pushes_carry_no_signal` proves the zero case;
`test_residual_arm_ranks_who_beats_the_number_not_who_wins` is the
known-answer test: a team (`CHALK`) that always wins the scoreboard by 14
but was favored by 24 every time (`ats_margin = -10` every game) ranks
**below** a team (`DOG`) that always loses the scoreboard by 14 but was a
24-point underdog every time (`ats_margin = +10` every game) under the
residual arm, and the ranking **inverts** under the raw-margin control arm.
That inversion is the direct evidence the two arms measure different
things.

The non-negative arm (`propagation="nonneg_pagerank"`, section 4) uses the
same signal but stores only `abs(s)` on the loser->winner edge (ties
contribute nothing, matching the signed arm) — this is the original
module's edge direction convention, uncompressed.

## 3. Weighting: uncompressed magnitude

The weight **is** the signal, in point units — `s` itself for the signed
arm, `abs(s)` for the non-negative arm. No floor, no cap, no compression
into a narrow band.

- **Ties**: `s == 0` contributes zero to both directions. A push carries no
  directional information, so this is not a special case bolted on — it
  falls out of the same formula (`W[H,A] += s` with `s = 0` is a no-op).
- **Blowouts**: uncapped. A 40-point story contributes roughly 40x what a
  1-point story contributes, not ~1.4x as the original's `[1.0, 3.0]` band
  would have. `test_uncompressed_magnitude_scales_the_rating_gap` proves
  this on the Katz primitive directly (see the note there on why it must be
  tested on the raw, unstandardized centrality rather than through the
  full standardized pipeline: with only two teams, cross-sectional z-scoring
  always yields an exact +-2.0 gap regardless of the underlying magnitude —
  population std with n=2 is exactly half the raw gap — which would mask
  the property rather than demonstrate it).
- Linear (identity) was chosen over a damped shape (sqrt, log1p) because (a)
  it is the most literal reading of "uncompressed," (b) it keeps the
  synthetic validation exact (weight equals signal, so a known-answer test
  can predict the matrix by hand), and (c) any runaway-blowout risk is
  handled at the row level, not the edge level (section 4) — deliberately
  separating "how big is one game's evidence" from "how much can one row
  trust its biggest single piece of evidence," which a per-edge damping
  function would conflate. A damped shape (e.g. `sign(s) * sqrt(abs(s))`)
  is a legitimate documented alternative for a future robustness screen but
  is not implemented here; committing to one weighting per the task's
  instruction, linear is it.

## 4. Signed propagation: Katz, not PageRank, and the row-Linf constraint

The owner asked whether bounding learned weights to `[-1, 1]` helps or
hurts propagation. **It hurts, and the reason is specific:**

- PageRank's convergence guarantee comes from the transition matrix being
  row-**stochastic** (rows sum to exactly 1), which gives spectral radius
  exactly 1 and a unique stationary distribution (Perron-Frobenius, which
  needs non-negativity).
- Clipping every entry of an N-node **signed** matrix to `[-1, 1]` lets a
  single row's absolute sum reach N. The spectral radius can then reach N,
  and the power iteration `x = v + alpha*W@x` diverges for any fixed
  `alpha < 1` once N is large enough that `alpha*N >= 1`.
- The actual guarantee is on the **operator norm** `||W||_inf` (the maximum
  row sum of absolute values): constraining that to `<= max_row_l1` bounds
  the spectral radius by the same amount (spectral radius `<=` any induced
  matrix norm). `_constrain_row_linf` (`graph_ratings_v2.py`) implements
  this: rows already under the bound are untouched; only rows that exceed
  it are rescaled down to exactly the bound, direction preserved. This is
  **more expressive** than entry-clipping, not less — one dominant edge is
  allowed to carry a row instead of every edge being forced small.
  `test_row_linf_constraint_rescales_only_rows_over_the_bound` proves the
  rescale-only-when-over-bound behavior and direction preservation.
- `GraphRatingV2Config.validate()` enforces `alpha * max_row_l1 < 1`
  explicitly — that product **is** the spectral-radius bound this module
  guarantees (`test_alpha_times_max_row_l1_must_be_strictly_below_one`).
  Default: `alpha=0.85`, `max_row_l1=1.0` (spectral-radius bound 0.85).

With negative entries, the fixed point `x = (I - alpha*W)^-1 v` is **not**
PageRank: there is no random walk, no stationary distribution, and no
"importance" interpretation once mass can be negative. This module names it
**signed Katz centrality** throughout (`signed_katz_centrality`,
`_katz_fixed_point`), and **drops HITS from the signed arm entirely**: HITS
needs Perron-Frobenius for a unique interpretable top eigenvector, which
needs non-negativity. HITS survives only on the non-negative arm
(`propagation="nonneg_pagerank"`), reproducing the original module's
offense/defense split with uncompressed edges.

**The convergence test** (the task's explicit requirement — a test that
*fails when the spectral radius constraint is violated*):
`test_katz_fixed_point_diverges_when_spectral_radius_bound_is_violated`
feeds `_katz_fixed_point` (the low-level iterator, exposed unconstrained on
purpose) a 5-node matrix with every row's absolute sum exactly 5 and
`alpha=0.85` (`alpha * 5 = 4.25`, far past the bound) and asserts the
iteration does **not** converge; the same matrix, passed through
`_constrain_row_linf` first, converges. This is direct evidence the
constraint is load-bearing, not decorative.
`test_signed_katz_matches_the_closed_form_linear_solve` separately proves
the iteration correctly implements `x = (I - alpha*W)^-1 v` by comparing
against `np.linalg.solve` on a random 8-node signed matrix.

## 5. Injury as an edge modifier, never a per-game attribution

The owner asked how much of an unexpected result could be attributed to a
specific injury. **Per-game attribution is not identifiable**: one
observation, many possible causes, no counterfactual game played without
the injury. This module does not attempt it, and says so in both the code
docstring and here.

What **is** estimable is a single shared coefficient. When the side that
fell short of the spread was also carrying heavy `injury_value_lost`
(`docs/injury_value_lost.md`'s construct; split-half reliability **read**
this session at 0.87-0.93 there, section 3.1; the task brief that
commissioned this rebuild cited 0.93 specifically -- **reported**, not
independently re-derived this session), discount that game's edge so
the graph does not conclude the opponent is strong from a game played
against a shorthanded team:

```
discount = 1 / (1 + beta * underperformer_injury_value_lost)
```

`underperformer_injury_value_lost` is read from whichever side actually
fell short (home's total when `s < 0`, away's when `s > 0` — see
`_injury_discount`); a push (`s == 0`) needs no discount since it already
contributes nothing. `beta` is a single scalar, shared across every game —
never a per-game or per-player attribution.

**Off by default** (`injury_beta=0.0`): `test_injury_modifier_is_a_true_no_op_at_beta_zero`
proves a dataset with real injury columns present but `injury_beta=0.0`
produces byte-identical output to a dataset lacking those columns entirely.
`test_injury_modifier_discounts_the_underperformers_edge_when_enabled`
proves the discount has a real, measurable effect once enabled (see that
test's docstring for the row-Linf-constraint interaction it had to control
for: a heavily-capped row can hide a weak discount, which is a real and
expected property of the two mechanisms composing, not a bug).

`beta` itself is **not** auto-tuned by this module and is deliberately
excluded from the CFB structural-fitting mechanism (section 6): CFB has no
player-level injury table, so this coefficient can only be fit on NFL
directly. Per this project's binding rule against inventing constraints,
fitting `beta` is deferred to a future session with its own leak-safe,
predeclared procedure. Until then, any caller that enables the injury arm
must supply `beta` explicitly and disclose it as undertuned.

## 6. Fitting where the parameters are affordable (CFB -> NFL, the XLG pattern)

`data/processed/cfb_game_features.parquet` holds **12,500 games** (measured
this session, `season` 2006-2025) versus NFL's **4,902** (measured,
`data/processed/game_features.parquet`, `season` 2009-2026). This project
has an established cross-league transfer pattern documented in
`docs/scaling_and_transfer.md` (the "XLG" family): fit structure where
there is enough data to support it, transfer the structure, refit only what
must differ per league.

**What transfers:** the structural hyperparameters that have many degrees
of freedom relative to NFL's smaller n — `alpha`, `half_life_weeks`,
`offseason_retention`, `max_row_l1`, `prior_weight`. **What refits:**
everything league-specific — the actual team-level ratings, always rebuilt
fresh from that league's own graph; `injury_beta`, which has no CFB analog
(section 5); and `edge_signal`/`propagation`, which are experimental arms
rather than tuned parameters and are compared on NFL directly once outcome
scoring begins.

`select_structural_config_on_cfb` / `cfb_structural_coherence`
(`graph_ratings_v2.py`) implement the mechanism: a leak-safe walk-forward
Pearson correlation between the pregame rating diff and the game's own
signal (`ats_margin` for the residual arm, `result` for the control),
computed on the CFB corpus. This is **explicitly not an ATS accuracy
number** — no call in this module touches `nfl_ats.weak_signals.record_signal`
or any rotation-registry command, and this document does not either.

### Measured demonstration (non-ATS, informational)

Run this session on the real CFB corpus (`select_structural_config_on_cfb`,
default `signed_katz` propagation, `min_games=16`):

| edge_signal | alpha | half_life_weeks | max_row_l1 | coherence |
|---|---|---|---|---|
| residual | 0.50 | 8.0 | 1.0 | -0.000287 |
| residual | 0.85 | 8.0 | 1.0 | -0.001013 |
| residual | 0.85 | 4.0 | 1.0 | -0.005681 |
| residual | 0.85 | 16.0 | 1.0 | +0.004119 |
| raw_margin | 0.85 | 8.0 | 1.0 | **+0.531023** |

A follow-up check (**measured**, same session) varied `max_row_l1` from 1.0
to 10.0 to 50.0 (with `alpha` scaled down to keep `alpha*max_row_l1 < 1`) for
the residual arm at `half_life_weeks=8.0`: coherence stayed at -0.00101,
-0.00093, -0.00373 respectively — the near-zero reading is not an artifact
of the row cap.

**Reading this honestly, not just reporting it:** the residual (market-
relative) signal shows essentially no linear graph-recoverable serial
structure on CFB at any tested hyperparameter setting, while the raw-margin
control shows a strong, robust +0.53 coherence. This is the SAME pattern
this project has found before by other instruments (raw team quality is
highly serially predictable; market-relative residuals are close to
white noise once the market has priced a team) — it is not a contradiction
of this module's premise, it is a second confirmation of it, and it is
good news for the control arm's validity as a control (it visibly recovers
the thing it is supposed to recover). It is a **caution**, not a closure,
for the residual arm's eventual NFL accuracy: a near-zero linear
correlation with next-game `ats_margin` does not rule out a real, useful
**sign**-agreement signal for forced-pick accuracy (a coarse, ~2-point
instrument per AGENTS.md) — correlation and forced-pick accuracy are
different questions, and per the binding closing-grounds taxonomy, this
non-ATS diagnostic reading zero is not evidence to reject or close the
residual arm. It is recorded here as what it is: a structural-fit
diagnostic, not a verdict.

The hyperparameter grid above is a small, illustrative candidate set (5
configs), not an exhaustive search — expanding it is future work, not
something this predeclaration authorizes as a tuning pass against NFL
outcomes.

## 7. Arms, summarized

| arm dimension | values | default | purpose |
|---|---|---|---|
| `edge_signal` | `residual` / `raw_margin` | `residual` | treatment (beats-the-number) vs. positive control (team quality) |
| `propagation` | `signed_katz` / `nonneg_pagerank` | `signed_katz` | signed Katz centrality vs. the non-negative PageRank+HITS arm |
| `injury_beta` | `0.0` (off) / `>0.0` | `0.0` | optional edge modifier; never a per-game attribution |

Every combination shares the same leak-safe weekly walk-forward
(`add_graph_ratings_v2_features`), namespaced output columns
(`katz_feature_columns`/`default_column_prefix`) so multiple arms can be
computed side by side on the same frame without colliding, and the same
uncompressed-magnitude, row-Linf-constrained mechanics.

## 8. Leakage

Two leakage regression tests mirror the pattern already proven in
`tests/test_graph_ratings.py` for the original module, extended to the new
inputs (`spread_line`, injury columns):

- `test_current_week_outcomes_cannot_change_current_ratings`: violently
  perturbing a week's own `result`/`spread_line` leaves that week's own
  already-computed ratings byte-identical (they were read from the graph
  as accumulated through the PRIOR week), and does change the following
  week's ratings (proving the perturbation is not simply inert).
- `test_future_outcomes_spread_and_input_order_cannot_change_prior_ratings`:
  blanking a future week's `result`/`spread_line`/scores and shuffling row
  order leaves every prior week's ratings unchanged.
- `test_future_injury_values_cannot_change_prior_ratings`: the same
  future-blanking discipline, specifically for the new injury-modifier
  arm's input columns.

Read of the leak-safety mechanism itself (not just its test coverage): the
weekly loop computes and assigns ratings for every game in a week from the
graph BEFORE that week's own results (or injury values) are absorbed into
the accumulator, exactly mirroring the original module's proven structure.

## 9. What has NOT been done, and the split for when it is

**No ATS number has been produced by this module or this document.** Per
the task's explicit instruction, the engine is built and validated on
synthetic graphs with known answers first (section 2's CHALK/DOG inversion,
section 3's magnitude scaling, section 4's convergence proofs, section 8's
leakage proofs — 36 tests total, `tests/test_graph_ratings_v2.py`), and
real-data use in this document is limited to the non-ATS structural
coherence diagnostic (section 6).

When the input-screen agent's statistic list exists and NFL outcome scoring
begins, the split discipline is:

- **Selection and scoring must not share a window.** Per
  `docs/overlay_subset_holdout_v2.md` (shrinkage factor measured worsening
  from 0.636 to 0.593 as a search widened), any structural or arm selection
  against NFL outcomes must be predeclared and use a disjoint window from
  the window used to report a final number.
- **Structural hyperparameters are frozen from the CFB fit** (section 6),
  not retuned on NFL — mirroring `docs/scaling_and_transfer.md`'s
  "Frozen parameters... no retuning" pattern for the joint/hierarchical/
  prior-mean arms. A future session may run a wider CFB grid before
  freezing, but the frozen values must come from CFB, not from an NFL
  accuracy readout.
- **A rotation-registry window is assigned, not hand-picked**, via
  `nfl-ats rotation assign` for a fresh family at the time the real
  screen is declared — this document does not pre-select a block, for the
  same reason `docs/scaling_and_transfer.md`'s predeclaration doesn't: the
  CLI computes the earliest eligible block at declaration time, and
  hand-picking one here would go stale.
- **Grade at the opener** for any comparison that could move a pick, per
  the binding "grade the decision at the opener" rule — a close-graded
  screen may run first (this project's established two-stage pattern), but
  no play/no-play or promotion decision may be settled on a close grade.
- **Report `probability_positive` for every arm and metric**, never a bare
  pass/fail and never "contains zero" as a rejection. An interval crossing
  zero is the expected shape for a real small signal at this project's
  ~2-point resolution; only a resolved wrong sign or a positive-control
  bound closes a line of work (closing-grounds taxonomy, `AGENTS.md`).
  Any terminal classification goes through `nfl-ats weak-signals record`
  with an admissible `--closing-ground`; anything else is
  `unresolved_below_power`.
- **The raw-margin control arm is the comparator of first resort.** A
  residual-arm result that does not beat the control on the same engine,
  same structure, same window is not evidence the residual construction
  did anything — this is the whole reason the control arm exists.

None of this was executed by this document or the session that wrote it. It
was written so a future session could run it mechanically, the same discipline
`docs/scaling_and_transfer.md`'s own predeclaration section modeled.

> **Executed 2026-08-26**, later the same day, in
> `docs/graph_ratings_v2_screen.md`. The gap that had to close first: this
> module accepted only `residual` and `raw_margin` edge signals, so there was
> no way to feed it a screened statistic at all. The `team_stat` arm
> (`edge_signal="team_stat"`, `signal_column=`, and `signal_column_pair=` for
> the suffix-form families) is that mechanism, and the paired control there is
> the statistic's own raw home-minus-away differential rather than the
> raw-margin arm -- same principle, correct comparator for the question being
> asked. `cfb_structural_coherence` refuses the `team_stat` arm by design: its
> correlation would be the rating diff against the very quantity its edges were
> built from. The arm inherits the structure frozen here; it does not refit it.
