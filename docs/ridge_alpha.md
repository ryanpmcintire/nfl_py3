# Deriving `ridge_alpha`

Written 2026-08-18. `ridge_alpha = 10.0` governs the active NFL model
(`market_residual` / `weak_stack` / `ridge`, `calibration: none`,
`model_id 118f31d9a98c815b`) and has never been derived — it is the sklearn
example value, inherited into `make_margin_estimator`'s default and copied
into every CLI flag that touches a ridge margin model
(`src/nfl_ats/margin.py:280`, `src/nfl_ats/cfb_benchmark.py:36`). Per the
standing rule (`docs/underived-constants-are-wrong` memory; ROADMAP MOD-12),
an unjustified constant that gates an irreversible decision is a defect until
derived. This document derives it, on the free CFB instrument, and reports
what the derivation actually buys.

**Bottom line up front.** For the metric this project is graded on (forced
-pick ATS accuracy), `ridge_alpha` is inert across **seven orders of
magnitude** (0.001 to 10,000) on 12,500 CFB games — no value in that range
beats the frozen 10.0 with a resolvable margin. For Brier/log-loss
(calibration), there is a real, resolvable, small effect, and the
walk-forward optimum sits around **alpha ≈ 2,000**, roughly 200x the frozen
value. Both of those statements are true at once. Neither supports promoting
a new `ridge_alpha` into the accuracy-graded production model today; both
support routing the calibration gain to a probability-quality consumer (the
Best Pick ranker) instead.

---

## 1. The rank deficiency, measured directly on the active design

The claim in circulation (`ROADMAP.md` MOD-12, `docs/pool_edge_plan.md`
"Where to look next" §1) is **"142 columns, rank 71."** That number is real
but **stale**: it was measured in `docs/groupwise_ridge.md` (2026-08-17) on
the `market_residual` / **`player`** profile (79 declared columns), which was
the active model *before* the same-day promotion to `weak_stack` (commit
`68b4dc0`, `docs/pool_edge_plan.md`). Measured directly on the **currently
active** design — `game_features_weak_stack.parquet`, the exact row/column
filter `fit_margin_model` applies (regular-season, non-push, `ats_margin`
notna, sorted, full refit), 90 declared feature columns, **4,431 training
games** (not 4,630):

| quantity | stale figure (quoted, `player` profile) | measured here (`weak_stack`, active) |
|---|---:|---:|
| declared feature columns | 79 | **90** |
| transformed columns (imputer + indicators + scaler) | 142 | **159** |
| numerical rank | 71 | **82** |
| training games | 4,431 | 4,431 (unchanged) |

The **shrinkage headline reproduces almost exactly** despite the column-count
correction: at `alpha = 10` the median non-null direction of the current
159-column design is shrunk **0.274%** (median eigenvalue 3,643 of `X'X`),
mean shrinkage **5.50%**, weakest-decile shrinkage **15.0%** — against the
quoted 0.27% / 6.4% / 16.0%. So the practical conclusion ("the model is
unregularised least squares on the signal-bearing bulk of the design") is
confirmed on the actual active design; only the column-count/rank figure
needed correcting.

### 1a. Where the 77 lost dimensions actually come from

The stale write-up attributes the null space entirely to `diff = home - away`.
That is only part of the story, and a smaller part than the missing-value
indicator columns `SimpleImputer(add_indicator=True)` adds. Splitting the
159 transformed columns into the 90 imputed-and-scaled feature columns and
the 69 missing-indicator flags, and computing each block's rank alone:

| block | columns | rank | lost |
|---|---:|---:|---:|
| raw features (imputed + scaled) | 90 | 74 | 16 |
| missing-indicator flags | 69 | **10** | **59** |
| combined | 159 | 82 | 77 |

**77% of the lost dimensions (59 of 77) come from the indicator block, not
the arithmetic identity.** The reason: `docs/modeling.md`'s warm-up rule
("at least three observed games are required for the initial state") gates
an entire team's rolling-state *vector* as one atomic condition, not each
metric independently. Every `home_*` column drawn from `STATE_METRICS` (15
metrics: offense, results, defense) goes missing in **exactly the same 48
rows** — so instead of one flag carrying "this team lacks history," the
pipeline carries **45 bit-for-bit identical copies of it** (one indicator
per `home_/away_/diff_` column across all 15 metrics). Measured directly:
the 69 indicator-bearing columns collapse to **13 distinct boolean
patterns**, and even those 13 aren't fully independent (rank 10 — three more
redundancies, e.g. `temp`/`wind` share one dome-game pattern). Group sizes:
one group of **45** (the shared team-state gate, 48 affected rows), one of 7
(the continuity-family gate, 1,056 rows — see `docs/underived-constants`
entry on `DEFAULT_OFFSEASON_RETENTION` for why continuity has its own,
larger, gap), one of 4, one of 3, one of 2 (`temp`/`wind`), and 8 singletons.

The `diff = home - away` identity is real but **only approximately survives
imputation**, and contributes less than the framing suggests. It holds
exactly on jointly-complete rows for all 15 triples (verified directly), but
`SimpleImputer(strategy="median")` fills `home`, `away`, and `diff` with
**three independent column medians**, so for the 48 rows requiring
imputation the identity breaks by up to 0.0832 raw points (largest break:
`point_diff`, the metric with the largest scale; smallest: `def_takeaway_rate`,
essentially zero). Of the 15 triples this affects, only **4** produce a
singular value distinguishable from float64 noise in this design (2.496,
0.605, 0.184, 0.0786); the other 11 breaks are small enough, relative to the
leading singular values (~250–460), to be indistinguishable from an exact
zero at machine precision. So the raw block's 16 lost dimensions are a mix
of triples that are *exactly* degenerate in this specific training cut and a
handful that are *near*-degenerate but not exactly so.

### 1b. Does the redundancy matter — benign, or does it decide the penalty split?

**Mostly benign, with one real exception.** For the 59 exact-duplicate
indicator dimensions (and the ~11 exactly-degenerate triples), the null
space consists of directions `v` with `Xv = 0` *identically*, in both
training and future scoring rows (the duplication is a property of the
feature-construction code, not a coincidence of one sample). Predictions are
`X @ beta`, which is invariant to how ridge splits coefficient mass among
columns that are literally the same vector — at `alpha = 10` or `alpha =
0.001` the fitted *values* on any held-out row are unaffected by this part
of the null space, only the individual coefficients on the 45 duplicate
columns are (uselessly) split. This matches `docs/groupwise_ridge.md`'s
framing, but that framing overstates the stakes: "the penalty is the only
thing choosing among infinitely many equivalent solutions" is true in a
narrow coefficient-identifiability sense and **irrelevant** to the
prediction the pool actually grades.

The one place `ridge_alpha` does real, prediction-relevant work is the
**4 near-but-not-exactly-degenerate directions** in the raw block
(eigenvalues 6.23, 0.366, 0.034, 0.0062 in this design's standardized
units). At `alpha = 10` these are shrunk **61.6% / 96.5% / 99.6% / 99.9%**
respectively — while the median *real* direction two paragraphs up is
shrunk 0.27%. So `alpha = 10` is doing exactly one substantive thing: it is
crushing a handful of already-marginal, largely-imputation-artifact
directions almost to zero, simply because their eigenvalues are tiny enough
that any alpha ≥ a few dominates them — not because 10 was chosen with that
in mind. The ~70 directions that carry the actual football signal are left
essentially untouched. **This is the practical meaning of "10.0 was
inherited, not derived": it happens to provide well-posedness on the
degenerate tail for free, but was never actually chosen to trade off bias
and variance on the part of the design that matters.**

Files: `src/nfl_ats/margin.py` (`make_margin_estimator`, `fit_margin_model`),
`src/nfl_ats/modeling.py` (`regular_season_rows`), measurement scripts left
in the scratchpad (not committed; see §4).

---

## 2. The CFB alpha screen

`scripts/ridge_alpha_screen.py` (new, this session) mirrors
`cfb_benchmark.cfb_walk_forward_benchmark` exactly — same weekly cutoffs,
same `CFB_MODEL_FEATURE_COLUMNS` contract, same 20% out-of-time residual
split — varying only the global `ridge_alpha` passed to
`fit_cfb_residual_model`. Free under rotation rule 8 (CFB is unreserved); no
NFL confirmation window spent. 12,500 games, 2006–2025; clean-core window
(2012–2019, 2021–2025) is 9,093 games / 8,933 evaluated non-push.

Two passes: a predeclared coarse log-grid (13 points, 1e-3 to 1e5, including
the frozen 10.0), then — after seeing the coarse Brier curve turn over near
1,000–3,000 — a second, **not blind**, refinement grid (1,500 / 2,000 /
2,500 / 4,000 / 5,000 / 7,000) to localize the optimum. The refinement is
flagged explicitly because it is a second look at the same instrument; it
narrows a point estimate, it does not license a stronger promotion claim
than the coarse grid already supports (§3, §5).

### 2a. Clean-core accuracy and Brier by alpha (coarse grid, `n=8,933` evaluated)

| alpha | accuracy | Δ accuracy vs 10 | P+ (week) | P+ (season) | Brier | Δ Brier vs 10 | P+ (week) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 0.51696 | +0.101 pt | 0.850 | **0.993** | 0.250082 | −0.000113 | 0.071 |
| 0.01 | 0.51718 | +0.123 pt | 0.928 | **0.994** | 0.250077 | −0.000108 | 0.076 |
| 0.1 | 0.51730 | +0.134 pt | **0.949** | **0.989** | 0.250060 | −0.000091 | 0.105 |
| 1 | 0.51629 | +0.034 pt | 0.632 | 0.606 | 0.250014 | −0.000045 | 0.198 |
| 3 | 0.51629 | +0.034 pt | 0.703 | 0.677 | 0.249990 | −0.000021 | 0.246 |
| **10 (frozen)** | 0.51595 | — | — | — | 0.249969 | — | — |
| 30 | 0.51562 | −0.034 pt | 0.277 | 0.264 | 0.249957 | +0.000012 | 0.709 |
| 100 | 0.51584 | −0.011 pt | 0.440 | 0.447 | 0.249948 | +0.000021 | 0.687 |
| 300 | 0.51774 | +0.179 pt | 0.811 | 0.815 | 0.249847 | +0.000122 | **0.966** |
| 1,000 | 0.51584 | −0.011 pt | 0.473 | 0.498 | 0.249733 | +0.000236 | **0.974** |
| 2,000\* | 0.51853 | +0.258 pt | 0.756 | 0.711 | 0.249674 | +0.000295 | **0.962** |
| 2,500\* | 0.51853 | +0.258 pt | 0.758 | 0.703 | 0.249655 | +0.000314 | **0.955** |
| 3,000 | 0.51808 | +0.213 pt | 0.716 | 0.659 | 0.249675 | +0.000294 | 0.933 |
| 10,000 | 0.51606 | +0.011 pt | 0.499 | 0.504 | 0.249777 | +0.000192 | 0.752 |
| 100,000 | 0.50901 | **−0.694 pt** | **0.104** | **0.060** | 0.250047 | −0.000078 | 0.408 |

\* fine-grid refinement point. P+ is `probability_positive` from a
2,000-draw blocked paired bootstrap against the alpha=10 reference, oriented
so positive means the candidate is better (higher accuracy, lower Brier).
Full tables: `artifacts/ridge_alpha_screen/` (coarse) and
`artifacts/ridge_alpha_screen_fine/` (refinement).

### 2b. Reading the curve

**Accuracy is flat from 0.001 to 10,000 — seven orders of magnitude.** The
point estimates wander within a 0.26-point band (0.5155–0.5185, excluding the
100,000 outlier) with no monotone trend, and **no candidate in that entire
range clears both week- and season-blocked P+ ≥ 0.90 simultaneously** at a
magnitude worth acting on. Two weak, non-overlapping pockets exist — very
small alpha (0.001–0.1: +0.10 to +0.13 points, P+ 0.85–0.99, both blockings
agree) and the 2,000–3,000 neighborhood (+0.21 to +0.26 points, P+ 0.66–0.76,
weaker and only one blocking clears 0.75) — pointing in *opposite* regularization
directions. Per AGENTS.md, neither interval crossing zero is grounds to
discard it, and both are recorded as category-3 below; but they do not
corroborate each other, which is itself evidence that the accuracy curve's
apparent bumps are noise scatter around a genuinely flat surface rather than
a real unimodal effect the grid just under-resolved. The **only unambiguous
signal in the accuracy objective is that extreme over-shrinkage is bad**:
alpha = 100,000 is worse with high confidence (P+ 0.06–0.10) — mechanically,
enough shrinkage eventually drags every prediction toward the market
baseline (`mean |predicted residual|` falls from 1.13 at alpha=10 to 0.23 at
100,000) and forced-pick accuracy decays toward the coin flip.

**Brier/log-loss is a real, resolvable, smooth curve.** It falls
monotonically from alpha=0.001 through a broad minimum spanning roughly
1,000–3,000, then rises again by 10,000–100,000 — the classic ridge
bias-variance U, and (unlike accuracy) every point from 300 up through 3,000
clears P+ ≥ 0.93 on the week-blocked bootstrap against the frozen 10.0. The
refined grid narrows the minimum to **alpha ≈ 2,000–2,500** (Brier 0.24966–
0.24967, indistinguishable from 1,000's 0.24973 or 3,000's 0.24968 — a broad,
shallow plateau, not a sharp point). The magnitude is small in absolute
terms: **+0.0003 Brier, ~0.12% relative** — the same order of magnitude as
the group-wise block-penalty screen's resolved effect
(`docs/groupwise_ridge.md` §4c: +0.000304, P+ 0.9915), and about a fifth of
MOD-08's ECDF-smoothing Brier gain.

**The two objectives disagree, and that disagreement is itself informative.**
The alpha that walk-forward-minimizes Brier (≈2,000) shows only P+ 0.66–0.76
on accuracy — not resolvable — while the alpha region with the *strongest*
accuracy lean (0.001–0.1) shows Brier moving the *wrong* way (P+ 0.07–0.11,
i.e. probably worse). This is the same pattern the group-wise screen found
for block allocation (`docs/groupwise_ridge.md` §5, verdict 3): shrinkage
level is a calibration lever, not a picking lever, on this design.

---

## 3. The derived answer

**Selecting on accuracy** (the project's primary, pool-relevant metric): no
value of `ridge_alpha` in `[0.001, 10,000]` is distinguishable from the
frozen 10.0 with a resolvable margin. The walk-forward selection procedure
returns **"can't tell" across seven orders of magnitude**, which is itself
the honest derived answer — `ridge_alpha` is **functionally inert for the
metric this project is graded on**, not because 10.0 happens to be optimal,
but because almost the entire usable range is statistically equivalent to it.
This is *not* the same claim as "10.0 is fine because it doesn't matter" —
it means **any value in that range is equally undefended**; 10.0 has no more
justification today than 1.0 or 1,000 would.

**Selecting on Brier/log-loss** (a legitimate secondary objective per
`docs/modeling.md`'s decision policy, but never the promotion gate for a
headline accuracy claim): the walk-forward optimum is **alpha ≈ 2,000**,
flat across roughly 1,000–3,000 (three orders of magnitude narrower than the
accuracy plateau, but still not a sharp point), and the gain is small
(+0.0003 Brier, P+ 0.93–0.97 in the resolvable band).

**"10.0 is wrong" and "10.0 is harmless" are both true, and should be said
plainly, exactly as the task anticipated.** It is wrong in the sense that
matters for defensibility: it was never chosen, and a walk-forward selection
on the identical instrument used to close every other hyperparameter
question in this project (MOD-06, the group-wise screen) returns a
different, derived number (≈2,000) for the one metric where the choice is
resolvable at all. It is harmless in the sense that matters for the pool: no
value in a seven-order-of-magnitude range moves the accuracy the pool is
graded on, so nothing about today's forecasts is silently miscalibrated by
this choice.

---

## 4. Recommendation

**Do not change the active NFL model.** Accuracy — the pool's bar — does not
discriminate between 10.0 and the derived Brier-optimum region on the CFB
instrument, and this document changes nothing about `artifacts/active_ats_model.json`,
`registry/rotation_registry.json`, or any tracked doc other than this new
file, per the task's constraints.

**Route the Brier gain to the Best Pick ranker, not to a re-fit production
model.** This is the second finding in a row (after the group-wise block
screen) where ridge penalty structure resolves cleanly on Brier/log-loss and
not on accuracy. `docs/pool_edge_plan.md`'s queue item 2 ("Best Pick lever is
currently unexploited: our confidence ordering is flat") is exactly a
probability-quality problem, and needs no NFL confirmation window at all —
unlike promoting a new `ridge_alpha` into the accuracy-graded model, which
would.

**Predeclaration text for a future session, if an NFL-side test is ever
justified.** Do not draw a rotation window on the accuracy hypothesis alone
— the CFB evidence does not support one. If a future session wants to spend
a window anyway (e.g. because the Best Pick work independently wants a
higher-alpha model and an accuracy check is a reasonable safety gate before
shipping it), freeze this before looking:

- **Family name:** `ridge_alpha_global` — new family, does not inherit
  reliability from `player_qb_continuity` or `groupwise_ridge_block_penalties`
  (different hypothesis, different mechanism).
- **Candidate:** `ridge_alpha = 2,000` (the walk-forward Brier optimum measured
  here) vs. the frozen reference `ridge_alpha = 10.0`. Both on
  `market_residual` / `weak_stack` / `ridge`, `calibration = none` — every
  other production setting held fixed.
- **Primary metric and predeclared hypothesis:** Brier / log-loss
  improvement, week- and season-blocked, on the earliest eligible NFL window
  in `registry/rotation_registry.json`. **Do not predeclare an accuracy
  hypothesis** — the CFB screen gives no reason to expect one, and framing it
  as an accuracy test would repeat the mistake `docs/groupwise_ridge.md`
  flagged in its own verdict §2.
- **Accuracy role:** reported as a safety check (must not resolve
  *negative* at P+ ≥ 0.90, mirroring the block-penalty screen's own guard),
  not as the promotion bar. Grade at the opener, per AGENTS.md.
- **Scope of promotion if it passes:** feeds the Best Pick / probability
  -quality consumer, not a swap of the headline accuracy model, unless a
  *separate*, explicitly accuracy-framed predeclaration is run and clears
  the pool's existing bar independently.

---

## 5. Category-3 record

The accuracy effect at the walk-forward Brier optimum is unresolved, not
refuted, and not bounded by a positive control — a category-3 result per
AGENTS.md, and must be recorded rather than silently dropped. Per this
task's constraints, the record is **not** written into
`registry/weak_signals.json` directly (another process owns that file this
session); the exact payload `nfl-ats weak-signals record` would need is at
`<scratchpad>/ridge_alpha/weak_signal_record.json` for a future session or
the file's owner to apply. Summary of what it records: `ridge_alpha=2,500`
(finest-resolution point) vs. the frozen 10.0 reference, clean-core CFB,
week-blocked, `+0.2575` accuracy points, 95% `[-0.410, +0.951]`,
`probability_positive 0.758` — unresolved, with the Brier half (`+0.000314`,
P+ 0.955) explicitly noted as the resolved companion, matching the same
structure as the already-recorded `groupwise_ridge_block_penalties` entry.

---

## 6. Files

- `scripts/ridge_alpha_screen.py` — the coarse + fine walk-forward alpha
  sweep (new, this session).
- `artifacts/ridge_alpha_screen/` — coarse-grid predictions, summaries,
  paired comparisons (`alpha_summary.csv`, `alpha_comparison.csv`, `grid.json`).
- `artifacts/ridge_alpha_screen_fine/` — refinement grid, same schema.
- `docs/groupwise_ridge.md` — the companion group-wise (block-allocation)
  screen this document was blocked behind; §2 of that document's eigenvalue
  table is the source for the CFB shrinkage-vs-alpha correspondence cited
  above.
- Rank/degeneracy measurement scripts used for §1 were run from the
  scratchpad and are not committed (research-only, reproducible from the
  commands in this document plus `src/nfl_ats/margin.margin_feature_columns`
  / `make_margin_estimator`).
