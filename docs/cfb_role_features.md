# CFB role-continuity feature family — predeclaration

Predeclared: 2026-08-17 (US), before any run in which the feature columns
below touched ATS outcomes, spreads-with-results, or the XLG-03 benchmark
evaluator. The frozen constants are mirrored in
`src/nfl_ats/cfb_role_features.py`; the benchmark runner records
`hypothesis_frozen_before_scoring: true` and this document's path in its
artifact metadata. Results are appended in a separate section below the
predeclaration and never edit it.

This is the XLG-04 follow-up called for by the roadmap: ONE frozen CFB
role-loss/role-continuity feature family (dropback and carry only), scored
against the frozen XLG-03 benchmark. It is a CFB-only experiment; no NFL
rows or outcomes are involved. Whatever the verdict, an NFL transfer claim
(XLG-05) remains a separate, separately predeclared step.

## Mechanism being operationalized

XLG-04 (`docs/cfb_role_replication.md`) replicated cross-league that a
player with a material trailing role who participates at all delivers
approximately that full role (CFB median delivered/prior 1.043 for
dropbacks, 0.995 for carries). The pregame-knowable trace of that mechanism
is *participation continuity*: whether the players holding a team's role
mass actually appeared in the team's most recent game. When they did, the
team's realized roles are highly predictable; when they did not, the team
enters the game with a disrupted, less predictable role structure that the
market may or may not fully price.

Receptions are excluded: XLG-04 recorded reception delivery as **not
replicated** (severe under-delivery gate), and that verdict stands.

## The departure prerequisite (satisfied first, participation data only)

The roadmap required separating permanent departures from temporary
absences before any absence-derived feature. The descriptive study
(`nfl-ats cfb-absence-separation`, artifact
`artifacts/cfb_role_experiments/20260817T105651Z`, participation data only
— no spreads or outcomes read) measured, over 2013–2025 FBS-vs-FBS games:

- **Season boundaries are dominated by departures.** Of qualified role
  holders at the end of a season, only **15.6%** (dropback) / **18.7%**
  (carry) ever appear for the same team the following season. A role mass
  that carried prior-season holders forward would therefore consist mostly
  of graduated/transferred/drafted players every September.
- **Within-season, reappearance probability decays fast with the missed-game
  streak.** Same-season return rates conditional on an absence episode
  reaching k consecutive missed valid games (dropback / carry):
  k=1: 40.3% / 45.2%; k=2: 25.4% / 24.7%; k=3: 16.4% / 12.4%;
  k=4: **10.2% / 6.8%**; k=5: 6.0% / 3.8%. By four straight missed games,
  roughly 80% of episodes never see the player again at all.

Frozen consequences: the active role mass is **scoped to the current
season** (a player enters it only after their first current-season
appearance), and a holder who misses **4** consecutive valid team-games
leaves it (treated as departed/out-for-season). Both rules are
pregame-knowable from participation alone.

## Frozen feature definition

Six columns (`CFB_ROLE_FEATURE_COLUMNS`), computed per canonical game:
`{home,away,diff}_{dropback,carry}_continuity`, with `diff = home − away`.

For one team, action type, and game t, evaluated strictly before game t's
own credits update any state:

- **Role state**: the XLG-04 appearance-only span-8 EWM share per
  (team, player, action type) — identical update rule, thresholds
  (dropback ≥ 0.50, carry ≥ 0.20), and ≥ 3 prior appearances, computed on
  the same credited-action definitions over the same valid team-games
  (≥ 10 team dropbacks / ≥ 10 carries; XLG-04's per-(season, action)
  coverage gate applies unchanged).
- **Active mass at game t**: qualified holders (state ≥ threshold,
  appearances ≥ 3) who have appeared for this team in game t's season and
  whose current missed-valid-game streak is < 4.
- **Continuity** = (state-weighted mass of active holders with streak 0,
  i.e. who appeared in the team's most recent valid game) ÷ (total active
  mass). Empty active mass → **1.0** (neutral: "no known disruption"). A
  canonical game side without a computable row (week one, invalid
  team-game) is likewise imputed 1.0.

Continuity lives in [0, 1]; 1 means every accustomed role holder
participated last game, 0 means none did.

## Frozen evaluation protocol

- **Evaluator**: the XLG-03 walk-forward recipe, unchanged — Ridge alpha 10,
  no calibration, `market_residual` target, ≥ 500 strictly-earlier training
  games, out-of-time empirical residual distribution, forced picks.
- **Three matched arms on identical weeks**: `market` (no-vig control),
  `market_residual` (the frozen benchmark contract), and
  `market_residual_roles` (the identical recipe with exactly the six
  continuity columns appended). Any week skipped for one arm is skipped for
  all.
- **Decision metric**: paired per-game **accuracy improvement** of
  `market_residual_roles` over `market_residual` on the **clean core**
  (2012–2019, 2021–2025), with the week-blocked bootstrap interval from
  `paired_feature_comparisons` (2,000 samples, seed 20260817).
- **Decision rule (frozen)**: the family **clears** only if the week-blocked
  95% interval on paired accuracy improvement excludes zero from below
  (lower bound > 0). Season-blocked intervals, Brier, and log-loss
  improvements are reported as coherence checks but do not override the
  rule in either direction.
- **One run.** No cap retuning, threshold tuning, imputation changes, or
  season-window changes after seeing results. Any variant is a new
  predeclaration. If the family does not clear, it is recorded as-is and no
  NFL transfer claim is predeclared from it.

Sensitivity context, fixed in advance: the XLG-03 positive-control audit
found the CFB evaluator's week-blocked machinery detects synthetic
0.5/1/2-point-per-SD effects in 1/8, 5/8, and 8/8 replicas. A non-cleared
result is therefore evidence of "not resolvably large," not proof of zero.

## Declared limitations

1. **Absence of credit is not absence.** Continuity measures observed
   participation only; a quiet game with zero credited dropbacks/carries by
   a role holder in a valid team-game reads as a miss. The XLG-04
   participation contract is inherited verbatim.
2. **The market prices injuries too.** CFB spreads move on QB news; the
   family may be entirely redundant with the close. That is exactly what
   the market-residual design tests, and a null is an informative outcome.
3. **Week-one blindness.** Season scoping makes every opener neutral; the
   family deliberately claims nothing about offseason roster change (that
   is XLG-06's territory, with roster/recruiting data).
4. **Mid-season departures inside the cap window** (weeks 1–3 of an
   absence) still depress continuity as if they were temporary; the study
   shows these are a minority of episode-games but they are not zero.
5. The neutral imputation (1.0) biases toward "no disruption"; teams with
   chronically thin volume read as continuously undisrupted.

---

## Results

### Voided first run (instrument failure, no information revealed)

The first execution (artifact `artifacts/cfb_role_experiments/20260817T110002Z`)
was **void**: the feature join keyed teams by name, but play-by-play uses
display names ("Minnesota Golden Gophers") while the canonical table uses
schedule names ("Minnesota"), so every game received the neutral value and
the candidate arm reproduced the baseline bit-for-bit (paired improvements
exactly 0.0 across all 8,933 games). Because the candidate arm was
numerically identical to the already-known baseline, this run revealed
nothing about the family's ATS value; repairing the join and re-running is
an instrument fix, not tuning. Two changes were made before the re-run,
neither touching the frozen definitions, gates, seeds, or decision rule:
the join now goes through ESPN team ids (pbp `pos_team_id` against the
canonical `home_id`/`away_id`), and the runner now **fails closed** if all
six role columns are constant, so a vacuous comparison can never complete
silently again.

### Frozen run (2026-08-17, artifact `artifacts/cfb_role_experiments/20260817T110541Z`)

Join verified: 31.7% of the 12,500 canonical games carry at least one
non-neutral continuity value (features exist for 2013–2025 only; pre-2013
games are all-neutral by construction and leave the pre-2013 arms
identical, as expected).

Clean core, 8,933 paired non-push games, candidate minus baseline
(positive = candidate better), 2,000 samples, seed 20260817:

| Metric | Estimate | Week-blocked 95% | Season-blocked 95% |
|---|---|---|---|
| Accuracy improvement | **−0.0067** (−0.67 pts; 50.92% vs 51.60%) | [−0.0133, +0.0001] | [−0.0168, +0.0040] |
| Brier improvement | −0.00058 | [−0.00102, −0.00015] | [−0.00115, −0.00010] |
| Log-loss improvement | −0.0018 | [−0.0038, −0.0004] | [−0.0042, −0.0002] |

Margin error also resolved worse (clean-core MAE 12.285 vs 12.256).

### Verdict: **not cleared** (frozen rule), recorded as a real negative

The week-blocked accuracy interval does not exclude zero from below — it
nearly excludes zero on the *wrong* side — and both probability metrics
resolved significantly worse under week and season blocking. The honest
mechanistic reading: participation-continuity disruption is information the
CFB market already prices (QB and lead-back news moves college spreads),
so conditioning the residual model on it added variance without signal.
XLG-04's participation-level replication stands — players who play deliver
their roles — but the pregame trace of that mechanism carries no residual
ATS value in this form.

Consequences, per the predeclaration:

- **No NFL transfer claim is predeclared from this family.** The XLG-05
  role-continuity transfer path is closed in this form.
- The family is retained as a negative result; no cap/threshold/imputation
  retuning on these outcomes. Any successor (e.g. roster-aware departure
  handling, replacement-quality weighting, or availability semantics from
  XLG-07 data) is a new predeclaration against this same benchmark.
- The benchmark instrument worked as designed: it resolved a ~0.6-point
  probability-metric degradation that the NFL evaluator could not have,
  which is exactly the detection power XLG-03 was built to provide.

---

## 2026-08-18 re-measurement: verdict downgraded from `closed_negative` to `unresolved_below_power`

Triggered by `docs/revisit_list.md` Tier 1 (a 2026-08-18 instrument audit found
two defects -- D1: the trailing residual/calibration step can distort small
effects, `docs/purged_cv.md` Sec.3; D2: reported intervals are 17-58% too
narrow because the block bootstrap never refits, `docs/estimation_variance.md`
-- and flagged this family's terminal negative as exactly the shape either
defect can manufacture). Everything below is **measured this session**,
reproducible via `scripts/cfb_role_continuity_remeasurement.py`
(`./.tools/uv.exe run --no-sync python scripts/cfb_role_continuity_remeasurement.py`,
~152s), full output at the script's scratch JSON (not tracked; rerun to
regenerate). No cap/threshold/imputation was retuned; every number below reads
the identical frozen feature build and frozen benchmark recipe the original
run used. CFB only -- no NFL rotation window touched (rule 8).

### 1. Reproduction: exact

Calling `cfb_role_features.cfb_role_benchmark` with the original
`bootstrap_samples=2000, bootstrap_seed=20260817` reproduces the recorded
week-blocked accuracy improvement **bit-for-bit**: estimate
`-0.006716668532407925`, interval `[-0.013330806116650032,
+0.00011418170713983402]`, 8,933 paired games. The original result is real,
not a bug in how it was computed.

### 2. D3 fix alone (samples 2,000 -> 20,000) flips the week-blocked interval fully negative

Re-running the identical weekly-cadence comparison at `samples=20,000` (the
now-default resolution, `experiments.py:150`) on the same predictions:

| Block | Estimate | 95% interval | `probability_positive` |
|---|---|---|---|
| week | -0.672 pts | [-1.325, -0.011] | **0.0228** |
| season | -0.672 pts | [-1.658, +0.382] | 0.1012 |

At 20,000 samples the week-blocked reading moves to `probability_positive`
0.0228 (season 0.1012), shifting the interval's upper bound by 0.0002 points
-- purely from tighter Monte Carlo resolution, not from new evidence (the
~0.03-point seed jitter D3 already documented). Taken alone this would look
like the negative got *stronger*. It does not survive the next two checks.

### 3. D1 cross-check: the sign-only estimator, identical folds, recovers 85% less negative

`sign(predicted_margin - spread_line)` -- the exact ablation `docs/purged_cv.md`
used on planted effects -- applied to the SAME weekly-cadence predictions, same
8,933 games, same folds:

| Estimator | Block | Estimate | 95% interval | `probability_positive` |
|---|---|---|---|---|
| Full pipeline (calibrated) | week | -0.672 pts | [-1.325, -0.011] | 0.0228 |
| **Sign-only** | week | **-0.101 pts** | [-0.632, +0.439] | **0.3498** |
| Full pipeline (calibrated) | season | -0.672 pts | [-1.658, +0.382] | 0.1012 |
| **Sign-only** | season | **-0.101 pts** | [-0.610, +0.459] | **0.3418** |

Bypassing the trailing residual/calibration step removes **85% of the
recorded negative** (-0.672 -> -0.101 pts) and moves `probability_positive`
from a near-refutation (0.023) to a near-coin-flip lean-negative (0.35).
`docs/purged_cv.md` Sec.3's caveat that D1 was "unverified on real effects"
was based on a *different* measurement (overall walk-forward-vs-purged-CV
accuracy, ~0.1-pt gap); this is the first direct sign-only-vs-full ablation on
this family's actual recorded paired delta, and the gap here is 5-6x larger.
**D1 is live for this specific result**, not merely a demonstrated-on-plants
concern.

### 4. Paired power arithmetic (`docs/estimation_variance.md`'s `MDE80 = 280*sqrt(f/n)`)

| Estimator | `n` | `f` (picks differ) | MDE80 |
|---|---|---|---|
| Full pipeline | 9,093 | 9.96% | **0.927 pts** |
| Sign-only | 9,093 | 9.16% | **0.889 pts** |

Both recorded point estimates (0.672 pts, 0.101 pts) sit **below** this
evaluator's own 80%-power detection floor at this `f`/`n` (0.89-0.93 pts) --
independent of D1 and D2, the magnitude itself was never large enough to
resolve at this instrument's power, which is a category-3 (unresolved), not a
category-1 (refuted), signature.

### 5. Honest, family-specific refit-aware interval (D2)

A weekly-cadence refit-aware bootstrap across 14 seasons was not affordable
this session (~15x an annual-cadence run per `docs/estimation_variance.md`
Sec.3's own cadence note). Instead of borrowing another family's width-inflation
factor, this family's OWN refit-aware interval was measured directly: annual
refit cadence (one fit per clean-core season, `N_BOOT=120`, reusing
`scripts/estvar_real_cfb_audit.py`'s `fit_seasons`/`refit_aware_paired_interval`
machinery), candidate = frozen CFB columns + the six role-continuity columns,
baseline = frozen CFB columns alone, 8,933 games:

| | Estimate | 95% interval | `probability_positive` |
|---|---|---|---|
| Naive (annual cadence) | -0.291 pts | [-0.851, +0.267] | 0.150 |
| **Honest (refit-aware)** | **-0.078 pts** | **[-0.956, +0.653]** | **0.367** |

Width inflation **1.438x** (44% wider) -- inside D2's documented 17-58% range,
toward the top, consistent with `docs/estimation_variance.md`'s own finding
that narrow-`f` single-feature-family comparisons (this one: `f`=9.0% annual)
are where the naive interval understates worst. Candidate flip rate under
training-row resampling: **19.4%**, matching the audit's cited 15-22% range
almost exactly and confirming the refit-variance mechanism is present at the
expected size for this family.

**Cadence note, stated plainly:** the annual-cadence *naive* point estimate
(-0.291 pts) is itself much smaller than the recorded weekly-cadence naive
estimate (-0.672 pts) on the same games -- unlike `docs/estimation_variance.md`'s
comparison A, where cadence barely moved the number. **Inferred, not measured
further this session:** weekly refitting invokes the noisy trailing-calibration
split (D1's channel) roughly 5x more often across a season than annual
refitting does, so if that channel's fold-to-fold bias does not fully average
out, more refits could compound rather than cancel it. This is a plausible
mechanism, not a demonstrated one; flagging it for anyone extending this work.

The two independent honest reads -- D1's sign-only ablation (-0.101 pts, P+
0.35, weekly cadence, identical folds to the recorded run) and D2's refit-aware
interval (-0.078 pts, P+ 0.37, annual cadence, family-specific) -- **converge**
on a small, unresolved, weakly-negative-leaning effect roughly a tenth of a
point in size, despite using unrelated methodologies. Neither resembles the
recorded -0.672.

### 6. Split-half reliability of the role-continuity trait itself

Per `docs/injury_value_lost.md` Sec.3.1's recipe (the AGENTS.md bar for
"refuted: no split-half reliability"): each team-season's continuity values
split by odd/even week, halves correlated across team-seasons with >=2 games
in each half, Spearman-Brown corrected to a full-length reliability, 4,000
bootstrap draws for the CI and `probability_positive`:

| Action type | `n` team-seasons | Pearson r | 95% CI | Spearman rho | Spearman-Brown reliability | `probability_positive` |
|---|---|---|---|---|---|---|
| dropback | 1,658 | 0.561 | [0.527, 0.596] | 0.665 | **0.719** | 1.000 |
| carry | 1,671 | 0.516 | [0.472, 0.559] | 0.577 | **0.680** | 1.000 |

Reliability 0.68-0.72 is a real, stably-measured team-season property --
comparable in strength to `injury_value_lost`'s 0.87-0.93 (kept) and far above
the reliabilities that closed other lines for lack of one (coach-quality
reputation 0.063, play-EPA dispersion 0.014, per `docs/pool_edge_plan.md`).
**This directly rules out "refuted mechanism: no split-half reliability."**

### 7. Positive control: none exists for this family

Checked directly: the only deliberate-leak positive control in the CFB
program bounds **opponent adjustment** (+0.0129 MAE, `probability_positive`
0.984, `docs/cfb_opponent_adjustment.md`), a different feature family --
per this task's instruction, not borrowed here. XLG-03's own general
sensitivity audit (RWB-15: detects synthetic 0.5/1/2-point-per-SD effects in
1/8, 5/8, 8/8 replicas) characterizes the evaluator's generic detection power,
not a bound proven for role-continuity specifically. **No positive control
supports "bounded by control" for this family.**

### Reclassification

Neither of the taxonomy's two closing criteria holds:

1. **Refuted mechanism** (wrong sign, or no split-half reliability) -- does
   not hold. The D1-clean sign-only estimate is a small, roughly coin-flip
   lean-negative (P+ 0.35), not a resolved wrong sign, and the trait's own
   split-half reliability (0.68-0.72) is strong.
2. **Bounded by positive control** -- does not hold. No family-specific
   positive control exists; the nearest one belongs to a different family and
   is explicitly not transferable.

**Verdict: `closed_negative` does not survive. Reclassified `unresolved_below_power`
(category 3).** The XLG-04 -> XLG-05 cross-league transfer path this family
was gating should be treated as open again for this family, not permanently
closed -- though re-opening XLG-05 itself is a separate, separately-scoped
decision, not made here. Proposed registry and weak-signal-ledger edits are in
the accompanying session report; this document does not write to
`registry/*.json`.
