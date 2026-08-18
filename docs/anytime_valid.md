# Anytime-valid inference for the paired feature-set comparison

Written 2026-08-18, revised twice same day after review. Implements and
validates `src/nfl_ats/anytime.py` alongside (never modifying) `experiments.
paired_feature_comparisons`. Empirical work runs on CFB
(`scripts/anytime_validate.py`) plus one independently-supplied NFL
measurement cited, not re-derived, below. `registry/rotation_registry.json`
is untouched; no NFL rotation window was spent producing anything here.

**Headline, in order of what actually matters:**

1. **Calibration under repeated peeking.** The existing fixed-sample block
   bootstrap false-alarms **38.0-48.5%** of the time when checked every
   week of a season — roughly 8-9x nominal. The confidence sequence built
   here false-alarms **0.00%**, including three different real-schedule
   horizons, at the project's operating configuration — and **0.77%**
   even under a deliberate stress test where the true within-week
   correlation is the full worst case while the sequence is told to assume
   independence. 2026 will be monitored every single week
   (`docs/prospective_evidence.md`), which is exactly the regime the
   existing method is not built for. This needs no new data.
2. **Independence within a week is a standing project decision, not an
   estimate.** `intraclass_correlation` is hardcoded to `0.0`. §2 has the
   full argument and the exchange that produced it.
3. **A weekly monitoring rule for the 2026 season** (§4,
   `scripts/anytime_weekly_monitor.py`) — concrete, runnable, and the actual
   point of this work.
4. **What the instrument will and will not resolve in one season** (§5): a
   fixed, quantified property, not a request for more data.

## 1. Method, and the assumption doing the work

A **normal-mixture test martingale / confidence sequence** (Robbins 1970;
formalised as the "Normal mixture" nonnegative supermartingale in Howard,
Ramdas, McAuliffe & Sesia 2021, *Time-uniform, nonparametric, nonasymptotic
confidence sequences*, Ann. Statist.; the same construction backs the
"always-valid p-values" used for continuously-monitored A/B tests in Johari,
Pekelis & Walsh 2015). Chosen over a WSR betting martingale or an
empirical-Bernstein sequence — the other two families named in this
module's brief — because it has a single closed form with no online
optimisation loop to get subtly wrong: an under-covering interval would be
worse than the status quo, and a closed form is the easiest kind of formula
to unit-test against algebra instead of against its own code
(`tests/test_anytime.py` pins the interval/e-value duality algebraically on
500 random inputs).

At look $t$, having accumulated per-game-improvement sum $S_t$ over a
predictable variance process $V_t$ (defined in §1.1) and a fixed mixing
("prior") variance $\rho$:

$$
\text{e-value: } M_t=\sqrt{\tfrac{1}{1+\rho V_t}}\,
  \exp\!\Big(\tfrac{\rho S_t^2}{2(1+\rho V_t)}\Big)
\qquad
\text{radius: } r_t=\tfrac1{N_t}\sqrt{\tfrac{2(1+\rho V_t)}{\rho}
  \log\!\big(\tfrac1\alpha\sqrt{1+\rho V_t}\big)}
$$

The interval $[\bar X_t-r_t,\ \bar X_t+r_t]$ excludes zero exactly when
$M_t\ge 1/\alpha$ — dual by construction, not just in practice. $\rho$
tunes power only; validity ($P(\exists t: M_t\ge 1/\alpha)\le\alpha$) holds
for *every* $\rho>0$, confirmed for several very different values in §3.

**The assumption doing the work:** each block's contribution is
conditionally mean-$\mu_0$ (under the null) given everything seen so far,
and bounded — nothing about *within*-block behaviour is required. That is
strictly weaker than the exchangeability the existing block bootstrap
already assumes. What *is* new, and is a real, stated assumption: a
**fixed variance proxy**, decided before monitoring starts and never
re-estimated online (§1.1, §2).

### 1.1 Handling within-week/season correlation

The unit fed to the martingale is **one whole block (week or season), never
a game**: block $i$'s contribution is the raw SUM of its $k_i$ per-game
improvements, deterministically bounded on $[-k_i,k_i]$ regardless of what
happens inside the block. Effective sample size is the **block count**, not
the game count — a 735-game CFB season is 16 looks, not 735.

Kish's cluster design effect gives the variance process:
$V_t=\sum_{i\le t} k_i\cdot s^2\cdot\big(1+(k_i-1)\cdot\text{icc}\big)$,
where $s^2$ (`per_game_variance_proxy`) bounds one game's own variance and
`icc` (0-1) bounds how much of a block moves in lockstep. `per_game_
variance_proxy` defaults to `1.0` (Hoeffding's worst case, no assumption
required) and is overridden per call with a measured value as a stated
assumption. `intraclass_correlation` defaults to `0.0` — see §2, this is
not the same kind of default. `WORST_CASE_INTRACLASS_CORRELATION = 1.0` is
kept as an explicit, named constant for stress-testing only. Both are fixed
BEFORE monitoring begins and never re-estimated from the stream being
watched, which is what keeps the construction valid rather than
adaptively-and-riskily narrow.

`log_loss_improvement` is excluded on purpose (unbounded, no fixed
sub-Gaussian proxy under this construction); only `accuracy_improvement`
and `brier_improvement` are supported.

## 2. Independence within a week: a standing decision, not an estimate

This section exists so no future session relitigates it. It records the
full exchange because the reasoning is more durable than the number.

**The claim.** Games within an NFL or CFB week involve disjoint teams
playing separate contests, with no shared outcome mechanism. That is a
property of the events, not a statistical hypothesis to be tested per
comparison — `intraclass_correlation = 0.0` is hardcoded in
`nfl_ats.anytime`, making the Kish design effect exactly `1.0` and reducing
the variance process to the fully independent case.

**What was tried first, and rejected.**

1. *Padded to 0.10, unmeasured.* The module's first version assumed
   `intraclass_correlation=0.10` as a "safety margin" with no measurement
   behind it — exactly the defect this project's own rule condemns: an
   unjustified constant gating a decision.
2. *Measured, then padded to a smaller value.* Corrected to measure the
   ICC properly (one-way random-effects ANOVA estimator,
   `anova_intraclass_correlation`, with a block-bootstrap confidence
   interval, `bootstrap_intraclass_correlation`) on three independent real
   CFB paired comparisons and reconcile against an independently-supplied
   NFL measurement, then pad the result to `0.01` "to be safe."
3. *An empirical, per-comparison auto-estimator.* Proposed and partially
   built (`predictable_intraclass_correlation`: an incrementally-updated,
   predictable running ANOVA estimate with a Fisher-formula confidence
   margin, using only strictly prior blocks) so the operating value would
   adapt per comparison instead of being fixed in advance.

**Why (3), and then (2), were both overridden.** The measured values from
four independent sources — three real CFB paired comparisons and one real
NFL comparison — are:

| Source | ICC(week) estimate |
|---|---:|
| NFL, fixed-vs-learned availability (`artifacts/availability_experiments/20260813T133345Z`, 2,075 games, 141 weeks) | −0.0054 |
| CFB, `market` vs `market_residual` (9,093 games, 199 weeks) | −0.0007 |
| CFB, `market_residual` vs `market_residual_roles` (9,093 games, 199 weeks) | +0.0021 |
| CFB, `market_residual` vs `market_residual_variance` (9,093 games, 199 weeks) | +0.0030 |

**These four values straddle zero and are all tiny — the signature of
sampling noise around a true value of exactly zero, not evidence of a
small positive correlation.** Their mean is **−0.00025**, which sits
**0.13 standard errors** from zero (using the CFB bootstrap SE, ≈0.0022);
if the true ICC were the padded 0.01, the observed mean would have to sit
**5.4 standard errors below it** — the data does not merely fail to
distinguish 0.01 from 0, it actively favours 0. An auto-estimator built on
this same data would keep producing small nonzero values driven by exactly
this noise, on every future comparison, silently reintroducing the cost
(3)-(2) were meant to eliminate. Physically this is what independence
predicts: distinct games between disjoint teams are independent events, and
that modelling decision, once made, does not need re-deriving from a noisy
estimate every time a new candidate is screened. `predictable_intraclass_
correlation` was therefore deleted rather than shipped; `anova_intraclass_
correlation`/`bootstrap_intraclass_correlation` remain in the module as
read-only diagnostics — `paired_anytime_comparisons` reports the measured
value for whatever comparison is at hand as `measured_icc_diagnostic`
alongside every trace, purely so a reader can sanity-check the independence
decision against new data — but neither function feeds the operating
value, and nothing in the module auto-estimates it.

**What the padded assumptions actually cost**, using Kish's design effect
$1+(m-1)\cdot\text{icc}$ at the real block sizes (games needed scales with
$\sqrt{\text{DEFF}}$, so "wider" below means the confidence sequence's
radius, not the game count):

| League | Games/week | DEFF at icc=0.01 | DEFF at icc=0.10 |
|---|---:|---:|---:|
| NFL | 14.7 | 1.14x games, 7% wider | 2.37x games, 54% wider |
| CFB | 45.7 | **1.45x games, 20% wider** | 5.47x games, 134% wider |

CFB is the row that matters operationally: CFB weeks are ~3x larger than
NFL weeks, so the design effect is ~3x more sensitive to the same pad, and
CFB is where all of this project's free screening happens. A 0.01 pad
chosen "to be safe" would have cost roughly a fifth of the effective
college sample on every future CFB screen, for a correlation the data does
not support — invisible in any single result, showing up only as more
screens returning "unresolved" than the true signal-to-noise ratio
justifies.

**The one number that matters, re-checked at the corrected (zero)
value:** false-alarm rate under repeated peeking must still hold at
nominal. It does — §3.

## 3. Calibration under peeking — the result that matters

Peek after every block, stop and report a false alarm at the first
exclusion; a true null (`true_mean=0`). `alpha=0.05`, 3,000 simulated
universes per row, real CFB weekly block-size sequences unless noted
(`scripts/anytime_validate.py`, `run_calibration_study`):

| Scenario | Looks | Games | Assumed icc | Simulated (true) icc | CS false-alarm | Fixed-sample-peeked false-alarm |
|---|---:|---:|---:|---:|---:|---:|
| CFB 2024 season, worst-case floor (icc=1, proxy=1) | 16 | 737 | 1.00 | 0.00 | **0.00%** | 38.0% |
| CFB 2024 season, operating (headline) | 16 | 737 | 0.00 | 0.00 | **0.00%** | 38.0% |
| CFB 2024 season, operating, stress-tested (true icc=1.0) | 16 | 737 | 0.00 | 1.00 | **0.77%** | 39.5% |
| CFB 2023-2025 (3 seasons) | 47 | 2,244 | 0.00 | 0.00 | **0.00%** | 48.5% |
| NFL-scale synthetic season — 2026's actual monitoring cadence | 18 | 285 | 0.00 | 0.00 | **0.00%** | 41.6% |

**This is the whole argument for the change.** The confidence sequence's
false-alarm rate stays at or below nominal in every configuration,
including the deliberately adversarial one (assumed independence, true
correlation the full worst case — the largest mismatch physically
expressible in this construction). The fixed-sample block bootstrap — the
exact algorithm `paired_feature_comparisons` uses, vectorized for the
simulation — false alarms **38-49%** of the time the moment it is peeked
weekly, roughly **8-9x nominal**, regardless of the ICC decision (peeking
inflation and ICC misspecification are different failure modes; fixing one
does not touch the other).

**Real-data confirmation, not just synthetic:** pairing CFB `market`
against a relabeled copy of itself (true effect exactly zero by
construction, real week/season correlation as it actually occurs) across
all 199 clean_core weeks (9,093 games) never excludes zero at any of the
199 sequential looks; final interval $[-0.130,\ +0.130]$ at the operating
configuration.

## 4. The weekly monitoring rule for the 2026 season

Picks are recorded pre-kickoff every week from the 2026-09-08 lock
(`docs/prospective_evidence.md`); `weekly-run`'s step 11
(`prospective-score`) settles them into one row per (entrant, game) in
`artifacts/prospective_scoring/<run>/settled_decisions.parquet`, already
carrying `correct_at_decision_line` (0/1, NaN if push/pending) and
`pick_side`. **`scripts/anytime_weekly_monitor.py` is what a reader — or a
dashboard — runs against that file each week.**

**What it computes, every week:**

1. Reads the latest settled ledger; takes the `active_model` entrant as
   baseline and `mod07_weak_signal_stack` (configurable) as challenger.
2. Reconstructs the two columns `paired_anytime_comparisons` needs from
   what the ledger already has: `home_cover_probability` as exactly
   1.0/0.0 from `pick_side` (accuracy improvement only ever tests
   `probability >= 0.5`, so nothing is lost), and `home_cover` recovered
   from `pick_side` + `correct_at_decision_line` together.
3. Runs `paired_anytime_comparisons(..., metric="accuracy_improvement",
   block="week", per_game_variance_proxy=0.55)` — `intraclass_correlation`
   left at its default, `0.0` (§2) — producing one confidence-sequence
   look per completed week.
4. Prints the current point estimate, 95% interval, e-value, and the
   diagnostic measured ICC for this specific comparison (informational
   only), translated into one of three plain-English readings.

**What a reader is entitled to conclude, at ANY week, having already
checked every week before it with no penalty for doing so:**

1. **Interval excludes zero on the positive side.** Strong, formally valid
   evidence the challenger is ahead. Rare within one season unless the true
   effect is large (§5) — if it happens, it does not need a second season
   or a rotation window to be trusted.
2. **Interval excludes zero on the negative side.** Strong evidence
   AGAINST the challenger. Different from "unresolved" — worth
   reconsidering its promotion.
3. **Interval still contains zero.** The expected outcome most weeks,
   especially early in the season, and **not a negative result**
   (`AGENTS.md`'s binding rule applies here exactly as it does to the
   fixed-sample method). Report the current point estimate and interval
   width as "how much the evidence has narrowed so far," and check again
   next week at zero validity cost — that is the entire point of building
   this instead of using the fixed-sample bootstrap weekly.

Tested end-to-end against a synthetic ledger matching the real schema (the
2026 season has not started, so there is no real settled ledger to run it
against yet); ready to point at the real artifact from Week 1 onward.

## 5. What the sequence will and will not do over one season

**A fixed, quantified property of the instrument, not a request for more
data.** NFL produces ~285 games a year; the usable NFL history is ~4,431
games and the CFB archive is ~12,500 — that is all there will ever be, and
no construction changes that. Games needed for the confidence sequence to
formally exclude zero, best-tuned mixing variance, at the operating
configuration (`per_game_variance_proxy=0.55`, `intraclass_correlation=
0.0`; `scripts/anytime_validate.py`, `instrument_property_table`):

| Effect (accuracy points) | Games needed, NFL scale (14.7/wk) | Games needed, CFB scale (45.7/wk) |
|---:|---:|---:|
| 0.5 | 202,668 | 202,696 |
| 1.0 | 50,670 | 50,674 |
| 1.3 | 29,988 | 30,020 |
| 2.0 | 12,671 | 12,702 |
| 3.0 | 5,644 | 5,665 |

NFL-scale and CFB-scale numbers are now nearly identical — a direct,
numeric confirmation that block granularity carries no cost once icc=0
(§2.3 of the dropped tightness investigation, below): the variance process
is linear in block size at independence, so week-sized and day-sized (or
game-sized) blocks would give the same answer.

**Say this once, plainly:** roughly 285 games — one NFL season — will not
exclude zero for anything under a few accuracy points. The sequence's job
this season is to catch a *large* effect early (a genuine 3+ point signal
could resolve inside one season) and to stop the pool from over-reading a
small one as it accumulates — not to adjudicate effects in the 0.5-2 point
range the project's own weak-signal work already treats as individually
unresolvable at any realistic sample size. That is not a gap this
instrument failed to close; it is what an honestly-bounded method costs at
this project's real signal-to-noise ratio, and chasing it with more seasons
is not on the table because more seasons are not coming.

### Why the tightness investigation was dropped

A follow-up asked whether a betting-style (WSR) or game-level
cluster-robust construction would meaningfully shrink the table above. One
finding from that investigation, before it was called off, is worth
keeping: the apparent gap between this method and a naive fixed-sample
guess was mostly the (then-unmeasured, later over-padded) ICC assumption,
not block granularity — a direct check (game-sized, size-1 blocks vs the
week-blocked default, at matched ICC) produced *identical* games-needed at
`icc=0`, refuting the specific "aggregating to week-blocks costs power"
hypothesis; §5's NFL-vs-CFB-scale agreement above is the same fact showing
up again now that icc=0 is the operating value, not just a hypothetical.
Beyond that one fact, the investigation was abandoned by design, not by
result: the project owner correctly ruled that any output of the form
"this needs N more games" is a refusal with arithmetic attached once N is
unreachable regardless of which construction produces it, and chasing
30,000 down to a hypothetically tighter 12,000 changes no decision when
neither number will ever be reached. No WSR or game-level variant was
built; the normal-mixture construction in §1 is the only one shipped, and
it is not further optimised.

## 6. The block-bootstrap degeneracy defect, and an audit of existing artifacts

**The defect.** The block bootstrap shared by `paired_feature_comparisons`
and `outcome_bootstrap_intervals` resamples $k$ blocks with replacement
from $k$ available blocks. At $k=1$ there is exactly one possible resample
— the interval collapses to a single point, which excludes zero unless the
realized mean lands exactly on it, a **guaranteed false alarm** with no
signal required. At $k=2$ or $3$ the distribution is not literally a point
mass but is extremely coarse: only $\binom{k+k-1}{k-1}$ distinct achievable
resample compositions exist (3 at $k=2$, **10 at $k=3$, 35 at $k=4$**
*(corrected 2026-08-18: this line previously read "27 at k=3, 256 at k=4",
which is $k^k$ — ordered tuples with repetition, the wrong count. The
multiset formula $\binom{k+k-1}{k-1}$ this same sentence names gives 10 and
35; §6 below independently computes $\binom{4+4-1}{3}=35$ the same way, so
the old numbers contradicted the rest of this document)*, so the
reported interval is a poor approximation of anything, though not
degenerate in the strict sense. This surfaced while building a fair
simulated comparator for §3 (`run_peeking_trial`'s
`min_blocks_before_fixed_sample_check`, default 4, which skips checking the
fixed-sample side below that floor so the calibration comparison in §3
measures the peeking problem, not this separate pathology). It is not a bug
in the anytime code — the confidence sequence handles a single block
correctly (wide, valid, usually inconclusive) by construction — it is a
property of the EXISTING bootstrap that this project should not treat any
interval built from fewer than ~4-5 blocks as reliable, regardless of how
narrow it looks.

*(Correction, 2026-08-18: "~4-5 blocks" undersells this — see the measured
coverage numbers immediately below. k=4 is not a safe floor, it is the
middle of the failure.)* **Measured coverage, by block count** (nominal
95%, against a known true value): **0.000 at k=1, 0.466 at k=2, 0.760 at
k=4, 0.896 at k=10, 0.944 at k=50** — a smooth climb toward nominal, not a
step function, so there is no single k at which the bootstrap suddenly
becomes trustworthy. The project's actual enforced floor is
**`MIN_BLOCKS_FOR_INTERVAL = 10`** (`src/nfl_ats/estimation_variance.py`),
which itself still only reaches 89.6% coverage — short of nominal, but the
shipped compromise, and well above the ad hoc "~4-5" figure this paragraph
used to suggest was adequate. With a discrete win/loss/push estimand the
coarseness bites hardest at the smallest k: **25% of 2-block and 2% of
4-block bootstrap intervals have literally zero width** (every achievable
resample composition happens to land on the same point estimate) — this,
not general narrowness, is the specific mechanism behind the degenerate
`[0.0, 2.2177]` interval discussed below.

**Audit.** Every `sample_blocks` value recorded in `registry/weak_signals.
json` (16 entries) and every window in `registry/rotation_registry.json`
(5 spent windows) was checked.

- `registry/weak_signals.json`: recorded block counts are `None`
  (bootstrap not run at the block level for that entry), 17, 35, 13, 199,
  and **4** for exactly one entry — no entry is recorded with fewer than 4.
- `registry/rotation_registry.json`: the two shortest windows are both
  2-season opener confirmations (`best_pick_ranker_opener`, confirmed,
  `probability_positive=0.865`; `mod07_weak_signal_stack`, unresolved,
  `probability_positive=0.8745`). Checked directly against
  `docs/best_pick_ranker.md` and `docs/mod07_stack.md`: both report
  **week-blocked** intervals (`[-7.00, +22.88]` and `[-1.10, +5.00]`
  respectively) — a 2-season window is still ~34 weeks, far above the
  floor. **Not affected.**

**One entry sat exactly at the floor and showed the defect's symptom —
and tracing it uncovered a second, separate error, now corrected in the
registry.** `player_qb_continuity_matched_alpha` in
`registry/weak_signals.json` was originally recorded as
`classification: refuted_mechanism`, effect +1.1033 accuracy points,
interval `[0.0, 2.2177]`, `sample_blocks: 4`, seasons [2014, 2017], source
`artifacts/qb_continuity_replication/20260816T143913Z/paired_comparisons.
csv`. Tracing the source CSV confirmed the degenerate interval was the
`block="season"` row on exactly 4 season-blocks (the `block="week"` row for
the same 997 games has ~68 blocks and is not at risk) — its lower bound
landing **exactly on 0.0** is the telltale symptom of a resampling
distribution with only $\binom{4+4-1}{3}=35$ achievable compositions rather
than a genuinely smooth 90%/95% quantile. That part of the finding stands.

**What did not survive tracing: the STRUCTURAL argument this document
originally gave for why the verdict "survives" regardless of the interval's
precision.** The text below, kept verbatim as the record of the error, read:

> 1. The entry's own `classification_evidence` argues `refuted_mechanism` on
>    STRUCTURAL grounds — the two compared arms differ only in ridge alpha
>    (1 vs 10), a near-null contrast by construction (0.03% vs 0.27% median
>    shrinkage difference, `docs/groupwise_ridge.md`) — independent of
>    trusting any interval's precision.
> 2. The companion **week-blocked** interval on the identical 997 games
>    (~68 blocks, well above the floor) is `[-0.011, +0.038]` at 90%
>    confidence — also comfortably straddling zero, qualitatively agreeing
>    with the coarse season-blocked number despite the latter's unreliable
>    precision.
>
> The verdict (`refuted_mechanism`, and by inheritance `player_qb_continuity`
> closed_negative in the rotation registry) stands.

**Reason 1 is factually wrong about this entry.** Checked directly against
`predictions.parquet` in the same artifact directory: the `candidate` and
`base_alpha1` arms — the pair this entry actually compares — are **both**
`ridge_alpha=1.0`; only the feature profile differs. The "alpha 1 vs 10,
near-null by construction" argument describes a *different, separate*
contrast (`base_alpha1` vs `base_alpha10`), which is real, resolvably
negative, and flips only 25/997 picks (2.51%). The feature contrast this
entry is actually named for flips **177/997 picks (17.75%)**, split 94-83,
paired SE 1.33, MDE80 3.74 — a real, if underpowered, effect, not a
near-null contrast. Reason 2's week-blocked interval number was also
computed on the wrong pairing for the same reason.

**Corrected, 2026-08-18:** `registry/weak_signals.json` now records this
entry as `classification: unresolved_below_power`, interval
`[-1.5126, 3.7192]`, `sample_blocks: 68` (not 4 — the week-blocked pairing
on the correct arms, not the degenerate 4-season-block one), `effect
+1.1033`, `probability_positive 0.796`. It is **not** `refuted_mechanism`
and does **not** close `player_qb_continuity` negative in the rotation
registry; that inheritance claim above is also void. The original
degeneracy finding (interval `[0.0, 2.2177]` being an artifact of a 4-block
resample) is still correct and is why the 68-block figure, not the 4-block
one, is now the one of record. No other recorded verdict in either registry
rests on a bootstrap below the `MIN_BLOCKS_FOR_INTERVAL` floor of 10.

## 7. What this replaces, what it does not

**Replaces:** the STOPPING RULE for continuous monitoring, not the
rationing rule. A family still draws at most one rotation window
(`docs/rotation_registry.md`'s mechanics are untouched), but prospective
2026 scoring — which already "needs no window at all" — can now be watched
every week with a valid, continuously-updating readout instead of a
fixed-sample bootstrap that was never built to survive being checked that
often.

**Does not replace:** window assignment, contamination inheritance, or the
2018-2025 multiplicity discount. Does not manufacture power that does not
exist in a 285-game season (§5). Does not license treating an unresolved
anytime interval as a negative — `AGENTS.md`'s rule applies here exactly as
it does to the fixed-sample method. Does not re-derive the independence
decision per comparison — §2's diagnostic functions report, they do not
decide.

## Reproducing this

```
.tools/uv.exe run --no-sync python scripts/anytime_validate.py \
    --output <scratch>/anytime            # calibration + ICC diagnostic + appendix, ~1 min
.tools/uv.exe run --no-sync python scripts/anytime_weekly_monitor.py \
    --settled-decisions <path>             # the weekly reading, once real 2026 data exists
.tools/uv.exe run --no-sync pytest tests/test_anytime.py
```

Uses the cached `artifacts/cfb_benchmark/20260818T115149Z/predictions.parquet`
and two other cached CFB comparisons by default; rebuild via
`nfl-ats cfb-benchmark` if those are ever pruned. No NFL outcome data is
read by this script; `registry/rotation_registry.json` is not written.
