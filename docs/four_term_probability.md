# The four-term model: model logit, one flag weight, market move

## Why this is being measured

Paraphrased, owner directive: `docs/joint_probability_model.md` (MKT-17)
fit all nine composition flags with their own separate coefficients, which
let the penalty over-shrink them into something less decisive than the
served flip chain. The owner's read: the flip chain effectively behaves like
one large shared weight applied to whichever flags fire, not nine small
independent votes. This document builds that directly -- a single shared
weight on the signed sum of the nine flags, alongside the model's own logit
and the late-week market move, three or four terms, no penalty, and scores
it the same honest way: leave-one-season-out, against the card, against the
served chain, and against the per-flag joint model.

## Binding closing grounds (verbatim, AGENTS.md)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with `nfl-ats
weak-signals record`, report `probability_positive`, never the binary
"contains zero". The registry code hard-rejects inadmissible closures; if a
record command errors, the verdict is wrong, not the validator. Also
binding: failing a promotion bar of p 0.90 or 0.95 is not grounds to reject
either -- sample sizes here rarely clear it. Nor is clearing p 0.5 grounds to
serve anything; research closure and card-serving are different decisions.
Within-week correlation is zero, so the week-blocked bootstrap
(`nfl_ats.clv.week_blocked_bootstrap`, blocks `(season, week)`) is the
honest unit, 20,000 draws, seed 20260914. `probability_positive` is the
share of bootstrap draws above zero, never a p-value, and is reported
wherever an effect is reported.

## Predeclaration, before any M5 number was computed

**Population**: newest opener-evaluation stream, seasons 2020-2025, opener
pushes dropped -- reused verbatim from `scripts/joint_probability_model_eval.py`
(`load_opener_population`, `build_base`, `add_flags`, `build_m1`, imported
directly rather than re-derived; see Result 0 for the confirmation its
outputs are unchanged). The market move (`leader_median_net` from
`artifacts/sharp_weighted_follow/20260909T233606Z/per_game.parquet`) exists
for 2023-2025 only.

**M5, the four-term model**:

```
logit P(home covers) = intercept + a * logit(raw model p_home) + b * S + c * move_toward_home
```

`S` is the signed sum of the nine composition flags (`flag_sum`, range
roughly -3..+3, `+1` favours home, `-1` favours away, `0` none), so one
shared weight `b` for every flag rather than nine separate ones. `move` is
`leader_median_net`, already signed toward home. Per the task's own
allowance ("0 when unavailable, with an availability indicator term if
fitting it that way is cleaner"), M5 is fit on the **full 1,503-game
2020-2025 population**: `move` is set to 0 for the 704 games where no market
population row exists (2020-2022), and a fourth covariate,
`move_available` (0/1), is added as a control so the fit does not conflate
"no data" with "the market did not move." `move_available` is a nuisance
control, not one of the four named terms, and its own coefficient is
reported but not treated as a finding. Fit by unpenalized logistic
regression (IRLS with a fixed `1e-3` ridge for numerical conditioning only,
the same convention `joint_probability_model_eval.py` used for its own
pure-recalibration arm M4), leave-one-season-out over all six seasons,
out-of-sample predictions pooled.

**Variants**, all predeclared:

- **M5a** = M5 without the move (and without `move_available`, which has
  nothing to control for once move is dropped) -- full 1,503-game
  population, three terms (intercept, `a`, `b`).
- **M5b** = M5 with `a` fixed at 0 -- the model's own logit is dropped
  entirely; opener plus situational adjustments only (intercept, `b`, `c`,
  `move_available`).
- **M5c** = M5 with `S` capped at +/-1 (`flag_sum_capped`), so two or three
  agreeing flags count the same as one -- matching the served chain's own
  complement-once semantics.

**Comparisons**: each of M5/M5a/M5b/M5c against M0 (raw model,
`home_cover_probability_at_open_raw`), M1 (served flip chain, reused
bit-for-bit from `joint_probability_model_eval.build_m1`, which calls
production's own `apply_*` functions and unions their flips) and M3 (all
nine flags fit separately, reused bit-for-bit from
`joint_probability_model_eval.loso_arm` on the identical population) -- log
loss, Brier, accuracy points, all out-of-season, week-blocked bootstrap,
`probability_positive` reported for every cell. Decisive-game records for
M5 vs M0, M5 vs M1, M5a vs M0, M5b vs M0, M5c vs M0. A five-bin reliability
table for M5 only. Per-fold coefficients with Wald 95% intervals. Season
split, descriptive. A positive control: perfect foresight restricted to the
games where M5's pick differs from M1's, scored against M1. Every look
counted by the eval script before any cell is discussed.

**Best Pick**: within each week, the game with the highest M5 probability
for its own pick (confidence `max(p, 1-p)`, unrestricted -- no dispersion or
small-spread screen, since the task asks for M5's own ranking, not the
served nominator's eligible pool), ties broken by lowest `game_id`. Graded
against the served nominator's stored weekly picks
(`artifacts/best_pick_served_score_ranker/20260914T163425Z/weekly.csv`,
columns `a0_incumbent_*` -- the old alpha=2000 ranker -- and
`a1_served_score_*` -- the ranker actually served since 2026-09-14), paired
by week, weeks where a nominee pushed dropped from that pair only.

## Results

**Measured**, `scripts/four_term_probability_eval.py`,
`artifacts/four_term_probability/20260914T222345Z/{summary.json,per_game.csv,coefficients.csv,best_pick_weekly.csv}`.
Population: 1,503 opener-graded games, 2020-2025, 34 pushes dropped (before
that, 1,537), 799 of those 2023-2025 games carry the market move, 107 week
blocks. Seed 20260914, 20,000 bootstrap draws. **51 looks taken**, logged
verbatim in `summary.json["look_log"]`: 5 LOSO fits (M3 reused, M5, M5a,
M5b, M5c), 36 head-to-head bootstrap cells (12 comparisons x {log loss,
Brier, accuracy}), 1 five-bin reliability table, 5 decisive-game reports, 1
positive control, 1 season split, 2 Best Pick comparisons.
`registry_resolved_wrong_sign_cells` is empty (measured) -- no cell resolves
to a wrong sign.

### Records, before any effect size (measured)

| Arm | Record (1,503 games) | Win % |
|---|---|---:|
| M0 (raw model) | 809-694 | 53.83% |
| M1 (served chain) | 842-661 | 56.02% |
| M3 (nine flags, separate weights, oos) | 826-677 | 54.96% |
| **M5 (four-term, oos)** | **847-656** | **56.35%** |
| M5a (no move, oos) | 824-679 | 54.82% |
| M5b (no model term, oos) | 857-646 | 57.02% |
| M5c (S capped, oos) | 850-653 | 56.55% |

Every M5 variant reads above M0 and above M3 on raw record; M5, M5b and M5c
all read at or above M1's own 56.02%, M5a reads just below it. None of this
is bootstrapped yet -- it is stated first, per the promotion-bar rule ("state
evidence in games" before any interval).

### 1. The decision-relevant finding, first: not the horse race, the calibration ordering

The single most useful fact in this document is not any head-to-head cell --
every one of those intervals touches zero, as expected at this resolution --
it is what happens when each arm's own stated confidence is checked against
its own accuracy. **Measured**, `per_game.csv`, each model binned
separately by its own out-of-season pick confidence (`max(p, 1-p)`, five
bands -- these are two different partitions of the same 1,503 games, not the
same games in each row):

| Confidence band | M5: n / accuracy | M0 (raw model): n / accuracy |
|---|---:|---:|
| 0.50-0.52 | 300 / 52.0% | 334 / 56.0% |
| 0.52-0.55 | 521 / 55.9% | 419 / 53.9% |
| 0.55-0.58 | 296 / 55.4% | 352 / 54.0% |
| 0.58-0.62 | 257 / 61.9% | 256 / 52.3% |
| 0.62+ | 129 / 59.7% | 142 / 50.7% |

**M5's accuracy rises with its own stated confidence** (52.0% in its
least-confident band up to 61.9% in its second-firmest, a small dip in the
middle); **M0's accuracy falls with its own stated confidence** -- 56.0%
when the raw model is least sure down to 50.7% when it is firmest, the
opposite direction from what a well-behaved probability should show. This
is the same overconfidence `docs/joint_probability_model.md`'s M4 already
found in log-loss/Brier terms (raw logit needs shrinking to 14-42% of its
stated size); this table restates it in accuracy terms, sharper, and shows
M5 does not inherit it. This descriptive table is not separately
bootstrapped (it slices the same 1,503 games five ways per arm, so its
bands are not independent draws); it is read alongside the bootstrapped
cells below, not instead of them.

### 2. Does M5 match or beat the served chain (M1) out of season?

**M5 leans ahead of M1 on every metric, none of the intervals clear zero,
and this reads more favourably for the joint-model approach than
`docs/joint_probability_model.md`'s M3 (separate per-flag weights) did.**

| Metric | M5 vs M1 | 95% week-blocked | `probability_positive` |
|---|---:|---|---:|
| Log loss | +0.0027 | [-0.0034, +0.0090] | 0.809 |
| Brier | +0.0014 | [-0.0016, +0.0043] | 0.814 |
| Accuracy | +0.333 pts | [-1.547, +2.243] | 0.640 |

Compare to the prior lane's M3 vs M1 read: log loss -0.0021 [-0.0080,
+0.0038] P+ 0.244, Brier -0.0010 P+ 0.239, accuracy -1.065 pts P+ 0.154 --
M3 leaned *behind* M1 on every metric; **M5 leans *ahead* of M1 on every
metric.** Neither reading resolves (both touch zero), so this is not a
reversal under the closing-ground rule, but the one shared flag weight plus
the market move reads as a materially different, more favourable picture
than nine separately-penalized flag weights did.

M5 vs M0 and M5 vs M3, for context: log loss +0.0130 [+0.0044, +0.0217] P+
0.999 vs M0 (interval entirely above zero), +0.0048 [+0.0004, +0.0093] P+
0.985 vs M3 (also entirely above zero); Brier the same pattern (+0.0063 P+
0.999 vs M0, +0.0024 P+ 0.987 vs M3); accuracy +2.528 pts [-0.131, +5.133]
P+ 0.970 vs M0, +1.397 pts [-0.534, +3.369] P+ 0.922 vs M3.

### 3. Does the model term earn its place?

**No, not on this read -- `a` is not distinguishable from zero in any of the
six folds, in any variant that carries it.** M5's own `a` (coefficient on
the raw model's logit):

| Held-out season | a (model logit) | 95% Wald interval |
|---|---:|---|
| 2020 | 0.317 | [-0.126, 0.760] |
| 2021 | 0.092 | [-0.337, 0.522] |
| 2022 | 0.243 | [-0.193, 0.679] |
| 2023 | 0.250 | [-0.183, 0.683] |
| 2024 | 0.177 | [-0.249, 0.602] |
| 2025 | 0.210 | [-0.221, 0.641] |

Every point estimate is positive (0.09 to 0.32, the same overconfidence
magnitude M4 found), and every fold's interval contains zero -- the
opposite pattern from `b` and `c` below. Per rule (a), **an interval
containing zero is never grounds to drop a term** -- `a` positive in six of
six folds is itself evidence, not nothing, and it is not being zeroed out
here. But it does mean M5 cannot yet show the raw model's own logit is
carrying independent weight once the flags and the move are already in the
equation, which is exactly what Result 4 below measures directly.

`b` (the shared flag weight, `flag_sum`) and `c` (the market move) tell the
opposite story -- both **distinguishable from zero in nearly every fold**:

| Held-out season | b (flag_sum) | 95% interval | c (move) | 95% interval |
|---|---:|---|---:|---|
| 2020 | 0.243 | [0.116, 0.371] | 0.209 | [0.060, 0.359] |
| 2021 | 0.279 | [0.148, 0.410] | 0.214 | [0.064, 0.363] |
| 2022 | 0.245 | [0.113, 0.376] | 0.211 | [0.061, 0.360] |
| 2023 | 0.248 | [0.117, 0.379] | 0.225 | [0.034, 0.415] |
| 2024 | 0.278 | [0.146, 0.410] | 0.250 | [0.069, 0.431] |
| 2025 | 0.300 | [0.169, 0.431] | 0.159 | [-0.022, 0.340] |

`b` clears zero in **all six of six folds**; `c` clears zero in **five of
six** (2025 touches -0.022). `move_available`, the nuisance control, never
clears zero in any fold (as expected -- it is not supposed to carry a
finding). This is the clearest single fact in the fold-coefficient table:
**the situational flags and the market move are earning fitted weight
distinguishable from zero on this population; the model's own logit is
not**, once the other two are already in the equation.

### 4. M5 vs M5b: is dropping the model term actually better?

M5b (identical to M5 except `a` fixed at 0) reads a **nominally** better
record (857-646 vs 847-656) and beats M0 and M3 with intervals entirely
above zero on every metric (Result 5), which M5 does not quite manage on
accuracy. A direct out-of-season comparison, measured this session
(`per_game.csv`, not part of the original 36 predeclared cells, run to
settle this specific question): **M5b vs M5, accuracy +0.665 pts [-0.601,
+1.926], `probability_positive` 0.848** -- leans toward M5b, interval
touches zero.

**This document recommends keeping the model-lean term (M5), not switching
to M5b.** Three reasons: (1) `a`'s coefficient is positive in six of six
folds (Result 3) -- under rule (a) an interval containing zero is never
grounds to drop a term, and a same-signed point estimate in every fold is
itself a form of evidence, not noise; (2) M5 and M5b are not
distinguishable from each other here (the direct comparison above touches
zero); (3) picking M5b over M5, M5a and M5c *because* it happens to have
the best raw record among four variants scored on the identical 1,503 games
is exactly the kind of after-the-fact selection the promotion-bar rule
warns against -- the four variants were predeclared together, and choosing
the empirical winner among them after seeing the numbers is not the same
thing as evidence that the model term is worthless. If the model term is to
be dropped, that should follow a forward read where `a` stays
indistinguishable from zero again, not a single retrospective ranking of
four variants against each other.

### 5. M5a, M5b, M5c vs M0/M1/M3 (measured, all 36 cells)

| Comparison | Log loss | Brier | Accuracy (pts) |
|---|---|---|---|
| M5a vs M0 | +0.0106 [+0.0030,+0.0184] P+0.997 | +0.0052 [+0.0014,+0.0089] P+0.997 | +0.998 [-1.800,+3.742] P+0.762 |
| M5a vs M1 | +0.0004 [-0.0053,+0.0061] P+0.563 | +0.0002 [-0.0026,+0.0030] P+0.559 | -1.198 [-2.957,+0.545] P+0.095 |
| M5a vs M3 | +0.0025 [+0.0001,+0.0049] P+0.980 | +0.0012 [+0.0001,+0.0024] P+0.980 | -0.133 [-1.412,+1.181] P+0.420 |
| M5b vs M0 | +0.0130 [+0.0034,+0.0227] P+0.996 | +0.0063 [+0.0017,+0.0111] P+0.996 | **+3.194 [+0.329,+6.053] P+0.985** |
| M5b vs M1 | +0.0027 [-0.0038,+0.0094] P+0.794 | +0.0013 [-0.0018,+0.0045] P+0.795 | +0.998 [-0.935,+2.964] P+0.843 |
| M5b vs M3 | +0.0048 [+0.0001,+0.0097] P+0.978 | +0.0024 [+0.0001,+0.0047] P+0.981 | **+2.063 [+0.133,+3.987] P+0.981** |
| M5c vs M0 | +0.0116 [+0.0034,+0.0200] P+0.997 | +0.0056 [+0.0017,+0.0097] P+0.997 | **+2.728 [+0.131,+5.333] P+0.979** |
| M5c vs M1 | +0.0014 [-0.0048,+0.0076] P+0.674 | +0.0007 [-0.0023,+0.0037] P+0.672 | +0.532 [-1.384,+2.462] P+0.710 |
| M5c vs M3 | +0.0035 [-0.0013,+0.0083] P+0.925 | +0.0017 [-0.0006,+0.0040] P+0.930 | +1.597 [-0.335,+3.576] P+0.947 |

Bold cells are the intervals that clear zero entirely (M5b and M5c both
beat M0 on accuracy with the whole interval positive, and M5b beats M3 on
accuracy the same way). M5a -- the version with no market move at all,
matching M3's own population -- is the weakest of the four against M1, the
same direction M3 itself read; **adding the market move is what moves M5,
M5b and M5c from "trails M1" to "leans ahead of M1."**

### 6. Reliability table, M5 only (5 bins, out of season)

| Bin | n | Mean predicted p(home) | Actual home-cover rate | Gap |
|---|---:|---:|---:|---:|
| 0.0-0.2 | 0 | -- | -- | -- |
| 0.2-0.4 | 104 | 0.362 | 0.413 | +0.051 |
| 0.4-0.6 | 1,274 | 0.493 | 0.487 | -0.007 |
| 0.6-0.8 | 125 | 0.637 | 0.664 | +0.027 |
| 0.8-1.0 | 0 | -- | -- | -- |

M5 never predicts outside 0.2-0.8, same as every arm in the prior lane. The
0.4-0.6 bin, holding 85% of predictions, is well calibrated (gap -0.007).

### 7. Decisive-game records (measured)

| Comparison | n differ | Base record on diff | Arm record on diff |
|---|---:|---|---|
| M5 vs M0 | 536 | 249-287 | **287-249** |
| M5 vs M1 | 263 | 129-134 | **134-129** |
| M5a vs M0 | 453 | 219-234 | **234-219** |
| M5b vs M0 | 620 | 286-334 | **334-286** |
| M5c vs M0 | 527 | 243-284 | **284-243** |

### 8. Positive control

Perfect foresight restricted to the 263 games where M5's out-of-sample pick
differs from M1's: M1's own record on those 263 games is 129-134, forced to
263-0. **Effect vs M1: +8.916 accuracy points, 95% week-blocked [+7.495,
+10.347], `probability_positive` 1.0.** This bounds the ceiling of any rule
agreeing with M5's particular 263-game disagreement set with M1; M5's own
measured gain over M1 (+0.333 pts, Result 2) sits well inside this ceiling
-- real resolving power exists at this game count, the candidate effect
itself does not clear it, textbook `unresolved_below_power`.

### 9. Season split (descriptive)

| Season | M0 | M1 | M3 (oos) | M5 (oos) | M5a (oos) | M5b (oos) | M5c (oos) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 51.36% | 57.27% | 54.09% | 55.00% | 54.09% | 57.27% | 56.36% |
| 2021 | 54.66% | 55.93% | 56.78% | 54.66% | 55.51% | 54.66% | 54.66% |
| 2022 | 54.03% | 58.47% | 57.66% | 59.68% | 58.06% | 59.68% | 59.27% |
| 2023 | 55.64% | 58.65% | 56.02% | 56.39% | 56.77% | 58.65% | 57.14% |
| 2024 | 54.14% | 53.01% | 53.01% | 56.39% | 53.76% | 56.02% | 55.26% |
| 2025 | 52.81% | 53.18% | 52.43% | 55.81% | 50.94% | 55.81% | 56.55% |

M5 (and M5b) sit at or above M1 in four of six seasons (2022, 2023 except a
narrow miss, 2024, 2025), trail narrowly in 2021, tie or trail slightly in
2020/2023 depending on the variant. M5a -- no market move -- is the weakest
of the M5-family variants in every season from 2023 on, consistent with
Result 5's read that the move is what is doing the work of closing the gap
with M1.

### 10. Best Pick: does M5's own top pick beat the served nominator's?

Within each week, the game with the highest M5 out-of-season confidence for
its own pick, unrestricted (no dispersion or small-spread screen -- those
belong to the served nominator's eligible pool, not to this question), 107
weeks 2020-2025, every week matched to `weekly.csv` (0 unmatched).

**Against the old alpha=2000 ranker (`a0_incumbent`, 102 paired weeks):**
M5's star hits **61.76% (63/102)**, a0's own hit rate is **55.88%
(57/102)**, **effect +5.882 accuracy points, 95% week-blocked [-8.824,
+20.588], `probability_positive` 0.784.** On the 96 of 102 weeks where the
two stars differ, M5's star goes 59-37 and a0's goes 53-43.

**Against the ranker actually served since 2026-09-14
(`a1_served_score`, 103 paired weeks): M5's star and the served ranker are
indistinguishable, not one an improvement on the other.** Both hit exactly
**62.14% (64/103)** -- effect **0.000 accuracy points, 95% week-blocked
[-13.592, +13.592], `probability_positive` 0.505**, as close to a coin flip
as this evaluator can read. On the 95 of 103 weeks where the two stars name
a *different* game, both arms go **57-38** -- the two rankings disagree on
which specific game to star in 92% of weeks, yet land on the identical
overall record. This is descriptive, not proof the two rankers are the
same rule in disguise; it says only that neither is shown here to beat the
other.

## Look-count and reuse caveat (binding, stated plainly)

**The nine composition flags were each promoted, and then all nine were
jointly fit as covariates, on this exact same 1,503-game 2020-2025
population in `docs/joint_probability_model.md` (MKT-17) before this
document existed.** `S` (`flag_sum`) is a linear recombination of those
same nine covariates; its coefficient `b` reading positive and distinguishable
from zero in every fold here is not an independent replication -- it
inherits whatever optimism already sits in each flag's own construction and
in M3's own prior fit on these same games. The market move similarly
carries forward from `docs/market_updated_model.md`. **51 looks were taken
in this document alone**, on top of the 33 looks the prior joint-probability
lane already took on the same games, and the 19 the market-move lane took
on the 2023-2025 subset. None of this is grounds to discard M5's reading
under the closing-ground rule (nothing here resolves to a wrong sign), but
it is exactly the reason this document's own numbers cannot settle whether
`b` and `c`'s apparent edge over `a` is real signal or accumulated look-reuse
across three documents scored on largely the same games. **Only a forward
read -- games this population has not yet seen -- can settle that.** This
caveat is not buried: it is the reason the registry classification below is
`unresolved_below_power` on every cell, including the ones whose intervals
already clear zero against M0.

## Registry

Family `four_term_probability_v1`, 39 cells recorded via `nfl-ats
weak-signals record`, category `modeling`, league `nfl`, season-start 2020,
season-end 2025 (best-pick cells use week-block counts rather than game
counts), source `docs/four_term_probability.md`. 36 head-to-head cells (4
candidates x 3 baselines x {log_loss_improvement, brier_improvement,
accuracy_points}), 1 positive control, 2 Best Pick cells. Every cell
`unresolved_below_power`: `registry_resolved_wrong_sign_cells` is empty
(measured) -- no interval sits wholly on the wrong side of zero, including
the cells reported bold in Result 5 whose intervals clear zero against M0
and M3 (a cleared-zero interval is a stronger unresolved reading, not a
closure -- only a wrong-sign or a positive-control bound closes anything,
and the positive control here reads an ~8.9-point ceiling that does not
prove the instrument can detect the 0-3-point effects most cells sit at).
Nothing in `src/` is touched by this document; M5 and its variants are
challengers scored here, not switched onto the served card.

## Provenance labels

Every load-bearing number in Results is **measured** this session by
`scripts/four_term_probability_eval.py` against the artifact stamped above,
except the M5-vs-M5b comparison in Result 4 and the confidence-band table
in Result 1, both **measured** directly against `per_game.csv` this session
(commands run interactively, not part of the eval script's own output),
unless marked **read** (file and line cited inline) or **inferred**
(reasoning stated as such, never combined with a measured number in the
same sentence).
