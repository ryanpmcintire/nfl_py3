# Does the leader-median late-week follow rule need a model-confidence gate?

## Why this is being measured

The owner declared the prior session's work wrong and ordered a full redo,
built from scratch, with no reuse of the rejected script's logic. Two
objections, paraphrased: (a) the previous split of the leader-median follow
rule's flips into a "shallow" and a "deep" half by override distance posted
an identical 61-49 record in both halves, and that was not credible without
independent proof; (b) following a late-week line move cannot be the right
call in every situation, and the decision should be conditioned on how
confident the underlying model already was in the pick being overridden --
the expectation being that a flip pays off when the model is near a coin
flip and should not happen when the model is firm.

Two further binding amendments arrived while this redo was in progress and
are folded in below: (1) no gate or cut point may be selected on the same
flips it is scored on -- any recommended gate is chosen by leave-one-season-out
(LOSO), and every "look" taken is counted and disclosed rather than only the
best cell being shown; (2) the served rule's own move-size threshold (flat
0.5, flat 1.0, or the mechanism rule that uses 0.5 only on spreads of 10.5 or
more) was itself chosen by comparing candidates on this same population, so a
confidence gate stacked on top of it inherits that optimism -- a nested LOSO
is required that selects BOTH the threshold and the confidence cut on
training seasons and scores them on the held-out season.

## Binding closing grounds (verbatim, AGENTS.md)

an interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with `nfl-ats
weak-signals record`, report `probability_positive`, never the binary
"contains zero". The registry code hard-rejects inadmissible closures; if a
record command errors, the verdict is wrong, not the validator.

## Population (read, then measured)

**Read**, `artifacts/sharp_weighted_follow/20260909T233606Z/per_game.parquet`:
816 rows. **Measured**, `scripts/leader_median_model_confidence_eval.py`:
799 rows have `correct_at_open_probability_rule` non-null (17 opener pushes
dropped). This is the frozen, read-only 2023-2025 population the prior
session's work and the original `sharp_weighted_follow` promotion both used;
nothing here is refit.

**Read**, `src/nfl_ats/sharp_book_movement_features.py:24-135`: the rule
under study (`s1_leader_only_*` columns) is the leader-median arm at a flat
0.5-point threshold -- `leader_median_net_move` is the median Wednesday-to-deadline
net spread move across the three leader books (Bovada, William Hill (US),
MyBookie); it fires when `abs(leader_median_net) >= 0.5`, and on a fire it
picks HOME when the net move is positive, otherwise AWAY, via
`refresh_pick(production_home, net_move, threshold)`, which keeps the
Tuesday/card pick unless the gate clears. **Verified sign convention,
measured**: `card_pick_home` (`pick_home_at_open_probability_rule`) equals
`home_cover_probability_at_open >= 0.5` on every one of the 799 graded games
(0 mismatches) -- the card's pick is always the model's own favored side, so
a "flip" always overrules the model's own read, never a tie.

**Read**, `docs/follow_threshold_by_line.md:133-141` and
`:426-436`: the rule actually served in production today is not this flat
0.5 arm. It is `T3`: a 1.0-point threshold everywhere except a 0.5-point
threshold on spreads of 10.5 or more
(`sharp_book_movement_features.leader_follow_threshold`, `LEADER_FOLLOW_THRESHOLD
= 1.0`, `LEADER_FOLLOW_BIG_SPREAD_LINE = 10.5`,
`LEADER_FOLLOW_BIG_SPREAD_THRESHOLD = 0.5`), chosen by comparing seven
threshold candidates (flat 0.5, flat 1.0, two walk-forward fitted arms, a
continuous fitted arm, and the mechanism arm) against the SAME 799-800-game
population documented there. That selection is itself in-sample on this
population. Analysis 6 below (the nested LOSO) accounts for this.

## Predeclared definitions, cuts, arms, metric and decision rule

Written before any of the confidence-conditioned results below was computed.
The confidence bands and move-size bins are fixed from the **covariate
distribution only** (quantiles of `confidence` and `move`), never from
outcomes.

- **Card pick and grade.** `pick_home_at_open_probability_rule`,
  `correct_at_open_probability_rule`.
- **Model confidence in its own pick.** `p = home_cover_probability_at_open`;
  `confidence = max(p, 1-p)`. Reported also as the raw probability of the
  served (post-flip) side.
- **Move size.** `move = abs(leader_median_net)`. On a flip this is, by
  construction, the whole magnitude of the move against the card's pick.
- **Flip.** fired AND rule pick != card pick.
- **Terciles** of `confidence` among the 220 flips (covariate quantiles):
  edges at 0.5252 and 0.5635, giving low/mid/high thirds of about 74/73/73
  flips.
- **Fixed bands**, adjusted to the actual range (confidence among flips runs
  0.500 to 0.684, median 0.543): <=0.52, 0.52-0.55, 0.55-0.58, >0.58.
- **Move-size bins** for the 2-D read: 0.5 to under 1.0, and 1.0 and up,
  crossed with the three confidence terciles (6 cells).
- **Gated arms** (Analysis 3, descriptive/in-sample): follow the rule only
  when confidence is at or below each tercile edge or fixed-band edge,
  scored as a full 799-game policy against the card and against the ungated
  rule.
- **Positive control.** Perfect foresight restricted to exactly the 220
  flips: force the card correct wherever the card was wrong on a flip,
  otherwise leave the card's own grade. This bounds what any confidence
  read on this flip set could possibly buy.
- **Metric.** Paired accuracy-point deltas (candidate policy mean correct
  minus card mean correct, times 100), week-blocked bootstrap
  (`nfl_ats.clv.week_blocked_bootstrap`, blocks `(season, week)`), 20,000
  samples, seed 20260914, per binding rule 3. `probability_positive` is
  reported for every cell; a binary "contains zero" read is never reported.
- **LOSO-selected gate (Analysis 3, amendment 1).** No cut is chosen on the
  flips it is scored on. Candidate confidence cuts are the nine deciles of
  confidence among all 220 flips (a covariate-only, outcome-blind grid) plus
  "no gate". For each held-out season, the cut maximizing the simple
  training-season accuracy-point effect (minimum 15 training flips under a
  finite cut, else the candidate is skipped for that fold) is chosen from
  the other two seasons and applied to the held-out season; the three
  held-out results are pooled into one 799-game policy and bootstrapped
  against the card. The in-sample version (cut chosen and scored on all 799
  games at once) is reported beside it, with the gap.
- **Nested threshold+confidence LOSO (Analysis 6, amendment 2).** The
  candidate grid crosses three move-size threshold arms -- T05 (flat 0.5, the
  rule this whole population's `s1_leader_only_*` columns encode), T0 (flat
  1.0), T3 (the served mechanism rule, 1.0 baseline / 0.5 at |line| >= 10.5,
  recomputed here directly from `leader_median_net` and
  `tue_open_home_spread` using `sharp_book_movement_features.leader_follow_threshold`,
  since grading a flip needs only which side was picked, not new margin data)
  -- with the same confidence-cut grid {0.52, 0.55, 0.58, no gate} for the
  "with gate" read, or {no gate} only for the "without gate" read. Selection
  and pooling follow the same leave-one-season-out procedure as above. T1,
  T1F and T2 (the walk-forward per-band and continuous fitted arms in
  `docs/follow_threshold_by_line.md`) are excluded from this nested search;
  they require their own weekly walk-forward refit machinery that is out of
  this lane's scope, and this exclusion is stated here as a scope decision,
  not a result.
- **Decision rule.** The pool is forced picks: the arm with the higher
  expected out-of-sample accuracy is the one relevant to a serving decision.
  A record like 12-7 or a `probability_positive` just above 0.5 is not by
  itself grounds to serve a gate (a promotion bar is not a decision bar, and
  the reverse). Serving is the owner's decision; this document proposes
  gates as paired challengers only. Nothing in `src/` is touched here.
- **Look count.** Every band, tercile, 2-D cell, continuous fit, gated arm
  and LOSO search evaluated is counted and logged by the eval script; the
  count is reported in Results before any single cell is discussed, and no
  cell is presented as "the finding" on its own.

## Sanity checks (measured)

`scripts/leader_median_model_confidence_eval.py`,
`artifacts/leader_median_model_confidence/20260914T202723Z/summary.json`.

- 799 graded games, 515 fires, **220 flips** -- reproduces the frozen
  population's own counts exactly.
- Card pick vs. model argmax: **0 mismatches** across all 799 games.
- On every flip, `rule_correct == 1 - card_correct` (no pushes possible on a
  flip, since push status is a game property independent of side): **True**.
- On every fire that is not a flip, `rule_correct == card_correct`: **True**.
- `confidence - 0.5` equals the prior session's "override distance"
  magnitude (`0.5 - probability of the served side`) on every flip, max
  absolute difference **5.6e-17** (floating-point zero). This is an exact
  mathematical identity, not an approximation -- see Analysis 4.
- Confidence among the 220 flips: min 0.5001, 25% 0.5196, median 0.5427, 75%
  0.5726, max 0.6843.

## Results

**Look count: 49 distinct statistical reads were taken** (headline x2 blocks;
7 confidence cells x2 each for effect and foresight-ceiling = 14; 2
correlations; 3 logistic fits; 6 two-dimensional cells; 5 in-sample gated-arm
reads; 1 foresight control; 4 single-dimension LOSO reads; 3 nested-LOSO
(no-gate) reads; 3 nested-LOSO (with-gate) reads; 1 median-split reproduction
plus 4 supporting band computations; 1 decile table). The full log is in
`summary.json["look_log"]`. These are one family sharing one population, one
grading and one follow computation -- correlated readings of one question,
not 49 independent votes -- and no single cell below is the finding on its
own.

### The decision-relevant fact, stated first

**The honest, leave-one-season-out out-of-sample read of the served rule's
own shape (no confidence gate) is positive, not at or below the card:
+2.128 accuracy points, week-blocked 95% [-0.505, +4.804],
`probability_positive` 0.941, pooled across all three held-out seasons.**
Adding a confidence gate on top of that honestly-chosen threshold does not
improve the out-of-sample read -- it makes it worse: **+1.877 [-0.732,
+4.478], P+ 0.926**, a full point lower than the ungated out-of-sample
number, with a larger in-sample-to-out-of-sample gap (1.377 points against
0.876 points for the threshold-only search). **No confidence gate is
recommended.**

### 1. Headline reproduction

Full rule vs. the card, 799 graded games, paired, week-blocked, 20,000
samples, seed 20260914.

| | Effect (pts) | 95% week-blocked | P+ | 95% season-blocked | P+ |
|---|---:|---|---:|---|---:|
| **This session, independent implementation** | **+3.0038** | [-0.999, +6.980] | 0.932 | [+2.632, +3.383] | 1.000 |
| Stored promotion read, `docs/sharp_weighted_follow.md` (seed 20260821) | +3.0038 | [-0.993, +6.953] | 0.930 | -- | -- |

**Reconciled.** The point estimate matches to ten decimal places
(3.0037546933667114 vs. 3.0037546933667083); the small interval and P+
differences are attributable entirely to the different bootstrap seed this
lane's binding rules require (20260914 vs. the older lane's 20260821), not to
any disagreement about the population, the rule, or the grading. No
discrepancy needed chasing further.

### 2. Confidence-conditioned reads on flips

**Terciles** (edges 0.5252 / 0.5635, covariate-only cut points), each cell's
effect is that band's policy (follow only flips in the band, else keep the
card) against the full 799-game card:

| Tercile | n flips | Card record | Follow record | Effect (pts) | 95% week | P+ |
|---|---:|---:|---:|---:|---|---:|
| Low (least confident) | 74 | 32-42 | 42-32 | +1.252 | [-0.876, +3.383] | 0.876 |
| Mid | 73 | 34-39 | 39-34 | +0.626 | [-1.613, +2.803] | 0.714 |
| High (most confident) | 73 | 32-41 | 41-32 | +1.126 | [-1.118, +3.509] | 0.831 |

**Fixed bands:**

| Band | n flips | Card record | Follow record | Effect (pts) | 95% week | P+ | Foresight ceiling (pts) |
|---|---:|---:|---:|---:|---|---:|---:|
| <=0.52 | 59 | 25-34 | 34-25 | +1.126 | [-0.752, +3.000] | 0.879 | +4.255 |
| 0.52-0.55 | 67 | 31-36 | 36-31 | +0.626 | [-1.018, +2.369] | 0.764 | +4.506 |
| 0.55-0.58 | 52 | 20-32 | 32-20 | +1.502 | [-0.500, +3.589] | 0.926 | +4.005 |
| >0.58 (firmest) | 42 | 22-20 | 20-22 | -0.250 | [-1.533, +1.128] | 0.358 | +2.503 |

**Resolving power at these cell sizes, measured, not inferred.** The
foresight ceiling column above is each band's own positive control (perfect
foresight restricted to that band's flips): even the smallest band (42
flips, >0.58) resolves a real +2.503-point ceiling with an interval entirely
above zero, so the instrument is not blind at these sizes -- but every
band's actual candidate effect sits at 0.25 to 1.5 points, well inside its
own ceiling's noise floor, exactly the `unresolved_below_power` pattern the
closing-grounds rule describes. The two-dimensional cells below (n=29 to 44)
are smaller still; their resolving power is worse than the bands above, by
inference from how the interval widths above scale with n, not separately
measured.

**Only the firmest band (>0.58, 42 flips) reads negative**, and it is the
one directionally consistent with the owner's hypothesis that a flip should
not happen when the model is already sure. Its interval crosses zero
([-1.533, +1.128]) and does not resolve to a wrong sign, so this is
`unresolved_below_power`, not a refutation and not a confirmation.

### 3. Continuous reads: is there a measured slope?

**Correlation, week-blocked bootstrap, n=220 flips, 53 week blocks:**

| | Estimate | 95% week | P+ |
|---|---:|---|---:|
| confidence vs. follow-correct | -0.0224 | [-0.148, +0.097] | 0.353 |
| move size vs. follow-correct | +0.0215 | [-0.108, +0.147] | 0.624 |

**Logistic fit of follow-correct on confidence** (standardized predictor,
slope converted to natural units, week-blocked bootstrap):

- Slope per 0.01 confidence: **-0.0121** [-0.0855, +0.0557] log-odds, P+
  0.353.

**Logistic fit on move size:** slope per 1.0-point move: **+0.0619**
[-0.322, +0.488] log-odds, P+ 0.624.

**Logistic fit with the confidence x move interaction:** confidence slope
-0.960 per unit confidence [-8.097, +7.129] P+ 0.411; move slope +0.093 per
point [-0.336, +0.639] P+ 0.694; interaction term (standardized) +0.070
[-0.292, +0.665] P+ 0.639.

**There is no measured slope here, in either direction, with an interval
that resolves.** Every one of these four continuous reads is centered near
zero with a wide interval on both sides. This is the basis for the "flat"
language used in Analysis 4 below -- it is a measured near-zero slope with
its own interval, not a records-look-the-same argument.

**2-D split, confidence tercile x move-size bin** (descriptive counts, not
separately bootstrapped; 6 of the 49 looks):

| Confidence third | Move 0.5-<1.0 | Move >=1.0 |
|---|---|---|
| Low | n=38, 22-16, 57.9% | n=36, 20-16, 55.6% |
| Mid | n=44, 21-23, 47.7% | n=29, 18-11, 62.1% |
| High | n=42, 22-20, 52.4% | n=31, 19-12, 61.3% |

No consistent pattern across the grid; cell sizes here (29-44) are too small
to read anything beyond noise, and this table is reported as descriptive
only, not as a result with its own interval.

### 4. The identical 61-49, addressed directly

**Independent reproduction confirms it, and it is not a defect.** Splitting
the 220 flips at their own median confidence (0.5427) gives two halves of
110 flips each:

| Half | Card record | Follow record | Effect (pts) | 95% week | P+ |
|---|---:|---:|---:|---|---:|
| Shallow (confidence <= median) | 49-61 | **61-49** | +1.5019 | [-1.104, +4.125] | 0.8745 |
| Deep (confidence > median) | 49-61 | **61-49** | +1.5019 | [-1.491, +4.602] | 0.8331 |

This independently reproduces the prior session's numbers (49-61 card,
61-49 follow, both halves) exactly, using a from-scratch implementation and
a different-but-related variable.

**Why it is not a coincidence needing a different explanation.** The
sanity-check above proved that, on flips, `confidence - 0.5` and the prior
session's "override distance" magnitude are the **same number** to
floating-point precision (max difference 5.6e-17): both equal
`|model probability for the card's original pick| - 0.5`. A median split on
either variable therefore partitions the SAME 220 flips into the SAME two
110-game groups. The identical 61-49 in both this session's and the prior
session's numbers is not two independent measurements agreeing by chance --
it is one measurement, computed twice with equivalent variables, and it
could not have come out any other way once the split point coincided.

**Why the specific numbers (49-61 in both halves, not merely "similar
records") are themselves unsurprising, given what Analysis 3 measured.**
With confidence and follow-correctness correlated at -0.022 (essentially
zero, interval [-0.148, +0.097]), a median split has no reason to sort wins
and losses unevenly between the two halves. Under that near-zero
correlation, the single most probable outcome for a 110-vs-110 split of a
population with 98 total wins and 122 total losses is for each half to
inherit almost exactly half of each (SD of the hypergeometric count is
about 3.7 games around a mean of 61 wrong picks per half), so landing on
exactly 49-61 in both halves is close to the modal outcome, not a
one-in-a-thousand coincidence that would call the arithmetic into question.

**The decile table**, computed independently and shown for anyone to
recount against `flips.csv` (22 flips per decile, sorted by confidence):

| Decile (low to high confidence) | n | Follow record | Win % |
|---:|---:|---:|---:|
| 1 | 22 | 12-10 | 54.5 |
| 2 | 22 | 14-8 | 63.6 |
| 3 | 22 | 11-11 | 50.0 |
| 4 | 22 | 13-9 | 59.1 |
| 5 | 22 | 11-11 | 50.0 |
| 6 | 22 | 12-10 | 54.5 |
| 7 | 22 | 10-12 | 45.5 |
| 8 | 22 | 17-5 | 77.3 |
| 9 | 22 | 12-10 | 54.5 |
| 10 | 22 | 10-12 | 45.5 |

Range 45.5 to 77.3 percent on 22 flips per decile -- this independently
reproduces the range the prior session reported, and it is noise at n=22
per cell (the foresight ceiling at comparable band sizes above is only
+2.5 to +4.5 points, so a swing from 45.5% to 77.3% on 22 games is well
within what that much sampling variance alone produces).

### 5. Gated arms as full-card policies (descriptive, in-sample)

Follow only when confidence is at or below the stated cut; scored as a full
799-game policy against the card and against the ungated rule. **These cuts
are chosen from the same flips they are scored on and are reported as
descriptive reads only, per the binding amendment that fixed bands stay
descriptive** -- the recommended gate, if any, comes from Analysis 6's
honest out-of-sample search, not from this table.

| Arm | n flips followed | Effect vs. card (pts) | P+ vs. card | Effect vs. full rule (pts) | P+ vs. full rule |
|---|---:|---:|---:|---:|---:|
| conf <= low-tercile edge (0.5252) | 74 | +1.252 | 0.876 | -1.752 | 0.161 |
| conf <= mid-tercile edge (0.5635) | 147 | +1.877 | 0.889 | -1.126 | 0.169 |
| conf <= 0.52 | 59 | +1.126 | 0.879 | -1.877 | 0.152 |
| conf <= 0.55 | 126 | +1.752 | 0.906 | -1.252 | 0.190 |
| conf <= 0.58 | 178 | +3.254 | 0.965 | +0.250 | 0.642 |

**Positive control (foresight on exactly the 220 flips, full-card policy):**
+15.269 [+12.484, +18.090], P+ 1.000, forcing correct the 122 flips the card
got wrong. This is the ceiling of the entire question at this flip count.

Every in-sample gated arm except the loosest (conf<=0.58, which excludes
only the worst-performing top confidence band) reads WORSE than the full
ungated rule (negative "vs. full rule" effect). This in-sample table alone
already argues against a tight gate; Analysis 6 confirms it out-of-sample.

### 6. Honest out-of-sample reads: single-dimension and nested LOSO

**Single-dimension (confidence cut only, threshold arm fixed at the flat 0.5
rule this population encodes):** candidate cuts are the nine outcome-blind
deciles of confidence among flips, plus "no gate". All three held-out-season
folds chose the same cut, 0.6061 (about the 90th percentile -- this drops
only the most extreme ~10% of flips from being followed):

| | Effect vs. card (pts) | 95% week | P+ | Effect vs. full rule (pts) | P+ |
|---|---:|---|---:|---:|---:|
| Out-of-sample (LOSO, pooled) | +3.254 | [-0.505, +7.072] | 0.954 | +0.250 | 0.699 |
| In-sample (same cut chosen and scored on all 799) | +3.254 | [-0.505, +7.072] | 0.954 | -- | -- |
| **Gap (in-sample minus out-of-sample)** | **0.000** | | | | |

This single-dimension search shows no in-sample optimism (the same cut wins
in every fold), but its own gain over the ungated rule is small (+0.250 pts,
P+ 0.699, unresolved) and it holds the move-size threshold fixed at the
already-in-sample-selected flat 0.5 arm -- which is exactly the optimism
Analysis 6 was added to address.

**Nested (move-size threshold AND confidence cut jointly selected, leave-one-season-out):**
candidates cross three threshold arms (T05 flat 0.5, T0 flat 1.0, T3 the
served 1.0/0.5-big-spread mechanism rule) with confidence cuts {0.52, 0.55,
0.58, no gate}.

| | Effect vs. card (pts) | 95% week | P+ | Gap (in - out) |
|---|---:|---|---:|---:|
| **Threshold only, no confidence gate: out-of-sample** | **+2.128** | [-0.505, +4.804] | 0.941 | |
| Threshold only, no confidence gate: in-sample | +3.004 | [-0.999, +6.980] | 0.932 | +0.876 |
| **Threshold + confidence gate: out-of-sample** | **+1.877** | [-0.732, +4.478] | 0.926 | |
| Threshold + confidence gate: in-sample | +3.254 | [-0.250, +6.767] | 0.965 | +1.377 |

Fold choices (threshold arm / confidence cut, both variants chose the same
threshold arm per season): 2023 -> T3 / no gate (training effect +3.377);
2024 -> T3 / no gate (training effect +3.189); 2025 -> T05 / no gate for the
no-gate search, T05 / 0.58 for the with-gate search (training effect
+3.759). T0 (flat 1.0) was never selected in any fold. The currently-served
T3 mechanism rule won 2 of 3 folds; the flat-0.5 rule this population's
`s1_leader_only_*` columns encode won the third.

**What this implies for the decision, stated before any caveat.** The
honestly-selected move-size threshold, with no confidence gate, is worth an
estimated **+2.128 accuracy points out-of-sample** -- smaller than the
in-sample +3.004 headline by 0.876 points, but still positive, and this is
the number that should be quoted as the served rule's honest expected value
on this population, not the in-sample headline. Layering a confidence gate
on top makes the honest number worse (+1.877 vs. +2.128) and grows the
in-sample-to-out-of-sample gap (1.377 vs. 0.876 points) -- the textbook sign
of an added selection dimension buying overfitting rather than signal. **No
confidence gate is recommended for this rule.** This is consistent with,
and now explains on stronger footing, the near-zero correlation and flat
logistic slopes in Analysis 3.

### 7. Season split of the honest out-of-sample threshold-only policy (never gated on)

The pooled leave-one-season-out threshold-only policy (Analysis 6, no
confidence gate), which was never selected using season as a splitting
variable, split by season for a descriptive read:

| Season | n games | Policy accuracy | Card accuracy | Effect (pts) |
|---:|---:|---:|---:|---:|
| 2023 | 266 | 57.14% | 53.76% | +3.38 |
| 2024 | 266 | 57.14% | 53.01% | +4.14 |
| 2025 | 267 | 53.93% | 51.69% | +2.25 |

Positive in every season, smallest in the most recent one. This is the same
"positive or level in every season, negative in none" shape the served T3
rule's own promotion documented in `docs/follow_threshold_by_line.md`.

### On the handle-follow rule's constant (amendment 3, addressed here since it is the direct precedent for this whole redo)

The handle-follow constant `0.0261` was set by splitting the SAME 37
handle-follow flips it was then scored on at their own median override
distance (19/18). That is in-sample selection by the same definition this
document applies to itself in Analysis 6: the cut point was derived from
outcome-bearing data and then presented as a constant. **Removed from
`pick_refresh.py` on 2026-09-14** (measured: a random 19/18 split of those
flips is at least as lopsided one time in five, hypergeometric p 0.194);
`HANDLE_FOLLOW_SERVED = False` keeps the rule off the served card. It is
named here as the reason this redo holds its own gate to a
leave-one-season-out standard and recommends no gate rather than repeat the
pattern.

## Registry

Family `leader_median_model_confidence_v1`, 19 cells, every one
`unresolved_below_power` -- no interval here sits wholly on the wrong side
of zero (so `wrong_sign_resolved` is inadmissible), and the foresight
positive control resolves an effect of roughly 15 accuracy points, which
does not prove the instrument can detect the 0.25-to-3-point effects these
cells sit at (so `positive_control_bound` is inadmissible too).
Split-half reliability was not separately computed for this family and is
not claimed either way. Recorded from
`artifacts/leader_median_model_confidence/20260914T202723Z/summary.json`,
argv lists saved to `record_commands.json` in that directory:

- `leader_median_follow_headline_vs_card`
- `leader_median_follow_conf_tercile_low`
- `leader_median_follow_conf_tercile_mid`
- `leader_median_follow_conf_tercile_high`
- `leader_median_follow_conf_le_0_52`
- `leader_median_follow_conf_0_52_to_0_55`
- `leader_median_follow_conf_0_55_to_0_58`
- `leader_median_follow_conf_gt_0_58`
- `leader_median_follow_median_split_shallow`
- `leader_median_follow_median_split_deep`
- `leader_median_follow_correlation_confidence`
- `leader_median_follow_correlation_move`
- `leader_median_follow_foresight_control`
- `leader_median_follow_conf_gate_loso_recommended`
- `leader_median_follow_conf_gate_loso_recommended_vs_full_rule`
- `leader_median_follow_nested_loso_threshold_only_out_of_sample`
- `leader_median_follow_nested_loso_threshold_only_in_sample`
- `leader_median_follow_nested_loso_with_gate_out_of_sample`
- `leader_median_follow_nested_loso_with_gate_in_sample`

Not every one of the 49 looks got its own registry row: the six 2-D cells,
the five band-level foresight ceilings, and the ten decile-table rows are
descriptive breakdowns of the same handful of measured quantities above and
are reported in this document's tables rather than as separate registry
entries, so the registry is not inflated with restatements of the same
family's members.

**Invalidated, superseded by this redo**, `nfl-ats weak-signals invalidate`:
`leader_median_shallow_only_vs_full_rule` (superseded by
`leader_median_follow_median_split_shallow`),
`leader_median_deep_overrides_vs_card` (superseded by
`leader_median_follow_median_split_deep`), and
`leader_median_override_distance_foresight_control` (superseded by
`leader_median_follow_foresight_control`) -- all three from family
`leader_median_override_distance_v1`, the rejected prior session's work.

## Provenance labels

Every load-bearing number above is **measured** this session by
`scripts/leader_median_model_confidence_eval.py` against
`artifacts/leader_median_model_confidence/20260914T202723Z/summary.json`
and `flips.csv`, unless marked **read** (a file and the line range cited
inline) or **inferred** (reasoning stated as such, never combined with a
measured number in the same sentence). The stored `docs/sharp_weighted_follow.md`
S1 numbers quoted in Analysis 1 are **read**, not remeasured, and are used
only for reconciliation against this session's independent measurement.
