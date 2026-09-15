# Does a combined model+market probability beat the hard follow rule?

## Why this is being measured

Paraphrased, owner directive 2026-09-14: a hard override rule ("the line
moved half a point against us, so flip the pick") is wrong on its face,
because a late-week market move is one more piece of evidence, not a veto
over the model. The decision should be a single combined number: the
model's own probability, updated by the late-week line move and whatever
else is situational, then the side that number favours is played. Under
this view, flips happen naturally where the model is already near a coin
flip and should not happen where the model is firm -- the hard rule cannot
express that because it fires on move size alone, blind to how confident the
model already was. The hard flip rule was taken off the served card the same
day (`LATE_WEEK_FOLLOW_SERVED = False` in `src/nfl_ats/pick_refresh.py`; it
keeps recording prospectively). This document builds and scores the combined
version honestly, leave-one-season-out, against the card and against the
hard rule it replaces.

## Binding closing grounds (verbatim, AGENTS.md)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with `nfl-ats
weak-signals record`, report `probability_positive`, never the binary
"contains zero". If a record command errors, the verdict is wrong, not the
validator. Also: failing p 0.90/0.95 is no reason to reject; a
`probability_positive` just above 0.5 is no reason to serve; state evidence
in games. Within-week correlation is zero, so the week-blocked bootstrap
(`nfl_ats.clv.week_blocked_bootstrap`, blocks `(season, week)`) is the
honest unit, 20,000 draws, seed 20260914.

## Population (read, then measured)

**Read**, `artifacts/sharp_weighted_follow/20260909T233606Z/per_game.parquet`:
816 rows, 2023-2025. **Measured**, this session: 799 rows have
`correct_at_open_probability_rule` non-null (17 opener pushes dropped), 266 /
266 / 267 games per season, 54 `(season, week)` blocks. This is the same
frozen population `docs/leader_median_model_confidence.md` and the original
`sharp_weighted_follow` promotion used; nothing in the archive itself is
refit here.

**Sign convention, read**, `src/nfl_ats/sharp_book_movement_features.py:138-144`
(`refresh_pick`) and confirmed in `docs/sharp_weighted_follow.md:94-95`: a
positive `leader_median_net` (`leader_median_net` column, the median
Wednesday-to-deadline net move across the three leader books) points HOME --
`net_move.gt(0)` picks home. So `leader_median_net` used directly is already
"move toward home" in the units the task calls for; no re-signing is needed.

**Measured, this session**: `home_cover_probability_at_open` (`p_home`)
ranges 0.3182 to 0.6845 over the 799 graded games; `pick_home_at_open_probability_rule`
equals `p_home >= 0.5` with 0 mismatches (reproducing the same identity
`docs/leader_median_model_confidence.md` found), so the card's pick is
always the model's own argmax. The true home-cover outcome is therefore
recoverable without new data: `home_covered = card_correct` where the card
picked home, else `1 - card_correct` (card_correct is `correct_at_open_probability_rule`);
measured mean 0.5081 across 799 games, values are a clean `{0, 1}` (no
pushes remain, since pushes were already dropped by the `correct_at_open_probability_rule.notna()`
filter).

## Predeclared definitions, arms, metric and decision rule

Written before any of the combined-model numbers below were computed.

- **C0 (card / model only).** `pick_home_at_open_probability_rule`,
  `correct_at_open_probability_rule` -- unchanged, the baseline every arm is
  paired against.
- **C1 (hard flip rule as served until today).** The `s1_leader_only_*`
  columns already in the population: fires when `abs(leader_median_net) >=
  0.5`, flips to the side the move favours. This is the flat-0.5 arm the
  population's own `s1_leader_only_*` columns encode; it is scored here for
  reference exactly as-is, not re-fit.
- **C2 (combined).** `logit(p_home_combined) = intercept + a * logit(p_home)
  + b * leader_median_net`, three coefficients. Fit by leave-one-season-out
  (LOSO) logistic regression on the true `home_covered` outcome: for each
  held-out season, `intercept, a, b` are fit by IRLS (with a small L2
  penalty for numerical stability, standardized predictors internally, then
  converted back to natural units) on the other two seasons only, and
  applied to the held-out season. Pick home iff `p_home_combined >= 0.5`.
  The optional third term the task allows (`c * move * |spread| band` or `c
  * books`) is deliberately **not** added to the headline arm: the
  overfitting discipline caps the model at 2-4 coefficients specifically to
  keep it from memorising 799 games, and 3 (intercept, a, b) already
  satisfies that band with the fewest parameters that can express "blend the
  model and the move." This choice is stated here, before any coefficient
  was fit.
- **C3 (thresholded move).** Identical to C2 except the move term is
  `leader_median_net` when `abs(leader_median_net) >= 0.5`, else `0.0` --
  same 3 coefficients, same LOSO procedure. Tests whether the served rule's
  own move-size gate adds anything once the model's own logit is already in
  the equation.
- **Positive control.** Perfect foresight restricted to exactly the games
  where C2's out-of-sample pick differs from C0's: force the card correct
  wherever the card was wrong on one of those games, otherwise leave the
  card's own grade. This bounds what any rule agreeing with C2's set of
  differences from the card could possibly buy, the same construction
  `docs/leader_median_model_confidence.md` used for its own foresight
  control.
- **In-sample companion.** For C2 and C3, the same 3-coefficient fit is also
  run once on all 799 games and scored on those same 799 games, reported
  beside the out-of-sample number with the gap (in-sample minus
  out-of-sample), the same overfitting-discipline check the sibling
  document used.
- **Metric.** Paired accuracy-point deltas (candidate mean correct minus
  baseline mean correct, times 100), week-blocked bootstrap, 20,000 samples,
  seed 20260914. `probability_positive` reported for every cell; a binary
  "contains zero" read is never reported. Records in games (wins-losses) are
  reported before any effect size, per the overfitting-discipline
  instruction.
- **Head-to-heads scored.** C1 vs C0 (reference); C2 out-of-sample vs C0; C2
  out-of-sample vs C1; C3 out-of-sample vs C0; C3 out-of-sample vs C1; C3
  out-of-sample vs C2 out-of-sample (does the threshold add anything once
  the model is in the equation -- the reason C3 exists); C2 and C3 in-sample
  vs C0 (for the gap only, not a registry cell).
- **Differ-from-card counts.** For C1, C2 (out-of-sample) and C3
  (out-of-sample): the count of the 799 games where that arm's pick differs
  from C0's, and the won-loss record on exactly those differing games for
  both the card and the arm.
- **Calibration.** C2's pooled out-of-sample `p_home_combined`, binned into
  5 fixed-width bins (0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0): n games,
  mean predicted probability, actual home-cover rate, and the gap between
  them, per bin.
- **Confidence-band behaviour.** Model confidence `= max(p_home, 1 -
  p_home)`, the same four fixed, covariate-only bands
  `docs/leader_median_model_confidence.md` already fixed from the sibling
  population (`<=0.52`, `0.52-0.55`, `0.55-0.58`, `>0.58`) -- reused rather
  than re-derived, since re-deriving cut points from this analysis's own
  outcomes would violate the no-cut-on-scored-data rule. For each band:
  games in the band, how many of C2's out-of-sample picks differ from C0's
  in that band, and the effect of applying C2 only inside that band (else
  keep the card) scored as a full 799-game policy against the card. The
  owner's stated expectation is that flips concentrate in the near-50/50
  band; this table is where that gets checked, not assumed.
- **Season split.** C0, C1, C2 (out-of-sample) and C3 (out-of-sample)
  accuracy by season, descriptive, not separately bootstrapped.
- **Look count.** Every band, head-to-head, LOSO fit, calibration bin table
  and season split counted by the eval script and reported in Results before
  any single cell is discussed.
- **Decision rule.** The pool is forced picks. A record like 12-7 or a
  `probability_positive` just above 0.5 is not by itself grounds to serve;
  research closure and card-serving are different decisions. What the
  out-of-sample numbers imply for the decision is stated before any caveat
  about them. Nothing in `src/` is touched by this document; C2/C3 are
  challengers scored here, not switched on.

## Results

**Measured**, `scripts/market_updated_model_eval.py`,
`artifacts/market_updated_model/20260914T210317Z/summary.json` and
`per_game.csv`, 799 graded games, 54 week blocks, seed 20260914, 20,000
draws. **19 looks taken**, logged in `summary.json["look_log"]`: 4 LOSO/
in-sample fits, 8 head-to-head bootstrap comparisons, 1 positive control, 1
five-bin calibration table, 4 confidence-band policy reads, 1 descriptive
season split.

### The decision-relevant fact, stated first

**The combined model (C2) beats the card out of season, but not by as much
as the hard rule it is meant to replace, and the evidence that it trails the
hard rule is fairly one-sided even though the interval still touches zero.**
Out-of-season: C2 vs card **+2.003 accuracy points**, week-blocked 95%
[-2.101, +6.045], `probability_positive` 0.838; C2 vs the hard rule (C1)
**-1.001 accuracy points**, 95% [-2.390, +0.373], `probability_positive`
0.075 (i.e. about 92.5% odds C1 was the better arm on this pooled read). The
interval on the C2-vs-C1 read still touches zero at the extreme (+0.373), so
this does **not** resolve to a wrong sign under the closing-ground rule -- it
is `unresolved_below_power`, not a refutation of the combined approach -- but
it is a materially more one-sided reading than C2-vs-card. **C3 (the
move-gated variant) is bit-for-bit identical to C2** on this population: see
Result 3 below for why, which is a measured structural fact about the
`leader_median_net` variable, not a defect in the fit.

### 1. Records in games, before any effect size

| Arm | Record (799 games) | Win % |
|---|---|---:|
| C0 (card) | 422-377 | 52.82% |
| C1 (hard rule, s1 arm) | 446-353 | 55.82% |
| C2 out-of-sample (combined) | 438-361 | 54.82% |
| C3 out-of-sample (thresholded combined) | 438-361 | 54.82% |

**On the 220 games where an arm's pick differs from the card** (C1's flip
set and C2's/C3's differ-set are each exactly 220 games, though not the
identical 220 -- see Result 3):

| Arm | Card record on its diff set | Arm record on its diff set |
|---|---|---|
| C1 vs C0 | 98-122 | **122-98** |
| C2 out-of-sample vs C0 | 102-118 | **118-102** |
| C3 out-of-sample vs C0 | 102-118 | **118-102** |

### 2. Head-to-head effects, out-of-sample beside in-sample

| Comparison | Effect (pts) | 95% week-blocked | P+ |
|---|---:|---|---:|
| C1 vs C0 (reference) | +3.0038 | [-0.999, +6.980] | 0.9323 |
| **C2 out-of-sample vs C0** | **+2.0025** | [-2.101, +6.045] | 0.8381 |
| C2 in-sample vs C0 | +3.7547 | [-0.248, +7.771] | 0.9666 |
| **Gap (in - out), C2** | **+1.752** | | |
| **C2 out-of-sample vs C1** | **-1.0013** | [-2.390, +0.373] | 0.0750 |
| C3 out-of-sample vs C0 | +2.0025 | [-2.101, +6.045] | 0.8381 |
| C3 in-sample vs C0 | +3.7547 | [-0.248, +7.771] | 0.9666 |
| Gap (in - out), C3 | +1.752 | | |
| C3 out-of-sample vs C1 | -1.0013 | [-2.390, +0.373] | 0.0750 |
| **C3 out-of-sample vs C2 out-of-sample** | **0.0000** | [0.000, 0.000] | 0.5000 |

C1's own reproduction here (+3.0038, P+ 0.9323) matches
`docs/leader_median_model_confidence.md`'s independent reproduction of the
same flat-0.5 arm to ten decimal places, on the same population and seed --
cross-check passes.

### 3. Why C3 is identical to C2 (measured, not a bug)

C3 was predeclared to test whether the served rule's own move-size gate adds
anything once the model's own logit is in the equation. **Measured**: on
this population, `leader_median_net` takes the value 0 on 284 games and is
never smaller than 0.5 in absolute value when it is nonzero -- the full
value set is `{0, ±0.5, ±1.0, ±1.5, ±2.0, ±2.5, ±3.0, ±4.0, ±4.5, ±6.0}`,
because it is the median of three books' own net moves, which themselves
move in half-point spread increments. Gating at `abs(move) >= 0.5` therefore
changes zero rows: `move_thresholded == move` on all 799 games (measured, 0
differing rows). So C3's fold coefficients, predictions, records and every
bootstrap comparison are bit-for-bit identical to C2's -- the answer to "does
the threshold add anything" is a clean, structural **no** on this
population, not a close call. This differs from the C1/T3 threshold
question in `docs/leader_median_model_confidence.md` (which compares
different FLAT cut levels, e.g. 0.5 vs 1.0, on top of a hard binary rule);
here the threshold is gating a continuous term already sitting on a coarser
grid than the gate itself.

### 4. C2 introduces its own new disagreements with the card

**Measured**, comparing the 220-game C1 flip set against the 220-game C2
out-of-sample diff set: 205 games overlap, 15 are C1-only (the market moved
and the hard rule flips, but C2's blended logit still favours the card's
original side), and 15 are C2-only -- **new** flips on games where
`leader_median_net` was exactly 0 (no market signal at all). Those 15
C2-only games all sit at `home_cover_probability_at_open` between 0.4979 and
0.5234 (mean model confidence 0.512, i.e. essentially a coin flip) -- the
tiny fitted intercept (-0.0385 to +0.0028 across folds) is enough to tip the
combined pick only when the model itself was already almost exactly
undecided. This is the mechanism the owner's directive predicted: the
combined read changes picks near 50/50, using the model's own calibration
rather than only market movement, and mostly (205 of 220) still agrees with
the hard rule where the market did move.

### 5. Positive control

Perfect foresight restricted to the 220 games where C2's out-of-sample pick
differs from the card's (the same construction
`docs/leader_median_model_confidence.md` used): card record on those 220
games 102-118, forced to 220-0. **Effect vs card: +14.768 accuracy points,
95% week-blocked [+12.219, +17.370], P+ 1.0.** This bounds the ceiling of
any rule that agrees with C2's particular set of 220 disagreements -- the
actual measured C2 gain (+2.003) sits well inside this ceiling, the same
`unresolved_below_power` pattern (real resolving power exists at this game
count, the candidate effect itself just does not clear it) as the sibling
document's own positive control.

### 6. Calibration of C2's out-of-sample probability (5 bins)

| Predicted p(home) bin | n | Mean predicted p | Actual home-cover rate | Gap |
|---|---:|---:|---:|---:|
| 0.0-0.2 | 0 | -- | -- | -- |
| 0.2-0.4 | 25 | 0.3475 | 0.5600 | +0.2125 |
| 0.4-0.6 | 733 | 0.5047 | 0.4980 | -0.0068 |
| 0.6-0.8 | 41 | 0.6418 | 0.6585 | +0.0167 |
| 0.8-1.0 | 0 | -- | -- | -- |

The fitted `a` and `b` coefficients are small enough (see Result 8) that
C2's predicted probability almost never leaves the 0.4-0.6 band (733 of 799
games); that band is well calibrated (-0.0068 gap). The 0.2-0.4 bin (n=25,
the games with the largest away-leaning combined read) shows a +0.21 gap --
home covered more than predicted there -- but at n=25 this is a thin read,
reported descriptively and not bootstrapped separately.

### 7. Confidence-band behaviour (owner's expectation: flips concentrate near 50/50)

Same four fixed, covariate-only bands `docs/leader_median_model_confidence.md`
already fixed. "Effect" is the policy of applying C2's out-of-sample pick
only inside that band (else keep the card), scored as a full 799-game policy
against the card.

| Confidence band | n games | n where C2 differs from card | Share differing | Effect of C2-in-band vs card (pts) | 95% week | P+ |
|---|---:|---:|---:|---:|---|---:|
| <=0.52 (least confident) | 179 | 73 | 40.8% | +0.626 | [-1.635, +2.792] | 0.718 |
| 0.52-0.55 | 251 | 68 | 27.1% | +0.501 | [-1.245, +2.267] | 0.710 |
| 0.55-0.58 | 180 | 49 | 27.2% | +1.126 | [-0.748, +3.086] | 0.875 |
| >0.58 (firmest) | 189 | 30 | 15.9% | -0.250 | [-1.358, +0.878] | 0.328 |

**The share of games where C2 disagrees with the card falls monotonically
as model confidence rises** -- 40.8% in the least-confident band down to
15.9% in the firmest -- which is the shape the owner's directive predicted.
The effect sizes do not fall as cleanly monotonically (0.55-0.58 reads
highest), but the **firmest band is the only one that reads negative**
(-0.250, P+ 0.328, `unresolved_below_power`, not a resolved wrong sign).
This is descriptively consistent with, though it does not prove, "don't
override a firm model."

### 8. Fitted coefficients, per fold and in-sample

C2 (and C3, identical): `logit(p_home_combined) = intercept + a * logit(p_home) + b * move`.

| Fold (held out) | n train | intercept | a (model logit) | b (per point of move) |
|---|---:|---:|---:|---:|
| 2023 | 533 | -0.0385 | 0.3301 | 0.2359 |
| 2024 | 533 | +0.0028 | 0.1531 | 0.2551 |
| 2025 | 532 | -0.0085 | 0.2833 | 0.1670 |
| In-sample (all 799) | 799 | -0.0126 | 0.2520 | 0.2189 |

`a` and `b` are both **positive in every fold** (the model's own logit and
the market move both push the combined read the expected direction in every
season), and both stay inside a roughly 2x band across folds (a: 0.153 to
0.330; b: 0.167 to 0.255) -- reasonably stable for a 3-coefficient fit on
~533-game training folds, not collapsing to zero or flipping sign in any
fold. `a` is consistently below 1.0, meaning the fit shrinks the model's own
raw logit toward 0.5 once the market term is added, rather than passing it
through unchanged.

### 9. Season split (descriptive, not separately bootstrapped)

| Season | n | C0 | C1 | C2 out-of-sample |
|---:|---:|---:|---:|---:|
| 2023 | 266 | 53.76% | 56.39% | **52.26%** |
| 2024 | 266 | 53.01% | 56.39% | 56.02% |
| 2025 | 267 | 51.69% | 54.68% | 56.18% |

**C2 reads below the card in 2023** (the fold trained on 2024+2025 only),
positive in 2024 and 2025. C1 (the hard rule) is at or above the card in
every season. This is the clearest single piece of evidence behind the
C2-vs-C1 head-to-head reading in Result 2 -- the combined model's
out-of-sample shortfall against the hard rule is concentrated in one
season, not spread evenly, which is the kind of detail a pooled 3-block
week-level bootstrap already reflects in its wide interval, not something
this descriptive table adds new uncertainty to.

### What this implies for the decision, stated before caveats

C2 is a genuine improvement over having no market-move rule at all (+2.00
pts vs card, P+ 0.838, and it changes picks in the theoretically right place
-- more often near 50/50, less often when the model is firm). It is **not**
shown to be an improvement over the hard rule it was built to replace on
this pooled read (-1.00 pt vs C1, P+ 0.075) -- the evidence leans toward C1
still being better out-of-sample, though the interval still grazes zero so
this is not a resolved wrong sign. Per the promotion-bar rule, neither a
`probability_positive` of 0.838 (for C2 vs card) nor one of 0.075 (for C2 vs
C1, read as ~0.925 in C1's favour) is by itself a serving decision --
research closure and card-serving are different questions. What this result
does settle is the mechanism question the owner asked: a combined
model+market probability **can** be built, is stable across LOSO folds
(same-sign `a`, `b` in every fold), and does concentrate its picks-that-
differ-from-the-card in the low-confidence band as predicted -- but on this
799-game population it has not been shown to beat the simpler hard rule it
would replace. The one-season concentration of C2's shortfall (Result 9)
is the detail worth a follow-up look before any serving decision, not a
reason to reject the combined-model approach itself: per the closing-ground
rule, an interval touching zero is never grounds to close this line of
work.

### Caveats

- The optional third coefficient the task allowed (interaction with spread
  band, or books) was not fit at all, per the predeclaration; it remains an
  open extension, not a result here.
- C1 in this document is the flat-0.5 `s1_leader_only` arm actually stored
  in the population, per the task's own naming ("C1 hard flip rule as
  served until today (s1 arm)"). The rule actually served in production
  before today used the mechanism threshold T3 (1.0 baseline, 0.5 on
  spreads >= 10.5), not flat 0.5. `docs/leader_median_model_confidence.md`
  measured T3's own honest leave-one-season-out read at +2.128 pts vs the
  card; C1 here is scored once, as coded, with no LOSO applied to it (it is
  a fixed reference rule, not a fit), so its +3.004 vs card is not directly
  comparable to that +2.128 -- both are reported as read/measured facts
  answering different questions, not reconciled into one number.
- The 2023 season split issue in Result 9 is descriptive; it was not used
  to select or gate anything in this document, and no cut point here was
  chosen using season as a splitting variable.

## Registry

Family `market_updated_model_v1`, 7 cells recorded via `nfl-ats weak-signals
record`, category `market`, units `accuracy_points`, league `nfl`,
season-start 2023, season-end 2025, source
`artifacts/market_updated_model/20260914T210317Z/summary.json`. Every cell
classified `unresolved_below_power`: no interval here sits wholly on the
wrong side of zero (`wrong_sign_resolved` inadmissible for all seven,
including C2-vs-C1 whose interval still touches +0.373), and the foresight
positive control resolves an effect of roughly 14.8 accuracy points on its
own 220-game subset, which does not prove the instrument can detect the
0-3-point effects most of these cells sit at (`positive_control_bound`
inadmissible too, same reasoning the sibling document used for its own
foresight control).

- `market_updated_model__c1_vs_c0`
- `market_updated_model__c2_oos_vs_c0`
- `market_updated_model__c2_oos_vs_c1`
- `market_updated_model__c3_oos_vs_c0`
- `market_updated_model__c3_oos_vs_c1`
- `market_updated_model__c3_oos_vs_c2_oos`
- `market_updated_model__positive_control_c2_diff_foresight`

Exact argv used for each `record` call is listed inline in the session
report; the same seven reads are also in
`artifacts/market_updated_model/20260914T210317Z/summary.json`.

## Provenance labels

Every load-bearing number in Results is **measured** this session by
`scripts/market_updated_model_eval.py` against the artifact stamped below,
unless marked **read** (file and line cited inline) or **inferred**
(reasoning stated as such, never combined with a measured number in the same
sentence).
