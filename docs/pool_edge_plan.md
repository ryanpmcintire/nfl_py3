# Pool edge plan: ceiling, gap accounting, and the build queue

Written 2026-08-17 at the end of the session that reframed the project
around its clarified **primary goal: beat the OPENING line the user's
Splash Sports pool grades against** (closing lines, CLV, vig are
secondary). This document captures the strategy analysis behind the
current queue so a new session can pick it up without re-deriving it.

## Where we stand (all measured, all committed)

- **52.50% against Tuesday openers** — the frozen active model, one
  predeclared look on 1,537 paired 2020–2025 games
  (`docs/opener_evaluation.md`). P(genuine skill vs coin flip) ~97–99%;
  season-blocked interval excludes 50%. Positive five of six seasons
  (the miss is COVID-2020).
- 51.09% against closes on the same games; the paired +1.35-point
  opener-vs-close delta has ~99.9% probability positive — the market
  drifting toward our number all week is real settlement value.
- **Movement oracle: 55.1%** — perfectly foreseeing Wednesday–Sunday line
  movement, with zero football knowledge, scores 55.1% against the frozen
  opener. This is the measured value of the midweek-information channel.
- Splash pool mechanics (researched, sourced): their legacy engine posts
  lines Tuesday, revises once Wednesday, then freezes for the week —
  half-point numbers, identical for all entrants. Commissioners can
  override lines; the native app's cadence is unconfirmed.

## The ceiling, and why

NFL margins scatter around the best possible pregame expectation with
σ ≈ 13.1 points (turnover bounces, in-game injuries, one-score-game coin
flips). Exchange rate: **1 point of true line error ≈ 3 points of ATS
accuracy** (Φ(1/13.1) ≈ 0.53).

> **σ corrected 13.5 → 13.1 on 2026-08-17** (measured: sd of the ATS
> residual is 13.130 over 4,431 completed 2009–2025 regular-season games;
> 12.78 for 2018–2025, 12.83 against the Tuesday opener). The exchange rate
> is unchanged to two figures, so nothing downstream moves. Note this is
> scatter around the *market* line; around a genuinely perfect expectation
> it can only be smaller, so 13.1 is a mild over-estimate and the ceilings
> below are, if anything, conservative.
>
> **The exchange rate was also checked for key-number curvature, and it
> survives — for the quantity this section is about.** The intuition that a
> point should be worth much more near 3 is correct for *shopping the
> settlement line* and wrong for *improving our own number*. Moving our
> centre reweights the whole key-number lattice rather than sliding a spike
> across a threshold, so that derivative is a covariance and stays flat:
> 2.97–3.29 ATS points per point across every line position. Moving the
> *settlement* line is the lumpy one — the last half-point is worth **6.66**
> points per point, and its value ranges from 0.84× the Gaussian rate in the
> dead zones (|line| < 2, and 8–10.5) to **2.67× at a line of 3**. Use the
> flat ≈3 for "is our model better"; use the curve for "does it matter that
> the pool posted a different number".

- Omniscient-pregame oracle vs the close (~1.5–2 pts RMS market error):
  **~55–56% ceiling**. Matches the best documented career bettors.
- Same oracle vs a frozen Tuesday pool line: **~57–58%** (adds the
  midweek channel; the movement oracle's 55.1% is one measured floor).
- Practical excellence band for us: **54–55% vs the frozen opener**.
- Honesty guardrail: any backtest showing 60%+ is a leak, not a
  breakthrough — the sport's noise floor does not permit it.

The ceiling is adversarial, not physical: it measures how much better we
can be than the entity that set the line. The pool's line-setter is
deliberately handicapped (frozen midweek), which is the entire strategic
opportunity.

## The pool's actual format (confirmed by the user, 2026-08-17 evening)

- Pick a side **against the spread for all 272 regular-season games and
  all 13 playoff games** (forced picks, no passes — exactly the
  forced-pick metric this project evaluates).
- One **"Best Pick"** (formerly "Key Pick") per regular-season week — a
  confidence-weighted selection.
- **Picks lock Tuesday at 12** — essentially at the opener.

Two immediate consequences, both measured the same evening:

- **The midweek channel is NOT harvestable for this pool.** Picks are
  due before Wednesday–Friday injury designations exist, so neither
  late-picking nor an event-aware close model can add pool points. The
  event-aware close model is demoted to the secondary goals (CLV,
  close prediction). The 52.50% opener grade is, almost exactly, the
  pool-relevant baseline, and improving the model AT the opener is the
  only path.
- **The Best Pick lever is currently unexploited: our confidence
  ordering is flat.** From the opener-evaluation artifact (read-only
  reporting, no selection): the weekly top-|residual| pick scored 48.6%
  over 107 weeks, and accuracy is non-monotone across confidence
  quartiles (53.2%, 47.3%, 55.7%, 53.7%). The model's *sign* carries
  the signal; its residual *magnitude* does not rank pick quality.
  Choosing any pick as Best Pick costs nothing today (all ≈ 52.5%), but
  a working confidence ranker is free pool points if one exists —
  candidates: calibrated cover probability instead of raw residual,
  key-number geometry, regime-aware calibration (MOD-11's open half).

Also discovered: **the feature pipeline contains zero playoff games**
(`game_features*.parquet` is `game_type == REG` only), while the pool
requires 13 playoff picks. Extending features/predictions to the
postseason is now a required work item before January.

## The standing lesson: measuring teams better is bounded near zero

Added 2026-08-17, after a screen that finally explained a long trail of
negative results rather than adding another one.

Our target is the **residual from the market line**. The market already
prices team quality — that is the one thing it is unambiguously good at.
So any feature whose contribution is "we now know how strong these teams
are, more precisely" is refining a quantity the spread has already
accounted for, and its achievable gain is bounded near zero however good
the measurement becomes.

This was measured, not argued. The CFB opponent-adjustment screen
(`docs/cfb_opponent_adjustment.md`) ran a **deliberate leak as a positive
control**: fit the adjustment once over all of 2006–2025 so the columns can
see the future. Perfect foreknowledge of team quality moved margin MAE by
**+0.0129 points** (`probability_positive` 0.984 — the instrument detects a
leak that small, so the honest null beside it is measured, not
underpowered). That figure is a **ceiling on the entire family**.

It retroactively explains the raw-PBP/drive bundle (−0.08 points), PBP-05
in both its additive and dimension-neutral forms, MOD-16's variance model,
and CFB role continuity. None of them failed for want of craft. They were
all measuring the same already-priced quantity.

**Use it as a filter.** Before building a feature, ask which of these it is:

1. *Measures team quality more precisely* → bounded near zero. Do not
   build without an argument for why this instance escapes the ceiling.
2. *Prices something the market prices badly* → this is where edge lives.
   Availability is the only candidate with a measured lean so far
   (`probability_positive` 0.899 in the MOD-07 ablation).
3. *Exploits the pool's format rather than the line* → a different
   objective entirely (Best Pick selection, pick popularity, contest
   utility), and largely unexplored.

A corollary from MOD-06, same date: any method whose whole effect is to
**rescale** the prediction — shrinkage, regularization, recalibration —
cannot change a forced pick, because the pick is `sign(predicted
residual)` and a positive scalar never changes a sign. Such methods can
improve calibration and confidence ordering, which matters for the Best
Pick, but they cannot move the headline accuracy.

## Gap accounting: 52.5% → ~57% (revised for the Tuesday lock)

1. **Midweek information channel, ~2.6 points — closed for the pool.**
   The 55.1% movement oracle needs post-Tuesday information; the pool
   locks Tuesday noon. What remains capturable at pick time is only
   Monday-night results and Monday/Tuesday-morning news the pool's own
   Tuesday line hasn't priced — a thin slice, largely already inside
   our features. (The channel remains fully relevant to the secondary
   goals and to any future contest with later locks.)
2. **Fundamental edge vs the close, ~3–4 points of space, hardest.**
   Concede ~2 (film-level and human-aggregation information we cannot
   ingest). Reachable slice ~0.5–1: hierarchical shrinkage pooling where
   our estimates are thin (backup-QB value from few snaps + position
   prior; early-season team states; CFB→NFL as a league-effect pooled
   prior — the XLG-05 "partially pooled" arm), QB-dependence interaction
   (team output conditioned on QB reliance), and the peer-reviewed
   opener biases (Week-1 playoff-holdover fade — holdovers covered
   35.6%; Week-2 anchoring; prior-week recency; low-visibility games
   move most).

   > **Both halves of this item were measured on 2026-08-17 and both need
   > correcting.**
   >
   > **Shrinkage: only the unit-level arm survives.** Coefficient-level
   > pooling is closed (MOD-06, 12,206 free CFB games). The reason is
   > structural and applies to anything proposed here in future: the pick
   > is `sign(predicted residual)`, and rescaling by a positive scalar
   > cannot change a sign, so a method whose whole effect is "be more
   > conservative" cannot move a single pick. What remains live is
   > shrinking a thin estimate toward a *position prior* instead of toward
   > zero — that changes relative values and can flip a pick.
   >
   > **The opener biases do not survive contact with our own data.** Three
   > of the four were already built and went into the MOD-07 stack. An
   > ablation on the already-spent window (free — attribution on data
   > already looked at costs no window) shows they contributed
   > **+0.22 points, probability_positive 0.505** — a coin flip. The
   > player-value/availability half carried the whole +1.97 (P+ 0.899).
   > The headline holdover figure also fails to replicate: 35.6% published
   > against **52.5% measured** on 120 Week-1 holdover favourites here
   > (season-blocked diff vs plain favourites −3.6 points, [−14.3, +6.6]).
   > Week-2 anchoring came out directionally *opposite* the hypothesis.
   > Low-visibility remains unbuilt and its two feasible proxies (book
   > count at the opener, standalone-window flag) were both null on the
   > only non-reserved data available. **Do not add more bias features.**
   > The availability thread is where the measured signal actually is.
3. **Estimation noise, ~0.5–1 point.** Coefficient noise from ~4,500
   training games; recovered by shrinkage and by stacking weak signals
   instead of discarding them (MOD-07), never by synthetic rows.
4. **Unreachable remainder.** Private information and oracle perfection.

## Three kinds of negative, and only two of them are real

Added 2026-08-17 after a "fully priced, drop it" verdict turned out to be an
underpowered interval read as a null — the RWB-16 error, committed again by
the people who wrote RWB-16. **Before recording any negative, state which of
these it is.**

1. **Refuted mechanism.** The effect has the wrong sign, or the thing being
   measured is not a stable property at all. Examples: game play volume
   correlates *negatively* with margin size (−0.20; blowouts kill clock), a
   team's play-EPA dispersion has split-half reliability 0.014, coach ATS
   reputation has 0.063. Nothing to forecast, so no amount of sample would
   help. **A sound close.**
2. **Bounded by a positive control.** We proved the instrument could detect an
   effect of the hypothesised size, and it did not. The gold standard here is
   the opponent-adjustment screen, which deliberately leaked the future and
   showed the whole family is worth ≤ +0.0129 points. **The strongest close
   available, and the one worth the extra effort.**
3. **Unresolved below detection power.** The confidence interval contains both
   zero *and* the hypothesised effect. This is **not a negative** and must not
   be recorded as one. The evaluator's demonstrated power is ~2 ATS points
   (RWB-15); most candidate features are worth a fraction of that, so this is
   the *default* outcome for a small signal and says nothing about it.

Worked example: 4th-down aggressiveness. The market test gives −0.038 points
for a one-sd matchup, 95% [−0.423, +0.417]. The completely-unpriced hypothesis
is +0.174 — inside the interval. Resolving it would need ~24,000 games, about
90 NFL seasons. It can never be confirmed standalone, so it belongs in the
weak-signal stacker or nowhere, and it must never be spent on a window.

The same discount applies to the MOD-07 ablation's finding that the three
opener-bias features contributed +0.22 points at `probability_positive` 0.505
on 456 games: that interval, too, cannot separate "nothing" from "a little".
What actually closes the holdover bias is the independent **replication
failure** (published 35.6% against 52.5% measured here on 120 games), which is
category 1, not the ablation.

**The practical rule:** a category-3 result means *build it into the stacker
and stop testing it individually*, not *delete it*. Only categories 1 and 2
justify closing a line of work.

### Where category-3 signals now go, and why keeping them pays

Category 3 results are recorded in `registry/weak_signals.json` (git-tracked,
schema-validated) through `nfl_ats.weak_signals`, and inspected with
`nfl-ats weak-signals status` / `nfl-ats weak-signals pool`. Nothing is
deleted. Two things accumulate there that a single experiment cannot buy:

**Directions accumulate faster than precision.** Under a true null, a point
estimate is equally likely to land either side of zero, so the *sign* of each
result is one clean bit of evidence regardless of how wide its interval is.
Ten of twelve independent candidates leaning the same way is a binomial event
with p ≈ 0.039 — a real finding assembled entirely out of individually
worthless results. That is `sign_test`, and it is the cheapest thing in this
document. Conversely, if the signs come out a coin flip, that is strong
evidence the discards genuinely had nothing in them, and we should stop.

**Pooling shrinks the standard error as √K.** Inverse-variance pooling of K
signals sharpens the estimate by √K. The honest arithmetic, pinned in
`tests/test_weak_signals.py`: a signal at 0.5σ needs **√K ≥ 3.92, so about
sixteen** independent companions before the pool crosses 1.96σ. Four is not
enough — it reaches 1.0σ, and believing otherwise is precisely the error this
section exists to prevent. Random effects is the default so that
between-signal disagreement inflates the variance rather than being assumed
away.

Three guards, because this machinery could otherwise manufacture findings:

- **Only category 3 is poolable.** A refuted mechanism and a control-bounded
  null are real negatives; folding either in would launder a known failure.
- **Shared seasons are reported.** Results measured on the same football have
  correlated errors, so pooling them overstates precision — the "pooling ten
  weak positives on the SAME window proves nothing" trap. `overlap_warnings`
  surfaces every overlapping pair instead of hiding it.
- **A pooled estimate is not a finding.** It is grounds for building ONE
  combined candidate and confirming it, predeclared, on a rotation window none
  of the inputs touched. The registry records `seasons_touched_by_inputs` so
  that window can be chosen honestly.

## Methodology agreements from this session (binding style, not just taste)

- Report **continuous evidence**: `probability_positive` (fraction of
  blocked resamples favoring the candidate) ships in
  `paired_feature_comparisons` and `week_blocked_bootstrap`. Never
  collapse "unresolved at this sample size" into "false".
- **Rotation registry** (to build): each new candidate family draws a
  logged confirmation window from 2009–2025 it has never touched;
  training always strictly prior (forward-chaining, never random
  k-fold); windows retire per-family, not globally. Iterate freely on
  CFB and non-reserved seasons.
- **Stack weak signals** (MOD-07): combine surviving small positives
  (learned availability ~61% P(positive), value-weighted injuries,
  bias features, contract-year and friction-event features) into one
  candidate judged once at the opener on a rotated window. Pooling ten
  weak positives on the SAME window proves nothing (shared noise);
  on a fresh window the variance-shrinkage logic is valid.
- **Synthetic data verdict** (researched): synthetic rows do not help
  this problem shape; the evidence-backed tools are block bootstrap
  (already standard here), hierarchical shrinkage (underused), and
  cross-league priors (the XLG design). Feature-level noise injection at
  most a minor regularizer; simulator-generated training games have no
  supporting evidence and a formal circularity objection.
- NFL contract-year effect ≈ null in the literature: minor stacker input
  at most, never a standalone candidate.

## The queue (revised for the confirmed format, in order)

> Execution status 2026-08-17 evening: item 1 is DONE (see
> `docs/postseason_support.md`). Items 2-4 plus the ops cadence and the
> postseason snapshot fetch are fully specified for the next sessions in
> **`docs/opus_execution_specs.md`** — start there; the specs make every
> design decision so execution is mechanical.

1. **Playoff coverage**: extend the feature build and weekly prediction
   flow to postseason games (13 pool picks currently unservable).
2. **Rotation registry** (experiment-registry extension) — the
   evaluation substrate for everything below.
3. **MOD-07 stacked candidate** through a rotated window, graded at the
   opener with probability_positive (inputs: learned availability,
   value-weighted injuries, the peer-reviewed opener biases, contract
   year + friction events).
4. **Best Pick ranker**: find a confidence signal that actually orders
   pick quality (calibrated probabilities, key-number geometry,
   regime-aware calibration); validated on top-k-per-week accuracy at
   the opener on rotated windows.
5. **Hierarchical pooling upgrades** (backup-QB value, early-season
   states; then the XLG-05 partially pooled CFB prior).
6. QB-dependence interaction feature.
7. **In-season ops cadence (from Week 1)**: nflverse data refresh +
   feature build + weekly card early Tuesday, published before the
   Tuesday-noon lock; prospective scoring at BOTH grades (opener
   primary); Week Board, predict-close, and the CLV ledger fail closed.
8. Event-aware close prediction — secondary goals only (CLV/close),
   no longer pool-relevant.

Negative results stay recorded (role-continuity family and the MOD-16
variance screen both closed at the CFB benchmark on 2026-08-17; see
`docs/cfb_role_features.md`, `docs/margin_variance.md`).
