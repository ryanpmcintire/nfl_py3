# Pool edge plan: ceiling, gap accounting, and the build queue

Written 2026-08-17 at the end of the session that reframed the project
around its clarified **primary goal: beat the OPENING line the user's
Splash Sports pool grades against** (closing lines, CLV, vig are
secondary). This document captures the strategy analysis behind the
current queue so a new session can pick it up without re-deriving it.

## Where we stand (all measured, all committed)

- **52.83% against Tuesday openers** — the frozen active `weak_stack` model,
  one predeclared look on 1,537 paired 2020–2025 games (`docs/opener_evaluation.md`).
  This improves on the `player` baseline's 52.50% by 0.33 points. **The
  stability evidence below was measured on that baseline, not on the promoted
  stack**, and is quoted here because it is what the 0.33 points sit on top of:
  P(genuine skill vs coin flip) ~97–99%, season-blocked interval excludes 50%,
  positive five of six seasons (the miss is COVID-2020). The stack's own +0.33
  is a play decision on expected value, not a resolved finding — its one
  registry look landed at `probability_positive` 0.8745, short of the
  predeclared 0.90. Prospective 2026 is what settles it.
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
- ~~**Picks lock Tuesday at 12** — essentially at the opener.~~
  **Owner-corrected 2026-08-20:** wrong. The pool's LINE locks Tuesday
  (revised once Wednesday, then frozen for grading) — our PICKS do not.
  Picks are editable any time before each game's own kickoff **(refined
  2026-08-20: SNF/MNF lock early, at Sunday 16:00 ET — the real per-game
  deadline is min(kickoff, Sunday 4pm ET), not kickoff itself)**. The
  Tuesday-opener number stays the grading target, because that is the
  frozen line the pool settles against, but nothing about our own
  submission timing is fixed at Tuesday noon.

Two immediate consequences (both measured the same evening, but drawn from
the wrong-lock premise above; **owner-corrected 2026-08-20** inline):

- ~~**The midweek channel is NOT harvestable for this pool.** Picks are
  due before Wednesday–Friday injury designations exist, so neither
  late-picking nor an event-aware close model can add pool points.~~
  **Corrected: the midweek channel IS harvestable, via a late-week pick
  refresh.** Picks are not due until each game's own kickoff (**refined
  2026-08-20: except SNF/MNF, which lock early at Sunday 16:00 ET, so
  Monday-only information cannot reach an MNF pick**), so
  Wednesday–Friday injury designations (and any other information that
  firms up during the week) can be folded into a re-picked card before
  Sunday/Monday kickoffs — it is graded against a Tuesday line that has by
  then gone stale, which is a stale-line edge, not a closed channel. The
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

> **RETRACTED 2026-08-17 (second session). The corollary below is wrong, and
> it was being used to reject work.**
>
> It claimed that any method whose whole effect is to **rescale** the
> prediction cannot change a forced pick, "because the pick is
> `sign(predicted residual)`". **That premise is false.** The production
> forced pick is `home_cover_probability >= 0.5` — see `pool.py:41` and
> `backtest.py:56` — which thresholds the *median of the out-of-time residual
> sample shifted by the prediction*, not the sign of the prediction. The two
> rules disagree on **11.8% of the 2,075 scored games** (244, measured
> directly). That sample's median is not zero, so rescaling the centre **can**
> flip picks.
>
> **The production rule is resolvably the better of the two, and that is a
> lead, not a footnote.** Scored on the standing 2018-2025 backtest
> (attribution on already-looked-at data, no window): probability rule
> **52.05%** vs sign rule **49.93%**, **+2.12 points**, season-blocked 95%
> **[+0.24, +4.17]**, `probability_positive` **0.990**; on the 244
> disagreements the production rule wins **59.0%**. The whole of that margin
> is the residual sample's *location offset* — and that offset is currently
> the unweighted empirical median of a ~500-900-draw trailing-20% holdout,
> a quantity nobody has ever modelled (no recency weighting, no shrinkage
> toward zero, no conditioning). If the crude version is worth 2.12 points,
> the derivative on estimating it better is worth measuring. Free on CFB.
> **Caveat that must not be dropped: the value of HAVING the correction is
> not the value of IMPROVING it.** The second could be much smaller. This is
> a lead to screen, not a result.
>
> Two independent routes reached this the same day. The decision-rule route
> above, and a coefficient-geometry route: generalized ridge gives
> `b_j = d_j·b_j^OLS/(d_j + λ_j)`, so two penalty vectors are proportional only
> if `(d_j + λ_j¹)/(d_j + λ_j⁰)` is the same constant for every `j` —
> impossible once the `λ_j` differ across blocks. Penalty changes rotate the
> coefficient vector; they were never in the "rescale" class at all. Measured:
> block penalties flip up to 18.6% of CFB picks, a global alpha change 10→1e4
> flips 20.1%, and a positive rescale flips exactly 0.
>
> **What survives:** MOD-06's *conclusion* rests on its own measurement —
> sweeping shrinkage over five orders of magnitude moved accuracy by under a
> point, in the wrong direction. Do not reopen MOD-06. But never again reject
> penalty-structure, calibration, or shrinkage work by citing this corollary;
> it does not license that.

A corollary from MOD-06, same date: any method whose whole effect is to
**rescale** the prediction — shrinkage, regularization, recalibration —
cannot change a forced pick, because the pick is `sign(predicted
residual)` and a positive scalar never changes a sign. Such methods can
improve calibration and confidence ordering, which matters for the Best
Pick, but they cannot move the headline accuracy.

## Gap accounting: 52.5% → ~57% ~~(revised for the Tuesday lock)~~

> **Owner-corrected 2026-08-20:** the heading's parenthetical and item 1's
> "closed for the pool" verdict both rest on the same wrong premise struck
> above — that our PICKS lock Tuesday noon. Only the pool's LINE locks
> then. Picks are editable up to each game's own kickoff (**refined
> 2026-08-20: SNF/MNF lock early at Sunday 16:00 ET, so the real per-game
> deadline is min(kickoff, Sunday 4pm ET)**), so everything
> knowable before that deadline is capturable at pick time; the Tuesday lock
> constrains only which line we are graded against. If anything this makes
> late-week information MORE valuable, not less — a pick placed Saturday
> is graded against a Tuesday line that has since gone stale, a stale-line
> edge rather than a closed channel. The measurements in item 1 below (the
> 2.39% PFT-headline-visibility figure, the 0.000-pt/P+0.3965 Tuesday-cutoff
> contrast, the +1.3158-pt Saturday-minus-Tuesday channel delta) all stand
> as measurements of what was known when; only the "not playable"
> conclusion drawn from them is retracted. See
> `docs/injury_news_sourcing.md` §5.1 and `docs/prospective_evidence.md`'s
> Tuesday-visibility audit for the corrected framing: a late-week pick
> refresh makes the Saturday-cutoff channel directly playable.

1. ~~**Midweek information channel, ~2.6 points — closed for the pool.**~~
   **Midweek information channel, ~2.6 points — playable via a late-week
   pick refresh, not closed.**
   The 55.1% movement oracle needs post-Tuesday information; ~~the pool
   locks Tuesday noon. What remains capturable at pick time is only
   Monday-night results and Monday/Tuesday-morning news the pool's own
   Tuesday line hasn't priced~~ — **owner-corrected 2026-08-20:** the
   pool's LINE locks Tuesday noon, but our picks do not, so anything known
   before that game's real deadline (**refined 2026-08-20: min(kickoff,
   Sunday 16:00 ET) — SNF/MNF lock early at Sunday 4pm, not at kickoff**)
   is capturable by re-picking that game later
   in the week — ~~a thin slice, largely already inside
   our features~~. **Correction, 2026-08-19 (measured, `docs/injury_news_sourcing.md`
   §5.1):** this claim was false. Only **2.39%** of Friday-final injury
   designations (1,838 of 76,782 rows) are already headline-visible via
   PFT injury news by that Tuesday-noon cutoff, and the `injury_value_lost`
   edge under a true Tuesday-noon decision cutoff is **0.000 accuracy
   points, `probability_positive` 0.3965** (vs. +1.316 pts, P+ 0.8875 at
   the previously-used Saturday-default cutoff) — ~~the previously measured
   edge is not currently playable by a forced Tuesday pick~~. **Owner-
   corrected 2026-08-20:** the Saturday-cutoff construction (+1.316 pts,
   P+ 0.8875) is exactly what a late-week pick refresh sees, so it is the
   playable figure, and the Tuesday-cutoff 0.000-pt reading describes only
   the Tuesday PUBLISH, not the pool's actual constraint; see lead 3
   below. (The channel remains fully relevant to the secondary
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
   > pooling is closed (MOD-06, 12,500 free CFB games *(corrected 2026-08-18;
   > was 12,206)*). The reason is
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
   > player-value/availability half carried **+1.75 of the +1.97**
   > (P+ 0.899) — corrected 2026-08-17 from "the whole +1.97", which
   > overstated it. That 0.899 was cited in five places with **no artifact
   > on disk**; it is now reproduced by `scripts/availability_ablation.py`,
   > which first rebuilds all six recorded MOD-07 quantities exactly.
   > The headline holdover figure also fails to replicate: 35.6% published
   > against **52.5% measured** on 120 Week-1 holdover favourites here
   > (season-blocked diff vs plain favourites −3.6 points, [−14.3, +6.6]).
   > Week-2 anchoring came out directionally *opposite* the hypothesis.
   > Low-visibility remains unbuilt and its two feasible proxies (book
   > count at the opener, standalone-window flag) were both null on the
   > only non-reserved data available. **Do not add more bias features.**
   > The availability thread is where the measured signal actually is.
   >
   > **Availability downgraded 2026-08-17 (second session), and the reason
   > is a methodology lesson.** The claim was five independent measurements
   > all positive, sign test p=0.0625. Every one sits on the *same* 2,075
   > games — there is no independent football in the family — and correcting
   > for the shared sample moves p to **0.098**. Worse, the "five" was never
   > written down, and the set that reproduces 0.0625 excludes a same-kind
   > negative (participation RAPM, −0.43 pts, sharing the `player_value`
   > baseline arm). Include it: **p=0.219**. Broadest defensible set:
   > **p=0.180**. The boundary was drawn after the signs were visible.
   > Category 3, unresolved — not a finding. **A sign test is only worth its
   > family definition, and the family must be declared before the signs are
   > seen.**
   >
   > **What replaced it is narrower and better.** Splitting the family along
   > its only two axes, the halves disagree. Injury *value lost*:
   > disagreement rank-biserial **+0.248, 95% [+0.046, +0.450], p=0.016**,
   > with a monotone accuracy gradient across terciles (−0.66 / +0.66 /
   > **+5.26** pts). Learned availability *rate*: −0.024, [−0.226, +0.178],
   > p=0.816, and its gradient runs *backwards*. Placebo axes (|spread|,
   > total, week) are non-monotone, so this is not tercile-slicing artifact.
   > The signal looks like **how much value is missing, not how likely
   > someone is to play**. Hypothesis, not promotion — see
   > `docs/availability_confirmation.md`.
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
  surfaces every overlapping pair instead of hiding it; since 2026-08-24 it is
  organized **per family** (`family_overlap_warnings`) — one row per correlated
  decomposition group (grades, era splits, battery cells of one construct),
  with pairwise totals kept alongside — after the raw list passed 55,000
  strings and stopped being readable (`docs/registry_correlation_audit_20260822.md`,
  risk #3).
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
> **`docs/archive/opus_execution_specs.md`** — start there; the specs make every
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

## Where to look next (2026-08-18)

**The promotion test on MOD-07 had two grades, one negative and one positive.**
Graded at the **close** — the nflverse spread on 2,075 games 2018-2025 — the
baseline `player` profile scored **52.05%** and `weak_stack` scored **51.57%**,
a 0.48-point loss on 4.5x the opener-window data. At the **opener** — the
Tuesday-lock grade on the same 1,537 paired games where the stack first won
+1.97 — the `player` profile scored **52.50%** and `weak_stack` scored
**52.83%**, a 0.33-point gain. The close result initially refused promotion.
That refusal was **wrong instrument** per AGENTS.md: the market drifts from
opener to close, and week-locked forced picks are settled against the opener,
not the sharp-market close. On the pool-relevant grade, the `weak_stack` was
promoted (commit 68b4dc0). The +1.97 measured on 456 games must not be quoted
as expected gain — regression of a selected effect on a smaller window is the
right lesson — and the 0.33 point on 1,537 paired games is the promoted claim.

**What the weak-signal pool says.** **Correction, 2026-08-18:** this section
previously quoted a pooled result of +0.724 accuracy points, 95%
[+0.056, +1.392], `probability_positive` 0.983, sharpening 1.46x over the
best single input, attributed to three signals at `probability_positive`
0.860 / 0.933 / 0.917. Audited later the same day, that pool does not
reproduce from `registry/weak_signals.json` — see AGENTS.md's "interval
crossing zero" section for the full correction — and the three inputs' numbers
look like split-half reliabilities conflated with `probability_positive`; the
pool tool has never emitted a `P+` field. The living reference is
`nfl-ats weak-signals pool --league nfl --effect-units accuracy_points`
(read-only; `weak-signals record` is the separate recorder that writes to
the registry). As of 2026-08-18 (107 signals recorded) it reports a
random-effects pool of **-0.023 accuracy points, 95% [-0.073, +0.028]**,
`excludes_zero: false` — the pile is close to a coin flip, leaning slightly
negative, and is not resolved, so it has not yet earned the ONE predeclared
combined look this framing anticipates. The pool now includes correlated
decompositions of shared windows (see `overlap_warnings`, extensive for this
batch), so the interval overstates precision; the sign-test (24 of 51
favouring the candidate, p=0.780) and per-entry rows are the safer read.
Re-run the command for the current state rather than quoting a fixed number
here. Since 2026-08-24 `overlap_warnings` is a per-family structure
(`families`, `within_family` rows, cross-family shared-window counts) rather
than an unbounded pairwise string list, so "extensive" no longer means
unreadable.

**Live leads, ordered.**

Leads 1 and 2 were both spent on 2026-08-18 and are now **answered**; lead 3
survived its decisive test and is the one still live.

1. **`ridge_alpha` — ANSWERED, leave it alone** (`docs/ridge_alpha.md`).
   Undefended and inert are both true. A free 19-point CFB sweep (1e-3 to 1e5,
   12,500 games) finds forced-pick accuracy **flat across seven orders of
   magnitude**; only 1e5 is clearly worse. Brier is resolvable and minimised
   near **α ≈ 2,000-2,500**, worth **+0.0003 (~0.12% relative)** across a broad
   300-10,000 plateau at `probability_positive` 0.75-0.97.
   **Re-valued 2026-08-18, and the first reading was too dismissive.** It was
   written off as "calibration, not picks" hours before `docs/variance_reduction.md`
   established that Brier is this project's **~9x most sample-efficient
   measurement channel** — so a resolvable Brier gain is now the currency we
   screen in, not a curiosity. It still moves no pick by itself, and the model
   should still be left alone. But it feeds the Best Pick ranker (a real pool
   lever: one nomination a week, 18+ weeks) and it belongs in the stack of small
   compounding wins, not in the discard pile.
   Two stale numbers corrected: "rank-71-of-142" was the retired `player`
   profile (active `weak_stack` is 90 → 159 transformed → **rank 82**), and
   **59 of the 77 lost dimensions are duplicated `SimpleImputer` indicators**,
   not `diff = home - away` — the warm-up rule gates a whole team-state vector
   at once, so 45 of 69 indicator columns are bit-identical.
2. **Residual location offset — MECHANISM FOUND, remedies REFUTED**
   (`docs/residual_location.md`). The +2.12 stands and is now explained: the
   residual sample comes from a temporary model fit on the leading 80% while
   the deployed estimator is refit on 100%, so the offset measures how far the
   target's mean moved between those slices (corr **0.944**, slope 1.13 on CFB;
   0.615 / 1.00 on NFL). It is a recency-aware intercept the unweighted
   expanding-window ridge lacks. The offset is **not** a stable bias to model —
   it crosses zero in both leagues, pooled mean ≈ 0. On the CFB remedies, state
   this precisely rather than as "all eight lose" (that phrasing was used on
   2026-08-18 and is the binary framing AGENTS.md bans): **two are refuted** —
   recency half-lives 200/400 resolve negative under both blockings
   (`probability_positive` 0.014 / 0.0005), and every recency arm worsens Brier
   at P+ 0.000 — while **six are unresolved, not negative**, at P+ 0.058-0.389;
   `shrink_025` at −0.03 is a null, not a loss. All eight are recorded in
   `registry/weak_signals.json`. **This validates production** — the accidental
   unweighted ECDF is already near-best in its family. The untested lever is recency-weighting the *mean model's
   training rows*, not the residual reader; predeclaration drafted.
3. **Injury value lost, not availability rate — STILL LIVE, refutation now
   ruled out** (`docs/injury_value_lost.md`). Split-half reliability on 384
   team-seasons is **0.933 [0.915, 0.948], P+ 1.000**, so the trait genuinely
   repeats and the "no split-half reliability" closer is off the table — only
   power is missing. It is orthogonal to what the market already moved on
   (r=0.029 with line movement) and is **not** a quarterback story: the effect
   persists on the 87% of games with no QB issue (r=0.188, p=0.086), which the
   code corroborates — `players.py:1263` excludes QBs from `skill_epa` by
   construction. A cleaner arm without the semantics-shift confound scores
   **+1.316 points, P+ 0.8875**; predeclare against that, not the conflated
   +1.75. CFB screening is impossible here as a data fact — no CFB source
   carries any pregame injury signal, so `severity` cannot be computed. Stays
   `unresolved_below_power`; the frozen predeclaration targets `[2022, 2023]`
   once the live 2026 prospective look lands.

   **Update, 2026-08-19** (`docs/injury_news_sourcing.md` §5.1,
   `scripts/injury_tuesday_cutoff_experiment.py`): the +1.316 pt / P+ 0.8875
   figure above is measured at the Saturday-default `decision_hours_before_kickoff=24`
   cutoff, not the pool's actual Tuesday-noon lock. Under a true Tuesday-noon
   cutoff (official injury report only), the same 456-game contrast
   collapses to +0.000 pts, P+ 0.3965; crediting every PFT-foreshadowed
   designation as Tuesday-known still only reaches -0.219 pts, P+ 0.248. The
   paired channel-delta test isolating exactly the Tuesday-to-Saturday
   information gap reads **+1.32 to +1.54 pts, P+ 0.90-0.92** — the channel
   is real (not refuted: both channel-delta intervals cross zero, and
   split-half reliability stays 0.933) ~~but **not currently playable by a
   forced Tuesday pick**. Any future `[2022, 2023]` predeclaration on this
   family must specify a Tuesday-noon (or later-information-excluded)
   decision cutoff explicitly, not the Saturday default, or it will
   re-measure a channel the pool cannot use.~~ **Owner-corrected
   2026-08-20:** the pool does not force a Tuesday pick — picks are
   editable until each game's real deadline (**refined 2026-08-20:
   min(kickoff, Sunday 16:00 ET) — SNF/MNF lock early at Sunday 4pm**) —
   so the Saturday-cutoff
   construction (`decision_hours_before_kickoff=24`) IS the playable one
   via a late-week refresh, not an unplayable channel. Any future
   `[2022, 2023]` predeclaration on this family should use the Saturday
   (or a genuinely pre-kickoff) decision cutoff, matching what a late-week
   refresh pass actually sees; the Tuesday-cutoff figures above describe
   only the Tuesday PUBLISH's starting point, not a hard constraint on what
   the pool can play. Four new registry entries:
   `injury_value_lost_tuesday_cutoff_official`,
   `injury_value_lost_tuesday_cutoff_pft_augmented`,
   `injury_value_lost_tuesday_saturday_channel_official_only`,
   `injury_value_lost_tuesday_saturday_channel_pft_augmented`, all
   `unresolved_below_power`.
4. **Pool format** is a multiplier to protect, not a lever to pull
   (`docs/pool_format_levers.md`): 52.5% already buys 6.56% of first place
   against 100 rivals vs a 0.99% fair share, and two accuracy points are worth
   **+11.8 pp** against the best format lever's +2.4 pp.

## 2026-08-31 registry state and next shots

This is a read-only survey pass, not an adjudication. Nothing below closes,
downgrades, or reclassifies any registry entry; every terminal verdict still
requires an admissible `closing_ground` and flows through `nfl-ats
weak-signals record` / `rotation record-look`, never through prose here.
Every claim is tagged **measured** (run this session), **read** (file:line,
opened this session), **reported** (a doc or prior session says so,
unverified), or **inferred** (reasoning/judgement, not evidence).

### 1. Measured pooled state

**Measured**, command `nfl-ats weak-signals pool --league nfl --effect-units
accuracy_points`, run this session:

- 533 NFL `accuracy_points` signals are currently poolable
  (`unresolved_below_power`, not refuted/control-bounded); 531 of those carry
  a resolvable standard error and enter the inverse-variance pool.
- Random-effects pooled effect: **+0.00196 accuracy points, 95% [-0.01202,
  +0.01594]**, `excludes_zero: false`, heterogeneity τ²=0.000747,
  `sharpening_vs_best_single` ≈0.0036.
- Sign test: **288 of 533 favour the candidate direction, 245 favour the
  baseline direction, p=0.0688** (tool's own interpretation string: "directions
  lean further than chance comfortably explains" — note this sits just above
  the conventional 0.05 line, so read it as a lean, not a resolution).
- Overlap: 122,721 pairwise overlapping pairs; 4,507 of those within one
  measurement family (correlated decompositions of the same window — one
  dependent vote each, not independent ones, per AGENTS.md and
  `docs/registry_correlation_audit_20260822.md` §3); 314 distinct families,
  26 with internal overlap. Largest overlapping families: `graph_input_screen`
  (82 members, NFL seasons 2020-2025), `graph_ratings_v2_team_stat` (38
  members, 2011-2013), `bias_battery` (25 members, 2009-2025),
  `weather_battery` (8, 2009-2025), `attention_battery` / `odds_microstructure`
  / `sagarin_battery` (7 each).
- Per AGENTS.md, this interval crossing zero is **not** grounds to close
  anything; the pile stays `unresolved_below_power`. Compared with the
  2026-08-18 read quoted above in this file (-0.023 accuracy points, 95%
  [-0.073, +0.028], sign test 24 of 51 favouring the candidate, p=0.780, 107
  signals recorded), the registry has grown roughly 5x (609 signals total
  across both leagues -- measured, `len(registry.signals)`: 542 NFL + 67 CFB,
  606 `unresolved_below_power` + 3 `refuted_mechanism`; 533 of the 542 NFL
  signals are in this specific `accuracy_points` pool) and the sign test has
  moved from indistinguishable-from-a-coin-flip
  toward the candidate side (p 0.780 → 0.069) without resolving. Re-run the
  command above for the current number rather than quoting either of these
  as fixed.

**Family breakdown (measured, same pool, grouped by `signal_family`).**
Within-family pooled reads below are informational only — they share one
window per AGENTS.md's overlap discipline and overstate precision; the sign
test and per-entry rows are the safer read, exactly as the living guidance in
this file already says.

| family (n) | seasons | sign test (cand/base, p) | pooled (informational) |
|---|---|---:|---|
| `graph_input_screen` (82) | 2020-2025 | 63/19, p≈0.0000 | +0.493 [+0.096,+0.890] |
| `graph_ratings_v2_team_stat` (38) | 2011-2013 | 12/26, p=0.0336 | -0.107 [-0.424,+0.209] (matches `docs/graph_ratings_v2_screen.md` §8 exactly) |
| `bias_battery` (25) | 2009-2025 | 6/19, p=0.0146 | -0.024 [-0.044,-0.003] |
| `weather_battery` (8) | 2009-2025 | 7/1, p=0.070 | +0.078 [-0.009,+0.166] |
| `attention_battery` (7) | 2016-2025 | 2/5, p=0.453 | -0.044 [-0.268,+0.180] |
| `odds_microstructure` (7) | 2020-2025 | 4/3, p=1.000 | +2.224 [-1.135,+5.582] (mixes real hypotheses with deliberately-leaked oracle controls -- see item 4 below) |
| `sagarin_battery` (7) | 2010-2025 | 2/5, p=0.453 | -0.439 [-2.113,+1.234] |
| `referee_battery` (6) | 2016-2025 | 4/2, p=0.688 | +0.058 [-0.134,+0.250] |
| `roof_battery` (6) | 2009-2025 | 2/4, p=0.688 | -0.042 [-0.062,-0.021] |

Two of these are worth flagging on their own terms (both **measured**, both
matching prior write-ups): `bias_battery`'s 25 members lean 6-favour/19-against
at p=0.0146 -- consistent with `docs/pool_edge_plan.md`'s own 2026-08-18
finding that the opener-bias features ablate to a coin flip (+0.22 pts,
P+ 0.505); and `graph_ratings_v2_team_stat`'s -0.107 [-0.424,+0.209] pooled
read reproduces `docs/graph_ratings_v2_screen.md` §8's number to three decimal
places, cross-checking that the family-overlap machinery and the dedicated
screen script agree. `graph_input_screen`'s 63/19 lopsidedness (p≈0.0000) is
the single most one-sided family in the registry, but it is 82 cells on one
2020-2025 window -- one dependent vote, not 82 independent ones.

### 2. Commensurable, genuinely non-overlapping pool candidates

The registry's big families (above) are the opposite of a pooling candidate:
same window, same population, correlated by construction. The interesting
cases are the small families whose members are **era-split** -- disjoint
NFL seasons, same construct -- because the two halves genuinely do not share
football, unlike a family's own full-range parent entry (which overlaps both
halves at once and was excluded from the sub-pools below). All read from
`registry/weak_signals.json` this session; each pooled figure is
**informational** -- a real predeclared confirmatory look would need to be
run fresh, not built by combining entries whose signs are already visible,
which is exactly the "family declared before signs are seen" rule this
project treats as binding.

**Consistent sign across disjoint eras, low heterogeneity (best candidates
for a future predeclared pooled confirmation):**

- `body_clock_west_road_early`: 2009-2016 era -0.031 pts vs 2017-2025 era
  -0.118 pts, both negative, τ²=0.000 -- pooled -0.069 [-0.294,+0.157].
- `body_clock_night_west_road_ge2000et`: 2009-2016 -0.105 pts (n=48 games) vs
  2017-2025 -0.064 pts (n=71 games), both negative, τ²=0.000 -- pooled -0.087
  [-0.209,+0.036]. Samples are thin (rare kickoff-slot/geography combination);
  a future pooled look here would still be sharpening a small effect, not
  discovering a large one.
- `altitude_deficit_4000ft` and `interim_hc_active` era splits are also
  low-heterogeneity and disjoint but both effects sit near zero in both eras
  (pooled +0.025 and -0.028 respectively) -- technically poolable, low EV.

**Sign flips across the era boundary (era-magnitude, not a pooling
candidate -- report per-era magnitude separately, per this project's standing
"era magnitude, not presence" rule, rather than averaging them away):**

- `pt_post_mnf_sunday`: 2009-2017 +0.255 pts vs 2018-2025 -0.137 pts.
- `sagarin_battery_large_divergence`: 2010-2016 +1.807 pts vs 2017-2025
  -2.299 pts, τ²=3.52 -- real detected heterogeneity, the clearest case in
  this pass where pooling would misrepresent two eras that are not measuring
  the same thing.
- `bye_overval_home_edge`: pre-2011 +0.271 pts vs post-2011 -0.330 pts,
  τ²=0.034 (plausibly the 2011 overtime-rule change, **inferred**, not
  verified against a rules-change date).

**Excluded despite being time-disjoint:** `surface_familiarity_r3`'s two era
halves (2009-2017 +0.528 pts, 2018-2025 +2.387 pts, both positive, τ²=0.000)
would otherwise be the strongest same-sign candidate in this pass, but the
2018-2025 half's own registry note flags it as reinforcing evidence for the
upstream weather-battery mining window rather than an independent read (its
own text: "because it lives in the 2018-2025 window -- the same window the
upstream weather battery was mined on"). Pooling it would launder an
already-played overlay's own decomposition, so it is excluded here and from
the ranked agenda below.

**A working precedent for how to do this right:**
`sbr_opener_pooled_2011_2021` (+0.928 pts, 95% [-0.873, +2.754], P+ 0.841,
2,832 games) is not a mechanical combination of its three era-split inputs
(`sbr_opener_era_2011_2014` -0.251, `era_2015_2019` +0.891,
`era_2020_2021` +3.295, whose naive random-effects pool reads +0.809
[-0.989,+2.606]) -- it was **measured directly on the 2011-2021 union
window**, which is the right way to take a predeclared combined look: on the
union of the data, not by algebra on top of separately-recorded point
estimates.

### 3. Ranked next-shots agenda

Ranked by EV (reliability, era stability, whether a marginal-on-production
test already ran), excluding today's three already-run shots
(`graph_team_stat_off_sack_rate_on_production`, the two `fluview_*_on_production`
reads) and the leads the 2026-08-26 continuity already ruled out
(`surface_familiarity_r3_era_2018_2025`, `movement_attribution_pop_threshold_injury`,
`odds_microstructure_*_oracle_*`, `special_teams_return_top_quartile`).

1. **`graph_team_stat_def_yards_per_play` → on-production marginal test.**
   **RESOLVED the same day it was written (2026-08-31; commit `0d9af09`), stale here until 2026-09-01:** the on-production marginal ran -- paired delta **-0.668 accuracy points**, week-blocked 95% [-2.243, +0.804], P+ **0.189**, 8.0th percentile of its own null, recorded `unresolved_below_power` under `graph_def_ypp_on_production`, window [2014, 2016] spent `unresolved` (`docs/graph_team_stat_def_ypp_on_production.md` section 7). Original text kept below for the record.
   Read, `docs/graph_ratings_v2_screen.md` §8: the family's other
   two-predeclared-candidates-remaining cell; leads the conservative
   (null-adjusted) reference at the **95.5th percentile** of its own
   permutation null (+2.145 pts vs zero, P+ 0.965, null centred at only
   +0.279 -- the *least* artifact-contaminated of the three cells the doc
   names). Independently, the same underlying feature scored +1.198 pts,
   P+ 0.711 on a completely disjoint holdout window
   (`graph_input_screen_def_yards_per_play`, opener-graded 2020-2025,
   n=1,503, reliability 0.985). Two non-overlapping windows, same sign, high
   reliability. EV case: `off_sack_rate` -- the family's best cell by raw
   magnitude -- went negative on production today (-0.935 pts, P+ 0.122);
   `def_yards_per_play` is the doc's own named next candidate specifically
   *because* it is the cell least explained by the home-tilt artifact, so
   this is a distinct bet, not a repeat of the one that just lost.
2. **`graph_team_stat_off_rush_epa_per_play` → on-production marginal test.**
   **RESOLVED 2026-09-01 (WP2, measured).** Ran close-graded on the rotation-ASSIGNED window [2014, 2016] (749 games, 51 weeks -- the same block both siblings drew): candidate `weak_stack_graph_off_rush_epa` 50.07% vs production `weak_stack` 51.00%, paired delta **-0.935 accuracy points**, week-blocked 95% CI [-1.998, +0.135], `probability_positive` **0.037**; season-blocked [-1.195, -0.800], P+ 0.000 (3 blocks, a low-power artifact of block count). Home-pick rates 37.63% vs 36.72%. Against its own 200-draw within-week null (centre +0.196) the observed delta sits at the 0.5th percentile. Positive control +48.999 pts, P+ 1.000. Recorded `unresolved_below_power` (family `graph_off_rush_epa_on_production`, NOT poolable with `graph_ratings_v2_team_stat` / `graph_input_screen` / either sibling); window spent `unresolved`; `remaining_eligible_windows: 0`. The item's own EV case is what the result speaks to: the family's highest split-half reliability (0.987), replicated in sign on a disjoint opener-graded window, did not predict survival once stacked on what is played -- reliability measures consistency of the trait, not whether the production chain already prices it. See also the CFB replication in section 4 below, which reads the whole graph lane as "better single feature, not additional information".
   Read, same doc: +1.609 pts vs zero, P+ 0.911, but only the **53.5th**
   percentile of its own null (null centred at +1.450 -- most of the apparent
   edge is plausibly the artifact, disclosed as such in the doc). Independently,
   `graph_input_screen_off_rush_epa_per_play` (opener-graded 2020-2025,
   n=1,503) reads +1.996 pts, P+ 0.828, reliability 0.987 -- the single
   highest reliability figure in the whole family. EV case: run second, after
   `def_yards_per_play` -- highest reliability in the family, but the doc's
   own disclosure means expect a more muted on-production read.
3. **`fluview_home_market_elevated` → fresh opener-graded confirmation on an
   **RESOLVED 2026-09-01 (WP3, measured).** `nfl-ats rotation assign --name fluview_home_elevated_opener` returned **(2022, 2023)** -- the family's last default-size opener block (`remaining_eligible_windows: 0` afterward). The screen (514 paired games, 36 weeks) read PRIMARY (production probability rule, week-blocked) delta **-1.751 accuracy points, 95% CI [-4.501, +0.986], P+ 0.094** -- a stronger lean against the candidate than the first window's -0.439, P+ 0.341; the secondary sign rule leans for it in both (P+ 0.522, then 0.802). Neither primary interval resolves the sign; the positive control (+42.023 pts, P+ 1.000) detects at ~24x the tested scale, so `bounded_by_control` is unavailable. Recorded `unresolved_below_power` (`fluview_home_market_elevated_opener_confirmation_2022_2023`), rotation verdict `unresolved`. A doc-only, post-hoc union read (2020-2023) gives PRIMARY -1.134 pts, P+ 0.113 (`docs/fluview_opener_look.md` sections 8-9). EV read: the rule that decides what gets played does not support adding this feature, now on two windows plus the CFB replication in section 4.
   unspent window.** Read: the close-graded on-production read across the
   full 2011-2025 population is +0.969 pts, P+ 0.792, reliability 0.9814 --
   one of the highest reliabilities recorded anywhere in this registry. The
   only opener-graded confirmation run so far used a thin two-season window
   (2020-2021, 456 games) and read -0.439 pts, P+ 0.341. Both intervals cross
   zero; neither resolves anything, and this is not a contradiction -- a
   construct this reliable deserves a properly powered opener-graded look on
   a fresh multi-season window before the thin 2020-2021 read is treated as
   the last word.
4. **`injury_value_lost_gradient` / `_narrowed` → on-production marginal test
   (queued, gating condition not yet met).** Read, split-half reliability
   0.933 [0.915, 0.948] -- among the highest measured in the registry -- with
   a clean-channel read of +1.316 pts, P+ 0.8875 (the Saturday-cutoff
   construction that matches what a late-week pick refresh actually sees).
   This file's own 2026-08-18 entry (above) freezes the next predeclaration
   at seasons [2022, 2023] "once the live 2026 prospective look lands."
   **Inferred:** since today is 2026-08-31 and the 2026 season has not yet
   kicked off, that gate is not yet satisfied. No on-production
   (candidate-vs-`weak_stack`) test of this construct exists yet at all --
   unlike the graph-ratings family, this one has never been tried against
   production. Once the gate clears, its reliability alone puts it above
   items 1-3 on this list.
5. **Best Pick ranker: dispersion-filtered chooser vs unfiltered, confirmed
   on a genuinely fresh window.** Read: current reading +3.922 pts, P+
   0.8132 (`best_pick_opener_ranker_dispersion_filtered_candidate_vs_unfiltered`)
   -- the strongest of the four current ranker variants -- but its own
   registry note states it is the "Third reuse of the same 107 opener weeks"
   with an explicit compounding-multiplicity discount attached. EV case:
   `docs/pool_edge_plan.md`'s own gap accounting (above) flags the Best Pick
   lever as free and currently unexploited (confidence ordering is flat,
   ~52.5% regardless of pick chosen); this is the strongest surviving
   variant and deserves one clean predeclared confirmation on an untouched
   window before being treated as evidence in either direction.
6. **Not ranked as a next shot, but worth carrying forward:** the
   **RESOLVED 2026-09-01 (WP4; read-only survey, nothing recorded).** `scripts/era_magnitude_report.py` + `docs/era_magnitude_report.md` deliver the per-era magnitude tables (9 era-split groups found programmatically) and mechanism proposals. Verified on the web: `bye_overval_home_edge`'s pre/post-2011 boundary coincides with the 2011 CBA practice-time caps and the regular-season OT rule (effective 2012) -- and `docs/bye_overvaluation_screen.md` already predeclares that exact "post-2011 CBA" mechanism, so the two era entries are two of its five cells, not a fresh hypothesis; `pt_post_mnf_sunday`'s 2017/2018 split is the primetime battery's own fixed convention with no coinciding rule change; `sagarin_battery_large_divergence`'s 2016/2017 split has no verified external cause, but the early era's Sagarin coverage is severely non-uniform (2012 at 0.0%), a data-grounded candidate mechanism now being fixed (`docs/sagarin_backfill.md` section 9). One unrun predeclarable cell is proposed per construct; the four consistent-sign/low-tau^2 constructs are carried forward for a future union-window confirmation per the `sbr_opener_pooled_2011_2021` precedent.
   era-magnitude constructs in section 2 above (`pt_post_mnf_sunday`,
   `sagarin_battery_large_divergence`, `bye_overval_home_edge`) show
   era-dependent sign flips -- real detected heterogeneity in the Sagarin
   case -- so the right next step for these is reporting per-era magnitude
   (per this project's standing rule) and, if a mechanism for the
   era-dependence is proposed, screening that mechanism directly, not
   pooling across the flip or spending an on-production window on either
   half alone.

### 4. 2026-09-01 afternoon fleet: what else moved (all measured that day unless labelled)

**Measured 2026-09-01 (afternoon fleet session, `nfl-ats weak-signals pool --league nfl --effect-units accuracy_points`):** 539 NFL `accuracy_points` signals pool to **+0.00168 accuracy points, 95% [-0.01234, +0.01570]** (random effects, tau^2=0.000759, `excludes_zero: False`); sign test **288 of 541 favour the candidate direction, 253 the baseline, p=0.144** (tool's own reading: "directions are consistent with a coin flip"). Per AGENTS.md this crossing-zero interval closes nothing; re-run the command rather than quoting this line.

- **FluView illness, the cross-league read comes back against the card (WP7, `docs/fluview_cfb_replication.md`).** The one positive reading this construct has is close-graded and NFL-only (+0.969 pts on production, P+ 0.792, 516 games). It now has three independent negative-leaning counterweights: the two NFL opener windows (-0.439 P+ 0.341; -1.751 P+ 0.094) and a college-football replication of the identical construct on **5,671 games** -- **-0.388 pts, 95% [-1.272, +0.460], P+ 0.200**, same sign in both CFB eras (away-market -0.423, P+ 0.213, whose eras carry opposite signs and are reported separately). None of the four is commensurable and they are never pooled; the trait's split-half reliability replicates at 0.9856 on CFB; all cells stay `unresolved_below_power`. What changes is the EV of the next move: further NFL window on the FluView home-market cell is a worse bet than it looked, and the cell should not enter the played chain on the close-graded +0.969 alone.
- **Where the graph lane stands (WP8, `docs/graph_team_stat_cfb_replication.md`).** On CFB, where the schedule graph is ~5x sparser (edge density 0.085 vs 0.42) and 73% of games are conference games, the graph transform as a *single feature* beats the raw statistic on all three available cells (+0.369 / +0.291 / +0.694 pts, P+ 0.798 / 0.765 / 0.897, 8,933 games, 17-26% of picks moved) -- the first corroboration of `graph_input_screen` from outside the NFL -- but as an *addition* to a model already carrying the raw statistic it is worth about nothing (-0.011 / +0.022 / -0.179 pts, P+ 0.467 / 0.535 / 0.266; picks moved 1.9-5.1%). Practical consequence: pursue the graph lane, if at all, as a *replacement* for a raw team-state input, never as a bolted-on column; and read the NFL on-production negatives (-0.935 / -0.668 / -0.935) as consistent with a null, not as harm. Nothing closes; no card moves on CFB evidence.
- **Totals (tiebreaker), WP1 + WP18 + WP14.** The market total beats our model total outright (MAE 10.4249 vs 10.5495 on 3,935 walk-forward games 2010-2025); the served guess is `total_line + 0.1 * predicted_residual`, worth +0.0008 points at P+ 0.583 -- a nudge taken because the pool forces a tiebreaker every week, not an edge (`totals_market_residual_blend`). Wave 2 (24 drive-pace columns) beats wave 1 by +0.0020 MAE points, P+ 0.8235 (`totals_market_residual_wave2_vs_wave1`) and is the EV favourite for the served view. The blend surfaced a hard-window edge in the tiebreaker's calibration neighborhood (a 0.042-point line move flipped the guess the WRONG way); the neighborhood is now kernel-weighted and monotone in the line. **Week 1 tiebreaker card now reads KC 22 - DEN 19** (was KC 23 - DEN 20).
- **PER-13 durability prior, Stage 1 (WP13, `docs/per13_durability_prior.md`):** availability Brier 0.09087 -> 0.08332 (+0.0075514, 95% [+0.0067549, +0.0083081], P+ 1.000, 52,382 player-games, placebo -0.0000871). EV gate met; Stage 2 on production is warranted and queued. Not an ATS number: PER-11's 0.00444 on the same metric bought +0.10 ATS points.
- **Prediction safety for Sep 8 (WP15, `docs/missingness_audit.md`):** no 2026 Week 1 production column carries a missingness state unseen in >=1% of 2025 rows -- no imputer/ridge extrapolation risk from that mechanism.
- **Registry housekeeping:** `EFFECT_UNITS` gained `correlation` and `mae/brier/log_loss_improvement` (positive = candidate better) plus `nfl-ats weak-signals retag-units` (WP16); `docs/movement_expansion_battery.md` got its missing Results section and XLG-06 Stage 1 was written up and recorded (WP11); BET-09 is guarded in code and `PoolRules` composes the deadline and tiebreak rules (WP12); the capture scheduler can catch up idempotent jobs and now captures player arrests every Tuesday (WP10); an artifact-retention planner exists in dry-run form and found `artifacts/` had no off-device backup (WP6).
