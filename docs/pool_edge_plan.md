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
σ ≈ 13.5 points (turnover bounces, in-game injuries, one-score-game coin
flips). Exchange rate: **1 point of true line error ≈ 3 points of ATS
accuracy** (Φ(1/13.5) ≈ 0.53).

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

## Gap accounting: 52.5% → ~57%

1. **Midweek information channel, ~2.6 points, most tractable.** The gap
   to the 55.1% movement oracle is Wednesday–Friday injury designations,
   weather, and the market's own digestion of news. Two paths depending
   on ONE unresolved operational fact — **when the user's specific
   contest locks picks** (Splash lock timing is per-contest: each game's
   kickoff, first game of slate, or Sunday 1pm ET):
   - *Kickoff/late lock*: submit picks as late as allowed; the live
     market line vs the frozen pool line IS realized movement — most of
     the channel is captured mechanically (compare current consensus to
     the pool number, take the moved-toward side, model as tiebreak).
   - *Early lock*: the channel must be predicted — the event-aware close
     model (close-minus-open ~ opener + post-Tuesday injury designations
     weighted by player value/position, hourly 2023–2025 odds + practice
     reports, fully clean prospectively in 2026).
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
3. **Estimation noise, ~0.5–1 point.** Coefficient noise from ~4,500
   training games; recovered by shrinkage and by stacking weak signals
   instead of discarding them (MOD-07), never by synthetic rows.
4. **Unreachable remainder.** Private information and oracle perfection.

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

## The queue (in order)

0. **User action: check the Splash contest's pick-lock rule** (contest
   details page) — it decides between the two midweek-channel paths.
1. Event-aware close prediction (or late-pick ops tooling if the lock
   allows) — the ~2.6-point channel.
2. Rotation registry (experiment-registry extension).
3. MOD-07 stacked candidate through a rotated window, graded at the
   opener with probability_positive.
4. Hierarchical pooling upgrades (backup-QB value, early-season states;
   then the XLG-05 partially pooled CFB prior).
5. QB-dependence interaction feature.
6. Prospective 2026: score active model + challengers at BOTH grades
   (opener primary) — Week Board, predict-close, and the CLV ledger are
   armed and fail closed.

Negative results stay recorded (role-continuity family and the MOD-16
variance screen both closed at the CFB benchmark on 2026-08-17; see
`docs/cfb_role_features.md`, `docs/margin_variance.md`).
