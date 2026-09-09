# Sharp-weighted late-week follow (MKT-15 follow-up) -- predeclaration

Written **before** any of the four arms below was computed. Nothing in this
file is a result; every number that appears here is either a frozen constant
of an already-served rule or a threshold chosen before the signs were seen.
The measured table lives in the report and in the experiment artifact under
`artifacts/sharp_weighted_follow/<UTC stamp>/`.

## The question

`docs/book_leadership.md` (SKY-04) is descriptive: Bovada, William Hill (US)
and MyBookie lead roughly three-fifths of their own line moves and the retail
books follow at 0.30-0.45. The served late-week refresh rule
(`src/nfl_ats/pick_refresh.py` `LATE_WEEK_MOVE_FOLLOW_POLICY`,
`src/nfl_ats/sharp_book_movement_features.py` `late_week_follow_frame`,
`docs/late_week_refresh.md`) throws that leadership away: it takes the
**equal-book mean** of every eligible book's Wednesday-to-deadline net spread
move and follows it against the Tuesday pick at 0.5 points.

MKT-15 asks whether the leadership structure is worth anything on the played
card: does listening to the three leading books -- only, or louder, or
earlier -- beat listening to all twelve equally?

## What is held identical across every arm

Everything except the aggregation of the per-book net moves.

* **Population.** The 2023-2025 hourly intraday archive, exactly the
  population the served rule's own evidence was measured on: the frozen quote
  and kickoff caches
  `artifacts/experiments/sharp_book_movement/quotes.parquet`
  (sha256 `b7d24760737ece78161b1962c80975e08a27fe6b02ed0fed590631e7485213d4`)
  and `kickoff.parquet`
  (sha256 `6630bb802216014b814aef55b744e08a483233b4a5a736a854c96939f927b68a`),
  816 archive games, 272 per season.
* **Quote loading, time guards, window and deadline logic.**
  `nfl_ats.sharp_book_movement_features.sharp_book_movement_features` called
  verbatim -- the same function `late_week_follow_frame` calls. Cutoff is
  `min(kickoff, that week's Sunday 16:00 ET)`; observations are kept only
  inside `[Monday, Sunday)` and strictly before the cutoff, with
  `bookmaker_last_update_utc <= observed_at_utc`; per book the line series is
  differenced over the whole window and only increments observed at or after
  that week's Wednesday are summed. That per-book Wednesday-to-deadline net
  move is the single quantity every arm below aggregates differently.
* **Book universe.** The frozen twelve: Bovada, William Hill (US), MyBookie,
  DraftKings, BetUS, BetRivers, PointsBet, Fanatics, LowVig, FanDuel, BetMGM,
  BetOnline. Leaders are the three `LEADER_BOOKS`: `bovada`,
  `williamhill_us`, `mybookieag`.
* **Threshold.** 0.5 points, the served rule's frozen threshold, for every
  arm. No threshold grid is searched here.
* **Side convention.** `net > 0` picks HOME, otherwise AWAY -- reused verbatim
  from `refresh_pick`. A net of exactly 0 cannot fire, since firing needs
  `|net| >= 0.5`.
* **Fallback.** An arm that does not fire on a game keeps the Tuesday pick for
  that game. A game with no usable quote evidence keeps the Tuesday pick under
  every arm.
* **Baseline card.** See "Which Tuesday card" below.
* **Grading.** Opener grading, `margin_vs_open`, pushes unscored -- the
  project's declared primary goal and the grading the served rule's +1.752 was
  measured at.

## Which Tuesday card the arms are scored against

The task asks for the Tuesday **nine-member** composed card
(`nfl_ats.four_overlay_composition.POLICY_ID`,
`overlay_union_coach_division_arrests_bye_coldvisitor_protection_interim_tank_precip_v3`).

**The archive harness cannot build that card for 2023-2025, so it is not
used.** Five of the nine members are functions of point-in-time inputs this
archive does not carry and cannot honestly reconstruct: the cold-visitor and
precipitation tilts read a Tuesday-noon and a kickoff-nearest weather forecast
fetched live (`fetch_tuesday_noon_forecast_temps_fail_open`,
`fetch_kickoff_nearest_forecasts_fail_open`); the arrest back-side member
reads a live arrest snapshot with a maximum age (`MAX_SNAPSHOT_AGE`); the
protection-mismatch and interim-head-coach tilts read point-in-time state
flags. Replaying them off today's tables would backfill information dated
after the Tuesday decision instant into a Tuesday decision -- exactly the
leakage this repository's pregame rule forbids.

**Every arm is therefore scored against the same baseline the served rule's
own +1.752 was measured against**: the raw production opener card,
`pick_home_at_open_probability_rule` / `correct_at_open_probability_rule` from
`nfl_ats.clv.opener_pick_evaluation`. One baseline, all five arms, so the
arm-vs-arm and arm-vs-S4 comparisons are exact; and S4 is directly comparable
to the number the served rule was promoted on.

## The four arms plus the positive control

Let `n_b(g)` be book `b`'s Wednesday-to-deadline net move on game `g`, as
defined above, over the books that actually contributed an increment.

| Arm | Aggregate | Fires when | Side |
|---|---|---|---|
| **S4** (served, replay) | `equal(g) = mean_b n_b(g)` over all contributing books | `abs(equal) >= 0.5` | `equal > 0` -> HOME |
| **S1** leader-only | `lead(g) = median n_b(g)` over the three leader books that contributed | `abs(lead) >= 0.5` | `lead > 0` -> HOME |
| **S2** leader-weighted | `lw(g) = sum_b w_b n_b(g) / sum_b w_b`, `w = 2` for the three leaders, `1` for the other nine | `abs(lw) >= 0.5` | `lw > 0` -> HOME |
| **S3** leader-first timing | fires on the leaders when they have moved, otherwise on the equal-book aggregate | `abs(lead) >= 0.5`, else `abs(equal) >= 0.5` | leader side when the leader clause fires, else equal side |
| **PC** positive control | perfect foresight | every game with at least one contributing book | the side that actually covered |

Predeclared tie-breaks and edge cases, fixed here before the signs were seen:

* **S3 precedence is leader-first, not equal-first.** When the leaders and the
  equal-book aggregate both fire and point at opposite sides, the leaders win.
  That is the whole point of the arm -- it is a timing claim ("the leaders got
  there first"), so deferring to the equal book when the leaders have already
  spoken would test nothing.
* **S1 with no leader-book evidence does not fall through to the equal-book
  aggregate.** It keeps the Tuesday pick. S1 is the strict "listen only to the
  leaders" arm; a fall-through would silently make it S3.
* **S2 uses the whole twelve-book universe** with a 2:1 weight, deliberately
  distinct from the already-measured `leader_net_move`, which weights all
  twelve books by their continuous leadership scores (`LEADERSHIP_WEIGHTS`).
  That continuous-weight arm was measured on 2026-09-05 at +1.627 points and
  is not re-run here.
* **The positive control is the ceiling of the reachable population**, not of
  the whole slate: it switches only where a follow rule could switch (at least
  one contributing book). It exists to prove the instrument -- 799 paired
  games, week-blocked -- can detect an effect at all, and to bound how much
  accuracy any aggregation of this archive's movement could possibly buy.

## Statistics, declared before the numbers

* Paired deltas against the same baseline, in accuracy points, game-weighted.
* **Week-blocked** bootstrap (blocks are `(season, week)`) and
  **season-blocked** bootstrap (blocks are seasons), 20,000 resamples,
  seed 20260821, for every cell. Within-week correlation is zero by owner
  mandate, so the week block is the honest unit; the season block is reported
  because three blocks is a deliberately brutal read and it is worth seeing.
* `probability_positive` is reported for every cell, computed as
  `P(delta > 0) + 0.5 * P(delta == 0)` via
  `nfl_ats.evidence_conventions.probability_positive_from_draws`.
* Per-season point estimates for every arm.
* Switch counts per arm, and the count of games where an arm's served side
  differs from S4's served side.
* The head-to-head arm-vs-S4 paired delta (same games, same blocks) is the
  decision quantity, reported with its own interval and
  `probability_positive`.

## The decision rule, declared before the numbers

The pool is forced picks. **The arm with the higher expected accuracy at the
deadline is the one to serve.** A 0.90-style threshold governs only what a
document may CLAIM; it never governs which card is played. If an arm beats S4
on the paired arm-vs-S4 delta, this document reports the exact function and
policy constant a promotion would touch -- and nothing is wired in this lane.

## Closing grounds (verbatim, binding)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (a) refuted mechanism -- a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (b) bounded by a
positive control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it, report `probability_positive`, never
"contains zero".

Every cell in this lane is recorded through `nfl-ats weak-signals record`
under names `sharp_weighted_follow_<arm>_<window>`, units `accuracy_points`,
league `nfl`, family `sharp_weighted_follow`. The family is declared here,
before the signs were seen, and its members share one window and one baseline,
so they are correlated readings of one question rather than independent votes.
