# Best Pick deadline renomination (LEAD-53 historical read)

**Question.** The pool scores one Best Pick per week separately. The served
nominator (`nfl_ats.best_pick_nomination.nominate_v2`) chooses that game once,
at the Tuesday lock, and never revisits it — while every ordinary pick stays
editable until `min(kickoff, Sunday 16:00 ET)` (`docs/late_week_refresh.md`).
**Does re-nominating the Best Pick at the pick deadline beat the Tuesday
nominee?**

A registered challenger for the live version of this already exists —
`best_pick_sunday_renomination` in `artifacts/prospective/challengers.json`,
recorded by `nfl_ats.best_pick_refresh_prospective` — with **no historical
evidence attached**. This document is the predeclaration and result for the
historical read.

## Predeclaration (written before any number in section "Results" was computed)

### Binding closing-grounds taxonomy, verbatim

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (a) refuted mechanism — a RESOLVED wrong sign (whole interval on
the wrong side of zero) or zero split-half reliability; (b) bounded by a
positive control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never "contains zero". The registry hard-rejects
inadmissible closures; if a record command errors, the verdict is wrong, not
the validator. Within-week game correlation is ZERO by owner mandate:
week-blocked bootstrap only. Decide on expected value — forced picks, so P+
above 0.5 on the played card is played; thresholds like 0.90 govern only what
docs may claim.

### Population

The 2023-2025 intersection of two frozen archives, no new fitting:

- **Nomination inputs** — `artifacts/ridge_alpha_promotion/20260818T221459Z/`
  (`opener_paired.parquet`, `opener_baseline.parquet`), the same 1,537-game /
  107-week paired opener archive `docs/opener_evaluation.md` and
  `scripts/best_pick_composed_rule_eval.py` use, restricted to 2023-2025.
  Supplies `candidate_dist` (the alpha=2000 `market_residual` probability's
  distance from 0.5), the played side (`baseline_pick_home`), the opener
  settlement (`margin_vs_open`, `baseline_correct_open`) and the frozen
  Tuesday line.
- **Tuesday cross-book dispersion** —
  `artifacts/odds_microstructure/20260818T225430Z/spread_novig_tue_open.parquet`'s
  `spread_std`, fed through production's own
  `nfl_ats.best_pick_nomination.dispersion_pool_from_frame` per week (the
  cross-check `scripts/best_pick_composed_rule_eval.py::build_archive_frame`
  already performs).
- **Late-week market** — the `intraday_hourly` decision archive for 2023-2025
  (`artifacts/experiments/sharp_book_movement/quotes.parquet` +
  `kickoff.parquet`, the exact cached load
  `scripts/observed_movement_channel.py::_load_intraday_with_kickoff` produces
  and `scripts/sharp_book_movement_on_production.py` consumed for the served
  MKT-15/CX18 promotion).

### The renomination instant

One instant per week: **Sunday 12:00 ET** of that week, derived from the
week's own first kickoff with the same Tue..Mon anchor
`nfl_ats.sharp_book_movement_features.sharp_book_movement_features` uses. It
is the latest moment at which every ordinary Sunday game is still open (the
earliest Sunday kickoff this project schedules against is 13:00 ET) and it
matches the window `nfl_ats.best_pick_refresh_prospective.record_best_pick_refresh`
already enforces for the live challenger (Sunday, local hour < 12).

A game is **playable** at that instant iff its own kickoff is still in the
future. Every market input is cut at `min(kickoff, instant)`; the follow
rule's own Wednesday-to-Saturday window is unchanged.

**Frozen-nominee rule, read from `best_pick_refresh_prospective.py:180-182`:**
if the Tuesday nominee's own deadline has already passed at the renomination
instant (a Thursday game), every arm keeps the Tuesday nominee — a Best Pick
on an already-played game cannot be withdrawn.

### Arms (one nomination per week, all graded at the frozen Tuesday opener)

- **A0 — Tuesday v2 nominee (incumbent).** `select_nominee` on the Tuesday
  below-median-`spread_std` pool, exactly as played.
- **A1 — deadline re-rank.** The same v2 ranking rerun at the instant, over
  the still-playable games, with the pool and the dispersion tie-break built
  from the **deadline's** cross-book `spread_std` (last quote per book before
  `min(kickoff, instant)`), through the same production
  `dispersion_pool_from_frame` / `select_nominee`.
- **A2 — flip-avoidant.** A0 unless the served late-week follow rule
  (`late_week_follow_frame`, equal-book Wednesday-to-deadline net move, 0.5
  threshold) has flipped that game's pick; then the next-ranked still-playable
  member of the Tuesday pool whose pick did not flip (obtained by re-calling
  production's `select_nominee` after removing each flipped choice). If every
  pool member flipped, A0.
- **A3 — biggest favourable move.** Among still-playable Tuesday-pool games,
  the one whose equal-book net move is at least 0.5 **toward** our Tuesday
  pick, largest magnitude first, ties on ascending `game_id`; else A0.
- **Positive control — perfect-foresight renomination.** Among still-playable
  Tuesday-pool games, any game whose graded pick is correct (`select_nominee`
  order among the correct ones); else A0.

### Grading

- **PRIMARY side** = the served late-week pick at the instant: the Tuesday
  played side, overridden by the follow rule when `|equal_net_move| >= 0.5`.
  This is what the card actually submits, so it holds the *side* rule constant
  across arms and isolates the *nomination* question.
- **SECONDARY side** = the frozen Tuesday side, no late-week override.
- Correctness is settled against `margin_vs_open` at the frozen Tuesday line;
  a push is NaN for every arm and drops that week from the paired cell.

### Metric

Best Pick correctness per week (one pick per week, 0/1), paired against A0,
week-blocked bootstrap (`nfl_ats.clv.week_blocked_bootstrap`, blocks
`(season, week)`), **20,000 samples, seed 20260821**, reported as
`probability_positive` plus the interval, in accuracy points. Raw counts
(weeks scored, weeks won, weeks the nominee differs) are reported alongside
because n is ~50 weeks and the interval is necessarily wide.

### Declared limitation, stated before the run

`nfl_ats.pick_refresh.plan_refresh` recomputes the model probability from
**current** features. The historical feature table is a single as-of build, so
a point-in-time deadline refit reproduces the Tuesday probability exactly:
`fit_margin_models_for_week` trains on games strictly before the week's first
kickoff at both instants and reads the same feature rows. Historically,
therefore, the only quantity that genuinely moves between Tuesday and the
deadline is the **market** — the dispersion pool (A1) and the late-week move
(A2, A3). A corollary follows and is stated in advance rather than presented
as a finding: the registered `best_pick_sunday_renomination` challenger, which
reuses the frozen Tuesday pool and only refreshes probabilities, is a
**mathematical no-op on a static feature table** and can only diverge in
production, where the feature table is rebuilt mid-week.

A second declared caveat: the Tuesday `spread_std` comes from the
decision-labeled `tue_open` snapshot store and the deadline `spread_std` from
the `intraday_hourly` archive; the two need not carry the same book universe.
Book counts for both are reported.

## Results

All measured 2026-09-09, command
`.\.tools\uv.exe run --no-sync python scripts/best_pick_deadline_renomination_eval.py --repo F:/Repos/nfl_py3`,
artifact `artifacts/best_pick_deadline_renomination/20260909T214500Z/`.

**Decision: keep the Tuesday nominee. No arm beats it on expected value.**
The incumbent A0 is the highest-scoring arm on both grades; the best
alternative is a dead heat that changes the nomination once in 51 weeks, and
the two arms that actually renominate both lean against the incumbent.

Population: 816 games, 54 weeks, 2023-2025, 17 pushes; 51 weeks where both
nominees resolved. The Tuesday nominee's own kickoff had already passed at
Sunday noon in 10 of 54 weeks (Thursday, Black-Friday, Saturday and
London/Munich/Berlin 09:30 ET games), and those weeks are frozen for every
arm. Zero quote rows were refused by the anti-backdating guard.

### Arms, paired against A0 at the frozen Tuesday opener

PRIMARY grade (served late-week pick side), 51 paired weeks:

| Arm | Weeks won | Accuracy | Effect (acc. pts) | 95% week-blocked | `probability_positive` | Weeks nominee differs |
|---|---:|---:|---:|---|---:|---:|
| **A0 Tuesday v2 (incumbent)** | 33/51 | 64.71% | — | — | — | — |
| A1 deadline re-rank | 30/51 | 58.82% | **-5.88** | [-17.65, +5.88] | **0.182** | 26 |
| A2 flip-avoidant | 33/51 | 64.71% | **0.00** | [0.00, 0.00] | **0.500** | 1 |
| A3 biggest favourable move | 28/51 | 54.90% | **-9.80** | [-23.53, +3.92] | **0.077** | 25 |
| Positive control (perfect foresight) | 46/51 | 90.20% | **+25.49** | [+13.73, +37.25] | **1.000** | 13 |

SECONDARY grade (frozen Tuesday pick side), same 51 weeks:

| Arm | Weeks won | Accuracy | Effect (acc. pts) | 95% week-blocked | `probability_positive` |
|---|---:|---:|---:|---|---:|
| **A0 Tuesday v2 (incumbent)** | 32/51 | 62.75% | — | — | — |
| A1 deadline re-rank | 29/51 | 56.86% | -5.88 | [-17.65, +5.88] | 0.159 |
| A2 flip-avoidant | 31/51 | 60.78% | -1.96 | [-5.88, 0.00] | 0.180 |
| A3 biggest favourable move | 26/51 | 50.98% | -11.76 | [-25.49, +1.96] | 0.049 |
| Positive control | 42/51 | 82.35% | +19.61 | [+9.80, +31.37] | 1.000 |

For scale: the same population's ordinary game-level opener accuracy is
**53.32%** on 799 non-push games (measured, `opener_paired.parquet` restricted
to 2023-2025), against the incumbent nominator's 64.71% on its 51 Best Picks.
That gap is on the same population family the v2 rule was selected on, so it
carries the standing look-reuse discount and is not a fresh confirmation.

### Nothing here closes, and why

Every cell is recorded `unresolved_below_power`
(`registry/weak_signals.json`, family `best_pick_deadline_renomination`;
the control under `..._control`). Neither terminal ground is available:

- **`wrong_sign_resolved`** — no interval sits wholly on the wrong side of
  zero. A3's secondary cell comes closest at [-25.49, +1.96] and still
  crosses.
- **`bounded_by_control`** — the positive control resolves cleanly at +25.49
  points, `probability_positive` 1.000, so the harness demonstrably sees a
  ~25-point renomination effect on 51 weeks. It was **not** shown able to see a
  1-2 point one, so it bounds nothing at the scale these arms plausibly live
  at. One nomination a week is 51 observations; at that n the interval is
  ±12 points wide and this instrument cannot resolve a small renomination
  edge either way.

### Why the arms behave the way they do (measured diagnostics)

- **A2 is nearly inert by construction.** The served late-week follow rule
  flips 18.17% of all games and 16.31% of Tuesday-pool games, but it flipped
  the **Best Pick nominee's own pick in only 3 of 54 weeks, and in only 1 of
  the 44 weeks where the nominee was still playable**. The mechanism is the
  nominator itself: v2 restricts to games the books agree on, and those games
  move less — mean `|equal_net_move|` is 0.599 across all games, 0.521 across
  Tuesday-pool games, and **0.485 on the nominee**, below the rule's own 0.5
  threshold. A rule that only fires when the flagged game flips has almost
  nothing to fire on.
- **A1 renominates constantly because deadline dispersion is a different
  ranking, not a refreshed one.** The median cross-book `spread_std` barely
  moves across the week (Tuesday 0.218, deadline 0.224, ~12.1 books
  contributing at the deadline), but the mean within-week correlation between
  a game's Tuesday dispersion and its deadline dispersion is only **0.255**.
  So the below-median pool re-forms around largely different games (mean pool
  7.48 games Tuesday, 6.18 at the deadline over ~13.2 playable games), the
  argmax of `candidate_dist` lands elsewhere in 26 of 51 weeks, and the
  incumbent's advantage is thrown away.
- **A3 is the weakest arm and the most interesting one.** Choosing the game
  the market moved furthest *toward* our pick is a different objective from
  choosing the game the model is surest about, and on this window it scores
  54.90% against the incumbent's 64.71%. Read as evidence about the follow
  rule: a favourable late-week move is a reason to keep a pick, not a reason
  to promote it to Best Pick.

### The registered challenger, and what would actually settle this

`best_pick_sunday_renomination` reuses the **frozen Tuesday pool** and only
refreshes the probabilities. On a static feature table the refreshed
probability equals the Tuesday probability, so its nominee equals A0 in every
historical week by construction — a mathematical no-op, declared before the
run, not a measured finding. Its entire live content is the half this document
cannot reach: a mid-week feature rebuild (Wednesday-to-Saturday injury
designations reaching `game_features_weak_stack.parquet`) genuinely moving
`candidate_dist`. That is what 2026's prospective pairs will measure, and
nothing here should be read as evidence about it.

If the owner ever wants to serve a renomination anyway, the wiring is:
`nfl_ats.best_pick_refresh_prospective.record_best_pick_refresh` already
computes the Sunday-morning nominee on the Sunday `refresh-picks` pass
(`docs/late_week_refresh.md`'s `sunday_morning_final`); serving it would mean
having `nfl_ats.publishing` read that arm instead of the frozen Tuesday
`is_best_pick`. **Not wired, and on this evidence it should not be** — the
best alternative arm is a dead heat and the two that move are 0.18 and 0.08
likely to help.

