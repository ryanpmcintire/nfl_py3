# Sunday-morning Best Pick re-nomination (LEAD-53 historical read)

**Question.** The pool pays one Best Pick per week. The served nominator
(`nfl_ats.best_pick_nomination.nominate_v2`) chooses that game once, at the
Tuesday lock, and never revisits it, while every ordinary pick stays editable
until `min(kickoff, Sunday 16:00 ET)`. **Does re-nominating the Best Pick on
Sunday morning, from the refreshed probabilities, beat the Tuesday nominee?**

A live challenger for this already exists — `best_pick_sunday_renomination`
in `artifacts/prospective/challengers.json`, paired each week by
`nfl_ats.best_pick_refresh_prospective` — with no historical evidence attached.
This document is the predeclaration and result for the historical read.

It is the second historical read of LEAD-53. The first,
`docs/best_pick_deadline_renomination.md` (2026-09-09), scored four arms on
2023-2025 that renominated by changing the **dispersion pool** or by following
the late-week market move, and left the probability itself untouched. This read
does the other half: it moves the **probability**, and it runs on twice the
population (2020-2025).

## Predeclaration

Everything in this section was written before any arm was scored.

### Binding closing-grounds taxonomy, verbatim

An interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (1) refuted mechanism — a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (2) bounded by a positive
control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". The registry code
hard-rejects inadmissible closures; if a record command errors, the verdict is
wrong, not the validator.

Decide on expected value: `probability_positive` above 0.5 favours playing it;
0.90 and 95% govern only what the docs may claim, never which card is played.
Within-week game correlation is zero by owner mandate, and the unit here is one
Best Pick per week, so the block is the week. Football margins are discrete and
multimodal: every cover probability below is read off the model's own empirical
residual sample, never a fitted normal.

### Predeclared direction

The refreshed nomination BEATS the Tuesday nomination (the ROADMAP LEAD-53 row
states this direction).

### Population

The 107 graded weeks / 1,537 games of the frozen paired opener archive
`artifacts/ridge_alpha_promotion/20260818T221459Z/opener_paired.parquet`
(2020-2025) — the same archive `docs/opener_evaluation.md` and POL-09's
composed-rule read use — intersected with a pool-cycle Sunday cross-book line
from `data/market/raw`, read through production's own `load_quote_history` and
`spread_consensus`.

This is the fifth analytical look at the 107-week opener population and the
second on the LEAD-53 question, so it carries the standing reuse discount:
nothing here is a fresh confirmation of the nominator itself, only a comparison
between nomination instants on a shared window.

### The renomination instant

**Pool-cycle Sunday 12:30 ET**, derived per week as
`nfl_ats.pick_refresh.sunday_pick_lock(kickoffs) - 3h30m`.

The reason is an archive fact, measured before the run and not chosen from
outcomes: the only pre-kickoff Sunday NFL capture that exists in every season
2020-2025 is the 12:25 ET snapshot. Hourly Sunday captures (00:55 ET through
10:55 ET) begin in 2023 and do not exist for 2020-2022. A 12:30 ET instant is
30 minutes before the earliest Sunday kickoff this project schedules against
(13:00 ET), so every Sunday-afternoon game is still open, and it is the latest
instant with coverage in all six seasons.

The live challenger's own window is stricter — Sunday, local hour below 12
(`best_pick_refresh_prospective.record_best_pick_refresh`). The 30-minute gap
is a limitation of the archive, not a design preference, and it is checked: a
**sensitivity arm at an 11:00 ET instant on 2023-2025 only** is reported
alongside.

A game is **playable** at the instant iff its own kickoff is strictly after it.
Sunday quotes are restricted to observations on the pool-cycle Sunday itself,
at or before the instant and strictly before the game's own kickoff; no
Saturday quote is allowed to stand in for a missing Sunday one.

**Frozen-nominee rule**, mirroring `best_pick_refresh_prospective.py`: if the
Tuesday nominee's own deadline (`min(kickoff, Sunday 16:00 ET)`) has already
passed at the instant — a Thursday, Friday, Saturday or 09:30 ET international
game — every arm keeps the Tuesday nominee. A Best Pick on an already-played
game cannot be withdrawn.

### Arms

One nomination per week. Every arm is graded identically, on the pool's Best
Pick rule: the **frozen Tuesday opener line** (`tue_open_home_spread`) and the
**frozen played side** (`baseline_pick_home`), settled by
`baseline_correct_open`. Holding the side and the grading line fixed is what
isolates the *nomination* question from the side question.

All probabilities come from production's own nomination candidate —
`nfl_ats.best_pick_nomination.fit_candidate_probabilities`, ridge `alpha=2000`,
`weak_stack`, `market_residual`, `min_train_games=500`, trained per week on
games strictly before that week — with the line in `spread_line` overridden to
the instant's line. Both the Tuesday and the Sunday arm are recomputed here
from the same estimator so they are commensurable; the recomputed Tuesday
probability is tied out against the archive's frozen `candidate_prob_open`.

- **T0 — Tuesday nomination (incumbent).** Candidate probability at the frozen
  Tuesday opener line; pool = games strictly below the week's median Tuesday
  cross-book opener dispersion (production `week_dispersion_pool`); nominee =
  the largest `|p - 0.5|`. This is the served rule.
- **T0b — Tuesday, frozen-side functional (control for the functional).** Same
  instant, same pool, but ranked by the probability the **frozen played side
  covers** at the frozen Tuesday line rather than by `|p - 0.5|`. T0b exists so
  that S2 minus T0 can be decomposed: T0b minus T0 is the functional, S2 minus
  T0b is the instant.
- **S1 — literal Sunday re-rank.** The same candidate model re-predicted with
  the Sunday-morning cross-book consensus line in `spread_line`, ranked by
  `|p_sunday - 0.5|` at that Sunday line. Tuesday pool, playable games only.
  This is the arm the live challenger's wording describes most literally.
- **S2 — Sunday re-rank at the frozen grading line (headline).** The Sunday-line
  predicted margin, evaluated against the **frozen Tuesday opener line** on the
  model's own empirical residual sample, ranked by the probability the frozen
  played side covers. This is the refreshed probability of the bet the pool
  actually holds: late market information enters the margin forecast, the
  grading line stays where the pool froze it. Tuesday pool, playable only.
- **S3 — full Sunday refresh.** S2's ranking over a **Sunday-morning**
  dispersion pool (games strictly below the week's median Sunday cross-book
  `spread_std`, through the same production `dispersion_pool_from_frame`),
  playable only.
- **PC — positive control, perfect foresight.** Among still-playable Tuesday-pool
  games, a game whose graded pick is correct, chosen under the same tie rule.

### Tie rule, predeclared, never alphabetical

The ranking statistic is a smoothed count over a finite residual sample, so
exact ties are common and must not be resolved by name. Ties are broken
lexicographically, identically for every arm:

1. Higher ranking statistic.
2. Lower cross-book dispersion (`spread_std`) at that arm's own instant —
   production's own tie-break in `select_nominee`. A missing dispersion ranks
   last; if every tied game's dispersion is missing or equal, this level does
   not discriminate.
3. Smaller absolute frozen Tuesday line. Ground: the served nominator already
   screens toward small spreads (`SERVED_SPREAD_THRESHOLD = 7.0`) and the model
   is measured weaker on large ones.
4. Earlier kickoff — the game whose information is most complete at the instant
   and whose deadline binds first.
5. Still tied: the week's score for that arm is the **mean of the tied games'
   outcomes** — the expectation over a uniform random tie-break — never one
   game chosen by name. POL-09's re-read showed that an alphabetical tie-break
   on this exact population manufactured a 7.8-point illusion; this rule
   removes the coin flip from the measurement instead of taking its variance.

T0 is additionally reported exactly as production serves it (dispersion, then
ascending `game_id`) purely as a tie-out against POL-09. The paired comparison
uses the predeclared rule on both arms.

### Metric and uncertainty

Best Pick correctness per week (one pick per week; fractional when the tie rule
reaches level 5), paired against T0 on weeks where both arms resolved, with a
push leaving that arm's week unresolved. Uncertainty from
`nfl_ats.clv.week_blocked_bootstrap(block="week")`, 20,000 samples, seed
20260910, reported as the effect in accuracy points, the 95% interval and
`probability_positive`. Raw counts (weeks scored, weeks won, weeks the nominee
differs, weeks the outcome differs) are reported alongside, because ~100 weeks
of a single pick is a wide instrument and the interval will be wide.

### Declared limitations, stated before the run

- **The historical feature table is a single as-of build.** Production's
  `plan_refresh` re-anchors the line to the frozen Tuesday spread and recomputes
  from the *current* feature table, so the half of the live refresh that comes
  from a mid-week feature rebuild (Wednesday-to-Saturday injury designations
  landing in `game_features_weak_stack.parquet`) is invisible here: at both
  instants the historical table carries the same, final, values. Everything
  measured below is therefore the **market** half of the refresh. The live
  challenger's feature half is not tested by this document.
- **The played side is the frozen archive's baseline arm**, recorded
  2026-08-18, not today's active model. That keeps the grading commensurable
  with POL-09 and with the 2026-09-09 read, at the cost of not being the card
  the site publishes today.
- **The functional difference is real and is measured, not assumed.** T0 ranks
  by `|p - 0.5|`, which is confidence in the *nominator's* side; S2 ranks by
  confidence in the *played* side. T0b separates the two.

## Results

All measured 2026-09-10 with
`.\.tools\uv.exe run --no-sync python scripts/best_pick_sunday_renomination_eval.py`;
artifact `artifacts/best_pick_sunday_renomination/20260911T021349Z/`. The run is
deterministic and was reproduced twice.

**Decision: on expected value, move the star to the Sunday-morning refresh.**
The full Sunday refresh (S3) is `probability_positive` **0.814** better than the
frozen Tuesday nominee and the refreshed-probability arm alone (S2) is
**0.742**. The pool is forced picks — a star is submitted every week either
way — so declining an arm that is 81% likely better is taking the 19/81 side of
the bet, not being careful. Nothing here is resolved, and nothing here needs to
be resolved to make that call.

### Population

107 weeks, 1,537 games, 2020-2025. Zero weeks were dropped: every week had a
pool-cycle Sunday cross-book line, at a median of **14 books**, for every
still-playable game. In **17 of 107 weeks** the Tuesday nominee's own deadline
had already passed at the instant (Thursday, Friday, Saturday and 09:30 ET
international games), and those weeks are frozen for every arm. The mean
absolute Tuesday-to-Sunday line move is **0.964 points**.

The recomputed Tuesday candidate probability ties out against the frozen
archive's `candidate_prob_open`: median within-week correlation **0.9969**
(minimum 0.7789 in one week). Under the predeclared tie rule, T0 selects the
same game production's served `select_nominee` does in **105 of 107 weeks**.

### Arms, paired against T0 at the frozen Tuesday opener

| Arm | Weeks | Best Pick accuracy | Effect (acc. pts) | 95% week-blocked | `probability_positive` | Nominee differs | Outcome differs (arm-base) |
|---|---:|---:|---:|---|---:|---:|---:|
| **T0 Tuesday nomination (incumbent)** | 104 | 52.88% (55) | — | — | — | — | — |
| T0b Tuesday, frozen-side functional | 104 | 52.88% (55) | **0.00** | [-2.88, +2.88] | **0.503** | 2 | 2 (1-1) |
| S1 literal Sunday re-rank | 104 | 51.92% (54) | **-0.96** | [-4.81, +2.88] | **0.334** | 6 | 5 (2-3) |
| **S2 Sunday re-rank at frozen line** | 103 | 55.34% (57) | **+2.91** | [-5.83, +11.65] | **0.742** | 40 | 21 (12-9) |
| **S3 full Sunday refresh** | 103 | 57.28% (59) | **+4.85** | [-5.83, +15.53] | **0.814** | 70 | 31 (18-13) |
| S2 vs T0b (instant alone) | 103 | 55.34% | **+2.91** | [-5.83, +11.65] | **0.741** | 39 | 21 (12-9) |
| Positive control (perfect foresight) | 104 | 87.50% | **+34.62** | [+25.96, +44.23] | **1.000** | 72 | 36 (36-0) |

Accuracies in the table are on each cell's own paired weeks; a push leaves that
arm's week unresolved, which is why the denominators differ by one or two.

### Per-season stability

| Season | Weeks | T0 | S1 | S2 | S3 | Control | S2 − T0 (pts) | S3 − T0 (pts) | S2 nominee differs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2020 | 17 | 64.7% | 64.7% | 58.8% | 64.7% | 94.1% | **-5.88** | 0.00 | 7 |
| 2021 | 18 | 50.0% | 50.0% | 58.8% | 66.7% | 83.3% | **+11.76** | +16.67 | 7 |
| 2022 | 18 | 47.1% | 35.3% | 47.1% | 52.9% | 77.8% | 0.00 | +5.88 | 4 |
| 2023 | 18 | 47.1% | 47.1% | 64.7% | 55.6% | 94.4% | **+17.65** | +11.76 | 7 |
| 2024 | 18 | 52.9% | 58.8% | 41.2% | 47.1% | 88.9% | **-11.76** | -5.88 | 6 |
| 2025 | 18 | 55.6% | 55.6% | 61.1% | 52.9% | 88.9% | **+5.56** | 0.00 | 9 |

S2 is positive in three seasons, flat in one and negative in two; S3 is positive
in three, flat in two and negative in one. A split-half read on the same weeks:
S2 scores **-4.00 points on odd-numbered weeks and +9.43 on even-numbered**
(the sign does not survive that split) but **+1.96 on 2020-2022 and +3.85 on
2023-2025** (it does survive the era split). S3 is the steadier of the two on
both splits: **+4.00 odd / +5.66 even** and **+7.69 on 2020-2022 / +1.96 on
2023-2025**. No `--reliability` was recorded for these cells: the quantity is a
per-week nomination delta, not a per-game trait with a split-half, and reporting
the season-half and week-parity consistency above is the honest substitute.

### Sensitivity to the instant (2023-2025 only, where hourly captures exist)

| Instant | Weeks | S2 effect | `probability_positive` | S3 effect | `probability_positive` |
|---|---:|---:|---:|---:|---:|
| 11:00 ET (the live challenger's window) | 52 / 51 | 0.00 [-13.46, +13.46] | **0.508** | +3.92 [-11.77, +19.61] | **0.679** |
| 12:30 ET (this document's instant) | 52 / 51 | +3.85 [-9.62, +17.31] | **0.708** | +1.96 [-13.73, +17.65] | **0.603** |

On 52 weeks neither instant separates from the other — every reading sits at or
above a dead heat, and the two arms swap order between them. The one thing the
table does say is that the 90 minutes between 11:00 and 12:30 ET is where S2's
lean on this subset comes from, which is consistent with the mechanism below.
The live challenger's pre-noon window is therefore not obviously the right one,
and the question of the instant is itself unresolved.

### The mechanism, measured

**The refreshed margin forecast tracks the Sunday line almost exactly one for
one.** The median per-week slope of (Sunday-line predicted margin − Tuesday-line
predicted margin) against the line move itself is **1.0067**. The model's
`market_residual` arm carries `spread_line` as a feature but barely leans on it,
so the refreshed cover probability *at the refreshed line* is nearly invariant:
across all 1,537 games the largest Tuesday-to-Sunday change in `|p − 0.5|` at
each arm's own line is **0.0066**, whereas at the frozen grading line it is
**0.2768**.

Three consequences, all of which the arms show:

- **S1 is a measured near-no-op, not an assumed one.** Re-predicting at the
  Sunday line and re-ranking by `|p − 0.5|` changes the nominee in 6 of 104
  weeks and scores -0.96 points. The 2026-09-09 read predicted this
  analytically; it is now measured.
- **S2 and S3 are the market's Sunday move, re-scored against the line the pool
  froze.** That is the honest description: the model's own view of a game
  barely moves, so what re-ranks the card is how far the market walked toward or
  away from the side we already hold.
- **The functional is inert; the instant is everything.** T0b — the same Tuesday
  instant, ranked by the frozen side's cover probability instead of by
  `|p − 0.5|` — changes the nominee in 2 of 104 weeks and scores exactly 0.00
  (`probability_positive` 0.503). So the whole of S2's +2.91 is the move to
  Sunday, not the change of ranking statistic.

### The tension with the 2026-09-09 read, stated plainly

`docs/best_pick_deadline_renomination.md`'s A3 arm — nominate the Tuesday-pool
game with the biggest favourable late-week move — scored **-9.80 points,
`probability_positive` 0.077** on 2023-2025, and that document read it as
"a favourable late-week move is a reason to keep a pick, not to promote it".
S2 and S3 lean the other way on an overlapping mechanism. They are not the same
measurement — A3 ranks by the *magnitude of the move alone*, S2 ranks by the
*refreshed probability*, which is the Tuesday edge plus the move, and A3 was
graded on the served late-week follow side at a 12:00 ET instant on 51 weeks —
but the two readings do bear on the same idea and point opposite ways. On the
overlapping 2023-2025 subset at 12:30 ET, S2 is +3.85 at 0.708 and A3 was -9.80
at 0.077. Neither is resolved; both are ~50-week reads of a one-pick-per-week
instrument. Anyone acting on this should treat "how the late move should enter
the nomination" as open, with the composite (edge + move) currently ahead of the
move alone.

### Nothing here closes, and why

Every cell is recorded `unresolved_below_power` in `registry/weak_signals.json`
under family `best_pick_renomination` (the control under
`best_pick_renomination_control`). Neither terminal ground is available:

- **`wrong_sign_resolved`** — no arm's interval sits wholly on the wrong side of
  zero. S1's is the closest at [-4.81, +2.88] and still crosses.
- **`bounded_by_control`** — the positive control resolves cleanly at +34.62
  points, `probability_positive` 1.000, so the instrument demonstrably sees a
  ~35-point renomination effect on 104 weeks. It was **not** shown able to see a
  3-to-5 point one, so it bounds nothing at the scale these arms live at. One
  pick a week is 104 observations; the interval is ±9 points wide and this
  instrument cannot resolve a small renomination edge in either direction.

### What would have to change to serve it

`nfl_ats.publishing` reads the frozen Tuesday `is_best_pick` from the recorded
card. Serving S3 would mean having it read the Sunday-morning arm that
`nfl_ats.best_pick_refresh_prospective.record_best_pick_refresh` already
computes on the Sunday `refresh-picks` pass, with two changes that this document
did not test in production code: the arm must rank by the **frozen side's cover
probability at the frozen line** (not by `|p − 0.5|` at the current line, which
is the near-no-op S1), and the dispersion pool must be rebuilt from the
**Sunday** cross-book spread over the still-playable games only. Both are
one-line changes to the ranking inputs; neither is wired, and the live
challenger currently implements S1, the weakest arm measured here.

The live challenger also carries the half this document cannot reach: a mid-week
feature rebuild moving the probability on its own. Nothing above is evidence
about that.
