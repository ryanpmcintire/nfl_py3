# Sharp-book-weighted late-week movement (MKT-15)

**Status:** design frozen 2026-09-10 **before** any leadership score, any arm
and any accuracy delta was computed. Sections 1-7 are the predeclaration.
Section 8 holds the measured numbers, appended from the frozen script in one
pass. Files: this document, `scripts/sharp_book_weighted_movement.py`,
artifact `artifacts/sharp_book_weighted_movement/<UTC stamp>/`.

## 1. What is left of MKT-15 after the 2026-09-09 promotion

`docs/book_leadership.md` (SKY-04) measured, descriptively, that Bovada,
William Hill (US) and MyBookie post a new spread first in roughly three-fifths
of the line moves they take part in, while the big US retail books follow at
0.30-0.45. `docs/sharp_weighted_follow.md` then spent that table the crude
way: it took **hard membership** of the three top books and served their
**median** Wednesday-to-deadline net move as the late-week follow's aggregate
(`src/nfl_ats/sharp_book_movement_features.py` `LEADER_BOOKS`,
`leader_median_net_move`; `docs/late_week_refresh.md`, "Promoted late-week
follow"). That is what is played today, at a full-point gate below a 10.5-point
line and half a point at or above it.

Two things the promotion did not do, and this lane does:

1. **The leadership numbers it leans on are in-sample.** `LEADERSHIP_WEIGHTS`
   and `LEADER_BOOKS` were both estimated on 2023-2025 and are applied to
   2023-2025. Nothing in the served rule knows which seasons it is allowed to
   have seen. A rule that will be run forward needs weights that were
   estimable **before** the season they are applied to.
2. **Membership is a step function of a continuous statistic.** The third
   leader (0.6276) and the fourth book (0.5569) are 7 points apart on a scale
   that runs from 0.22 to 0.64; the served rule gives the first a full vote and
   the second none. A score-weighted aggregate spends the whole ordering.

So MKT-15's remaining question, stated once: **does a leadership score that
was estimable before the season, spent as a weight on every book, beat the
served hard-membership median of three books on the card that is played?**

## 2. The leadership score (definition frozen here)

A book leads when it gets to the new number before the consensus does. The
statistic is literally that, at one-capture resolution.

For a **pool** of seasons `P` (which seasons go in the pool is section 3), over
every NFL spread quote in `data/market/raw` belonging to a game of a season in
`P`, from the frozen twelve-book universe
(`sharp_book_movement_features.LEADERSHIP_WEIGHTS`'s keys):

1. Keep rows with a finite `home_spread_line`, `bookmaker_last_update_utc <=
   observed_at_utc` (the provider cannot have stamped an update later than the
   capture that carries it) and `observed_at_utc < commence_time_utc`.
   Deduplicate to one line per `(game, book, capture)`.
2. Per game, order the distinct capture instants `c_0 < c_1 < ... < c_n` and
   build `L[i, b]`, book `b`'s line at capture `c_i`, forward-filled within the
   game from its own last quote. Before a book's first quote its entry is
   missing.
3. For each book `b`, the **leave-one-out consensus** `M_b[i]` is the median of
   `L[i, .]` over the other eleven books present at `c_i`. Leave-one-out on
   purpose: a book that moves early drags a consensus that contains it, which
   would credit leadership to the arithmetic rather than to the book.
4. A **consensus move event** for `b` is an index `i >= 1` with `M_b[i] !=
   M_b[i-1]`, direction `d = sign(M_b[i] - M_b[i-1])`, where `L[i-1, b]` and
   `L[i, b]` both exist (the book is observable across the event). Every such
   event is one **participation** for `b`.
5. `b` **leads** that event when `d * (L[i-1, b] - M_b[i-1]) > 0`: at the
   capture **before** the consensus moved, `b` was already quoting a line
   displaced in the direction the consensus was about to take. That is
   "precedes the consensus by at least one capture", and nothing weaker counts
   — a book that arrives in the same capture as the consensus scores zero for
   that event.
6. `leadership(b) = leads / participations`.
7. A book with fewer than **200** participations in the pool is not estimated;
   it takes the participation-weighted mean score of the books that clear 200.
   That is a neutral prior, not a zero: an unmeasured book must not be silently
   muted, and must not be silently amplified either.

Declared limitations, before the numbers: provider clocks are trusted as given,
so a book whose feed stamps late reads as a leader; capture cadence bounds the
resolution, and the pre-2023 archive is far coarser than the 2023-2025 hourly
archive (measured in section 3); and "early" is not "right" — this statistic
never looks at an outcome.

## 3. Chronology: which seasons may set the weights

**The weights used to score season `S` are estimated only on seasons strictly
before `S`.** No exceptions, no pooled-across-everything variant is scored.

The scored population is 2023-2025 (section 4), so:

| Scored season | Leadership pool |
|---|---|
| 2023 | 2020, 2021, 2022 |
| 2024 | 2020, 2021, 2022, 2023 |
| 2025 | 2020, 2021, 2022, 2023, 2024 |

**Disclosed up front, because it is the weakest joint in this lane:** the
2020-2022 archive is not the hourly archive. It carries roughly 7 captures per
game against 2023-2025's ~325, so the one-capture-resolution statistic that
defines leadership is measured at a much coarser resolution there, and the
2023 weights are the coarsest of the three. Coarse resolution pushes every
book's score **down** (a genuine lead that lands inside the same snapshot as
the consensus move is scored as a non-lead), which compresses the weights
toward each other and therefore toward the equal-book aggregate. The pool
weights are normalised inside each arm, so a uniform compression cancels; what
does not cancel is a change in the **ordering** between the coarse and fine
eras, and section 8 reports exactly that as the era split-half.

## 4. Population, surfaces and grading

Held identical to `docs/follow_threshold_live_card.md`, which is the lane that
last graded this rule on the card that is played.

- **Follow-replay population.** The 2023-2025 `intraday_hourly`
  historical-backfill archive under `data/market/raw`, the only capture set
  the served rule's own evidence has ever been replayed on. 816 archive games,
  272 per season.
- **Stated deviation.** The served path loads `capture_kind="live"` and the
  store holds no live capture before 2026-08-17, so 2023-2025 is reachable only
  through the historical backfill. Same substitution as
  `docs/served_card_harness.md` and `docs/follow_threshold_live_card.md`; this
  is evidence about the rule's shape, not a re-grade of a live card.
- **Leadership-estimation population.** Every NFL spread capture in
  `data/market/raw` for the pool seasons, hourly and non-hourly alike, because
  leadership is a property of a book rather than of a capture cadence and the
  pre-2023 seasons have no hourly captures at all.
- **CARD surface (primary).** The served nine-member joint OR
  `overlay_union_coach_division_arrests_bye_coldvisitor_protection_interim_tank_precip_v3`,
  rebuilt with `nfl_ats.unserved_tilt_marginals.served_card_flip_set(card="served")`
  on the current active model's opener archive. This is the surface that
  decides the played card, so it is the surface the decision is taken on.
- **RAW surface (reconciliation).** The active model's own opener pick,
  `home_cover_probability_at_open >= 0.5`, no card. Reported so this lane sits
  in the same table as `docs/sharp_weighted_follow.md`, which used a raw
  surface.
- **Grading.** Opener grading, `margin_vs_open`, pushes unscored — the
  project's declared primary goal and the grade the served rule was promoted
  at.
- **The instant.** Each game's cutoff is its own pick deadline
  `min(kickoff, that week's Sunday 16:00 ET)`, passed to the frozen
  `late_week_follow_frame`; the per-book net move is the served construct
  exactly — difference each book's line over `[Monday, Sunday 00:00 ET)` and
  sum only the increments observed at or after that week's Wednesday. Sunday
  moves are outside the rule on every arm, including the new ones, because
  changing the window would stop this being a test of the aggregation.

**Parity check, run before any arm is scored and reported as a number rather
than absorbed:** the per-book net moves this lane reconstructs must reproduce
`sharp_book_movement_features`'s own `equal_net_move`, `eligible_books`,
`leader_median_net_move` and `leader_books` on every game, to 1e-9. If they do
not, the lane reports the mismatch and stops.

## 5. The arms (a declared grid; nothing is chosen after a sign is seen)

Let `n_b(g)` be book `b`'s Wednesday-to-deadline net move on game `g` over the
books that contributed an increment, and `w_b(S)` book `b`'s leadership score
from the seasons before `g`'s season. `gate(g)` is the served per-game gate
`sharp_book_movement_features.leader_follow_threshold(tue_open_home_spread)` —
a full point below a 10.5-point line, half a point at or above it. Side
convention is the served one throughout: `net > 0` picks HOME.

| Arm | Aggregate | Gate |
|---|---|---|
| **B0** | no follow; the surface's own pick | — |
| **B1** (served replay; the baseline of every head-to-head) | median of `n_b` over the three `LEADER_BOOKS` that contributed | `gate(g)` |
| **B1f** | the same leader median | flat 1.0 |
| **W1s** | `sum_b w_b n_b / sum_b w_b` over every contributing book | `gate(g)` |
| **W2s** | weighted median of `{n_b}` with weights `w_b` | `gate(g)` |
| **W1f** | leader-weighted mean | flat 1.0 |
| **W2f** | leader-weighted median | flat 1.0 |
| **W1h** | leader-weighted mean | flat 0.5 |
| **W2h** | leader-weighted median | flat 0.5 |
| **PC** | perfect foresight on every game with at least one contributing book | — |

Predeclared, before the signs:

* **Both a weighted mean and a weighted median are declared, and neither is a
  fallback for the other.** The mean is the natural reading of "weight the
  moves by leadership"; the median is declared beside it because the served
  arm is a median and a mean over twelve books lives off the half-point
  lattice the gate is defined on — `docs/late_week_refresh.md` already
  measured 198 games where the leaders cleared the gate and the twelve-book
  mean was diluted below it. Scoring only the mean would confound aggregation
  with dilution. Both are reported whatever they say.
* **The 0.5 arms (W1h, W2h) are declared here, before scoring**, which is the
  only condition under which the task allows them to be read at all. They are
  reported as a sensitivity on the gate, never as a promotion candidate chosen
  after the fact.
* **The weighted median** is the smallest `n_b` whose cumulative weight, over
  books sorted by `n_b`, reaches half the total weight; with an exact
  half-weight tie it is the midpoint of the two neighbouring values.
* **An arm that does not fire keeps the surface's pick.** A game with no
  contributing book keeps it under every arm. W1/W2 do not fall through to
  anything.
* **The positive control** switches only where a follow rule could switch (at
  least one contributing book). It exists to prove the instrument can resolve
  an effect on this population and to say what it does and does not bound.

## 6. Statistics, declared before the numbers

* Paired deltas in accuracy points, game-weighted, on the games both arms
  score.
* `nfl_ats.overlay_composition.blocked_bootstrap_matrix`, the repository's
  block bootstrap, **week-blocked** (`season, week`) and **season-blocked**,
  20,000 resamples, seed 20260821 — the seed and sample count
  `docs/sharp_weighted_follow.md` and `docs/follow_threshold_live_card.md` both
  used, so the three lanes' intervals are drawn the same way.
* `probability_positive` on every cell, `P(delta > 0) + 0.5 * P(delta == 0)` via
  `nfl_ats.evidence_conventions.probability_positive_from_draws`.
* **The decision quantity is `W* - B1` on the CARD surface, week-blocked** —
  the weighted arm against the served rule, on the same games and the same
  blocks. Every arm is additionally reported against B0 (no follow) for
  context, per season, and on the RAW surface.
* Picks changed against B1, and against B0, per arm and per surface.
* Split-half reliability of the leadership scores themselves: odd-week versus
  even-week halves within the pool, and the coarse era (2020-2022) versus the
  fine era (2023-2025), Pearson and Spearman across the twelve books.

## 7. The decision rule, declared before the numbers

The pool is forced picks: 285 cards get submitted either way, so declining an
arm that is more likely than not to be better is taking the other side of that
bet. **The arm with the higher expected accuracy at the deadline is the one to
serve**, and `probability_positive` above 0.5 favours playing it. A 0.90-style
bar governs only what this document may CLAIM, never which card is played.

Nothing in this lane is wired. If the leader-weighted arm is ahead, section 8
says so plainly and names the exact function and constant a promotion would
touch.

### Closing grounds (verbatim, binding)

An interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (1) refuted mechanism — a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (2) bounded by a positive
control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero".

Every cell is recorded through `nfl-ats weak-signals record` under names
`sharp_book_weighted_movement_<surface>_<arm>_vs_<baseline>_<block>`, units
`accuracy_points`, league `nfl`, family `sharp_book_weighted_movement`,
category `market`. The family is declared here before the signs were seen; its
members share one window, one population and one baseline, so they are
correlated readings of one question, never independent votes.

## 8. Results

Measured 2026-09-10/11 in one pass by `scripts/sharp_book_weighted_movement.py`,
artifact `artifacts/sharp_book_weighted_movement/20260911T024752Z/`
(`results.json`, `cells.csv`, `leadership.csv`, `per_game.parquet`). Opener
archive `artifacts/opener_evaluation/20260910T211255Z/per_game.parquet`, active
model `d49194e04945a5e5`.

One provenance note, stated rather than hidden: the script was re-run into the
same stamped directory after a formatter pass changed its bytes, so
`configuration.script_sha256` matches the script on disk and
`configuration.predeclaration_sha256` is the hash of this document **with**
section 8 rather than of the frozen sections 1-7. Every number reproduced
identically across the two runs (the bootstrap seed is fixed), which is what
the second run was for.

### 8.0 The two checks that had to pass first

**Parity is exact.** The per-book net moves this lane reconstructs reproduce
`sharp_book_movement_features`'s own aggregates on all 816 games with a maximum
absolute difference of **0.0** on both `equal_net_move` and
`leader_median_net_move`, and **0** book-count mismatches on both
`eligible_books` and `leader_books`. 0 quote rows were refused by the time
guards. So every arm below differs from the served rule in exactly one place:
how the per-book moves are aggregated.

**Both surfaces land where the earlier lanes published them.** The card with no
follow scores **56.0701%** on the 799 scored games, the figure
`docs/served_card_harness.md` publishes to three decimals; the raw model's own
opener pick scores **55.0688%**, the figure `docs/follow_threshold_live_card.md`
expected. 816 games in scope, 799 scored, 17 opener pushes, 54 week blocks,
3 season blocks, 260 nine-member flips in window.

### 8.1 The leadership table, recomputed chronologically

Leads / participations / score, per book, per pool. A pool contains only
seasons strictly before the season it weights. Fanatics never contributes a
capture before 2025, so it never clears the 200-participation bar and takes the
pool's neutral prior in all three columns (shown in brackets).

| Book | 2023 weights (pool 2020-2022) | 2024 weights (pool 2020-2023) | 2025 weights (pool 2020-2024) |
|---|---|---|---|
| fanduel | **0.2955** (607/2054) | **0.3890** (1844/4740) | **0.4142** (2790/6736) |
| draftkings | 0.2481 (536/2160) | 0.3580 (1780/4972) | 0.3891 (2768/7114) |
| betus | 0.2421 (277/1144) | 0.3413 (1279/3747) | 0.3541 (2010/5676) |
| betmgm | 0.2418 (397/1642) | 0.3027 (1043/3446) | 0.3051 (1425/4670) |
| betrivers | 0.2282 (491/2152) | 0.2846 (1212/4258) | 0.2949 (1717/5823) |
| lowvig | 0.2281 (433/1898) | 0.3113 (1316/4228) | 0.3186 (1881/5904) |
| fanatics | [0.2253] (0/0) | [0.3156] (0/0) | [0.3332] (0/0) |
| pointsbetus | 0.2188 (368/1682) | 0.3167 (1097/3464) | 0.3167 (1097/3464) |
| betonlineag | 0.2159 (443/2052) | 0.3031 (1347/4444) | 0.3125 (1913/6121) |
| **bovada** | 0.2068 (405/1958) | 0.3091 (1230/3979) | 0.3442 (1758/5107) |
| **mybookieag** | 0.1866 (361/1935) | 0.2622 (1067/4069) | 0.2920 (1707/5846) |
| **williamhill_us** | **0.1731** (366/2114) | 0.2826 (1407/4978) | 0.2981 (2147/7202) |

Bold-named rows are the three books the served rule listens to. **On this
statistic they are not the leaders.** In the 2020-2022 pool that weights 2023
they finish 10th, 11th and 12th of twelve; in the 2020-2024 pool that weights
2025 they finish 4th, 10th and 12th. FanDuel and DraftKings lead every pool.

The scores rise across pools (0.17-0.30 for the coarse pool, 0.29-0.41 for the
widest) exactly as section 3 predeclared: adding the fine-cadence 2023-2024
seasons lets a lead land in its own capture instead of being swallowed by the
snapshot that also carries the consensus move.

### 8.2 Diagnostic: is this the same trait SKY-04 measured?

Post-hoc, named as post-hoc, computed after the arms were scored because it is
the mechanism for what the arms did. Correlating this lane's per-book score
against `docs/book_leadership.md`'s first-mover share
(`sharp_book_movement_features.LEADERSHIP_WEIGHTS`), over the books that clear
200 participations:

| Pool | Pearson | Spearman | Books |
|---|---|---|---|
| 2023-2025 (fine cadence) | **+0.196** | +0.238 | 12 |
| 2020-2022 (coarse cadence) | **-0.649** | -0.500 | 11 |

**These are not the same trait.** SKY-04 scores a book by whether its
`bookmaker_last_update_utc` is the earliest among books that changed in a
capture; this lane never reads a provider timestamp and scores a book by
whether it is already displaced toward the new number one capture before the
leave-one-out consensus gets there. SKY-04's own document warned that its
statistic confounds clock skew with speed, and the staleness column it
published is consistent with that: Bovada 48 s, MyBookie 73 s, William Hill
100 s against DraftKings 46 s and FanDuel 35 s. The counter-caution against
this lane's statistic is equally plain: it rewards being displaced from
consensus, so a book that habitually hangs an idiosyncratic number scores well
for a reason that need not be anticipation. **Neither statistic is validated
against an outcome here.** What is measured is which one, spent as a weight,
picks more winners — section 8.3.

For completeness, the same statistic computed on the fine era alone (2023-2025,
the seasons SKY-04 used) orders the books: fanduel 0.4665, draftkings 0.4510,
bovada 0.4471, fanatics 0.4376, pointsbetus 0.4091, betus 0.3955, mybookieag
0.3751, lowvig 0.3728, betonlineag 0.3712, williamhill_us 0.3584, betrivers
0.3535, betmgm 0.3475.

### 8.3 The arms: weighted move versus the served median

CARD surface (primary), 799 opener-scored games, week-blocked bootstrap,
20,000 resamples, seed 20260821. "vs B1" is the decision quantity: the
candidate against the served rule on the same games and the same blocks.

| Arm | Fires | Accuracy | Picks changed vs B1 | Delta vs B1 | 95% (week) | `probability_positive` | 95% (season) |
|---|---|---|---|---|---|---|---|
| B0 no follow | — | 56.070% | 109 | — | — | — | — |
| **B1 served leader median, served gate** | 241 | **57.697%** | — | — | — | — | — |
| B1f served leader median, flat 1.0 | 228 | 57.322% | 9 | -0.375 | [-1.125, +0.373] | 0.157 | [-0.752, 0.000] |
| **W2s weighted MEDIAN, served gate** | 234 | 57.572% | 11 | **-0.125** | [-0.896, +0.631] | **0.379** | [-0.749, +0.376] |
| W2f weighted MEDIAN, flat 1.0 | 224 | 56.946% | 16 | -0.751 | [-1.639, +0.125] | 0.049 | [-0.752, -0.749] |
| W1s weighted MEAN, served gate | 160 | 56.320% | 41 | -1.377 | [-2.750, 0.000] | 0.023 | [-1.873, -0.752] |
| W1f weighted MEAN, flat 1.0 | 150 | 55.945% | 48 | -1.752 | [-3.354, -0.247] | 0.013 | [-2.996, -0.752] |
| W1h weighted MEAN, flat 0.5 | 317 | 55.820% | 47 | -1.877 | [-3.234, -0.624] | 0.001 | [-2.256, -1.504] |
| W2h weighted MEDIAN, flat 0.5 | 493 | 55.069% | 125 | -2.628 | [-5.112, -0.124] | 0.022 | [-5.993, 0.000] |
| PC positive control | 816 | 100% | 346 | +42.303 | [+39.130, +45.512] | 1.000 | [+40.602, +43.609] |

Against no follow at all (B0), for context: B1 **+1.627** [-0.752, +4.066]
P+ 0.909; W2s **+1.502** [-0.881, +3.980] P+ 0.889; W2f +0.876 P+ 0.753;
W1s +0.250 P+ 0.597; W1f -0.125 P+ 0.440; W1h -0.250 P+ 0.434; W2h -1.001
P+ 0.271; PC +43.930 P+ 1.000.

Per season, candidate minus B1 (2023 / 2024 / 2025): W2s 0.00 / +0.376 /
-0.749; W1s -1.504 / -0.752 / -1.873; W1f -1.504 / -0.752 / -2.996;
W2f -0.752 / -0.752 / -0.749; W1h -2.256 / -1.504 / -1.873; W2h -1.880 /
0.000 / -5.993. B1 against B0 per season: 0.00 / +2.256 / +2.622.

RAW surface (reconciliation), same games, against the served rule:
W2s -0.375 [-1.152, +0.381] P+ 0.183; W2f -0.751 [-1.735, +0.248] P+ 0.065;
W1s -2.003 [-3.491, -0.615] P+ 0.003; W1f -2.128 [-3.741, -0.502] P+ 0.005;
W1h -1.377 [-3.011, +0.250] P+ 0.048; W2h -1.252 [-3.837, +1.391] P+ 0.175;
B1f -0.375 [-1.005, +0.251] P+ 0.128. The served arm beats no-follow on this
surface by +2.128 [-0.126, +4.485] P+ 0.966.

### 8.4 Split-half reliability of the leadership score

| Split | Pearson | Spearman | Books |
|---|---|---|---|
| Odd weeks vs even weeks, all seasons | **+0.963** | +0.937 | 12 |
| Coarse era (2020-2022) vs fine era (2023-2025) | +0.519 | +0.345 | 11 |
| Season pairs, mean of 15 | +0.431 | — | 9-11 |

Within the fine era the season-to-season correlations are +0.860 (2023-2024),
+0.733 (2023-2025) and +0.967 (2024-2025); within the coarse era they are
+0.227, +0.238 and +0.320; the weakest pair of all is 2020 against 2025
(+0.151).

**The trait is real and it is measured well.** A 0.963 odd/even split-half is
about as reliable as anything in this repository's registry, so
`no_split_half_reliability` is not available as a closing ground for anything
in this lane, and nothing here is a measurement-noise story. What the era split
shows instead is the weak joint section 3 declared in advance: the ORDERING
survives the coarse-to-fine cadence change only at +0.519 Pearson / +0.345
Spearman, so the 2023 weights — the ones drawn from the coarsest pool — are the
ones least entitled to be trusted, and 2023 is indeed the season W2s is flat
and W1s is at its worst.

### 8.5 What this implies for the decision, before what is wrong with it

**No leader-weighted arm should be served. The served three-book leader median
stays.** The best weighted arm is W2s, the weighted median at the served gate,
and against the served rule it reads **-0.125 accuracy points,
`probability_positive` 0.379** on the card that is played. `probability_positive`
below 0.5 means the served side is the higher-expectation side of that bet, so
the EV decision is to leave the played card alone — the same forced-pick logic
that promoted the leader median in the first place, pointing the other way this
time. Nothing in this lane is wired and nothing should be.

**Two arms are closed, and only these two.** The leadership-weighted MEAN at a
flat 1.0 gate (card, -1.752, week [-3.354, -0.247], season [-2.996, -0.752])
and at a flat 0.5 gate (card, -1.877, week [-3.234, -0.624], season
[-2.256, -1.504]) have their whole week-blocked AND whole season-blocked
intervals below zero against the served rule, on the primary surface; the same
two mean arms resolve the same way on the raw surface (W1s -2.003
[-3.491, -0.615], W1f -2.128 [-3.741, -0.502]). Those four cells are recorded
`refuted_mechanism` on `wrong_sign_resolved`. **What is refuted is narrow and
should be stated narrowly:** spreading a leadership weight across all twelve
books as a MEAN is worse than the served three-book median, because the weight
lands hardest on FanDuel and DraftKings (section 8.1) and because a mean over
twelve books lives off the half-point lattice the gate is defined on — it fires
on only 150-160 games against the served rule's 241. It does not refute "sharp
books lead", it does not refute the leadership score (reliability 0.963), and
it does not refute a weighted median: **W2s, W2f, W2h, W1s and B1f are all
`unresolved_below_power`** — their intervals cross or touch zero, and an
interval containing zero is never a rejection ground (AGENTS.md).

**Nothing here is bounded by a control.** The positive control resolves
**+42.303** accuracy points on 346 changed picks (week [+39.130, +45.512]), so
the instrument demonstrably works at 42 points and has never been shown able to
separate one point from zero on 799 games. It bounds nothing in the 0-2 point
range, which is where every real cell in this lane sits.

**The one result worth carrying forward** is section 8.2, not the arms: the
served rule's three books were chosen on a provider-timestamp statistic that
its own document flagged as confounded with feed staleness, and a
clock-independent statistic measured on the capture sequence does not agree
with it (+0.196 in the fine era, -0.649 in the coarse era) and ranks those
three books low. The served rule still wins the card, which is the only test
that decided anything here — but the reason it wins is not established to be
"those three books are the fast ones", and MKT-15's descriptive premise is
weaker after this lane than before it.

### 8.6 If it had been ahead, or is ahead on a later read

Named here so the row is actionable without re-deriving it. Serving a
leader-weighted aggregate needs three edits, all inside
`src/nfl_ats/sharp_book_movement_features.py`:

1. `LEADERSHIP_WEIGHTS` stops being a frozen dict of 2023-2025 in-sample shares
   and becomes a per-season table loaded from a stamped artifact, with the
   season's own weights computed only from seasons before it (this lane's
   `leadership_counts` / `scores_from_counts` are the reference implementation);
2. `sharp_book_movement_features` gains a `weighted_median_net_move` column
   beside `leader_median_net_move` and `equal_net_move`, so one computation
   still feeds the served pick and every paired ledger arm;
3. `late_week_follow_frame` reads that column for the served side, and the
   leader median moves to a paired OFF challenger
   (`late_week_leader_median_follow_off_incumbent`), exactly the pattern the
   equal-book rule already follows.

`src/nfl_ats/pick_refresh.py`'s `LATE_WEEK_LEADER_MEDIAN_FOLLOW_POLICY` id would
change with it, and `docs/late_week_refresh.md`'s "Promoted late-week follow"
section would need the new rule stated in the same voice. None of that was done,
because the measurement says not to.

### 8.7 Registry

All 24 cells are recorded under family `sharp_book_weighted_movement`,
category `market`, units `accuracy_points`, league `nfl`, seasons 2023-2025,
each carrying the leadership score's own split-half reliability (0.963) — 20
`unresolved_below_power` and 4 `refuted_mechanism` on `wrong_sign_resolved`
(`card_w1f_vs_b1`, `card_w1h_vs_b1`, `raw_w1s_vs_b1`, `raw_w1f_vs_b1`). The
exact commands are in
`artifacts/sharp_book_weighted_movement/20260911T024752Z/record_commands.json`
and the runner that issued them is `record.py` beside it.

Rotation family `sharp_book_weighted_movement` was declared (grade `opener`,
mined-acknowledged) and assigned window **[2020, 2021]** before any score was
computed, and the look was recorded `unresolved` at
`probability_positive` 0.378725 on the headline W2s-vs-B1 cell. Disclosed with
it, as LEAD-01 disclosed the same thing: the assigned window is spent by the
**leadership estimation**, which is what the window governs here, while the
follow replay is 2023-2025 because the archive holds no hourly capture before
2023 and the served rule has never been replayable on an earlier season.
