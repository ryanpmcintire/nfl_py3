# Grading the model at a single book's posted opener

Frozen 2026-09-11, before any candidate number was computed and before the
book coverage survey was run. Family `single_book_opener_grade_v1`.
Everything above the "Measured" heading was written first and its bytes are
digested into `artifacts/research/laneP2/reproduction.json`
(`predeclaration_sha256`).

## The gap this lane closes

Read: `docs/mod18_pool_shaped_read.md`, closing section "What this lane will
leave open, named now" -- grade the family "at a SINGLE BOOK's posted half
point, rather than at the cross-book consensus, using
`data/market/raw/*/quotes.parquet` at `decision_label` `tue_open`", because
"that requires refitting the residual, the home-side offset and the graded
outcome at that book's line, so it is a new opener evaluation rather than a
slice of this one. It is the only way to measure the family at literally the
number a pool posts."

Read: the same doc's measurement that the opener evaluation's
`tue_open_home_spread` is the CONSENSUS median across the captured books and
is a half point on 842 of 1,537 archive games, taking quarter-point values
elsewhere. A quarter point is not a number any book posts and not a number the
pool can post; a consensus median is an average of prices, not a price.

The owner's pool always quotes a half point at ONE number. So the
decision-relevant grade is the one taken against a line that a book actually
posted, and this lane runs the opener evaluation at that line rather than
slicing the consensus one.

## What changes when the line changes, stated before the run

Substituting the opener line changes BOTH sides of the grade at once:

- the model's input -- `spread_line` is a feature, so the predicted residual
  and the served cover probability are recomputed at the new number;
- the graded outcome -- `margin_vs_open` is `result - line`, so which side
  covered, and whether the game pushes at all, are decided at the new number;
- the served home-side offset, which is fitted walk-forward on prior weeks'
  `(line, point, result)` stream and is therefore refitted at the new lines.

A difference between the two grades is therefore NOT attributable to the model
alone. It is the answer to "how well does this model do against the number a
pool could post", which is the project's stated primary goal, and it is
reported as that and never as a model comparison.

## The line series, and the rule for choosing the book

Two series are built from the per-book opener archive, `tue_open` snapshots
only, `market == "spreads"`, `outcome_side == "HOME"`, the book's LAST quote
before kickoff in that snapshot (the same de-duplication
`decision_market_consensus` applies before it takes its median):

1. **`BOOK`** -- the posted `home_spread_line` of ONE named book. The book is
   chosen by COVERAGE, before any grade is computed, by this rule, declared
   here so the choice cannot be made from a result:

   - the candidate set is the major US retail sportsbooks whose line a pool
     engine is most likely to mirror, i.e. those of
     `{draftkings, fanduel, betmgm, caesars, williamhill_us, pointsbetus,
     betrivers, espnbet, barstool, unibet_us, wynnbet, superbook, betonlineag,
     bovada, mybookieag, lowvig, betus, betfair, foxbet, twinspires,
     sugarhouse, circasports, hardrockbet, fanatics, ballybet}` that appear in
     the `tue_open` archive at all;
   - among them, the book with the MOST distinct games carrying a `tue_open`
     HOME spread quote over 2020-2025;
   - ties broken by total quote rows, then alphabetically by bookmaker key.

   The chosen book, its game coverage, and the runners-up are reported before
   any accuracy number below.

2. **`HALFPOINT`** -- the fallback series named by the task: per game, the
   MEDIAN of the `tue_open` HOME spread quotes whose posted line is exactly a
   half point, across all books in the archive (not only the candidate set).
   A game with no half-point quote at the opener is absent from this series.
   This series is a pool-shaped line by construction; it is a synthetic
   consensus of only the half-point prices, and is reported as the second
   series, never as "the book's line".

For each series the run reports: games covered, games covered that are also in
the consensus archive (the paired set every comparison is taken on), how often
the series equals the consensus line exactly, the mean absolute gap to the
consensus, and the share of the series that is a half point, by season.

## Domains, frozen

`d` is the distance from `abs(line)` to the nearest atom in `{3, 7}`, measured
at the series' own line.

| Domain | Definition |
|---|---|
| **ALL** | every paired game the series covers |
| **HALF** | `ALL` and the series line is exactly a half point |
| **NEAR** | `HALF` and `d <= 0.5`, i.e. `abs(line)` in `{2.5, 3.5, 6.5, 7.5}` |
| **NEAR_3** | `abs(line)` in `{2.5, 3.5}` |
| **NEAR_7** | `abs(line)` in `{6.5, 7.5}` |
| **AWAY** | `HALF` minus `NEAR` |
| **WHOLE** | `ALL` minus `HALF`, reported for contrast only |

Per-season cells are cut inside `ALL` and inside `HALF`.

## Stage 1 -- the headline grade

The ACTIVE model (`artifacts/active_ats_model.json`, confirmed by digest
before anything runs) is run through `opener_pick_evaluation` at each series'
line, into a NEW stamped directory that is NOT
`artifacts/opener_evaluation/`, so no published page can ever mistake a
line-source run for the active model's own opener evaluation. Its metadata
names the line source, its digest, the book, and the coverage.

Reported on the paired games, at the opener, for the model's served
probability-rule pick:

- accuracy at the single-book line and at the consensus line, the difference
  in accuracy points, the week-blocked interval, `probability_positive`, `n`;
- the same by season;
- the same for the served rule chain (the probability rule with the served
  home-side offset, and the raw rule without it) wherever the evaluator emits
  it.

## Stage 2 -- the MOD-18 mapping family at the posted line

Arms imported, not reimplemented, exactly as lane P1 imported them:
`nfl_ats.discrete_margin_mapping.ARMS` applied through `apply_arm`, whose
candidate probability comes from `nfl_ats.mass_preserving_lattice`'s
walk-forward band read, with the prior pool built at the SAME series line so
the lattice reads the distribution conditional on the number actually posted.

- **G2** -- touched when `d <= 0.5` for atoms `{3, 7}`.
- **KL1b** -- `abs(line)` exactly 3 or 7, the SERVED rule, carried as a
  baseline.
- **S3** -- the served smooth read with the home-side offset, the evaluator's
  own `home_cover_probability_at_open`.

Scored on `NEAR`, `NEAR_3`, `NEAR_7` and, for context, `HALF` and `ALL`, both
standalone and through the played three-member overlay union (coach fade,
division revenge, player arrests), membership frozen before scoring, no subset
search.

**Predeclared structural prediction, to be verified and not assumed.** On a
half-point line KL1b can touch nothing, so on `HALF` it must be bit-identical
to S3; and G2's touch set on `HALF` is exactly `NEAR`. A failure of either
stops the run before scoring.

## Metric and protocol

- Forced-pick ATS accuracy at the **OPENER**, paired per game, week-blocked
  block bootstrap over (season, week) blocks, 20,000 draws, seed 20260817.
  Within-week game correlation is ZERO by owner mandate: never estimated,
  never padded.
- Pushes at the series line are dropped from the graded set, per the
  evaluator's own `pick_correct`.
- `probability_positive` is reported for every cell. The binary "the interval
  contains zero" is never reported and is never a verdict.

## Decision rule, declared before the numbers are seen

Stated as an action, not a bar:

1. If the headline opener accuracy at the posted line differs from the
   consensus grade, the DECISION implication -- what the pool's real number is
   worth to this model -- is stated first, before any limitation.
2. A mapping arm whose `probability_positive` against the served read on
   `NEAR` (through the played card, the decision read) exceeds 0.5 is
   registered as an **ACTIVE_PROSPECTIVE paired challenger** in
   `artifacts/prospective/challengers.json`, following
   `key_line_pick_read_off_incumbent`. 0.5 is the expected-value line for the
   action, not a bar on what may be measured or recorded.

This lane does **not** change the served card, the active model, or any
published forecast, and never writes into `artifacts/opener_evaluation/`.

## Leakage gates

- Only `tue_open` snapshots enter the line series; every quote in them is
  observed before kickoff (`decision_market_consensus` already drops quotes at
  or after `commence_time_utc`).
- The evaluator refits the ridge model weekly on strictly prior completed
  games and fits the home-side offset on a stream of strictly prior weeks;
  both are unchanged by this lane.
- The lattice prior pool for a week contains no game from that week or later
  and never the target game, enforced inside
  `nfl_ats.mass_preserving_lattice.prior_pool_for_week`.
- The feature-table digest must equal the active model's before anything runs.
- The S3 replay (algebraic inversion of the served `gaussian_median` read)
  must reproduce the run's own served probability to 1e-9 before any candidate
  is built; the run fails closed above that.

## Reuse discount, disclosed before the run

The games are the same 2020-2025 seasons every other lane has read, and the
atom set `{3, 7}` is lane T's, chosen after seeing lane K's bucket-7 gain on
these same games. What is new is the LINE: a per-book posted number that no
prior lane has graded at, so the graded outcome of every game differs from the
consensus archive on the games where the book and the consensus disagree.
These are descriptive reused-era measurements, not independent confirmation.
No numerical discount is invented and no rotation window is spent: precedent
`docs/player_arrests_policy_eval.md`, MOD-17 and lane P1, all promotion-style
looks on this archive without a `rotation record-look` entry.

## Closing-grounds taxonomy (binding, verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close
a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else
is `unresolved_below_power`: record it with `nfl-ats weak-signals record`,
report `probability_positive`, never the binary "contains zero". The registry
code hard-rejects inadmissible closures; if a record command errors, the
verdict is wrong, not the validator. Verdicts flow through
`nfl-ats weak-signals record` / `nfl-ats rotation record-look`, never through
prose in a doc. Never compute or state that something needs N more games: the
data is fixed and the project is model-limited. Never use 95% or 0.90 as a
DECISION bar; decide on expected value (`probability_positive` above 0.5
favours playing). Within-week game correlation is ZERO by owner mandate;
never estimate or pad it. Grade at the OPENER; a close-graded number may
never veto a play.

## Where the code lives

- `src/nfl_ats/single_book_opener.py` -- the line series builder.
- `src/nfl_ats/clv.py` -- `opener_pick_evaluation`, given an opener-line
  override rather than forked.
- `src/nfl_ats/cli_commands/clv.py` -- `nfl-ats opener-evaluation
  --opener-line-source`.
- `scripts/single_book_opener_grade.py` -- this lane's run.
- `artifacts/research/laneP2/` -- its artifacts.

---

# Measured (2026-09-11, after the predeclaration above was frozen)

Everything above this line was written before the book coverage survey ran;
its bytes are digested into `artifacts/research/laneP2/headline.json` as
`predeclaration_sha256`
`43e17e5334439cb14fab88b2831c613ba4625fd8124854d78999afcb51561462`.

Run: `scripts/single_book_opener_grade.py --stage headline|map|score|record|posthoc|report`,
on top of `nfl-ats opener-line-series` and `nfl-ats opener-evaluation
--opener-line-source`. Active model `7786467eabe418b8`, feature table
`10c82814...`, confirmed against `artifacts/active_ats_model.json` before
anything ran.

## The decision first

**The project's headline opener number is about seven tenths of a point
optimistic about the line a pool actually posts.** Graded at DraftKings'
posted Tuesday opener rather than at the cross-book consensus, the same active
model, on the same 1,477 paired graded games, scores **53.83% against 54.50%**
-- a difference of **-0.677 accuracy points, 95% [-1.279, -0.134],
`probability_positive` 0.009** for the single-book grade being the higher one.
Every season runs the same way: 2020 -0.488, 2021 -1.282, 2022 -0.813, 2023
-0.379, 2024 -0.380, 2025 -0.755.

Nothing is closed by that and no rule is refused. It is a re-basing: the
number the pool will settle is the lower one, and the difference is a property
of the LINE, not of a candidate.

**No mapping arm beats the served read at the posted half-point line**, so the
predeclared registration rule is not met and **no challenger was registered**.
On the 485 games where DraftKings posted 2.5, 3.5, 6.5 or 7.5, G2 through the
played card is **-1.443 [-4.366, +1.452], `probability_positive` 0.166**, and
the served exact-atom rule KL1b cannot fire on a single one of them.

## The line series, and the book

Measured on the per-book opener archive: 241 historical `tue_open` snapshots
over 2020-2025, **21,521 HOME spread quote rows, 1,613 games, 24 books** after
the own-week and pre-kickoff filters and one last quote per book per game.

The predeclared coverage rule selects **DraftKings**, with **1,597 games**, the
most of any candidate book. Runners-up: betrivers 1,559, fanduel 1,531,
betonlineag 1,506, williamhill_us 1,468, betmgm 1,465.

| Series | Games | Half-point share | Equals the consensus | Mean absolute gap |
|---|---:|---:|---:|---:|
| `BOOK` (draftkings) | 1,597 | **56.92%** | 74.26% (1,186) | 0.139 (max 2.0) |
| `HALFPOINT` (median of half-point quotes) | 1,480 | 98.24% | 60.68% (898) | 0.228 (max 14.5) |

Two things in that table matter and neither was predicted.

**A single book posts a whole number on 43% of its openers.** The pool-shaped
domain does not arrive by picking one book; DraftKings' opener is a half point
on 909 of 1,597 games, barely more often than the consensus' 54.8%. Lane P1's
`POOL` domain and this lane's `HALF` domain are therefore the same kind of
restriction, taken on a real posted price rather than on a median.

**The `HALFPOINT` series is thin where it is most synthetic.** 145 of its
1,480 games rest on a single book's half-point quote, and its worst
disagreements with the consensus are exactly those: 2020_08_LAC_DEN at +11.5
against a consensus of -3.0, 2020_06_CIN_IND at +18.5 against +8.0. 26 of its
lines are not half points at all, because a median of an even number of
straddling half-point quotes lands on the whole number between them. It is
reported as the predeclared fallback and is not this lane's answer.

## Stage 1 -- the headline grade, paired

`artifacts/opener_evaluation_line_source/20260911T164253Z` (DraftKings, 1,521
games) and `.../20260911T164325Z` (half-point median, 1,410 games) against
`artifacts/opener_evaluation/20260911T161354Z` (the consensus grade of the
same active model, 1,537 games). Neither line-source run writes into
`artifacts/opener_evaluation/` and neither carries a `model_id`, so no
published page can mistake one for the active model's own evaluation.

| Series / rule | n | At the series line | At the consensus | Delta | P+ |
|---|---:|---:|---:|---|---:|
| DraftKings, card rule | 1,477 | **53.825%** | **54.502%** | **-0.677** [-1.279, -0.134] | **0.009** |
| DraftKings, before the home-side nudge | 1,477 | 53.622% | 53.825% | -0.203 [-0.546, +0.136] | 0.131 |
| DraftKings, points-edge sign | 1,477 | 52.674% | 52.945% | -0.271 [-0.545, -0.067] | 0.009 |
| DraftKings, half-point games only | 852 | 53.756% | 54.108% | -0.352 [-1.119, +0.371] | 0.184 |
| Half-point median, card rule | 1,383 | 54.085% | 54.736% | -0.651 [-1.373, +0.000] | 0.031 |
| Half-point median, before the nudge | 1,383 | 53.941% | 53.868% | +0.072 [-0.221, +0.422] | 0.671 |

## Where the gap lives (post-hoc, labelled as such)

Not predeclared; cut after the headline signs were visible and recorded in its
own family `single_book_opener_grade_posthoc_v1` so it can never be pooled
with the predeclared cells.

Only **20 of 1,521 picks change side** when the line changes. The gap is
overwhelmingly the OUTCOME moving -- which side covered at the other number --
not the model changing its mind.

| Split | n | Delta | P+ |
|---|---:|---|---:|
| Games where DraftKings posted the consensus number | 1,104 | -0.362 [-0.900, +0.092] | 0.073 |
| Games where it posted a different number | 373 | -1.609 [-3.457, +0.247] | 0.038 |
| Value of the served home-side nudge, at the posted book line | 1,489 | **+0.201** [-0.754, +1.188] | 0.657 |
| Value of the served home-side nudge, at the consensus line | 1,487 | **+0.672** [-0.199, +1.547] | 0.939 |

The last two rows are the substantive ones. The home-side offset is worth
about **two thirds of a point at the consensus opener and about a fifth of a
point at a posted book opener**, refit walk-forward at each. My reading, and
it is inference rather than measurement: an offset fitted against a median of
prices is partly fitting the median itself -- quarter points and cross-book
disagreement have no counterpart on a card -- and the part that survives at a
real posted number is the smaller part. That is a modelling defect to work on,
not a flip to bolt on.

## Stage 2 -- the MOD-18 family at the posted line

Structural invariants, measured not assumed (`mapping.json`): the S3 replay
gate landed at **1.67e-16**, sixteen orders inside its 1e-9 refusal; **KL1b
touches 0 of the 864 half-point DraftKings games** and is bit-identical to S3
there; G2's touch set on those games is exactly the 485-game `NEAR` domain.
KL1b fires on 280 archive games, every one a whole 3 or 7.

| Domain | n | G2 standalone (P+) | G2 through the card (P+) |
|---|---:|---|---|
| `NEAR` 2.5/3.5/6.5/7.5 | 485 | -2.474 [-5.945, +1.002] 0.080 | **-1.443** [-4.366, +1.452] **0.166** |
| `NEAR_3` 2.5/3.5 | 337 | -2.374 [-6.925, +2.102] 0.151 | -1.484 [-5.203, +2.188] 0.218 |
| `NEAR_7` 6.5/7.5 | 148 | -2.703 [-8.824, +3.145] 0.187 | -1.351 [-6.122, +3.185] 0.282 |
| `HALF` every half-point line | 864 | -1.389 [-3.382, +0.572] 0.082 | -0.810 [-2.488, +0.832] 0.168 |
| `WHOLE` every whole-number line | 625 | -0.160 [-2.070, +1.843] 0.435 | **+0.480** [-1.077, +2.091] **0.724** |
| `ALL` | 1,489 | -0.873 [-2.194, +0.477] 0.104 | -0.269 [-1.538, +1.011] 0.344 |

On `WHOLE`, G2 and KL1b are the SAME picks, because a whole number within half
a point of 3 or 7 is 3 or 7. So this reproduces lane P1's finding at a real
posted price: **the family's whole measured value is the served exact-atom
rule, and it lives only on whole numbers.** DraftKings posts those on 43% of
games; the owner's pool posts them never.

Per season on `HALF`, G2 through the card: 2020 +0.699 (0.647), 2021 +0.917
(0.713), 2022 -1.681 (0.155), 2023 -0.676 (0.406), 2024 -0.714 (0.389), 2025
-2.439 [-5.314, +0.000] (0.029).

## Recorded

**72 rows** under family `single_book_opener_grade_v1` and **4** under
`single_book_opener_grade_posthoc_v1`, every one `unresolved_below_power` with
no closing ground and a plain-English summary (registry 6,364 to 6,440 rows,
zero failures). All 76 argv were first executed against an isolated registry
under `artifacts/research/laneP2/registry` and all returned 0, so the emitted
argv is admissible to the real validator. Arm cells in which no graded pick
moved at all are named in `registry_names.json` under `skipped_no_flip` and
deliberately not recorded, because a bootstrap of zeros carries a zero-width
band for what is really "this rule cannot fire here" -- a fact the structural
invariants already state exactly.

No rotation window spent: a promotion-style look on the reused 2020-2025
archive, the precedent `docs/player_arrests_policy_eval.md`, MOD-17 and lane
P1 all set.

## What this lane leaves open, named now

1. **The home-side offset is fitted at the wrong line.** It is worth +0.672
   points at the consensus and +0.201 at a posted book opener. Refitting it at
   a posted half-point line, and grading it there, is a lane of its own.
2. **A pool-line series, not a book series.** DraftKings posts a whole number
   43% of the time; the pool never does. The nearest thing to the pool's own
   number is "the half point this book posted, or the half point it moved
   through", which needs the intraday archive rather than the single Tuesday
   snapshot.
3. **FanDuel posts a half point on 74.7% of openers against DraftKings' 56.9%**
   -- measured here, outside the predeclared coverage rule, and not graded.
   A book chosen for half-point discipline rather than for coverage is a
   different and possibly better proxy for a pool engine.
