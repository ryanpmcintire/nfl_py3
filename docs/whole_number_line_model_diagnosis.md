# Why the model is weaker when the opening line is a whole number

Predeclaration frozen 2026-09-11, before any accuracy, bias, calibration or
effect in this lane was computed. Family
`whole_number_line_model_diagnosis_v1`. Everything above the "Measured"
heading was written first and its bytes are digested into
`artifacts/research/laneQ2/reproduction.json` as `predeclaration_sha256`.

## The observation this lane has to explain

Lane Q1 (`docs/overlay_union_line_shape.md`, family
`overlay_union_line_shape_mechanism_v1`) set out to explain why the played
three-member overlay union helps on whole-number opening lines and not on
half-point ones. It failed to explain it, and on the way it measured something
that is not about the overlays at all:

> the model's own pick is **-4.538 accuracy points [-9.641, +0.500],
> `probability_positive` 0.039** on whole-number opener lines relative to
> half-point ones.

That reading is on the previous active model's opener evaluation
(`artifacts/opener_evaluation/20260910T211255Z`). This lane re-measures it on
the model that is active now, `7786467eabe418b8`, opener evaluation
`artifacts/opener_evaluation/20260911T161354Z`, and then asks the successor
question Q1 named and did not answer: **why**.

The question matters for a reason that has nothing to do with the overlays.
The pool only ever posts a half point, so the served card lives entirely on the
shape where the model is stronger. But the archive-graded headline the site
publishes blends both shapes. If the whole-number weakness is a modelling
defect, fixing it raises the number a reader sees and, more importantly, tells
us something true about where the model's read breaks. If it is a property of
how the archive's decision line was constructed, the headline is measuring the
capture and not the model.

## What this lane will and will not do

It will not change the served card, the active model, any published forecast,
any overlay's membership rule, or any served probability mapping. It produces a
mechanism verdict, registry rows, and a named fix direction. Under every
possible result, a rule conditioned on line shape is forbidden on the played
card: an accuracy dip located only at a line-shape threshold is a diagnosis to
publish and a modelling defect to fix, never a flip to bolt on.

## The data, the quantity, and the harness

- **Frame**: `artifacts/opener_evaluation/20260911T161354Z/per_game.parquet`,
  the opener evaluation matched to the active model by
  `nfl_ats.public_board.find_matching_opener_evaluation`, 1,537 games,
  seasons 2020-2025. The match and the feature-table digest are asserted before
  anything is computed; if the active model moves under this lane, the run
  stops rather than mixing two models.
- **Dispersion join**: the `tue_open` rows of
  `nfl_ats.clv.cached_pairing_table(data/market/raw, labels=('tue_open',
  *CLOSE_LABEL_PRIORITY), schedule=...)`, which carry `spread_books`,
  `spread_min`, `spread_max` and `spread_std` for each game's Tuesday
  consensus. Gate: the joined `home_spread` must equal
  `tue_open_home_spread` and `spread_books` must equal `opener_books` on every
  row, to a tolerance of 1e-9, or the run stops.
- **The pick**: `pick_home_at_open_probability_rule`, which is the served
  probability rule -- `home_cover_probability_at_open >= 0.5` with the served
  home-side offset applied. Asserted equal to that comparison before scoring.
  `pick_home_at_open` (the sign of the unoffset residual) and
  `pick_home_at_open_probability_rule_raw` (the probability rule without the
  offset) are the two comparison picks used in D6.
- **The quantity**: forced-pick ATS accuracy at the opener on graded games
  (`margin_vs_open != 0`), in accuracy points (percent).
- **The shape gap**: `GAP = 100 * (accuracy on WHOLE - accuracy on HALF)`,
  computed with lane Q1's between-group week-blocked block bootstrap
  `scripts/overlay_union_line_shape.py::gap_effect`, imported unchanged, with
  `values` set to the per-game correctness indicator. 20,000 draws, blocks are
  (season, week), seed 20260817. `probability_positive` comes from
  `nfl_ats.evidence_conventions.probability_positive_from_draws`.
- Within-week game correlation is ZERO by owner mandate: never estimated,
  never padded.
- `probability_positive` is reported for every cell. The binary "the interval
  contains zero" is never reported and is never a verdict.

### Shapes, frozen (lane Q1's definitions, reused unchanged)

`size = abs(tue_open_home_spread)`.

| Shape | Definition |
|---|---|
| **HALF** | `size % 1 == 0.5` -- the only shape the pool posts |
| **WHOLE** | `size % 1 == 0` |
| **QUARTER** | `size % 1` in `{0.25, 0.75}` -- no book posts one |

`shape_labels` is imported from lane Q1 rather than rewritten, so the two lanes
cannot drift.

## What was already known before the hypotheses were written

Declared so the reader can tell what this lane saw first. Three things, all of
them code reads or descriptive distributions with no accuracy in them:

1. The Tuesday consensus is a **median** over books
   (`nfl_ats.clv.decision_market_consensus`), not a mean, and the pairing table
   already carries `spread_min`, `spread_max` and `spread_std` per game.
   Across all 1,537 games `spread_std` has median 0.224 and a first quartile of
   0.125, so unanimous and near-unanimous consensuses are common. No accuracy
   was computed.
2. `spread_line` is one of the 90 features the active model
   (`market_residual`, `weak_stack`, ridge alpha 10) is fitted on, and the
   opener evaluation **substitutes** `tue_open_home_spread` into that column at
   scoring time while every other feature stays as the feature table built it.
   The training target `ats_margin` is `result` minus the feature table's own
   `spread_line`, which is a different line from the Tuesday opener.
3. The served key-line discrete override
   (`nfl_ats.key_line_pick_read`) fires only when `abs(line)` is **exactly**
   3.0 or 7.0, which are whole numbers, so it is structurally inapplicable to
   every line the pool posts. The archive's own probability is the smooth
   `gaussian_median` read, not the discrete one.

## Hypotheses, declared before measurement

### D1 -- the consensus itself

A whole-number consensus may be the median of books that disagree, in which
case the decision line is a number no book posted and the model's residual is
measured against an artefact. The opposite is equally possible and is declared
here so the sign cannot be chosen afterwards: books "sit" on key numbers and
move the price instead of the line, so a whole-number consensus may be the
*most* unanimous kind, and unanimity would mean the line is at its sharpest.

Tests:

1. **D1a dispersion by shape.** `spread_std`, `spread_max - spread_min` and
   `spread_books` per shape: mean, median, and the share of games where every
   captured book posted the same number (`spread_max == spread_min`).
2. **D1b unanimous games.** The shape gap restricted to unanimous games, and
   restricted to games where books disagreed. **This is the decisive test for
   D1.** On a unanimous game the consensus IS a posted price at every book, so
   the averaging story cannot apply. Declared reading: if the gap collapses
   among unanimous games, the defect is that the model is graded and fitted
   against an averaged line no book posts; if the gap survives unanimity, the
   averaging is not the mechanism and the mechanism is the NUMBER itself.
3. **D1c dispersion strata.** Model accuracy within each shape by dispersion
   tercile of `spread_max - spread_min`, and `GAP` standardised over
   dispersion strata holding at least 20 games of each shape.

### D2 -- key-number exposure and the treatment of the atom

Whole-number lines at 3 and 7 put the push atom on the line. Football margins
are multimodal with mass on the key numbers, so at a 3.0 line the modal outcome
is a push and is deleted from the grade, while at a 3.5 line the same mass
falls on one side. The smooth `gaussian_median` read the archive is graded on
has no atom at all, so it may misplace that mass.

Tests:

1. **D2a where inside WHOLE.** Model accuracy on whole-number lines with
   `abs(line)` in {3, 7} against other whole numbers, and against {3, 7, 10,
   14}; counts per distinct `abs(line)` value per shape. Inside HALF, accuracy
   on the hook lines {2.5, 3.5, 6.5, 7.5} against the rest.
2. **D2b push exposure.** Pushes per shape, asserted rather than assumed (HALF
   must have zero). `GAP` restricted to games decided by at least a full point,
   which both shapes can supply; and `GAP` with the whole-number pushes
   reinstated, once graded home-favourable and once away-favourable.
3. **D2c calibration by shape.** Reliability of the served
   `home_cover_probability_at_open` in five bins, per shape: mean stated
   probability against realised home-cover rate, plus the Brier score, the mean
   stated confidence `max(p, 1-p)` and the realised accuracy. Declared reading:
   if stated confidence is the same in both shapes while realised accuracy is
   not, the read is overconfident on whole numbers and the defect is in the
   probability mapping.
4. **D2d served-override applicability.** How many archive games sit exactly on
   a served atom, what share of the whole-number domain they are, and the model's
   accuracy on them. Descriptive; it says where the discrete override would fire
   if the archive were graded on it.
5. **D2e magnitude, held fixed.** The key numbers are whole, so the two shapes
   need not be drawn from the same magnitude distribution. Line-size mix per
   shape, and `GAP` standardised over integer-floor buckets of `abs(line)`
   capped at 14, and over the coarse buckets `[0,2) [2,4) [4,7) [7,10)
   [10,inf)`, in each case over strata holding at least 20 games of each shape.
   Declared reading as in D1c. Added to this section before any number in this
   lane was computed.

Declared reading for D2 as a whole: if the whole-number deficit is concentrated
at `abs(line)` in {3, 7} and the stated confidence there matches the rest, the
defect is the atom's treatment and the fix belongs in
`src/nfl_ats/discrete_margin_mapping.py`. If the deficit is flat across
whole-number magnitudes, the atom is not the mechanism.

### D3 -- the market-residual target

The model predicts `ats_margin`, which in training is `result` minus the
feature table's `spread_line`, and is applied with `spread_line` replaced by
the Tuesday opener. Two separable consequences, both declared:

1. **D3a the model's point error by shape.** `error = margin_vs_open -
   residual_at_open_served`, the model's fair-line-minus-opener error. Mean
   (bias), standard deviation (noise) and mean absolute error per shape, and
   per shape within coarse magnitude buckets. Declared reading: a shape
   difference in the MEAN is a mis-centred point and a real defect; a shape
   difference only in the SPREAD is games that are harder for anyone.
2. **D3b the line substitution.** `line_shift = tue_open_home_spread -
   spread_line` from the feature table, per shape: mean, sd, and the share of
   games where the two lines differ at all. Model accuracy within each shape by
   `line_shift` bucket, and `GAP` standardised over those buckets. Declared
   reading: if standardising over the shift collapses the gap, the defect is
   that the model is fitted against one line and served against another.
3. **D3c the model's own edge, and its distribution.** `abs(residual_at_open)`
   per shape: mean, median, and the share under 0.5, 1 and 2 points. Model
   accuracy within each shape by `abs(residual_at_open)` bucket, and `GAP`
   standardised over those buckets. **This is the decisive test for "defect
   versus difficulty":** if the accuracy-versus-edge curve is the same in both
   shapes and only the mix of edges differs, the model simply has less to work
   with on whole numbers; if the curve itself sits lower on whole numbers at
   the same stated edge, the model is genuinely worse there.
4. **D3d agreement with the close.** Mean `abs(open_move)` per shape; the share
   of games where `sign(residual_at_open)` agrees with `sign(open_move)` per
   shape; the model's accuracy at the CLOSE per shape and the opener-minus-close
   difference per shape. Declared reading: if the model beats the close equally
   on both shapes but beats the opener only on half points, the whole-number
   opener is the stickier price and the model's apparent opener edge is simply
   not there.

### D4 -- era

Whole-number share by season and captured-book count by season, and `GAP` per
season, and `GAP` standardised over season. If the whole-number share and the
gap are both concentrated in the seasons with the thinnest capture, the gap is
an artefact of how much market was recorded, not of the football.

### D5 -- reliability of the gap itself

1. **D5a odd/even seasons**, the primary: `GAP` on 2021/2023/2025 and on
   2020/2022/2024, each with the same week-blocked bootstrap. With two halves
   there is no correlation to take; two halves agreeing in sign is consistency,
   two halves disagreeing is the honest form of "this does not reproduce".
2. **D5b odd/even weeks within season**: the per-season gap on odd weeks
   against even weeks, Spearman and Pearson over the six pairs,
   Spearman-Brown corrected.
3. **D5c block-level**: inside each (season, week) block, games sorted by
   `game_id` and split alternately; the paired per-block gaps correlated across
   blocks holding both shapes in both halves.

Declared reading, verbatim from the taxonomy: a reliability whose point
estimate and interval sit at zero is the only admissible
`no_split_half_reliability` closing ground, and it would close only the claim
"the shape of the opening line locates where the model is weak". It closes
nothing about the model, the discrete mapping, or the served card.

### D6 -- the served pick rule and the home-side offset (this lane's addition)

Declared before any number is seen. The served pick is not the model's raw
read: `home_side_offset_big_spreads_v2` shifts the centre on big spreads, and
the active evaluation reports 401 games with a non-zero offset and 46 opener
picks changed by it. An offset fitted on spread magnitude could easily land
asymmetrically on the two shapes.

Tests: non-zero offset counts and mean absolute offset per shape; `GAP`
computed three times, on the served probability pick, on the raw probability
pick without the offset, and on the sign of the unoffset residual. Declared
reading: if the gap is already there on the raw pick, the offset is not the
mechanism; if the gap appears only after the offset, the defect is the offset
policy and the fix belongs there.

### D7 -- home/away base rate and a direction-matched null (this lane's addition)

Declared before any number is seen. Every forced pick has a side. If the
realised home-cover rate differs between the shapes and the model's picks tilt
one way, part of the gap is a base-rate artefact rather than a statement about
the model's read.

Tests: realised home-cover rate and the model's home-pick rate per shape; and a
side-matched permutation null in which, inside each shape, the model's picks
are replaced by a random assignment with the same number of home picks, 2,000
draws, seed 20260911. The model's excess over that null is reported per shape,
and `GAP` is recomputed on the excess. Declared reading: if the excess over the
null is the same in both shapes, the gap is the base rate and the model's read
is equally good on both.

### D8 -- is it the model, or are the games harder? (this lane's addition)

Declared before any number is seen. Two reference predictors that exist in the
same frame and know nothing about the model:

1. `oracle_correct_at_open` -- take the side the market moved toward between
   the opener and the close, graded at the opener. It is not available before
   kickoff and can never be a rule; it is a reference that measures how much
   signal the market itself extracted from these games.
2. `correct_at_close_probability_rule` -- the same model graded at the close.

Both are measured by shape and as a gap. Declared reading: if the movement
oracle is also weaker on whole-number lines by a similar margin, whole-number
games are harder for everyone and the gap is not a defect of this model; if
the oracle is flat across shapes while the model is not, the weakness is the
model's.

## What would count as an explanation

Declared now so it cannot be chosen afterwards. `GAP` is negative, so the
comparison is on magnitude. A hypothesis **EXPLAINS** the gap if, under its
correction, `abs(GAP)` falls below a third of its raw value; it **CONTRIBUTES**
if `abs(GAP)` falls by between a tenth and a third; otherwise it does not
account for the gap. These are descriptive bands for a written verdict, not
decision bars, and no cell is rejected, closed or withheld on them.

## Decision rule, declared before the numbers

The action this lane may take is bounded in advance.

- Nothing here touches the served card, the active model, the published
  forecast, the discrete mapping as served, or any overlay. This lane writes a
  diagnosis and registry rows.
- **If the defect is the averaged consensus (D1b or D3b):** the named fix
  direction is the one MOD-18 already carries -- grade and fit against a single
  book's posted half-point line rather than a cross-book median, so the model
  is fitted and served on the same line convention.
- **If the defect is the atom's treatment (D2a with D2c):** the fix belongs in
  `src/nfl_ats/discrete_margin_mapping.py` and the key-line read, not in the
  card.
- **If the defect is the home-side offset (D6):** the fix belongs in the offset
  policy.
- **If it is difficulty rather than defect (D3c or D8):** the finding is that
  the archive-graded headline mixes two populations of different difficulty,
  and the reader-facing consequence is that the pool's own shape is the
  stronger one -- a fact to publish, not a rule.
- **If no hypothesis accounts for it:** it is recorded
  `unresolved_below_power`, published as a diagnosis, and it explicitly may NOT
  gate any pick.

No threshold -- 0.5, 0.90 or 0.95 -- is used as a bar on what may be MEASURED
or RECORDED, and no promotion bar governs a play.

## Closing-grounds taxonomy (binding, verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval on
the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". The registry code
hard-rejects inadmissible closures; if a record command errors, the verdict is
wrong, not the validator. Verdicts flow through `nfl-ats weak-signals record` /
`nfl-ats rotation record-look`, never through prose in a doc. Never state that
something needs N more games. Never use 95% or 0.90 as a decision bar; decide
on expected value. Within-week game correlation is ZERO by owner mandate.
Football margins are multimodal with mass on the key numbers, and a served
cover probability is computed against the discrete margin distribution
conditional on the line. No unexplained threshold flips on the played card: a
dip located at a line-shape threshold is a diagnosis to publish and a modelling
defect to fix, never a rule to bolt on.

## Reuse discount, disclosed before the run

The same mined 1,537-game Tuesday-opener archive that lanes K, T, H, S, V, C2,
P1 and Q1 were selected on. The observation this lane explains was itself
produced post-hoc by lane P1 and measured by lane Q1; this lane predeclares the
MECHANISM tests, not the observation, and it cannot un-see the observation.
These are descriptive reused-era measurements, not independent confirmation.
No rotation window is spent: a diagnostic look at an already-scored archive,
the precedent `docs/player_arrests_policy_eval.md` and MOD-17 both set.

## Leakage gates

- The active model and its opener evaluation are matched by
  `find_matching_opener_evaluation` and the feature-table digest is compared to
  `artifacts/active_ats_model.json` before anything is computed.
- The dispersion join is gated: `home_spread` must equal
  `tue_open_home_spread` and `spread_books` must equal `opener_books` on all
  1,537 rows.
- `pick_home_at_open_probability_rule` is asserted equal to
  `home_cover_probability_at_open >= 0.5`.
- D2a's hook split, D3a's error decomposition and D8's movement oracle
  CONDITION ON THE REALISED RESULT or on information unavailable before
  kickoff. They are labelled diagnostics in the code, the artifact and the
  write-up, and none of them may become a rule.

## Where the code lives

- `scripts/whole_number_line_model_diagnosis.py` -- this lane's run.
- `artifacts/research/laneQ2/` -- its artifacts.
- `scripts/mod18_pool_shaped_read_eval.py` and
  `scripts/overlay_union_line_shape.py` -- lanes P1 and Q1's helpers, imported.

---

# Measured (2026-09-11, after the predeclaration above was frozen)

Everything above this line was written first; its bytes are digested into
`artifacts/research/laneQ2/reproduction.json` as `predeclaration_sha256`
`d577459e34d7cae01a48fb3ae8f78ca12241c2e201b6bc73c47ffe09ec9d6b9e` and copied
verbatim to `artifacts/research/laneQ2/predeclaration.md`, which still carries
exactly those bytes and no measured number.

Run: `scripts/whole_number_line_model_diagnosis.py --stage run|record|report`,
against the active model `7786467eabe418b8` and its opener evaluation
`artifacts/opener_evaluation/20260911T161354Z`, 1,537 games over 107 (season,
week) blocks, seasons 2020-2025. All four predeclared gates passed: the
evaluation matched the active model, the feature digest matched, the Tuesday
consensus join reproduced the archive's decision line and book count on all
1,537 rows, and the served pick equalled the served probability rule.

## The decision first

**The fix direction is the line the model is graded and fitted against, not the
discrete mapping.** Three numbers carry that:

1. **The atom is exonerated.** Whole-number lines that sit exactly on 3 or 7 --
   where the push mass lands on the line -- are the *strongest* part of the
   whole-number domain, not the weakest: **+2.355 [-6.011, +10.488],
   `probability_positive` 0.712** against other whole numbers, 53.41% on their
   249 graded games against 51.99% for whole numbers overall. Strip the key
   numbers out of both shapes and the shape gap gets **bigger**, -6.395
   [-13.307, +0.657]. Reinstating the 34 pushes, or grading both shapes only on
   games decided by a full point, moves it from -4.538 to between -4.248 and
   -4.792. Nothing here asks for a change in
   `src/nfl_ats/discrete_margin_mapping.py`.
2. **The whole-number opener is an away-biased price, and three quarters of the
   error is the line rather than the model.** Home teams beat a whole-number
   Tuesday opener by **+0.671 points** on average and a half-point one by
   **-0.031** -- a 0.702-point swing in the line's own bias. The model's mean
   residual moves the other way by 0.226 points (+0.085 on half-point openers,
   -0.141 on whole-number ones). The two compound into a 0.929-point swing in
   the model's point error (-0.116 against +0.813) that is the same sign in all
   five magnitude buckets. The model picks the home side on 42.1% of
   whole-number openers, where home covered 51.2% of the time.
3. **The model's edge is a book-disagreement edge, and the pool posts one
   book's number.** On half-point openers the model reads **57.96%** on the 647
   games where the captured books disagreed and **51.79%** on the 195 where
   every book posted the same number: **+6.165 [-0.789, +13.340],
   `probability_positive` 0.958**. Pooled over both shapes the same contrast is
   +4.195 [-2.074, +10.504], `probability_positive` 0.906. Among unanimous
   openers the shape gap itself falls from -4.538 to **-1.795 [-13.587,
   +9.714]**, and both shapes sit near a coin flip (51.79% and 50.00%).

So: **grade and fit against a single book's posted half-point line**, which is
the step `docs/mod18_pool_shaped_read.md` already names and which lane P2 is
running today as `docs/single_book_opener_grade.md`. The archive's
decision line is a median over a book set that disagrees on three games in
four, the number the pool actually posts is one draw from that set, and the
part of the model's measured edge that is largest is exactly the part measured
where the median is furthest from any posted price.

Two secondary consequences for the decision:

- **The served probability does not know any of this.** Mean stated confidence
  is 55.85% on half-point openers and 55.58% on whole-number ones -- the same
  number -- against realised accuracy of 56.53% and 51.99%. Confidence minus
  accuracy is **-0.7 points on half-point lines and +3.6 points on
  whole-number ones**. That is the flat-confidence defect, located on a shape,
  and it belongs in the Model page's weak-spots table.
- **Between a third and two thirds of the gap is not this model's to fix.** The
  market's own movement signal is also weaker on whole-number openers (55.89%
  against 54.11%, gap **-1.774 [-7.739, +3.990]**, 39% of the model's gap), and
  the same model regraded at the close still carries **-2.811 [-7.750,
  +2.107]**, 62% of it.

**Nothing here may touch the played card.** The pool posts a half point and
only a half point, so every served pick already lives on the stronger shape;
the reader-facing consequence is that the archive-graded headline blends two
populations of different difficulty, not that any pick should change. All **40
recorded cells** are `unresolved_below_power` with no closing ground.

## The observation, re-measured on the active model

| Shape | Games | Graded | Pushes | Model accuracy | Home covered |
|---|---:|---:|---:|---:|---:|
| **half point** (the pool's shape) | 842 | 842 | 0 | **56.53%** | 48.34% |
| **whole number** | 661 | 627 | 34 | **51.99%** | 51.20% |
| quarter point | 34 | 34 | 0 | 50.00% | 52.94% |
| every line | 1,537 | 1,503 | 34 | 54.49% | 49.63% |

`GAP` = whole minus half = **-4.538 [-9.612, +0.484]**,
`probability_positive` 0.040. Lane Q1 measured -4.538 [-9.641, +0.500] on the
previous model's evaluation; the point estimate is unchanged to three decimals
on a differently fitted model.

## D1 the consensus: an interaction, not a composition

Dispersion barely differs by shape -- mean book range 1.140 on half-point
openers against 0.958 on whole-number ones, median `spread_std` 0.213 against
0.229, unanimous share 23.2% against 17.7% -- and standardising the gap over
dispersion strata leaves it at **-4.787 [-9.982, +0.497]**, slightly larger
than raw. Composition accounts for none of it.

What the split does show is an interaction:

| Books | half point | whole number | shape gap |
|---|---:|---:|---|
| every book posted the same number | 51.79% (195) | 50.00% (110) | **-1.795** [-13.587, +9.714] P+ 0.376 |
| the books disagreed | 57.96% (647) | 52.42% (517) | **-5.542** [-11.119, -0.037] P+ 0.024 |

The model's edge, on the shape the pool posts, is **+6.165 [-0.789, +13.340]**
accuracy points larger where the books disagreed than where they did not. The
shape gap exists among disagreeing games and nearly vanishes among unanimous
ones; unanimity cuts it by 60%, which is the largest single move any correction
in this lane produced and still short of the predeclared third.

## D2 the key numbers and the atom: accounts for none of it

| Contrast | Effect | P+ |
|---|---|---:|
| whole lines on 3 or 7, against other whole lines | **+2.355** [-6.011, +10.488] | 0.712 |
| whole lines on 3, 7, 10 or 14, against other whole lines | +2.485 [-5.401, +10.259] | 0.735 |
| half-point lines beside 3 or 7, against other half-point lines | -1.243 [-8.238, +5.780] | 0.360 |
| the shape gap with the key numbers removed from both shapes | **-6.395** [-13.307, +0.657] | 0.038 |
| the shape gap on games decided by at least a point | -4.248 [-9.704, +1.068] | 0.060 |
| the shape gap with the pushes reinstated, home-favourable | -4.792 [-9.747, +0.139] | 0.029 |
| the shape gap with the pushes reinstated, away-favourable | -4.490 [-9.680, +0.600] | 0.044 |

The predeclared invariant held, measured rather than assumed: **all 34 pushes
sit on whole-number lines and half-point lines have zero.** 272 games -- 41.1%
of the whole-number domain -- sit exactly on a served atom, and the model is
53.41% on them, above the whole-number average. Per distinct line size, the
whole-number weakness is spread across the non-key numbers (48.78% on 2, 48.28%
on 4, 46.15% on 8, 25.00% on the 20 games at 9) while 3 reads 52.51% and 7
reads 55.71%.

Standardising the gap over integer-floor buckets of the line size leaves it at
**-4.055 [-9.514, +1.702]**, a fall of 10.6% -- the low end of the predeclared
CONTRIBUTES band and no more.

## D3 the market-residual target: the point is mis-centred, and mostly by the line

The model's point error at the opener, `margin_vs_open` minus the served
residual, decomposes as:

| Shape | Home beat the opener by | Model's mean residual | Point error | Error sd | Mean absolute error |
|---|---:|---:|---:|---:|---:|
| half point | -0.031 | +0.085 | **-0.116** | 12.71 | 9.94 |
| whole number | **+0.671** | **-0.141** | **+0.813** | 13.41 | 10.46 |

0.702 of the 0.929-point swing is the line's own bias and 0.226 is the model's
residual leaning the other way -- 76% line, 24% model. The whole-number mean
error is positive in every one of the five magnitude buckets (+0.62, +0.73,
+1.28, +0.91, +0.08) while the half-point mean error is negative in three of
them.

**The line substitution is real but is not the mechanism.** The feature table's
`spread_line`, which the training target is measured against, equals the close
on 68.1% of games (mean absolute distance 0.196) and equals the Tuesday opener
on 24.7% (mean absolute distance 0.998), so the model is fitted against
something very close to the closing line and served against the Tuesday
opener. That substitution is the same size on both shapes (mean absolute shift
1.009 against 0.986; identical on 26.0% against 24.4%), and standardising over
it leaves the gap at **-4.888 [-10.054, +0.420]**. What the strata do show is
that the gap is concentrated where the opener never moved: -12.449 [-22.663,
-1.691] on the 371 games whose opener equals the training line, against -0.092
on the 427 where the two differ by under a point.

**The model's edge is the same size on both shapes; only its yield differs.**
Mean `abs(residual_at_open)` is 1.631 on half-point openers and 1.594 on
whole-number ones, and the shares under half a point, a point and two points
match to within a point. Standardising the gap over edge buckets leaves it at
**-4.518 [-9.643, +0.673]**. The whole deficit sits in one bucket: at a stated
edge of one to two points the model reads 57.32% on half-point lines and
**46.56%** on whole-number ones, a gap of **-10.756 [-19.862, -2.044]**,
`probability_positive` 0.007 on 435 games.

**The market moves the same amount on both shapes.** Mean absolute
open-to-close move is 0.971 on half-point openers and 0.951 on whole-number
ones; the model's residual agrees with the direction of the market's own move
55.4% of the time on half points and 54.6% on whole numbers. The stickiness
story is not supported.

## D4 era: accounts for none of it

The whole-number share swings from 31.7% (2020) to 52.5% (2022) and the mean
captured book count from 8.9 (2024) to 19.1 (2022), so the mix genuinely
differs by season. Standardising over season leaves the gap at **-4.859
[-10.240, +0.354]**. The per-season gaps are -1.204, -2.520, -12.060, -1.653,
+0.489 and -11.435 for 2020 through 2025: five of six negative, two of them
large, none decisive on its own.

## D5 reliability: no closing ground, and the gap reproduces

- **D5a odd/even seasons**, the predeclared primary: **-5.151 [-12.554,
  +2.342]** (P+ 0.090) on 2021/2023/2025 and **-3.806 [-10.585, +2.954]**
  (P+ 0.135) on 2020/2022/2024. **Both halves negative.**
- **D5b odd/even weeks within season**: Pearson **+0.692** (Spearman-Brown
  +0.818), Spearman +0.486 across the six pairs. The gap has positive
  split-half reliability at the season level, so the
  `no_split_half_reliability` closing ground is **not** available and is not
  claimed.
- **D5c block-level**: Pearson +0.056 [-0.128, +0.238] on 100 blocks. A block
  half holds about seven games, so this construction returns a near-zero
  correlation for a real effect too; it settles nothing either way.

## D6 the home-side offset: not the mechanism, and it helps

| Pick | Shape gap | P+ |
|---|---|---:|
| the sign of the unoffset residual | **-7.969** [-13.184, -2.819] | 0.001 |
| the probability rule without the offset | **-5.577** [-10.622, -0.529] | 0.014 |
| the served probability rule (the offset applied) | **-4.538** [-9.612, +0.484] | 0.040 |
| games the offset left alone | -4.336 [-10.617, +1.807] | 0.086 |

The served home-side offset touches 28.9% of whole-number openers and 23.5% of
half-point ones and changes 28 whole-number picks against 16 half-point ones.
It **reduces** the shape gap by 1.04 points rather than creating it, and the
gap is larger still on the raw residual sign. No fix is indicated in the offset
policy.

## D7 home/away base rate: contributes 9%

| Shape | Model picks home | Home covered | Side-matched null | Observed | Excess |
|---|---:|---:|---:|---:|---:|
| half point | 43.82% | 48.34% | 50.21% | 56.53% | **+6.327** |
| whole number | 42.11% | 51.20% | 49.81% | 51.99% | **+2.182** |

The null replaces the model's picks with a random assignment holding the same
number of home picks, 2,000 draws, seed 20260911; the permutation and analytic
nulls agree to 0.02 points. The gap in excess over that null is **-4.144
[-9.384, +0.948]**, so the home/away base rate accounts for 0.394 of the
4.538-point gap -- 8.7%, below the predeclared CONTRIBUTES band.

## D8 is it the model or the games? Between a third and two thirds is the games

| Reference | half point | whole number | Gap | P+ |
|---|---:|---:|---|---:|
| the side the market moved toward | 55.89% (637) | 54.11% (462) | **-1.774** [-7.739, +3.990] | 0.273 |
| the same model graded at the close | 54.90% (827) | 52.09% (647) | **-2.811** [-7.750, +2.107] | 0.131 |

Both references condition on information that does not exist before kickoff and
neither may become a rule. The market's own movement signal carries 39% of the
model's gap and the same model regraded at the close carries 62% of it: games
whose Tuesday opener landed on a whole number are harder for a market-residual
predictor generally, not only for this one.

## Summary against the predeclared bands

`GAP` is -4.538. A hypothesis EXPLAINS if `abs(GAP)` falls below a third of
that under its correction, CONTRIBUTES if it falls by a tenth to a third.

| Hypothesis | `GAP` under its correction | Verdict |
|---|---|---|
| D1 unanimous books only | -1.795 (a 60% fall) | largest mover; short of EXPLAINS |
| D1c standardised by dispersion | -4.787 | does not account |
| D2a key numbers removed | -6.395 | does not account |
| D2b push handling | -4.248 to -4.792 | does not account |
| D2e standardised by line size | -4.055 | **contributes, 11%** |
| D3b standardised by line substitution | -4.888 | does not account |
| D3c standardised by model edge | -4.518 | does not account |
| D4 standardised by season | -4.859 | does not account |
| D6 the home-side offset | -5.577 without it | does not account; the offset helps |
| D7 side-matched null | -4.144 | contributes, 9% |
| D8 a market-only reference | -1.774 of it is in the market too | 39% is not this model's |

**Verdict: no single correction explains the gap, and the mechanism is located
rather than closed.** The whole-number Tuesday opener is a price the home side
beat by 0.67 points on average while the model leaned 0.14 points the other
way; the model states the same confidence on both shapes regardless; and the
edge the model does have is concentrated on games where the captured books
disagreed, which is precisely where a cross-book median is least like the
single posted price the pool uses. It is not the push atom, not the key
numbers, not the home-side offset, not season mix, and not the size of the
model's own edge.

## Recorded

40 rows under family `whole_number_line_model_diagnosis_v1`, every one
`unresolved_below_power` with no closing ground, each carrying a plain-English
summary and the post-hoc discount in its notes so it can never be pooled with a
predeclared family. The registry went from 6,440 to 6,480 signals with zero
failures. The same 40 argv were first executed against an isolated registry
under the session scratchpad and all 40 were accepted, so the emitted argv is
admissible to the real validator. The two D8 rows carry an explicit note that
they condition on information unavailable before kickoff and can never become
rules.

No rotation window spent: a diagnostic look at an already-scored archive, the
precedent `docs/player_arrests_policy_eval.md` and MOD-17 both set.

## What this lane leaves open, named now

Two follow-ups, both about the line rather than about a threshold:

1. **Regrade the archive against a single book's posted line** --
   `docs/single_book_opener_grade.md` (lane P2) is doing exactly this today.
   The pairing table already carries `spread_min`, `spread_max` and every
   book's quote is in `data/market/raw`. Grading the same model against one
   consistently captured book, on that book's own half-point prices, would say
   directly how much of the 56.53% half-point read survives when the decision
   line is a price somebody actually posted. This lane's D1 interaction is a
   reason to expect the answer to move: the model's half-point edge is 6.165
   accuracy points larger where the captured books disagreed, and a single
   book's price is one draw from that disagreement rather than its median.
2. **Give the stated confidence something to vary with.** A read that says
   55.6% on a population it hits 52.0% of the time, and 55.9% on one it hits
   56.5% of the time, is a calibration defect the weak-spots table can publish
   today and the probability mapping should eventually absorb. No flip belongs
   on the card in the meantime.
