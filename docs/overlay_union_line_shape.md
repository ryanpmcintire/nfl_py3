# Why the played overlays split on the shape of the opening line

Predeclaration frozen 2026-09-11, before any cover rate, accuracy or effect in
this lane was computed. Family `overlay_union_line_shape_mechanism_v1`.
Everything above the "Measured" heading was written first and its bytes are
digested into `artifacts/research/laneQ1/reproduction.json` as
`predeclaration_sha256`.

## The observation this lane has to explain

Reported by lane P1 (`docs/mod18_pool_shaped_read.md`, section "An
unpredeclared finding this lane fell over"), recorded in family
`mod18_overlay_union_by_line_shape_v1`, and **not predeclared there**: on the
active model's opener evaluation
(`artifacts/opener_evaluation/20260910T211255Z/per_game.parquet`, 1,537 games,
decision line `tue_open_home_spread`), the played three-member overlay union --
coach fade, division revenge, player arrests -- reads

| Domain | n | Overlay union vs the model's own pick | P+ |
|---|---:|---|---:|
| half-point opener lines | 842 | -1.425 [-3.933, +1.160] | 0.136 |
| everything else | 661 graded | +4.841 [+1.506, +8.235] | 0.999 |

The owner's pool only ever posts a half point. So the overlays' entire measured
contribution on this archive sits on lines the pool never posts.

That is a finding about the ARCHIVE until someone names a mechanism. The
opener line here is a cross-book consensus, not a posted price, so "the line
was a half point" is a property of how the capture averaged nine books, not a
property of the football game. A rule conditioned on it would be a threshold
with no named mechanism, which the repository forbids on the played card. This
lane's job is to find out which of several ordinary explanations produces the
split, and it is written before any of them is measured.

## What this lane will and will not do

It will not change the served card, the active model, any published forecast,
or any overlay's membership rule. It produces a mechanism verdict and registry
rows. If a mechanism is found that is a property of a GAME rather than of the
consensus line's granularity, the output is a named mechanism and a proposal
for a predeclared prospective challenger -- never a flip bolted onto the card.

## The data, the effect, and the harness

Everything is imported from lane P1 rather than reimplemented.

- **Frame**: `artifacts/research/laneP1/scored.parquet`, 1,537 games over 107
  (season, week) blocks, seasons 2020-2025. It carries the replayed served
  probability `p_S3` (gated at 1e-9 against the archive's own served number),
  the composed played card `card_S3`, `opener_books`, `margin_vs_open` and
  `tue_open_home_spread`.
- **Effect**: `scripts/mod18_pool_shaped_read_eval.py::comparison`, imported
  unchanged -- forced-pick ATS accuracy at the OPENER, paired per game,
  week-blocked block bootstrap over (season, week) blocks, 20,000 draws, seed
  20260817, `probability_positive` from
  `nfl_ats.evidence_conventions.probability_positive_from_draws`. Candidate is
  `card_S3` (the played union), baseline is `p_S3 >= 0.5` (the model's own
  pick). Positive favours the overlays. Games with `margin_vs_open == 0` are
  excluded by that function; H2 exists because that exclusion is not
  shape-neutral.
- Within-week game correlation is ZERO by owner mandate: never estimated,
  never padded.
- `probability_positive` is reported for every cell. The binary "the interval
  contains zero" is never reported and is never a verdict.

### Shapes, frozen

`size = abs(tue_open_home_spread)`.

| Shape | Definition |
|---|---|
| **HALF** | `size % 1 == 0.5` -- the only shape the pool posts |
| **WHOLE** | `size % 1 == 0` |
| **QUARTER** | `size % 1` in `{0.25, 0.75}` -- a consensus artifact; no book posts one |

Lane P1's `OFF_POOL` is `WHOLE + QUARTER` pooled. Splitting it is itself part
of H1: if the positive side is carried by quarter points, the split is a
property of how the consensus was averaged and of nothing else.

The lane's headline quantity is the **shape gap**

    GAP = effect(WHOLE) - effect(HALF)

in accuracy points, and every hypothesis below is a candidate explanation for
it. `GAP` is also computed as `effect(OFF_POOL) - effect(HALF)` so it is
comparable to lane P1's numbers.

## Hypotheses, declared before measurement

### H1 -- composition

The whole-number games are a different population: a different season mix, or a
different number of captured books, or both. A consensus over nine books lands
on a whole number under different conditions than a consensus over four.

Tests:

1. **H1a season mix.** Share of each shape per season; the union's effect
   within each shape per season.
2. **H1b book count.** Distribution of `opener_books` per shape; the union's
   effect within each shape, stratified by `opener_books` (strata: the observed
   values, pooled into terciles if any stratum holds fewer than 40 games of a
   shape).
3. **H1c direct standardisation.** Recompute `GAP` as the sample-weighted
   average of within-stratum gaps over (season x book-count tercile) cells that
   hold at least 15 games of each shape. Declared reading: if the standardised
   gap is materially smaller than the raw gap, composition carries it; if it is
   unchanged, composition does not.

### H2 -- push handling

A whole-number line can push; a half-point line cannot. `comparison` drops
`margin_vs_open == 0`, so the WHOLE domain is graded only after its exactly-on-
the-line games have been removed, while the HALF domain keeps every game it
has. Worse, the closest a half-point game can come to the line is 0.5 -- the
hook -- and those games are genuine coin flips clustered at the key numbers,
while the equivalent whole-number games have been deleted as pushes. That
asymmetry alone would dilute any real effect on HALF and concentrate it on
WHOLE.

Tests:

1. **H2a push counts.** Pushes per shape. HALF must have zero; that is
   asserted, not assumed.
2. **H2b a common sub-domain.** The union's effect on each shape restricted to
   `abs(margin_vs_open) >= 1.0`, which both shapes can supply and which removes
   the hook games from HALF exactly as pushes were removed from WHOLE.
   Declared reading: if `GAP` collapses on this sub-domain, push handling is
   the mechanism.
3. **H2c pushes reinstated.** Regrade WHOLE with its pushes added back, once
   graded home-favourable (line shifted +0.5) and once away-favourable (line
   shifted -0.5), so the whole-number domain is graded as a half-point domain
   would be. Declared reading: if the WHOLE effect survives both shifts, push
   exclusion is not the mechanism.
4. **H2d hook decomposition.** Within HALF, the union's effect on
   `abs(margin_vs_open) == 0.5` against `> 0.5`. **This conditions on the
   outcome and is a diagnostic only.** It can never be a rule, because the
   margin is not known when the card is submitted. It is declared here so the
   temptation to read it as one is foreclosed in advance.

### H3 -- one member, or all three

The union is three rules with different firing conditions. Coach fade and
division revenge fire off schedule facts; the arrest policy fires off an
incident window. If the split belongs to one member, it is that member's
mechanism to explain, not the union's.

Test: each member's flip set rebuilt separately
(`nfl_ats.overlay_composition.run_overlays` for coach fade and division
revenge, `reconstruct_arrest_flip_set` for the arrests policy -- the same calls
`composed_picks` makes), each scored against the model's own pick within each
shape, with its flip count per shape reported alongside.

### H4 -- spread magnitude

The key numbers are whole: 3, 7, 10, 14. A whole-number consensus line is
therefore not drawn from the same magnitude distribution as a half-point one,
and the overlays may simply help more on some magnitudes.

Tests:

1. **H4a magnitude mix.** Distribution of `size` per shape.
2. **H4b within-bucket effects.** The union's effect within each shape inside
   integer-floor buckets of `size` (`floor(size)`, capped at 14 and above),
   reported for every bucket holding at least 20 games of each shape; and
   inside a coarse set `[0,2) [2,4) [4,7) [7,10) [10,inf)` as a robustness
   read.
3. **H4c direct standardisation.** `GAP` recomputed as the sample-weighted
   average of within-bucket gaps. Declared reading as in H1c.

### H5 -- split-half reliability of the gap itself

If the shape gap has no split-half reliability, it is noise, and "line shape
predicts where the overlays help" is a refuted mechanism -- for that claim
only, and for nothing else about the overlays.

Tests, both declared, the first primary:

1. **H5a block-level.** Inside each (season, week) block, sort games by
   `game_id` and assign alternating games to half A and half B. Compute the
   block's gap in each half (mean of `card_S3 correct - raw correct` on WHOLE
   minus the same on HALF, on graded games). Pearson and Spearman correlation
   of the paired block gaps across blocks holding at least one game of each
   shape in both halves, Spearman-Brown corrected.
2. **H5b season-level.** Per season, the gap computed on odd weeks and on even
   weeks; Spearman correlation of the six pairs, Spearman-Brown corrected.
3. **H5c odd/even seasons.** `GAP` computed separately on seasons 2021, 2023,
   2025 and on 2020, 2022, 2024, each with the same week-blocked bootstrap.
   This is a two-half agreement read rather than a correlation -- with two
   halves there is no correlation to take -- and it is reported as the most
   directly powered of the three. Two halves agreeing in sign is consistency;
   two halves disagreeing in sign is the honest form of "this does not
   replicate across the archive".

Declared reading: a reliability whose point estimate and interval sit at zero
is an admissible `no_split_half_reliability` closing ground for the claim
"the shape of the opening line predicts where the overlays help", recorded as
such. It closes nothing about the overlays themselves, nothing about the
discrete-mapping family, and nothing about the HALF-domain reading.

### H6 -- flip rate against hit rate (this lane's own addition)

Declared before any number is seen. On a graded game the union either leaves
the pick alone, contributing zero, or flips it, contributing +1 when the flip
is right and -1 when it is wrong. So exactly

    effect (accuracy points) = 100 * (flips / n) * (2 * hit - 1)

where `flips` is the number of graded games the union flips in the domain and
`hit` is the share of those flips the union gets right. The shape gap
decomposes into a **flip-rate** term and a **hit-rate** term. Declared
reading: if the flip rate is the same in both shapes and the hit rate differs,
the split is about which games the rules land on; if the flip rate differs, the
split is about how often they fire, and the firing conditions are where to
look. This identity is asserted numerically against the bootstrap point
estimate and the run stops if it does not hold to 1e-9.

### H7 -- direction and base rate (this lane's own addition)

Declared before any number is seen. Every overlay flip has a direction: to the
home side or to the away side. If the realised home-cover rate differs between
the shapes and the union flips predominantly one way, the gap is a base-rate
artifact and not a statement about the rules' game selection at all.

Tests:

1. **H7a** realised home-cover rate per shape, and the share of union flips
   that move to the home side per shape.
2. **H7b a direction-matched permutation null.** Within each shape, replace the
   union's flip set with a random set of the same size that flips the same
   number of games to the home side and the same number to the away side,
   drawn from that shape's games with the matching raw pick. 2,000 draws, seed
   20260911. Report where the observed effect sits in that null as a
   `probability_positive`-style share. Declared reading: if the observed effect
   sits in the middle of the direction-matched null in both shapes, the gap is
   direction and base rate, not game selection.

## What would count as an explanation

Declared now so it cannot be chosen afterwards. A hypothesis EXPLAINS the gap
if, under its correction, `GAP` falls to under a third of its raw value; it
CONTRIBUTES if `GAP` falls by between a tenth and a third; otherwise it does
not account for the gap. These are descriptive bands for a written verdict,
not decision bars, and no cell is rejected, closed or withheld on them.

## Decision rule, declared before the numbers

The action this lane may take is bounded in advance:

- If a mechanism is found that is a property of the GAME (magnitude, push
  exposure, a single member's firing condition), the finding is that mechanism,
  and the proposal is a predeclared prospective challenger. The served card is
  not touched by this lane under any result.
- If the gap is composition, push handling or direction, the finding is that
  lane P1's unpredeclared table measures the archive rather than the overlays,
  and the HALF-domain reading is the one that describes the pool.
- If no hypothesis accounts for it, the gap stays an unexplained threshold, it
  is recorded `unresolved_below_power`, and it explicitly may NOT be used to
  gate overlay membership -- an accuracy split located only at a line-shape
  threshold is a diagnosis to publish and a defect to understand, never a flip
  to bolt on.

No threshold -- 0.5, 0.90 or 0.95 -- is used as a bar on what may be MEASURED
or RECORDED.

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
on expected value. Within-week game correlation is ZERO by owner mandate. No
unexplained threshold flips on the played card: a rule that changes a pick must
name a mechanism, and a dip located only at a line-shape threshold is a
diagnosis to publish and a defect to understand, never a flip to bolt on.

## Reuse discount, disclosed before the run

The same mined 1,537-game Tuesday-opener archive lanes K, T, H, S, V, C2 and P1
were selected on. The shape split itself was noticed post-hoc by lane P1 after
its signs were visible; this lane predeclares the MECHANISM tests, not the
observation, and it cannot un-see the observation. These are descriptive
reused-era measurements, not independent confirmation. No rotation window is
spent: this is a diagnostic look at an already-scored archive, the precedent
`docs/player_arrests_policy_eval.md` and MOD-17 both set.

## Leakage gates

- The frame is lane P1's replay, already gated at 1e-9 against the archive's
  served probability; this lane recomputes nothing upstream of the card.
- The union rebuilt member-by-member for H3 must reproduce `card_S3` exactly
  when the three members are re-unioned; the run stops otherwise.
- H2d and H7a condition on realised outcomes and are labelled diagnostics in
  the code, the artifact and the write-up; neither may become a rule.
- The H6 identity is asserted to 1e-9 against the measured point estimate.

## Where the code lives

- `scripts/overlay_union_line_shape.py` -- this lane's run.
- `artifacts/research/laneQ1/` -- its artifacts.
- `scripts/mod18_pool_shaped_read_eval.py` -- lane P1's helpers, imported.

---

# Measured (2026-09-11, after the predeclaration above was frozen)

Everything above this line was written first; its bytes are digested into
`artifacts/research/laneQ1/reproduction.json` as `predeclaration_sha256`
`4d1115df0c5edfedb7e666c72c0c14fb53df1fccecd82c54129954cd2953c934` and copied
verbatim to `artifacts/research/laneQ1/predeclaration.md`.

Run: `scripts/overlay_union_line_shape.py --stage run|record|report`, against
`artifacts/research/laneP1/scored.parquet`
(`8f5f85b5e4fd1e7e0cd29c0ec3ecd26971ceb9772eabca6c88149856de336454`), 1,537
games over 107 (season, week) blocks, seasons 2020-2025. The three members
re-unioned reproduce the played card `card_S3` exactly, asserted before
anything was scored.

## The decision first

**The overlays may not be gated on the shape of the opening line, and the
reason is now a measured one rather than a rule of thumb: every correction that
would make the split a property of the line failed to move it.** Season mix and
captured book count leave it at +5.956 against a raw +5.891. Spread magnitude
leaves it at +6.841 (fine buckets) or +6.093 (coarse). Grading both shapes only
on games decided by at least a full point -- the correction that removes the
push asymmetry -- leaves it at +6.174. None of the three accounts for any of
it.

What did move is smaller and points elsewhere. **The model's own pick is 4.538
accuracy points WORSE on whole-number opener lines than on half-point ones**
(-4.538 [-9.641, +0.500], `probability_positive` 0.039 for the whole-number
side being better -- so it is the half-point side that is stronger). A flip rule
firing on one game in six therefore has a far higher bar to clear on the shape
the pool posts. Measured against a random flip of the same size and the same
home/away direction mix, **the played union is positive on BOTH shapes**: +0.763
accuracy points on half-point lines and +5.285 on whole-number lines. That
correction accounts for 1.369 of the 5.891-point gap, 23%.

The rest of the gap is **located** -- not explained -- in one member. The
division-revenge rule carries 4.615 of the 5.891 points on its own, 78%.

So, for the decision:

1. **The played union stays.** On the whole archive it reads **+1.331
   [-0.727, +3.393], `probability_positive` 0.898** on 1,503 graded games and
   272 flips. Declining a candidate that is 90% likely better is taking the
   other side of a 90/10 bet.
2. **Nothing about line shape may touch the card.** The split survives every
   predeclared correction, which is precisely the case the repository's rule
   covers: an accuracy split located only at a line-shape threshold is a
   diagnosis to publish and a modelling defect to understand, never a flip to
   bolt on.
3. **Lane P1's headline needs its denominator quoted with it.** "On the lines
   the pool posts the overlays are 86% likely a loss" is right against the
   alternative of leaving every pick alone. It is not right as "the overlays
   have no value on pool-shaped lines": against the cost of flipping 138 picks
   at all, they are +0.763 points ahead on exactly those games. The -1.425 is
   mostly the arithmetic price of flipping against a base pick that is already
   right 56.53% of the time on half-point lines and only 51.99% on
   whole-number ones.
4. **The one number worth a follow-up is about a rule, not a threshold.** The
   division-revenge overlay on the whole archive reads +0.732 [-0.735, +2.197],
   `probability_positive` 0.834 -- as a rule it stays. On half-point lines
   alone it reads -1.425 [-3.147, +0.241], `probability_positive` 0.049, on 78
   flips. Acting on the second without a mechanism for why a division rematch
   should behave differently when nine books happened to average to a whole
   number would be the exact forbidden move. No such mechanism was found here.

Nothing is closed. All **60 recorded cells** are `unresolved_below_power` with
no closing ground. The served card, the active model and every published
forecast are untouched by this lane.

## The observation, restated on the three-way shape split

Lane P1 pooled whole and quarter points into one contrast slice. Split apart
(positive favours the overlay adjustment over the model's own pick):

| Shape | n graded | Flips | Union vs the model's own pick | P+ |
|---|---:|---:|---|---:|
| **half point** (the pool's shape) | 842 | 138 | **-1.425** [-3.933, +1.160] | **0.136** |
| **whole number** | 627 | 128 | **+4.466** [+1.102, +7.870] | **0.996** |
| quarter point | 34 | 6 | +11.765 [-3.030, +26.471] | 0.940 |
| every line | 1,503 | 272 | **+1.331** [-0.727, +3.393] | **0.898** |

`GAP` = whole minus half = **+5.891 [+1.734, +10.022]**, `probability_positive`
0.997. Against lane P1's pooled slice it is +6.266 [+2.079, +10.452]. The
quarter points are 34 games and six flips and decide nothing either way.

## H1 composition: accounts for none of it

Half-point share per season swings from 67.0% (2020) to 46.7% (2024), so the
season mix genuinely differs by shape. Captured book count barely does: mean
`opener_books` 13.08 on half-point lines against 13.70 on whole-number ones,
the same median of 13.

The union by season and shape shows the sign split is not a single season's
doing -- half-point lines are negative in four of six seasons and whole-number
lines positive in five of six -- but no season is decisive on its own:

| Season | half point | whole number |
|---|---|---|
| 2020 | +1.316 (P+ 0.692) | +6.154 (0.867) |
| 2021 | -0.730 (0.403) | +0.000 (0.491) |
| 2022 | -5.000 (0.059) | +10.236 (0.999) |
| 2023 | -1.418 (0.353) | +5.932 (0.920) |
| 2024 | -3.937 (0.127) | +1.613 (0.664) |
| 2025 | +0.000 (0.490) | +2.041 (0.660) |

Direct standardisation over eight (season x book-count tercile) strata holding
at least fifteen games of each shape returns a standardised gap of **+5.956
[+1.337, +10.407]**, `probability_positive` 0.995, against a raw +5.891. It is
*larger*, not smaller. Composition does not account for the gap.

## H2 push handling: accounts for none of it

The predeclared invariant holds, measured rather than assumed: **all 34 pushes
in the archive sit on whole-number lines and half-point lines have zero.** So
the whole-number domain really is graded after its on-the-line games have been
deleted while the half-point domain keeps everything.

Correcting it three ways changes nothing:

| Correction | half point | whole number | GAP |
|---|---|---|---|
| Raw | -1.425 (P+ 0.136) | +4.466 (0.996) | +5.891 [+1.734, +10.022] |
| Only games decided by at least a point | -1.708 (0.091) | +4.466 (0.996) | **+6.174** [+1.959, +10.387] |
| Pushes reinstated, home-favourable | -- | +4.387 (0.996) | -- |
| Pushes reinstated, away-favourable | -- | +4.085 (0.994) | -- |

The hook decomposition (H2d, **a diagnostic that conditions on the realised
result and can never be a rule**): inside the half-point domain the union is
+1.235 [-8.046, +10.844] on the 81 games decided by the half point itself and
-1.708 [-4.177, +0.810] on the 761 decided by more. The hook games are not
where the negative lives.

## H3 one member: locates 78% of the gap, and does not explain it

| Rule | Flips half / whole | half point | whole number | every line |
|---|---:|---|---|---|
| **division revenge** | 78 / 70 | **-1.425** [-3.147, +0.241] P+ **0.049** | **+3.190** [+0.640, +5.776] P+ **0.994** | +0.732 [-0.735, +2.197] P+ 0.834 |
| coach fade | 50 / 51 | +0.000 [-1.557, +1.675] 0.487 | +0.797 [-1.297, +2.946] 0.768 | +0.399 [-0.872, +1.714] 0.725 |
| player arrests | 14 / 9 | +0.238 [-0.712, +1.168] 0.692 | +0.797 [-0.147, +1.783] 0.958 | +0.466 [-0.199, +1.137] 0.915 |

Division revenge splits by 4.615 points, 78% of the union's 5.891. All three
members split the same way, which is what would happen if a common cause were
acting on all of them, but the other two split by 0.80 and 0.56 points.

**Locating is not explaining, and this is a post-hoc three-way split.** One of
three members was always going to be the largest. What the row does establish is
where to look if anyone wants to look further: nothing about a division rematch
plausibly depends on whether nine books averaged to a whole number, so the
likeliest remaining reading is that the division-revenge flips happen to land on
games where the model is weak, and line shape is a correlate of that weakness
rather than its cause.

## H4 spread magnitude: accounts for none of it

The magnitude mix barely differs -- mean `abs(line)` 5.18 on half-point lines
against 5.07 on whole-number ones -- and the coarse bucket counts track each
other closely. Within buckets the whole-number side is ahead in four of five:

| Line size | half point | whole number |
|---|---|---|
| 0-2 | -2.597 (P+ 0.265) | +3.077 (0.761) |
| 2-4 | -3.216 (0.060) | +5.909 (0.986) |
| 4-7 | +1.339 (0.675) | +6.494 (0.982) |
| 7-10 | -4.202 (0.135) | -1.724 (0.361) |
| 10+ | +3.750 (0.762) | +6.944 (0.922) |

Standardised over integer-floor buckets of `abs(line)`: **+6.841 [+2.175,
+11.460]**, `probability_positive` 1.000. Over coarse buckets: +6.093 [+1.992,
+10.200]. Both are *larger* than the raw gap. The key numbers sitting on whole
numbers do not account for it.

## H5 reliability: no closing ground, and the gap reproduces in sign

- **H5a block-level**, the predeclared primary: on 100 (season, week) blocks
  with both shapes present in both halves, the split-half correlation of the
  per-block gap is Pearson **-0.121 [-0.290, +0.053]** (Spearman-Brown -0.275),
  Spearman -0.081. The interval does not sit at zero, so the predeclared
  `no_split_half_reliability` ground is **not** met. It is also confounded:
  a per-block half holds about seven games and most of them carry a zero
  difference, so a near-zero correlation is what this construction returns for
  a real effect too. It is reported and it settles nothing.
- **H5b season-level**: Spearman -0.029 across six odd-week / even-week pairs,
  Pearson +0.102. Six pairs; uninformative either way.
- **H5c odd/even seasons**, the most directly powered: the gap is **+3.571
  [-2.542, +9.670]** (P+ 0.877) on 2021/2023/2025 and **+8.268 [+2.733,
  +13.894]** (P+ 0.999) on 2020/2022/2024. **Both halves are positive.** The
  half-point side is negative in both (-0.677 and -2.256) and the whole-number
  side positive in both (+2.894 and +6.013). The split reproduces in sign
  across the archive.

No closing ground is available from H5, and none is claimed. Every cell is
recorded `unresolved_below_power`.

## H6 the arithmetic: it is the hit rate, not the firing rate

The identity `effect = 100 * (flips / n) * (2 * hit - 1)` was asserted against
the measured point estimate and holds to 9e-16 on every shape.

| Shape | Flip rate | Hit rate on flips | Effect |
|---|---:|---:|---|
| half point | 16.39% | **45.65%** | -1.425 |
| whole number | 20.41% | **60.94%** | +4.466 |

Decomposing the 5.891-point gap: the flip-rate term is **-0.350**, the hit-rate
term **+5.010**, the interaction +1.231. Eighty-five per cent of the gap is the
hit rate. The rules fire at a similar rate on both shapes; when they fire on a
whole-number line they are right on three flips in five, and on a half-point
line on fewer than one in two. The firing conditions are not where the split
lives -- the games the flips land on are.

## H7 direction and base rate: contributes 23%, and reframes the headline

| Shape | Home cover rate | Model picks home | Flips to home / away | Observed | Direction-matched null mean (sd) | Excess |
|---|---:|---:|---:|---|---|---:|
| half point | 48.34% | 43.59% | 86 / 52 | -1.425 | **-2.188** (1.293) | **+0.763** |
| whole number | 51.20% | 42.42% | 76 / 52 | +4.466 | **-0.819** (1.582) | **+5.285** |

The null draws a random flip set of the same size with the same home/away
direction mix from the same shape's games, 2,000 draws, seed 20260911. Its mean
is negative in both shapes because flipping a pick that is right more than half
the time costs accuracy -- and it is more than twice as negative on half-point
lines, because the model is better there and home covered less often there.

The observed union sits at the 73rd percentile of that null on half-point lines
and above all 2,000 draws on whole-number lines. **It beats a same-size,
same-direction random flip on both shapes.** The gap in excess-over-null is
+4.522, so direction and base rate account for **1.369 of 5.891 points, 23%** --
a contribution under the predeclared bands, not an explanation.

The recordable form of the same mechanism: the model's own pick reads **-4.538
[-9.641, +0.500]** accuracy points on whole-number lines relative to half-point
ones, `probability_positive` 0.039. The pool's own shape is where the model is
already strongest and hardest to improve on.

## Summary against the predeclared bands

A hypothesis EXPLAINS if `GAP` falls below a third of +5.891 under its
correction, CONTRIBUTES if it falls by a tenth to a third.

| Hypothesis | GAP under its correction | Verdict |
|---|---|---|
| H1 composition (season x books) | +5.956 | does not account |
| H2 push handling (decided by a point) | +6.174 | does not account |
| H4 spread magnitude (standardised) | +6.841 / +6.093 | does not account |
| H7 direction and base rate | +4.522 | **contributes, 23%** |
| H3 one member | 4.615 of 5.891 located in division revenge | locates 78%, explains none |
| H5 reliability | no closing ground; both season halves positive | not resolved |

**Verdict: the gap is not explained.** Twenty-three per cent of it is the cost
of flipping against a stronger base pick on the pool's own shape, and the rest
is concentrated in one rule for reasons no test here named. It stays an
unexplained threshold, it is published as a diagnosis, and it may not gate
overlay membership.

## Recorded

60 rows under family `overlay_union_line_shape_mechanism_v1`, every one
`unresolved_below_power` with no closing ground, each carrying a plain-English
summary and the post-hoc discount in its notes so it can never be pooled with a
predeclared family. The registry went from 6,304 to 6,364 signals with zero
failures. The same 60 argv were first executed against an isolated registry
under the session scratchpad and all returned 0, so the emitted argv is
admissible to the real validator. The two H2d rows and the H7 direction figures
carry an explicit note that they condition on the realised result and can never
become rules.

No rotation window spent: a diagnostic look at an already-scored archive, the
precedent `docs/player_arrests_policy_eval.md` and MOD-17 both set.

## What this lane leaves open, named now

The remaining live question is not about line shape at all. It is why the
model's own pick is 4.5 accuracy points weaker on whole-number opener lines,
and whether that weakness has a game-level correlate the overlays are
accidentally tracking. That is a model diagnosis -- the same weak-spots reading
`docs/mod18_discrete_margin_mapping.md` and the Model page's weak-spots table
already carry -- and it is the honest successor to this lane, rather than any
rule conditioned on how a consensus rounded.
