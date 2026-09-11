# Contest utility optimizer (POL-05)

**Question.** Given the pool's scoring rules and a field of other entries, does
any deviation from *"submit the max-probability side on every game and star the
max-probability game"* raise the expected season payout, and by how much?

`docs/pool_format_levers.md` (2026-08-17) answered a version of this against the
then-current 52.5% card, an arbitrary Best Pick and a hypothetical field. Three
things have changed since and none of them were folded back in: the served card
is measured higher, the Best Pick is now chosen by a measured nominator and
re-nominated on the Sunday pass (`docs/best_pick_sunday_renomination.md`), and a
tiebreaker shade is served (`docs/tiebreaker_low_side_shading.md`). This
document re-runs the contest-utility question on those served numbers and adds
the payout exchange rate the owner asked for, so research lanes can be compared
in payout units instead of accuracy units.

Nothing here changes served behaviour. Every proposal is a proposal.

---

## Predeclaration

Everything in this section was written before any cell was scored.

### Binding closing-grounds taxonomy, verbatim

An interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (1) refuted mechanism — a RESOLVED wrong sign (whole interval on the wrong
side of zero) or zero split-half reliability; (2) bounded by a positive control
proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". The registry code
hard-rejects inadmissible closures; if a record command errors, the verdict is
wrong, not the validator.

Decide on expected value. 0.90 and 95% govern what the docs may claim, never
which card is submitted. Within-week game correlation is zero by owner mandate.
Football margins are discrete and multimodal: nothing below fits or assumes a
Gaussian margin — every cover probability used here is an empirical hit rate
measured from the opener archive, read off graded outcomes.

### What is measured, what is assumed

**Measured** (from artifacts, this session):

- Per-game cover probability as a function of the frozen Tuesday opener spread,
  fitted on the active model's own opener evaluation
  (`artifacts/opener_evaluation/20260910T211255Z/per_game.parquet`, 1,503 graded
  games, 107 weeks, 2020-2025). This is an empirical hit-rate curve on graded
  outcomes, not a margin distribution.
- The share of our own picks that sit on the betting favourite, the only
  obtainable proxy for the field's popular side (`docs/pool_format_levers.md` §4
  closed POL-04: no pick-popularity feed with history exists).
- The served Best Pick hit rate, from LEAD-53's artifact
  (`artifacts/best_pick_sunday_renomination/20260911T021349Z/summary.json`).
- The real 2025 weekly game counts and playoff count
  (`data/raw/20260908T162105Z/schedules.parquet`), so the season has the pool's
  real shape: 272 regular-season picks in 18 weeks plus 13 playoff picks, 285
  forced picks, 18 Best Picks.
- The tiebreaker shade's measured effect on the closest-total tiebreak, from
  LEAD-54 (`docs/tiebreaker_low_side_shading.md`).

**Assumed** (declared as a sensitivity grid, never as data). The pool's field
size and prize structure are still uncaptured: LEAD-52 built the capture path
(`nfl-ats pool-observables`) and `data/pool_observables/` does not exist yet, so
every field parameter below is an assumption with a grid, not a measurement.

| parameter | grid | headline |
|---|---|---|
| field size (entries, ours included) | 26, 101, 285, 1001 | **285** (the task's figure; note the repo's own 285 is the count of forced *picks* per season, not a captured field size) |
| prize structure | winner-take-all, top 10%, top 15%, top 25% | winner-take-all and top 15% reported side by side |
| field public lean `L` — probability a rival takes the popular (favourite) side of a game | 0.50, 0.55, 0.65, 0.75, 0.85 | 0.65 |
| Best Pick bonus | +1 (double points, researched) and +2 (triple) | +1 |

`L` is the field-correlation axis cell 2 asks for: `L = 0.50` is a field
uncorrelated with the popular side (and so with each other), `L = 0.85` is a
heavily herded field. Rivals are i.i.d. conditional on outcomes, so `L` also
sets how much rivals resemble each other.

### Instrument

`nfl_ats.pool` — `PoolFormat`, `FieldModel`, `Entry`, `simulate_pool_finish`,
validated in `docs/pool_format_levers.md` §3 against five closed forms. The
script adds one capability the library lacks and the cells need, and keeps it in
the script rather than in `src/`: a detailed finish simulation that additionally
records **boundary ties** (the event where a tiebreak, not a score, decides
whether we are paid). It calls the library's own field-scoring routine so the
field model is identical, and it is tied out against
`simulate_pool_finish` on a shared seed.

### Cells, predeclared

1. **Best Pick weighting.** What share of the season's score variance, and of
   the season's expected score, comes from the 18 Best Picks versus the 267
   ordinary picks, at the served hit rates. Predeclared expectation: the Best
   Pick is a minority of the variance (under 25%), because 18 doubled picks
   cannot outweigh 267 single ones.
2. **Star a higher-variance game.** Holding all 285 sides fixed and changing
   only *which* game carries the star: does starring a bigger-spread, more
   contrarian, or closer-to-coin-flip game beat starring the max-probability
   game, under winner-take-all and under a top-N payout, as a function of `L`?
   Arms: `max_probability` (incumbent), `big_spread`, `max_variance`,
   `contrarian` (highest probability among games where our pick is the
   underdog), `contrarian_big`, `arbitrary` (a uniformly random eligible game —
   the control that says what the chooser itself is worth). Predeclared
   direction: `big_spread` and `max_variance` lose everywhere; `contrarian` is
   the one that could plausibly win under winner-take-all, sign not predeclared.
   **Extension, same cell:** the other half of the headline question — flipping
   *ordinary* sides away from the model. Flip the `k` popular-side picks whose
   probability is closest to 0.5, `k` in 0, 10, 25, 50, paying the measured cost
   of each flip.
3. **Tiebreaker shade.** Does the served `-1` shade on the tiebreak total change
   the expected payout materially? Decomposed as (probability a tiebreak decides
   whether we are paid) x (LEAD-54's measured change in the tiebreak win rate).
   Predeclared expectation: immaterial, under 0.05 accuracy-point equivalents.
4. **Payout units.** The value of one extra point of ordinary forced-pick
   accuracy in P(first), in P(paid) and in expected finishing percentile, at
   each field size — the exchange rate that lets the owner price a research lane
   in payout units. Every cell-2 and cell-3 result is then restated in
   **ordinary-accuracy-point equivalents** through this rate.

### Metric, uncertainty and the honest unit

Each replicate draws its inputs afresh: a **week-blocked bootstrap** of the
opener archive (block = one graded week, matching this project's standing
convention and the owner's zero-within-week-correlation mandate), refits the
hit-rate curve on that resample, and draws a fresh 285-pick season from that
resample's joint distribution of spread and favourite flag. Arms within a
replicate share the drawn inputs **and the simulator seed** (common random
numbers), so an arm difference is paired and its interval is the paired
difference's interval, not the difference of two interval endpoints. Reported:
the mean effect, the 2.5/97.5 percentiles across replicates, and
`probability_positive` under the half-credit convention
(`nfl_ats.evidence_conventions.probability_positive_from_draws`).

The interval therefore carries *both* simulation error and sampling uncertainty
about the measured hit rates. It does **not** carry uncertainty about the
assumed field parameters; that is what the grid is for.

**On units.** The weak-signal registry's `--effect-units` has no payout unit
(`nfl_ats.weak_signals.EFFECT_UNITS`), and LEAD-51 recorded nothing for exactly
that reason. Rather than record nothing, every payout comparison here is
converted through cell 4's measured exchange rate into
**ordinary-accuracy-point equivalents** — "this lever is worth as much P(first)
as *x* points of forced-pick accuracy would be" — and recorded as
`accuracy_points` with that conversion stated in the description. Cells whose
quantity is a genuine accuracy (the spread gradient) are recorded as measured,
with no conversion.

### Declared limitations, stated before the run

- **The field has never been fitted to a real field.** No pick-popularity feed
  with history exists (POL-04). Every conclusion that depends on `L` is
  conditional on the grid.
- **Field size and prize structure are uncaptured.** Both are observable the
  first time the pool is entered (LEAD-52) and both move cell 2's answer.
- **Rivals are i.i.d. given outcomes.** Real entries correlate through shared
  information beyond the favourite lean; correlation of that kind raises the
  value of differentiation, so cell 2's contrarian arms are, if anything,
  understated.
- **The Best Pick bonus is researched, not owner-confirmed.** Every cell-1 and
  cell-2 number is reported at +1 and +2.
- **No pushes.** The pool posts half-point numbers.
- **The 13 playoff picks sit in the last weekly block** so the season totals 285
  picks across 18 Best Pick weeks; the star choosers are restricted to
  regular-season games. The field's 18th Best Pick is drawn from that enlarged
  block — a declared, negligible asymmetry.
- **The hit-rate curve is a function of the opener spread only.** It is the one
  pick-quality axis this project has measured repeatedly; the model's own stated
  confidence is measured flat and is deliberately not used as `p`.

---

## Results

All measured 2026-09-11 with
`.\.tools\uv.exe run --no-sync python scripts/contest_utility_optimizer.py --replicates 20 --sample-budget 5000`;
artifact `artifacts/contest_utility_optimizer/20260911T034605Z/` (`results.json`,
`derived.json`, `star_arms.csv`, `side_flips.csv`, `boundary_ties.csv`,
`accuracy_curve.csv`, `star_equivalents.csv`, `tiebreaker_value.csv`). The
three side cells were run separately from the same script and the same inputs
(`--gradient-only`, `--controls-only`, `--best-pick-value-only`).

### The answer, first

**Almost nothing beats "max-probability side on every game, star the
max-probability game", and the one thing that does is worth about a
twenty-ninth of a single accuracy point.**

- Every deviation on the **sides** loses or is a coin flip. Flipping 25
  near-50/50 popular-side picks costs -0.02 accuracy-point equivalents at
  winner-take-all (`probability_positive` 0.55) and -0.19 under a top-15%
  payout (0.05); flipping 50 costs -0.21 and -0.55.
- Every deviation on **which game carries the star** loses except one:
  starring the highest-probability game *among the games where our pick is the
  underdog* is worth **+0.035 accuracy-point equivalents** at winner-take-all
  (`probability_positive` 0.85 at the assumed 0.65 field lean, rising to 0.95 at
  0.85), and is **negative under every wide payout** (-0.027 at top-15%,
  `probability_positive` 0.20). Starring a big-spread game (-0.29), a
  coin-flip game (-0.24), or a big contrarian game (-0.22) all lose.
- **The big number is not a deviation at all.** One point of ordinary
  forced-pick accuracy is worth **+5.80 percentage points of P(first)** at 285
  entries, 95% [+4.67, +6.37], `probability_positive` 1.000 — 29x the best star
  deviation. And LEAD-53's already-served Sunday Best Pick re-nomination is
  worth **+2.08 pp of P(first)**, 95% [+0.83, +3.31], `probability_positive`
  1.000 — **+0.36 accuracy-point equivalents**, ten times the star-position
  lever.

So POL-05's question has an answer, and it is "yes, one, and it is small and
conditional on a prize structure nobody has written down yet."

### Measured inputs

| input | measured | source |
|---|---|---|
| ordinary forced-pick hit rate at the frozen Tuesday opener | **52.96%** (1,503 games, 107 weeks) | `artifacts/opener_evaluation/20260910T211255Z/per_game.parquet`, `correct_at_open` |
| same, today's served probability rule | 54.56% | same artifact, `correct_at_open_probability_rule` |
| served Best Pick hit rate (Sunday re-nomination, S3) | **56.73%** (104 weeks) | `artifacts/best_pick_sunday_renomination/20260911T021349Z/summary.json` |
| Tuesday Best Pick hit rate (T0, the retired instant) | 52.88% (104 weeks) | same |
| our picks on the betting favourite | 46.7% | same opener artifact |
| hit-rate curve | `logit(p) = 0.2392 - 0.02349 x |spread|` | fitted on the opener artifact |
| season shape | 272 regular-season picks in 18 weeks + 13 playoff picks = 285 | `data/raw/20260908T162105Z/schedules.parquet` (2025) |

The curve reads 55.4% at a 1-point spread, 54.2% at 3, 52.5% at 6, 50.7% at 9
and 48.3% at 13. The measured hit rate on picks at spreads of 4.5 or less is
**53.92%** (829 games) against **48.31%** on spreads of 7.5 or more (325
games) — a gradient of **+5.61 accuracy points, week-blocked 95% [-0.80,
+12.14], `probability_positive` 0.9566** on 107 week blocks. It is positive in
both season halves independently (odd seasons 2021/23/25 **+8.59** points, even
seasons 2020/22/24 **+2.56**) — an era-magnitude difference, not an absence.
This gradient is the mechanism the whole star question rests on: it is why the
max-probability game is the small-spread game and why the big-spread star is
starring a pick measured below a coin flip.

Sanity tie-out: the curve's value at the smallest spread in a week averages
about 55%, and LEAD-53's served nominator measures 56.73%. The simulated star
is therefore a slightly *conservative* stand-in for the nominator that is
actually served; cell 2 compares star *positions* at a common curve, which is
what isolates the position question from the nominator-quality question that
cell 5 prices separately.

### Instrument check

| check | result |
|---|---|
| script's detailed simulator vs `nfl_ats.pool.simulate_pool_finish`, shared seed | **identical to 1e-12** on P(first), expected rank, score sd and tied-first |
| positive control: a perfect card | P(first) = **1.000** |
| a pure coin-flip card whose side profile matches the field's own favourite lean | P(first) **0.54%** vs a fair share of 0.35% (**1.54x**) |
| a pure coin-flip card at our measured 46.7% favourite share | P(first) **0.96%** (**2.73x** a fair share) |

Read the last two rows as a warning label, not a finding. **The field model
awards a differentiation premium to a zero-edge card**, because rivals are
correlated with each other through the popular side and we are not, and it
awards more of it the further our side profile sits from theirs. Absolute
P(first) levels below are therefore optimistic and should not be quoted as
"our chance of winning the pool". Every cell is reported as a **paired delta at
a fixed field setting**, where that level bias cancels.

### Cell 1 — Best Pick weighting

Exact arithmetic at the measured hit rates (267 ordinary picks at 52.96%, 18
starred picks at 56.73%), with the simulator's own resampling alongside:

| quantity | double points (+1) | triple points (+2) |
|---|---:|---:|
| season expected score | 161.83 | 172.04 |
| Best Pick share of expected score | **12.6%** | 17.8% |
| season score standard deviation | 9.18 | 10.31 |
| Best Pick share of score variance | **21.0%** | 37.4% |
| variance **added** by starring at all (exact) | 15.7% | 33.3% |
| same, simulated (20 replicates) | **15.9%**, 95% [15.1%, 16.9%] | **33.5%**, 95% [32.2%, 34.8%] |

**Roughly five-sixths of the season's score variance, and seven-eighths of its
expected points, come from the ordinary picks.** The predeclared expectation
(under 25%) holds at double points and fails at triple — worth knowing, since
the multiplier is researched rather than confirmed. The practical reading: the
Best Pick is where a *per-pick* improvement is worth the most (each starred
pick counts double), and simultaneously it is far too small a slice of the
season to be where a *card-level* improvement should be sought.

### Cell 2 — starring a higher-variance game

285 entries, field lean 0.65, double points, 20 replicates, common random
numbers. Every arm submits the **same 285 sides**; only the starred game moves.

| star rule | P(first) | delta (pp) | 95% | `probability_positive` | accuracy-point equivalent |
|---|---:|---:|---|---:|---:|
| **max probability (incumbent)** | 9.57% | — | — | — | — |
| contrarian (best pick among our underdogs) | 9.77% | **+0.20** | [-0.29, +0.60] | **0.85** | **+0.035** |
| arbitrary (random eligible game) | 8.91% | -0.66 | [-2.03, -0.06] | 0.00 | -0.114 |
| max variance (p closest to 0.5) | 8.18% | -1.39 | [-3.22, -0.15] | 0.00 | -0.240 |
| contrarian big (biggest spread among our underdogs) | 8.28% | -1.29 | [-3.86, +0.28] | 0.15 | -0.222 |
| big spread | 7.88% | -1.69 | [-4.50, 0.00] | 0.05 | -0.291 |

The contrarian star is nearly free: it costs **0.07 expected points a season**
(161.21 vs 161.28) because the best underdog pick in a week is usually almost
as good as the best pick overall. Every other deviation pays 1.0-1.2 expected
points for its variance, which is why they all lose.

**As a function of the field's assumed correlation with our card** (285
entries, winner-take-all), the contrarian star's value is monotone in how
herded the field is — exactly the predicted mechanism:

| field lean `L` | contrarian delta (pp) | 95% | `probability_positive` |
|---:|---:|---|---:|
| 0.50 (uncorrelated field) | -0.08 | [-0.39, +0.20] | 0.40 |
| 0.55 | +0.00 | [-0.29, +0.18] | 0.60 |
| 0.65 | +0.20 | [-0.29, +0.60] | 0.85 |
| 0.75 | +0.36 | [-0.13, +0.87] | 0.90 |
| 0.85 (heavily herded) | +0.34 | [-0.04, +0.90] | 0.95 |

**And it reverses as the payout widens** (285 entries, `L` = 0.65):

On P(paid), with a tie at the cutoff counted as paid:

| payout | contrarian delta (pp) | `probability_positive` |
|---|---:|---:|
| winner-take-all (1 place) | +0.17 | **0.75** |
| top 10% (28 places) | -0.21 | 0.375 |
| top 15% (43 places) | -0.32 | 0.20 |
| top 25% (71 places) | -0.55 | 0.05 |

This is the same shape `docs/pool_format_levers.md` §5 found for *side* flips,
now measured for the *star*: differentiation buys the top tail by selling the
middle, and a wide prize pays for the middle. Field size barely changes it
(+0.11 pp at 26 entries, +0.20 at 101, +0.20 at 285, +0.14 at 1001, all at
`L` = 0.65).

**Side deviation (the other half of the headline question).** Flipping the `k`
popular-side picks closest to a coin flip, paying each flip's measured cost:

| flips | mean pick probability | P(first) delta (pp) | `probability_positive` | top-15% delta (pp) | `probability_positive` |
|---:|---:|---:|---:|---:|---:|
| 10 | 53.04% | +0.17 | 0.65 | -0.57 | 0.275 |
| 25 | 52.89% | -0.10 | 0.55 | -2.31 | 0.050 |
| 50 | 52.50% | -1.25 | 0.30 | -6.56 | 0.000 |

Ten flips is a dead heat at winner-take-all and negative on every wider payout;
25 and 50 are worse. **The sides should not move.** This reproduces the 2026-08-17
finding on the current card and current hit rates rather than the retired ones.

### Cell 3 — the tiebreaker shade

The tiebreak only matters when a tie decides whether we are paid. Simulated
(285 entries, `L` = 0.65), that happens in:

| payout | P(a tiebreak decides our payment) |
|---|---:|
| winner-take-all | **1.63%** of seasons, 95% [0.66%, 3.14%] |
| top 10% | 3.84% [3.23%, 4.35%] |
| top 15% | 3.87% [2.94%, 4.47%] |
| top 25% | 3.62% [2.35%, 4.74%] |

Combined with LEAD-54's measured change in the tiebreak win rate from the
served `-1` shade (**-0.137 pp, 95% [-0.935, +0.661]**, `probability_positive`
0.362 on the closest-total win-rate metric), the expected change in P(paid)
over a season is **-0.0000051**, 95% [-0.0000355, +0.0000248],
`probability_positive` **0.367** at 285 entries and a top-15% payout —
**-0.00043 accuracy-point equivalents**, 95% [-0.0030, +0.0021]. The answer is
the same to three significant figures at every field size and every payout in
the grid.

**Immaterial, as predeclared: about one two-thousandth of a single accuracy
point.** Nothing here argues for changing the served shade in either direction.
LEAD-54 chose it on the closest-total absolute-error metric, where it reads
`probability_positive` 0.816 full-sample and 0.843 out-of-sample; this cell
only establishes that the *payout* consequence is too small to enter any
decision. The shade stays served on LEAD-54's grounds, and the contest-utility
axis has no opinion.

### Cell 4 — payout units

285 entries, `L` = 0.65, double points, 20 replicates:

| forced-pick accuracy | P(first) | P(top 15%) | expected finish (percentile, lower is better) |
|---:|---:|---:|---:|
| 50.0% | 1.00% | 21.2% | 47.7th |
| 52.0% | 4.49% | 43.3% | 29.5th |
| **53.0%** | **8.11%** | **56.0%** | **21.8th** |
| 54.0% | 13.91% | 67.9% | 15.5th |
| 55.0% | 21.67% | 78.3% | 10.5th |
| 57.0% | 42.84% | 92.3% | 4.3rd |

**One accuracy point, priced four ways** (from 53.0% to 54.0%):

| field | P(first) gain | 95% | `probability_positive` | finishing-percentile gain |
|---:|---:|---|---:|---:|
| 26 entries | +10.37 pp | [+9.42, +11.52] | 1.000 | 6.13 |
| 101 | +7.79 pp | [+6.84, +8.40] | 1.000 | 6.30 |
| **285** | **+5.80 pp** | [+4.67, +6.37] | **1.000** | **6.32** |
| 1001 | +3.76 pp | [+2.57, +4.65] | 1.000 | 6.34 |

**The finishing-percentile rate is the same at every field size: one point of
forced-pick accuracy moves us up about 6.3 percentiles of the field.** That is
the exchange rate to price a research lane with. At 285 entries it is also
+5.80 pp of P(first) and +11.85 pp of P(top 15%).

### Cell 5 (added) — what the Best Pick nominator is worth in payout units

Same season, same sides, same star *position*; only the starred pick's true hit
rate changes, to each of the three rates LEAD-53 measured. 285 entries,
`L` = 0.65, 20 replicates, common random numbers, baseline = a star that hits
at the card average:

| starred pick hits at | P(first) | delta (pp) | 95% | `probability_positive` | accuracy-point equivalent |
|---|---:|---:|---|---:|---:|
| 52.96% (card average, an arbitrary star) | 7.93% | — | — | — | — |
| 52.88% (Tuesday nominator, the retired instant) | 8.04% | +0.11 | [-0.35, +0.81] | 0.60 | +0.019 |
| **56.73% (Sunday re-nomination, served)** | **10.02%** | **+2.08** | [+0.83, +3.31] | **1.000** | **+0.359** |

**The served Sunday re-nomination is the largest contest-format lever this
project has, worth about 0.36 ordinary accuracy points** — ten times the star
*position* lever and a third of a full accuracy point, for a change that was
already made. The Tuesday nominator was worth essentially nothing, which is
LEAD-53's own finding arriving through a second route.

One caveat that belongs in the same breath: this prices a nominator whose true
hit rate is 56.73%, and LEAD-53's 56.73% is itself unresolved (S3 beats the
Tuesday arm at `probability_positive` 0.814 on 104 weeks, interval
[-5.83, +15.53]). The payout value is conditional on the hit rate being real;
what is *not* conditional is the shape — a starred pick is worth double, so a
point on it is worth two points anywhere else, and 18 of them are worth 36
ordinary picks' worth of movement.

---

## What should change in how the card is submitted

**Nothing, today.** Stated as expected value and not as caution:

1. **Sides: do not deviate.** Ten flips is a dead heat at winner-take-all
   (`probability_positive` 0.65) and negative on every wider payout; 25 and 50
   are worse on both. There is no version of side deviation whose expected
   payout is positive across the grid.
2. **Star position: keep the max-probability rule for now, and switch to the
   contrarian rule if and only if the pool turns out to be winner-take-all.**
   The arm is +0.17 pp at winner-take-all (0.75) and -0.32 pp at top-15%
   (0.20). Splash's researched default is roughly top 15%, so the expected
   value *over what is currently known* is negative, and the switch becomes
   positive only on an observation nobody has made yet. This is a conditional
   decision waiting on a free measurement, not a deferral.
3. **Tiebreaker shade: leave it exactly as served.** Its payout consequence is
   -0.0004 accuracy-point equivalents; it is decided by LEAD-54's own metric,
   not by contest utility.
4. **Capture the two observables at entry** (`nfl-ats pool-observables`,
   LEAD-52). Field size and prize structure are the only two inputs that change
   any answer above, both are free at entry, and `data/pool_observables/` is
   still empty. The prize structure alone flips item 2's sign.
5. **Spend research effort on accuracy, and on the Best Pick.** One accuracy
   point is 6.3 percentiles of finishing position at any field size; the entire
   star-position question is 0.035 of one. The Best Pick is the exception among
   format levers — it pays double, so it is the one place where a small
   per-pick improvement buys a third of an accuracy point (cell 5).

### Proposal, not served

If Week 1's entry shows a winner-take-all structure, the star rule becomes:
among the still-playable games in the Sunday dispersion pool, take the highest
frozen-side cover probability **restricted to games where our pick is the
underdog**, falling back to the unrestricted maximum when we hold no underdog
pick that week. That is a one-line filter on
`nfl_ats.best_pick_nomination.select_nominee`'s candidate set. It is **not
wired** and must not be wired before the prize structure is observed, because
under the researched top-15% structure it is measured negative.

---

## What is recorded, and what must not be pooled with it

Thirteen cells are recorded via `nfl-ats weak-signals record`, family
`contest_utility`, category `modeling`, all `unresolved_below_power`
(registry entries 6,003 through 6,015).

**Twelve of the thirteen are simulated payout quantities converted to
ordinary-accuracy-point equivalents through cell 4's exchange rate**, and they
are recorded in `accuracy_points` because the registry has no payout unit
(`nfl_ats.weak_signals.EFFECT_UNITS`) and LEAD-51's alternative — recording
nothing — loses the result entirely. The conversion is stated in every
description. **They are not commensurable with measured game-level
`accuracy_points` and must not be pooled with them**: they are Monte Carlo
quantities conditional on an assumed field, not graded outcomes. Only
`contest_utility_spread_accuracy_gradient` is a real measured accuracy.

**Nothing closes.** Neither terminal ground is available:

- `wrong_sign_resolved` — two arms (max-variance star, arbitrary star) do have
  a 95% Monte Carlo interval wholly below zero, but that interval is
  simulation-plus-input error *conditional on assumed field parameters*, not a
  sampling interval over graded outcomes. A grid assumption cannot refute a
  mechanism, so every cell is `unresolved_below_power` and the doc's
  recommendation carries the decision instead.
- `bounded_by_control` — the positive control (a perfect card) resolves at
  P(first) 1.000 and the accuracy sweep resolves a one-point effect at
  `probability_positive` 1.000, so the instrument demonstrably sees effects of
  ~5 pp. It was **not** shown able to resolve a 0.2 pp star-position effect, so
  it bounds nothing at the scale cell 2 lives at.

No rotation window is spent: nothing here fits a model, selects a feature, or
grades a new arm on held-out seasons. It re-reads an already-mined 107-week
opener archive for two descriptive inputs (the hit-rate curve and the favourite
share) and simulates on top of them.

**2026-09-11 addendum: the twelve simulated cells were re-recorded in a native
payout unit.** A new `nfl_ats.weak_signals.EFFECT_UNITS` entry,
`payout_first_pp` (percentage points of a simulated pool-finish probability),
was added so these twelve no longer have to borrow `accuracy_points` to be
recorded at all. All twelve (every row above except
`contest_utility_spread_accuracy_gradient`) were re-recorded via
`nfl-ats weak-signals record --replace` under `effect_units=payout_first_pp`,
same names, same classification (`unresolved_below_power`), same
`sample_games`/`sample_blocks`/`seasons`, with the raw simulated percentage
points read from the artifact (`derived.json`, `results.json`,
`best_pick_value.json`, `tiebreaker_value.csv` under
`artifacts/contest_utility_optimizer/20260911T034605Z/`) rather than the
accuracy-point conversion or this document's own rounding — one entry,
`contest_utility_tiebreaker_shade_payout`, exposed that this document's inline
prose figure (`-0.0000051`) is roughly 10x too small versus the artifact's own
`tiebreaker_value.csv` row (`-0.00005139...`); the registry now carries the
artifact figure. Verified: `nfl-ats weak-signals pool --league nfl
--effect-units accuracy_points` eligible count dropped from 3,063 to 3,051 (12
fewer) and now contains only `contest_utility_spread_accuracy_gradient` from
this family; `nfl-ats weak-signals pool --league nfl --effect-units
payout_first_pp` pools exactly these 12 signals and nothing else (random-effects
+0.347 payout pp, 95% [-0.933, +1.628], `probability_positive` 0.702 — an
interval crossing zero, not grounds to close anything, and not commensurable
with any `accuracy_points` number since the two units are never pooled
together). `docs/weak_signal_registry.md`'s unit vocabulary table documents
the new unit.

## Limitations that survived the run

- **The field is assumed.** The `L` grid is the honest form of "we do not know
  how herded this pool is"; the instrument check shows the field model hands a
  zero-edge card a 1.5-2.7x differentiation premium on its own, so cell 2's
  contrarian result is a property of that model as much as of the pool.
- **Field size and prize structure are still uncaptured**, and the prize
  structure decides the sign of the only positive deviation found.
- **20 replicates** put a floor of 0.05 on `probability_positive`, so a
  reported 0.00 means "negative in all 20 paired replicates", not zero.
- **The star chooser is a proxy.** Cell 2 ranks by the fitted hit-rate curve,
  not by production's dispersion-pool nominator; it answers "which *kind* of
  game should carry the star", not "is the served nominator the best one".
  Cell 5 answers the second question on LEAD-53's measured rates instead.
- **The hit-rate curve is a function of the opener spread only**, and its
  gradient is +5.61 points with a 95% interval that reaches -0.80. A flatter
  true curve would shrink every cell-2 difference toward zero without changing
  a sign.

## Files

- `scripts/contest_utility_optimizer.py` — the four predeclared cells, the
  measured inputs, the instrument controls, the spread gradient, the Best Pick
  payout pricing, and the payout-unit derivation
  (`--derive-from`); writes `artifacts/contest_utility_optimizer/<UTC stamp>/`.
- This document.
- `ROADMAP.md` POL-05 row (updated with these numbers; no other row touched).
- 13 `nfl-ats weak-signals record` entries, family `contest_utility`, category
  `modeling` (registry `registry/weak_signals.json`, entries 6,003-6,015).

Nothing served was changed: no model, no card, no nominator, no tiebreaker, no
challenger, no ledger.
