# The number the reader sees: calibrating the displayed confidence (MOD-18 lane AH)

Follow-on to `docs/spread_hole_diagnosis.md` and `docs/spread_hole_target.md`.
Lane N ends by naming this arm: *"What is NOT bounded, and what none of the
seven arms has touched, is the flat 55.8-56.5% stated confidence in every
bucket — the model has one uncertainty and reuses it everywhere. That is a
per-bucket, mass-correct probability problem, not a side problem, and it is the
one place this program's own measurements keep pointing"* (read:
`docs/spread_hole_target.md:412-422`).

This lane changes a **displayed number only**. No side changes, on any game, in
any bucket, ever. There is no threshold rule and nothing flips.

Closing-grounds taxonomy, verbatim, because this document reports intervals:
an interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (1) refuted mechanism — a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (2) bounded by a positive
control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". Never use 95%, 0.90,
or any threshold as a DECISION bar; decide on expected value
(`probability_positive` above 0.5 favours serving it), thresholds only govern
what docs may CLAIM.

## The defect

Measured 2026-09-09 by lane F from the served opener archive (read:
`docs/spread_hole_diagnosis.md:23-28`): the model's stated confidence is
55.6% / 56.5% / 56.2% / 56.0% across the four line-size buckets while its
realised accuracy as served is 56.16% / 51.35% / 51.55% / 47.33%. The stated
number is bucket-blind; the realised number is not. Lane N then measured the
same flatness surviving all four of its arms (read:
`docs/spread_hole_target.md:371-377`).

The card and the This Week page print that stated confidence as the reader's
"Decision score" (read: `CURRENT_PREDICTIONS.md:15`,
`src/nfl_ats/publishing.py:104`, `src/nfl_ats/public_board.py:735`), and the
per-pick explanation says "The model makes ARI a 64.2% cover" (read:
`src/nfl_ats/card_explanation.py:672-675`). Week 1 2026's two most confident
lines are ARI +9.5 at 64.2% and WAS +5.5 at 63.3%; the first sits in the
7.5-10 bucket, where the model has been right 51.55% of the time.

A percentage a reader acts on that is 64.2% when the honest answer is not
64.2% is exactly what the "no number on the site may go stale" rule bans
(AGENTS.md): a figure standing in for a measurement it does not carry.

## Design, frozen 2026-09-10T01:07:59Z before any candidate was scored

Family `mod18_displayed_confidence_v1`. One candidate, one baseline, no tuning
after scoring, no subset search, no threshold rule, no pick flip.

### Baseline — the stated probability, exactly as served today

For each archive game, the served home-cover probability at the opener
(`home_cover_probability_at_open`, which already carries the S3 home-side
offset), oriented to the model's own probability-rule pick
(`pick_home_at_open_probability_rule`). That oriented number is what the card
prints.

### Candidate — a walk-forward reliability transform on that displayed number

For each game, the displayed probability is replaced by the shrunken realised
correctness of its **cell**, computed from completed games only.

**Cells: four line-size buckets by three stated-probability bands, twelve in
all.**

| dimension | levels |
|---|---|
| line size `abs(spread)` | `0-6.5`, `7` (exactly 7), `7.5-10`, `10.5+` |
| stated pick probability | `[0.50, 0.55)`, `[0.55, 0.60)`, `[0.60, 1]` |

The buckets are the four the diagnosis and the Model page already report to
the reader (read: `docs/spread_hole_diagnosis.md:23-28`), so the transform is
stated in the same vocabulary the defect is stated in. The bands are MOD-18
C3's, verbatim (read: `docs/spread_regime_program.md:63-66`).

**Estimator: MOD-18 C3's, verbatim and not re-tuned.** For a game with stated
pick probability `p` in cell `c`,

```
calibrated = (prior wins in c + 20 * p) / (prior games in c + 20)
```

The count 20 is fixed regularization carried over from C3 (read:
`docs/spread_regime_program.md:66-68`) and is not tuned in this lane. No
history in the cell means the calibrated value equals `p` exactly. The shrink
target is the game's OWN stated probability, not a pooled constant, so a cell
with little history barely moves.

**Walk-forward: expanding, 2020 onward, cut at the game's own Tuesday.** Only
games from a strictly earlier `(season, week)` than the target's enter its
history, so nothing from the target week — including the target game — can
reach its own calibration. The archive's weeks complete on Monday night and
the card is published on Tuesday, so "strictly earlier week" is exactly
"completed before that game's Tuesday". History is expanding rather than
trailing-N: every completed archive game from 2020 forward is eligible. The
first archive weeks therefore have no history and calibrate to themselves.

Where MOD-18 C3 also splits cells by favourite/underdog/pick'em and by five
line buckets, this lane uses the four display buckets and no side split. That
is stated here, before scoring, as a deliberate coarsening for cell size (the
7.5-10 bucket holds 194 archive games in total), not a search over cell
definitions: no other cell grid is scored, in this lane or any other.

**Sides never change.** C3's "flip only if calibrated pick correctness falls
below 0.5" clause is explicitly NOT adopted. A calibrated value below 0.5 is
displayed as it is; the pick stands. This is required by the owner's ban on
unexplained threshold flips (AGENTS.md, "No unexplained threshold flips on the
played card") and is the whole point of the lane: the reader is told how sure
the model has actually been, and the forced pick is made anyway.

### Population and grading

The served opener archive
`artifacts/opener_evaluation/20260910T005854Z/per_game.parquet` (active model
`2e8c616b476dd0d2`, 1,537 archived games 2020-2025), restricted to games that
are not a push at the opener — 1,503 games. Grading is at the **opener**,
which is the grade the pool settles on.

**Primary quantity: the paired Brier-score difference (stated minus
calibrated, so positive favours the candidate) on those 1,503 games, with its
`probability_positive`.** Predeclared secondary reads: the same paired
difference in log loss; both metrics per line bucket; a reliability table
(stated band → mean stated, realised accuracy, n) before and after; the
per-cell calibration table fitted on the full archive; and the sixteen Week 1
2026 displayed scores before and after.

Brier is `mean((displayed - correct)^2)` and log loss is
`-mean(correct*ln(displayed) + (1-correct)*ln(1-displayed))`, both on the
pick-oriented displayed probability against whether that pick was right. Both
are orientation-invariant on non-push games.

Paired **week-blocked** bootstrap, 20,000 samples, seed 20260821, within-week
correlation ZERO by owner mandate; whole `(season, week)` blocks resampled
with replacement. `probability_positive` is
`P(draw > 0) + 0.5 * P(draw == 0)`
(`nfl_ats.evidence_conventions.probability_positive_from_draws`).

Every cell is recorded with `nfl-ats weak-signals record`, league `nfl`, units
`brier` and `log_loss`, named
`displayed_confidence_<metric>_2020_2025`. Only a whole-interval wrong sign may
be recorded as `refuted_mechanism` / `wrong_sign_resolved`; everything else is
`unresolved_below_power`.

### Reuse discount, stated up front

The cells are motivated by a descriptive bucket table computed on the same
2020-2025 archive this lane grades on. That is the same stated discount
`docs/spread_regime_program.md` declares, not independent confirmation. What
protects the read is that the transform itself is walk-forward: no game's
displayed number is calibrated with its own week's results, so the paired
Brier difference is an honest out-of-time read even though the cell grid was
chosen with the archive in view.

### The serving decision, declared before scoring

**If the calibrated Brier is better than the stated Brier with
`probability_positive` above 0.5, the calibrated number is served as the
displayed "Decision score"** on the tracked Markdown card, on the This Week
page, and in the per-pick explanation sentence. That is an expected-value
call, not a threshold call: the reader is shown a number either way, and
declining a display that is more likely right than wrong is taking the other
side of that bet.

The raw stated probability is not deleted. It stays in the forecast's
`recommendations.csv` (`home_cover_probability`, untouched), in
`explanations.json`, and in the card's lineage sidecar, so every number the
reader sees remains traceable to the model's own output.

The served calibration table is fitted at publish time from the opener
evaluation matched to the ACTIVE model
(`nfl_ats.public_board.find_matching_opener_evaluation`), the same
model-keyed sourcing the served S3 home-side offset already uses (read:
`src/nfl_ats/home_side_location.py:262-296`). No percentage is a constant.

**The Best Pick nominator is not changed by this lane.** It has its own served
rule (`nominate_v2_small_spread`, restricted to spreads of 6.5 or less, owner
decision 2026-09-09) and continues to run on the model's own un-overlaid,
un-calibrated probabilities.

Thresholds govern only what this document may CLAIM. What is DISPLAYED is an
expected-value decision on the opener grade, full stop.

## Results

Every number in this section is **measured this session** by the commands at
the bottom; the tables live in
`artifacts/displayed_confidence/20260910T010759Z/`. Nothing above the Results
heading was written after any candidate number was seen.

### The defect, reproduced on this archive

The four-bucket table, recomputed here from the active model's own opener
archive (`bucket_diagnosis.csv`):

| bucket | n | mean stated | realised accuracy | stated overstatement |
|---|---|---|---|---|
| 0-6.5 | 1,104 | 55.61% | 56.16% | +0.55 pts |
| 7 | 74 | 56.51% | 51.35% | **-5.16 pts** |
| 7.5-10 | 194 | 56.17% | 51.55% | **-4.62 pts** |
| 10.5+ | 131 | 55.95% | 47.33% | **-8.62 pts** |

Stated confidence varies by 0.9 points across the four buckets; realised
accuracy varies by 8.8. That is the whole defect in one table.

Sharper still, the same 1,503 games sliced by what the model SAID rather than
by the line (`reliability.csv`, "before" rows):

| stated band | n | mean stated | realised accuracy | gap |
|---|---|---|---|---|
| [0.50, 0.55) | 741 | 52.29% | **56.28%** | +3.98 pts |
| [0.55, 0.60) | 522 | 57.24% | **53.07%** | -4.17 pts |
| [0.60, 1] | 240 | 63.22% | **52.50%** | **-10.72 pts** |

The model's confidence is not merely flat, it is **inverted**: the picks it is
least sure about have been its best, and the 240 it called better than 60%
came in at 52.50%. Those are the rows a reader leans on hardest.

### Headline

Paired difference, stated minus calibrated, so positive favours the calibrated
display. 1,503 non-push opener-graded games; week-blocked paired bootstrap,
20,000 samples, seed 20260821.

| metric | stated (served today) | calibrated | paired difference | 95% week-blocked | `probability_positive` |
|---|---|---|---|---|---|
| Brier | 0.251581 | 0.250468 | **+0.001113** | [-0.001233, +0.003516] | **0.8224** |
| log loss | 0.696618 | 0.694229 | **+0.002388** | [-0.002454, +0.007348] | **0.8333** |

Per line bucket:

| metric | bucket | n | stated | calibrated | difference | 95% week-blocked | `probability_positive` |
|---|---|---|---|---|---|---|---|
| Brier | 0-6.5 | 1,104 | 0.248991 | 0.248614 | +0.000377 | [-0.002010, +0.002772] | 0.6234 |
| Brier | 7 | 74 | 0.250155 | 0.253548 | -0.003394 | [-0.011806, +0.004916] | 0.2102 |
| Brier | 7.5-10 | 194 | 0.260299 | 0.252859 | **+0.007440** | [-0.002682, +0.017563] | **0.9253** |
| Brier | 10.5+ | 131 | 0.261304 | 0.260816 | +0.000488 | [-0.010270, +0.011747] | 0.5316 |
| log loss | 0-6.5 | 1,104 | 0.691368 | 0.690459 | +0.000910 | [-0.004008, +0.005868] | 0.6442 |
| log loss | 7 | 74 | 0.694072 | 0.701226 | -0.007154 | [-0.024485, +0.009865] | 0.2039 |
| log loss | 7.5-10 | 194 | 0.714204 | 0.698979 | **+0.015225** | [-0.005394, +0.035886] | **0.9263** |
| log loss | 10.5+ | 131 | 0.716248 | 0.715021 | +0.001228 | [-0.020682, +0.024264] | 0.5391 |

Both overall metrics favour the calibrated display, and the largest gain is in
7.5-10, the bucket the diagnosis named. Exactly-7 goes the other way
(`probability_positive` 0.2102 on Brier, 0.2039 on log loss) on 74 games; it
is recorded `unresolved_below_power` and stays open, and it is not grounds to
withhold the transform from a display whose overall read is 0.82.

Where the overstatement ends up, per bucket (`bucket_diagnosis.csv`):

| bucket | stated gap | calibrated gap |
|---|---|---|
| 0-6.5 | +0.55 pts | -0.34 pts |
| 7 | -5.16 pts | -3.64 pts |
| 7.5-10 | -4.62 pts | **-0.89 pts** |
| 10.5+ | -8.62 pts | **-3.73 pts** |

The gap between what the display promises and what the model delivers shrinks
in all three problem buckets, most of all on 10.5+ where it more than halves.

### Reliability, before and after

| stage | displayed band | n | mean displayed | realised accuracy | gap |
|---|---|---|---|---|---|
| before | [0.50, 0.55) | 741 | 52.29% | 56.28% | +3.98 pts |
| before | [0.55, 0.60) | 522 | 57.24% | 53.07% | -4.17 pts |
| before | [0.60, 1] | 240 | 63.22% | 52.50% | -10.72 pts |
| after | [0.50, 0.55) | 537 | 51.03% | 54.93% | +3.91 pts |
| after | [0.55, 0.60) | 848 | 57.31% | 55.54% | -1.77 pts |
| after | [0.60, 1] | 118 | 61.90% | 45.76% | -16.14 pts |

**The last row needs its explanation stated, not buried.** It looks worse after
than before, and it is the one place the transform does not visibly help. The
reason is mechanical: 100 of those 118 games are 2020-2022, when the cells had
little or no history and the shrinkage toward the game's own stated value left
a high number almost unchanged. By season the band holds 48 games in 2020, 27
in 2021, 25 in 2022, 13 in 2023, 2 in 2024 and 3 in 2025 — it drains as history
accumulates. A 2026 card, which sees the full 2020-2025 archive, prints nothing
above 0.60 at all (see the Week 1 table below). The archive's warm-up years are
carried in the walk-forward read on purpose; they are not what a reader will
see.

The honest summary of the two tables is: the transform removes the inversion in
the two large bands (the 0.55-0.60 band's overstatement falls from 4.17 points
to 1.77 on 848 games) and empties the band that was worst, rather than
correcting it in place.

### The served calibration table

Fitted on the whole 2020-2025 archive, which is what a Week 1 2026 card sees
(`calibration_cells.csv`):

| bucket | band | prior games | prior wins | realised |
|---|---|---|---|---|
| 0-6.5 | [0.50, 0.55) | 557 | 318 | 57.09% |
| 0-6.5 | [0.55, 0.60) | 379 | 210 | 55.41% |
| 0-6.5 | [0.60, 1] | 168 | 92 | 54.76% |
| 7 | [0.50, 0.55) | 34 | 15 | 44.12% |
| 7 | [0.55, 0.60) | 22 | 14 | 63.64% |
| 7 | [0.60, 1] | 18 | 9 | 50.00% |
| 7.5-10 | [0.50, 0.55) | 89 | 54 | 60.67% |
| 7.5-10 | [0.55, 0.60) | 71 | 30 | 42.25% |
| 7.5-10 | [0.60, 1] | 34 | 16 | 47.06% |
| 10.5+ | [0.50, 0.55) | 61 | 30 | 49.18% |
| 10.5+ | [0.55, 0.60) | 50 | 23 | 46.00% |
| 10.5+ | [0.60, 1] | 20 | 9 | 45.00% |

The 10.5+ row a reader never saw: on 20 archive games where the model said
better than 60%, it was right 45.00% of the time. With 20 pseudo-observations
pulling toward a stated 0.64 the served display lands near 0.54 — not 0.64, and
not 0.45 either.

The small cells cut both ways and this is stated plainly rather than smoothed:
`7.5-10 / [0.50, 0.55)` is 54 of 89 (60.67%), so the transform RAISES a
modest-looking pick on a big spread. That is the predeclared rule doing what it
says on a 89-game cell, not a separate judgement, and it is why the exactly-7
bucket lands on the wrong side of zero above.

### Week 1 2026, the sixteen scores a reader sees

Measured through the served publish path (`_publication_context` ->
`_published_card`), so this is the tracked Markdown card's own table. **Sides
are identical in all sixteen rows** (`week1_display.csv`, `side_changed` false
on every row; the two card tables below differ in one column only).

| matchup | pick | line size | bucket | before | after | change |
|---|---|---|---|---|---|---|
| ARI at LAC | ARI +9.5 | 9.5 | 7.5-10 | **64.2%** | **53.4%** | -10.8 |
| WAS at PHI | WAS +5.5 | 5.5 | 0-6.5 | **63.3%** | **55.7%** | -7.6 |
| MIA at LV | ★ MIA +3.5 | 3.5 | 0-6.5 | 56.1% | 55.4% | -0.7 |
| SF at LA | SF +3.5 | 3.5 | 0-6.5 | 56.0% | 55.4% | -0.6 |
| ATL at PIT | PIT -3.5 | 3.5 | 0-6.5 | 55.9% | 55.4% | -0.5 |
| BAL at IND | IND +3.5 | 3.5 | 0-6.5 | 55.4% | 55.4% | +0.0 |
| BUF at HOU | HOU +1.5 | 1.5 | 0-6.5 | 54.2% | 57.0% | +2.8 |
| GB at MIN | MIN -1.5 | 1.5 | 0-6.5 | 54.1% | 57.0% | +2.9 |
| CLE at JAX | JAX -8.5 | 8.5 | 7.5-10 | 53.3% | 59.3% | +6.0 |
| NO at DET | NO +6.5 | 6.5 | 0-6.5 | 52.3% | 56.9% | +4.7 |
| DEN at KC | DEN +2.5 | 2.5 | 0-6.5 | 51.8% | 56.9% | +5.1 |
| CHI at CAR | CAR +2.5 | 2.5 | 0-6.5 | 51.6% | 56.9% | +5.3 |
| NYJ at TEN | NYJ +1.5 | 1.5 | 0-6.5 | 51.6% | 56.9% | +5.3 |
| TB at CIN | CIN -3.5 | 3.5 | 0-6.5 | 50.9% | 56.9% | +6.0 |
| NE at SEA | NE +3.5 | 3.5 | 0-6.5 | 50.4% | 56.9% | +6.5 |
| DAL at NYG | DAL -2.5 | 2.5 | 0-6.5 | 50.2% | 56.9% | +6.7 |

The two lines the brief names lose the most: ARI +9.5 drops 10.8 points and
WAS +5.5 drops 7.6. The spread of displayed scores narrows from 50.2-64.2 to
53.4-59.3, which is the transform saying, correctly, that this model does not
separate its own picks as sharply as its raw probability implies.

The per-pick sentence changes with it, measured through `explain_pick`:

- before: *"ARI +9.5. The model makes ARI a 64.2% cover, a strong lean."*
- after: *"ARI +9.5. The model gives ARI a 53.4% chance to cover once its
  record on spreads this size is counted in, a lean."*

and `explanations.json` carries both numbers on every pick
(`model_probability.probability` 0.5340835792301,
`model_probability.stated_probability` 0.6420256639212699 for that game). The
forecast's `recommendations.csv` is not rewritten at all — its
`home_cover_probability` stays the model's raw output.

### The one thing this change made worse, and how it was repaired

The board's three-word strength meter (`confidence_word`) is applied to the
DISPLAYED number, by its own design, so that the word and the number can never
contradict each other. Its two edges used to be hand-set literals -- above 0.56
"strong", 0.53 and up "lean", else "slight" -- and compressing the displayed
numbers into 53-59% pushed almost everything over the 0.56 line. Measured on
the rendered This Week page:

| | slight | lean | strong |
|---|---|---|---|
| before this lane | 7 | 6 | 3 |
| calibrated score, hand-set edges | 0 | 6 | 10 |
| calibrated score, derived edges | 1 | 14 | 1 |

**The edges are now derived, not chosen.** They are the terciles of the
displayed score's own walk-forward distribution on the 1,503-game 2020-2025
opener archive of the ACTIVE model, rounded to the three decimals the board
prints (`derive_strength_bands`), so a Week 1 2026 card reads
`lean_min` 0.547 and `strong_min` 0.572. They are recomputed from the
archive whenever the active model changes and are written into the forecast's
`displayed_confidence.json` beside the cells; no percentage on the meter is a
constant, and a board with no archive behind it shows no strength word rather
than a made-up one.

**Why terciles and not realised-accuracy breakpoints.** The other candidate was
to cut the score where a reader's expectation genuinely changes -- bands at,
say, 55% / 58% / 61% realised. Measured on the same archive, under the served
edges above, those breakpoints do not exist:

| band | games | mean displayed | realised accuracy |
|---|---|---|---|
| slight | 496 | 50.71% | 54.23% |
| lean | 500 | 56.18% | 56.60% |
| strong | 507 | 59.29% | 52.86% |

Realised accuracy does not rise across the three bands; the top third is the
worst of them, which is the same inversion the reliability table above
records, surviving the transform. Cutting at accuracy breakpoints would
therefore have meant either inventing edges the data does not support or
labelling the model's most confident picks "slight" -- an unexplained flip in
everything but name. Terciles say something the data does support: **where this
pick sits among the reads this model actually produces**, one third of the
archive in each band by construction. That is what the board's legend now says
in pool-player words, and it is deliberately not a promised hit rate. The
per-bucket weak spots stay where they belong, on the Model page's own table.

The 1 / 14 / 1 split on Week 1 2026 is not the meter failing. Fourteen of the
sixteen calibrated scores land between 0.547 and 0.572 because this card's
picks fall into two adjacent cells, which is the transform reporting -- as the
headline table already did -- that the model does not separate these sixteen
picks as sharply as its raw probability implied.

### Decision

**Serve the calibrated number.** The paired Brier difference is +0.001113 with
`probability_positive` 0.8224 and the paired log-loss difference is +0.002388
at 0.8333, both on the pool's own opener grade. Showing the raw number instead
is taking the 18% side of that bet, for a figure the owner reads and acts on.
Every bucket cell is `unresolved_below_power` and stays open, including
exactly-7, which currently favours the stated number at 0.2102.

**No side changed**, on any of the 1,503 archive games or any of the sixteen
Week 1 picks. The pick is fixed from `home_cover_probability` before the
transform runs, and the transform writes a separate, pick-oriented column that
no side is ever derived from.

### Registry

Ten cells recorded under family `mod18_displayed_confidence_v1`:
`displayed_confidence_brier_2020_2025`,
`displayed_confidence_log_loss_2020_2025`, and the eight per-bucket cells
`displayed_confidence_<metric>_<bucket>_2020_2025`. All ten are
`unresolved_below_power` — a favourable interval is not a closing ground
either, and neither is an unfavourable one. The exact argv lists are saved as
`record_commands.json` in the artifact directory.

## What the reader is told

The card's footnote and the This Week page's column now say, in pool-player
words, that the score is the model's cover chance **adjusted for how it has
actually done on spreads this size**, and that big favourites and big underdogs
have been its weak spot. The column header changes from "Cover prob" to
"Cover chance". That sentence is the mechanism, stated where the number is.

## Commands run

```
python scripts/displayed_confidence_measure.py \
  --archive artifacts/opener_evaluation/20260910T005854Z \
  --forecast artifacts/margin_predictions/2026-week-01-20260910T005452Z \
  --data-root data \
  --out artifacts/displayed_confidence/20260910T010759Z

nfl-ats weak-signals record ...   (10 cells, mod18_displayed_confidence_v1;
the exact argv lists are saved as record_commands.json in the artifact
directory)
```
