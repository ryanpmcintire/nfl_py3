# MKT-15: leader-weighted late-week refresh

## Predeclaration — 2026-09-05, before scoring

Measured (`artifacts/experiments/sharp_book_movement/coverage_first.csv`):
the existing observed-movement intraday archive contains 272 regular-season
games in each of 2023, 2024 and 2025; 259, 253 and 259 respectively have at
least one Wednesday–Saturday spread move from Bovada, William Hill or MyBookie,
observed strictly before min(kickoff, Sunday 16:00 America/New_York).
Coverage was computed without loading outcomes or evaluating picks.

Read (`docs/book_leadership.md:41–57`): freeze these measured descriptive lead
shares as weights, without refitting: bovada .6135, williamhill_us .6405,
mybookieag .6276, draftkings .4477, betus .4035, betrivers .3563,
pointsbetus .5397, fanatics .5569, lowvig .2885, fanduel .3045,
betmgm .3026, betonlineag .2200. Exclude unlisted books from BOTH arms.
The three named leaders define the coverage diagnostic, not an outcome filter.

For each game/book, deduplicate spread sides at each observed timestamp and
compute consecutive changes in standardized home spread (positive means toward
home). Sum only changes observed Wednesday through Saturday of the game's own
scheduled week, strictly before the decision cutoff; the preceding quote may
be from Monday/Tuesday of that same week. Both observations must precede the
cutoff; reject quotes whose provider update is later than observed time.
The first observed quote is not a move. Use each eligible book with at least
one measurable late-week increment, including zero increments, once in the
denominator. Net movement is sum(weight * book net change) / sum(weight).
This normalization avoids mistaking capture frequency for stronger evidence.

Freeze TWO arms: leadership weights above and the SAME computation with every
listed book weighted 1.0. FOLLOW the net movement if its absolute value is
at least **0.5 point**; otherwise retain production. Missing exposure retains
production. No tuning, no search over thresholds or sign reversals.

Read (`scripts/observed_movement_channel.py:303–336`): reuse its intraday loader
and true-week correction, 2023–2025. Reuse `clv.opener_pick_evaluation` with
the active model configuration and current production feature table, weekly
chronological refits, production `pick_home_at_open_probability_rule` and
`margin_vs_open`. This is a refresh overlay screen, not a new fitted feature.
Both arms and control use the identical full paired archive population,
including unflagged games; exclude opener pushes only from accuracy.
Retain game-level predictions, exposure, cutoffs, season results and flips.

Freeze a week-blocked bootstrap: 20,000 draws, seed **2026090518**, whole
(season, week) blocks sampled with replacement, game-weighted paired accuracy
delta multiplied by 100 (accuracy_points), 95% percentile interval and fraction
of draws strictly positive. Compute using block sums/counts (equivalent to
concatenating sampled game rows). Report season stability, never select a season.
Also report leader-minus-equal on these same games, as a correlated comparison.

Positive control: FOLLOW nonzero closing movement from the Tuesday opener,
retain production on ties, grade at the Tuesday opener on the SAME games.
This is an oracle instrument diagnostic: a close after the pool deadline is
unavailable to a refresh and cannot itself be played or establish a
candidate-sized detection bound.

Exposure reliability: each team-season's fraction of games with an arm flag,
separately on odd and even scheduled weeks; Pearson correlation across paired
team-seasons, report pair count and raw correlation (undefined if constant).
Do not use outcomes for this diagnostic or manufacture zero reliability from
an undefined correlation.

Read (`docs/opener_evaluation.md:34–36`): the inherited opener evaluator swaps
spread but retains other close-era inputs; disclose that approximation.
Read (`docs/book_leadership.md:15–17`): weights were described using 2023–2025
quotes; this retrospective same-era screen is not independent temporal
confirmation of learned weights. Weights are frozen for subsequent use.
Read (`artifacts/movement_leads_battery/20260905T042928Z/metadata.json:1–5`):
that battery used an older opener artifact; this screen recomputes the active
configuration rather than inheriting its pick stream. Related observed-movement
results share games; these arms are correlated evidence, not independent votes.

## Binding taxonomy

An interval or CI that contains zero is NEVER grounds to reject, fail, or close an experiment. Only two grounds ever close a line of work: (1) refuted mechanism - a RESOLVED wrong sign (whole interval on the wrong side of zero) or zero split-half reliability; (2) bounded by a positive control proven able to detect an effect that size. Everything else is unresolved_below_power: record it, report probability_positive, never the binary "contains zero". If a record command errors, the verdict is wrong, not the validator. Decisions are expected value: probability_positive above 0.5 favours playing it as a refresh overlay; state what the result implies for the DECISION before what is wrong with it. Never say a lead needs more games; the data is fixed and the project is model-limited.

Record both arms in accuracy_points through `nfl-ats weak-signals record`;
the control and direct comparison share a declared family with the two arms.
The screen does not alter the published card or production wiring.

## Results — measured 2026-09-05

Inferred: I think the decision favours a late-week refresh over keeping the
production probability pick, and favours equal weights over leadership weights
when choosing between these two predeclared arms.

Measured (`artifacts/experiments/sharp_book_movement/20260905T205038Z/metadata.json`):
all arms share 816 games, 799 non-push grades and 54 weeks across 2023–2025;
production gets 422/799 correct (52.8160%). The active configuration was
recomputed with Gaussian probabilities and its feature-table digest checked.

| Measured arm (same artifact) | Correct / 799 | Accuracy | Delta accuracy_points | 95% week bootstrap | probability_positive | Flags / flips (816 games) |
|---|---:|---:|---:|---|---:|---:|
| Leadership weights | 435 | 54.4431% | +1.62703 | [-0.99751, +4.28752] | 0.88025 | 320 / 140 |
| Equal weights | 436 | 54.5682% | +1.75219 | [-0.86849, +4.37500] | 0.89890 | 329 / 143 |
| Closing-move control | 444 | 55.5695% | +2.75344 | [-1.10837, +6.68348] | 0.91255 | 600 / 257 |
| Leadership minus equal | — | — | -0.12516 | [-0.63532, +0.37975] | 0.25340 | — |

Measured (`.../per_game.parquet`, grouped by season): production gets
143/266, 141/266, 138/267 correct in 2023/2024/2025; leadership weights get
145/266, 144/266, 146/267; equal weights get 146/266, 144/266, 146/267.
Leadership season deltas are +0.75188, +1.12782, +2.99625 points; equal-weight
deltas are +1.12782, +1.12782, +2.99625. No season was selected after scoring.

Measured (`.../metadata.json`, exposure_reliability): odd/even team-season
flag-rate correlations are 0.0940792 for leadership and 0.1008829 for equal
weights, each over 96 paired team-seasons. These are raw correlations,
not transformed probabilities. Measured (`.../per_game.parquet`): 310 games
have kickoff after the Sunday pool deadline; the closing control is an oracle,
not a playable late refresh. Inferred: I think this control supports the
movement direction but does not prove a candidate-sized detection bound.

Measured (`nfl-ats weak-signals record`, exact argument vectors in
`artifacts/experiments/sharp_book_movement/20260905T205038Z/record_commands.json`):
four entries were recorded as `unresolved_below_power` in accuracy_points:
`sharp_book_movement_leader_2023_2025`,
`sharp_book_movement_equal_2023_2025`,
`sharp_book_movement_closing_control_2023_2025`, and
`sharp_book_movement_leader_minus_equal_2023_2025`.
Their family is `sharp_book_movement_refresh_2023_2025`; shared games,
same-era descriptive weights, the inherited close-era input approximation,
and the direct comparison's different baseline are disclosed in registry notes.
Inferred: I think the positive refresh decision and unresolved mechanism
classification are compatible; there is no measured leadership-weighting gain
over equal weighting in this screen, and no admissible terminal ground.

Measured (targeted pytest): 33 tests pass, including 11 new cases; the new
bootstrap fixture reproduces the existing row-resampling evaluator on unequal
week sizes with pushes excluded. Measured (`ruff format`, `ruff check`,
`mypy src`): formatting and lint pass; mypy reports no issues in 243 source files.
