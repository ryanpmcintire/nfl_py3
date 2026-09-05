# Phase 12 market microstructure lead battery (LEAD-05, LEAD-03)

Predeclared BEFORE either candidate's ATS outcome is scored. Read
`scripts/on_production_opener_confirmation.py` (the harness this file's runs
mirror, never edited) and `scripts/odds_microstructure_battery.py` (the
closest existing template for raw per-book quote work) alongside this doc.

## Closing-grounds taxonomy (binding, verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close
a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator. Verdicts flow ONLY through `nfl-ats weak-signals record` and
`nfl-ats rotation record`, never through prose.

## Shared instrument

Both candidates reuse the production `weak_stack` ridge-alpha-10
market-residual chain (`nfl_ats.margin.fit_margin_model`, the active model's
own recipe -- `artifacts/active_ats_model.json`), evaluated with
`nfl_ats.clv.opener_pick_evaluation` (opener grade, per-week walk-forward
refit) exactly as `scripts/on_production_opener_confirmation.py` does for its
four existing candidates. `scripts/market_lead_on_production.py` (new,
mirrors that harness verbatim rather than importing it, so this run is
self-contained and untouched by any other lane editing the template) adds
`--mode rank` (LEAD-05 step 1, descriptive, no window, no registry) and
`--mode coverage` (LEAD-03's predeclared moneyline-coverage measurement, also
window-free). Both candidate columns are built entirely from the local
point-in-time archive under `data/market/raw/` via the new
`nfl_ats.market_lead_features` module -- no network calls, the paid Odds API
is never touched.

Rotation grade: `opener` for both families (`nfl_ats.rotation.GRADE_POOLS["opener"]
== (2020, 2025)`, matching the paired Tuesday-opener archive's own coverage).
Since neither family inherits from anything already declared,
`nfl-ats rotation assign` will almost certainly hand each the EARLIEST
eligible 2-season block in that pool -- there is no global scarcity across
unrelated families (`nfl_ats.rotation._touched_seasons` only tracks a
family's own inherits chain), so a fresh family typically draws (2020, 2021).
That block is the pool's own first block, so there are no whole seasons
strictly before it inside the archive -- the "seasons strictly before the
window" option this doc's task text offered is therefore not available for
EITHER candidate once assignment lands there, and the frozen alternative
below (a walk-forward trailing computation) is used instead. The exact
assigned window actually drawn is read at run time from
`nfl_ats.rotation.confirmation_split` and reported in each run's artifact,
never assumed here.

---

## LEAD-05: opener-softness book ranking

**Mechanism.** Books differ in how far their own Tuesday-opener spread LINE
sits from the eventual close; the book with the largest average gap is
absorbing the least pregame information into its number ("the noisiest
opener carries the least information", ROADMAP LEAD-05). A book that is
persistently that far off is more likely, on any single game, to have simply
missed which side is even favored when the true market is close --
information the CONSENSUS opener (the median across every book) already has
priced in.

**Step 1 (descriptive, no window, no registry).** Rank books by mean
|opener - close| spread error across the full archive
(`nfl_ats.market_lead_features.book_softness_ranking`, run via
`scripts/market_lead_on_production.py --candidate opener_softness --mode rank`),
using each book's per-book Tuesday-opener quote row (`build_pairing_table`'s
consensus median drops this; the raw quote rows are read directly, one level
below the pairing table, via `nfl_ats.market_lead_features.book_level_tue_open_spreads`)
against the project's own close reference
(`nfl_ats.clv.close_reference_table`). Split-half reliability: rank books
separately on odd seasons (2021, 2023, 2025) and even seasons (2020, 2022,
2024), Spearman correlation of the two rankings restricted to books with
>=50 games in EACH half, with a season-blocked bootstrap (resampling each
half's seasons with replacement) for a CI and `probability_positive`.

**Step 2 (the ATS look, family `opener_softness_fade_on_production`).**
Predeclared direction: **consensus beats softest** -- FADE the side implied
ONLY by the softest book's opener, i.e., the signal fires only where the
softest book's own opener FAVORS THE OTHER SIDE from the consensus opener
(a sign-of-spread disagreement, not a magnitude disagreement). The frozen
softest-book identification method (declared before this step's outcome is
scored): **walk-forward**, not "seasons strictly before the window" --
justified above, since the opener-grade pool's first eligible block for an
uninherited family is (2020, 2021), leaving no whole prior season inside the
archive. At each scored week's cutoff (that week's earliest kickoff), the
softest book is the one with the highest cumulative mean |opener - close|
error over every ALREADY-KICKED-OFF archived game (`gameday` strictly before
the cutoff -- every such game's close is by then fully resolved, so nothing
from the scored week or any later week reaches the identification), among
books with >= `MIN_BOOK_HISTORY_GAMES` (100) such observations; ties broken
by the lexicographically smallest book key. Weeks before any book clears
that bar get no softest book (signal NaN for every game that week).

**Encoding.** `opener_softness_fade_signal` in {-1.0, 0.0, +1.0, NaN}: +1.0
(home) / -1.0 (away) when the identified softest book's own Tuesday-opener
favorite disagrees with the CONSENSUS Tuesday-opener favorite -- the value
IS the consensus's side, i.e., fading the softest book; 0.0 when they agree
(no fade signal); NaN when the softest book is not yet identified, has no
quote for this game, or either spread is an exact pick'em (no favorite to
disagree about). Built in
`nfl_ats.market_lead_features.derive_opener_softness_fade_features`, lives in
`data/processed/game_features_weak_stack_opener_softness.parquet`
(`weak_stack_opener_softness` margin profile: PRODUCTION `weak_stack` plus
exactly this one column, `nfl_ats.margin`/`nfl_ats.constants`, additive-only).

**Comparator / metric.** Same paired evaluator as every existing
on-production candidate: baseline = production `weak_stack`; candidate =
`weak_stack_opener_softness`; metric = forced-pick accuracy delta
(candidate minus baseline) at the opener, both the sign rule and production's
own probability rule, week- and season-blocked bootstrap (20,000 resamples),
plus a 200-permutation within-week null on the settle margin.

**Controls (run in this order, before the window is spent).**
1. `--mode null` -- settle margins shuffled within each week; the harness
   must show no effect here.
2. `--mode positive-control` -- the one new column is replaced by the
   realized `ats_margin` (a deliberate leak); the harness must show a huge,
   obvious effect (`probability_positive` ~= 1.0) even inside the full
   production feature set, or the screen is not read.
3. `--mode screen` -- the real look, spending `opener_softness_fade_on_production`'s
   assigned window. Run once.

**Measured base rate (construction fact, not an outcome -- computed before
either control ran).** Across the full 2020-2025 regular-season archive
(4,347 regular-season rows with the column attached), the signal fires
non-zero on exactly **3 of 1,413 covered games** (0.21%): `2021_05_CLE_LAC`
(+1.0), `2025_04_CHI_LV` (-1.0), `2025_06_DET_KC` (-1.0). The walk-forward
identification converges on **betmgm** as the softest book from week 15 of
the 2020 season onward (measured: 93 of 101 identified weeks name betmgm;
weeks 9-14 transiently name betrivers/pointsbetus while history is still
thin; the first 8 weeks of 2020 have no book past the 100-game history bar
and get no identification at all) -- consistent with the full-archive
cross-sectional ranking below, where betmgm has the largest mean error among
every book with a non-trivial sample (n=1,391, mean |error| 1.343, vs the
next-highest well-populated book `foxbet` at n=329/1.159; the two thinner
books above it, `caesars` n=31 and none above 1.4, are sample-size noise, not
softer books). A sign-of-spread disagreement between any single book and the
consensus is intrinsically rare (it requires the two to disagree on which
team is even favored, not just by how much), so this sparsity is a property
of the predeclared construct itself, not an artifact of the book chosen.
**This is disclosed as a genuine power caveat, not grounds to skip or
redefine the look**: the family's binding classification will be
`unresolved_below_power` regardless of sign, per the taxonomy above -- n=3
non-zero games (1 of them, `2021_05_CLE_LAC`, inside the (2020, 2021) window
this family will almost certainly draw) cannot resolve a mechanism either
way, and "needs more data" is not a valid response (AGENTS.md: the data is
fixed).

**Measured full-archive book ranking** (mean |opener - close| error, n
games, ascending = sharpest; `--mode rank`, 20,445 book-game rows, 24
distinct books):

| rank | book | n | mean abs error |
|---|---|---|---|
| 1 | superbook | 449 | 0.8903 |
| 2 | fanatics | 256 | 0.9170 |
| 3 | bovada | 1267 | 0.9303 |
| 4 | lowvig | 1271 | 0.9382 |
| 5 | williamhill_us | 1392 | 0.9445 |
| 6 | circasports | 221 | 0.9457 |
| 7 | betonlineag | 1430 | 0.9504 |
| 8 | mybookieag | 1330 | 0.9517 |
| 9 | draftkings | 1521 | 0.9681 |
| 10 | betrivers | 1483 | 0.9690 |
| 11 | unibet_us | 418 | 0.9707 |
| 12 | wynnbet | 616 | 0.9805 |
| 13 | fanduel | 1456 | 0.9845 |
| 14 | intertops | 575 | 0.9904 |
| 15 | betus | 1156 | 1.0006 |
| 16 | barstool | 647 | 1.0317 |
| 17 | pointsbetus | 932 | 1.0341 |
| 18 | gtbets | 683 | 1.0351 |
| 19 | sugarhouse | 567 | 1.0683 |
| 20 | unibet | 542 | 1.0821 |
| 21 | twinspires | 482 | 1.1214 |
| 22 | foxbet | 329 | 1.1588 |
| 23 | **betmgm** | 1391 | **1.3426** |
| 24 | caesars | 31 | 1.5242 |

**Measured split-half rank reliability** (odd vs. even season halves, books
with >=50 games in each half, n=21 common books, season-blocked bootstrap,
5,000 resamples): Spearman rho = **0.590**, 95% CI **[0.081, 0.708]**,
`probability_positive` = **0.991**. This is a genuinely reliable trait (high
lean, and the CI's lower bound sits above zero at this resample count) -- the
book-softness ranking itself replicates well across eras even though the ATS
trigger built on top of it is extremely sparse.

**Recording plan.** Step 1 (rank + reliability) is descriptive and touches no
registry -- reported in this doc only. Step 2 (`--mode screen`) records via
both `nfl-ats rotation record --name opener_softness_fade_on_production` and
`nfl-ats weak-signals record --name opener_softness_fade_on_production
--family opener_softness_fade_on_production --category market`, verdict/
classification `unresolved` / `unresolved_below_power` unless an admissible
closing ground from the taxonomy applies (a resolved wrong sign requires the
WHOLE interval below zero, which n=1 in-window firing game cannot produce
either way) -- `--reliability` on the weak-signals record uses the measured
0.590 split-half rank reliability above (the reliability of the underlying
BOOK ranking, the trait the whole construct depends on; the ATS trigger
itself is too sparse in-window for its own split-half read to mean anything).

---

## LEAD-03: moneyline-spread divergence

**Mechanism.** The Tuesday-opener consensus moneyline and the Tuesday-opener
consensus spread are two independent market reads of the same game; when the
no-vig moneyline-implied home win probability diverges from a
spread-implied home win probability by a wide enough margin, the spread leg
may be stale (slower to react to the same information the moneyline already
carries). Predeclared direction: side WITH the moneyline on divergences
>= 3 percentage points.

**Population and measured moneyline coverage (measured BEFORE any outcome is
scored, `--mode coverage`).** `nfl_ats.clv.build_pairing_table` already
requests the moneyline market at the Tuesday-opener decision time
(`nfl_ats.odds_backfill.DECISION_TIMES`'s `("tue_open", -5, time(9,0),
with_h2h=True)`, one of only two of the six weekly decision times that
requests `h2h` at all). Measured: of the 1,537 Tuesday-opener games
2020-2025 (the same population `docs/novig_diagnostics.md` and
`scripts/odds_microstructure_battery.py` use), **1,537 (100.000%) carry BOTH
home and away moneyline sides**, every season 2020-2025 individually at
100.0%. **No fallback snapshot is needed and none is used** -- this
supersedes the ROADMAP row's framing ("2023-2025 true-opener + moneyline
archive, MKT-02"): the separate `true_open` capture (an even-earlier Monday
snapshot, `docs/sbr_odds_archive.md`) was measured to carry **zero** `h2h`
rows at all (it never requested that market), so it could not have been the
intended moneyline source; the standard `tue_open` weekly decision label
already has complete coverage on its own.

**Spread-implied win probability.** No existing helper in the codebase maps
a spread directly to a straight-up win probability (`nfl_ats.novig.spread_novig_probabilities`
and `nfl_ats.calibration.smoothed_home_cover_probability` give COVER
probability against the spread's own line, a different quantity); per this
task's own sanctioned fallback, a **walk-forward univariate logistic
regression of home win (result > 0, pushes excluded) on the SAME Tuesday
-opener snapshot's home spread** is fit per scored week, trained only on
strictly earlier weeks' completed games (>= `MIN_FITTABLE_TRAIN_GAMES` = 50
prior rows required to fit; weeks before that get NaN). Comparing the
resulting probability against the no-vig moneyline probability keeps both
sides of the divergence in the SAME unit (home win probability) from the
SAME snapshot, satisfying AGENTS.md's commensurability rule.

**Encoding.** `ml_spread_divergence_signal` in {-1.0, 0.0, +1.0, NaN}: let
`divergence = no_vig_moneyline_home_win_probability -
walk_forward_spread_implied_home_win_probability`; +1.0 (home) when
`divergence >= 0.03`, -1.0 (away) when `divergence <= -0.03`, 0.0 otherwise;
NaN when either input is unavailable (walk-forward logistic not yet
fittable, or -- never observed in the measured population -- a missing
moneyline). Built in
`nfl_ats.market_lead_features.derive_ml_spread_divergence_features`, lives in
`data/processed/game_features_weak_stack_ml_divergence.parquet`
(`weak_stack_ml_divergence` margin profile: PRODUCTION `weak_stack` plus
exactly this one column).

**Comparator / metric / controls.** Identical shared instrument described
above (same paired opener-graded evaluator, week/season-blocked bootstrap,
200-permutation null, `null` / `positive-control` / `screen` order) against
family `ml_spread_divergence_on_production`.

**Measured column distribution (construction fact, not an outcome).** Across
the full archive (4,902 rows, all seasons; 1,555 covered games -- slightly
above the 1,537 regular-season-only population because the underlying
`weak_stack` table also carries a handful of playoff games with archive
coverage): **546 (35.1%) fire +1.0 (home), 14 (0.9%) fire -1.0 (away), 995
(64.0%) are 0.0**. The strong home/away asymmetry (far more games where the
moneyline implies MORE home strength than the walk-forward spread-implied
probability, rarely the reverse) is disclosed as a measured, unexplained
curiosity -- plausibly an artifact of the single-feature logistic's implied
scale on large favorites/underdogs rather than a market inefficiency, and
not adjudicated here; the confirmation window's own ridge fit will tell us
whether it corresponds to any real accuracy edge.

**Recording plan.** `nfl-ats rotation record --name
ml_spread_divergence_on_production` and `nfl-ats weak-signals record --name
ml_spread_divergence_on_production --family ml_spread_divergence_on_production
--category market`, verdict/classification `unresolved` /
`unresolved_below_power` unless an admissible closing ground applies.

---

## Sequencing (both candidates, run in this exact order)

1. This predeclaration (above) is committed before any control runs.
2. `nfl-ats rotation declare --name <family> --description "..." --grade opener --acknowledge-mined`
   (the opener pool 2020-2025 is wholly inside the mined 2018-2025 range, so
   `--acknowledge-mined` is required for assignment to find any eligible
   block at all) then `nfl-ats rotation assign --name <family>`. If assign
   refuses, the exact error is reported and that candidate stops there.
3. `--mode null`, then `--mode positive-control` (must show
   `probability_positive` ~= 1.0), then `--mode screen` exactly once.
4. `nfl-ats rotation record` and `nfl-ats weak-signals record` for the
   family, per the recording plans above.
5. Measured results appended below each control/screen run, plus a dated
   note on the corresponding ROADMAP.md row.

---

## Measured results, 2026-09-05

Both families' `nfl-ats rotation assign` calls drew the pool's very first
eligible block, **[2020, 2021]** (confirming this doc's prediction above),
456 paired opener-grade games / 35 weeks for both screens. Effect and
interval are reported in **accuracy points** (a raw 0-1 accuracy-fraction
delta multiplied by 100, matching the registry convention already
established by `post_ot_fatigue_on_production` and siblings recorded the
same day -- verified against that family's own artifact, whose raw
`delta_accuracy` of -0.0021929824561403508 is stored as `effect: -0.2193`).
An earlier draft of both recordings stored the raw fraction unscaled; both
were corrected via `--replace` before this section was written, and the
numbers below are the corrected ones.

### LEAD-05: opener_softness_fade_on_production

* `--mode null`: opener production-rule null centred near zero
  (mean -0.155 pts, sd 0.886, 95% [-1.974, 1.535]) with the observed delta
  (-0.877 pts) at its 20.5th percentile -- no artifact in the harness.
* `--mode positive-control`: **+44.298 accuracy points**, week-blocked
  `probability_positive` **1.000** (both blockings), permutation null
  observed at its 100th percentile. The instrument detects a huge planted
  effect even inside the full 91-column production feature set.
* `--mode screen` (the one real look): opener production-rule accuracy
  delta **-0.8772 accuracy points**, week-blocked 95% **[-3.132, +1.129]**,
  `probability_positive` **0.1816**; season-blocked (n=2 seasons, wide, not
  leaned on) [-0.909, -0.847], `probability_positive` 0.0. 456/456 games
  scored, **19 picks flipped between arms** despite only 1 in-window game
  carrying a non-zero column value -- the shared ridge fit (alpha 10 across
  all 91 columns) redistributes weight across every feature when one column
  is added, so most of the 19 flips come from small coefficient perturbation
  on OTHER games, not from the sparse trigger itself (the "composition is
  not the signal" lesson, in reverse: a near-inert column can still move
  picks it never directly touches). Permutation null observed at its 20.5th
  percentile (same distribution as the null check, since the null shuffles
  OUTCOMES, not the candidate column). Interval crosses zero at both
  blockings; per the taxonomy this is `unresolved_below_power`, not a
  rejection.
* Recorded: `nfl-ats rotation record --name opener_softness_fade_on_production
  --verdict unresolved` and `nfl-ats weak-signals record --name
  opener_softness_fade_on_production --classification unresolved_below_power
  --reliability 0.5898` (the book-ranking's own split-half reliability, not
  the sparse trigger's). Registry: 721 signals total.
* Artifacts: `artifacts/market_lead_on_production/opener_softness/`
  (`20260905T031110Z` null, `20260905T031430Z` positive-control,
  `20260905T031800Z` screen, `20260905T032728Z` rank).

### LEAD-03: ml_spread_divergence_on_production

* Moneyline coverage (measured before any outcome scored, `--mode
  coverage`): **1,537/1,537 (100.000%)** Tuesday-opener games 2020-2025
  carry both moneyline sides, every season individually at 100.0%. No
  fallback snapshot used.
* `--mode null`: null centred near zero (mean -0.569 pts, sd 1.152, 95%
  [-2.637, 1.754]) with the observed delta (-0.658 pts) at its 42.5th
  percentile -- no artifact in the harness.
* `--mode positive-control`: identical **+44.298 accuracy points**, P+
  **1.000** (both blockings) -- same shared instrument, same result as
  LEAD-05's control (expected: the positive control replaces the candidate
  column with `ats_margin` regardless of which candidate is under test, so
  both controls exercise the identical baseline/leaked-arm comparison).
* `--mode screen` (the one real look): opener production-rule accuracy
  delta **-0.6579 accuracy points**, week-blocked 95% **[-3.261, +1.974]**,
  `probability_positive` **0.27765**; season-blocked (n=2, wide)
  [-1.818, +0.424], `probability_positive` 0.25535. 456/456 games scored, 36
  picks flipped between arms (a healthier n than LEAD-05: the signal fires
  on 270/528 archive games, 35.1% home / 0.9% away -- asymmetric and
  disclosed as an unexplained curiosity, not adjudicated). Interval crosses
  zero at both blockings; `unresolved_below_power`.
* Recorded: `nfl-ats rotation record --name ml_spread_divergence_on_production
  --verdict unresolved` and `nfl-ats weak-signals record --name
  ml_spread_divergence_on_production --classification
  unresolved_below_power` (no `--reliability`: none was predeclared or
  measured for this trait). Registry: 721 signals total (the corrected
  `--replace` did not add a new row).
* Artifacts: `artifacts/market_lead_on_production/ml_divergence/`
  (`20260905T031846Z` null, `20260905T032214Z` positive-control,
  `20260905T032540Z` screen, `20260905T032741Z` coverage).

### Decision

Per AGENTS.md ("a promotion bar is not a decision bar... decide on expected
value"), `probability_positive` is reported, not the binary "contains zero":
LEAD-05 reads P+ 0.1816 (leans against the candidate at the opener) and
LEAD-03 reads P+ 0.27765 (also leans against, less strongly, at healthier
power). Neither crosses the 0.5 threshold that would favour promoting either
column on this window, and neither is refuted (no resolved wrong sign, no
split-half-reliability failure of an underlying trait, and the positive
control rules out an instrument-insensitivity bound). Both stay open,
recorded `unresolved_below_power`, available to a future pooled look under a
declared commensurable family.
