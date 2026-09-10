# Best Pick nomination across eras (lane AX)

**Question.** The Best Pick nominator served since 2026-09-09
(`nfl_ats.best_pick_nomination.nominate_v2_small_spread`) is v2 restricted to
candidates whose frozen decision spread is 6.5 or less. It was promoted on
**+1.96 Best-Pick accuracy points over unrestricted v2 on 2020-2025 (60/102 vs
58/102, `probability_positive` 0.688)** (read:
`docs/best_pick_bucket_confidence.md:159-191`), but the served-chain lane
measured the same restriction at **-4.00 points, `probability_positive` 0.24**
on 2023-2025 alone (read: `docs/served_refresh_card.md`, lane's own summary) --
so the whole promoted gain sits in 2020-2022. The pool scores the Best Pick
separately, so this ranking is worth a careful look.

**Does the restriction hold up across eras, and is there a better nominator
than either the restricted or the unrestricted rule?**

This lane changes a **RANKING**, never a side. Every game's forced pick stays
exactly as published; only which single game carries the star can move. No
threshold flip is proposed on any side, and nothing here is wired.

## Predeclaration (written before any number in "Results" was computed)

### Binding closing-grounds taxonomy, verbatim

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (a) refuted mechanism -- a RESOLVED wrong sign (whole interval on
the wrong side of zero) or zero split-half reliability; (b) bounded by a
positive control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it, report `probability_positive`, never
"contains zero". Within-week correlation is ZERO; week-blocked bootstrap. Era
readings differ in MAGNITUDE; a weaker era is never absence, but a SIGN
reversal across eras is a diagnosis to publish. Decide on expected value;
predeclared thresholds govern only claims.

### Population

The frozen 2020-2025 opener archive, no new fitting, identical to lane U's:

- **Nomination inputs** -- `artifacts/ridge_alpha_promotion/20260818T221459Z/`
  (`opener_paired.parquet`, `opener_baseline.parquet`): 1,537 games, 107 REG
  weeks, seasons 2020-2025. Supplies `candidate_dist` (v2's own ranking score)
  and the frozen Tuesday opener line `tue_open_home_spread`.
- **Tuesday cross-book dispersion** --
  `artifacts/odds_microstructure/20260818T225430Z/spread_novig_tue_open.parquet`'s
  `spread_std`, fed through production's own `dispersion_pool_from_frame` per
  week, cross-checked row-for-row against the eval-script pool; the run aborts
  on any disagreement.
- **Grades and the calibrated display** --
  `artifacts/opener_evaluation/20260909T183120Z/per_game.parquet`, active model
  `c657058903f3232b`, same 1,537 games / 107 weeks.

### Two grades, both reported; PRIMARY named in advance

Lane U named `correct_at_open` / `pick_home_at_open` its "served" grade. Read
in code, those two columns are the **sign rule** on the un-offset residual
(`src/nfl_ats/clv.py:2234-2236`, `result["pick_home_at_open"] =
result["residual_at_open"].gt(0.0)`), while `pick_home_at_open_probability_rule`
is `home_cover_probability_at_open >= 0.5` on the offset-applied probability
(`src/nfl_ats/clv.py:2240-2246`), and `docs/opener_evaluation.md:165-210` states
plainly that **production has always played the probability rule**. The two
disagree on **231 of 1,537** games (measured, see Results). This lane therefore
names its primary differently from lane U, on purpose, and reports both so the
two lanes reconcile:

- **PRIMARY -- PRODUCTION grade.** `correct_at_open_probability_rule` /
  `pick_home_at_open_probability_rule`. The rule every published pick has used,
  offset included, and the same stream
  `nfl_ats.displayed_confidence.archive_display_stream` orients the card's
  displayed score on -- so N3/N4 are internally consistent by construction.
- **SECONDARY -- SIGN-RULE grade.** `correct_at_open` / `pick_home_at_open`,
  the columns lane U named PRIMARY, kept so N0/N1 reconcile with
  `docs/best_pick_bucket_confidence.md`'s 60/102 vs 58/102.

N2's walk-forward bucket reliability is built from whichever stream is being
graded, so that arm never mixes them. N3/N4's ranking score is production's own
calibrated display, which exists only on the probability-rule stream
(`archive_display_stream` hard-codes those columns), so the SAME score ranks
both grades -- stated here rather than silently re-derived, and one more reason
the production grade is named primary.

### Spread buckets

`|tue_open_home_spread|` in **0-6.5**, **exactly 7**, **7.5-10**, **10.5+** --
the four buckets `docs/spread_hole_diagnosis.md` reports realised accuracy on
and `nfl_ats.displayed_confidence.DISPLAY_BUCKETS` serves, used unchanged so no
boundary is mined in this lane.

### Eras

**2020-21 / 2022-23 / 2024-25**, three two-season blocks, declared before any
number was computed. Reported for **every** arm as a magnitude, never as
presence/absence; a weaker era is never read as no effect. The
**sign-agreement count** (how many of the three eras carry the same sign as the
full-window estimate) is reported for every arm, because a sign reversal across
eras is a diagnosis to publish.

### Arms (one nomination per week, all graded at the frozen Tuesday opener)

Every arm's nominee comes from production's own
`nfl_ats.best_pick_nomination.select_nominee` -- ranking by a score, ties broken
by lower `spread_std` then ascending `game_id`. Only the score and the eligible
pool differ. **No arm is added after scoring begins.**

- **N0 -- served small-spread v2 (the incumbent).** `select_nominee` on v2's
  below-median-`spread_std` pool restricted to `|tue_open_home_spread| <= 6.5`;
  falls back to the unrestricted v2 pool when no eligible game exists that week.
  This is `nominate_v2_small_spread` at its served `SERVED_SPREAD_THRESHOLD`
  7.0, and lane U's B2.
- **N1 -- unrestricted v2.** `select_nominee` on v2's pool, unrestricted. This
  is `nominate_v2` and lane U's B0, and it is the registered paired OFF arm
  (`CHALLENGER_ID = best_pick_nomination_v2`).
- **N2 -- most-reliable-bucket v2.** Each week, the shrunk realised accuracy of
  the graded stream in each of the four buckets over **all prior completed
  games** (strictly earlier `(season, week)`, expanding from 2020), shrunk with
  **20 pseudo-observations toward 0.5**: `(prior wins + 20*0.5)/(prior n + 20)`
  -- the same count and form MOD-18 C3 and lane U's B1 use
  (`docs/spread_regime_program.md:61-71`). The single bucket with the highest
  shrunk reliability is the week's eligible bucket; v2's pool is restricted to
  games in it, falling back to the unrestricted v2 pool when that leaves the
  week empty. Ties in bucket reliability break in the declared bucket order
  (`0-6.5`, `7`, `7.5-10`, `10.5+`), so the first week of 2020 -- where every
  bucket sits at exactly 0.5 with no history -- selects `0-6.5`. This is the
  data-chosen cutoff N0 fixes at 6.5 by hand.
- **N3 -- highest calibrated displayed score, inside v2's pool.** The score is
  `nfl_ats.displayed_confidence.walk_forward_displayed_confidence` on
  `archive_display_stream` of the same opener archive -- the bucket-by-band
  calibrated probability now printed on the card, fitted only on strictly
  earlier weeks. Ranked descending inside v2's below-median-`spread_std` pool,
  ties through production's own `select_nominee`.
- **N4 -- highest calibrated displayed score, no dispersion pool.** N3 over the
  whole week's card instead of v2's eligibility pool.
- **Positive control -- perfect-foresight nomination within the pool.** Among
  v2's eligible pool, any game whose graded pick is correct (`select_nominee`
  order among the correct ones); else N1's nominee. The instrument's ceiling on
  this exact population.

### Metric

Best-Pick correctness per week (one pick per week, 0/1) at the frozen Tuesday
opener line, over 2020-2025 weeks with a v2 nominee (~102 after pushes),
**paired against N0 and, separately, against N1**, week-blocked bootstrap
(`nfl_ats.clv.week_blocked_bootstrap`, blocks `(season, week)`), **20,000
samples, seed 20260821**, in accuracy points, reported as
`probability_positive` plus the interval. Raw win counts and the number of
weeks the nominee differs are reported alongside because one pick a week makes
the interval necessarily wide. A push on either nominee is NaN and drops that
week from the paired cell. Every cell is computed on the full window and on
each of the three eras.

### Registry cells

Every cell is recorded, one command at a time, argv in the artifact's
`record_commands.json`, family `best_pick_eras`:

- `best_pick_eras_<arm>_vs_n0_<window>` for arm in `n1_v2_unrestricted`,
  `n2_reliable_bucket`, `n3_calibrated_pool`, `n4_calibrated_open`,
  `control_perfect_foresight`;
- `best_pick_eras_<arm>_vs_n1_<window>` for arm in `n0_small_spread`,
  `n2_reliable_bucket`, `n3_calibrated_pool`, `n4_calibrated_open`,
  `control_perfect_foresight`;

with `<window>` in `2020_2025`, `2020_2021`, `2022_2023`, `2024_2025` -- 40
cells on the PRIMARY grade. The three era cells are a correlated decomposition
of their own full-window cell and say so in their notes, so the pooler's
`overlap_warnings` can discount them.

### Declared limitations, stated before the run

1. **Look reuse.** This is the sixth reuse of the ~107-week opener population
   for the Best Pick family (ridge_alpha promotion look, odds-microstructure
   battery, the 2026-08-18 ranker screen that SELECTED v2, the 2026-08-19 v3
   audit, the 2026-09-09 composed-rule scoring, the 2026-09-09 bucket-confidence
   lane that SELECTED N0). Every number here carries that compounding discount.
   N0 was chosen on this window; comparing it here is a re-read of its own
   selection window, and any N0-favouring number is inflated by that.
2. **N2 is a walk-forward version of the same re-weighting N0 hard-codes.** Its
   reliability is an expanding mean of the same graded stream it is scored on,
   one week ahead. It cannot leak an outcome, but it is not independent
   evidence about buckets.
3. **N3/N4 rank by a number fitted on this archive.** The calibrated display is
   walk-forward per week, but its cell structure (four buckets by three bands)
   was chosen on this same window in MOD-18 lane AH.
4. **Eras are ~34-36 weeks each.** An era cell is a ~35-observation instrument;
   its interval will be roughly twice the full window's. Era readings are
   magnitudes, never presence/absence.
5. **One pick a week is a ~102-observation instrument even on the full window.**
   A 2-3 accuracy-point nomination edge is far below what it can resolve, which
   is exactly why the positive control is run.

## Results

All measured 2026-09-10; commands at the bottom. Every arm's nominee comes from
production's own `nfl_ats.best_pick_nomination.select_nominee`; nothing was
refit. Artifact:
`artifacts/best_pick_eras/20260910T033510Z/` (`summary.json`,
`weekly_production.csv`, `weekly_sign_rule.csv`, `record_commands.json`).

Population: 1,537 games, 107 REG weeks, 2020-2025, 34 pushes on each grade, the
dispersion pool falling back to the full week in 10 of 107 (8 missing data, 2
empty filter), the production pool agreeing with the eval pool on every game.
The production and sign-rule picks disagree on **231 of 1,537** games; the
production rule scores 54.56% per game against the sign rule's 52.96%.

### The replay check, and what it exposes

On the SECONDARY sign-rule grade -- the columns
`docs/best_pick_bucket_confidence.md` used -- this lane reproduces that lane
exactly: N0 vs N1 is **60/102 vs 58/102, +1.96 accuracy points, 95%
[-5.88, +9.80], `probability_positive` 0.6885, 26 weeks the nominee differs**,
and the control is **+41.75, [+32.04, +51.46]**. Same numbers to the decimal.

On the PRIMARY production grade -- `correct_at_open_probability_rule`, the rule
every published pick has actually used -- the same comparison on the same 102
weeks is **57/102 vs 57/102, +0.00 points, 95% [-6.86, +6.86],
`probability_positive` 0.5014**, still 26 weeks with a different nominee, split
7 wins and 7 losses. **The small-spread restriction's promoted gain is a
sign-rule reading. On its own decision grade it is a dead heat.** That is not a
refutation and closes nothing -- a dead heat is exactly `unresolved_below_power`
-- but it removes the evidence the promotion rested on.

### Arms, PRIMARY production grade, full window

| Arm | Weeks won | Accuracy | vs N0 (pts) | 95% week | P+ vs N0 | vs N1 (pts) | P+ vs N1 |
|---|---:|---:|---:|---|---:|---:|---:|
| **N0 served small-spread v2** | 57/102 | 55.88% | — | — | — | +0.00 | 0.5014 |
| N1 unrestricted v2 | 57/103 | 55.34% | +0.00 | [−6.86, +6.86] | 0.4986 | — | — |
| N2 most-reliable bucket | 58/101 | 57.43% | **+0.99** | [+0.00, +2.97] | **0.8176** | +0.99 | 0.6076 |
| **N3 calibrated score in v2's pool** | **61/104** | **58.65%** | **+3.96** | [−6.93, +14.85] | **0.7665** | **+2.94** | **0.7065** |
| N4 calibrated score, no pool | 59/106 | 55.66% | +0.98 | [−10.78, +12.75] | 0.5670 | +0.97 | 0.5686 |
| Positive control | 106/107 | 99.07% | +43.14 | [+33.33, +52.94] | 1.0000 | +43.69 | 1.0000 |

Paired cells use only the weeks both nominees resolved, so an arm's paired
record differs from its own full record: N3 is 60/101 against N0's 56/101 and
60/102 against N1's 57/102.

### Every arm per era, as magnitude

Effects in accuracy points against **N0**, PRIMARY grade:

| Arm | 2020-21 (35 wk) | 2022-23 (36 wk) | 2024-25 (36 wk) | Full | Eras positive / zero / negative | Sign reversal |
|---|---:|---:|---:|---:|---|---|
| N1 unrestricted v2 | +0.00 | −5.88 | +6.06 | +0.00 | 1 / 1 / 1 | **yes** |
| N2 most-reliable bucket | +2.94 | +0.00 | +0.00 | +0.99 | 1 / 2 / 0 | no |
| **N3 calibrated in pool** | **+2.86** | **+8.82** | **+0.00** | **+3.96** | **2 / 1 / 0** | **no** |
| N4 calibrated, no pool | +0.00 | +23.53 | −21.21 | +0.98 | 1 / 1 / 1 | **yes** |
| Positive control | +48.57 | +47.06 | +33.33 | +43.14 | 3 / 0 / 0 | no |

Against **N1**, PRIMARY grade:

| Arm | 2020-21 | 2022-23 | 2024-25 | Full | Sign reversal |
|---|---:|---:|---:|---:|---|
| N0 served small-spread | +0.00 | +5.88 | −6.06 | +0.00 | **yes** |
| N2 most-reliable bucket | +2.94 | +5.88 | −6.06 | +0.99 | **yes** |
| N3 calibrated in pool | +2.86 | +14.71 | −9.09 | +2.94 | **yes** |
| N4 calibrated, no pool | +0.00 | +29.41 | −26.47 | +0.97 | **yes** |
| Positive control | +48.57 | +52.94 | +29.41 | +43.69 | no |

**No arm is positive in all three eras.** Only the positive control is, and it
is the oracle. N3 vs N0 is the one candidate comparison that is **never
negative in any era** (+2.86 / +8.82 / +0.00). Every other candidate reverses
sign somewhere, including the served N0 against the rule it replaced.

### The sign reversal worth publishing

N4 -- the calibrated score with **no** dispersion pool -- swings from
**+23.53 points in 2022-23 (`probability_positive` 0.9944, 25/34 vs 17/34)** to
**−21.21 in 2024-25 (`probability_positive` 0.0276, 15/33 vs 22/33)**. Per
season it runs 58.8 / 44.4 / 66.7 / 77.8 / 47.1 / 38.9 percent. N3, the same
score inside v2's below-median cross-book dispersion pool, does not do this:
58.8 / 50.0 / 61.1 / 55.6 / 58.8 / 68.8. The reading is a magnitude statement
about the two eras, and the mechanism it points at is that **the dispersion
pool is doing real work in the recent era** -- the calibrated score alone will
walk into a game the books disagree about, and since 2024 that has cost. It is
a diagnosis, not a closure, and it is why N4 is not the recommendation despite
sharing N3's score.

### What N2 says about the 6.5 cutoff

N2 lets the data choose the eligible bucket every week instead of fixing it at
6.5. It chooses **`0-6.5` in 102 of 107 weeks** (`7` in 4, `7.5-10` in 1), so
the hand-set cutoff is the walk-forward-best bucket almost always, and N2's
nominee differs from N0's in only **4 of 101 paired weeks** -- +0.99 points,
95% [+0.00, +2.97], `probability_positive` 0.8176. The narrow interval is a
consequence of the arms being nearly the same rule, not of precision about
buckets: a four-week difference cannot resolve anything. Read it as
confirmation that 6.5 is the right boundary if a boundary is used at all, not
as a second candidate.

### Nominee mix, PRIMARY grade

| Arm | 0-6.5 | exactly 7 | 7.5-10 | 10.5+ |
|---|---|---|---|---|
| N0 | 102 wk, 55.88% | — | — | — |
| N1 | 76 wk, 57.89% | 9 wk, 44.44% | 12 wk, 58.33% | 6 wk, 33.33% |
| N2 | 97 wk, 56.70% | 3 wk, 66.67% | 1 wk, 100% | — |
| N3 | 84 wk, 59.52% | 11 wk, 45.45% | 6 wk, 83.33% | 3 wk, 33.33% |
| N4 | 73 wk, 60.27% | 16 wk, 50.00% | 14 wk, 42.86% | 3 wk, 33.33% |

N3 does **not** avoid big spreads the way N0 does -- it sends 20 of 104
nominations into 7-or-more -- and still beats N0. Its gain does not come from
the spread screen; it comes from ranking by the number that already knows how
the model has done in that cell, rather than by raw distance from 0.5.

### Decision

**On the grade the pool actually settles, N3 -- the highest calibrated
displayed score inside v2's low-disagreement pool -- has the higher expected
Best-Pick accuracy than either served rule**: +3.96 points over the served N0
(`probability_positive` 0.7665) and +2.94 over the registered OFF arm N1
(`probability_positive` 0.7065), 61/104 raw against 57/102 and 57/103. On a
forced weekly nomination those are 77/23 and 71/29 bets, and per `AGENTS.md` a
promotion bar governs claims, not plays. **No arm is positive in all three
eras**, and N3 is the only candidate that is never negative in one.

Second, separately: the evidence that promoted N0 over N1 is grade-specific.
On the sign-rule columns it is +1.96 (`probability_positive` 0.6885); on the
production grade it is +0.00 (`probability_positive` 0.5014). Both are
`unresolved_below_power`; neither closes.

**Nothing here is wired.** This lane recorded 40 cells and changed no code path.

### What serving N3 would take (not done)

1. **A new nominator.** `nfl_ats.best_pick_nomination` needs a
   `nominate_v2_displayed_confidence` beside `nominate_v2_small_spread`,
   sharing `_nominate`'s orchestration but ranking the eligible pool by the
   calibrated displayed probability instead of `candidate_dist`. The score is
   already produced in production by
   `nfl_ats.displayed_confidence.fit_production_displayed_confidence(...).calibrate(stated, spread_line)`
   on the card's own pick-oriented `home_cover_probability` -- the number
   already printed beside each pick -- so this is a new ranking key over
   existing production values, not a new model. The eligibility pool
   (`week_dispersion_pool`), the tie-break (`select_nominee`: lower
   `spread_std`, then ascending `game_id`) and the REG-only gate stay
   byte-for-byte.
2. **The published call site.** `nfl_ats.publishing._publication_context` passes
   `nominate_v2_fn=nominate_v2_small_spread` into
   `nfl_ats.card_view.resolve_card_view` (read: `src/nfl_ats/publishing.py:14`,
   `:189`, `:198`) and stars `view.nomination.active_game_id`.
   `card_view.resolve_card_view` / `compute_v2_nomination` /
   `card_view_for_predictions` each default that same parameter (read:
   `src/nfl_ats/card_view.py:261`, `:307`, `:373`). Those four defaults plus the
   one import are the whole served surface.
3. **The OFF-arm registration.** The played star is recorded through the active
   model's own `is_best_pick` flag on `artifacts/clv_ledger/decisions.parquet`
   (`nfl_ats.clv.record_paper_decisions`); the Tuesday nomination side-ledger is
   `record_nomination_challenger_decisions` under
   `CHALLENGER_ID = "best_pick_nomination_v2"`, registered in
   `artifacts/prospective/challengers.json` and pinned to the active model's
   configuration fingerprint. Serving N3 means registering a new active
   challenger for it and keeping `best_pick_nomination_v2` recording as the
   unrestricted OFF arm, so the season scores the served rule against both
   predecessors weekly.
4. **The card's star sentence.** `NOMINATION_V2_METHOD_SENTENCE` and the
   small-spread clauses in `best_pick_nomination.py`, plus `_best_pick_note` in
   `publishing.py`, would have to name the new rule in reader words -- and per
   `AGENTS.md`'s board rule, without the words "calibrated", "bucket" or any
   column name.

### Nothing closes, and why

All 40 cells are recorded `unresolved_below_power`
(`registry/weak_signals.json`, family `best_pick_eras`, names
`best_pick_eras_<arm>_vs_<n0|n1>_<window>`; argv in the artifact's
`record_commands.json`). Neither terminal ground is available:

- **`wrong_sign_resolved`** requires the WHOLE interval on the wrong side of
  zero. Two cells come close -- N4 vs N1 in 2024-25 at [−47.06, −5.88] and N4
  vs N0 in 2024-25 at [−42.42, +0.00] -- and the first is technically wholly
  below zero on a 34-week era slice whose sign is opposite in the adjacent era
  (+29.41). A sign that flips between eras is by construction not a RESOLVED
  wrong sign; per this repository's own era rule it is a magnitude difference
  and a diagnosis to publish. It is recorded unresolved.
- **`bounded_by_control`** requires an instrument PROVEN able to see an effect
  this size. The positive control resolves at **+43.14 points**,
  `probability_positive` 1.000, so this instrument demonstrably sees a
  ~43-point renomination effect over 102 weeks. It was never shown able to see
  a 1-4 point one; at ~102 weekly observations the interval is roughly ±7
  points wide and ±15 on an era. It bounds nothing at the scale N0, N2 and N3
  live at.

### Stated against the decision

- **Sixth reuse.** N0 was selected on this same window one day earlier, and the
  bucket table that motivated it was measured on it too. Any N0-favouring
  reading is inflated; so, less obviously, is N3's, because the calibrated
  score's cell structure was fitted on these weeks in MOD-18 lane AH.
- **N3 re-nominates 62 of 101 weeks against N0** -- it is a substantially
  different rule, not a tweak, and the outcome differs in 30 of those 62 (17 to
  N3, 13 to N0). Two extra correct weeks a season is what +3.96 points means.
- **N3's own recent era is a dead heat** (+0.00 in 2024-25 against N0), and its
  advantage against N1 turns negative there (−9.09). The full-window number is
  carried by 2022-23.
- **The grade change is doing work.** Reporting on the production rule rather
  than the sign rule is a defensible correction, but it is a correction made in
  this lane, and it is the reason N0's promotion evidence evaporates. It should
  be checked by whoever owns the opener-evaluation columns before anything is
  re-served on it.

## Commands run

```
uv run --no-sync python scripts/best_pick_eras_eval.py \
  --repo F:/Repos/nfl_py3 \
  --out-dir F:/Repos/nfl_py3/artifacts/best_pick_eras/20260910T033510Z

uv run --no-sync python scripts/best_pick_eras_eval.py --record \
  --artifact F:/Repos/nfl_py3/artifacts/best_pick_eras/20260910T033510Z/summary.json
  # 40 cells, family best_pick_eras, exact argv in the artifact's record_commands.json
```

