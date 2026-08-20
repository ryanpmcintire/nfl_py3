# SBR-derived opener evaluation: era-stratified grading of the production model

Predeclared 2026-08-19 (US), before `scripts/sbr_era_opener_eval.py` computes
any era-level accuracy number. Provenance tags used throughout: **measured**
(run this session, command/path given), **read** (file opened this session),
**reported** (another doc's claim, not reverified here), **inferred**
(reasoning, not evidence).

## Binding closing-grounds taxonomy (pasted verbatim, per AGENTS.md/CLAUDE.md)

> An interval or CI that contains zero is NEVER grounds to reject, fail, or
> close an experiment. At this evaluator's ~2-point resolution, "contains
> zero" is the EXPECTED outcome for a real small signal. Only two grounds
> ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
> (whole interval on the wrong side of zero) or zero split-half reliability;
> (2) bounded by a positive control proven able to detect an effect that
> size. Everything else is `unresolved_below_power`: record it with
> `nfl-ats weak-signals record`, report `probability_positive`, never the
> binary "contains zero". The registry code hard-rejects inadmissible
> closures; if a record command errors, the verdict is wrong, not the
> validator.

## 0. Task-1 status: the archive extension is already done, not new work here

`docs/sbr_odds_archive.md` (**read** this session) ingested all 15 populated
SBR season pages (2007-08 through 2021-22) in one pass on 2026-08-19 -- not
just the scout's headline "13 net-new" 2007-08..2019-20 pages, but those plus
2020-21 and 2021-22 as well (needed for the OPENER validation and for any
2020-2021 comparison arm). **Measured this session**: a fresh fetch of the
index page (`https://www.sportsbookreviewsonline.com/scoresoddsarchives/nfl/nfloddsarchives.htm`)
still lists exactly the same 15 slugs, and a fresh fetch of
`nfl-odds-2022-23` still returns HTTP 200 with 0 `<tr>` rows -- the archive
has not changed since the earlier ingestion this session, so there is
nothing net-new to scrape. **Measured this session**, re-running
`scripts/ingest_sbr_odds.py --skip-fetch --validate` against the existing
raw snapshot (`data/raw/sbr_odds/20260819T192226Z/`, no re-fetch): the CLOSE,
OPENER, and COVERAGE diagnostics reproduce `docs/sbr_odds_archive.md`'s
already-published tables exactly (2009-2020 CLOSE mean |diff| 0.19-0.38 pts;
2021 CLOSE mean |diff| 0.595 pts, the measured outlier already documented;
OPENER check 491 matched games, mean |diff| 1.357, r reported separately at
0.949; COVERAGE 100% match rate 2009-2021, 0% for 2007-2008 which predates
`game_features.parquet`'s own floor). This document's Task-1 contribution is
therefore **verification that the extension is complete and stable**, not a
new scrape -- reported as measured zero-change, not assumed.

## 1. What this document is

An **era-stratified** re-slice of the frozen production model's opener-graded
accuracy, using SBR's Open consistently as the settlement/prediction line
across the full range SBR's archive supports for the model's own warm-up
floor (seasons 2011-2021 -- see Section 3). This directly answers the task's
primary question: **at SBR-derived openers, does the model's edge look
different by era, and if so where does it live?** Per the owner's standing
rule (`docs/era_magnitude_profile.md`, **read** this session): era variation
is expected to be a change in **magnitude**, not presence/absence, so every
era cell below is reported as a number and an interval, never collapsed to
a binary "worked / didn't work."

## 2. Blindness disclosure (label how you know it)

This predeclaration is **not** blind for the whole population, and that is
stated plainly rather than glossed over:

- **Not blind, 2011-2019**: `docs/proxy_opener_replication.md` (**read** this
  session, required background per this task's own instructions) already
  reports the full per-season sign/production accuracy for every season
  2011-2019 (e.g. 2011: 45.20%, 2013: 52.46%, 2018: 47.13%, production rule).
  Those numbers were seen before this document was written. What is
  predeclared here for that range is therefore the **aggregation rule**
  (which era boundaries, which bootstrap spec, which registry names) --
  not a blind look at the signs themselves. This is disclosed rather than
  claimed as a blind test, per AGENTS.md's "label how you know it" rule.
- **Blind, 2020-2021**: no per-season or per-era number has been computed or
  seen for the FULL (non-restricted) SBR-Open-graded 2020-2021 population.
  `docs/proxy_opener_replication.md`'s calibration arm reports a dual grade
  on a **466-game subset restricted to games that also have a true `tue_open`
  quote** (of 528 SBR-matched REG games) -- a different, smaller population
  built for a different purpose (measuring the proxy-vs-true discount on a
  paired set). This document's Era C population is the full ~528-game SBR-
  matched set, graded once, independently of whether a true-opener quote also
  exists. That number is genuinely unseen before this script runs.
- **Overlap disclosure vs `proxy_opener_production_rule_2009_2019`**
  (registry, **read** this session): Eras A (2011-2014) and B (2015-2019)
  pooled together are, by construction, the **exact same population and
  scoring pass** as that already-recorded registry entry's 2011-2019 pooled
  figure (2,304 games) -- same script logic (`build_population` +
  `sbr_proxy_pick_evaluation`, reused by import, not reimplemented), same
  model config, same games. This document does not re-measure that finding;
  it **re-slices** it into two era buckets instead of one decade-pool, which
  is new information (per-era magnitude) even though the underlying games
  and grades are not new. Era C (2020-2021, full SBR population) is the only
  genuinely new SBR-graded population in this document.

## 3. Population, model, and pick rule (declared exactly)

- **Model artifact**: `artifacts/active_ats_model.json` (**read** this
  session): `feature_profile="weak_stack"`, `regressor="ridge"`,
  `ridge_alpha=10.0`, `method="market_residual"`. Single frozen production
  arm -- no candidate, no tuning, no variant sweep.
- **Feature table**: `data/processed/game_features_weak_stack.parquet`,
  regular season only (`game_type == "REG"`, via
  `nfl_ats.modeling.regular_season_rows`).
- **Games**: every REG game 2009-2021 with a non-null SBR `open_home_spread`
  (inner join on `game_id`, `data/processed/sbr_odds.parquet`), scored
  subject to the same 500-game (`DEFAULT_MIN_TRAIN_GAMES`) walk-forward
  warm-up floor `nfl_ats.clv.opener_pick_evaluation` and
  `scripts/proxy_opener_replication.py` already use. This reproduces the
  already-documented warm-up consequence exactly: 2009-2010 fail the floor
  entirely (max cumulative training games before 2010's last cutoff is 496),
  2011 clears it at week 1 (512 cumulative games) -- so the scorable
  population starts at **2011**, not 2009, identically to
  `docs/proxy_opener_replication.md`.
- **Settlement line**: `spread_line` is overridden to SBR's
  `open_home_spread` at SCORING time only; training rows keep their native
  close-era `spread_line` (the same "only the settling line changes"
  convention `opener_pick_evaluation` and `proxy_opener_replication.py` both
  use). `margin_vs_open_proxy = result - open_home_spread`; pushes excluded
  via `nfl_ats.clv.pick_correct`'s existing convention.
- **Pick rules, both reported, from the same model output**:
  - **Probability rule (primary)** -- what `pool.py`/`backtest.py` actually
    play: `home_cover_probability_at_open_proxy >= 0.5`.
  - **Sign rule (secondary, historical protocol comparability)**:
    `residual_at_open_proxy > 0`.
- **Implementation**: `scripts/sbr_era_opener_eval.py`, which imports
  `build_population` and `sbr_proxy_pick_evaluation` from
  `scripts/proxy_opener_replication.py` **by reference, unmodified** (not
  reimplemented) so that Eras A/B are provably the same scored games as the
  already-recorded registry entry, and runs ONE weekly-refit scoring pass
  over 2009-2021 (population includes 2009-2010 rows; the walk-forward loop
  itself is what drops them via the warm-up floor, exactly as the original
  script does for 2009-2019).

## 4. Era scheme (declared before any era-level number is computed)

Three eras, chosen to reuse `docs/era_magnitude_profile.md`'s own Stage-1
fixed-era convention (`2009-2014` / `2015-2019` / `2020-2025`) as closely as
this population's two hard boundaries allow -- the warm-up floor (2011) on
the early end and SBR's archive ceiling (2021-22 is the last populated
season) on the late end -- rather than inventing a fresh scheme for this
document:

| Era | Seasons | n seasons | Why this boundary |
|---|---|---|---|
| A | 2011-2014 | 4 | Earliest era the warm-up floor allows; left edge of era_magnitude_profile's `2009-2014` bucket, clipped at 2011 |
| B | 2015-2019 | 5 | Identical to era_magnitude_profile's own `2015-2019` bucket, unchanged |
| C | 2020-2021 | 2 | Right edge of era_magnitude_profile's `2020-2025` bucket, clipped at 2021 -- SBR's archive ceiling. This is also the only era with independent true-Tuesday-opener ground truth to compare against |

Plus a **pooled 2011-2021 read** (all three eras combined, 11 seasons) as
the headline "where does the SBR-graded edge live" number.

Every era is reported with its own week-blocked interval and
`probability_positive` regardless of sign or width -- a weaker or
zero-crossing era reading is a magnitude reading, not evidence of absence,
per the binding rule above and per `docs/era_magnitude_profile.md`'s
owner-approved framing.

## 5. Bootstrap specification

- **Primary**: week-blocked bootstrap (`nfl_ats.clv.week_blocked_bootstrap`,
  `block="week"`), 20,000 samples, **seed 20260819** -- the exact seed this
  task's own brief specified. Disclosed: this differs from the project's
  older "standing" opener-bootstrap seed (20260817, used by
  `scripts/proxy_opener_replication.py`, `scripts/surface_profile_opener_eval.py`,
  `scripts/era_magnitude_profile.py`); 20260819 is not a fresh arbitrary
  choice, it is the seed given directly in this task's instructions (today's
  date, matching the pattern of the standing seed being an earlier date).
  Both seeds are legitimate fixed choices; this document uses the one the
  task specified rather than silently substituting the older one.
- **Secondary**: season-blocked bootstrap, same samples/seed, reported
  alongside every week-blocked number. Coverage caveat carried forward
  explicitly: `src/nfl_ats/estimation_variance.py`'s own measured finding
  (`MIN_BLOCKS_FOR_INTERVAL = 10`, **read** this session) is that
  season-blocked coverage at small block counts falls well short of nominal
  (2-block coverage measured at 0.466 vs nominal 0.95 elsewhere in this
  repo, `docs/estimation_variance.md`, **reported**, not rerun here). Every
  era in this document has 2, 4, or 5 season-blocks -- all under 10 -- so
  season-blocked intervals here are reported for completeness, not treated
  as the primary read; week-blocked is primary throughout, matching
  `docs/proxy_opener_replication.md`'s own convention.
- Within-week correlation is treated as zero by project mandate (no
  estimation, no padding) -- not separately re-derived here.

## 6. Reporting plan (no accept/reject gate -- this is a measurement)

For each era (A, B, C) and for the pooled 2011-2021 read:

- games, seasons present, week count
- absolute accuracy, both rules
- week-blocked 95% interval (pts above 50) and `probability_positive`, both
  rules
- season-blocked 95% interval and `probability_positive` (secondary, with
  the coverage caveat above), both rules

Plus, stated plainly once computed:

- A cross-check that Era A + Era B pooled reproduces the already-recorded
  `proxy_opener_production_rule_2009_2019` registry entry's 2011-2019 pooled
  figure (same games, same config -- this is a re-slice, not a re-measurement,
  and the cross-check is how that claim is verified rather than assumed).
- A comparison of Era C (SBR-Open grade, full 2020-2021 population) against
  (a) the true-Tuesday-opener grade on the identical two seasons
  (`docs/proxy_opener_replication.md`'s calibration arm, 53.73% production
  rule on the 466-game paired subset, **reported** from that document) and
  (b) `era_magnitude_profile.md`'s own `2020-2025` true-opener era cell
  (+3.360 [+0.605, +6.040] pts, P+0.991, **reported** from that document) --
  stating what a genuinely SBR-graded (rather than true-opener-graded) read
  of the same two seasons looks like, side by side with both.
- Where the model's opener edge appears concentrated (or not) across eras,
  read from the numbers -- a magnitude statement, never a presence/absence
  one, and never a claim that a zero-crossing era interval means "no edge
  there."

## 7. Registry recording plan

Four entries, `nfl-ats weak-signals record`, `league=nfl`,
`effect_units=accuracy_points`, effect = week-blocked pooled
**production-rule** accuracy above 50 (`(accuracy - 0.5) * 100`), matching
the existing registry's own convention for this metric family:

- `sbr_opener_era_2011_2014`
- `sbr_opener_era_2015_2019`
- `sbr_opener_era_2020_2021`
- `sbr_opener_pooled_2011_2021`

None of these names collides with any of the 177 entries already in
`registry/weak_signals.json` (**measured** this session, checked directly).
Classification for every entry is `unresolved_below_power` UNLESS the whole
week-blocked interval sits below zero, which is the only admissible
`wrong_sign_resolved` condition (`--closing-ground wrong_sign_resolved`); no
positive control is run in this document, so `bounded_by_control` is not
available to any entry here. Every numeric CLI argument is read
programmatically from this run's artifact JSON by
`scripts/sbr_era_opener_record.py` -- no hand-typed numbers, matching
`scripts/proxy_opener_replication_record.py`'s and
`scripts/era_magnitude_profile_record.py`'s precedent. The registry is read
back after recording to verify each write.

## Files

- `scripts/sbr_era_opener_eval.py` -- implementation (imports
  `scripts/proxy_opener_replication.py`'s population/scoring functions
  unmodified; era-slices and bootstraps the single resulting scored frame).
- `scripts/sbr_era_opener_record.py` -- reads the artifact JSON and calls
  `nfl-ats weak-signals record` four times.
- `artifacts/sbr_era_opener_eval/<run-id>/` -- output artifact (summary JSON
  plus the full scored per-game parquet).

---

## Results

**Measured**, `scripts/sbr_era_opener_eval.py`, artifact
`artifacts/sbr_era_opener_eval/20260819T233013Z/summary.json`, run
2026-08-19. Config: `weak_stack`/ridge/alpha=10.0/`market_residual` (matches
`artifacts/active_ats_model.json` exactly), `min_train_games=500`, 20,000
bootstrap samples, seed 20260819.

Population confirms the predeclaration exactly: 3,344 of 3,344 REG games
2009-2021 have an SBR Open (100% match, 5 flagged `open_ambiguous`, kept).
Warm-up reproduces the predeclared floor exactly: **2009 and 2010 score zero
weeks** (34 weeks skipped, all logged as `below_min_train_games`; max
cumulative training games before 2010's last cutoff is 496), 2011-2021 score
all weeks. Scored population: **2,832 games, 188 weeks, 11 seasons
(2011-2021)**.

### Cross-check: does Era A + B reproduce the already-recorded registry entry?

**Yes.** Era A + Era B combined (2011-2019, n=2,304, 153 weeks) is the exact
same population and scored games as the already-recorded
`proxy_opener_production_rule_2009_2019` entry's 2011-2019 pooled figure.
Production-rule absolute accuracy here: **50.38%**, week-blocked
+0.381 pts [-1.584, +2.368] pts, P+ 0.6450 -- the registry's own recorded
figure (different bootstrap seed, 20260817) is 50.38% absolute, week-blocked
[-1.600, +2.370] pts, P+ 0.6468. The two reproduce each other to within
ordinary seed-to-seed bootstrap noise (~0.02 pts), confirming Eras A and B
below are a genuine re-slice of that entry, not a divergent re-run.

### Era x rule matrix

Effect in accuracy points above 50, week-blocked 95% interval (primary),
`probability_positive` (P+). Season-blocked is reported alongside but is
**not primary**: every era here has 2, 4, or 5 season-blocks, all under
`estimation_variance.MIN_BLOCKS_FOR_INTERVAL = 10`, so season-blocked
coverage is understated per that module's own established finding -- most
visibly in Era C, where only 2 season-blocks exist and the season-blocked
interval is correspondingly (and misleadingly) tight.

| Era | Seasons | Games | Rule | Absolute | Week-blocked 95% | Week-blocked P+ | Season-blocked 95% (caveat: <10 blocks) |
|---|---|---:|---|---:|---:|---:|---:|
| A | 2011-2014 | 1,024 | Production (primary) | 49.75% | [-3.035, +2.584] | 0.4137 | [-3.240, +1.911] |
| A | 2011-2014 | 1,024 | Sign (secondary) | 50.15% | [-2.977, +3.253] | 0.5232 | [-1.944, +1.911] |
| B | 2015-2019 | 1,280 | Production (primary) | 50.89% | [-1.840, +3.639] | 0.7340 | [-1.142, +2.516] |
| B | 2015-2019 | 1,280 | Sign (secondary) | 50.73% | [-2.195, +3.635] | 0.6816 | [-0.567, +2.508] |
| **C** | **2020-2021** | **528** | **Production (primary)** | **53.29%** | **[-1.351, +7.634]** | **0.9178** | [+3.012, +3.558] (n=2 blocks, near-degenerate) |
| **C** | **2020-2021** | **528** | **Sign (secondary)** | **55.23%** | **[+0.898, +9.351]** | **0.9903** | [+4.618, +5.805] (n=2 blocks, near-degenerate) |
| Pooled | 2011-2021 | 2,832 | Production (primary) | 50.93% | [-0.873, +2.754] | 0.8406 | [-0.681, +2.260] |
| Pooled | 2011-2021 | 2,832 | Sign (secondary) | 51.37% | [-0.542, +3.287] | 0.9197 | [-0.147, +2.909] |

**Only one cell excludes zero in the raw week-blocked view: Era C, sign
rule** (+5.233 pts, [+0.898, +9.351], P+ 0.9903 -- the whole interval sits
above zero). Per the binding taxonomy above, this is a strong positive lean,
**not a closing ground** either way -- this project's taxonomy has no
"resolved positive" terminal classification (only `refuted_mechanism` via a
resolved wrong sign, or `bounded_by_control`), so even this excludes-zero
cell is recorded `unresolved_below_power`, same as every other cell here.
Every other cell contains zero, which per the binding rule is the **expected**
shape for a real small signal at this resolution -- reported as a number and
a lean, never as absence.

### Where the model's SBR-graded opener edge lives, read plainly

**A clean magnitude gradient, era to era, both rules:**
Era A (2011-2014) reads flat-to-slightly-negative (production P+ 0.414,
i.e. 58.6% of week-blocked draws are negative) -> Era B (2015-2019) reads
moderately positive (P+ 0.734) -> Era C (2020-2021) reads strongly positive
(P+ 0.918 production, P+ 0.990 sign, the one excludes-zero cell in the
table). None of these individually resolves under the binding taxonomy
(Era A's interval does not sit entirely below zero either -- upper bound
+2.584 -- so `wrong_sign_resolved` does not apply there), but the **direction
of the lean across eras is monotonic and consistent with three independent
prior readings**, none of which this document re-runs, all previously
**read** this session:

1. `docs/era_magnitude_profile.md`'s free-break changepoint search on the
   production model's own opener-proxy edge (a DIFFERENT, blended
   proxy+true-opener series) independently locates its break at **2019**,
   never told this document's era boundaries.
2. That same document's season-trend OLS slope for the identical signal is
   +0.347 pts/season [-0.021, +0.708], P+ 0.968 -- leaning positive across
   the calendar, not resolved.
3. `docs/era_magnitude_profile.md`'s **2020-2025 true-opener era cell**
   (6 seasons, TRUE Tuesday opener throughout, not SBR) is **+3.360
   [+0.605, +6.040] pts, P+ 0.991 -- fully resolved positive**. This
   document's Era C (SBR-Open grade, 2020-2021 only, 528 games) lands at
   **+3.295 pts production-rule** -- a near-identical point estimate reached
   by a completely different instrument (SBR's Open, not the purchased
   Tuesday-opener archive) on a 2-season subset of that same 6-season range,
   with a much wider interval ([-1.351, +7.634]) that does not resolve,
   purely because n is roughly a third the size (528 games/35 weeks vs. the
   full 6-season population). Same direction, same rough magnitude,
   different power -- not the same finding restated, but a consistent one.

**Comparison to the true-opener grade on the identical two seasons**
(`docs/proxy_opener_replication.md`'s calibration arm, **read** this
session, restricted to the 466-game subset with BOTH an SBR Open and a true
`tue_open` quote): true-opener production-rule absolute accuracy there is
53.73%; SBR-Open production-rule on that same restricted subset is 52.86%.
This document's Era C uses the FULL 528-game SBR-matched population (not
restricted to the paired subset) and reads **53.29%** -- between the two,
consistent with the calibration arm's own measured (but unresolved, P+
0.167) finding that SBR-Open reads modestly lower than the true opener under
the production rule, without that discount erasing the late-era edge.

**Reading, stated as a magnitude finding, not a presence/absence one**: the
model's SBR-graded opener edge over the market looks small-to-flat in the
earliest scorable era (2011-2014), moderate in the middle era (2015-2019),
and largest in the most recent era this archive can reach (2020-2021) --
consistent with, not proof of, the same 2018-2019-ish inflection three
independent readings above already pointed to. **This does not mean the
edge is "absent" in 2011-2014** -- Era A's interval does not exclude zero on
either side, so the honest statement is "not resolved, leaning flat," never
"no edge there," per the owner's binding magnitude-not-presence framing.

### Overlap disclosure, restated with numbers now in hand

Eras A and B, pooled, are a bit-for-bit re-slice of the population underlying
`proxy_opener_production_rule_2009_2019` (cross-check above). The only
genuinely new number in this document is **Era C: 528 REG games, 2020-2021,
graded at SBR's Open** -- a population never scored in full before (the
existing calibration arm restricted to a 466-game paired subset for a
different purpose). Nothing in Eras A/B changes what is already on record;
Era C adds a same-instrument, full-population read for the two seasons where
SBR's archive overlaps the purchased Tuesday-opener archive.

## Registry recording

Recorded via `scripts/sbr_era_opener_record.py` (reads
`artifacts/sbr_era_opener_eval/20260819T233013Z/summary.json`
programmatically; no hand-typed numbers). Four entries: `sbr_opener_era_2011_2014`,
`sbr_opener_era_2015_2019`, `sbr_opener_era_2020_2021`,
`sbr_opener_pooled_2011_2021`. Classification decided mechanically per entry
(whole week-blocked production-rule interval below zero -> `wrong_sign_resolved`;
otherwise `unresolved_below_power`) -- none of the four production-rule
intervals sits entirely below zero, so all four record
`unresolved_below_power`, including Era C despite its strong positive lean
(no "resolved positive" terminal state exists in this project's taxonomy).
Registry read back after write to confirm.
