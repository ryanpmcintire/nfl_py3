# Illness-designation battery: predeclaration

Written 2026-08-26, **before any cover-rate sign in this battery has been
examined**, per the task brief's Part B and `docs/new_lead_classes_20260826.md`
section 1 ("Roster-level contagion — the `illness` designation on the injury
report"). Mechanism: a team carrying several players designated `illness` in
one week has a roster-wide contagion event, not N independent injuries. It is
correlated across players, transient, and degrades players who still dress
and play. The market's injury pricing is calibrated on musculoskeletal
availability (binary: does a valuable player play or not), so an
`illness — Questionable` player who plays almost always grades near zero to
that machinery — the correlated, roster-wide, transient nature of contagion
has no representation in it at all (**inferred**, mechanism argument, not a
measurement). This mirrors the project's most reliable measured trait
(state-level flu, split-half reliability 0.981, **reported** from the parent
task brief, not re-derived here) but observed INSIDE the building rather than
inferred from the surrounding state.

This document freezes the data source, point-in-time construction, population,
cells, and predicted signs before any cell is scored. Section 6's reliability
check and this section's density counts are the only admissible pre-scoring
exceptions (computed on the PREDICTOR's own distribution, never on a
cover-rate outcome), matching `docs/fluview_battery.md`'s identical
precedent.

## Binding taxonomy (owned verbatim, per AGENTS.md / CLAUDE.md)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds
ever close a line of work: (1) refuted mechanism — a RESOLVED wrong sign
(the whole interval on the wrong side of zero) or zero split-half
reliability; (2) bounded by a positive control proven able to detect an
effect that size. Everything else is `unresolved_below_power`: record it
with `nfl-ats weak-signals record`, report `probability_positive`, never
the binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator. Every cell below is recorded regardless of sign or interval
shape.

## 1. Data source (measured, this session)

`scripts/nflverse_injuries_ingest.py` (this session, Part A) pulled the
FULL-column nflverse injuries release via `nflreadpy.load_injuries`, one
season at a time, 2009-2025 (17 seasons, 90,752 rows). Snapshot:
`data/raw/nflverse_injuries/20260826T122850Z/injuries.parquet`, manifest at
the sibling `manifest.json`. Columns include `report_primary_injury`,
`report_secondary_injury`, `practice_primary_injury`,
`practice_secondary_injury` (the reason text, dropped by the repo's existing
`nfl_ats.players.canonicalize_injuries`) and `date_modified` (per-row
revision timestamp — the as-of field this whole battery depends on).

**Illness flag (measured, this session)**: a player-week row is
illness-flagged if any of the four reason columns, lower-cased, contains the
substring `"illness"`. Measured variant strings across the full 2009-2025
table: `illness`, `illness (non-covid)`, `knee illness`, `medical illness`,
`non-football illness`, `chest/ankle/illness`, `calf/illness`,
`ribs / shoulder / illness`, plus several two-body-part combinations on the
secondary columns (`hamstring illness`, `hand, illness`, `rib, illness`,
`shoulder, illness`, `abdomen / illness`, `finger illness`, `knee, illness`).
All are illness-flagged under the substring rule.

**Point-in-time gap in `date_modified`, measured this session** (repeated
from `scripts/nflverse_injuries_ingest.py`'s own manifest note because it
governs this battery's population): 0 nulls for seasons 2011-2024; 62/4,491
(~1.4%) null for 2010; 4,804/4,821 (~99.6%) null for 2009; **6,068/6,068
(100%) null for 2025** — the 2025 release carries no `date_modified` column
at all (replaced by a `season_type` column instead, a genuine upstream
schema change). Consequence for this battery: 2009 and 2025 cannot be
resolved to any as-of illness state and are excluded from the scored
population (their checkpoint tables are empty by construction, matching
`docs/fluview_battery.md` section 3's identical treatment of its own
pre-2017 gap). 2010 is included; its 62 date-modified-null rows individually
resolve to missing exactly like fluview's `ny`-region gap, not corrected
around.

## 2. Reconciliation against the existing NFL.com scrape (measured, this
session)

`scripts/nflverse_injuries_reconcile.py`, run this session:
`artifacts/nflverse_injuries_reconcile/20260826T123409Z/agreement.json`.
Joins nflverse's FINAL-STATE row per (season, week, team, gsis_id) (latest
`date_modified`) against
`data/raw/nflcom_injuries/20260821T222602Z/injuries.parquet` (17,483 rows,
2022-2024) on normalized full name, with a fallback first-initial+last-name
match — identical two-tier rule to `scripts/ingest_nflcom_injuries.py`'s own
`agreement()`.

- Match rate: 16,783 of 17,483 nflcom rows matched (96.0%); 16,783 of 16,853
  nflverse final-state rows matched (99.6%).
- Report-status (Q/D/O) agreement: 97.66% (16,391/16,783).
- **Illness-designation agreement: 97.13% (16,301/16,783), disagreement rate
  2.87%.** Confusion: both-not-illness 15,903; nflverse-only-illness 480;
  both-illness 398; nflcom-only-illness 2.
- **Measured asymmetry, disclosed rather than smoothed over**: nflverse's
  final-state illness flag fires on 883 player-weeks vs nflcom's 409 over
  the same three seasons — nflverse flags roughly 2.2x more illness cases,
  and the disagreement is almost entirely one-directional (480 nflverse-only
  vs 2 nflcom-only). This is not explained further here (**inferred**
  candidate causes: NFL.com's single per-week page may display only one
  injury reason per player when several are reported across the week, or
  collapse a resolved mid-week illness note that nflverse's revision history
  still shows in an intermediate row even after taking the LATEST revision).
  The two sources are NOT interchangeable at the player level; nflverse is
  the richer source and is what this battery uses. Sources are broadly
  reconcilable at the aggregate level (agreement > 97%), which is the level
  this battery actually scores at (team-week counts, not player identity).

## 3. Decision cutoff and point-in-time construction (measured algorithm,
reusing existing repo machinery — not reimplemented)

**Cutoff**: `nfl_ats.pick_refresh.pick_deadline(kickoff, sunday_lock)`,
imported directly (not reimplemented) — the project's own binding per-game
pick deadline (owner-corrected 2026-08-20, `src/nfl_ats/pick_refresh.py`):
`min(that game's own kickoff, that week's Sunday 16:00 ET)`. This is the
correct cutoff for an injury-report feature, NOT the Tuesday line-freeze
`docs/fluview_battery.md` uses — FluView's ILI reading does not depend on
that week's own practice reports, but illness designations are populated
Wednesday through Friday of the SAME game week, so a Tuesday cutoff would
show near-100% missingness on every game. `sunday_pick_lock` is computed per
(season, week) from that week's own kickoffs (mode Tue..Mon cycle Sunday),
exactly as `nfl_ats.prospective` already does for its own leakage guards.

**Per-entity checkpoint + as-of resolution**: the entity is
(season, week, team, gsis_id) — one player's report history within one game
week. Unlike FluView's continuously-revising STATE series (which must carry
a value forward across many weeks via `merge_asof`), an injury report is
naturally re-anchored fresh each week per player; it never needs to carry
forward across week boundaries. The as-of resolution is therefore: filter
that entity's rows to `date_modified <= cutoff`, and take the row with the
LATEST surviving `date_modified` (the merge_asof "backward" operation
specialized to a group whose calendar span is one game week, rather than a
multi-week series) — same guarantee as fluview's `merge_asof`
(`direction="backward"`): a revision issued after the cutoff can never be
selected, and if EVERY row for that entity postdates the cutoff, the entity
resolves to unknown, not zero.

**Team-week aggregation**: `illness_count` = count of distinct `gsis_id`
resolved illness-flagged as of that team's own game's cutoff.
`active_illness_count` = the same restricted to players whose as-of
`report_status` is NOT in `{"Out", "Doubtful"}` (i.e., expected to suit up
despite the designation — including a null `report_status`, which does not
confirm they were ruled out). **Missing, not zero**: if a (season, week,
team) has ZERO report rows with `date_modified <= cutoff` at all (the report
had not been filed yet, or the season is outside the PIT-recoverable window),
the team-week resolves to missing and is excluded from both the subset and
complement of every cell — never defaulted to a 0 illness count.

## 4. 2020 (COVID-era) handling (measured, stated before scoring)

Season 2020 is EXCLUDED from the primary population. The `illness`
designation's meaning plausibly changed under COVID-era protocols (contact
tracing, cautious/mandatory quarantine listings rather than a player's own
felt symptoms), so pooling it with 2010-2019/2021-2024 would mix two
different constructs under one label — exactly the "not naively pooled"
instruction. **Measured density, predictor-only, this session** (final-state
diagnostic, not the true as-of feature, used only to confirm 2020 is not
degenerate): 511 team-weeks, 177 with `n_ill>=1`, 55 with `n_ill>=2`, 17 with
`n_ill>=3` — visibly elevated vs. the 2010-2024-excl-2020 base rates below,
consistent with the regime-change hypothesis. 2020 is scored SEPARATELY as
its own supplementary stratum (same five cell definitions, its own
week-blocked bootstrap; season-blocked is degenerate with one season and
reported as such, not corrected around) and is NOT pooled into the primary
estimate or used to gate any classification.

## 5. Population and predeclared cells (5)

**Primary population**: NFL REG, seasons 2010-2024 EXCLUDING 2020 (14
seasons) — the PIT-recoverable window measured in section 1, minus the
COVID-era stratum handled separately in section 4. Unlike
`docs/fluview_battery.md` (which restricts to `location == "Home"` games,
since its mechanism is specifically about the home team's own home market),
this battery does NOT restrict on `location` — illness burden is a property
of the TEAM's own roster, not the venue, so a neutral-site game still
carries a real illness signal for both sides and is kept in scope. Close-graded via
`schedules.spread_line` (`nfl_ats.features.add_ats_outcomes`, pushes
dropped), identical convention to `docs/fluview_battery.md`. Method: joint
week-blocked bootstrap (block = `season*100+week`) PRIMARY, season-blocked
bootstrap SECONDARY, both via `block_bootstrap_two_group`
(`scripts/_common.py` / identical algorithm to
`scripts/fluview_battery_screen.py`'s own copy). Full-slate effect scaling
via `nfl_ats.experiment_runner.scale_subset_effect`, `accuracy_points`
units. 20,000 bootstrap samples, seed 20260826 (repo convention: today's
date). Within-week correlation is zero by owner mandate — no ICC term.

**Measured density (predictor-only, this session, before any cover-rate sign
was examined)**, final-state diagnostic over the primary population (7,288
team-weeks, 2010-2024 excl. 2020): `n_ill>=1` on 1,930 team-weeks (26.5%),
`n_ill>=2` on 523 (7.2%), `n_ill>=3` on 152 (2.1%); `n_active_ill>=1` (illness
AND not ruled out) on 1,800 (24.7%). The true as-of feature (section 3) will
show somewhat lower density than this final-state diagnostic, since some
reports are incomplete before a game's own cutoff — reported, not corrected
for.

Choosing a SMALL set (not a threshold sweep): the task brief names four
candidate angles — team-week count above a threshold, the 2+-ill-players
team-week specifically, the home-vs-away differential, and whether a
designated-ill player actually played. The mechanism paragraph above itself
says "SEVERAL players" (plural, a cluster) constitutes contagion as opposed
to an isolated case, so **threshold = 2** is used throughout (not selected
from the outcome; it is the same reading `docs/new_lead_classes_20260826.md`
already used and the density check above confirms it is not degenerate:
7.2% of team-weeks). "Whether a designated-ill player actually played" is
not a legitimate PREGAME feature as literally stated (whether a Questionable
player suits up is unknown until kickoff) — it is operationalized instead as
`active_illness_count` (report_status not in `{Out, Doubtful}` as of the
cutoff), the pregame-knowable proxy for "expected to play through it," which
is exactly the "plays through it, market prices it at ~zero" mechanism from
this document's opening paragraph.

**I1. `illness_home_ge2`** — home team's `illness_count >= 2` (as of cutoff)
vs. `< 2`, response `home_cover`. **Predicted sign: NEGATIVE** — a
contagion cluster in the home team's own building plausibly degrades the
home team specifically.

**I2. `illness_away_ge2`** — away team's `illness_count >= 2` vs. `< 2`,
response `home_cover`. **Predicted sign: POSITIVE** — mirror mechanism, the
away team's own cluster degrades them, favoring the home side.

**I3. `illness_differential_home_worse`** — restricted to games where
EXACTLY one side has `illness_count >= 2` (home XOR away, both sides'
as-of values required non-missing); subset = home>=2 & away<2, complement =
away>=2 & home<2. **Predicted sign: NEGATIVE** — isolates the relative form
of the same mechanism, the cleanest test since it removes games where both
or neither side carries the exposure (mirrors `docs/fluview_battery.md`'s
F3 exactly).

**I4. `illness_home_active_ge1`** — home team's `active_illness_count >= 1`
(at least one illness-flagged player NOT ruled out, i.e. expected to play
through it) vs. `0`. **Predicted sign: NEGATIVE** — the specific "plays
through it, market misprices it at ~zero" mechanism from the opening
paragraph, distinct from I1/I2's raw cluster-size test.

**I5. `illness_away_active_ge1`** — away team's `active_illness_count >= 1`
vs. `0`. **Predicted sign: POSITIVE** — mirror of I4.

Every cell excludes rows with a missing required as-of value (either side,
as applicable) from BOTH the subset and complement, reported as
`n_excluded_missing`, never defaulted.

## 6. Reliability check (measured, run before cover-rate scoring)

Split-half reliability of the underlying team-week illness trait, via
`nfl_ats.cfb_qb_dependence.split_half_reliability` (reused directly, the
same function the FluView battery and `injury_value_lost` figures were built
on), applied to a (team, season, week) panel — `metric = illness_count` (the
raw as-of count, not the `>= 2` flag; Pearson r is scale-invariant, so raw
vs. thresholded is immaterial to the reliability figure itself), one row per
team per week (all 32 teams, whether home or away that week — a genuinely
team-specific panel, unlike FluView's shared-by-state panel). Primary
population only (2010-2024 excl. 2020). This tests whether "this team is
running an illness-prone locker room this season" is a real, non-noise
persistent trait within a season (odd/even week split), which is the
assumption every threshold-based cell above depends on. Per AGENTS.md, a
`no_split_half_reliability` closing ground requires this figure's CI to sit
AT (not near) zero — an interval crossing zero here is, as everywhere else,
not grounds to close.

## 7. Caveat (stated up front, per task instruction)

`illness` is a self-reported team designation, and clubs are known to differ
in how liberally they use it (the same caveat `docs/new_lead_classes_20260826.md`
already flagged). This makes the construct a FLOOR on true illness burden,
not a census, and a source of between-team measurement heterogeneity that is
NOT the contagion signal itself. The expected effect of this is attenuation
toward zero and added noise, not a spurious positive — a team culture that
under-reports illness will show fewer flagged team-weeks than its true
burden, diluting rather than inflating any measured effect. This caveat is
disclosed, not corrected for.

## 8. Files

- `scripts/nflverse_injuries_ingest.py` — Part A ingest (this document's
  data source).
- `scripts/nflverse_injuries_reconcile.py` — Part A reconciliation (section
  2 above).
- `scripts/illness_battery_screen.py` — as-of construction, cell scoring,
  writes `artifacts/illness_battery/<UTC>/results.json` (measure-only, no
  registry writes).
- `scripts/illness_battery_record.py` — records all 5 cells to
  `registry/weak_signals.json` via `nfl-ats weak-signals record`, verifies
  after writing.
- `tests/test_illness_battery_leakage.py` — leakage regression test: a
  revision issued after a game's decision cutoff must never reach that
  game's as-of illness feature.
