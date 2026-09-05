# Injury-report text hygiene (LEAD-18, LEAD-19; predeclaration + results)

**Status:** predeclared 2026-09-05, BEFORE any number in section 8 was
computed. Sections 0-7 are frozen; section 8 is appended after the run and
nothing above it is edited afterwards.

**Owning work package:** Phase 12 availability-model measurement leads
(ROADMAP.md `LEAD-18`, `LEAD-19`). Both are QUALITY / feature-hygiene
measurements: **no ATS direction, no rotation window, no
`nfl_ats.rotation`/`nfl_ats.weak_signals` involvement of any kind.** They
exist to check `nfl_ats.availability`'s `report_category` /
`practice_category` / `position_group` feature construction
(`src/nfl_ats/availability.py`) against what actually happens on Sunday.
Files: this document, `src/nfl_ats/injury_report_hygiene.py`,
`scripts/injury_report_hygiene_screen.py`,
`tests/test_injury_report_hygiene.py`, `artifacts/injury_report_hygiene/`.

---

## 0. Binding closing-grounds taxonomy (AGENTS.md, pasted verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (1) refuted mechanism — a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (2) bounded by a positive
control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". The registry code
hard-rejects inadmissible closures; if a record command errors, the verdict is
wrong, not the validator. Verdicts flow only through `nfl-ats weak-signals
record`, never through prose.

This taxonomy governs how a *rejected/refuted* verdict may be written down.
Section 7 below explains why neither LEAD-18 nor LEAD-19 ever reaches the
registry that taxonomy protects (a units problem, not a strength-of-evidence
problem) — that is a separate, narrower conclusion than "this result is
refuted," and section 8 states every number's `probability_positive` plainly
rather than defaulting to a zero-crossing "no effect" read.

## 1. What each lead asks (frozen)

- **LEAD-18** (Concussion recovery by position): among player-weeks that
  carry a Wednesday-equivalent DNP for a concussion, do skill positions
  (QB/RB/WR/TE) sit out the following Sunday MORE often than literal
  offensive-/defensive-line positions (OL/DL)? Predeclared direction:
  skill-minus-line sit-rate gap is positive. Compared against the same
  split for non-concussion DNPs, so the concussion-specific part of any gap
  is visible rather than just reproducing a generic "skill positions get
  rested more" pattern.
- **LEAD-19** (Personal-matter DNP exclusion audit): what is the Sunday
  played-rate for player-weeks whose injury-report text is a non-injury
  "personal matter" designation, compared against (a) genuine-injury DNPs
  at the same practice status and (b) "rest day" DNPs? No predeclared ATS
  direction — this is a feature-hygiene audit that is valuable either way
  (exclude, keep, or keep-with-separate-flag are all live outcomes).

## 2. Population (frozen)

**Data (read, `data/raw/nflverse_injuries/20260826T122850Z/manifest.json`
and `injuries.parquet`; `data/players/raw/20260817T184901Z/snap_counts.parquet`
+ `weekly_rosters.parquet`):** nflverse injury reports, 90,752 raw rows,
requested range 2009-2025; PFR-sourced snap counts and nflverse weekly
rosters, 2013-2025 (snap counts) / 2009-2025 (rosters).

**Season range and an already-disclosed data gap (read, the snapshot's own
`manifest.json`):** the manifest's `point_in_time_note` states plainly that
`date_modified` is "6068/6068 (100%, entirely absent -- replaced by a
season_type column instead) for 2025" and that "the point-in-time-recoverable
window is therefore 2010-2024, not the nominal 2009-2025 ingest range."
`nfl_ats.players.canonicalize_injuries` (production's own injury
canonicalizer) enforces `date_modified.notna()` and this module's
`canonicalize_injury_text_rows` mirrors that same requirement, so **both
LEAD-18 and LEAD-19's nominal "2013-2025" population is, measured, actually
2013-2024** (measured:
`nfl_ats.players.canonicalize_injuries(injuries)["season"]` has max 2024
against the same input file; confirmed independently in this module's own
`build_player_week_frame` output). This is not a bug introduced by this
work — it reproduces a limitation the snapshot already disclosed and that
production's live injury features already carry. It is disclosed again here
because the ROADMAP row and this document's own title both say "2013-2025."
Per AGENTS.md's "verify before quoting anything that gates a decision"
rule, this was read from the manifest and independently reproduced against
production's own `canonicalize_injuries`, not assumed from the ROADMAP
row's stated range.

**Outcome:** `played` = any recorded offense/defense/special-teams snap for
that player-week in the PFR snap-count table (`nfl_ats.players.canonicalize_snaps`
+ `canonicalize_rosters` + `attach_snap_player_ids`, reused verbatim, no new
join logic). A player-week absent from the snap table is treated as
`played=False`: nflverse's snap-count feed only emits a row for players who
recorded at least one snap, so absence from the table is itself the
zero-snap signal, not missing data.

**One report per player-week, measured:** grouping the canonicalized
2013-2024 injury rows by (season, week, team, gsis_id) yields 62,489
distinct player-weeks with only **2** carrying more than one revision
(0.003%). Among the 2,872 (2009-2025, REG) / 1,265 (2013-2024 DNP subset
used below) concussion-flagged player-weeks specifically, **0** carry more
than one revision. "Earliest report of the week" is therefore not really an
approximation for this population — nflverse's injury feed already
publishes one row per player-week — but the code still selects the earliest
`date_modified` explicitly (`select_earliest_revision_per_player_week`)
rather than assuming it, and reports the multi-revision count every run so
a future snapshot with more revision granularity would be caught, not
silently mishandled.

**DNP status (frozen, exact string):** `practice_status ==
"Did Not Participate In Practice"`. Deliberately excludes the stricter,
much rarer `"Out (Definitely Will Not Play)"` practice status (974 raw
rows), which is a separate designation this document does not fold in.

**LEAD-18 position groups (frozen, narrower than the rest of the codebase
on purpose):** `CONCUSSION_SKILL_POSITIONS = {QB, RB, WR, TE, FB, HB}`,
`CONCUSSION_LINE_POSITIONS = {C, G, OG, OL, OT, T, DE, DL, DT, NT, EDGE}`.
This is narrower than `nfl_ats.availability.position_group`'s `_FRONT`
group (which folds LB/OLB/ILB in with DE/DT/NT) because LEAD-18's ROADMAP
row names "OL/DL" specifically — using the codebase's existing 4-group
split would silently substitute a different, broader comparison than the
one predeclared.

**LEAD-18 concussion flag (frozen):** `report_primary_injury == "Concussion"`
(exact match, case-sensitive as nflverse writes it) — this is the same
definition this task's own briefing used to report "2,963 rows" for the
full 2009-2025 raw file, confirmed by measurement here
(`report_primary_injury.eq("Concussion")` on the raw REG+POST frame:
2,963 rows). `practice_primary_injury == "Concussion"` is a broader,
overlapping field (879 rows tag concussion there but not in
`report_primary_injury`) that is NOT used for the population definition,
to keep it anchored to the field the ROADMAP row and this task's briefing
both cite.

**LEAD-19 designation vocabulary (frozen, exact lowercase/stripped match
against EITHER `report_primary_injury` or `practice_primary_injury`,
never substring):** built by grepping the distinct values of both columns
(2013-2024 REG rows) for "personal", "not injury", "rest", "illness",
"coach", "suspend", "travel", "discipline", "team decision" and
hand-sorting the distinct hits into five disjoint buckets:

| Bucket | Frozen strings |
|---|---|
| `personal_matter` | `personal matter`, `not injury related - personal matter`, `personal` |
| `rest_day` | `rest`, `rested`, `resting veteran`, `not injury related - resting player`, `not injury related -- resting veteran` |
| `illness` | `illness`, `illness (non-covid)`, `medical illness`, `non-football illness` |
| `coach_team_decision` | `coach's decision`, `coaching`, `coaching decision`, `not injury related - coach's decision`, `not injury related - coaching decision`, `not injury related - team decision` |
| `other_non_injury` | `not injury related`, `not injury related - other`, `not injury related - discipline`, `not injury related - returning from suspension`, `not injury related - did not travel`, `not injury related - travel`, `travel after trade` |

Everything else (every genuine body-part/medical string, PLUS compound
strings that mix a body part with a non-injury tag such as `"Ankle [Not
Injury Related - Personal, Thursday Only]"` or `"Knee/Rested"`, PLUS
one-off narrative sentences such as "Did not travel to Brazil due to a
personal matter...") classifies as `injury` — the conservative direction
for an exclusion audit, since it can only shrink the personal-matter/rest
populations, never inflate them with an ambiguous string. Full grep output
lives in this lane's session transcript; the frozen sets above are also the
literal module-level constants in `src/nfl_ats/injury_report_hygiene.py`.

## 3. Statistic (frozen)

Season-blocked bootstrap (2,000 draws, fixed seed 20260905) of
`mean(outcome | group_a) - mean(outcome | group_b)`: whole seasons are
resampled with replacement (never individual player-weeks), and BOTH
groups reuse the identical per-draw season-resampling weights, so the two
groups' resampled rates are correlated exactly as they would be under a
literal "resample seasons, then split by group" bootstrap rather than two
independently-seeded ones. Seasons, not weeks, are the block unit: AGENTS.md's
"within-week correlation is zero" mandate governs game-level correlation
within one week's slate — a different population from this
player-week-across-seasons frame. `probability_positive` = fraction of
draws with the resampled gap `> 0`.

- LEAD-18 outcome: `sat_out` = `1 - played` (predeclared positive direction:
  skill sits MORE than line).
- LEAD-19 outcome: `played_f` = `played` as a float (no predeclared
  direction; both gaps are reported as measured).

## 4. Comparisons (frozen)

- LEAD-18: the concussion+DNP skill-vs-line gap is reported alongside the
  identical statistic computed on non-concussion DNPs (any other
  designation), so a concussion-specific effect is distinguishable from a
  generic "skill positions get protected on any DNP" pattern.
- LEAD-19: `personal_matter` vs `injury` (same practice status) and
  `personal_matter` vs `rest_day` are both reported. `illness` and
  `coach_team_decision` base rates are reported descriptively (section 8)
  but not bootstrapped as primary comparisons — outside this document's two
  frozen contrasts.
- Both leads: per-era split 2013-2017 vs 2018-2024 (2018-2025 nominally;
  see section 2 on the 2025 gap). Per AGENTS.md, a weaker- or absent-era
  reading is never treated as "no effect," and an empty era slice is
  reported as empty, not silently skipped.

## 5. Leakage (frozen)

None applicable: this is a purely descriptive, completed-season,
outcome-vs-designation measurement with no pregame model application, no
feature-table join, and no card/pick output. `tests/test_injury_report_hygiene.py`
still pins a leakage-style regression: the outcome (`played`) must never be
allowed to influence which player-weeks are classified into which
population (frozen string membership and practice-status filtering happen
on injury-report fields alone, verified on a synthetic frame where a
population flag is computed and then the outcome is permuted, and the
membership must not change).

## 6. Test contract (release-blocking)

`tests/test_injury_report_hygiene.py` covers, without network access, all
on synthetic frames: the frozen designation string sets classify correctly
(including that a compound/narrative string that is NOT in a frozen set
falls through to `injury`, never guessed); `concussion_position_group`'s
skill/line/other split; `select_earliest_revision_per_player_week` picks
the earliest `date_modified` and correctly counts multi-revision
player-weeks; `attach_played_outcome`'s Sunday-action join (present in the
snap table => played, absent => not played) on a hand-built frame;
`season_block_bootstrap_gap`'s determinism (same seed, same draws) and its
graceful handling of an empty-group input (raises `DataContractError`
rather than returning a garbage interval); and the leakage-style
population/outcome-independence check described in section 5.

## 7. Decision rule and registry-recording rule (frozen)

**Registry:** `nfl_ats.weak_signals.EFFECT_UNITS` is `(ats_points,
accuracy_points, brier, log_loss, mae, correlation, mae_improvement,
brier_improvement, log_loss_improvement)` (read,
`src/nfl_ats/weak_signals.py`). Every one of these is either an ATS-pick
metric (`ats_points`, `accuracy_points`), a probabilistic-forecast error
metric (`brier`, `log_loss`, `mae` and their `*_improvement` variants), or
a correlation coefficient. LEAD-18's skill-minus-line sit-rate gap and
LEAD-19's personal-vs-injury/rest-day played-rate gap are neither: they are
raw differences in a Sunday-availability RATE between two subpopulations,
with no forecast, no model, and no ATS pick anywhere in the computation.
Forcing either into `accuracy_points` (the closest-sounding unit) would
silently misrepresent a player-availability base-rate gap as a forced-pick
accuracy delta — exactly the unit-mismatch AGENTS.md's commensurability
rule warns against ("A pooled number built from a production quantity plus
a subset cover-rate gap is not a finding; it collapses under the next
audit"). **Conclusion: no admissible `EFFECT_UNITS` value fits a rate
difference. Neither result is recorded via `nfl-ats weak-signals record`,
under any outcome.** This document plus the
`artifacts/injury_report_hygiene/<stamp>/results.json` artifact ARE the
record.

**Decision, stated before caveats (per AGENTS.md's "state what a result
implies for the decision before what is wrong with it"):**

- LEAD-18: the concussion-DNP population sits out Sunday **99.13% of the
  time overall regardless of position** (1,254 of 1,265 player-weeks) — a
  ceiling effect that leaves very little room for a positional gap to show
  up, and the point estimate is small and sign-unstable across cuts
  (positive in 2013-2017, negative in 2018-2024, see section 8). The
  predeclared skill-sits-more hypothesis is **not supported** by this
  population, but nor is it refuted (no interval sits wholly on the wrong
  side of zero) — this is `unresolved_below_power` in substance, reported
  here as `probability_positive` per cut rather than recorded, since no
  registry unit fits (see above). The DECISION this implies for the
  availability model: a concussion-specific positional adjustment on top of
  the existing severity/position-group unavailability prior is not
  supported by 2013-2024 data and should not be added on the strength of
  this measurement; the generic (non-concussion) DNP population DOES show a
  robust, whole-interval-positive skill-vs-line gap in both eras, so the
  existing position-group split in `nfl_ats.availability.position_group`
  remains justified for ordinary injuries — it just does not extend to
  concussions specifically.
- LEAD-19: personal-matter DNPs play the following Sunday **53.3% of the
  time** (227 player-weeks, 2018-2024; zero personal-matter rows exist
  2013-2017 — see section 8), dramatically higher than genuine-injury DNPs
  at the same practice status (**7.9%**, n=14,752, whole-interval-positive
  gap, P+ 1.0) and dramatically LOWER than rest-day DNPs (**95.7%**, n=897,
  whole-interval-negative gap, P+ 0.0). Personal-matter DNPs are
  statistically distinct from BOTH of the categories a feature designer
  might be tempted to fold them into. **Decision implied: keep-with-
  separate-flag, not exclude and not merge.** Excluding these 227
  player-weeks outright would throw away real information (a 53.3% played
  rate is far from uninformative); folding them into the standard
  injury-DNP bucket would bias that bucket's learned unavailability rate
  upward from its true ~7.9% floor; folding them into rest-day would bias
  it downward from rest-day's ~95.7% ceiling. The `no_report_status_mixed_cell_diagnostic`
  in section 8 shows this is not a hypothetical risk: the specific combo
  cell `nfl_ats.availability`'s learned-rate table already builds today
  (blank `report_status` + DNP practice status) mixes played rates from
  30.3% (genuine injury) to 96.7% (rest day) within ONE combo/position-group
  cell, with genuine injury actually a MINORITY (28.7%) of that cell's
  2,672 rows.

## 8. Results (measured, `scripts/injury_report_hygiene_screen.py`, artifact
`artifacts/injury_report_hygiene/20260905T044503Z/results.json`, 62,489
total player-weeks, 2 multi-revision player-weeks, seed 20260905, 2,000
bootstrap samples per interval)

### LEAD-18: concussion-DNP skill-vs-line sit rate

Population: 1,265 concussion+DNP player-weeks (skill 452, line 359, other
454); non-concussion DNP comparison: 16,243 player-weeks (skill 4,777,
line 5,463).

| Cut | n skill / n line | skill sit rate | line sit rate | gap (skill-line) | 95% CI | P+ |
|---|---|---|---|---|---|---|
| Concussion DNP, 2013-2024 | 452 / 359 | 0.9912 | 0.9944 | -0.00328 | [-0.01674, +0.00849] | 0.3365 |
| Concussion DNP, 2013-2017 | 190 / 163 | 0.9947 | 0.9877 | +0.00701 | [-0.01282, +0.02105] | 0.7765 |
| Concussion DNP, 2018-2024 | 262 / 196 | 0.9885 | 1.0000 | -0.01145 | [-0.02536, +0.00000] | 0.0000 |
| Non-concussion DNP, 2013-2024 | 4,777 / 5,463 | 0.8355 | 0.7393 | +0.09612 | [+0.07513, +0.11674] | 1.0000 |
| Non-concussion DNP, 2013-2017 | 1,796 / 2,000 | 0.8797 | 0.8155 | +0.06423 | [+0.03658, +0.09203] | 1.0000 |
| Non-concussion DNP, 2018-2024 | 2,981 / 3,463 | 0.8088 | 0.6954 | +0.11344 | [+0.09235, +0.13329] | 1.0000 |

Reading the taxonomy on the 2018-2024 concussion cut: the CI's upper bound
is exactly 0.0, not below it, so this does NOT meet `wrong_sign_resolved`
(the taxonomy requires the WHOLE interval below zero) — it stays
`unresolved_below_power` in substance despite `probability_positive` 0.0,
because `line`'s point rate is a rare literal 1.0 (196 of 196 played=False,
i.e. zero line-position concussion-DNP player-weeks 2018-2024 played) that
mechanically floors the bootstrap draws' upper tail at zero.

### LEAD-19: personal-matter vs injury vs rest-day Sunday-action rate

Population: 17,508 DNP player-weeks by designation — `injury` 14,752,
`other_non_injury` 900, `rest_day` 897, `illness` 728, `personal_matter`
227, `coach_team_decision` 4.

| Comparison | n personal / n other | personal rate | other rate | gap | 95% CI | P+ |
|---|---|---|---|---|---|---|
| personal_matter vs injury, 2013-2024 | 227 / 14,752 | 0.5330 | 0.0790 | +0.45400 | [+0.26512, +0.60454] | 1.0000 |
| personal_matter vs rest_day, 2013-2024 | 227 / 897 | 0.5330 | 0.9565 | -0.42348 | [-0.63455, -0.25271] | 0.0000 |
| personal_matter vs injury, 2018-2024 | 227 / 8,639 | 0.5330 | 0.0815 | +0.45155 | [+0.26824, +0.60378] | 1.0000 |
| personal_matter vs rest_day, 2018-2024 | 227 / 896 | 0.5330 | 0.9565 | -0.42343 | [-0.59902, -0.26485] | 0.0000 |
| personal_matter vs injury/rest_day, 2013-2017 | 0 / — | — | — | skipped: 0 personal-matter rows this era | — | — |

By-designation base rates (DNP population, no bootstrap): `injury` 0.0790
(n=14,752), `illness` 0.5137 (n=728), `personal_matter` 0.5330 (n=227),
`other_non_injury` 0.8289 (n=900), `rest_day` 0.9565 (n=897),
`coach_team_decision` 0.0000 (n=4, too small to interpret).

By season, `personal_matter` DNPs: 2019: 1, 2021: 78, 2022: 54, 2023: 56,
2024: 38 (zero 2013-2018, 2020). `rest_day` DNPs: 2017: 1, 2019: 15, 2020:
11, 2021: 256, 2022: 221, 2023: 249, 2024: 144 (zero 2013-2016, 2018).
Both designations are effectively a 2021-onward phenomenon in this
snapshot — inferred (not measured) explanation: these read as
standardized-vocabulary designations that entered common use around the
2020-2021 CBA/reporting-practice changes, not evidence that "personal
matter" absences did not occur before 2021.

By standard position group (`nfl_ats.availability.position_group`),
played rate for `personal_matter`: skill 0.375 (n=80), offensive_line
0.432 (n=37), front 0.673 (n=55), secondary 0.632 (n=38), other 0.824
(n=17) — every group sits between the `injury` and `rest_day` floors/
ceilings for its own group, none matching either.

`no_report_status_mixed_cell_diagnostic` (DNP rows with blank
`report_status` — the specific combo cell `nfl_ats.availability`'s
`report_category="none"` x `practice_category="dnp"` pools today): `injury`
0.3029 (n=766, 28.7% of this cell), `illness` 0.6797 (n=153, 5.7%),
`personal_matter` 0.8116 (n=138, 5.2%), `other_non_injury` 0.9065 (n=738,
27.6%), `rest_day` 0.9669 (n=877, 32.8%) — total 2,672 rows, played rate
ranging 30.3%-96.7% within one combo cell the model currently treats as
homogeneous.

## 9. Producer/feature-model design inputs (measured, for the future builder)

1. Do not add a concussion-specific positional unavailability adjustment on
   top of the existing severity/position-group prior (LEAD-18: not
   supported, ceiling effect, sign-unstable across eras).
2. The existing skill-vs-line/front split for ordinary (non-concussion)
   DNPs IS supported (whole-interval-positive both eras) and should be left
   as-is.
3. `personal_matter` (and, less urgently given its near-ceiling rate
   already resembling a genuine "healthy" prior, `rest_day`) should get
   their own designation-level entries in the learned availability-rate
   table (`nfl_ats.availability.build_season_lagged_availability_rates`'s
   `report_category`/`practice_category` combo keys) rather than falling
   through to the shared blank-report-status/DNP cell, which currently
   mixes a 30.3%-96.7% range of true behavior into one number.
