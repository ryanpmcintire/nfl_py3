# Interim head coach screen

Predeclared and screened 2026-08-20, per `docs/data_source_scout_v3.md`
section 5 ("Interim head coach games", rank 5, effort S). Mechanism: a
documented motivation/effort discontinuity in sports-betting research when a
team fires its head coach mid-season and plays under an interim. Every claim
below is tagged **measured** (run this session, command/path given),
**read** (a file opened this session), **reported** (a source found via
search, not independently re-verified), or **inferred** (reasoning, not
evidence), per `AGENTS.md`'s labeling rule.

**Binding closing-grounds taxonomy** (governs every verdict in this
document): an interval or CI that contains zero is NEVER grounds to reject,
fail, or close an experiment. At this evaluator's ~2-point resolution,
"contains zero" is the EXPECTED outcome for a real small signal. Only two
grounds ever close a line of work: (1) refuted mechanism — a RESOLVED wrong
sign (whole interval on the wrong side of zero) or zero split-half
reliability; (2) bounded by a positive control proven able to detect an
effect that size. Everything else is `unresolved_below_power`: recorded with
`nfl-ats weak-signals record` (here, via `nfl-ats experiment run`), reporting
`probability_positive`, never the binary "contains zero". Every cell below
was classified `unresolved_below_power` by the experiment runner's own
mechanical classifier — none crossed the `refuted_mechanism` bar (a
week-blocked interval sitting entirely below zero AND needing >1.099x
widening to re-cross), and none is claimed as `bounded_by_control` (no
positive control was run in this pass).

## 1. Data source: Pro Football Rumors interim-coach list

**Measured** this session: `curl` (not WebFetch, to get the untouched raw
HTML) against
`https://www.profootballrumors.com/2025/10/the-nfls-interim-coaches-since-2000`
returned HTTP 200, 89,817 bytes. The article's own JSON-LD carries
`datePublished: 2025-10-13` and `dateModified: 2025-11-10` (the page is kept
current as new interim hires happen — the Nov 10, 2025 Giants entry is the
newest row). Raw HTML, a hand-transcribed `parsed_table.csv` (52 rows), and a
`manifest.json` documenting the fetch, parsing method, and cross-checks are
snapshotted at `data/raw/interim_coaches/20260820T105234Z/`.

**Cross-checks (measured/reported this session):** 3 entries selected by a
seeded `random.sample(range(1,53), 3)` (seed 20260820, reproducible) —
entries 32 (Steve Spagnuolo/Giants 2017), 37 (Romeo Crennel/Texans 2020), 44
(Jerry Rosburg/Broncos 2022) — were independently verified via WebSearch
against outlets with no PFR affiliation (CBS News/NFL.com/SI.com; ABC13/Fox26
Houston; CNN/CBS Colorado/ESPN). All 3 agree with the PFR list exactly (dates
and facts). Two more entries (Perry Fewell/Bills 2009, Eric Studesville/
Broncos 2010) were separately spot-checked **directly against
`schedules.parquet`'s own coach field** (measured, not a news search) and
also matched exactly. Combined: 6 of 6 spot-checked entries agree; no
disagreements found. Full detail in `manifest.json`'s `cross_checks` object.

## 2. Join methodology

**Primary join (measured):** rather than reconstruct a date-range cutoff
from PFR's prose dates, this join matches on
`(team, season, credited_coach_name)` directly against the newest
`data/raw/*/schedules.parquet` snapshot's own **per-game**
`home_coach`/`away_coach` field. This is strictly more precise where it
works: three spot-checked team-seasons (BUF 2009, DEN 2010, NYG 2017) show
the coach-name transition landing on exactly the week boundary PFR's stated
takeover date implies, and this join is what resolved the two 2012 Saints
entries (Aaron Kromer, Joe Vitt) that PFR's own article gives no date for at
all — `schedules.parquet` shows NO's credited coach as Kromer for weeks 1-7
and Vitt for weeks 8-17 of 2012, unprompted.

**Fallback join (measured):** for 10 of 39 joinable entries,
`schedules.parquet`'s coach field does **not** reflect the in-season change
at all — it credits the fired coach for every remaining game of the season
(a data-quality gap in nflverse's own source, not this project's). Those 10
entries (Dan Campbell/MIA 2015, Mike Mularkey/TEN 2015, John Fassel/LA 2016,
Perry Fewell/CAR 2019, Jay/Jerry Rosburg/DEN 2022, Jeff Ulbrich/NYJ 2024,
Darren Rizzi/NO 2024, Thomas Brown/CHI 2024, Mike McCoy/TEN 2025, Mike
Kafka/NYG 2025) fall back to PFR's own stated `takeover_date_iso` as a date
cutoff (`team`, `season`, `gameday >= takeover_date`). The join code
(`nfl_ats.experiment_runner._build_interim_coach_trait_data`) raises loudly
if even the fallback finds zero games for an entry, or if the final matched-
entry count ever differs from the joinable-entry count — this was exercised
for real while building it (see `manifest.json`'s `join_key_design`).

**Coverage floor (measured):** `data/processed/game_features.parquet` and
every `data/raw/*/schedules.parquet` snapshot in this repo both start at
`season=2009`. Of the 52 listed interim stints (2000-2025), **39 are
joinable** to this project's own graded ATS archive; the 13 from 2000-2008
are kept in `parsed_table.csv` for provenance but excluded from every cell
below — an honest coverage limit, not a defect in the source list.

**Result:** 39 joinable interim stints produce **250 REG-season, non-push
team-games** under an interim head coach, 2009-2025 (some seasons as few as
3, one as many as 28 — mid-season firings are a genuinely rare event, so
small-n is expected and reported honestly, not treated as a problem).

## 3. Overlap with the live `hc_year_one_fade_overlay` challenger

Explicitly checked, per the task brief. `hc_year_one_fade_overlay`
(`nfl_ats.coach_fade_overlay`) flags a team whose CURRENT-season coach
differs from LAST season's — a whole-season condition, active only weeks
1-8. This family flags a team whose coach changed WITHIN the current season
— narrower, and defined without regard to week number.

**Measured** overlap: of the 250 under-interim team-games, 174 (69.6%) are
ALSO flagged `year_one` by the whole-season construct (expected — most
interims weren't the team's coach last season either). But `hc_year_one_
fade_overlay` only ever ACTS in weeks 1-8, and only **32 of the 250**
under-interim games (12.8%) fall in that window at all; of the `first_game`
sub-cell specifically (see below), only **11 of 39** fall in weeks 1-8, of
which 8 would also be touched by the other overlay. Practical read: the two
families rarely collide in practice, but a genuinely early-season firing (a
real possibility — Gregg Williams/CLE took over week 8 of 2018, Bill
Callahan/WAS week 5 of 2019) could trigger both simultaneously. Any future
live wiring of an interim-coach overlay needs a stated precedence rule for
that overlap case; none exists yet because neither overlay is wired to
collide today.

## 4. Predeclared cell family

Predeclared in this document, in full, **before any cover-rate sign was
computed** (only join mechanics/counts were inspected first). Four cells,
plus two era re-slices of the headline cell:

| Cell | Flag | Population | Predeclared sign / mechanism |
|---|---|---|---|
| `interim_hc_active` | Team currently under an interim HC | vs. rest of league | **+1**: motivation/effort discontinuity lifts cover rate |
| `interim_hc_first_game` | Team's FIRST game under a NEW interim | vs. rest of league | **+1**: the specific bettor-folklore claim |
| `interim_hc_home_within_interim` | Home vs. road, restricted to interim games | two-sided, within interim pop. | **arbitrary convention** (+1 = home > road); NO predeclared mechanism — descriptive only, reported honestly regardless of sign |
| `interim_hc_fired_year_one` | Fired coach was himself in year 1 of his own tenure | restricted to interim games with known predecessor tenure | **-1**: a year-1 firing signals organizational chaos/panic (vs. a considered reset), hypothesized to BLUNT the interim bump |

`interim_hc_active` was also re-run (a) excluding the 2012 Saints
(`predecessor_status == "suspended"` — Sean Payton was suspended for the
Bounty Scandal, not fired, the one entry that doesn't cleanly match the
stated "firing" mechanism) as a sensitivity check, and (b) over two eras,
2009-2017 / 2018-2025 (the same boundary `hc_year_one_fade` already uses),
per the owner rule that era differences are reported as magnitude
differences, never absence.

`fired_coach_was_year_one` reuses the EXACT year-1 definition
`hc_year_one_fade_overlay` already uses
(`nfl_ats.coach_fade_overlay.team_season_primary_coach`), applied to the
season BEFORE the takeover season — this is a genuinely different
population/question than that overlay's own construct (which asks about the
CURRENT interim/new coach's own tenure, not the fired predecessor's), so
there is no double-counting between this cell and the live challenger.

"Interim window length so far" (`interim_game_number`, computed for every
row) is reported descriptively below rather than run through the pipeline as
its own registry cell: it is a continuous covariate, and `interim_hc_first_
game` already supplies the natural single boolean split of it (game 1 vs.
everyone else) that the folklore claim itself calls for.

## 5. Results (all cells, measured via `nfl-ats experiment run`)

Week-blocked is the primary interval per the spec (default
`blocking.primary`); season-blocked is secondary. All runs: 20,000 resamples,
seed 20260820, `population.grade="close"`.

| Cell | n games (flag) | Raw cover rate: flag vs. complement | Effect (full-slate, accuracy pts) | 95% week-blocked interval | P+ (week) | P+ (season) | Classification |
|---|---|---|---|---|---|---|---|
| `interim_hc_active` | 8,634 (250) | 49.20% vs. 50.02% | -0.024 | [-0.204, +0.165] | 0.386 | 0.384 | `unresolved_below_power` |
| `interim_hc_active_excl_suspension` | 8,618 (234) | (Saints 2012 excluded) | -0.024 | [-0.198, +0.158] | 0.395 | 0.384 | `unresolved_below_power` |
| `interim_hc_first_game` | 8,634 (39) | **58.97%** vs. 49.96% | **+0.041** | [-0.034, +0.111] | **0.845** | 0.834 | `unresolved_below_power` |
| `interim_hc_home_within_interim` | 250 (127 home / 123 road) | 51.97% (home) vs. 46.34% (road) | +0.163 | [-0.194, +0.518] | 0.811 | **0.952** | `unresolved_below_power` |
| `interim_hc_fired_year_one` | 225 (38) | 47.37% (year-1-fired) vs. 48.66% (not) | +0.034 | [-0.413, +0.502] | 0.559 | 0.568 | `unresolved_below_power` |
| `interim_hc_active_era_2009_2017` | 4,484 (98) | — | -0.046 | [-0.275, +0.183] | 0.333 | n/a (9 blocks, degenerate) | `unresolved_below_power` |
| `interim_hc_active_era_2018_2025` | 4,150 (152) | — | +0.000 | [-0.293, +0.294] | 0.487 | n/a (8 blocks, degenerate) | `unresolved_below_power` |

Season-blocked intervals for both era slices came back flagged
**degenerate** by the runner's own D4 guard (8-9 blocks, below
`MIN_BLOCKS_FOR_INTERVAL=10`) — reported as `P+` only, per the runner's own
warning ("report the estimate and probability_positive, not this interval").
Week-blocked has 141-153 blocks in both eras and is unaffected.

All 7 entries confirmed present via `nfl-ats weak-signals status` after
recording (no parallel-writer race). Registry total: 271 signals.

## 6. The decomposition: folklore holds for game 1, not for the stint overall

The headline `interim_hc_active` cell (any game under an interim, any point
in the stint) is flat-to-slightly-negative (49.20% vs. 50.02%, P+ 0.386) —
**not** what the folklore predicts if read as "interim coaches help cover
rate broadly." But splitting the 250 under-interim games by
`interim_game_number` (measured, not part of the registry cells — a
descriptive decomposition) tells a sharper story:

- **Game 1 only** (n=39): 58.97% cover.
- **Games 2+** (n=211): 47.39% cover — actually below the 250-game average
  AND below the league baseline.

The folklore claim — "teams often cover their first game under a new
interim" — has real support in this data (P+ 0.845, the strongest lean of
any cell here). It does not extrapolate to the rest of the stint; if
anything the reverse. This is exactly the kind of finding AGENTS.md's
pooling section describes as legitimate: a flat aggregate can decompose into
a real, sharper piece plus an offsetting piece, and the piece is the finding
worth keeping — not the aggregate that hides it.

The `interim_hc_home_within_interim` cell (51.97% home vs. 46.34% road,
P+ 0.811-0.952) is reported for completeness with its predeclared caveat: no
directional mechanism was stated in advance, so this is descriptive context
for the wiring discussion below, not a claim on its own.

`interim_hc_fired_year_one` points in the hypothesized direction (both
sub-groups sit below 50%, and the year-1-fired group is lower: 47.37% vs.
48.66%) but the interval is very wide (n=38 flag rows) and P+ sits almost
exactly at a coin flip (0.559/0.568) — genuinely unresolved, reported
honestly rather than rounded up or down.

## 7. Wiring status: WIRED as a dual-tracked challenger (2026-08-20)

**Updated 2026-08-20, same day as the sketch below was written.** The
sketch that follows this note originally described `interim_hc_first_game`
as a recommendation only ("NOT wired"). It has since been built exactly
along those lines, as a no-window-cost, dual-tracked prospective challenger
— never applied to the published card, spends no rotation-registry window,
and changes no pick anyone actually plays. This is a positive-EV wiring
decision per `AGENTS.md`'s "a promotion bar is not a decision bar" (P+
0.845 > 0.5, and a dual-tracked challenger has no downside to wiring), not
a claim that the underlying cell is a proven edge — it remains
`unresolved_below_power` in the registry, exactly as measured in section 5.

**What was actually built, mapped onto the sketch's own five numbered
steps:**

1. **Prospective ingestion:** not rebuilt in this pass — the challenger
   reads whatever `data/raw/interim_coaches/*/parsed_table.csv` snapshot is
   newest at run time (same snapshot this document's own measurement used).
   A weekly re-fetch-and-diff step (as the sketch proposed) is still future
   work; until it exists, the challenger's live reads use a snapshot that is
   only as fresh as the last manual fetch. This is a live-data freshness gap,
   not a correctness gap — see the fail-open behavior below.
2. **The pregame-safe per-game flag** is
   `nfl_ats.interim_hc_first_game_tilt_overlay.interim_first_game_flag_by_game_fail_open`.
   Rather than promoting `_build_interim_coach_trait_data` out of
   `nfl_ats.experiment_runner`'s private namespace into a new shared module
   (the sketch's original suggestion), the overlay module imports it
   directly (a local, function-scoped import, mirroring how
   `_build_interim_coach_trait_data` itself locally imports
   `nfl_ats.coach_fade_overlay.team_season_primary_coach`) — this reuses the
   exact join verbatim without touching `experiment_runner.py`, lower-risk
   while other agents are concurrently working in this tree. **New,
   FAIL-OPEN by design** (not in the original sketch): any join failure —
   no snapshot fetched yet, a malformed source file, a missing schedules
   snapshot — is caught, logged as a `RuntimeWarning`, and folds into "zero
   games flagged" rather than raising, mirroring
   `forecast_cold_visitor_tilt_overlay`'s fail-open live-fetch wrapper. This
   is what makes Week 1 2026 (trivially zero interim coaches) and any other
   week where the local snapshot is stale/missing render safely as a no-op
   rather than a build failure.
3. **`apply_interim_hc_first_game_tilt_overlay(predictions, repo_root)`**
   (`src/nfl_ats/interim_hc_first_game_tilt_overlay.py`): the pick-level
   transform, built exactly as sketched — when the model's own pick does NOT
   already side with the interim-coached team in its first game, flip toward
   it. No week restriction, exactly as predicted (the cell has no week
   dependency by construction). One addition beyond the sketch: a rare
   simultaneous case (both sides of a matchup are each in their own interim
   stint's first game at once) is detected and left untouched — no measured
   direction for that case — rather than flipping on an arbitrary tie-break.
4. **Precedence with `hc_year_one_fade_overlay`: resolved as "both track
   independently," not as one overlay taking precedence over the other.**
   The sketch's own proposed resolution ("year-1 overlay takes precedence")
   turned out to be unnecessary: neither overlay is composed with the other
   in code — each is a dual-tracked challenger that transforms the SAME
   un-overlaid base card independently and records its own arm to the
   prospective challenger ledger. `interim_hc_first_game_tilt_overlay` never
   sees, and is never seen by, `hc_year_one_fade_overlay`'s flips, mirroring
   the same independence `spread_gap_zone_fade_overlay` already documents
   for its own relationship to that overlay. A precedence rule would only
   become necessary if both overlays were ever played on the real card
   simultaneously, which is not the case today (only `hc_year_one_fade_overlay`
   is applied to the published card).
5. **Recording:**
   `nfl_ats.interim_hc_first_game_tilt_overlay.record_interim_hc_first_game_tilt_challenger_decisions`,
   built exactly as sketched (the same `bet_side="PASS"`/`edge=NaN` pattern
   every sibling overlay challenger uses), wired into
   `nfl-ats publish-predictions --record-decisions` in `cli.py`.

**Registration:** `interim_hc_first_game_tilt_overlay` in
`artifacts/prospective/challengers.json`, status `ACTIVE_PROSPECTIVE`. Tests:
`tests/test_interim_hc_first_game_tilt_overlay.py` (flag derivation,
fail-open behavior, the pick-level flip including the REG-only gate and the
simultaneous-both-sides case, the disclosure note, and challenger-ledger
recording including the fingerprint-mismatch and inactive-registration
refusals). The dashboard's challenger board
(`src/nfl_ats/public_board.py`) previews this challenger's hypothetical
weekly flip the same way it previews `division_revenge_tilt_overlay` and
`surface_switch_tilt_overlay` — Week 1 2026 renders the honest "no games
matched its rule this week" sentence, since mid-season firings cannot exist
in Week 1 by construction.

**Original sketch, kept below verbatim for the historical record of what
was proposed before it was built:**

`interim_hc_first_game` is the candidate worth sketching: P+ 0.845 (week) /
0.834 (season), the strongest lean of any cell in this family, on a
mechanism distinct from every other live challenger, with a raw effect (+9.0
points of raw cover-rate gap, +0.041 accuracy points at full-slate scale)
that is not small even though `n=39` is. Per `AGENTS.md`'s "a promotion bar
is not a decision bar": this is not being proposed as a proven edge — it is
`unresolved_below_power`, reported as `probability_positive`, and the
project's own EV logic (not a 0.90 confidence bar) is what would decide
whether to ever play it.

**Why live availability is trivially safe:** interim-coach firings are
public news, announced well before the team's next kickoff, with a proven
free source (PFR's own list is kept current — the Nov 10, 2025 Giants entry
is the newest row, added within the same article this session fetched). No
new paid data or scraping infrastructure is needed to know "this team's
coach just changed" — it is exactly as pregame-safe as the already-wired
`hc_year_one_fade_overlay`.

**Sketch, if ever built** (cloning `nfl_ats.coach_fade_overlay`'s existing
pattern almost exactly):

1. **Prospective ingestion (new, small):** since new interim hires are rare
   (0-3/season league-wide), a full scraper is overkill. Re-fetch
   `profootballrumors.com`'s interim-coaches article weekly during the
   season (same URL, same parse this session already validated) and diff
   against the stored `parsed_table.csv` for new rows — effort S, reusing
   this session's exact fetch/parse code.
2. **`first_game_under_interim_by_game(schedules, parsed_table)`:** a
   pregame-safe per-game flag, same shape as `coach_fade_overlay.year_one_
   by_game` — already built this session as
   `nfl_ats.experiment_runner._build_interim_coach_trait_data` (would need
   promoting out of the experiment-runner's private namespace into a small
   shared module, e.g. `nfl_ats.interim_coach`, if wired for real).
3. **`apply_interim_first_game_overlay(predictions, schedules, parsed_
   table)`:** pick-level transform, same shape as `apply_coach_fade_
   overlay` — when the model's own pick does NOT already side with the
   interim-coached team in its first game, flip toward it (direction implied
   by the measured 58.97% cover rate). No week-9+ claim exists (unlike the
   year-1 overlay's own week 1-8 restriction, this cell has no week
   dependency by construction — it fires whenever a first interim game
   happens to fall, any week 1-18).
4. **Precedence with `hc_year_one_fade_overlay`:** measured 11 of 39
   historical first-games would ALSO have been eligible for the other
   overlay's weeks-1-8 window. Any real wiring needs an explicit rule for
   that collision (e.g. year-1 overlay takes precedence, since it has a
   larger sample and a resolved-if-still-unresolved history); not decided
   here.
5. **Recording:** same `nfl_ats.prospective_scoring` PASS/bet_side pattern
   `record_overlay_challenger_decisions` already uses, so both arms (with
   and without the flip) get tracked cleanly regardless of whether it is
   ever actually played.

This sketch is preserved as written; the "What was actually built" mapping
above is the current, accurate account of the wiring.

## Files

- `data/raw/interim_coaches/20260820T105234Z/` — `raw.html`,
  `response_headers.txt`, `parsed_table.csv` (52 rows), `manifest.json`
  (fetch provenance, join design, cross-checks, known caveats).
- `src/nfl_ats/experiment_runner.py` — `interim_hc_active`,
  `interim_hc_first_game`, `interim_hc_home`, `interim_hc_fired_year_one`
  flag builders (`FLAG_BUILDERS` registry), plus the shared join
  (`_build_interim_coach_trait_data`, `_interim_coach_team_game_table`).
- `registry/experiment_specs/interim_hc_active.json`,
  `interim_hc_active_excl_suspension.json`, `interim_hc_first_game.json`,
  `interim_hc_home_within_interim.json`, `interim_hc_fired_year_one.json`,
  `interim_hc_active_era_2009_2017.json`, `interim_hc_active_era_2018_2025.json`.
- `registry/weak_signals.json` — 7 new entries (see above), all
  `unresolved_below_power`.
- `src/nfl_ats/interim_hc_first_game_tilt_overlay.py` — the wired dual-tracked
  challenger (section 7): `interim_first_game_flag_by_game_fail_open`,
  `apply_interim_hc_first_game_tilt_overlay`, `overlay_disclosure_note`,
  `record_interim_hc_first_game_tilt_challenger_decisions`.
- `tests/test_interim_hc_first_game_tilt_overlay.py` — flag derivation,
  fail-open behavior, the pick-level flip (including the REG-only gate and
  the simultaneous-both-sides case), the disclosure note, and
  challenger-ledger recording.
- `artifacts/prospective/challengers.json` — `interim_hc_first_game_tilt_overlay`
  entry, status `ACTIVE_PROSPECTIVE`.
- `src/nfl_ats/cli.py` — `publish-predictions --record-decisions` now also
  calls `record_interim_hc_first_game_tilt_challenger_decisions`.
- `src/nfl_ats/public_board.py` — the challenger board's weekly preview
  dispatcher now includes `interim_hc_first_game_tilt_overlay`.
