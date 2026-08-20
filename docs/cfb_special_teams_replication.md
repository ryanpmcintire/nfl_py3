# PBP-06 cross-league replication: CFB special-teams return + net-punting

Written and frozen 2026-08-19, **before any CFB cover-rate sign has been
computed**. Follows the `docs/surface_switch_tilt_overlay.md` /
`scripts/cfb_surface_familiarity_screen.py` precedent for wiring an NFL lead
plus an independent CFB replication (ROADMAP `ENV-02` row). The feasibility
audit below (which trait dimensions can be built at all from local CFB PBP,
and which seasons are usable) WAS run first and is the one predeclared,
admissible exception to "no look before the doc is frozen" -- identical to
the `special_teams_battery.md` / `team_style.md` reliability-gate precedent --
but no punt/return cover-rate has been computed for any CFB game.

## Binding taxonomy (owned verbatim, per AGENTS.md/CLAUDE.md)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with `nfl-ats
weak-signals record`, report `probability_positive`, never the binary
"contains zero". The registry code hard-rejects inadmissible closures; if a
record command errors, the verdict is wrong, not the validator. Every cell
below is recorded to the registry regardless of sign or interval shape.

## What is being replicated

Two NFL PBP-06 cells (`registry/weak_signals.json`, both **read** this
session, both `unresolved_below_power`, seasons 2009-2025, n=8,634):

| NFL entry | Effect (accuracy pts) | Week-blocked 95% CI | P+ | Reliability (YoY r) |
|---|---|---|---|---|
| `special_teams_return_top_quartile` | +0.4986 | [-0.0742, +1.0797] | 0.9547 | componentwise +0.109 (punt) / +0.158 (kickoff) |
| `special_teams_punt_net_top_quartile` | -0.3890 | [-0.9674, +0.2038] | 0.0946 | +0.313 |

`special_teams_return_top_quartile` is today's strongest new NFL lead: teams
whose PRIOR-SEASON punt-return and kickoff-return yardage (mean of the two
z-scored, "return_composite_z") sit in the top quartile across the league
cover their spread more often. `special_teams_punt_net_top_quartile` is its
**inverse-lean sibling in the same battery**: predeclared sign was +1 (top
net-punting teams predicted to outperform, mirroring the return cell's own
predeclared direction), but the measured NFL result leans the OTHER way
(P+ 0.0946) despite `punt_net_yards` having the STRONGEST year-over-year
reliability of the four kept dimensions in the battery (+0.313). That
combination -- reliable trait, wrong-leaning cover result -- is exactly the
kind of surprising pattern a cross-league replication is built to stress-test:
if CFB shows the same split (return positive, punt-net negative-or-null), the
pattern is not an NFL sampling artifact; if CFB disagrees, the NFL reads are
more likely noise around zero on both cells.

This replication spends **no NFL rotation-registry window** -- CFB is this
project's sanctioned free cross-league replication ground (rule 8).

## Feasibility audit (MEASURED, run first, before any cover-rate sign)

**Source**: local CFB PBP snapshots only (`data/cfb/pbp/raw/*/season=*/plays.parquet`,
the XLG-02 55-column canonical contract table, never the full-column
upstream file). No fresh CFBD PBP fetch was made. Snapshots span 2004-2025
across six separate ingestion runs (different UTC timestamps hold different
season partitions); all were read directly.

**No `kick_distance`/`return_yards`/`touchback` columns exist in the CFB
canonical contract** (confirmed by reading `CFB_PBP_SNAPSHOT_COLUMNS`,
`src/nfl_ats/cfb.py` line 363) -- the CFB table is even narrower than the NFL
`PBP_SNAPSHOT_COLUMNS` special-teams gap the NFL battery's own predeclaration
described. Two things the CFB table DOES carry substitute for it:

1. **`statYardage`** is a clean, pre-extracted numeric yardage-of-the-play
   column (not a text field) that on a punt/kickoff-return row already equals
   the return yardage, and on a fair-catch/touchback/downed/out-of-bounds row
   is 0. MEASURED (2009 season, kickoff-return rows only, `n=6,169`
   candidate real returns): a naive text-regex parse of "returned ... for N
   yards" **disagrees with `statYardage` on 40% of rows** -- not because
   `statYardage` is wrong, but because the free-text sentence has a "tackled
   by X" clause between "returned by" and the yardage number in that era's
   phrasing, which a simple regex anchored on "returned...for" cannot skip.
   MEASURED fix-check: restricting to the ~13 rows where a corrected parse
   (loss-of-N-yards sign handling) actually disagreed with `statYardage`,
   the true error rate is 13/6,169 = 0.2%, i.e. `statYardage` was right and
   the text-regex needed more work, not the other way around. **Conclusion:
   `statYardage` is used as the return-yardage ground truth directly; text is
   used only for classification (touchback/fair catch/downed/out-of-bounds/
   blocked/has-return), never for parsing the return-yardage NUMBER.** This
   is a stronger data foundation than a naive replication would have used.
2. **`text`** (free-form play description) is used to (a) classify a
   punt/kickoff row into touchback / fair-catch / downed / out-of-bounds /
   blocked / real-return (substring search, not numeric parsing -- robust
   across every era and text-format variant sampled), and (b) parse the KICK
   DISTANCE, which has no numeric column and must come from text (e.g. "punt
   for 38 yds" / "punt 44 yards to the MIA39" -- two different upstream feed
   formats coexist even within recent seasons). MEASURED kick-distance parse
   rate, one season sampled per era: 2004 0% (see below), 2005 98.2%, 2009
   98.6%, 2013 98.9%, 2014 98.9%, 2018 98.6%, 2022 98.0%, 2023 98.9%, 2024
   99.0%, 2025 99.1% (a regex covering both the "for N yards" and "N yards
   to" phrasings was required to reach the 2025 figure; the narrower NFL-style
   pattern alone parsed only 64% of 2025 punts before the fix).

**2004 is unusable and excluded.** MEASURED: the 2004 CFB PBP snapshot has
only 144 rows classified as a punt play type for the entire season (real CFB
seasons run ~7,000-8,700 punts) and a 0% kick-distance parse rate on those
144 -- the season's `type.text`/`text` special-teams detail is effectively
absent, not merely thin. This is worse than `docs/cfb_data.md`'s documented
"2004-2013 thinner early ESPN coverage" characterization for 2004
specifically. **2005-2025 (21 seasons) are usable** -- kick-distance parse
rates cluster at 98-99% across every sampled season in that range, and
`statYardage` (the return-yardage ground truth) has no missingness issue at
all, since it is a genuine numeric column, not a text-derived one.

**Touchback field-position cap.** The NFL punt-net formula caps a touchback's
net value at `yardline_100 - 20`. CFB's `start.yardsToEndzone` (the
punting team's distance to the opponent's end zone at the snap) is the exact
analogue of NFL `yardline_100` and is **100% populated on every touchback
punt row sampled** across all nine probed seasons (2005-2025). NCAA punt
touchbacks are spotted at the 20-yard line (unlike NCAA kickoff touchbacks,
moved to the 25 in 2012 -- this battery only touches punts for the net-yards
trait, so the kickoff touchback rule change does not enter this formula), so
the CFB cap uses the same `- 20` constant as the NFL formula, verbatim.

**Team identity.** CFB PBP carries ESPN numeric team ids (`pos_team_id`
kicking team, `def_pos_team_id` receiving team) rather than name strings, and
the canonical benchmark table `data/processed/cfb_game_features.parquet`
already carries `home_id`/`away_id` in the SAME id space. MEASURED: every one
of the 134 team ids appearing in the 2024 clean-core game-features rows
resolves inside the 2024 PBP's team-id set (134/134, 0 unresolved) -- no name
join, no relocation-alias table needed (ESPN ids are stable; unlike NFL
abbreviation codes, CFB programs are not observed to change id on a
conference move in this window).

**Trailing-basis coverage.** The task's population is the XLG-03 clean core
(2012-2019 + 2021-2025); its trailing prior seasons are 2011-2018, 2020, and
2024 -- all inside the usable 2005-2025 range, so **no clean-core season is
truncated for missing special-teams history** (unlike the NFL battery, which
drops 2009 for lacking a 2008 prior season within its own declared range).
Per-team-season missingness (a team fielding zero usable punts/returns in
its prior season, e.g. a first-year FBS transition or a season with heavy
text-parse gaps) is measured and reported at build time, not assumed zero.

**Blocked kicks.** MEASURED: blocked-punt play text is the messiest variant
sampled (concatenated duplicate fragments, inconsistent distance mentions).
The net-yards formula is applied uniformly to every punt-type row exactly as
the NFL script applies it (no special-case exclusion), consistent with the
NFL convention of a blocked kick counting as a real (typically very low or
negative) net outcome; a distance that fails to parse on a blocked-punt row
falls out as missing, not zero-defaulted.

## Trait definitions (ported from `docs/special_teams_battery.md`, CFB data substituted)

Built by `scripts/cfb_special_teams_screen.py`. Every raw dimension is pooled
directly from underlying play-level attempts at the team-season grain (never
averaged from a lower granularity), then centered against its own season's
unweighted team mean (era-drift removal, identical convention).

| dimension | definition |
|---|---|
| `punt_net_yards` | `kick_distance - statYardage` (return yards) for a normal punt, capped at `start.yardsToEndzone - 20` on a touchback. Applies to every punt-type play (`Punt`, `Blocked Punt`, `Punt Return Touchdown`, `Punt Team Fumble Recovery`, `Blocked Punt Touchdown`, `Punt (Safety)`), grouped by the KICKING team (`pos_team_id`). |
| `punt_return_yards` | mean `statYardage` on punts with a genuine return (excludes fair catch, touchback, downed, out-of-bounds, blocked, and any row without the word "return" in its text), grouped by the RETURNING team (`def_pos_team_id`). |
| `kickoff_return_yards` | mirror of the above for kickoff-type plays (`Kickoff`, `Kickoff Return (Offense)`, `Kickoff Return Touchdown`, `Kickoff Team Fumble Recovery`), excludes touchback/fair catch/downed/out-of-bounds. |

Composite: `return_composite_z` = mean of (`punt_return_yards_z`,
`kickoff_return_yards_z`), each z-scored by the pooled standard deviation of
its centered value across the full 2005-2025 CFB team-season panel -- the
identical construction `special_teams_screen.py::add_composites` uses for
NFL. `punt_net_yards_z` is the single centered-and-z-scored dimension for the
punt-net cell, no composite.

**Trailing basis**: prior FULL REGULAR season (`seasonType` code `"2"`,
matching `CFB_PBP_SEASON_TYPE_CODES`), not trailing-N-games -- identical
convention to the NFL battery and to `cfb_opponent_adjustment.py`'s own
point-in-time contract. A team's first tracked season (2005) has no prior
season and is excluded (missing, not defaulted) from every cell, as is any
team-season with zero usable punts or zero usable returns in the prior
season.

## Predeclared cells (2, matching the task's stated scope exactly)

**Population**: the XLG-03 CFB clean core, reused verbatim, not redeclared --
`data/processed/cfb_game_features.parquet` restricted to
`nfl_ats.cfb_benchmark.CFB_CLEAN_CORE_SEASONS` (2012-2019 + 2021-2025),
`home_cover` not null (pushes/missing dropped). **MEASURED this session: this
yields exactly 8,933 games**, matching `docs/cfb_opponent_adjustment.md`'s
own stated headline sample size exactly. Unlike
`scripts/cfb_surface_familiarity_screen.py`'s population, **neutral-site
games are KEPT** (267 of the 8,933) -- a team's return/net-punting skill is
not a home-venue-contingent fact the way modal home surface is, and the task
brief names the 8,933-game population explicitly, which only matches with
neutral sites included (8,933 measured with them in; 8,666 would result from
excluding them, which does not match).

**Design**: team-perspective long table (one row per team per game,
`team_covered`), the exact pattern `scripts/special_teams_screen.py::build_long_table`
uses for NFL -- a special-teams trait belongs to whichever of the two
competing teams has it, home or away.

**Method**: joint week-blocked bootstrap (block = `season*100+week`) PRIMARY,
season-blocked bootstrap SECONDARY, `block_bootstrap_two_group`-identical to
`scripts/special_teams_screen.py` / `scripts/cfb_surface_familiarity_screen.py`
(vectorized joint multinomial block resample). Full-slate effect scaling via
`nfl_ats.experiment_runner.scale_subset_effect` (imported directly, not
reimplemented, matching `special_teams_screen.py`'s own convention) --
`sign * raw_gap_fraction * 100 * fraction_of_slate`, `accuracy_points` units.
20,000 bootstrap samples, seed 20260819, fixed and deterministic.

Quartile threshold (0.75 quantile, TOP quartile only -- the task's stated
scope is these two top-quartile cells, not their bottom-quartile mirrors) of
each dimension across the full CFB 2005-2025 team-season panel, computed
ONCE and reused, matching the NFL anti-drift convention.

**Predeclared signs, matching the NFL cells' own predeclared direction
exactly (not the NFL cells' measured/revealed direction)**:

1. **`cfb_special_teams_return_top_quartile`** -- top-quartile teams by
   prior-season `return_composite_z` (mean z of punt-return and
   kickoff-return yards) vs the field. Sign: **+1**, identical to the NFL
   cell's own predeclared sign. Mechanism: a team's return unit generates
   field position a market model built primarily on offense/defense
   efficiency plausibly underweights.
2. **`cfb_special_teams_punt_net_top_quartile`** -- top-quartile teams by
   prior-season `punt_net_yards_z` vs the field. Sign: **+1**, identical to
   the NFL cell's own predeclared sign (even though the NFL MEASURED result
   leans the opposite way at P+ 0.0946 -- the predeclaration mirrors the
   NFL predeclaration, not the NFL finding, exactly as any honest
   replication must).

## What counts as replication (stated before the run)

- **Sign match on `cfb_special_teams_return_top_quartile`**: CFB point
  estimate positive (regardless of whether its interval excludes zero) counts
  as directional replication of the NFL lead, matching the
  `surface_switch_tilt_overlay` precedent's own reading (that document called
  a CFB read "replicating" on sign-plus-comparable-magnitude even with both
  intervals crossing zero).
- **Sign match on `cfb_special_teams_punt_net_top_quartile`**: CFB point
  estimate negative (or a P+ decisively below 0.5) would replicate the NFL
  battery's surprising inverse-lean pattern; a CFB point estimate positive
  (or P+ near/above 0.5) would NOT replicate it, meaning the NFL punt-net
  read looks more like sampling noise specific to that league/window.
- Neither outcome, in either direction, is grounds to close either line --
  see the taxonomy above. Both cells are recorded regardless of sign.

## Not built here

No CFB analogue of `special_teams_fg_kicker_*`, `special_teams_composite_edge_*`,
or either trait's BOTTOM-quartile mirror is built -- the task's stated scope
is exactly these two top-quartile cells. A future session could extend this
battery the same way the NFL one was extended, but that is a new
predeclaration, not implied by this one.

## Files

- `scripts/cfb_special_teams_screen.py` -- feature build + screen, single
  script (team-season aggregation is small enough not to warrant a separate
  features script the way the NFL battery split fetch/build from screen).
- This document, frozen before any cover-rate sign was computed.
- Results and the two registry recordings (`nfl-ats weak-signals record
  --league cfb`), reported in the session's final summary.
