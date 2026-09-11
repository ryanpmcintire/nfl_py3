# Late-season all-star crews (LEAD-33)

Binding closing-grounds taxonomy (verbatim, per the owner's session contract):
an interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds
ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
(whole interval on the wrong side of zero) or zero split-half reliability;
(2) bounded by a positive control proven able to detect an effect that
size. Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator.

## Status recap

`ROADMAP.md`'s LEAD-33 row previously stayed `🔬` (skipped) because
`officials.parquet` carries no all-star/Pro-Bowl/crew-designation column and
a prior session's own task explicitly ruled out inventing a
week-17/18-assignment marker as a substitute (`docs/officials_crew_leads.md`).
This session's task explicitly predeclares two admissible operational
constructions instead of a bare "late week" proxy -- a referee whose prior
season included a playoff assignment, or a regular-season crew reassembled
from multiple crews in weeks 15-18 -- and directs building one before
scoring. This session uses the first (prior-season playoff assignment),
which is directly measurable from real historical assignment records already
in `officials.parquet` (no invented marker).

## Data sources (read, not rebuilt)

- **Officials feed**: `data/raw/officials/20260819T190537Z/officials.parquet`,
  2015-2025, 21,900 rows, `position`/`season_type`/`season`/`week` columns
  already present per-row (no crosswalk needed for the playoff-history
  lookup). Loaded via `nfl_ats.officials_archive.load_officials(...,
  include_archive=False)` -- the nflverse feed only. **LEAD-59 status
  (read, `docs/officials_archive_probe.md` and the `ROADMAP.md` LEAD-59 row):
  2009-2014 is a separate Wayback-sourced archive, gated off by
  `INCLUDE_ARCHIVE_DEFAULT = False` because the crew traits this repo builds
  join to `game_penalties.parquet`, which is 2015-2025 only (turning the
  archive on is a measured no-op for every existing crew trait per the
  2026-09-08 LEAD-59 note). This session therefore uses 2015-2025 as the
  population, matching every other crew-trait screen in this repo, and does
  not touch `src/nfl_ats/` to change that.**
- **Penalty counts**: `data/raw/officials/20260819T190537Z/game_penalties.parquet`
  (3,028 games, `penalties_total`/`penalties_on_home`/`penalties_on_away`,
  standard `game_id` already).
- **Play counts**: `data/pbp/raw/20260817T184927Z/season=<year>/plays.parquet`,
  rows with `posteam` not null, grouped by `game_id` -- the same "plays"
  definition `nfl_ats.weak_stack_v3_features.team_season_penalty_rate`
  already uses.
- **Physical-underdog trait**: `data/processed/game_features_pbp.parquet`'s
  `home_pbp_off_pass_rate` / `away_pbp_off_pass_rate` -- an existing,
  prior-rolling, pregame-safe feature. `src/nfl_ats/experiment_runner.py`
  (read, lines 1440 and 1578) already treats the BOTTOM quartile of this
  pass rate as "run-heavy" for a directly analogous referee/penalty-type
  screen (`docs/penalty_crew_tendencies.md` cell C), so "physical underdog"
  reuses that exact convention rather than inventing a new one.
- **Production opener grade**: `nfl_ats.public_board.find_matching_opener_evaluation`
  -> `per_game.parquet` for the active model (`d49194e04945a5e5`, **measured**,
  `artifacts/opener_evaluation/20260910T211255Z`), 1,537 games, 2020-2025.
  This is the only season range the active model's opener evaluation covers,
  so the underdog cover-rate comparison and the production screen are both
  restricted to 2020-2025 (stated here, not hidden); the tightness
  comparison and the reliability check use the full 2015-2025 officials
  population instead, since they need no per-game production grade.
- **Standard schedule crosswalk**: `nfl_ats.snapshots.latest_snapshot` +
  `load_snapshot` (`data/raw/20260908T162105Z/schedules.parquet`),
  `old_game_id -> game_id` for REG games only.
- **Bootstrap harness**: `nfl_ats.clv.week_blocked_bootstrap` and
  `nfl_ats.clv.pick_correct` for the production screen (matching every other
  on-production candidate in this repo, e.g. `scripts/veteran_rest_day_tell.py`);
  a small self-contained percentile bootstrap (`np.random.default_rng`,
  20,000 draws) for the tightness/cover-rate/reliability comparisons, since
  those are not per-game ATS grades and do not need week-blocking. All draws
  use `nfl_ats.evidence_conventions.probability_positive_from_draws`.

Script: `scripts/late_season_crews_screen.py`. Run:
`.\.tools\uv.exe run --no-sync python scripts\late_season_crews_screen.py --season-start 2020 --season-end 2021`
(the `--season-start/--season-end` pair is the rotation-assigned window,
below). Artifact: `artifacts/experiments/late_season_crews/20260911T035551Z/`.

## (1) Operational definition, predeclared before scoring

**All-star crew** = a REG-season game in weeks 15-18 whose assigned Referee
(`position == "Referee"`, matching every other crew construct in this repo)
officiated at least one POSTSEASON game (`season_type` in
`{POST, WC, DIV, CON, SB}` -- the officials feed uses `POST` as an aggregate
label for the seasons before the league started publishing separate
`WC`/`DIV`/`CON`/`SB` rows, so both are included) **as Referee, in the
immediately preceding season**. Population restricted to seasons 2016-2025
(so every game has a full prior season on record; 2015 is excluded as the
population's first season). Direction, unchanged from the row: **BACK the
favorite** when the underdog is also a "physical" (run-heavy) team, defined
in (3) below.

**Measured population** (`late_season_crews_screen.py`, coverage block):
2,896 REG-season Referee rows crosswalk to a standard `game_id` at a
99.86% rate (2,892 matched -- reproducing `docs/referee_battery.md`'s own
figure exactly, confirming the loader). Of **556** weeks-15-18 REG games
with a matched Referee (2016-2025), **341 (61.3%)** are worked by an
all-star crew by this definition, spanning **24 of 29** distinct referees
who ever appear in the 2016-2025 archive (i.e. the large majority of active
referees pick up at least one playoff assignment across the study window).
**This construction is measurably broad, not a small elite subset** -- the
NFL runs 13 playoff games across four rounds through a pool of only
~17-19 active referees each season, so most regular officials rotate
through at least one playoff assignment somewhere in a season. Stated
plainly rather than softened: a reader should not picture "all-star crew"
as a rare few games.

## (2) Do they call tighter? Measured: no -- resolved wrong sign

Both comparisons below use "penalties per game" (`penalties_total`, both
teams combined) and "penalties per play" (`penalties_total /` PBP-derived
scrimmage-play count), with a 20,000-draw percentile bootstrap
(`np.random.default_rng`, seeds 20260910/20260911/20260912/20260913).

**Cross-sectional, weeks 15-18 only** (all-star vs. other crews, 341 vs.
215 games):

| Metric | All-star mean | Other mean | Diff | 95% CI | P+ (diff > 0) |
|---|---|---|---|---|---|
| Penalties/game | 10.865 | 11.726 | -0.860 | [-1.589, -0.128] | 0.010 |
| Penalties/play | 0.06608 | 0.07099 | -0.00491 | [-0.00913, -0.00076] | 0.011 |

**Paired, same referee's own weeks-1-14 rate that same season** (100
all-star referee-season units with >= 1 early-season game):

| Metric | Late (wk 15-18) | Early (wk 1-14) | Diff | 95% CI | P+ (diff > 0) |
|---|---|---|---|---|---|
| Penalties/game | 10.865 (subset mean) | 12.596 (subset mean) | -1.731 | [-2.223, -1.241] | 0.000 |
| Penalties/play | -- | -- | -0.00943 | [-0.01215, -0.00675] | 0.000 |

**All four intervals sit entirely on the negative side of zero.** The
row's predeclared mechanism ("hand-picked late-season crews call tighter")
is **measured wrong-signed**: all-star-flagged crews call FEWER penalties
in weeks 15-18 than other crews call in the same weeks, and fewer than
those SAME referees called earlier in the SAME season. This is a resolved
result by the binding taxonomy above (whole interval on the wrong side of
zero, corroborated by two independent comparisons and two units of
measurement) -- the stated MECHANISM is refuted. This is a descriptive
mechanism check, not itself an ATS/accuracy-unit signal, so per the
`docs/officials_crew_leads.md` LEAD-34 precedent (its crew-familiarity
frequency/penalty gap was reported the same way) it is not pushed through
`nfl-ats weak-signals record`, whose `--effect-units` enum has no "penalty
rate" unit -- the numbers are reported here in full instead, not softened
into prose that merely alludes to a doc.

## (3) Physical underdog cover rate under all-star vs. other crews, and the production screen

**Physical underdog, predeclared**: the underdog side (by the sign of
`tue_open_home_spread`; positive means home favored) whose prior-rolling
pregame `pbp_off_pass_rate` sits at or below the pooled home+away bottom
quartile computed over the full REG-season `game_features_pbp.parquet`
archive (cut = **0.5386**; below this a team is run-heavy relative to the
rest of the league). This is the same bottom-quartile convention
`src/nfl_ats/experiment_runner.py` already uses for "run-heavy" home teams
(read, lines 1440/1578, `docs/penalty_crew_tendencies.md` cell C), applied
here to whichever side (home or away) is the underdog and pooled across
both sides for a single, symmetric cut.

**Underdog cover rate** (2020-2025, the active model's opener-evaluation
population; 376 physical-underdog games):

| Crew | n | Underdog cover rate |
|---|---|---|
| All-star | 78 | 43.59% |
| Other | 298 | 56.04% |

Framed as the predeclared "back the favorite" edge (favorite win rate
under all-star minus under other crews, i.e. the sign flip of the raw
underdog-cover diff): **+12.451 accuracy points, 95% CI [+0.120, +24.781],
probability_positive 0.9755** (20,000-draw bootstrap, seed 20260914). The
whole interval sits on the predeclared side of zero -- directionally
**consistent** with the row's predicted pick effect, even though the
"calls tighter" mechanism measured in (2) is wrong-signed. This is not a
contradiction to paper over: whatever is producing the underdog
under-performance against all-star-flagged crews, it measurably is not
"more penalties called." A plausible confound not chased further this
session: playoff assignments may correlate with which TEAMS a referee is
staffed to (stronger, more nationally featured matchups), not with
in-game officiating behavior -- flagged here as an open question, not
resolved.

**Production screen** (stacked on the active model's opener pick;
`nfl_ats.clv.week_blocked_bootstrap`, block="week", 20,000 draws, seed
20260910). Trigger: `all_star_crew AND physical_underdog`; on a trigger
game the candidate always picks the favorite, otherwise it matches
production. Positive control (a same-population tautological oracle,
`margin_vs_open > 0`) is included to prove the harness is sensitive.

| Population | n games | n triggers | n flips | Production acc. | Candidate acc. | Candidate effect (pts) | 95% CI | P+ | Positive control (pts) | Control P+ |
|---|---|---|---|---|---|---|---|---|---|---|
| Full archive (2020-2025) | 1,537 | 80 | 26 | 54.558% | 54.291% | -0.266 | [-0.853, +0.267] | 0.169 | +45.442 | 1.000 |
| Assigned window (2020-2021) | 466 | 14 | 2 | 53.289% | 53.289% | 0.000 | [0.000, 0.000] | 0.500 | +46.711 | 1.000 |

The positive control confirms the harness detects a huge effect when one is
truly present (a tautological oracle beats production by ~45-47 points at
P+ 1.000 in both populations), so the flat candidate reading is not a
harness-insensitivity artifact -- it is a **measured, underpowered** result:
only 14 trigger games fall inside the rotation-assigned 2020-2021 window,
and the 2 resulting flips happened to net to exactly zero. **Both CIs
contain zero (the assigned window literally degenerates to a point at
zero); per the binding taxonomy this is the EXPECTED outcome for a real
small signal at this evaluator's resolution and is NOT grounds to reject
the candidate** -- it is recorded `unresolved_below_power`, and the EV-decision
rule (probability_positive vs. 0.5, never a significance bar) is what
should gate whether it is played, not this interval.

## (4) Split-half reliability (odd vs. even seasons) of the crew tightness trait

Is "how tight a referee calls" a repeatable, season-independent trait, or
is what was measured in (2) noise? For every referee with >= 5 REG games
in at least one odd season (2015/17/19/21/23/25) and >= 5 in at least one
even season (2016/18/20/22/24), pooled penalty rate (`sum(penalties_total)
/ sum(plays_from_pbp)`, and separately mean penalties/game) over all their
odd-season games vs. all their even-season games; Pearson r with a
20,000-draw referee-level bootstrap (seed 20260915/20260916).

| Metric | n referees | Pearson r | 95% CI | P+ |
|---|---|---|---|---|
| Penalties/play | 27 | 0.6414 | [0.3452, 0.8174] | 0.99985 |
| Penalties/game | 27 | 0.6895 | [0.4166, 0.8412] | 0.99990 |

**A referee's penalty-calling tendency is a real, strongly reliable
personal trait across disjoint season halves** (both intervals sit well
clear of zero, both measures agree). This matters for interpreting (2):
the "calls tighter" story is not refuted because tightness itself is
unmeasurable noise -- it is refuted because the SPECIFIC subgroup this
session's operational rule flags (playoff-experienced referees, in weeks
15-18) calls looser, not tighter, with a trait that is otherwise a stable
and real thing to measure.

## Weak-signal and rotation recording

Family `late_season_crews` (per this session's task; `--category onfield`
throughout). Two records fit the registry's `--effect-units` schema
(`correlation`, `accuracy_points`); the tightness mechanism check in (2)
does not (no "penalty rate" unit exists) and is reported above only,
matching the `docs/officials_crew_leads.md` LEAD-34 precedent for a
descriptive, non-ATS-unit comparison.

```
.\.tools\uv.exe run --no-sync nfl-ats weak-signals record `
  --name late_season_crew_tightness_reliability `
  --description "LEAD-33: odd-vs-even-season split-half reliability of a referee's penalty rate per play (2015-2025 REG games, officials.parquet crosswalked to game_penalties.parquet/PBP play counts)" `
  --source docs/late_season_crews.md `
  --effect 0.6414 --effect-units correlation `
  --interval-low 0.3452 --interval-high 0.8174 `
  --probability-positive 0.99985 `
  --classification unresolved_below_power `
  --league nfl --season-start 2015 --season-end 2025 `
  --sample-games 2892 --sample-blocks 27 `
  --family late_season_crews --category onfield `
  --plain-summary "Some referees really do call a tighter or looser game than others, and that tendency holds up from one set of seasons to another." `
  --notes "Not a closing ground either way -- reliability is strongly positive but the registry has no confirmed-positive bucket, only unresolved_below_power / refuted_mechanism / bounded_by_control."

.\.tools\uv.exe run --no-sync nfl-ats weak-signals record `
  --name physical_underdog_cover_under_all_star_crew `
  --description "LEAD-33: physical (run-heavy) underdog cover rate under all-star (prior-season-playoff-referee) crews vs. other crews, weeks 15-18, 2020-2025 opener-graded population, framed as the favorite-side edge" `
  --source docs/late_season_crews.md `
  --effect 12.4505 --effect-units accuracy_points `
  --interval-low 0.1205 --interval-high 24.7806 `
  --probability-positive 0.9755 `
  --classification unresolved_below_power `
  --league nfl --season-start 2020 --season-end 2025 `
  --sample-games 376 --sample-blocks 105 `
  --family late_season_crews --category onfield `
  --plain-summary "When a run-heavy underdog draws a crew that worked the playoffs last year, that underdog covers less often than it does under any other crew." `
  --notes "Whole interval on the predeclared side of zero, but the calls-tighter mechanism in (2) is wrong-signed for the same crews, so this is reported as an open, unresolved directional finding, not a confirmed mechanism."

.\.tools\uv.exe run --no-sync nfl-ats weak-signals record `
  --name all_star_crew_underdog_on_production `
  --description "LEAD-33: on-production opener screen, BACK the favorite when all_star_crew AND physical_underdog, rotation-assigned window 2020-2021" `
  --source docs/late_season_crews.md `
  --effect 0.0 --effect-units accuracy_points `
  --interval-low 0.0 --interval-high 0.0 `
  --probability-positive 0.5 `
  --classification unresolved_below_power `
  --league nfl --season-start 2020 --season-end 2021 `
  --sample-games 466 --sample-blocks 35 `
  --family late_season_crews --category onfield `
  --plain-summary "Backing the favorite when a run-heavy underdog draws a playoff-tested crew was too rare an event in the current test window (14 games, 2 actual pick changes) to move the needle either way." `
  --notes "Full 2020-2025 archive read (not itself a rotation spend): -0.266 pts [-0.853, +0.267], P+ 0.169, 80 triggers/26 flips of 1,537 games. Positive control (tautological oracle) confirms harness sensitivity: +45.4 to +46.7 pts, P+ 1.000, both populations."

.\.tools\uv.exe run --no-sync nfl-ats rotation record `
  --name all_star_crew_underdog_on_production `
  --artifact artifacts/experiments/late_season_crews/20260911T035551Z/results.json `
  --verdict unresolved `
  --probability-positive 0.5 `
  --effect 0.0 --effect-units accuracy_points `
  --interval-low 0.0 --interval-high 0.0 `
  --sample-blocks 35 `
  --notes "Assigned window 2020-2021, opener grade: 466 games, 14 triggers, 2 flips, net zero accuracy delta. Full 2020-2025 archive (not a separate spend): -0.266 pts [-0.853,+0.267] P+0.169. Positive control +45.4/+46.7 pts P+1.000 confirms sensitivity. Per AGENTS.md an interval containing zero is not grounds to close; EV-decision rule (P+ vs 0.5, never a significance bar) should gate whether this is played going forward."
```

## What was NOT done / open follow-ups

- The row's OTHER named construction ("a referee's regular-season crew
  reassembled from multiple crews in weeks 15-18") was not built this
  session; the prior-season-playoff-assignment construction was used
  instead and is reported in full above. A future session could build the
  crew-reassembly statistic and compare it against this one (LEAD-59's
  2026-09-08 note already mentions a "crew-scramble statistic" computed in
  a research harness for a related lead, -0.33 standalone / +0.20 through
  the played card, neither committed to `src/`).
- The underdog cover-rate finding in (3) is directionally consistent with
  the row's predicted PICK effect despite the MECHANISM in (2) being
  wrong-signed; a plausible confound (which teams/games get staffed with
  playoff-tested crews) is named but not investigated further this
  session.
- 2009-2014 (LEAD-59's Wayback archive) is not used; 2015-2025 is the
  stated population throughout, per `INCLUDE_ARCHIVE_DEFAULT = False` and
  the archive-on no-op finding already on record for existing crew traits.
