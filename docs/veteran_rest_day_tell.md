# LEAD-11: veteran rest-day tell

Read (`ROADMAP.md:978`): predeclared direction is that a Wednesday DNP
labelled rest or not-injury-related (NIR) for a 30+ veteran proxies a hidden
limitation rather than pure load management, so teams leaning on rest-day
veterans should be FADED (their market line lags the decline). A prior pass
(2026-09-05, CX14, `docs/injury_trajectory_leads.md`) recorded this construct
at **zero covered team-games** because its NFL.com source carried no birth
dates (the existing age axis in the codebase, `src/nfl_ats/age_curves.py`, is
career experience, not chronological age) and no Wednesday/Thursday/Friday
revision sequence. This session rebuilds the construct on different sources
per this lane's task brief and gets non-zero coverage throughout.

## Binding verdict taxonomy (verbatim, per session instructions)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
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

## Data sources (read/measured)

- Injury reason text and practice/report status: **measured**, latest
  `data/raw/nflverse_injuries/20260910T203021Z/injuries.parquet` (90,891
  rows, `nflreadpy.load_injuries`). This table carries exactly one row per
  player-week -- **measured** (`groupby(["season","week","team","gsis_id"]).size()`
  on the canonical copy returns 90,887 groups of size 1 and only 2 of size 2
  across the whole 2009-2026 archive) -- so it is the FINAL observed report
  for that player-week, not a stored Wednesday/Thursday/Friday sequence.
  `date_modified`'s weekday is a genuine signal, not an ingest artifact:
  **measured**, among the 3,083 rest/NIR-tagged rows league-wide, a
  Wednesday-final row is DNP 407 times, Limited 152, Full only 32, while a
  Friday-final row is DNP 383, Limited 129, Full 843 -- exactly the shape
  expected if the archive keeps whichever day's status was the LAST one
  filed that week. This is why the flag (below) is built from the row's own
  final `practice_status` plus its reason text, not from requiring the
  timestamp to literally read Wednesday.
- Point-in-time basis: **measured**, latest
  `data/players/raw/20260910T205112Z/injuries.parquet` carries
  `observed_at_basis`/`observed_at_is_proxy` per row (manifest:
  `injury_proxy_hours_before_kickoff: 24`, `injury_timestamp_fallback:
  "week_proxy"`). Real timestamps cover 2011-2024 at 89-100% per season;
  2025-2026 are proxy-only (a week-anchored fallback, not a leaked
  post-game value) and 2009-2010 are partly proxy. The report row itself is
  always pregame information (it is the league's own pregame disclosure)
  regardless of proxy status, so the proxy flag governs the WEEKDAY framing
  disclosed above, not the pregame validity of the flag.
- Chronological age: **measured**, `nflreadpy.load_players()` (24,823
  players, `gsis_id`/`birth_date`/`pfr_id`, cached to
  `data/players/raw/20260911T020107Z/players.parquet`,
  sha256 `15949a57b84e206cd68d87975eb6a35318b01774d5810570cab0eeb1454d589c`).
  **Measured**: 32 of 24,823 players (0.13%) have a null `birth_date`
  league-wide; among the 3,045 rest/NIR-tagged rows in 2019-2026, **zero**
  are missing a birth date. This is a materially better source than CX14's
  NFL.com feed (which had zero birth-date coverage) and directly answers
  this lane's brief to check `data/players/` for a roster/player table
  before assuming the career-experience axis was the only option.
- Snap shares and starter status: **measured**, latest
  `data/players/raw/20260910T205112Z/snap_counts.parquet` (2013-2025,
  `pfr_player_id` keyed) and `weekly_rosters.parquet` (roster/status,
  `gsis_id`+`pfr_id`, both usable for the gsis-to-pfr crosswalk).
- Schedules/kickoffs/team codes: **measured**, `nfl_ats.snapshots.latest_snapshot`
  (`data/raw/20260908T162105Z`, 4,902 REG+POST games back to 2009). Team
  codes are canonicalised with `nfl_ats.constants.TEAM_ABBREVIATION_ALIASES`
  (OAK->LV, SD->LAC, STL->LA) before every join.
- Production baseline and opener grade: **measured**,
  `nfl_ats.public_board.find_matching_opener_evaluation` against the active
  model (`d49194e04945a5e5`), resolving to
  `artifacts/opener_evaluation/20260910T211255Z/per_game.parquet` (1,537
  REG games, 2020-2025). `pick_home_at_open_probability_rule` /
  `correct_at_open_probability_rule` is used as PRODUCTION, matching the
  convention every sibling LEAD-03..LEAD-17 row uses ("stacked on
  PRODUCTION `weak_stack`") -- the model's own opener probability-rule pick,
  not a re-derivation of the full nine-member weather/arrest/coach overlay
  composition (`src/nfl_ats/four_overlay_composition.py`), which needs
  inputs (weather forecasts, protection-mismatch flags) this lane's brief
  did not ask for and which is itself a moving target (the three-member
  union was retired 2026-09-09, one day before this session:
  `RETIRED_THREE_MEMBER_CHALLENGER_ID = "overlay_three_member_union_retired_20260909"`
  in `src/nfl_ats/four_overlay_composition.py`).

Script: `scripts/veteran_rest_day_tell.py`. No code comments or docstrings
(repo rule); no new test files (moratorium) -- validated by direct runs of
the script, not a pytest file.

## 1. The flag

Predeclared population and construct, adapted to what the single-row-per-week
archive can actually support (disclosed above): a player-week is **flagged**
when (a) the injury row's reason text (`report_primary_injury`,
`report_secondary_injury`, `practice_primary_injury`,
`practice_secondary_injury`) matches `\brest(?:ed|ing)?\b|load management`
(case-insensitive) -- this catches "Rest", "Rested", "Resting Veteran/Vet",
"Not injury related - resting player", "Load Management", and excludes
other NIR reasons that are not rest (personal matter, travel, coach's
decision, discipline, suspension-return); (b) the player is chronologically
**30 or older** at that game's kickoff; (c) the row's own final
`practice_status` is **not** "Did Not Participate In Practice" -- i.e., by
the time the archive's one snapshot for that week was taken, the player had
progressed to Limited or Full participation. Given the archive keeps
whichever day was reported LAST, (c) is the operationalisation of "then
practising later in the week": a rest-tagged row whose final state is still
DNP means the absence was never resolved that week (see the weekday
crosstab above), which is the opposite of what this lead's mechanism needs.

Measured (`scripts/veteran_rest_day_tell.py`, REG season only, all team
codes canonicalised):

| Season | Injury-report rows | Rest/NIR-tagged rows | ...age 30+ | ...missing birth date | **Flagged player-weeks** | Real-timestamp share |
|---:|---:|---:|---:|---:|---:|---:|
| 2009 | 4,596 | 1 | 0 | 0 | 0 | 0.004 |
| 2010 | 4,324 | 23 | 13 | 0 | 12 | 0.885 |
| 2011 | 4,748 | 0 | 0 | 0 | 0 | 0.898 |
| 2012 | 5,266 | 0 | 0 | 0 | 0 | 0.919 |
| 2013 | 4,913 | 0 | 0 | 0 | 0 | 0.921 |
| 2014 | 4,912 | 0 | 0 | 0 | 0 | 0.911 |
| 2015 | 5,009 | 0 | 0 | 0 | 0 | 0.924 |
| 2016 | 4,928 | 2 | 1 | 0 | 1 | 0.948 |
| 2017 | 4,949 | 8 | 6 | 0 | 5 | 0.969 |
| 2018 | 4,961 | 1 | 0 | 0 | 0 | 0.966 |
| 2019 | 5,202 | 52 | 43 | 0 | 30 | 0.973 |
| 2020 | 5,414 | 23 | 18 | 0 | 3 | 1.000 |
| 2021 | 5,348 | 506 | 388 | 0 | 169 | 1.000 |
| 2022 | 5,450 | 631 | 358 | 0 | 208 | 1.000 |
| 2023 | 5,451 | 478 | 302 | 0 | 120 | 1.000 |
| 2024 | 5,954 | 608 | 383 | 0 | 249 | 1.000 |
| 2025 | 5,783 | 593 | 415 | 0 | 288 | 0.000 |
| 2026 (partial) | 139 | 6 | 3 | 0 | 1 | 1.000 |

**Measured, not inferred:** the rest/NIR-tagged reason text is essentially
absent before 2019 (0-25 rows/season) and jumps an order of magnitude from
2021 onward (506-631/season). **Inferred:** this reads as a reporting/tagging
change (nflverse's practice-injury text field becoming more granular),
not a real change in how often veterans are rested, since the underlying
football practice of resting veterans predates 2021. Because of this, the
outcome-comparison and reliability sections below restrict to 2021-2025,
where the construct has real, comparable coverage every season; the full
2009-2026 table above is reported for transparency, per this project's
"label how you know it" rule, not cherry-picked to hide the pre-2021 gap.

## 2. On-field outcome: flagged veterans vs. matched non-flagged veterans

Population: 2021-2025 (five seasons), REG only, age >= 30, active roster
(`weekly_rosters.status == "ACT"`), player has a `pfr_player_id` crosswalk
(via `weekly_rosters.pfr_id` or `nflreadpy.load_players().pfr_id`).
"Production drop" = that week's snap-share (`max(offense_pct, defense_pct)`)
minus the player's own trailing mean share over up to the prior 4 games that
season (strictly before the game -- pregame-safe, though this section is a
descriptive outcome check, not a pregame feature). "Missed within 4 weeks" =
the player's team played a game in weeks w+1..w+4 that season and the player
recorded zero snaps or no row at all in that game.

**Measured, raw comparison** (flagged vs. every other rostered 30+ veteran
that week, unmatched on role):

| Group | n player-weeks | Mean same-week snap share | Mean production drop (n) | 4-week miss rate (n) |
|---|---:|---:|---|---|
| Flagged | 1,022 | 0.7267 | -0.00375 (991) | 0.200 (955) |
| All other 30+ veterans | 20,282 | 0.4680 | +0.00282 (17,780) | 0.480 (19,091) |

Flagged veterans look nothing like the average 30+ roster player on the raw
cut -- **inferred**: this is a selection effect, not a finding. Only players
important enough to warrant a scheduled rest day get this reason text in the
first place, so the "all other veterans" pool is diluted by backups and
fringe roster players who churn on and off rosters for unrelated reasons
(that is almost certainly why their raw 48% four-week miss rate is so high).

**Measured, starter-matched comparison** (both groups restricted to players
whose OWN trailing snap share was >= 0.5, i.e., comparing rested starters to
other non-rested starters, not to the whole roster):

| Group | n player-weeks | Mean same-week snap share | Mean production drop | 4-week miss rate |
|---|---:|---:|---:|---:|
| Flagged starters | 790 | 0.8139 | **-0.0154** | **0.195** |
| Other 30+ starters (not flagged) | 8,381 | 0.7931 | **-0.0273** | **0.251** |

Once matched to comparable workload, flagged veterans show a **smaller**
production drop from their own baseline and a **lower** four-week miss rate
than other starting-caliber 30+ veterans -- the opposite of the predeclared
"hidden limitation" direction. Split by season half:

| Half | Flagged production drop | Control production drop | Flagged miss rate | Control miss rate |
|---|---:|---:|---:|---:|
| Odd (2021, 2023, 2025) | -0.0161 (n=437) | -0.0305 (n=5,167) | 0.180 (n=410) | 0.259 (n=4,881) |
| Even (2022, 2024) | -0.0145 (n=353) | -0.0221 (n=3,214) | 0.214 (n=327) | 0.239 (n=3,033) |

**Measured**: the direction (flagged veterans decline less and miss less
than matched controls) replicates in both halves; the gap is larger in the
odd half than the even half. **Inferred**: read together with the
production screen below, the more plausible mechanism is that a Wednesday
rest day is, on the whole, doing what it says -- protecting a valuable
veteran's remaining-season production -- rather than masking a decline the
market has not priced. This does not prove the predeclared mechanism is
false (see the closing-grounds test below); it is a directional read against
it from the properly matched cut, and it is consistent with the production
screen's own negative lean.

### Reliability: is "resting veterans" a stable team trait?

Measured (`nfl_ats.evidence_conventions` bootstrap convention, 20,000 draws,
seed 20260910): split 2021-2025 into odd seasons (2021/23/25) and even
seasons (2022/24), compute each of the 32 teams' rate of team-games with at
least one flagged veteran in each half, and correlate the two halves across
teams.

**Pearson r = 0.5633, 95% [0.285, 0.768], probability_positive = 0.99985,
n = 32 teams.**

This clears the "no split-half reliability" bar by a wide margin: a team's
propensity to rest-tag its 30+ veterans is a real, stable team-season trait,
not sampling noise. It rules out `no_split_half_reliability` as a closing
ground for this family (see the taxonomy above); it does not by itself say
anything about whether the trait is predictive of a spread edge, which is
what section 3 tests.

## 3. Production screen

Team-game feature: for each team-game, sum the **trailing snap share**
(the same up-to-4-game trailing mean used above, so this is pregame-safe --
never the current game's own snaps) of every FLAGGED veteran whose trailing
share is also >= 0.5 (a starter gate). This is the "value-weighted count of
flagged starters" the task specifies. `home_flagged`/`away_flagged` = that
side's summed value > 0. The candidate flips PRODUCTION's
`pick_home_at_open_probability_rule` toward the side WITHOUT the flag only
when production currently backs the flagged (fade-worthy) side -- the same
flip rule used by the retired CX14 injury-trajectory screens and by
`src/nfl_ats/four_overlay_composition.py`'s fade overlays: `flip = (home_flagged
!= away_flagged) & (production_pick_home == home_flagged)`.

Rotation: **measured** (`nfl-ats rotation declare` + `assign`), a fresh
family `veteran_rest_day_tell_on_production` (opener grade,
acknowledges_mined_2018_2025) drew window **[2020, 2021]** -- the same
window LEAD-62 used, exactly as this lane's task brief asked to prefer,
and the registry did not refuse it.

Bootstrap: `nfl_ats.clv.week_blocked_bootstrap`, 20,000 draws, seed
20260910, `probability_positive_from_draws` (the shared, zero-atom-credited
helper every registry entry uses).

**Measured, assigned window (2020-2021, 466 opener-graded games / 35
weeks) -- the recorded confirmation cell:**

| | Production | Candidate |
|---|---:|---:|
| Accuracy | 53.29% | 50.66% |
| Forced picks flipped | -- | 39 / 466 |

Candidate effect: **-2.6316 accuracy points, week-blocked 95% [-5.765,
+0.222], probability_positive = 0.038225.**

**Measured, full 2020-2025 opener archive (1,537 games) -- descriptive
context only, NOT the confirmation cell** (these seasons were never assigned
to this family; using them as if they were the test would be exactly the
"iterate on the sample until it resolves" behaviour this project's window
discipline exists to prevent, per `docs/injury_value_lost.md` section 7 and
the rotation-registry rules): production 54.56% vs. candidate 51.76%, delta
**-2.7944 accuracy points, 95% [-4.573, -1.002], probability_positive =
0.00105**. This descriptive interval happens to sit wholly below zero, but
it is not treated as a resolved wrong sign here because it mixes seasons
this family never spent -- the assigned-window cell above is the one that
governs the registry classification.

Positive control (candidate replaced by the realized margin sign, perfect
foresight, same window/blocking): **+46.71 accuracy points,
probability_positive = 1.0** -- the harness is proven able to detect a large
effect on this exact population before this screen's own small effect is
read.

### Closing-grounds check (binding taxonomy applied, not asserted)

- `wrong_sign_resolved`: **inadmissible.** The assigned-window interval
  [-5.765, +0.222] crosses zero. The full-archive interval does not cross
  zero, but per the reasoning above it is descriptive, not the confirmation
  cell, so it is not used to claim this ground.
- `no_split_half_reliability`: **inadmissible.** Section 2's reliability
  read (r=0.5633, 95% [0.285, 0.768]) is decisively non-zero.
- `positive_control_bound`: **not established.** The positive control shows
  the harness CAN detect a large effect; it does not show the harness was
  tested against and failed to detect an effect this candidate's own size.
- **Classification: `unresolved_below_power`.** Recorded exactly that way
  in both registries (below). Per this project's EV rule, `probability_positive
  = 0.038225` on the confirmation cell favours retaining production over
  playing this fade screen -- that is a legitimate, low-P+, EV-based
  decision, not a claim that the underlying "rest masks decline" mechanism
  has been refuted. It stays available to revisit with a different
  construction (e.g. position-stratified value weights, or a longer
  trailing window) without needing a positive-control bound to reopen it,
  since nothing here closes it.

## Recorded

Measured (`nfl-ats rotation declare`, `nfl-ats rotation assign`, `nfl-ats
rotation record`):

```
nfl-ats rotation declare --name veteran_rest_day_tell_on_production \
  --description "LEAD-11 veteran rest-day tell: ..." --grade opener \
  --acknowledge-mined --plain-summary "Checks whether teams that rest ..."
nfl-ats rotation assign --name veteran_rest_day_tell_on_production
  -> assigned window [2020, 2021]
nfl-ats rotation record --name veteran_rest_day_tell_on_production \
  --artifact artifacts/experiments/veteran_rest_day_tell/20260911T020731Z/results.json \
  --verdict unresolved --probability-positive 0.038225 \
  --effect -2.6315789473684212 --effect-units accuracy_points \
  --interval-low -5.764966740576496 --interval-high 0.22172949002217296 \
  --sample-blocks 35 --notes "..."
  -> family veteran_rest_day_tell_on_production window [2020,2021] state=spent verdict=unresolved
```

Measured (`nfl-ats weak-signals record`):

```
nfl-ats weak-signals record --name veteran_rest_day_tell_on_production \
  --description "Value-weighted count of flagged 30+ veteran starters ..." \
  --source artifacts/experiments/veteran_rest_day_tell/20260911T020731Z/results.json \
  --effect -2.6315789473684212 --effect-units accuracy_points \
  --classification unresolved_below_power --league nfl \
  --season-start 2020 --season-end 2021 \
  --interval-low -5.764966740576496 --interval-high 0.22172949002217296 \
  --probability-positive 0.038225 --sample-games 466 --sample-blocks 35 \
  --reliability 0.5632573223589222 --family veteran_rest_day_tell \
  --category health --classification-evidence "..." --plain-summary "..." --notes "..."
  -> classification unresolved_below_power, effect -2.6316 accuracy_points,
     favours_candidate false, registry total_signals 5810
```

## Artifacts

- `scripts/veteran_rest_day_tell.py` -- flag build, outcome comparison,
  reliability, production screen, and registry-artifact writer.
- `artifacts/experiments/veteran_rest_day_tell/20260911T020731Z/results.json`
  -- full measured output (season flag table, outcome comparison including
  the starter-matched and odd/even cuts, reliability, both screen cells).
- `artifacts/experiments/veteran_rest_day_tell/20260911T020731Z/flags.parquet`,
  `team_value.parquet` -- underlying per-player-week and per-team-game
  tables.
- `data/players/raw/20260911T020107Z/players.parquet` -- cached
  `nflreadpy.load_players()` snapshot (birth dates, static biographical
  data, never inferred from performance) used for age.

## What this session did not do

- Did not stratify the team-value feature by position group (e.g. weighting
  a rested left tackle differently from a rested slot receiver); the
  trailing-share weight already captures workload but not positional
  importance.
- Did not attempt a longer trailing-share window than 4 games or a
  different starter threshold; per this project's rule against iterating a
  construction until it resolves, the first predeclared construction is
  reported as measured, not re-tuned toward a better-looking number.
- Did not reconstruct the full nine-member overlay composition
  (`four_overlay_composition.py`) as the production baseline; used the same
  base probability-rule pick convention every sibling LEAD-03..LEAD-17 row
  uses, disclosed above.
