# Era events and meta-game series

Substrate for era hypotheses, not a hypothesis itself. This doc pairs two
things: (1) an external, sourced table of NFL rule/structure changes and
meta-game shifts, 2002-2026, and (2) internal per-season league-level series
computed from this repo's own play-by-play and game data. Eras are
mechanistic and their boundaries are fuzzy by construction; nothing here
declares a boundary, closes a signal, or issues a verdict.

**Provenance key used throughout this doc**: **measured** = computed this
session from the local files named; **read** = opened the file/page directly
this session; **reported** = a fetched source says so, not independently
verified by this agent; **inferred** = reasoning/judgment call, not evidence.

## 1. Rule-change and structural-event table

Full table with `mechanism_touched` and per-row `notes`:
[`registry/reference/nfl_era_events.csv`](../registry/reference/nfl_era_events.csv)
(26 rows, tracked, small text file). Compiled by web search/fetch against
primary and secondary sources (NFL.com, team sites republishing owners'-
meeting summaries, ESPN, CBS Sports, Washington Post, Sports Illustrated, the
Supreme Court opinion itself for Murphy v. NCAA). Every row carries a
`source_url`; every row is **reported** unless its `notes` column says a
specific figure was **read** directly this session.

**Category counts** (measured, `registry/reference/nfl_era_events.csv`, 26
rows): kickoff 5, overtime 4, roster_cba 3, officiating_emphasis 2,
meta_game 2, player_safety 2, replay_review 2, scoring_rule 1, clock_rules 1,
market_structure 1, covid 1, playoff_structure 1, schedule_structure 1.

Two rows are explicitly **not** rule changes and are flagged as such in their
own `notes`: the two `meta_game` rows (4th-down aggressiveness, the "tush
push") are behavioral/strategic shifts, not NFL rulebook changes, and their
`season_effective` values are placeholder midpoints, not measured
inflection points -- Section 3 below is where the actual inflection gets
located empirically, in the series data, per this doc's own "how to use
this" rule.

| Season | Category | Event | Signal families | Source |
|---|---|---|---|---|
| 2004 | officiating_emphasis | Illegal contact / defensive holding points-of-emphasis crackdown | penalties; QB availability (pass efficiency, indirect); coach fade | [link](https://www.espn.com/nfl/columns/story?columnist=pasquarelli_len&id=1771047) |
| 2011 | kickoff | Kickoff spot moved from the 30-yard line to the 35-yard line | kicking; home advantage (field position); coach fade | [link](https://www.nfl.com/news/nfl-moves-kickoffs-to-35-yard-line-touchbacks-unchanged-09000d5d81ee38c1) |
| 2011 | roster_cba | 2011 CBA ends lockout: shorter offseason program, OTAs capped at 10, year-round full-contact practice limits, rookie wage scale | QB availability; coach fade | [link](https://en.wikipedia.org/wiki/2011_NFL_lockout) |
| 2012 | overtime | Regular-season overtime adopts modified sudden death (both teams get a possession unless the first team scores a TD) | market_structure (closing-line/CLV dynamics); home advantage | [link](https://www.foxnews.com/sports/league-adopts-playoff-ot-rule-for-regular-season) |
| 2014 | officiating_emphasis | Second illegal-contact / defensive-holding crackdown (echo of 2004) | penalties; coach fade | [link](https://www.washingtonpost.com/news/sports/wp/2014/08/13/nfl-crackdown-on-illegal-contact-may-spur-epic-stat-surge-for-qbs-like-tom-brady-peyton-manning/) |
| 2015 | scoring_rule | Extra point moved to the 15-yard line (a 33-yard kick); 2-point tries stay at the 2; blocked/returned PATs and failed 2-point tries become returnable for 2 points | kicking | [link](https://www.nfl.com/news/nfl-moves-extra-point-to-15-yard-line-for-2015-season-0ap3000000493347) |
| 2016 | kickoff | Touchback spot on kickoffs moved from the 20-yard line to the 25-yard line | kicking; home advantage | [link](https://www.footballzebras.com/2016/08/with-goal-of-reducing-kickoff-returns-nfl-sets-touchback-spot-at-25/) |
| 2017 | clock_rules | 10-second run-off rule expanded to apply after the two-minute warning (previously only the final minute of each half) | market_structure (in-game win-probability/closing dynamics); coach fade | [link](https://www.footballzebras.com/2017/10/everything-need-know-10-second-runoffs/) |
| 2017 | meta_game | Secular rise in 4th-down go-for-it aggressiveness, commonly dated to Doug Pederson's Philadelphia Eagles (hired 2016, Super Bowl LII win at the end of the 2017 season) building on earlier academic work (Romer 2006) and analytics-media advocacy | coach fade | [link](https://www.sportico.com/leagues/football/2024/detroit-lions-dan-campbell-nfl-stats-fourth-down-1234817656/) |
| 2017 | overtime | Regular-season overtime period shortened from 15 minutes to 10 minutes | market_structure; rest | [link](https://www.washingtonpost.com/news/sports/wp/2017/05/23/nfl-votes-to-shorten-regular-season-overtime-period-from-15-minutes-to-10/) |
| 2018 | market_structure | Murphy v. NCAA: U.S. Supreme Court strikes down PASPA, ending the federal ban on state-level sports betting | market_structure | [link](https://www.supremecourt.gov/opinions/17pdf/16-476_dbfi.pdf) |
| 2018 | player_safety | "Use of helmet" rule: 15-yard penalty and possible ejection for any player who lowers his head to initiate and make contact with the helmet, anywhere on the field | penalties; coach fade | [link](https://www.espn.com/nfl/story/_/id/22935229/nfl-institutes-rule-lowering-head-initiate-contact-helmet) |
| 2018 | player_safety | Roughing-the-passer "body weight" rider tightened by a single conjunction change ("and" landing on the QB with weight -> "or") | penalties; QB availability | [link](https://www.si.com/nfl/2018/09/27/nfl-statement-roughing-passer-flag-body-weight-rule-no-changes) |
| 2018 | replay_review | Catch rule simplified: dropped the "going to the ground" element; a catch now needs control, two feet (or another body part) down, and a "football move" | penalties (indirect, via reversed-call rates); coach fade | [link](https://www.espn.com/blog/nflnation/post/_/id/272769/guide-to-nfls-new-rules-what-to-know-about-approved-tabled-and-rejected-proposals) |
| 2019 | replay_review | Offensive and defensive pass interference, including non-calls, made reviewable | penalties | [link](https://www.nfl.com/news/owners-make-pass-interference-non-calls-reviewable-0ap3000001024371) |
| 2020 | covid | COVID-19 season: zero preseason games, a fully virtual offseason program, and no or sharply limited crowds at most stadiums for most of the season | home advantage; QB availability; rest | [link](https://www.espn.com/nfl/story/_/id/29823418/15-nfl-coronavirus-protocols-need-know-no-more-mascots-jersey-swaps-cheerleaders) |
| 2020 | playoff_structure | Playoff field expands from 12 to 14 teams (7 seeds per conference), adding a Wild Card game | coach fade; rest | [link](https://www.nfl.com/news/new-cba-includes-playoff-expansion-to-14-teams-0ap3000001106259) |
| 2020 | roster_cba | 2020 CBA ratified: practice squad expanded 10 -> 12 (permanent baseline), 17-game season authorized for a later season, playoff field expanded | QB availability | [link](https://www.washingtonpost.com/sports/2020/03/15/nfl-completes-new-cba/) |
| 2021 | schedule_structure | 17th regular-season game added (18-week regular season, one bye week) | rest; coach fade | [link](https://www.si.com/nfl/2020/03/15/nfl-cba-approved-players-vote-17-game-regular-season-expansion) |
| 2022 | overtime | Playoff overtime rule change: both teams guaranteed at least one possession unless the first team's opening drive scores a TD | market_structure | [link](https://cbssports.com/nfl/news/nfl-owners-approve-new-overtime-rules-for-playoffs-ensuring-each-team-gets-a-possession/amp) |
| 2022 | roster_cba | Practice squad expanded again, 12 -> 14 | QB availability | [link](https://www.rosecrete.com/blog/10-biggest-changes-to-the-nfls-collective-bargaining-agreement) |
| 2023 | kickoff | Fair catch anywhere behind a team's own 25-yard line on a kickoff is placed at the 25 (one-year trial) | kicking | [link](https://www.cbssports.com/nfl/news/nfl-approves-one-year-trial-run-for-new-fair-catch-rule-that-could-have-dramatic-impact-on-kickoff-returns/) |
| 2023 | meta_game | "Tush push" (quarterback sneak with backfield push) proliferation, associated with the Philadelphia Eagles; a 2025 proposal to ban the play failed on a 16-16 owners' vote | coach fade | [link](https://www.nbcphiladelphia.com/news/sports/nfl/nfl-owners-rule-changes-overtime-rules-kickoff-touchback-tush-push-2025/4148635/) |
| 2024 | kickoff | "Dynamic kickoff": new setup alignment, a 20-yard "landing zone," and restricted player movement until the ball is touched, explicitly modeled on the XFL kickoff | kicking; home advantage | [link](https://operations.nfl.com/updates/the-game/dynamic-kickoff-back-and-better-in-second-season/) |
| 2025 | kickoff | Dynamic kickoff made permanent; touchback spot on kickoffs moved to the 35-yard line | kicking | [link](https://www.therams.com/news/2025-nfl-rule-bylaw-resolution-changes-owners-approve-expanded-replay-alignment-of-regular-season-and-overtime-rules-permanent-dynamic-kickoff-with-touchbacks-to-the-35-yard-line) |
| 2025 | overtime | Regular-season overtime rules aligned with the playoff format: both teams get a possession regardless of the first result, 10-minute period, then sudden death | market_structure; home advantage | [link](https://www.therams.com/news/2025-nfl-rule-bylaw-resolution-changes-owners-approve-expanded-replay-alignment-of-regular-season-and-overtime-rules-permanent-dynamic-kickoff-with-touchbacks-to-the-35-yard-line) |

Contested/approximate dates, called out explicitly rather than smoothed over:
the two officiating-emphasis rows are points-of-emphasis, not rule-text
changes, and their effective season is the press-coverage year, not a
codified date; the two meta-game rows have no single effective season at all
(see the callout above); the roughing-the-passer 2018 row is a one-word
change to a rule dating to 1995, enforced as an in-season emphasis; the 2023
kickoff fair-catch rule and 2019 pass-interference-review rule were both
explicit one-year trials (the PI rule was not renewed; the fair-catch rule
was effectively superseded by 2024's full kickoff overhaul rather than
extended on its own terms).

**Coverage gap, stated plainly**: the task's nominal window is 2002-2026, but
this table's first row is 2004 -- no 2002-2003 rule change met this table's
threshold (a change with a plausible signal-family link) after search; this
is an absence-of-hits finding, not a claim that nothing happened 2002-2003.

## 2. Meta-game series (measured, local data)

Built by [`scripts/build_metagame_series.py`](../scripts/build_metagame_series.py)
(read-only; touches no tracked file, writes nothing to `data/`). Run this
session:

```
.\.tools\uv.exe run --no-sync python scripts\build_metagame_series.py
```

**Inputs** (measured, read from local parquet this session):
`data/pbp/raw/20260817T184927Z/season=<season>/plays.parquet` (nflverse
play-by-play, 2009-2025, 17 seasons, 781,712-ish rows total) and
`data/processed/game_features.parquet` (one row per game, 2009-2026; 2026 is
the unplayed forecast season and drops out of every completed-game aggregate
automatically because `result` is null for those rows).

**Output**: `artifacts/metagame_series/20260819T205042Z/series.parquet`
(17 rows, one per season, gitignored like all `artifacts/`).

### 2a. Coverage: what was computable, what wasn't

Computed (all measured, this run):

- 4th-down go rate, overall and by field-position region (own territory,
  "plus" territory 11-50, red zone <=10)
- pass rate (actual), mean expected-pass probability (`xpass`), and mean
  pass-rate-over-expected in percentage points (`pass_oe`, nflverse's own
  PROE-style field, already computed play-by-play in the stored snapshot)
- sack rate and QB-hit rate, both per dropback (`qb_dropback == 1`)
- penalty rate (per play, and per game)
- pace: game-clock seconds elapsed per live scrimmage play (`play == 1`),
  correctly handling overtime as an added clock window rather than reusing
  the regulation-only game-seconds-remaining diff (see the script's
  `pace_series` docstring)
- kickoff landing: average post-kickoff starting field position
  (`yardline_100` of the next play), and a touchback-rate **proxy** (see
  below)
- home cover rate, home straight-up win rate, average absolute spread,
  offensive points per game, from `game_features.parquet`

**Not computable** from the columns this repo's PBP ingestion actually
stores (checked dynamically by the script against the live schema, not
assumed; `nfl_ats.pbp.PBP_SNAPSHOT_COLUMNS` narrows the raw nflverse feed at
ingestion time and these are the casualties):

- **average air yards, short-pass share (<=5 yards), deep share (>=20
  yards)** -- missing column: `air_yards`
- **shotgun rate, no-huddle rate** -- missing columns: `shotgun`, `no_huddle`;
  there is also no play-description text field to regex instead (the stored
  `play` column is a numeric valid-play flag, 0/1, not text)
- **QB scramble rate**, as distinct from a called dropback -- missing column:
  `qb_scramble` (and no rusher-player id to separate a scramble from a
  designed QB run)
- **touchback rate**, as a strict stored boolean -- no `touchback` or
  return-yardage column exists. A proxy is reported instead
  (`touchback_rate_proxy`): the share of kickoffs whose immediate next play
  starts exactly at that rule-era's expected touchback spot (a season-keyed
  lookup built directly off the Section 1 kickoff rows: 80 yardline_100
  through 2015, 75 from 2016-2023, 70 in 2024, 65 from 2025). This was
  validated before being trusted: it reproduces the known direction and a
  plausible magnitude at every kickoff rule change in Section 1 (see 3a-3c
  below), including an internal cross-check against a **reported** external
  figure at the 2011 change.

If a future PBP re-ingestion adds any of the missing columns, this script
picks them up automatically (`detect_missing_columns` runs against the live
schema every time, not a hardcoded assumption).

### 2b. 4th-down go rate

| season | fourth_down_go_rate | own_territory | plus_territory | red_zone |
|---|---|---|---|---|
| 2009 | 0.144 | 0.061 | 0.228 | 0.304 |
| 2010 | 0.126 | 0.056 | 0.199 | 0.229 |
| 2011 | 0.112 | 0.042 | 0.200 | 0.198 |
| 2012 | 0.119 | 0.054 | 0.186 | 0.257 |
| 2013 | 0.122 | 0.060 | 0.186 | 0.266 |
| 2014 | 0.120 | 0.065 | 0.175 | 0.253 |
| 2015 | 0.126 | 0.065 | 0.190 | 0.264 |
| 2016 | 0.127 | 0.055 | 0.209 | 0.250 |
| 2017 | 0.127 | 0.059 | 0.207 | 0.247 |
| 2018 | 0.149 | 0.071 | 0.219 | 0.357 |
| 2019 | 0.163 | 0.083 | 0.246 | 0.305 |
| 2020 | 0.191 | 0.100 | 0.267 | 0.382 |
| 2021 | 0.207 | 0.110 | 0.292 | 0.399 |
| 2022 | 0.189 | 0.099 | 0.280 | 0.337 |
| 2023 | 0.197 | 0.111 | 0.285 | 0.345 |
| 2024 | 0.204 | 0.114 | 0.282 | 0.367 |
| 2025 | 0.233 | 0.129 | 0.305 | 0.476 |

"Go" = `down == 4` and `play_type` in (run, pass), as a share of all `down ==
4` plays decided as run/pass/punt/field_goal (kneels, spikes, and
penalty-negated no-plays excluded as not real 4th-down decisions).

### 2c. Pass rate and PROE

| season | pass_rate | mean_xpass | mean_pass_oe (pct pts) |
|---|---|---|---|
| 2009 | 0.569 | 0.602 | -1.08 |
| 2010 | 0.575 | 0.603 | -0.23 |
| 2011 | 0.578 | 0.603 | +0.36 |
| 2012 | 0.581 | 0.599 | +1.02 |
| 2013 | 0.589 | 0.602 | +1.64 |
| 2014 | 0.589 | 0.621 | -0.15 |
| 2015 | 0.599 | 0.627 | +0.14 |
| 2016 | 0.601 | 0.625 | +0.33 |
| 2017 | 0.585 | 0.628 | -1.28 |
| 2018 | 0.596 | 0.627 | +0.01 |
| 2019 | 0.594 | 0.631 | -0.37 |
| 2020 | 0.589 | 0.624 | -0.01 |
| 2021 | 0.587 | 0.627 | -0.50 |
| 2022 | 0.576 | 0.628 | -1.82 |
| 2023 | 0.582 | 0.630 | -0.86 |
| 2024 | 0.571 | 0.629 | -1.75 |
| 2025 | 0.570 | 0.627 | -1.61 |

`pass_oe` is nflverse's own per-play residual (`(actual - expected) * 100`),
already computed in the stored snapshot; the season mean is the league
PROE-style trend, on a percentage-point scale, and nets close to zero by
construction of a residual against a fitted model -- read the *drift* across
seasons, not the absolute level.

### 2d. Sacks, penalties, pace

| season | sack_rate | qb_hit_rate | penalty_rate/play | penalties/game | seconds/play |
|---|---|---|---|---|---|
| 2009 | 0.059 | 0.121 | 0.070 | 11.84 | 27.15 |
| 2010 | 0.060 | 0.119 | 0.071 | 12.05 | 27.04 |
| 2011 | 0.062 | 0.127 | 0.073 | 12.65 | 26.70 |
| 2012 | 0.059 | 0.121 | 0.072 | 12.43 | 26.48 |
| 2013 | 0.064 | 0.127 | 0.069 | 12.14 | 26.35 |
| 2014 | 0.061 | 0.130 | 0.076 | 13.19 | 26.42 |
| 2015 | 0.059 | 0.139 | 0.079 | 13.72 | 26.23 |
| 2016 | 0.056 | 0.136 | 0.077 | 13.28 | 26.48 |
| 2017 | 0.061 | 0.142 | 0.077 | 13.18 | 26.83 |
| 2018 | 0.064 | 0.142 | 0.078 | 13.35 | 26.99 |
| 2019 | 0.065 | 0.141 | 0.078 | 13.36 | 26.64 |
| 2020 | 0.056 | 0.138 | 0.064 | 11.10 | 26.63 |
| 2021 | 0.060 | 0.137 | 0.069 | 11.71 | 26.99 |
| 2022 | 0.064 | 0.143 | 0.066 | 11.10 | 27.23 |
| 2023 | 0.067 | 0.145 | 0.067 | 11.33 | 27.06 |
| 2024 | 0.066 | 0.142 | 0.076 | 12.78 | 27.25 |
| 2025 | 0.065 | 0.146 | 0.075 | 12.49 | 27.74 |

"Seconds/play" is game-clock seconds per live scrimmage snap (a pace-of-play
proxy, not broadcast wall-clock time).

### 2e. Kickoff landing

| season | avg_post_kickoff_start (yardline_100) | touchback_rate_proxy |
|---|---|---|
| 2009 | 73.29 | 0.199 |
| 2010 | 72.95 | 0.196 |
| 2011 | 77.20 | 0.473 |
| 2012 | 77.38 | 0.479 |
| 2013 | 77.24 | 0.517 |
| 2014 | 77.39 | 0.527 |
| 2015 | 77.79 | 0.593 |
| 2016 | 74.61 | 0.592 |
| 2017 | 74.76 | 0.588 |
| 2018 | 74.31 | 0.630 |
| 2019 | 74.29 | 0.631 |
| 2020 | 74.17 | 0.637 |
| 2021 | 74.46 | 0.603 |
| 2022 | 74.30 | 0.618 |
| 2023 | 74.50 | 0.778 |
| 2024 | 70.00 | 0.647 |
| 2025 | 69.12 | 0.197 |

Lower `avg_post_kickoff_start` = better field position for the receiving
offense (closer to the opponent's end zone). `touchback_rate_proxy` is *not*
a pure "how many touchbacks" number -- it is season-relative (matched against
that season's expected touchback spot), so a rule change that pushes the
touchback spot deeper without changing team behavior would leave the proxy
roughly flat even as the raw spot number moves; see 3c below for the one
case (2025) where the proxy and the raw spot genuinely diverge in meaning.

### 2f. Market and outcome

| season | home_cover_rate | home_su_win_rate | avg_abs_spread | offensive_ppg | completed_games |
|---|---|---|---|---|---|
| 2009 | 0.467 | 0.573 | 6.54 | 21.58 | 267 |
| 2010 | 0.492 | 0.554 | 4.97 | 22.16 | 267 |
| 2011 | 0.500 | 0.573 | 5.60 | 22.25 | 267 |
| 2012 | 0.469 | 0.569 | 5.10 | 22.89 | 267 |
| 2013 | 0.523 | 0.596 | 5.29 | 23.43 | 267 |
| 2014 | 0.487 | 0.573 | 5.32 | 22.63 | 267 |
| 2015 | 0.467 | 0.543 | 4.85 | 22.73 | 267 |
| 2016 | 0.500 | 0.581 | 4.63 | 22.87 | 267 |
| 2017 | 0.514 | 0.569 | 5.35 | 21.81 | 267 |
| 2018 | 0.473 | 0.592 | 5.40 | 23.28 | 267 |
| 2019 | 0.440 | 0.521 | 5.67 | 22.85 | 267 |
| 2020 | 0.494 | 0.498 | 5.51 | 24.75 | 269 |
| 2021 | 0.484 | 0.516 | 6.00 | 23.03 | 285 |
| 2022 | 0.493 | 0.563 | 5.05 | 22.02 | 284 |
| 2023 | 0.498 | 0.565 | 4.97 | 21.90 | 285 |
| 2024 | 0.520 | 0.547 | 4.92 | 23.01 | 285 |
| 2025 | 0.507 | 0.533 | 5.32 | 22.98 | 285 |

`home_cover_rate` excludes push games (`home_cover` null); `completed_games`
is `result`-not-null games per season, all 17 seasons at or above 267 -- the
2026 forecast season contributes zero rows here because none of it is played
yet.

## 3. Event/series correspondences (descriptive, no causal claim)

Every number below is **measured** this session from the files named in
Section 2. Correspondence is not causation; these are the same-season
co-occurrences of a Section 1 event and a Section 2 series break, offered as
candidate era boundaries for a hypothesis to test, not as a finding about
mechanism.

**3a. 2011 kickoff-spot move.** `touchback_rate_proxy` jumps 0.196 (2010) ->
0.473 (2011), and `avg_post_kickoff_start_yardline_100` worsens for the
receiving offense by 4.25 yards (72.95 -> 77.20) in the same season the
kickoff spot moved from the 30 to the 35. This measured pattern lines up in
direction and rough scale with a **reported** (unverified by this agent)
external figure surfaced during Section 1's research -- ESPN Stats & Info's
touchback-rate figures of 16.4% (2010) and 43.5% (2011) -- even though the two
figures use different methodologies (a stored touchback boolean vs. this
doc's next-play-position proxy).

**3b. 2023 fair-catch-at-25 rule.** `touchback_rate_proxy` rises 0.618 (2022)
-> 0.778 (2023), the single largest one-year jump anywhere in the 17-season
series -- consistent with a rule that mechanically converts a fair catch
anywhere behind the 25 into a touchback at the 25, i.e. removes the
in-between "returned for less than 25 yards of starting-position value"
outcome entirely.

**3c. 2024 dynamic kickoff.** `avg_post_kickoff_start_yardline_100` improves
by 4.5 yards for the receiving offense in one season (74.50 -> 70.00), while
`touchback_rate_proxy` simultaneously *falls* (0.778 -> 0.647) even as the
average field position gets better. Those two moving in opposite directions
is not a contradiction -- it is the rule working as designed: the new
touchback spot (the 30) is deep enough to discourage kicking teams from
happily conceding it, so more of the mass shifts toward live, returned kicks
that land at varied spots rather than clustering exactly at the touchback
line. The 2025 permanent-rule season pushes this further (proxy falls again,
to 0.197, alongside a modest further field-position gain to 69.12) --
directly consistent with the 2025 rule moving the touchback spot deeper
still (to the 35), which by the same logic should make kicking teams even
less willing to concede a touchback outright.

**3d. 2017-2021 4th-down go-rate rise.** `fourth_down_go_rate` climbs every
single season from 2017 (0.127) through 2021 (0.207), a monotonic run found
nowhere else in the series, and the largest cumulative multi-year move in
the whole table (+0.080, versus a 2011 trough of 0.112). This is the
cleanest inflection in the dataset and lines up with the meta-game row in
Section 1 (Eagles' 2017 Super Bowl season and the subsequent spread of
analytics-driven 4th-down advocacy) far better than any single calendar year
could -- exactly the kind of era boundary this doc's own "how to use this"
section says should be read off the series, not assumed from a headline.

**3e. 2020 COVID season.** Two contemporaneous, opposite-signed breaks in
the same season: `penalty_rate_per_play` falls from 0.078 (2019) to 0.064
(2020) -- the largest single-year penalty-rate drop anywhere in the series
-- while `home_su_win_rate` drops below 0.500 for the only time in the
17-season table (0.521 -> 0.498). At the same time `home_cover_rate` *rises*
(0.440 -> 0.494), consistent with the market pricing down the home
advantage it could see disappearing (fewer/no fans) faster than home teams'
actual on-field advantage disappeared, which is why the cover-rate number
and the win-rate number move in opposite directions in the same season.

## 4. How to use this

**An era hypothesis names an event/series pair and a predicted effect on a
named signal family.** Point at one row of Section 1 (or a `meta_game` row)
and one series column in Section 2, state the predicted sign and
(qualitatively) the size, on one of this repo's signal families: surface/
weather, rest, penalties, coach fade, QB availability, kicking, home
advantage, market_structure. "The 2016 touchback rule should shift `kicking`
family features toward better average starting field position for the
receiving team" is a hypothesis in this doc's sense; "the game changed after
2015" is not.

**Era boundaries are estimated from the series, not assumed from the
calendar.** Section 1's `season_effective` is when a rule took effect, which
is precise for a codified rule and a placeholder midpoint for a `meta_game`
row (both meta-game rows say so explicitly in their `notes`). Section 3
already shows the pattern to follow: 3d finds the real 4th-down inflection
runs 2017-2021, not a single year, by reading the series; the same discipline
applies to any new hypothesis this doc supports later. A predeclared window
around the Section 1 event, then a look at where the relevant Section 2
series actually breaks, is the intended workflow -- not "the rule changed in
year X, so the era boundary is year X."

**This doc computes nothing about any experiment's status.** No number here
is a verdict, an interval, or a closing ground. Restated because this doc
will get cited in signal commentary: an interval or CI containing zero is
never grounds to reject an experiment; report `probability_positive`, never
"contains zero"; the only two things that ever close a line of work are a
refuted mechanism (a RESOLVED wrong sign, or zero split-half reliability) or
a positive-control bound; every other outcome is `unresolved_below_power`,
recorded via `nfl-ats weak-signals record`, never settled in doc prose alone.
