# Playcaller-change leads (LEAD-29 first-game bump, LEAD-28 post-bye x new playcaller)

Predeclared 2026-09-07 (lane R) BEFORE any cover outcome was read. Family
`lead29_playcaller_change_v1` for every cell below; distinct signal ids per
cell. This is a SCREEN in the shape of `docs/interim_coach_screen.md`: it
spends no rotation window, wires no overlay or challenger, and changes no
played card. Every claim carries its provenance tag (measured / read /
reported / inferred) per `AGENTS.md`.

## Closing-grounds taxonomy (verbatim, binding, applies to every experiment you run or judge)

An interval or CI that contains zero is NEVER grounds to reject, fail, or close an experiment. At this
evaluator's ~2-point resolution, "contains zero" is the EXPECTED outcome for a real small signal. Only two
grounds ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (2) bounded by a positive control proven able to
detect an effect that size. Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the binary "contains zero". The
registry code hard-rejects inadmissible closures; if a record command errors, the verdict is wrong, not
the validator. Never use 95%, 0.90, or any threshold as a DECISION bar; decide on expected value
(`probability_positive` above 0.5 favours playing it), thresholds only govern what docs may CLAIM.
Grade at the OPENER (the pool's grade); a close-graded number may never veto a play. Within-week game
correlation is ZERO by owner mandate: never estimate or pad it. Never say something "needs N more
games": the data is fixed and the project is model-limited.

## Source

**Read** (`docs/coordinator_history_source.md`, `docs/coordinator_change_on_production.md`):
`data/raw/coordinators/20260907T213814366437Z/coordinator_history.parquet`
holds dated HC/OC/DC assignments for all 544 team-seasons 2009-2025, each
observation carrying its real Wikipedia staff-template revision timestamp
(`effective_observed_at`, basis `wikipedia_revision`), sampled at a
September 1 cutoff (`sample_mode == "preseason"`) plus every in-season
revision 2022-2025 (`sample_mode == "inseason"`). **Measured**
(`pandas.read_parquet` this session): 2,976 rows, 1,566 preseason, 1,410
in-season; preseason OC rows cover 28-32 teams per season (28 in 2009, 32 in
2012, 2013, 2015, 2025), at most one preseason OC row per team-season, and
every preseason observation is at or before its `sampled_as_of` cutoff.
`change_validation.json` in the same directory adjudicates the 16 in-season
OC/DC identity edits into `staff_change` (10), `playcaller_role` (2),
`role_correction` (1), `reverted_identity_edit` (2) and
`unverified_identity_edit` (1).

**Information boundary.** Every in-season event's boundary is the revision
instant (`revision_at`, UTC), never midnight of that day. A game counts as
"after" an event only if the event's revision instant is strictly before
the game's decision timestamp. The decision timestamp used here is the
game's kickoff (`game_features.parquet`'s UTC `kickoff` column), as the lane
brief specifies. All twelve counted revision instants fall on a Monday,
Tuesday or Wednesday, so the pool's own earlier per-game deadline
(min(kickoff, Sunday 16:00 ET), `AGENTS.md` memory) would not move any
flag; this is checked mechanically by the script (it reports the minimum
lead time from revision instant to the flagged kickoff).

## R1: `playcaller_first_game` (LEAD-29)

**Events counted** (**read** from `change_validation.json`; both
`staff_change` and `playcaller_role` adjudications; the `role_correction`,
`reverted_identity_edit` and `unverified_identity_edit` rows are excluded):

| # | Season | Team | Side | Change (previous -> new) | Revision instant (UTC) | Adjudication |
|---|---|---|---|---|---|---|
| 1 | 2022 | CAR | defence | Phil Snow -> Al Holcomb | 2022-10-10 19:33:25 | staff_change |
| 2 | 2022 | IND | offence | Marcus Brady -> Parks Frazier | 2022-11-09 15:46:43 | playcaller_role |
| 3 | 2023 | BUF | offence | Ken Dorsey -> Joe Brady | 2023-11-14 16:40:27 | staff_change |
| 4 | 2023 | LV | offence | Mick Lombardi -> Bo Hardegree | 2023-11-01 15:13:06 | staff_change |
| 5 | 2023 | PIT | offence | Matt Canada -> Eddie Faulkner (Mike Sullivan calls plays) | 2023-11-21 18:02:37 | staff_change |
| 6 | 2024 | CHI | offence | Shane Waldron -> Thomas Brown | 2024-11-12 15:21:40 | staff_change |
| 7 | 2024 | CHI | offence | Thomas Brown -> Chris Beatty | 2024-12-02 18:26:04 | staff_change |
| 8 | 2024 | LV | offence | Luke Getsy -> Scott Turner | 2024-11-05 20:07:29 | staff_change |
| 9 | 2025 | LV | offence | Chip Kelly -> Greg Olson | 2025-11-24 20:13:05 | playcaller_role |
| 10 | 2025 | NYG | offence | Mike Kafka -> Tim Kelly (Kafka keeps calling plays) | 2025-11-12 17:44:03 | staff_change |
| 11 | 2025 | NYG | defence | Shane Bowen -> Charlie Bullen | 2025-11-24 17:19:51 | staff_change |
| 12 | 2025 | NYJ | defence | Steve Wilks -> Chris Harris | 2025-12-15 19:25:32 | staff_change |

Nine offensive and three defensive changes. The PIT 2023 pair of edits
(`role_correction` Canada -> Sullivan at 13:58, `staff_change` Sullivan ->
Faulkner at 18:02, same day) is ONE event, counted once at the later
`staff_change` revision instant. **Read** (validation notes): event 10 is
a coordinator-title change with the playcaller unchanged; it is counted
because the brief says to count every `staff_change`, and the event table
above discloses it so a reader can discount it.

**Definition.** For each team-season with counted events, every REG-season
game whose kickoff is strictly after an event's revision instant gets
`game_number_after_change` = 1 + the number of that team's earlier REG games
whose kickoff is also after the MOST RECENT event before this game's
kickoff. Game 1 after the most recent change is the flag. When a team has
two events (CHI 2024, NYG 2025) the count restarts at the second event; a
game before any event, or whose kickoff is not strictly after the revision
instant, is never flagged. Built by
`nfl_ats.coordinator_changes.games_after_coordinator_change` (additive
helper, fails closed on an event with no season match) with a leakage
regression test that a later revision cannot change an earlier row's flag.

**Cells.**
- `lead29_playcaller_change_v1_first_game`: flag = game 1 after the change;
  complement = every other team-game in the population. Direction: BACK
  (sign +1). Outcome `team_covered`.
- `lead29_playcaller_change_v1_games_2_to_4`: descriptive; flag = games 2-4
  after the change vs everyone else, same direction convention (+1) purely
  so the sign reads the same way; the interim-HC screen found its bump in
  game 1 only, so this cell is reported, not hypothesised.

**Population and grade.** Primary: the OPENER grade -- the paired
Tuesday-opener archive, `artifacts/opener_evaluation/20260907T152026Z/per_game.parquet`
(1,537 REG games 2020-2025, the same archive lane M used; line =
`tue_open_home_spread`, the consensus Tuesday opener; `team_covered` from
`result - tue_open_home_spread`, pushes dropped). Every counted event is
2022-2025, inside that archive. Secondary, descriptive only: the close
grade (`game_features.parquet`'s nflverse `spread_line`) on the same
2020-2025 window.

**Measurement.** Covered-minus-baseline gap in accuracy points (flag cover
rate minus complement cover rate, x100), week-blocked bootstrap
(`nfl_ats.experiment_runner._block_bootstrap_subset_gap`, joint
multinomial over week blocks), 20,000 draws, seed 20260817, within-week
correlation ZERO (no design-effect padding), reported with
`probability_positive`; season-blocked interval as a secondary read. The
registry `--effect` is the full-slate-scaled value
(`scale_subset_effect`, gap x fraction of slate flagged), the unit every
existing `subset_bias` entry in `registry/weak_signals.json` uses so the
pool stays commensurable; the raw gap and its interval are carried in the
evidence text. Classification by
`nfl_ats.experiment_runner.classify_subset_bias_result` (the only admissible
mechanical terminal verdict is a resolved wrong sign); no positive control
is run here, so `bounded_by_control` cannot be claimed. Expected outcome
with twelve events: `unresolved_below_power`, reported with the raw record.

## R2: `post_bye_new_playcaller` (LEAD-28)

**Playcaller proxy.** The team's OC at the September 1 preseason
observation is the offensive-playcaller proxy (the source separates
coordinator title from play-calling duty only for the 2022-2025 in-season
edits, not for season starts). Stated plainly: this is OC tenure, not
measured play-calling responsibility.

**Tenure at the September 1 observation** (preseason rows only; the
in-season rows are never read for this cell):
- year 1: OC(S) differs from OC(S-1);
- year 2: OC(S) == OC(S-1) and OC(S-1) differs from OC(S-2);
- year 3+: OC(S) == OC(S-1) == OC(S-2);
- unknown (excluded, fail closed): any needed observation missing. So 2009
  is entirely unknown, 2010 can only supply year-1 flags, and 2011-2025 are
  fully classifiable.
Names are compared after stripping Wikipedia's parenthetical disambiguator
(`"Joe Brady (American football coach)"` -> `"Joe Brady"`), case-folded;
**measured** this session, five base names carry two title variants across
seasons in the preseason OC rows. Built by
`nfl_ats.coordinator_changes.oc_tenure_at_season_start` (additive helper),
with a leakage regression test that an in-season revision or a
post-cutoff correction cannot change a season-start tenure.

**Post-bye flag.** `nfl_ats.bye_edge_fade_overlay.bye_edge_flag_by_game`
on the newest `data/raw/*/schedules.parquet` snapshot (strict bye: >= 12-day
gap to the team's own immediately preceding REG game in the same season),
the live bye-fade overlay's own definition, reused verbatim.

**Cells** (population: REG team-games, pushes dropped, seasons 2009-2025,
tenure known; each cell states its own comparison):
- `lead29_playcaller_change_v1_post_bye_new_oc_back`: flag = post-bye AND
  OC in year 1 or 2; complement = every other team-game in the population.
  Direction: BACK (sign +1).
- `lead29_playcaller_change_v1_post_bye_veteran_oc_fade`: flag = post-bye
  AND OC in year 3+; complement = every other team-game. Direction: FADE
  (sign -1).
- `lead29_playcaller_change_v1_post_bye_oc_interaction`: eligible = post-bye
  team-games with known tenure; flag = year 1/2 vs year 3+ within that
  population. Direction: year-1/2 covers more (sign +1).

**Population and grade.** Primary: the close grade (`game_features.parquet`
nflverse `spread_line`), seasons 2009-2025 -- the opener archive starts in
2020 and this cell's point is its full-history sample size, so the
full-history read is the decision read. Secondary: the same three cells on
the opener grade 2020-2025 (same archive as R1), reported alongside; the
pool's grade is the opener, so the opener read is what a play would be
judged on, and it may not be vetoed by the close read (or vice versa --
both are reported, neither is hidden).

**Measurement.** Same bootstrap, seed, draw count and scaling rules as R1.

## Pregame safety

Only observations whose revision timestamp is strictly before the decision
timestamp are used: R1 uses the revision instant vs kickoff; R2 uses
preseason observations (all at or before the September 1 cutoff, measured)
vs REG kickoffs (all after September 1), and the script asserts
`observed_at < kickoff` on every flagged row. `tests/test_playcaller_change_screen.py`
carries the leakage regression tests (a later revision cannot change an
earlier row's flag; a revision after kickoff never flags that game; an
in-season revision never changes a season-start tenure).

## No other variants

No other flag definitions, sign selections, week restrictions, era slices
or tuning will be tried. Any number outside the cells above is descriptive
context (counts, raw records, overlap with the interim-HC first-game
challenger) and is labelled as such.

## Completed screen (2026-09-07, lane R)

**Measured** (`scripts/playcaller_change_screen.py --output-dir <fleet scratch>/laneR/run1`,
`results.json` stamped by `write_stamped_artifact`; inputs
`data/raw/coordinators/20260907T213814366437Z/coordinator_history.parquet` +
`change_validation.json`, `artifacts/opener_evaluation/20260907T152026Z/per_game.parquet`,
`data/processed/game_features.parquet`, `data/raw/20260905T211016Z/schedules.parquet`).
All twelve events produced a first game; the shortest revision-to-kickoff
lead is 96.3 hours (2025 NYG DC, Monday revision, Monday-night kickoff the
following week). Every interval is week-blocked, 20,000 draws, seed
20260817, within-week correlation zero; gaps are flag-minus-complement
cover rate in accuracy points, signed so positive favours the predeclared
direction. **Registry effect** = full-slate-scaled (see predeclaration).

### R1 (opener grade, 2020-2025; population 3,006 non-push team-games)

| Cell | n | Record (cover %) vs rest | Raw gap | 95% week-blocked | P+ (week / season) | Scaled effect | Class |
|---|---|---|---|---|---|---|---|
| `lead29_playcaller_change_v1_first_game` | 12 | 6-6 (50.00%) vs 50.00% | +0.00 | [-28.71, +33.40] | 0.450 / 0.430 | +0.000 [-0.115, +0.133] | unresolved_below_power |
| `lead29_playcaller_change_v1_games_2_to_4` (descriptive) | 31 | 17-14 (54.84%) vs 49.95% | +4.89 | [-11.41, +21.53] | 0.693 / 0.912 | +0.050 [-0.118, +0.222] | unresolved_below_power |

Season-blocked intervals here have 6 blocks and are degenerate under
`MIN_BLOCKS_FOR_INTERVAL`; only their P+ is quoted. **Measured** first
games: LV 2023 wk 9 (+2.5, covered), BUF 2023 wk 11 (+7.0, covered), CHI
2024 wk 11 (-6.0, covered), NYG 2025 wk 11 (-7.5, covered), CAR 2022 wk 6
(-11.0, lost), IND 2022 wk 10 (-6.5, covered), PIT 2023 wk 12 (+1.0,
covered), LV 2024 wk 11 (-7.5, lost), CHI 2024 wk 14 (-4.0, lost), LV 2025
wk 13 (-9.5, lost), NYG 2025 wk 13 (-7.5, lost), NYJ 2025 wk 16 (-4.5,
lost); spreads are the team's own opener line. Cover rate by game number
after the change (opener): game 1 6/12, game 2 6/12, game 3 6/10, game 4
5/9, game 5 4/8, game 6 5/7, game 7 5/5, games 8+ 3/6. **Measured**
overlap: 5 of the 12 first games are ALSO the interim-HC first game the
live `interim_hc_first_game_tilt_overlay` challenger flags (LV 2023, NYG
2025 wk 11, CAR 2022, IND 2022, CHI 2024 wk 14), so R1 is not independent
of that cell. Secondary close grade (same window): first game 6-6, gap
+0.00 [-28.70, +33.40], P+ 0.450; games 2-4 14-17 (45.16% vs 50.05%), gap
-4.89 [-22.10, +12.63], P+ 0.256.

**Inferred (decision):** R1's P+ is 0.450, below 0.5, so no overlay or
challenger is proposed; the cell stays `unresolved_below_power` (no
admissible closing ground: the interval is not resolved on either side and
no positive control was run).

### R2 (primary: close grade, 2009-2025; population 6,898 team-games with known OC tenure, 422 of them post-bye)

| Cell | n | Record (cover %) vs comparison | Raw gap (signed) | 95% week-blocked | P+ (week / season) | Scaled effect | Class |
|---|---|---|---|---|---|---|---|
| `lead29_playcaller_change_v1_post_bye_new_oc_back` | 303 | 145-158 (47.85%) vs 50.19% | -2.33 | [-8.53, +3.85] | 0.229 / 0.216 | -0.103 [-0.375, +0.169] | unresolved_below_power |
| `lead29_playcaller_change_v1_post_bye_veteran_oc_fade` | 119 | 63-56 (52.94%) vs 50.04% | -2.90 | [-12.14, +6.35] | 0.268 / 0.263 | -0.050 [-0.209, +0.110] | unresolved_below_power |
| `lead29_playcaller_change_v1_post_bye_oc_interaction` | 303 vs 119 | 47.85% vs 52.94% | -5.09 | [-17.13, +6.78] | 0.203 / 0.224 | -0.311 [-1.048, +0.415] | unresolved_below_power |

Secondary, opener grade 2020-2025 (2,665 team-games with known tenure):
new-OC back 56-54 (50.91% vs 49.82%), gap +1.09 [-7.85, +10.25], P+ 0.590;
veteran-OC fade 20-26 (43.48% vs 49.98%), gap +6.50 [-10.44, +22.73], P+
0.774; interaction 50.91% vs 43.48%, gap +7.43 [-12.42, +27.16], P+ 0.762
(season-blocked reads on 6 blocks are degenerate). **Measured** tenure
coverage: 2009 0 of 496 team-game rows classifiable, 2010 91 of 502, then
416-525 of 490-542 per season 2011-2025.

**Inferred (decision):** the predeclared full-history close read leans
AGAINST every R2 direction (P+ 0.20-0.27) while the 2020-2025 opener read
leans FOR them (P+ 0.59-0.77) on a subset of the same seasons, which means
the 2009-2019 close-graded portion is the adverse part. Neither read is
resolved. Per `AGENTS.md` a close-graded number may not veto a play, and a
screen makes no play; the honest summary is that the sign of LEAD-28 is
era-dependent in this data and unresolved either way. All three cells are
recorded `unresolved_below_power` (no admissible closing ground).

### Recorded

**Measured** (`registry/weak_signals.json`, 1,090 -> 1,095 signals): the five
cells above under family `lead29_playcaller_change_v1`, each with a
pool-player `plain_summary`, category `offfield`, the raw record and both
intervals in `classification_evidence`, and the secondary-grade read in
`notes`. No rotation window was spent (screen), no overlay or challenger
was wired, no model, card or published page changed.

### Caveats

- **Inferred:** twelve events is the whole in-season census the source
  offers for 2022-2025; a 6-6 record carries a 62-point-wide interval.
  This is the data, not a request for more of it.
- **Read** (validation notes): event 10 (NYG 2025 OC) kept the playcaller;
  the PIT 2023 event moved the title and the play-calling to two different
  people. R2's tenure proxy is the OC title, not play-calling duty.
- **Measured:** 5 of 12 R1 first games coincide with an interim-HC first
  game, so R1 and `interim_hc_first_game` share outcomes and are not
  independent votes in any pool.
- **Inferred:** preseason coverage gaps (2009 entirely, KC 2016 etc.)
  exclude rows rather than guessing; the 2010 season can only supply
  year-1 flags.

## Rebuild, split-half reliability, and production screens (2026-09-10)

**Read** (`git log`, `ROADMAP.md` PER-07/LEAD-28/LEAD-29 rows): the
2026-08-18/09-09/09-10 repository cut (`b7ed31d`, "469,660 -> 216,083 Python
lines") removed `scripts/playcaller_change_screen.py` and
`src/nfl_ats/coordinator_changes.py` (only their `.pyc` files survive on
disk); the underlying data (`data/raw/coordinators/20260907T213814366437Z/`,
`data/processed/game_features.parquet`, the opener archive) was untouched.
This session rebuilt the screen as a single standalone script with no
dependency on the deleted module, reusing the exact cell definitions,
sources and sign conventions above, and extended it with the production
screens and split-half reliability this lane's brief asks for that the
2026-09-07 run never computed. Closing-grounds taxonomy from the top of
this document applies to every number below without restatement.

### Rebuild fidelity check

**Measured** (`scripts/playcaller_change_screen.py screen`,
`artifacts/playcaller_change_leads/screen_results.json`): the rebuild
reproduces the 2026-09-07 run within ordinary bootstrap-seed noise on every
number that is a decision input. R1 population 3,006 (exact), R2 primary
population 6,898 (exact), R2 secondary opener population 2,665 (exact), the
same 12 first-game events at the same team/week/cover outcomes (6-6), and
the signed fade/interaction gaps match to two decimal places (e.g. primary
`post_bye_veteran_oc_fade` signed gap -2.90 both times; secondary opener
interaction +7.43 both times). One descriptive-only count differs: R1's
`games_2_to_4` cell reads 32 games here against 31 in the 2026-09-07 table,
most likely a one-game difference between that run's schedule snapshot and
the newer `data/raw/20260908T162105Z/schedules.parquet` used here (a
rescheduled/added game); it does not change any cell's sign or population
size and is descriptive, not a decision input. `first_game`'s P+ also moved
from 0.450 to 0.511 on an unchanged +0.00 point estimate -- expected
bootstrap-implementation variance around an exact tie, not a sign change.

### Split-half reliability (odd vs even seasons)

**LEAD-29 (`R1`):** `playcaller_first_game` is a rare in-season event flag
(12 events across only 4 seasons, 2022-2025), not a repeated-measures
per-team trait with enough seasons to correlate -- a formal split-half
coefficient does not apply the way it would to a persistent trait, matching
this project's convention for other thin per-event flags
(`docs/snow_game_home_prep.md` section 4, `reliability_check.method =
not_applicable`). **Measured**: the first-game-vs-population cover gap
computed independently on odd-numbered seasons (7 first games, 2023+2025)
reads +7.18 accuracy points, week-blocked 95% [-35.87, +50.20], P+ 0.6407;
even-numbered seasons (5 first games, 2022+2024) reads -10.03 points, 95%
[-50.10, +50.03], P+ 0.2858. The two halves disagree in sign, which is the
expected shape of noise at n=7/n=5, not a zero-reliability finding -- the
binding closing ground requires a *resolved* near-zero coefficient, not two
noisy halves pointing different ways on a thin sample.

**LEAD-28 (`R2`):** the `post_bye_oc_interaction` trait (new-OC post-bye
cover rate minus veteran-OC post-bye cover rate) has enough seasons
(2011-2024, since 2009-2010 can only supply year-1/2 flags) to run a real
season-level correlation. **Measured**: pairing each odd season with the
following even season (2011-2012, 2013-2014, ..., 2023-2024; 7 pairs, both
seasons needed at least one back-flagged and one fade-flagged post-bye game)
and correlating the two halves' per-season interaction gaps (season-pair
bootstrap, 20,000 draws): Pearson r = **+0.2691**, 95% [-0.5674, +0.9187],
`probability_positive` **0.754**, Spearman-Brown full-length correction
+0.4241. The interval crosses zero and is wide on 7 pairs, but it leans
positive, not toward zero -- per the binding rule this stays
`unresolved_below_power`, not a `no_split_half_reliability` closure (that
ground needs a coefficient resolved AT zero, which this is not). A second,
simpler read pooling each half wholesale (not pair-by-pair, same convention
as `docs/tiebreaker_low_side_shading.md` section 2): odd-numbered seasons
(154 back-flagged games) gap -2.497 points, P+ 0.296; even-numbered seasons
(149 back-flagged games) gap -2.195 points, P+ 0.311 -- both halves lean the
same (negative) direction as the pooled full-history primary read (-2.33
points, P+ 0.230), i.e. the trait's *sign* replicates across halves even
though the season-pair correlation of its *magnitude* is itself uncertain.
These are different estimands (as `docs/coaching_leads.md`'s Denver Q4
finding and `docs/tiebreaker_low_side_shading.md` section 2 both note for
their own traits) and neither result closes the other.

### Production screens (opener-graded, PRODUCTION weak_stack)

**Measured**, `scripts/playcaller_change_screen.py production-lead28` /
`production-lead29`, `artifacts/playcaller_change_leads/production_results_lead28.json`
/ `_lead29.json`. Candidate: production's (`weak_stack`, active model
`d49194e04945a5e5`) opener probability-rule pick, with a clean-case tilt
applied on top (flip to the flagged side only when exactly one side of the
game carries a signal; leave the pick untouched when both sides do or
neither does, matching this repo's `coach_fade_overlay`/`bye_edge_fade_overlay`
convention). Two reads each: the assigned rotation window (confirmation
accounting) and the full 2020-2025 predeclared descriptive archive (the
same 1,537-game opener archive R1/R2 use above).

**LEAD-28** (`lead28_post_bye_new_playcaller_on_production`, window
[2020, 2021] assigned via `nfl-ats rotation assign`, the same block other
lanes drew tonight):

| Read | Games (graded) | Flagged | Picks changed | Effect | 95% week-blocked | P+ (week) | P+ (season) |
|---|---:|---:|---:|---:|---|---:|---:|
| Assigned window 2020-2021 | 456 | 43 | 22 | **+0.877** pts | [-0.887, +2.643] | 0.8287 | 0.7494 |
| Full archive 2020-2025 (descriptive) | 1,503 | 145 | 71 | **+0.333** pts | [-0.728, +1.390] | 0.7337 | 0.7631 |

Both reads lean positive and both cross zero -- per the binding rule that
is not grounds to reject; `probability_positive` of 0.73-0.83 favours
playing this tilt on expected value, not vetoing it. Recorded
`unresolved_below_power` (no admissible closing ground either way); the
2020-2021 window is now spent for this family.

**LEAD-29** (`lead29_playcaller_first_game_on_production`, window
[2020, 2021] assigned the same way):

| Read | Games (graded) | Flagged | Picks changed | Effect | 95% week-blocked | P+ (week) | P+ (season) |
|---|---:|---:|---:|---:|---|---:|---:|
| Assigned window 2020-2021 | 456 | 0 | 0 | 0.000 pts (degenerate) | [0.000, 0.000] | 0.5000 | 0.5000 |
| Full archive 2020-2025 (descriptive) | 1,503 | 12 | 4 | **-0.133** pts | [-0.401, +0.132] | 0.1605 | 0.0431 (95% [-0.263, 0.000]) |

**Inferred:** the assigned window is a structural zero -- none of the 12
counted in-season OC/DC changes (all 2022-2025) fall inside 2020-2021, so
the confirmation-accounting read is uninformative by construction, not
evidence against the rule. The full-archive descriptive read (the one that
actually contains all 12 events, only 4 of which disagree with production's
existing pick) leans slightly negative; its week-blocked upper bound
(+0.132) is positive and its season-blocked upper bound sits exactly at,
not below, zero, so per the same rule this is **not** a resolved wrong sign
either -- recorded `unresolved_below_power` on both reads, and the 2020-2021
window is now spent for this family too.

**Read plainly**, this production-conditional read (does flipping
production's own pick help, on the residual cases where the model's pick
disagrees with the flag) is not the same estimand as the unconditional
subset-bias screen above (does the flagged group cover more/less than the
population, full stop) -- R2's raw close-grade interaction cell leans P+
0.20 (against the predeclared direction) over 2009-2025, its opener-grade
secondary leans P+ 0.76 (for it) over a smaller 2020-2025 slice, and its
production-conditional read leans P+ 0.73-0.83 (for it) over the same
2020-2025 archive; R1's raw first-game cell is a coin flip (P+ 0.51) while
its production-conditional read leans P+ 0.16 (against it). None of these
four numbers resolves any other; they are reported side by side rather than
collapsed into one verdict, per this file's own instruction to state the
number and the interval rather than a one-word summary.

### Registry and rotation

**Measured** (`registry/weak_signals.json`, family `playcaller_change`, 6
new entries, registry 5,900 -> 5,906): `playcaller_change_lead28_production
_screen_window`, `_full_archive`, `playcaller_change_lead29_production
_screen_window`, `_full_archive`, `playcaller_change_lead28_reliability
_season_pair` (`--effect-units correlation`, the actually-measured unit, not
accuracy_points), `playcaller_change_lead29_reliability_odd_even_seasons`.
All six `unresolved_below_power`; none closed.

**Measured** (`registry/rotation_registry.json`): both
`lead28_post_bye_new_playcaller_on_production` and
`lead29_playcaller_first_game_on_production` declared (grade `opener`,
`--acknowledge-mined`), assigned the [2020, 2021] window (the block other
lanes drew the same night), and recorded `unresolved` via `nfl-ats rotation
record`; both windows are now spent. No overlay, challenger, model or
published card was changed by this lane.

### Files

`scripts/playcaller_change_screen.py` (subcommands `screen`,
`production-lead28`, `production-lead29`);
`artifacts/playcaller_change_leads/screen_results.json`,
`r1_population.parquet`, `r2_population.parquet`,
`production_results_lead28.json`, `production_results_lead29.json`,
`production_paired_lead{28,29}_{window,full}.parquet`.

## Post-bye new-playcaller BACK overlay: a no-window-cost prospective challenger (2026-09-10)

Follows the `veteran_rest_back_overlay` / `backup_qb_fade_overlay` precedent
(`docs/veteran_rest_back_overlay.md`, `docs/backup_qb_fade_overlay.md`) for
wiring a pick-level, post-prediction transform into the prospective
challenger ledger at zero rotation-registry window cost. Closing-grounds
taxonomy from the top of this document applies without restatement.

### What is built, and what it is not

This wires `post_bye_new_playcaller_back_overlay`: it flips production's
opener pick onto a team playing its first game off its own bye
(`nfl_ats.bye_edge_fade_overlay.bye_edge_flag_by_game`, the same >=12-day-gap
definition the live `bye_edge_fade_overlay` challenger and the R2 screen
above both use) whose offensive coordinator is in year 1 or 2 of tenure at
that season's September-1 preseason observation, whenever the opponent is
NOT also uniquely flagged and production's own pick is not already on that
side. **This is deliberately ONLY the back half** of the production
screen's combined tilt above (`lead28_post_bye_new_playcaller_on_production`,
family `playcaller_change`) -- it never fades a post-bye team with a
year-3+ ("veteran") offensive coordinator. The `+0.877` / `+0.333`
accuracy-point registry entries quoted throughout this document
(`playcaller_change_lead28_production_screen_window` /
`_full_archive`) are readings of the COMBINED back+fade tilt, not an
isolated remeasurement of this challenger's own back-only rule.

**Measured 2026-09-10** against the saved production-screen ledgers
(`artifacts/playcaller_change_leads/production_paired_lead28_{window,full}.parquet`),
diagnostically, not as a new registry entry: of the picks the combined tilt
changed, 16 of 23 (assigned window) and 53 of 73 (full archive) are
attributable to the back-only signal, the remainder to the fade-only
signal, with zero games in either read carrying both signals at once. The
back component is the majority contributor to the combined read, but its
own effect size was never isolated and bootstrapped on its own -- this
challenger's 2026+ prospective ledger is the only measurement that will
speak to the back-only construct specifically, exactly as
`veteran_rest_back_overlay.md`'s own arithmetic-mirror section disclaims
its own reversed figure.

**Reused, not re-derived.** The OC-tenure classification (year 1: this
season's OC differs from last season's; year 2: same as last season, which
differed from the season before; year 3+: same OC three seasons running;
unknown when a needed prior-season observation is missing) is the identical
three-branch rule `scripts/playcaller_change_screen.py`'s
`load_oc_tenure_table` uses, ported into
`nfl_ats.post_bye_new_playcaller_back_overlay.oc_tenure_by_team_season`
(same preseason-only OC rows, same disambiguator-stripped/case-folded name
comparison) rather than importing the script -- production code depending on
a `scripts/` file would be backwards, matching how
`backup_qb_fade_overlay.md` describes its own construct as "ported, not
redesigned" from its screen's script. The bye flag itself is not ported at
all: it calls `nfl_ats.bye_edge_fade_overlay.bye_edge_flag_by_game` directly,
the same function the screen imports.

**Pregame-safe, and how a midseason change is dated.** Tenure is read only
from each season's own September-1 preseason OC observation -- the in-season
revision rows in `coordinator_history.parquet` are never read for this
construct, matching R2's own design above ("the in-season rows are never
read for this cell"). A coordinator fired or promoted mid-season therefore
never changes that season's already-fixed year-1/2/3+ classification for any
team; the earliest that classification can move is the FOLLOWING season's
own September-1 cutoff. Since every preseason observation is dated at or
before September 1 and every REG kickoff falls after it, tenure is fixed
before any of that season's games are decided -- there is no path for a
same-season event to leak into a game's flag.

**Known gap, disclosed up front.** Measured 2026-09-10
(`pandas.read_parquet` against the newest snapshot,
`data/raw/coordinators/20260907T213814366437Z/coordinator_history.parquet`):
the table's seasons run 2009-2025 only. There is no 2026 preseason OC
observation for any team yet, so every 2026 team-game currently resolves to
an "unknown" tenure and the back flag cannot fire for any 2026 game --
independently of whether that week has any byes -- until a fresh
coordinator-history snapshot with a 2026 preseason cutoff is captured. This
fails closed (no flag, not a wrong flag) and does not compromise pregame
safety; it just means the overlay is presently a structural no-op league-wide
for 2026, not only in weeks with no byes.

### Confirmed live against the current week (2026 Week 1)

**Measured, 2026-09-10**, against the active forecast
(`artifacts/margin_predictions/2026-week-01-20260910T210852Z`, model
`d49194e04945a5e5`): 2026 Week 1 has zero post-bye games league-wide (a bye
is structurally impossible in the first week of any season;
`bye_edge_flag_by_game` on the newest schedule snapshot confirms 0 of 16
Week 1 games have either side off a bye), and separately no team's 2026 OC
tenure is yet classifiable (see known_gap above) -- both reasons
independently guarantee zero flags this week, so the overlay produced:

```
flip_count: 0
flipped_game_ids: []
both_flagged_games: []
```

`record_post_bye_new_playcaller_back_overlay_decisions` was run once this
session to confirm the recording path end to end: `recorded: 14`,
`post_kickoff_skipped: 2` (`2026_01_NE_SEA` and `2026_01_SF_LA` had already
kicked off by the time of the run, the same two games every other overlay
recorded this week already excluded), `ledger_rows: 530` after the append.
With zero flips, all 14 recorded rows are byte-identical to production's own
raw pick on the same 14 games -- a clean 1:1 match, not a coincidence of a
flip landing on an already-excluded game (there were no flips to exclude).
This does not change what was confirmed: the overlay computed a correct,
non-trivial-to-reach value (zero real flags is the mechanically correct
answer for Week 1, not a silent failure) and the recording path wrote real
rows to the shared ledger without touching `recommendations.csv`,
`CURRENT_PREDICTIONS.md`, or any served page.

### What is and is not wired in

- `src/nfl_ats/post_bye_new_playcaller_back_overlay.py`: the transform
  (`apply_post_bye_new_playcaller_back_overlay`), the signal readers
  (`post_bye_new_oc_flag_by_game`, `oc_tenure_by_team_season`,
  `load_coordinator_history`), the disclosure sentence
  (`overlay_disclosure_note`, not currently surfaced anywhere), and the
  recorder (`record_post_bye_new_playcaller_back_overlay_decisions`).
- `src/nfl_ats/cli_commands/publishing.py`'s `orchestrate_publish_predictions`:
  one more purely additive, fail-open `try`/`except` block (mirroring the
  existing ones) that calls the recorder when `--record-decisions` is
  passed, plus a `PUBLISH_CHALLENGER_RESULT_KEYS` entry
  (`post_bye_new_playcaller_back_overlay` ->
  `post_bye_new_playcaller_back_overlay_challenger_ledger`) and a matching
  `skipped` placeholder in the no-`--record-decisions` branch. This writes
  ONLY to `artifacts/prospective/challenger_decisions.parquet`; it never
  touches `recommendations.csv`, `CURRENT_PREDICTIONS.md`, `README.md`, or
  the public site. **The production pick path (`publish_active_predictions`)
  is untouched by this build.**
- `artifacts/prospective/challengers.json`: registered as
  `post_bye_new_playcaller_back_overlay`, status `ACTIVE_PROSPECTIVE`, `model`
  block a snapshot of the active configuration at registration time (for
  fingerprint-mismatch detection only, mirroring every other overlay
  challenger).
- `src/nfl_ats/dashboard/findings_content.py`'s `CHALLENGER_DISPLAY_NAMES`:
  `post_bye_new_playcaller_back_overlay` -> `"Post-bye new playcaller back"`.
- **Not wired anywhere:** there is no switch that applies this overlay to
  the published card. Playing this on the real card is a separate owner
  decision this document does not make.
- **Tracked independently against the active model's own card, not stacked
  on the other overlays**, matching every sibling overlay's pattern exactly.

### No new tests (moratorium)

Per the standing test moratorium, no test file or test function was added
for this module. Verification here is the direct run reported above
(`record_post_bye_new_playcaller_back_overlay_decisions` against the live
2026 Week 1 card), not a pytest fixture.

### Decision left to the orchestrator

Nothing here decides anything -- it only starts a free, honestly-labelled
evidence stream. Once 2026 accrues enough weeks (and, separately, once a
2026 preseason coordinator snapshot exists so the OC-tenure half of the flag
can fire at all), `nfl-ats prospective-score` reports this challenger's
`probability_positive` at both grades, paired against the active model,
exactly like every other overlay challenger. That number -- not the combined
back+fade production-screen figures quoted above -- is what will actually
say whether backing a new playcaller off a bye holds up on its own.
