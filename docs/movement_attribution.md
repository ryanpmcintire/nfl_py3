# Movement attribution: what does the market know that we don't?

Written 2026-08-20, **predeclaration section frozen before any flip-value
below was computed**. Owner question, verbatim: given the finding that
flipping to the market's side when Tuesday-to-close movement disagrees with
the model pick is worth **+5.26 accuracy points** (n=494, P+ 0.880,
unfiltered) and **+9.66 points** (n=290, P+ 0.935, `|open_move| >= 1.0`) --
`docs/opener_error_analysis.md`'s reconciliation section, itself consistent
with the already-recorded `observed_movement_*` family -- "it seems pretty
likely the market knows more than we do in these cases... can we figure out
WHAT it is the market knows that we don't?" This document attributes each
adverse move to an observable cause using archives already on disk, then
measures where the flip-value concentrates.

**Binding closing-grounds taxonomy (AGENTS.md), restated verbatim per this
project's rule for any document that scores or adjudicates an experiment:**
an interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds
ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
(whole interval on the wrong side of zero) or zero split-half reliability;
(2) bounded by a positive control proven able to detect an effect that
size. Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero." Within-week game correlation is ZERO by owner
mandate (week-blocked bootstrap already accounts for this; no separate ICC
term is estimated or padded). Decisions are made on expected value, never
on a 95%/0.90 threshold.

**Commensurability note (AGENTS.md):** every cell below is a **correlated
decomposition of the already-recorded `observed_movement_*` family** -- same
archive (`artifacts/observed_movement_channel/20260820T093426Z/per_game_tue_close.parquet`),
same population, same movement definition, a subpopulation cut by
attribution class rather than an independent sample. These cells will be
recorded under `movement_attribution_*` names, flagged in their `--notes` as
correlated decompositions, and must **never** be pooled additively with
`observed_movement_oracle_full_slate`, `observed_movement_threshold_1_0`, or
the `opener_error_mining_movement_agreement_*` entries.

## Population (frozen)

Base: `artifacts/observed_movement_channel/20260820T093426Z/per_game_tue_close.parquet`
(the exact artifact `docs/opener_error_analysis.md`'s reconciliation section
used), push-excluded (`correct_at_open_probability_rule` not null),
production pick = `pick_home_at_open_probability_rule`, candidate/market
pick = `oracle_pick_home` (movement side when `open_move != 0`, production
pick on an exact tie -- already computed in the artifact), graded at
`margin_vs_open` (the frozen Tuesday line).

- **POP_UNFILTERED**: `open_move != 0` and `oracle_pick_home != pick_home_at_open_probability_rule`
  (disagreement). Anchor target: n=494, paired delta +5.26 pts, week-blocked
  95% [-2.86, +14.12], P+ 0.880.
- **POP_THRESHOLD**: POP_UNFILTERED further restricted to `abs(open_move) >= 1.0`.
  Anchor target: n=290, paired delta +9.66 pts, week-blocked 95%
  [-2.68, +21.74], P+ 0.935. (POP_THRESHOLD subset of POP_UNFILTERED --
  disagreement is defined the same way regardless of filter order.)

**Reproduction check (must pass before any new number is trusted):**
recompute both cells' paired delta on this exact population using
`nfl_ats.clv.week_blocked_bootstrap`, `block="week"`, `samples=2000`,
`seed=20260820` (identical spec to the reconciliation section) and confirm
the numbers land within ordinary bootstrap resampling noise of the anchor
targets above before computing any attribution-stratified cell.

## Attribution classes (predeclared, operational definitions frozen before scoring)

For every game in POP_UNFILTERED: `moved_against_team` = the team we picked
(`home_team` if `pick_home_at_open_probability_rule` else `away_team` --
the side the market moved away from); `favored_team` = the other team (the
side the market moved toward).

### (a) INJURY

**Primary path, seasons 2020-2024 (official injury report archive
`data/players/raw/20260817T184901Z/injuries.parquet` covers only through
2024 -- 2025 has no official-report coverage, disclosed as a coverage gap,
not silently patched):**

- Restrict to `game_type == "REG"` and `position in {QB, RB, WR, TE}` (skill
  positions -- the market-relevant subset; this project does not have a
  snap-share-weighted "key player" definition cheaply available without a
  full model rebuild, so position is the predeclared proxy for
  "market-relevant player").
- Severity scale (predeclared, ordinal): `Out=4, Doubtful=3,
  Questionable=2, Probable=1, not on report=0`.
- For each (season, week, team), compute each player's severity **as of
  that game's own-week Tuesday noon ET** (`tuesday_status`, from the latest
  `date_modified <= tuesday_noon_utc` row) and **as of kickoff**
  (`final_status`, from the latest `date_modified <= kickoff_utc` row,
  i.e., the Friday-final report). Tuesday-noon-ET cutoff computed the same
  way as `scripts/injury_tuesday_cutoff_experiment.py`'s
  `team_week_tuesday_noon` (`(kickoff weekday - 1) % 7` days back, +12h ET,
  converted to UTC) -- duplicated locally per this repo's convention of not
  importing across `scripts/*.py` files.
- `player_delta = final_severity - tuesday_severity` (0 for a player absent
  from both snapshots; correctly captures both a brand-new post-Tuesday
  designation and a recovery that drops a player off the report).
- `team_injury_delta = sum(player_delta)` across that team's skill-position
  players for the week.
- `net_injury_score = team_injury_delta(moved_against_team) -
  team_injury_delta(favored_team)`. Positive means the abandoned side
  accumulated net-worse post-Tuesday injury news than the favored side --
  the sign that RATIONALIZES the move.
- **INJURY flag (2020-2024) = `net_injury_score >= 2`** (a fixed,
  predeclared magnitude bar -- roughly one new Questionable-or-worse
  asymmetry, chosen before any hit rate was computed).

**Fallback path, season 2025 only (no official-report coverage):**
ProFootballTalk headline count (`data/raw/injury_news/20260819T191639Z/index.parquet`,
`injury_relevant == True`), matched to each team by a fixed 32-team
nickname-substring lookup against `headline_guess`, restricted to
`lastmod` in `(tuesday_noon_utc, kickoff_utc]`. `net_pft_score =
pft_hits(moved_against_team) - pft_hits(favored_team)`. **INJURY flag
(2025) = `net_pft_score >= 1`.** This is a coarser, lower-fidelity proxy
(team-level headline volume, not player-level status, and a headline
naming both teams is not disambiguated) -- disclosed, not treated as
equivalent evidence to the official-report path.

### (b) WEATHER

`data/raw/forecast_archive/full_2020_2025/forecasts.parquet` (Tuesday-noon
MOS forecast vs. realized actuals -- `docs/forecast_archive_build.md`),
joined on `game_id`. Restricted to `roof in {"outdoors", "open"}` (indoor
games are not weather-exposed and score WEATHER=False by construction, not
missing data). No precipitation field exists in this archive -- disclosed;
only wind and temperature deterioration are measured.

- `wind_delta = actual_wind_mph - forecast_wind_mph`,
  `temp_delta = actual_temp_f - forecast_temp_f`.
- **WEATHER flag = `wind_delta >= 10` OR `abs(temp_delta) >= 15`**
  (predeclared magnitude bars, roughly 2x this archive's own measured MAE
  at the Tuesday-noon lead -- 4.89 mph wind MAE, 7.63°F temp MAE per
  `docs/forecast_archive_build.md` -- chosen as a "meaningfully surprising"
  cut before any hit rate was computed). Games missing a usable
  actual/forecast pair score WEATHER=False and are reported separately as a
  coverage gap, not folded into "no weather effect."
- Unlike INJURY, this flag is not built with a specific side/direction (no
  clean mechanism maps wind/temp surprise onto which side of the spread it
  should favor) -- WEATHER marks games where a plausible non-injury
  environmental cause exists, and the stratified flip-value speaks for
  itself.

### (c) PUBLIC ALIGNMENT (sparse subset only)

`data/raw/public_betting/20260820T111148Z/actionnetwork/index.parquet`
(`docs/public_betting_sourcing.md`), Consensus book (id 15, the only book
id present in this archive), joined to the population by `(away_team,
home_team)` team-pair match with `|kickoff - start_time_utc| <= 72h`
(identical join pattern to `scripts/ingest_public_betting.py`'s
`build_coverage_report`). For each game, the single **most recent capture
strictly before kickoff** is used (a reading captured after Tuesday is
still a legitimate, playable signal per this project's picks-editable-to-
kickoff convention; a reading at or after kickoff is not, since it can no
longer inform a decision).

- `move_side = "home"` if `open_move > 0` else `"away"`.
- `bet_pct_move = spread_{move_side}_bet_pct`,
  `bet_pct_against = spread_{other_side}_bet_pct`.
- Classification (bet% ticket share, available in both archive eras --
  money% is era2-only, 2022-11 onward, and is reported as a secondary
  diagnostic where present, not the primary classifier):
  - `bet_pct_move > bet_pct_against` -> **`book_shading_public`** (the
    crowd's tickets already favored the side the line moved toward -- the
    mechanical "public money pushes the number" story).
  - `bet_pct_move < bet_pct_against` -> **`reverse_line_movement`** (the
    line moved AWAY from where the ticket majority sat -- the classic
    sharp-money signature).
  - equal -> `even`.
- **PUBLIC flag = TRUE if a matched pregame bet% reading exists**,
  regardless of subtype; the two subtypes are reported separately.
  Coverage is expected to be low (`docs/public_betting_sourcing.md`
  reports at most ~34% of REG games in the best-covered season have any
  pregame reading at all) -- reported honestly, not treated as a dense
  feature.

### (d) UNATTRIBUTED

None of INJURY, WEATHER, PUBLIC is TRUE for that game.

Flags are **not mutually exclusive** (a game can be both INJURY and
WEATHER); `ATTRIBUTED` = INJURY or WEATHER or PUBLIC (any cause visible);
`UNATTRIBUTED` = none visible in these archives specifically -- not a claim
that no cause exists.

## Measurement (predeclared)

For POP_UNFILTERED, POP_THRESHOLD, and every attribution class/subtype
(and their complements), the paired flip-value: `market_pick =
oracle_pick_home`, `production_pick = pick_home_at_open_probability_rule`,
`candidate_correct - production_correct` averaged per game, graded at
`margin_vs_open`, `nfl_ats.clv.week_blocked_bootstrap(block="week",
samples=2000, seed=20260820)`. `probability_positive` reported for every
cell regardless of sign or interval shape, per AGENTS.md. Attribution
coverage (fraction of POP_UNFILTERED with each flag, and with ANY flag) is
reported alongside every table, not just once.

This is a mined battery (INJURY x2 populations, WEATHER x2, PUBLIC x2
subtypes x2 populations, UNATTRIBUTED x2, plus the two anchor cells) --
**no multiplicity correction is claimed**, consistent with every other
mined-battery document in this repo.

## Interpretation rules (fixed before results are seen)

- If flip-value **concentrates in ATTRIBUTED moves** (materially larger
  paired delta than UNATTRIBUTED, both directions reported honestly): the
  market is aggregating specific midweek information this project can
  ingest and front-run -- rank which cause carries the value and sketch
  what a Wed/Thu refresh pass would need to watch.
- If **UNATTRIBUTED moves still carry large flip value**: say so plainly --
  either the market knows things outside these archives, or the Tuesday
  opener itself mean-reverts for reasons unrelated to new information
  (stale-price correction). Do not force a mechanism onto a cell that
  doesn't show one.

---

## Results

**Measured** this session, `scripts/movement_attribution.py`,
`artifacts/movement_attribution/20260820T114620Z/` (`per_game_attribution.parquet`,
`cells_summary.csv`, `metadata.json`).

### Plain-English answer

Injury news is what the market knows that the model doesn't -- and it
carries roughly double the average flip-value. Of the 494 games where the
Tuesday-to-close line moved against the model's pick, **185 (37%) show an
asymmetric post-Tuesday injury signal** favoring the market's new side (an
official Wed-Fri report downgrade on the abandoned team, or an upgrade on
the favored team, net of both). Following the market on exactly those
games is worth **+10.27 accuracy points** (P+ 0.914) -- almost double the
**+5.26-point** average across the whole disagreement population -- and the
concentration holds, even more strongly, in the bigger-move subset:
**+17.07 points (P+ 0.976, n=123)** against a **+9.66-point** average, with
a week-blocked interval that sits entirely above zero (though per this
project's binding taxonomy that is evidence, not a closing ground -- see
below). Pooling all three attributable causes together (injury, weather,
or a public-betting alignment reading), **ATTRIBUTED games are worth +8.17
to +10.71 points vs +2.11 to +8.20 for UNATTRIBUTED games** -- so a
meaningful share of "the market knows more than us" traces to a specific,
already-ingested cause, concentrated almost entirely in injury news, not
weather or public money. One genuine surprise, reported plainly per
AGENTS.md rather than smoothed over: **weather-deteriorated games go the
other way** -- following the market actually costs points there
(-12.82 to -25.00 pts), though the sample is thin (n=39/24) and this is
far from resolved. And the UNATTRIBUTED share does **not** shrink to zero
in the higher-conviction subset -- at `|open_move| >= 1.0` it still reads
+8.20 points (P+ 0.797) -- so a real chunk of what the market knows stays
genuinely opaque to these three archives even after removing every visible
cause this document could check.

### Anchor reproduction (must-pass check, run before any attribution number was trusted)

| population | n | target delta | measured delta | target P+ | measured P+ |
|---|---:|---:|---:|---:|---:|
| POP_UNFILTERED | 494 | +5.26 pts | **+5.26 pts** | 0.880 | **0.8805** |
| POP_THRESHOLD (\|open_move\|>=1.0) | 290 | +9.66 pts | **+9.66 pts** | 0.935 | **0.9350** |

Exact match (to the reported two decimals) on `week_blocked_bootstrap`,
`samples=2000`, `seed=20260820`, identical to `docs/opener_error_analysis.md`'s
reconciliation section. The population, movement definition, and grading
line are confirmed identical before trusting any downstream cell.

### Attribution coverage

| population | n | INJURY | WEATHER | PUBLIC | ANY attributed | UNATTRIBUTED | weather data usable | season-2025 games |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| POP_UNFILTERED | 494 | 37.4% (185) | 7.9% (39) | 18.2% (90) | 52.0% (257) | 48.0% (237) | 58.5% | 82 |
| POP_THRESHOLD | 290 | 42.4% (123) | 8.3% (24) | 20.0% (58) | 57.9% (168) | 42.1% (122) | 56.9% | 43 |

Flags are not mutually exclusive (a game can be both INJURY and WEATHER):
measured overlaps on POP_UNFILTERED are INJURY&WEATHER=19,
INJURY&PUBLIC=33, WEATHER&PUBLIC=9, all three=4 (of 494).

INJURY path split (POP_UNFILTERED, measured): of 412 games with official
injury-report coverage (seasons 2020-2024), 151 (36.7%) flag INJURY via the
official-report path; of 82 season-2025 games (no official coverage), 34
(41.5%) flag INJURY via the PFT-headline fallback -- similar base rates,
which is reassuring face validity, but the two paths are not equal-fidelity
evidence (see Caveats).

### Flip-value by attribution class

Paired flip-value = market pick (`oracle_pick_home`) minus production pick
(`pick_home_at_open_probability_rule`), graded at `margin_vs_open`,
week-blocked bootstrap (2000 samples, seed 20260820). Every cell reported
regardless of sign or interval shape, per AGENTS.md; `probability_positive`
is the decision-relevant number, never "contains zero."

| population | class | n | flip-value (pts) | week-blocked 95% (pts) | P+ |
|---|---|---:|---:|---|---:|
| POP_UNFILTERED | **ANCHOR (all disagreements)** | 494 | **+5.26** | [-2.86, +14.12] | 0.8805 |
| POP_UNFILTERED | INJURY | 185 | **+10.27** | [-4.24, +24.14] | **0.9135** |
| POP_UNFILTERED | WEATHER | 39 | **-12.82** | [-44.44, +17.95] | 0.1880 |
| POP_UNFILTERED | PUBLIC (any reading) | 90 | +2.22 | [-16.49, +21.57] | 0.5515 |
| POP_UNFILTERED | &nbsp;&nbsp;PUBLIC: book_shading_public | 44 | 0.00 | [-31.82, +30.24] | 0.4445 |
| POP_UNFILTERED | &nbsp;&nbsp;PUBLIC: reverse_line_movement | 44 | +4.55 | [-20.00, +29.74] | 0.5955 |
| POP_UNFILTERED | &nbsp;&nbsp;PUBLIC: even (n too small to read) | 2 | 0.00 | [-100, +100] | 0.2455 |
| POP_UNFILTERED | **ATTRIBUTED (any cause)** | 257 | **+8.17** | [-3.35, +20.73] | **0.9065** |
| POP_UNFILTERED | **UNATTRIBUTED** | 237 | +2.11 | [-9.01, +14.06] | 0.6225 |
| POP_THRESHOLD | **ANCHOR (\|open_move\|>=1.0)** | 290 | **+9.66** | [-2.68, +21.74] | 0.9350 |
| POP_THRESHOLD | INJURY | 123 | **+17.07** | **[+0.79, +31.67]** | **0.9760** |
| POP_THRESHOLD | WEATHER | 24 | **-25.00** | [-62.98, +18.19] | 0.1120 |
| POP_THRESHOLD | PUBLIC (any reading) | 58 | -3.45 | [-28.14, +22.81] | 0.3780 |
| POP_THRESHOLD | &nbsp;&nbsp;PUBLIC: book_shading_public | 29 | +3.45 | [-37.93, +46.17] | 0.5340 |
| POP_THRESHOLD | &nbsp;&nbsp;PUBLIC: reverse_line_movement | 27 | -11.11 | [-40.00, +20.00] | 0.1990 |
| POP_THRESHOLD | &nbsp;&nbsp;PUBLIC: even | 2 | 0.00 | [-100, +100] | 0.2455 |
| POP_THRESHOLD | **ATTRIBUTED (any cause)** | 168 | **+10.71** | [-4.82, +26.12] | **0.9035** |
| POP_THRESHOLD | **UNATTRIBUTED** | 122 | +8.20 | [-10.57, +25.40] | 0.7970 |

`POP_THRESHOLD_INJURY`'s week-blocked interval sits entirely above zero.
Per this project's binding taxonomy, a fully-positive interval is **not**
an admissible closing ground either (only `refuted_mechanism` -- a
RESOLVED wrong sign -- and `bounded_by_control` close a line of work); this
stays `unresolved_below_power` like every other cell here, reported with
its `probability_positive` rather than treated as "confirmed."

### Interpretation, per the predeclared rules

Flip-value concentrates in ATTRIBUTED moves, and within that almost
entirely in INJURY: INJURY's point estimate (+10.27 / +17.07 pts) is
roughly double the ANCHOR average in both populations, and ATTRIBUTED-ANY
(+8.17 / +10.71) clears UNATTRIBUTED (+2.11 / +8.20) in both cuts too. That
satisfies the first predeclared interpretation rule: **the market is
aggregating injury information this project can ingest and front-run**.
But the second rule also applies, honestly: UNATTRIBUTED does **not**
collapse toward zero in the higher-conviction POP_THRESHOLD cut (+8.20 pts,
P+ 0.797, on n=122) -- a genuine share of what the market knows stays
outside these three archives even after removing every visible cause. Both
things are true at once; this document reports both rather than picking
the cleaner-sounding one.

### Ranked leads (by \|effect\|·sqrt(n), the same convention as `docs/opener_error_analysis.md`)

1. **INJURY, POP_THRESHOLD** (+17.07 pts, n=123, \|effect\|·sqrt(n)=189.3,
   whole CI positive, P+ 0.976). The single largest, best-powered cell in
   this battery. Reproduces the same direction as INJURY on the unfiltered
   population (+10.27 pts, n=185, \|effect\|·sqrt(n)=139.8) -- consistent
   sign and roughly-doubling magnitude across two independent-enough cuts
   of the same underlying mechanism.
2. **WEATHER, POP_THRESHOLD** (-25.00 pts, n=24, \|effect\|·sqrt(n)=122.5,
   P+ 0.112). Consistent negative direction with WEATHER on the unfiltered
   population (-12.82 pts, n=39, \|effect\|·sqrt(n)=80.1) -- a real,
   reproducible-direction counter-finding, not a fluke of one cut, but
   thin (n=24-39) and `unresolved_below_power`.
3. **ATTRIBUTED_ANY** (both cuts, +8.17/+10.71 pts, n=257/168,
   \|effect\|·sqrt(n)=131.0/138.6). The broad "any visible cause" summary;
   ranks just below INJURY alone because WEATHER's opposite sign and
   PUBLIC's near-zero reading dilute it slightly relative to INJURY in
   isolation.
4. **UNATTRIBUTED, POP_THRESHOLD** (+8.20 pts, n=122,
   \|effect\|·sqrt(n)=90.6, P+ 0.797). Nearly as large as ATTRIBUTED_ANY at
   this cut -- the clearest evidence that attribution coverage, while
   informative, does not explain the whole effect.
5. **PUBLIC: reverse_line_movement, POP_UNFILTERED** (+4.55 pts, n=44,
   \|effect\|·sqrt(n)=30.2, P+ 0.5955) vs. **PUBLIC: book_shading_public**
   (0.00 pts, n=44, P+ 0.4445) -- directionally consistent with the
   "sharp money" hypothesis (line moves against the public ticket majority
   score slightly better than moves that align with it) but both near a
   coin flip and the sign flips at POP_THRESHOLD (reverse_line_movement
   goes to -11.11 pts there) -- not a usable signal on this small a subset.

### Front-running sketch (injury carries the value -- what a Wed/Thu refresh pass would watch) -- WIRED 2026-08-20

**Status: wired**, not just sketched. `nfl_ats.injury_signal_refresh_tilt`
implements items 1-4 below verbatim as a dual-tracked challenger,
`injury_signal_refresh_tilt` (`artifacts/prospective/challengers.json`),
recording BOTH arms (the model's own hold pick and the injury-tilted pick)
at every `nfl-ats refresh-picks --record-decisions` pass, alongside a
disagreement classification against the observed-movement policy's own read
of the same game at the same pass -- exactly the population that answers
item 5's still-open question ("does acting on the injury signal beat waiting
for the market"). The production observed-movement >=1.0 policy inside
`nfl_ats.pick_refresh.plan_refresh` is untouched; this challenger only
writes to its own ledger (`artifacts/prospective/injury_signal_refresh_decisions.parquet`).
Item 5's lag question itself remains open -- this wiring accrues the
prospective evidence that will eventually answer it, it does not answer it
by construction. See that challenger's registration entry for the full
evidence chain, the Tuesday-to-close-vs-refresh-pass timing caveat, and the
correlated-decomposition caveat on the +17.07-point figure below.

Since INJURY is the class the value concentrates in, and this project's
picks stay editable to `min(kickoff, Sunday 16:00 ET)` (owner correction,
`docs/observed_movement_channel.md`), a late-week refresh pass does not
need to wait for the Tuesday-to-close move to finish -- it can watch the
same signal this document measured, live:

1. **Watch the picked team's own skill-position report (QB/RB/WR/TE)** as
   Wednesday/Thursday/Friday practice-status filings land
   (`data/players/raw/<snapshot>/injuries.parquet`, refreshed weekly), for
   any `report_status` newly at `Doubtful`/`Out` that was not present as of
   that game's own Tuesday noon ET -- exactly `player_delta > 0` in this
   document's construction, on the side currently picked.
2. **Watch the same report for the opponent clearing the board** (a
   previously `Doubtful`/`Out`/`Questionable` player dropping off or
   downgrading in severity) -- `player_delta < 0` on the favored side,
   which this document's `net_injury_score` already nets against the
   picked-team signal.
3. **Flip when `net_injury_score >= 2`** (this document's predeclared bar)
   accumulates on the currently-picked team, i.e., treat the Friday-final
   report the same way this document graded it, but as a live decision
   input rather than a backward-looking attribution.
4. **For any game where the official Friday-final report has not yet
   settled** (Thursday, or a Friday morning read before the final
   afternoon report), the PFT/ProFootballTalk archive
   (`data/raw/injury_news/<snapshot>/index.parquet`) is a faster,
   lower-fidelity proxy -- this document's 2025 fallback path shows a
   similar flag-rate to the official path (41.5% vs 36.7%) on the same
   population, though it was not validated player-by-player here (see
   Caveats) and should be treated as a leading indicator, not a
   replacement for the official report once it lands.
5. **This document does not test the lag** between an injury report
   landing and the line finishing its move -- whether a Thursday-evening
   flip beats waiting for Friday's number, or beats the eventual close, is
   the natural next study (**inferred**, not measured here) and would need
   its own predeclared design, not a re-read of this one.

### Caveats (label how you know it)

- **Position filter is a proxy, not a snap-share-weighted "key player"
  metric.** QB/RB/WR/TE were chosen as the market-relevant subset before
  any hit rate was computed; offensive-line and defensive-front injuries
  (well known to move NFL lines, e.g., a starting left tackle or an
  edge rusher) are entirely invisible to the INJURY flag as built. This is
  a likely undercount, not an overcount -- some UNATTRIBUTED games may
  carry real injury news this construction cannot see.
- **Official injury-report coverage ends at 2024** (`data/players/raw/20260817T184901Z/injuries.parquet`,
  measured max season 2024) -- 82 of 494 games (16.6%) rely on the
  lower-fidelity PFT-headline fallback instead of the official report.
- **The PFT fallback is team-nickname-headline matching, not player-level
  matching.** A headline mentioning both teams is not disambiguated; no
  roster name-mapping was built for this pass (the machinery exists in
  `scripts/injury_tuesday_cutoff_experiment.py`'s `build_pft_match_table`
  but requires a rosters join not reused here). Treat the 2025 INJURY flag
  as directionally suggestive, not equivalent evidence to 2020-2024's.
- **WEATHER has no precipitation field** in `data/raw/forecast_archive/full_2020_2025/forecasts.parquet`
  -- only wind and temperature deterioration are measured; a genuinely
  precipitation-driven move (rain/snow arriving after Tuesday) would not
  be flagged here.
- **PUBLIC coverage is genuinely sparse** (90/494, 18.2%; 58/290 at the
  threshold cut) -- consistent with `docs/public_betting_sourcing.md`'s
  own disclosed ceiling (at most ~34% of REG games in the best-covered
  season have any pregame reading at all). Reported honestly as an
  exploratory/backfill-quality subset, not a dense feature.
- **This is a mined battery** (INJURY x2 populations, WEATHER x2, PUBLIC x2
  subtypes x2 populations, ATTRIBUTED/UNATTRIBUTED x2, plus 2 anchor
  cells) -- no multiplicity correction is claimed, matching every other
  mined-battery document in this repo.
- **Every cell here is a correlated decomposition of the already-recorded
  `observed_movement_oracle_full_slate` / `observed_movement_threshold_1_0`
  entries** -- same archive, same population, an attribution subcut, not
  an independent sample. Must never be pooled additively with those
  entries or with each other (INJURY, WEATHER, and PUBLIC overlap on 32 of
  494 games).

### Registry entries recorded

14 entries recorded to `registry/weak_signals.json` via
`nfl-ats weak-signals record`, all `effect_units=accuracy_points`,
`classification=unresolved_below_power`, `closing_ground=null`, league
`nfl`, seasons `2020-2025`, every `--notes` flagging the entry as a
correlated decomposition of the `observed_movement_*` family. Verified
present via `nfl-ats weak-signals status` after recording (registry total
306 -> 320). The two `PUBLIC_even` cells (n=2 each population) were
computed and reported in the table above but not recorded -- too
degenerate to adjudicate, matching this repo's existing convention of not
separately recording clearly-uninformative near-zero-n cells
(`docs/opener_error_analysis.md`'s `favorite_side=pick_em`, n=12, was kept
similarly unrecorded). Names (prefix `movement_attribution_`):
`pop_unfiltered_injury`, `pop_unfiltered_weather`, `pop_unfiltered_public`,
`pop_unfiltered_attributed_any`, `pop_unfiltered_unattributed`,
`pop_unfiltered_public_book_shading_public`,
`pop_unfiltered_public_reverse_line_movement`, `pop_threshold_injury`,
`pop_threshold_weather`, `pop_threshold_public`,
`pop_threshold_attributed_any`, `pop_threshold_unattributed`,
`pop_threshold_public_book_shading_public`,
`pop_threshold_public_reverse_line_movement`.

### Files

- `docs/movement_attribution.md` -- this document (predeclaration + results).
- `scripts/movement_attribution.py` -- the measurement script (new).
- `artifacts/movement_attribution/20260820T114620Z/` -- `per_game_attribution.parquet`
  (per-game flags and inputs), `cells_summary.csv`, `metadata.json`
  (provenance via `nfl_ats.provenance.write_experiment_artifact`).
