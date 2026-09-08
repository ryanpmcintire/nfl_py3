# Turning the officials archive on for the crew battery (LEAD-59 / LEAD-33)

Lane AD, 2026-09-08. **This predeclaration section was written before any
outcome — any accuracy, cover rate, graded delta or interval — was computed.**
The one thing measured before it was written is the Part A trait- and
flag-change accounting (how many games' trait values move when the archive is
switched on): that is a diagnostic of the data plumbing, it reads no game
result, and it decided how much of Part B was worth running. It is reported
below with its own measured-first label.

Everything here runs read-only. It writes only
`artifacts/research/laneAD/**`, this document, `scripts/officials_archive_battery_eval.py`
and `tests/test_officials_archive_battery.py`. It never flips
`INCLUDE_ARCHIVE_DEFAULT`, never edits `src/nfl_ats`, never publishes, and
never records to the registry itself — every record command it would run is
written to `artifacts/research/laneAD/record_commands.ps1` for the
coordinator.

## Binding closing-grounds taxonomy (verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". The registry code
hard-rejects inadmissible closures; if a record command errors, the verdict is
wrong, not the validator.

Nothing in this lane "needs more games". The seasons are what they are; the
decision is expected value.

## What the battery is (read, before anything was run)

Five things in this repository compute an officiating-crew trait. Each is
named here with the file and line that defines it, because "the battery"
is not one object:

1. `nfl_ats.experiment_runner._build_referee_trait_data`
   (`src/nfl_ats/experiment_runner.py:1282`) — per-game head-referee
   `prior_seasons_experience`, `lag_penalty_rate_quartile`,
   `lag_home_away_diff_quartile`. Feeds `docs/referee_battery.md`'s six
   close-graded cells and LEAD-31's rookie flag.
2. `nfl_ats.experiment_runner._build_referee_type_trait_data`
   (`src/nfl_ats/experiment_runner.py:1656`) — per-referee lagged
   defensive-pass-interference and offensive-holding rate quartiles. Feeds
   `nfl_ats.crew_tilt_refresh_overlay`, the one crew cell registered as a
   live prospective challenger (`crew_tilt_refresh_v1`).
3. `nfl_ats.officials_flag_features.home_away_penalty_game_table`
   (`src/nfl_ats/officials_flag_features.py:111`) — the home-minus-away
   penalty differential table behind LEAD-32's reliability read and its
   trailing-home-bias flag.
4. `nfl_ats.officials_flag_features.crew_familiarity_table` (`:327`) —
   LEAD-34's crew second-meeting flag.
5. `nfl_ats.officials_flag_features.describe_referee_left_censoring` (`:447`)
   — the disclosure of how many referees' tenure is unknowable from the feed.

Two of these are graded at the opener against the active model, stacked on
production `weak_stack`, through `scripts/officials_flags_on_production.py`:
`crew_second_meeting_favorite_on_production` (LEAD-34) and
`rookie_crew_underdog_on_production` (LEAD-31). Read
(`registry/rotation_registry.json:959`, `:6534`): both windows are `[2020,
2021]`, both `spent`, both `unresolved`, effects -0.6579 and -1.0965
accuracy points. Read: neither flag appears in `nfl_ats.overlay_composition`
or `nfl_ats.four_overlay_composition`, so **no battery cell is a member of
the played card**; the only crew cell tracked as a challenger is
`crew_tilt_refresh_v1`, which is a late-week refresh channel that by its own
contract never alters the played pick.

## Arms (frozen)

Active model, read from `artifacts/active_ats_model.json`: `a4c757efd2525da6`,
`weak_stack`, ridge alpha 10, `gaussian_median`, served walk-forward home-side
offset. The opener-graded population is whatever
`nfl_ats.clv.opener_pick_evaluation` scores against the Tuesday-opener
consensus store: 2020-2025, because that is the only span the opener store
covers (measured: 239/252/268/284/285/285 games with a `tue_open` quote for
2020-2025 and nothing before).

**Replay gate first.** Before any arm is scored, the baseline is re-run and
compared with the served opener evaluation artifact that matches the active
model. The lane stops if the served probability-rule accuracy does not
reproduce exactly.

| Arm | Column | Officials source |
| --- | --- | --- |
| A0 | none (production `weak_stack`) | n/a |
| A1 | `rookie_crew_underdog_flag` | shipped path, archive OFF |
| A2 | `rookie_crew_underdog_flag` | shipped path, `include_archive=True` at the loader — the ONLY change |
| A3 | `rookie_crew_underdog_flag` | archive-aware tenure (below) |
| B1 | `crew_second_meeting_favorite_flag` | shipped path, archive OFF |
| B2 | `crew_second_meeting_favorite_flag` | shipped path, `include_archive=True` |
| B3 | `crew_second_meeting_favorite_flag` | archive-aware crew population |

**Archive-aware tenure (A3/B3)** is the trait exactly as
`docs/referee_battery.md` and `docs/officials_crew_leads.md` DEFINE it — "the
count of distinct PRIOR seasons that official appears as `Referee` in this
dataset" — computed over the officials table itself rather than over the
subset of it that survives an inner join to a per-game penalty aggregate. It
is a second declared arm, not a claim that the shipped implementation is
wrong about what it stores; it is the arm that answers "what would the archive
be worth if the plumbing let it through".

Each candidate arm is scored twice: **marginal on production** (candidate vs
A0, the number the rotation registry already holds for A1 and B1) and
**archive delta** (A2 vs A1, A3 vs A1, B2 vs B1, B3 vs B1). Predeclared: when
two arms' candidate columns are bit-identical on every game, their feature
tables are identical, the model is deterministic, and the paired delta is
exactly 0.000 with zero picks changed — that is asserted by column comparison
and no bootstrap is run for it. This is stated in advance so that a zero is
reported as a zero rather than dressed up as an interval.

Through-card read: `nfl-ats overlay-composition --per-game-artifact` on any
arm whose opener picks differ from A0's, using the played three-member union
(the spread-gap zone flip was retired from the played card on 2026-09-07).

## Grading

- Opener grade, probability rule (`home_cover_probability_at_open >= 0.5`) —
  what production plays.
- Paired, same games both arms. Week-blocked bootstrap, 20,000 draws, seed
  20260817. **Within-week correlation is ZERO by owner mandate; it is never
  estimated and never padded.**
- Report `probability_positive`, never "contains zero".
- Primary window 2020-2025 (every opener-graded game). The recorded
  `[2020, 2021]` subset is reported beside it because that is the window the
  registry rows hold; it is a reused window and carries that discount, not a
  ban.

## Part C: the era-extended read

The archive's whole point is games the feed cannot see. Those are graded with
the project's own pre-opener recipe, read from `docs/spread_regime_program.md`
lines 36-41 and implemented in `scripts/spread_regime_opener_eval.py`'s
`build_stream`: a weekly walk-forward ridge on completed games strictly before
the week's first kickoff, minimum 500 training rows, scored at the archived
nflverse spread as a CLOSE PROXY where no Tuesday opener exists. Grade labels
stay explicit; a close-proxy number never stands in for an opener number.

The feature table starts in 2009, and 500 prior completed games is roughly two
seasons, so the earliest gradeable season is 2011. That is a property of the
model's training requirement, not of the archive: the archive supplies crews
from 2009, and 2009-2010 crews are used as PRIOR seasons for the 2011-2014
traits even though those two seasons cannot themselves be graded.

Eras, reported as magnitudes per era and never as absence:

| Era | Grade | Crew source |
| --- | --- | --- |
| 2011-2014 | nflverse spread (close proxy) | archive only — impossible without it |
| 2015-2019 | nflverse spread (close proxy) | feed |
| 2020-2025 | Tuesday opener | feed |

## Part D: LEAD-33's crew-composition statistic

LEAD-33's source gap is not closed and no proxy for an all-star designation is
invented (`ROADMAP.md:962`; `docs/officials_archive.md`). What the archive adds
is that the full seven-person crew is now available for 2009-2014 as well as
2015-2025, which makes crew COMPOSITION measurable across two officiating eras.

**Definition, mechanical.** Within a season, each official has a modal crew:
the referee under whom that official works most often. For a game, count how
many of the seven on-field positions are filled by an official whose modal
crew's referee is NOT this game's referee. Two variants:

- `positions_off_modal_crew_season` — modal crew from the official's whole
  season. **Descriptive only.** It uses games after the one being described,
  so it may never be graded against an outcome.
- `positions_off_modal_crew_prior` — modal crew from that official's games
  strictly EARLIER in the same season; ties broken by the most recent prior
  game's referee; an official with no prior game this season is unresolved and
  is not counted. `positions_resolved` records how many of the seven were
  resolvable. **This is the pregame-safe variant and the only one graded.**

**The graded cell.** `crew_scramble_backs_favorite`: on a flagged game, back
the favourite at the line; on every other game, keep the model's own pick.
Flagged when `positions_off_modal_crew_prior >= 4` — a strict majority of a
seven-person crew drawn from other crews. Predeclared contingency, fixed here
before the distribution was read: if fewer than 30 games in the graded window
are flagged at 4, the threshold falls back to 3 and the fallback is reported
as such. Direction: **BACK the favourite**, which is LEAD-33's own predeclared
direction on `ROADMAP.md:962` ("hand-picked late-season crews call tighter,
neutralizing physical underdogs"), applied to the only measurable handle on a
hand-picked crew — a crew assembled out of several regular crews.

This cell is a pick RULE on top of the served picks, not a feature in the
ridge, because adding a feature to the model would require registering a new
feature profile in `src/nfl_ats`, which this lane may not edit. That is a real
limitation and it is disclosed rather than worked around: a rule-overlay read
and a stacked-feature read are not the same measurement.

**Disclosed as descriptive.** This is one predeclared cell on a mined
population, with the direction fixed in advance and no threshold search. The
distribution of the statistic by season and by week-of-season is reported
plainly alongside.

## Recording plan

Every cell that produces an effect is written as a `nfl-ats weak-signals
record` line into `artifacts/research/laneAD/record_commands.ps1`, family
`officials_archive_battery`, category `onfield`, classification
`unresolved_below_power` unless one of the two admissible closing grounds
applies, with a `--plain-summary` in pool-player words. This lane does not run
those commands; the coordinator does.

---

# Measured results

*(Written after the predeclaration above and after the runs. Every number here
was produced by `.\.tools\uv.exe run --no-sync python
scripts/officials_archive_battery_eval.py --stage <stage>` on 2026-09-08, and
every stage writes a provenance-stamped artifact under
`artifacts/research/laneAD/`.)*

## The decision, first

**Turning the archive on for the served battery, on its own, changes nothing —
not one pick, not one trait value, on any game the pool grades.** That is not a
small effect; it is an exact identity, and the reason is a plumbing detail, not
a fact about officiating. Two things have to change alongside the loader flag
before the 2009-2014 crews can reach a played number, and once they do, the
measured direction is favourable:

- Through the **served opener harness** (replay gate passed), the rookie-crew
  rule rebuilt with the archive and a close-proxy line beats its recorded
  feed-only build by **+0.73 accuracy points, 95% [+0.00, +1.48],
  `probability_positive` 0.972**, and moves from -0.60 against production to
  **+0.13, [-0.20, +0.53], P+ 0.700**. The repeat-crew rule gains **+0.80,
  [-0.73, +2.24], P+ 0.849** over its own feed-only build.
- LEAD-33's crew-scramble rule is **-0.33 standalone (P+ 0.081)** but **+0.20
  through the played card, [-0.27, +0.67], P+ 0.759** and **+0.33 through the
  four-member union, P+ 0.904**.

Both of those carry a second-look discount stated in full below. Neither is a
promotion; both are worth more than the zero the shipped wiring currently
delivers.

## Which seasons this lane actually used (read at run time, and it moved)

`describe_archive_coverage()` was re-read at the start and the end of the lane,
because the 2009-2013 Wayback sweep was still running the whole time. It grew
underneath the runs:

| Read | Archived games | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| start of lane | 1,452 | 248 | 254 | 254 | 256 | **185** | 255 |
| end of lane | 1,518 | 248 | 254 | 254 | 256 | **251** | 255 |

Only 2013 moved. The `traits` stage saw 1,486 extra referee-games, the `era`
stage 1,507, and the coverage is 1,518 games (1,517 complete crews) as this is
written. **So the 2011-2014 era rows are graded on a 2013 that was still
filling**, and re-running the era stage today would score slightly more 2013
games. That is a reproducibility caveat on the era block, not on Part A or Part
B (both of which are confined to 2015-2025 games the archive never touches).
The seasons used are **2009-2014 from the archive** and **2015-2025 from the
nflverse feed**, merged with nflverse winning any overlap.

## Part A: what the loader flag actually moves (stage `traits`)

Artifact: `artifacts/research/laneAD/traits.json`. No game outcome is read
here.

| Trait / consumer | Rows, archive off → on | Games whose value changed |
| --- | ---: | ---: |
| `_build_referee_trait_data.game_trait` `prior_seasons_experience` | 2,892 → 2,892 | **0** |
| ... its `lag_penalty_rate_quartile` | 2,892 → 2,892 | **0** |
| ... its `lag_home_away_diff_quartile` | 2,892 → 2,892 | **0** |
| `home_away_penalty_game_table` (whole frame) | 2,892 → 2,892 | **0** (frames compare `equals`) |
| `rookie_crew_underdog_flag` | 2,892 → 2,892 | **0** |
| `crew_second_meeting_favorite_flag` | 2,892 → 2,892 | **0** |
| `describe_referee_left_censoring` | 29 → **53** officials | 2015-censored debuts **17 → 1** |

**Why every trait is a no-op.** `_build_referee_trait_data` builds its
per-(official, season) aggregates from `merged_games`, an INNER join of the
referee assignments to `data/raw/officials/<snapshot>/game_penalties.parquet`
(measured: 3,028 rows, seasons 2015-2025 only). Every 2009-2014 archive crew is
dropped by that join before a single trait is computed, so tenure and both
quartiles come out bit-identical. `home_away_penalty_game_table` has the same
inner join and the same outcome. The premise this lane was given — "the archive
changes tenure and quartile cut-points, so it is not a no-op even on those
seasons" — is **falsified by measurement**: it would be true if the trait
builders read the crew table, and they read the penalty table instead.

**One consumer does not no-op; it raises.**
`_build_referee_type_trait_data` (the builder behind `crew_tilt_refresh_v1`,
the only crew cell registered as a live challenger) LEFT-joins its penalty-type
counts and `fillna(0.0)`. With the archive on, all 1,486 extra referee-games
enter with a fabricated zero-penalty count, the lower quartile cut-points
collapse onto each other, and `pd.qcut` refuses:

```
ValueError: Bin edges must be unique: Index([0.0, 0.0, 0.75, 1.0666666666666667, 1.9375], ...)
```

Both penalty types fail this way (defensive pass interference and offensive
holding). This is the good outcome — it fails closed rather than quietly
scoring 2009-2014 crews as if they never drew a flag — but it means **the one
tracked challenger cannot be run with the archive on at all** without a change
in `src/nfl_ats/experiment_runner.py`, which this lane may not make. That is
the "cannot be re-run without editing src" case, named precisely.

**The archive-aware tenure arm.** Recomputing tenure the way the docs define it
(distinct prior seasons the official appears as `Referee`, no penalty gate)
reproduces the shipped values exactly on all 2,892 feed games when the archive
is off (0 of 2,892 differ — a pinned second path), and with the archive on it
moves **1,691 of 2,892 games (58.5%), mean +2.39 prior seasons, max +6**.
Referee-games rise 2,892 → 4,378 and (official, season) pairs 187 → 308.

But that large tenure movement produces almost no movement in the *rookie*
label, because by 2018 a 2015-debut referee is no longer "first or second
season" either way:

| Season | Rookie-crew games, archive off | archive on |
| --- | ---: | ---: |
| 2015 | 0 | 0 |
| **2016** | **256** | **16** |
| 2017 | 0 | 0 |
| 2018 | 60 | 60 |
| 2019 | 105 | 105 |
| 2020-2025 | 60 / 31 / 16 / 32 / 16 / 16 | identical |

The entire left-censoring correction lands in **2016**, and 2016 is outside the
opener-graded window.

## Part B: opener-graded, against the active model (stage `opener`)

Artifacts: `opener.json`, `opener_picks.parquet`.

**Replay gate passed** before anything was scored: the harness reproduces the
served opener evaluation `artifacts/opener_evaluation/20260908T115957Z`
on all 1,537 games, maximum probability gap **3.8e-15**, **0** pick
disagreements, feature digest `457aafb7…`.

All seven arms, 2020-2025, 1,503 non-push games, 107 week blocks, week-blocked
bootstrap 20,000 draws seed 20260817, within-week correlation zero:

| Cell | Effect (pts) | 95% CI | P+ | Picks changed |
| --- | ---: | --- | ---: | ---: |
| rookie-crew rule vs production (all three archive variants) | -0.599 | [-1.340, +0.068] | 0.037 | 33 |
| repeat-crew rule vs production (both archive variants) | -0.931 | [-2.303, +0.473] | 0.087 | 86 |
| archive-on vs archive-off, either rule | **exactly 0.000** | — | — | **0** |

The archive-delta rows are an identity, not an estimate: the candidate columns
are equal on every game, so the fitted model, the picks and the delta are the
same object twice. They are reported as zeros and are deliberately NOT written
as registry signals with a degenerate `probability_positive` — the
`record_commands.ps1` file carries a `# SKIPPED` line saying so for each.

On the recorded `[2020, 2021]` sub-window this harness gives the rookie rule
-1.316 [-2.995, +0.214] P+ 0.027 against the registry's recorded -1.0965
[-2.6906, +0.2217] P+ 0.0456, and the repeat-crew rule -0.439 [-3.965, +3.516]
P+ 0.371 against the recorded -0.6579 [-4.2129, +3.2967] P+ 0.32515. Same sign,
same magnitude, not identical — the 2026-09-05 screens ran before the served
home-side offset existed and used the harness's own ECDF probability default
rather than the active model's `gaussian_median`. Read that as agreement on the
finding, not as a bit-for-bit reproduction.

## Part C: the era-extended read (stages `era`, `served-proxy`)

### Amendment, disclosed

The predeclared Part C could not answer its own question, and the reason is a
second 2015-shaped constant: `ROOKIE_ELIGIBLE_SEASON_FLOOR = 2016`
(`src/nfl_ats/officials_flag_features.py:78`) makes the rookie flag identically
zero on every pre-2016 game no matter what the archive knows, and
`crew_familiarity_table` inherits the penalty-table gate, so the repeat-crew
flag is identically zero before 2015. Measured: the three predeclared arms
score **0 flips and exactly 0.000** on 2011-2014.

So two arms were added **after** that zero was seen, each the shipped rule with
exactly one constraint lifted:

- `rookie_archive_era_floor` — the shipped rookie rule with the season floor
  set to the archive's own first season plus one (2010), mirroring how 2016 was
  derived from the feed's 2015.
- `second_meeting_archive_era` — the shipped repeat-crew rule rebuilt on the
  archive-extended referee table instead of the penalty table. **Pinned**: it
  reproduces the shipped builder's flag on all 2,892 shared games with **0
  disagreements** before it is used.

These two are a second look at a window whose first look was already read.
They are descriptive, they are not confirmation, and every number below carries
that discount.

Nonzero flag counts show what the archive buys: pre-2015 games carrying a flag
at all — 389 for the rookie rule, 412 for the repeat-crew rule, against **0**
for both feed-only builds.

### Per-era magnitudes (never absence)

Walk-forward ridge, training strictly before each week, minimum 500 training
games (so 2009-2010 cannot be graded — the model's training requirement, not
the archive's coverage; those two seasons still serve as prior seasons for the
2011-2014 traits). Pre-2020 rows are graded at the archived nflverse spread as
a **close proxy** and that label never stands in for an opener number.

| Era | Grade | Arm | Effect | 95% CI | P+ | Flips | n / weeks |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
| 2011-2014 | close proxy | rookie, feed only | 0.000 | — | — | 0 | 996 / 68 |
| 2011-2014 | close proxy | rookie, archive floor | -0.201 | [-0.908, +0.596] | 0.254 | 16 | 996 / 68 |
| 2011-2014 | close proxy | repeat-crew, feed only | 0.000 | — | — | 0 | 996 / 68 |
| 2011-2014 | close proxy | repeat-crew, archive | +0.100 | [-0.599, +0.711] | 0.578 | 9 | 996 / 68 |
| 2015-2019 | close proxy | rookie, feed only | +0.161 | [-1.117, +1.449] | 0.572 | 66 | 1,240 / 85 |
| 2015-2019 | close proxy | rookie, archive tenure | **+0.403** | [-0.082, +0.902] | **0.919** | 15 | 1,240 / 85 |
| 2015-2019 | close proxy | rookie, archive floor | +0.081 | [-0.401, +0.566] | 0.566 | 11 | 1,240 / 85 |
| 2015-2019 | close proxy | repeat-crew, feed only | +0.242 | [-1.267, +1.787] | 0.595 | 85 | 1,240 / 85 |
| 2015-2019 | close proxy | repeat-crew, archive | -0.323 | [-1.123, +0.410] | 0.177 | 26 | 1,240 / 85 |
| 2020-2025 | opener | rookie, feed only | -0.317 | [-1.064, +0.438] | 0.180 | 33 | 1,579 / 107 |
| 2020-2025 | opener | rookie, archive tenure | -0.190 | [-0.758, +0.318] | 0.219 | 19 | 1,579 / 107 |
| 2020-2025 | opener | rookie, archive floor | **+0.190** | [-0.063, +0.503] | **0.878** | 7 | 1,579 / 107 |
| 2020-2025 | opener | repeat-crew, feed only | -0.190 | [-0.701, +0.316] | 0.196 | 15 | 1,579 / 107 |
| 2020-2025 | opener | repeat-crew, archive | +0.127 | [-0.378, +0.633] | 0.641 | 14 | 1,579 / 107 |

The 2011-2014 magnitudes are small and both signs appear; they are recorded
`unresolved_below_power`, and the honest statement is that this is the first
time either rule could be measured there at all. The 2015-2019 row for
archive-aware tenure (**+0.403, P+ 0.919**) is the strongest cell in the lane
and sits exactly where the left-censoring correction bites.

### The decision-grade version (stage `served-proxy`)

The era stream is its own recipe (no served home-side offset), so the arms that
looked good there were re-scored through the SAME served harness Part B's
replay gate validated — replay gate passed again, 1,537 games, gap 3.8e-15, 0
disagreements. The only relaxation is the flag's own line source: the archived
nflverse spread fills the pre-2020 seasons the Tuesday-opener store cannot
reach.

| Cell | Effect | 95% CI | P+ | Flips |
| --- | ---: | --- | ---: | ---: |
| rookie, archive + proxy line, vs production | +0.133 | [-0.203, +0.528] | 0.700 | 10 |
| rookie, archive + proxy line, **vs its opener-line twin** | **+0.732** | [+0.000, +1.483] | **0.972** | 33 |
| repeat-crew, archive + proxy line, vs production | -0.133 | [-0.665, +0.397] | 0.267 | 18 |
| repeat-crew, archive + proxy line, **vs its opener-line twin** | **+0.798** | [-0.730, +2.240] | **0.849** | 96 |

Those two "vs its opener-line twin" rows are the archive-delta cells that Part
B could only report as zero. The mechanism is explicit: the archive-extended
flag has real values on 2009-2019 training games, which changes the fitted
coefficient the model brings into 2020-2025. Both cells are
`unresolved_below_power` and both carry the second-look discount.

## Part D: LEAD-33's crew-composition statistic (stages `composition`, `card`)

Artifacts: `composition.json`, `crew_composition.parquet`, `card.json`.

Computed over **all seventeen seasons, 2009-2025** — which is only possible
with the archive on; the feed alone reaches 2015. Distribution of
`positions_off_modal_crew_prior` over 4,385 games: 3,322 at 0, 858 at 1, 112 at
2, 17 at 3, 15 at 4, 13 at 5, 47 at 6, 1 at 7. Scrambled crews are rare and the
tail is lumpy.

By season (mean positions off the referee's modal crew, prior-games-only):

| 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.23 | 0.13 | 0.25 | **0.53** | 0.17 | 0.40 | 0.35 | 0.45 | 0.43 | 0.40 | 0.19 | **1.13** | 0.34 | 0.25 | 0.23 | 0.18 | 0.33 |

The two spikes are 2020 (1.13) and 2012 (0.53) — the COVID season and the
replacement-referee season. Both are real historical events and both are the
kind of thing the pre-2015 half of the archive exists to make visible; neither
is verified here against an outside source.

By week of season: week 1 is structurally unresolved (no prior games, mean
0.011 with 0.01 of 7 positions resolvable), the mid-season plateau sits near
0.30-0.40, and the elevated weeks are **13 (0.54), 17 (0.47) and 16 (0.45)**,
with week 18 the lowest of all (0.15). LEAD-33's premise expects late-season
crews to be reshuffled; weeks 13-17 lean that way and week 18 does not.
Descriptive, one look, no claim.

**The graded cell.** Threshold 4 as predeclared; the fallback to 3 was NOT
needed (37 flagged games in the graded window against the predeclared minimum
of 30). Backing the favourite on those games:

| Read | Effect | 95% CI | P+ | Flips |
| --- | ---: | --- | ---: | ---: |
| Standalone, against the served picks | -0.333 | [-0.916, +0.196] | 0.081 | 19 |
| **Through the played three-member card** | **+0.200** | [-0.268, +0.667] | **0.759** | 19 |
| Through the four-member union (zone included) | +0.333 | [-0.130, +0.813] | 0.904 | 19 |

The standalone and composed reads point opposite ways on the same 19 games.
That is the composition effect this project has already recorded elsewhere, and
it is the reason the composed read is the one that matters for a decision.

## What this means for LEAD-59 and LEAD-33

1. **Flipping `INCLUDE_ARCHIVE_DEFAULT` alone is safe and worthless.** It moves
   no trait, no flag, no pick — and it breaks
   `_build_referee_type_trait_data` outright. Do not flip it on its own.
2. **Three specific changes in `src/nfl_ats` are what the archive is waiting
   on**, in decreasing order of measured value: (a) let the crew traits be
   built from the crew table rather than gated on the 2015-2025 penalty
   aggregate; (b) derive the rookie season floor from the loaded population's
   own first season instead of the constant 2016; (c) make
   `_build_referee_type_trait_data` refuse (or drop) games with no penalty-type
   coverage instead of filling them with zero.
3. **With (a) and (b) done, the crew battery's own rules move from clearly
   worse than production to slightly better** — +0.73 and +0.80 accuracy points
   against their recorded builds, P+ 0.972 and 0.849. That is the number the
   next promotion look should start from.
4. **LEAD-33's source gap is still open** — no all-star marker exists in either
   source and none is invented here — but the crew-composition handle is now
   computable over seventeen seasons, and its composed read (+0.200, P+ 0.759)
   is the first LEAD-33 number this project has ever had.

Everything above is `unresolved_below_power`. Nothing here is refuted: no
interval sits wholly on the wrong side of zero, and no positive control bounds
any of it.
