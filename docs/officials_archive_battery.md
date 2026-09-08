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

*(This section was written after the predeclaration above and after the runs.
Every number carries the command that produced it.)*
