# Pace-mismatch dog tilt overlay: a no-window-cost prospective challenger

Written 2026-09-01 (ORCH-C worker W6). Follows the
`surface_switch_tilt_overlay` / `spread_gap_zone_fade_overlay` /
`pbp08_protection_mismatch_tilt_overlay` precedent
(`docs/surface_switch_tilt_overlay.md`, `docs/spread_gap_zone_fade_overlay.md`,
`docs/pbp08_matchup_screen.md`) for wiring a pick-level, post-prediction
transform into the prospective challenger ledger at zero rotation-registry
window cost.

## Evidence (read from the registry before building, as assigned)

`registry/weak_signals.json:team_style_pace_mismatch_dog_cover`, PBP-08 team
"personality" battery (`scripts/team_style_screen.py`, predeclaration
`docs/team_style.md`, one of 5 predeclared cells, mined lineage, uncorrected
multiplicity across the battery), `unresolved_below_power`:

Top-quartile absolute difference between the two teams' PRIOR-SEASON,
league-centered `seconds_per_play_pace`, scored on `dog_cover` (the
underdog's cover indicator; true pick'ems are undefined and dropped from
this cell's population, not folded into the complement). REG 2009-2025,
n=4,313 games (n_flag=1,018, n_missing_required_data=248):

| Read | Window | n | Effect (accuracy points) | 95% interval | `probability_positive` |
|---|---|---|---|---|---|
| Week-blocked (primary) | REG 2009-2025 | 4,313 | +0.2292 | [-0.5587, +1.0401] | 0.71125 |
| Season-blocked (secondary) | REG 2009-2025 | 4,313 | +0.2292 | [-0.1830, +0.6153] | 0.8711 |

**Reliability** (the highest of the 9 style dimensions in this battery):
`seconds_per_play_pace`'s YoY Pearson r **+0.489**, 95% CI [+0.405, +0.567],
n=512 team-season pairs (`scripts/team_style_screen.py:374-376`).

**Direction check, done and PASSED, not skipped.** Measured directly from
`artifacts/team_style_screen/20260819T210011Z/results.json`: in flagged
(top-quartile pace-mismatch) games the underdog covers `subset_mean`
0.518664 (51.8664%) against a `complement_mean` 0.508953 (50.8953%) field --
the underdog covers MORE in the flagged population than outside it, exactly
the predeclared POSITIVE-on-`dog_cover` direction
(`scripts/team_style_screen.py:479-483`: "variance mechanism: fewer
possessions favour the dog"). A sibling cell in this same predeclared batch
measured the opposite of its own predicted direction and was NOT built into
an overlay; this one passed and is.

**The week-blocked interval crosses zero.** Per AGENTS.md/CLAUDE.md, at this
evaluator's ~2-point resolution that is the EXPECTED shape for a real small
signal, never grounds to decline building a no-window-cost prospective
challenger. Neither admissible closing ground applies (no resolved wrong
sign -- the point estimate and the season-blocked secondary both sit on the
predicted side; no positive-control bound was run for this cell), so it
stays `unresolved_below_power`.

**Battery-multiplicity caveat, stated up front, not buried.** This is one of
5 predeclared cells (2 identity, 3 matchup: this pace-mismatch cell, a
short-game-vs-pressure-defense cell, and a deep-ball-outdoor-wind cell). No
multiple-comparison correction is applied across the battery; the
`probability_positive` figures above are this ONE cell's own bootstrap read.

## Trait and quartile cut, transcribed verbatim from the screen

- `seconds_per_play_pace` -- drive time-of-possession divided by drive play
  count, pooled directly from plays/drives per (season, team), NOT an
  average of per-game rates (avoids Simpson's-paradox bias from uneven
  per-game play counts) -- `scripts/team_style_features.py:358-366`
  (`build_team_season_style`'s pace block) via
  `scripts/team_style_features.py:162-184` (`_drive_pace_table`, reusing
  `nfl_ats.pbp.build_drive_table` verbatim, the same drive aggregation the
  production PBP-05 pipeline uses).
- "Centered" -- each dimension minus ITS OWN SEASON's unweighted
  across-team mean, so leaguewide pace drift over 2009-2025 does not read
  as a team identity -- `scripts/team_style_features.py:413-422`
  (`add_league_centered`).
- Prior-season join -- `scripts/team_style_screen.py:151-161` (`_prior`):
  shift the (season, team) table forward one season, so joining on `season`
  pulls the PRIOR season's centered value onto this season's game.
- `pace_diff_abs = abs(home_prior_pace_centered - away_prior_pace_centered)`
  -- `scripts/team_style_screen.py:203-216` (`build_game_table`'s pace
  block).
- Quartile cut -- top quartile (`QUARTILE = 0.75`,
  `scripts/team_style_screen.py:76`) of `pace_diff_abs` over the REG
  population, computed over ALL REG games including pick'ems, before the
  pick'em population restriction for the `dog_cover` value column
  (`scripts/team_style_screen.py:366`). The MEASURED numeric threshold,
  frozen exactly as measured and never recomputed by this overlay (the same
  discipline `spread_gap_zone_fade_overlay.SPREAD_GAP_LOWER_BOUND`/
  `UPPER_BOUND` use): **2.1685022294778378**, read from
  `artifacts/team_style_screen/20260819T210011Z/results.json:pace_diff_abs_threshold`.
- Flag comparator -- `>=` the threshold
  (`scripts/team_style_screen.py:468`: `flag_b2 = game_b2["pace_diff_abs"] >= pace_threshold`).

## Spread convention, verified independently

`scripts/team_style_screen.py:121-123`: "spread_line > 0 -> HOME favored;
spread_line < 0 -> AWAY favored", cross-checked there against
`nfl_ats.features.add_ats_outcomes`'s `ats_margin = result - spread_line`
convention. Independently re-verified this session against real schedule
data (`data/raw/20260824T115346Z/schedules.parquet`): the 2013 week-6 DEN
(home)-vs-JAX (away) game carries `spread_line=+27.0` with Denver a known
lopsided home favorite; the 2019 week-2 MIA (home)-vs-NE (away) game carries
`spread_line=-18.0` with New England a known lopsided ROAD favorite. Both
confirm `spread_line > 0` means HOME favored and `spread_line < 0` means AWAY
favored.

## The rule, exactly as built (frozen, parameter-free)

```
pace_diff_abs = abs(home_prior_pace_centered - away_prior_pace_centered)

flag when:
    game_type == "REG"
    AND pace_diff_abs >= PACE_DIFF_ABS_THRESHOLD (2.1685022294778378)

flip when:
    flag is True
    AND spread_line != 0  (no defined underdog for a pick'em)
    AND the model's own pick is on the FAVOURITE:
        (spread_line > 0 AND home picked)  OR  (spread_line < 0 AND away picked)

flip target: the UNDERDOG (the complement of the model's current pick)
```

In plain language: **when the top-quartile prior-season pace mismatch fires
AND the model currently has the market favourite, flip the pick to the
underdog.** If the model already has the underdog, leave it. Pick'em games
are never touched. Missing prior-season pace data (a new franchise's first
tracked season, or an incomplete cache) means no flag, never an error.

Implemented in `src/nfl_ats/pace_mismatch_dog_tilt_overlay.py`:
`pace_mismatch_flag_by_game` derives the pregame-safe, data-derived flag
(schedule + the team-season pace cache); `apply_pace_mismatch_dog_tilt_overlay`
applies the pick-level transform; `overlay_disclosure_note` produces the
plain-English provenance sentence (not currently surfaced anywhere).

**Fail-open, like the PBP-derived sibling.** The team-season pace cache
(`data/pbp/team_style/team_season_style.parquet`) is a bespoke, gitignored,
network-fetched research artifact, NOT part of the standard captured
raw-schedule snapshot pipeline. `pace_mismatch_flags_fail_open` folds a
missing cache or schedule snapshot into ZERO flags and a documented no-op,
never an exception that could break a weekly record call -- mirroring
`pbp08_protection_mismatch_tilt_overlay.flags_for_week_fail_open`.

## Pregame safety

The trait is a PRIOR-SEASON aggregate (shifted by exactly one season before
joining), and the flag never reads `result`, `spread_line`, or any outcome
column. Three leakage regression tests in
`tests/test_pace_mismatch_dog_tilt_overlay.py` prove this empirically:
mutating a game's own outcome columns has no bearing; a future season's
schedule/style data never changes an earlier season's already-computed
flags; and -- the construct-specific check this rule needs -- a mutated
CURRENT-season pace row for a team playing THIS season never changes that
game's flag (only the season-minus-one row is ever read). The flip's own
favourite/underdog determination reads `spread_line`, which is on the card
before kickoff (the Tuesday-opener line for the pool's own decision).

## Stacked-on-production back-test (mined-seasons read, context not a gate)

`scripts/pace_mismatch_dog_stacked_backtest.py` applies this overlay ON TOP
OF the PLAYED four-member overlay union (`coach_fade_overlay`,
`division_revenge_tilt_overlay`, `player_arrests_back_side_policy`,
`spread_gap_zone_fade_overlay` -- `nfl_ats.four_overlay_composition.PLAYED_UNION`),
not a bare baseline, using the SAME frozen archive every other overlay
back-test uses (`artifacts/opener_evaluation/20260819T174244Z/per_game.parquet`,
the active weak_stack model's 1,537 REG games 2020-2025, graded at the
Tuesday opener; baseline 53.36% on 1,503 scored games; production -- the
PLAYED union alone -- 55.42% on this archive).

**Spread column used: `spread_line`, seeded from `tue_open_home_spread` by
`overlay_stack_backtest.build_predictions_frame`** -- the Tuesday-OPENER
decision line, matching (a) the pool's own primary goal (AGENTS.md: "grade
the decision at the opener"), (b) this archive's own opener grading, and (c)
the exact field every sibling overlay recorder reads for
`decision_home_spread`. The pace-mismatch flag itself never reads
`spread_line`; only the favourite/underdog side determination does.

Result (`artifacts/pace_mismatch_dog_stacked/20260901T195112Z/results.json`,
seed 20260901, 20,000 bootstrap samples):

| Blocking | Candidate minus production | 95% interval | `probability_positive` |
|---|---|---|---|
| Week-blocked (primary) | -0.4657 pts | [-1.9515, +0.9987] | 0.2498 |
| Season-blocked (secondary) | -0.4657 pts | [-1.3559, +0.6267] | 0.1680 |

`n_flipped`: 180 games flip on the archive under this overlay alone; 120 of
those are NET-NEW beyond what the played union already flips (60 overlap
with a member that had already flipped the same game, so they cannot move
the candidate-vs-production delta further). Measured flip-set overlap with
`spread_gap_zone_fade_overlay` (both key off the card's own `spread_line`):
**27 of this overlay's 180 flips** are also flipped by
`spread_gap_zone_fade_overlay`.

**Verdict.** Neither blocking has its whole 95% interval on the wrong side
of zero (week-blocked upper bound +0.9987, season-blocked upper bound
+0.6267 -- both above zero), so the RESOLVED-wrong-sign closing ground does
NOT apply despite the negative point estimate. Recorded to
`registry/weak_signals.json` as `pace_mismatch_dog_tilt_stacked_on_production`
(family `team_style_pace_mismatch_stacked_on_production`), classification
`unresolved_below_power`. Per AGENTS.md this is CONTEXT on the stacked form
only, never a gate on the underlying, already-registered pregame-safe cell
-- the challenger is registered regardless.

**Distinct from a different worker's sibling study.** A separate registry
family, `team_style_pace_on_production` (entry
`team_style_pace_mismatch_on_production`, source
`artifacts/team_style_pace_on_production/20260901T194505Z/results.json`),
already exists: it uses an EXPANDING strictly-prior-season quartile cut
against the retrained production ridge chain, on a rotation-ASSIGNED
`[2011, 2013]` window with a positive control (+2.011 accuracy points). That
is a different, non-poolable comparator (retrained-chain accuracy delta vs
this overlay's pick-flip-on-the-played-union delta) using a different
quartile construction (expanding strictly-prior vs this overlay's frozen
whole-panel cut, which matches the registered bare-baseline cell exactly).
The two studies must not be conflated or pooled.

## Week 1 2026 preview (dry-run only, no ledger write)

Applied to `artifacts/margin_predictions/2026-week-01-20260824T120725Z`
(16 games): **3 of 16 games flag** (NO@DET pace gap 2.53, BUF@HOU 2.37,
ARI@LAC 2.29 -- all above the frozen 2.1685 cut), but only **1 pick actually
flips**: NO@DET, where DET was the 7-point home favourite and the model's
own pick (0.5385 home-cover probability) was already on DET -- the flip
moves the pick to NO. The other two flagged games (BUF@HOU, ARI@LAC) have
the model's pick already on the underdog side, so the asymmetric rule
correctly leaves them untouched. This rule uses a PRIOR-SEASON trait plus
the card's own spread, so it is expected to fire in Week 1, and it does.

## Interaction with other overlays on the production card

This challenger is tracked **INDEPENDENTLY** against the active model's own
UN-flipped card, exactly like every other overlay challenger. Its picks
overlap the played four-member OR-union chain and `spread_gap_zone_fade_overlay`
specifically (see the measured 27-game overlap above); it is not an
independent arm from that overlay and must not be pooled with it as if it
were.

## What is and is not wired in

- `src/nfl_ats/pace_mismatch_dog_tilt_overlay.py`: the flag builder
  (`pace_mismatch_flag_by_game`), the fail-open loader
  (`pace_mismatch_flags_fail_open`), the transform
  (`apply_pace_mismatch_dog_tilt_overlay`), the disclosure sentence
  (`overlay_disclosure_note`, not currently surfaced anywhere), and the
  recorder (`record_pace_mismatch_dog_tilt_challenger_decisions`).
- `scripts/record_pace_mismatch_dog_challenger.py`: the standalone weekly
  recording entry point. **Not yet wired into `cli.py`'s generic
  weekly-run publish/record loop** -- `src/nfl_ats/cli.py` was off-limits to
  this build; wiring it in (and adding this challenger to
  `cli.PUBLISH_CHALLENGER_RESULT_KEYS`) is the pending follow-up before the
  2026-09-08 Week 1 lock.
- `artifacts/prospective/challengers.json`: proposed entry (not yet applied
  by this worker -- the orchestrator integrates it), `pace_mismatch_dog_tilt_overlay`,
  status `ACTIVE_PROSPECTIVE`, `model` block a snapshot of the active
  configuration at registration time (fingerprint `bc77638d47e2748c`,
  matching every other live pick-level overlay challenger).
- **Not wired anywhere:** there is no switch that applies this tilt to the
  published card. Playing it for real is a separate owner decision this
  document does not make.
- **Each challenger is tracked independently against the active model's own
  card, not stacked on the other overlays** (that is what the "Interaction"
  section and the stacked back-test above are for -- separate questions).

## Both grades, from Week 1

`nfl-ats prospective-score` settles every `ACTIVE_PROSPECTIVE` challenger,
including this one, at both grades automatically once picks are recorded:
`decision_line` (the spread the pick was actually made at -- primary) and
`close_line` (the same pick re-settled at the close -- secondary). No
challenger-specific scoring code was needed.

## Tests

`tests/test_pace_mismatch_dog_tilt_overlay.py`:

1. `pace_mismatch_flag_by_game`: reproduces the screen's frozen quartile cut
   (`>=`, inclusive boundary tested), correctly returns `False` (never an
   error) when a team has no prior-season row at all or a NaN prior-season
   value, and three leakage regression tests -- no outcome columns read, no
   future-season contamination, and no CURRENT-season contamination (only
   the prior season may ever be read).
2. `apply_pace_mismatch_dog_tilt_overlay`: flips a home-favourite pick to
   the underdog AND flips an away-favourite pick to the underdog (both
   spread directions exercised explicitly), leaves an already-underdog pick
   untouched, never touches a pick'em game, has no effect outside the
   flagged population (asserted as an exact flip-set equality, not just
   spot checks), leaves postseason games untouched, is a no-op when
   disabled or given an empty flag table, and changes only
   `home_cover_probability` on flipped rows.
3. `overlay_disclosure_note`: empty when nothing flipped, states the flip
   count and matchups when something did, and explicitly says "not applied
   to the published card".
4. `record_pace_mismatch_dog_tilt_challenger_decisions`: records the tilt's
   own arm (which can diverge from the active model's raw pick), is
   append-only and idempotent, refuses outside the recording lock window,
   refuses a fingerprint mismatch (an active-model promotion under the
   challenger's feet), and refuses an inactive registration.

## Decision left to the orchestrator

Nothing here decides anything -- it only starts a free evidence stream. Once
2026 accrues enough weeks, `nfl-ats prospective-score` reports this
challenger's `probability_positive` at both grades, paired against the
active model, exactly like the other overlay challengers. That number
answers whether the pace-mismatch dog lean actually holds up out-of-sample
against the LIVE, currently-active model -- it is not resolved by assumption
in this document, and the stacked back-test above is context for that
decision, never a substitute for it.
