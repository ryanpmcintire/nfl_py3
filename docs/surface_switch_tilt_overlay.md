# Surface-switch tilt overlay: a no-window-cost prospective challenger

Written 2026-08-19. Follows the `backup_qb_fade_overlay` /
`division_revenge_tilt_overlay` / `injury_value_lost_tilt_overlay` /
`hc_year_one_fade_overlay` precedent (`docs/backup_qb_fade_overlay.md`,
`docs/division_revenge_tilt_overlay.md`, `docs/injury_value_lost_tilt_overlay.md`,
`docs/coach_fade_overlay.md`) for wiring a pick-level, post-prediction
transform into the prospective challenger ledger at zero rotation-registry
window cost.

## Mined lineage (read from the registry before building, as the task required)

Three reads, all measured/recorded 2026-08-19, all `unresolved_below_power`:

| Entry | League / population | Effect (accuracy pts) | Interval | `probability_positive` |
|---|---|---|---|---|
| `weather_battery_surface_switch_grass_to_turf` | NFL, mined, week-blocked, REG 2009-2025, n=4,317 | +1.1618 | [+0.2896, +2.038] | 0.995 |
| `surface_familiarity_r1_turf_venue_visitor_split` | NFL, venue-controlled follow-up, REG 2009-2025, n_pair=1,857 | +1.4579 | [-0.4774, +3.3655] | 0.9332 |
| `cfb_surface_familiarity_turf_venue_visitor_split` | CFB, cross-league replication, 2012-2025, n_pair=6,133 | +1.5579 | [-0.6665, +3.7421] | 0.9156 |

1. **Weather battery cell.** `weather_battery_surface_switch_grass_to_turf`
   is one of 8 predeclared cells in the NFL weather/environment bias battery
   (`scripts/nfl_weather_battery_screen.py`, mined, uncorrected multiplicity):
   away team's modal home surface this season normalizes to grass AND this
   game's surface normalizes to turf.
2. **Venue-controlled follow-up.** `surface_familiarity_r1_turf_venue_visitor_split`
   (`scripts/surface_familiarity_screen.py`) isolates the SAME mechanism
   WITHIN turf-venue games only, holding venue fixed -- so the gap cannot be
   explained away as "turf venues just play differently" -- and finds a
   full-slate effect even LARGER than the unconditional battery cell.
3. **CFB cross-league replication.** `cfb_surface_familiarity_turf_venue_visitor_split`
   (`scripts/cfb_surface_familiarity_screen.py`) replicates the identical
   venue-controlled construct on FBS games: same sign, a comparable (if
   anything larger) point estimate, an independent league and data source.

All three entries were **read**, directly from `registry/weak_signals.json`,
before this module was built. All three intervals cross zero. Per AGENTS.md,
an interval crossing zero at this evaluator's ~2-point resolution is the
EXPECTED shape for a real small signal, never grounds to close the line or
decline building a no-window-cost prospective challenger.

## The grass-venue mirror is null in BOTH leagues (stated up front, not buried)

If this were a clean, bilateral "surface-switch cost" mechanism, the
symmetric GRASS-venue mirror (turf-modal visitors playing on grass, vs.
grass-modal visitors) should show the SAME-SIZED positive gap. It does not,
in either league:

| Entry | League | Effect (accuracy pts) | Interval | `probability_positive` |
|---|---|---|---|---|
| `surface_familiarity_r2_grass_venue_mirror` | NFL | -0.4995 | [-2.6259, +1.6077] | 0.3205 |
| `cfb_surface_familiarity_grass_venue_mirror` | CFB | -0.1218 | [-1.3109, +1.0721] | 0.4291 |

Both mirror reads are **near a coin flip, leaning the WRONG way** (negative
point estimate against the predicted positive direction) in both leagues.
This is the reason the rule below is deliberately ASYMMETRIC (see "The rule,
exactly as built" below): it fires only in the one direction (grass-modal
visitor onto turf) that both leagues' primary reads actually support, never
the mirror direction that neither league corroborates.

## Era caveat: the NFL effect concentrates in 2018-2025

The venue-controlled follow-up's own era split shows the SAME sign in both
halves (sign-stable) but a roughly 4.5x larger magnitude in the later era:

| Entry | Seasons | Effect (accuracy pts) | Interval | `probability_positive` |
|---|---|---|---|---|
| `surface_familiarity_r3_era_2009_2017` | 2009-2017, n=973 | +0.5277 | [-2.1556, +3.1865] | 0.6482 |
| `surface_familiarity_r3_era_2018_2025` | 2018-2025, n=884 | +2.3869 | [-0.3446, +5.0651] | 0.95775 |

Both eras stay `unresolved_below_power`. This is a caveat on the effect's
stability -- reported here, not buried -- not grounds to restrict the
overlay's eligible weeks: the overlay applies to all of 2026 regardless of
era, since 2026 postdates both measured halves and the sign is consistent
across them.

## What this is not

None of the five registry entries above spend a rotation-registry window --
all are bias-battery/follow-up/cross-league re-screens, not an opener-window
confirmation run. This document and its registration do not change that,
and nothing here is an owner decision to play the tilt on the real card
(unlike `hc_year_one_fade_overlay`). It is dual-tracked only.

## The construct, exactly as measured (ported, not redesigned)

Ported **verbatim** from `scripts/nfl_weather_battery_screen.py`:

- `_normalize_surface` (lower-case, strip whitespace, match against two
  frozen sets):
  ```python
  GRASS_SURFACES = frozenset({"grass", "dessograss"})
  TURF_SURFACES = frozenset(
      {"fieldturf", "sportturf", "matrixturf", "astroturf", "a_turf", "astroplay"}
  )
  ```
- `load_population`'s modal-home-surface derivation: for every
  `(home_team, season)`, the MODE of that team's normalized home-game
  surface across the FULL regular season
  (`s.mode().iat[0] if not s.mode(dropna=True).empty else None`), looked up
  for each game by its AWAY team and season.
- The flag itself (cell 6 of `build_cells`):
  `away_modal_surface == "grass" AND surface_norm == "turf"`.

Implemented in `src/nfl_ats/surface_switch_tilt_overlay.py`'s
`surface_switch_flag_by_game`, which reads the newest local schedule
snapshot (`data/raw/<snapshot>/schedules.parquet`) directly, mirroring
`coach_fade_overlay.year_one_by_game`'s data source.

**Why a full-season aggregate is pregame-safe here, unlike the coach and
QB-continuity overlays' strictly-prior-only aggregates.** A team's home-
stadium surface is a STRUCTURAL, stadium-level fact fixed for essentially
the entire season and public knowledge before Week 1 -- unlike a head
coach (who can be fired mid-season) or a starting QB (who can change any
week), it is not an outcome and this derivation never reads `result` or
`spread_line` at all. The source script's own comment makes the same point
in code: "roof/surface is a stadium fact, not a cover outcome". Two leakage
regression tests (`tests/test_surface_switch_tilt_overlay.py`) prove this
empirically: the flag is unaffected by any outcome-bearing column
(`result`/`spread_line` are not even read), and a future season's surface
data never changes an earlier season's already-computed flags.

Team codes are canonicalized (`TEAM_ABBREVIATION_ALIASES`) before the
`(home_team, season)` grouping and the away-team lookup -- a merge-safety
measure only, so the output joins cleanly against the predictions frame's
own team codes; it does not change which surface a team's home games in a
given season actually used, so it does not alter the measured construct
itself.

## The rule, exactly as built (parameter-free, frozen, deliberately asymmetric)

```
flag = surface_switch_flag_by_game(schedules)   # away modal surface grass AND this game's surface turf
model_pick_away = home_cover_probability < 0.5

flip when:
    flag  AND  model_pick_away
```

In plain language: **when the flag is set AND the active model's own pick is
the AWAY team, flip to the home team.** REG season only (every measured read
above was scored on regular-season games); missing surface data means no
flip (folded into `False` by `surface_switch_flag_by_game`).

**Deliberately asymmetric, unlike the sibling tilt overlays**
(`division_revenge_tilt_overlay`, `injury_value_tilt_overlay`): this never
flips a HOME pick to AWAY when the flag fires, because the mirror direction
is near-null in both leagues (see above) -- there is no measured direction
to fade INTO the flagged side, only away from picking the grass-modal
visitor on turf.

Implemented in `src/nfl_ats/surface_switch_tilt_overlay.py`:
`surface_switch_flag_by_game` derives the flag;
`apply_surface_switch_tilt_overlay` applies the rule at pick level;
`overlay_disclosure_note` produces the plain-English provenance sentence
(not currently surfaced anywhere).

## Why this construction, not a full retrained-model challenger

Same two reasons the four sibling overlay docs give for their own tilts,
restated for this construct: (1) a full retrained challenger dual-tracked
against whatever model is currently active would answer a different
question than the isolated, venue-controlled, cross-league-replicated
construct measured above, and (2) the task explicitly named the pick-level
tilt precedents as the pattern to follow, and a pick-level design touches
zero training frames, zero feature profiles, and zero stored model
artifacts.

## What is and is not wired in

- `src/nfl_ats/surface_switch_tilt_overlay.py`: the transform
  (`apply_surface_switch_tilt_overlay`), the signal reader
  (`surface_switch_flag_by_game`), the disclosure sentence
  (`overlay_disclosure_note`, not currently surfaced anywhere), and the
  recorder (`record_surface_switch_tilt_challenger_decisions`).
- `src/nfl_ats/cli.py`'s `_cmd_publish_predictions`: a **seventh**, purely
  additive, fail-open `try`/`except` block (mirroring the existing six)
  that calls the recorder when `--record-decisions` is passed. This writes
  ONLY to `artifacts/prospective/challenger_decisions.parquet`; it never
  touches `recommendations.csv`, `CURRENT_PREDICTIONS.md`, `README.md`, or
  the public site. **The production pick path
  (`publish_active_predictions`) is untouched by this build.**
- `artifacts/prospective/challengers.json`: registered as
  `surface_switch_tilt_overlay`, status `ACTIVE_PROSPECTIVE`, `model` block
  a snapshot of the active configuration at registration time (for
  fingerprint-mismatch detection only, mirroring the six existing
  challengers).
- **Not wired anywhere:** there is no `OVERLAY_ENABLED`-style switch that
  applies the tilt to the published card. Playing this tilt for real is a
  separate owner decision this document does not make.
- **Each challenger is tracked independently against the active model's own
  card, not stacked on the other overlays.** `_cmd_publish_predictions`
  calls every overlay/nomination recorder in SEPARATE try/except blocks,
  each reading the SAME un-flipped active-model card and applying its own
  transform independently -- one overlay's flip never feeds another
  overlay's input. This matches the existing pattern exactly.

## Both grades, from Week 1

`nfl-ats prospective-score` settles every `ACTIVE_PROSPECTIVE` challenger,
including this one, at both grades automatically once picks are recorded:
`decision_line` (the spread the pick was actually made at -- primary) and
`close_line` (the same pick re-settled at the close -- secondary). No
challenger-specific scoring code was needed; this is the same generic
machinery the other overlay challengers already use
(`docs/prospective_evidence.md`).

## Tests

`tests/test_surface_switch_tilt_overlay.py`, mirroring
`tests/test_division_revenge_tilt_overlay.py`'s structure:

1. `surface_switch_flag_by_game`: fires when the away team's modal home
   surface this season is grass and this game's surface is turf, does not
   fire when the surfaces are not mismatched in that direction, does not
   fire when either surface is missing/unresolved, raises on missing
   columns, and two leakage regression tests (the flag never depends on any
   outcome-bearing column, since none are even read; a future season's
   surface data does not move an earlier season's flags).
2. `apply_surface_switch_tilt_overlay`: flips an away pick to home on the
   flagged side, does NOT flip a home pick even when the flag fires (the
   asymmetric design), does not flip when the flag is absent, leaves
   postseason games untouched, treats a missing schedule row as no signal,
   is a no-op when disabled, and changes only `home_cover_probability` on
   flipped rows (byte-identical everywhere else).
3. `overlay_disclosure_note`: empty when nothing flipped, states the flip
   count and matchups when something did, and explicitly says "not applied
   to the published card".
4. `record_surface_switch_tilt_challenger_decisions`: records the tilt's
   own arm (which can diverge from the active model's raw pick), is
   append-only and idempotent, refuses outside the recording lock window,
   refuses a fingerprint mismatch (an active-model promotion under the
   challenger's feet), and refuses an inactive registration.

## Decision left to the orchestrator

Nothing here decides anything -- it only starts a free evidence stream. Once
2026 accrues enough weeks, `nfl-ats prospective-score` reports this
challenger's `probability_positive` at both grades, paired against the
active model, exactly like the other overlay challengers. That number is
new, independent evidence about the TILT rule specifically -- it neither
replicates nor substitutes for the three registry reads above.
