# Third-down mean-reversion fade overlay: a no-window-cost prospective challenger

Written 2026-09-01. Follows the `tank_zone_fade_tilt_overlay` /
`pbp08_protection_mismatch_tilt_overlay` / `coach_fade_overlay` /
`surface_switch_tilt_overlay` precedent (`docs/tank_zone_fade_tilt_overlay.md`,
`docs/pbp08_matchup_screen.md`, `docs/coach_fade_overlay.md`,
`docs/surface_switch_tilt_overlay.md`) for wiring a pick-level, post-prediction
transform into the prospective challenger ledger at zero rotation-registry
window cost.

## Evidence (read from the registry before building, as the task required)

`registry/weak_signals.json:redzone_reversion_c3_third_down_over_fade`:
effect **+0.36652412950519364** accuracy points, week-blocked 95%
**[-0.2587, +0.999]**, `probability_positive` **0.87185**, **reliability
0.407** (trait year-over-year Pearson +0.407, 95% [+0.337, +0.473]), n=8634
team-games, sample_blocks 294, seasons 2009-2025, category `onfield`,
classification `unresolved_below_power`.

Measured by `scripts/redzone_reversion_screen.py`, artifact
`artifacts/redzone_reversion_screen/20260821T181025Z/results.json`, cell
`third_down_over_fade` (predeclared cell C3 of a 6-cell battery,
`docs/redzone_reversion_screen.md`):

| Read | 95% interval | `probability_positive` | n | n_blocks |
|---|---|---|---|---|
| Week-blocked (primary) | [-0.2587, +0.999] | 0.87185 | 8,634 | 294 |
| Season-blocked (secondary) | [-0.2576, +0.965] | 0.87135 | 8,634 | 17 |

**The interval crosses zero.** Per AGENTS.md, at this evaluator's ~2-point
resolution that is the EXPECTED shape for a real-but-small signal and is
NEVER grounds to decline building a no-window-cost prospective challenger.
Neither admissible closing ground applies (no resolved wrong sign, no
positive-control bound), so the cell stays `unresolved_below_power` in the
registry; wiring it here is an EV-positive dual-tracked play
(`probability_positive` 0.87185 > 0.5), not a claim of a proven edge.

## Direction check (verified, not assumed)

A sibling cell in this same mined battery failed its own direction check and
was dropped from the batch before registration -- this one was verified to
pass. Read directly from
`artifacts/redzone_reversion_screen/20260821T181025Z/results.json`, cell
`third_down_over_fade`: `sign_dir` is `-1` (a FADE cell -- the predeclared
prediction is NEGATIVE on `team_covered`), `subset_mean` **0.4884947267497603**
and `complement_mean` **0.503665241295052**. Flagged (prior-season elite
third-down) teams covered **48.85%** against a **50.37%** field -- the
predeclared fade is exactly what the data shows.

## Shared-trait mirror caveat (stated up front, not buried)

The mirror cell `third_down_under_rebound` (same trait, bottom quartile,
predicting a POSITIVE rebound) reads `full_slate_effect_pts`
**-0.3564386470499031**, `probability_positive` **0.09885** (week-blocked) --
the mirror's OWN prediction is CONTRADICTED (bottom-quartile teams did not
rebound). Both cells key off the SAME underlying trait split at the SAME two
tails of the SAME panel, so they are ONE signal read from two ends, not two
independent votes -- the registry entry's own note already says "mirror c4
shares trait, not independent". This overlay builds ONLY the C3 (top-quartile
fade) side. It must never be pooled with a hypothetical bottom-quartile
challenger as if the two were independent evidence.

## The trait, transcribed VERBATIM (not re-derived)

Cited from `scripts/redzone_reversion_screen.py` (not importable as a
library -- a standalone CLI with its own `sys.path` hacks -- so the
construction is PORTED into `src/nfl_ats/third_down_reversion_fade_overlay.py`,
exactly as `pbp08_matchup_flags.py` ports `pbp08_matchup_screen.py`):

1. **Third-down conversion rate**, per (season, team): every play with
   `down == 3.0` (`build_efficiency_panels`, line 126) grouped by
   `(season, posteam)`; `n_third_downs` is the play count and
   `third_conversions` is the sum of `first_down` (lines 127-132);
   `third_down_conv_rate = third_conversions / n_third_downs` (lines
   133-135). Plays are `nfl_ats.pbp.analysis_plays`' documented v1 efficiency
   filter, REG season only (line 94), team codes canonicalized (line 97).
2. **"Centered"** means: subtract that SEASON's own cross-team mean rate --
   `league_mean = offense.groupby("season")[trait].transform("mean")` then
   `offense[f"{trait}_centered"] = offense[trait] - league_mean` (lines
   151-153). A team's centered value is its OWN rate minus the average of
   every team THAT SAME SEASON -- never a cross-season comparison.
3. **The top-quartile cut is GLOBAL, not within-season.** `thresholds["third_down_q75"]
   = float(offense["third_down_conv_rate_centered"].quantile(0.75))` (line
   383) takes the 0.75 quantile of the ENTIRE pooled `offense` panel --
   every team-season 2009-2025 at once -- not a quantile recomputed
   separately inside each season.
4. **Prior-season lookup.** `_prior` (lines 191-196) shifts a team-season's
   row forward one season (`season = season + 1`) before joining onto the
   schedule by `(team, season)` -- so a game in season *S* reads the
   centered rate the team posted in season *S-1*, never season *S* itself.

## The frozen threshold (an underived constant would be a defect)

The measured GLOBAL top-quartile threshold is **0.03392624406886406** (read,
`artifacts/redzone_reversion_screen/20260821T181025Z/results.json:349`,
`thresholds.third_down_q75`; the same value
`scripts/redzone_reversion_screen.py:383` computed). This overlay FREEZES
that exact value as `THIRD_DOWN_TOP_QUARTILE_CENTERED` rather than
recomputing a quantile live, mirroring
`spread_gap_zone_fade_overlay.SPREAD_GAP_LOWER_BOUND` /
`SPREAD_GAP_UPPER_BOUND`'s identical choice: it is pregame-safe by
construction (a frozen constant carries zero risk of ever reading a future
season's data) and it keeps the rule the registry cell's own measured
number, cited, per AGENTS.md's "underived constants are defects" rule.

`tests/test_third_down_reversion_fade_overlay.py` includes a fixture
(`test_flag_uses_the_frozen_global_threshold_not_a_locally_recomputed_quantile`)
constructed so that a locally recomputed within-sample quantile would flag a
team the correct, frozen-global rule does not -- proving the implementation
really uses the GLOBAL frozen cutoff, not a within-season/within-sample
recomputation.

## The rule, exactly as built (frozen, parameter-free)

```
flag(team, season) = prior_third_down_conv_rate_centered(team, season - 1)
                       >= THIRD_DOWN_TOP_QUARTILE_CENTERED (0.03392624406886406)

flip when:
    game_type == "REG"  (when present)
    AND  EXACTLY ONE of the two teams is flagged
    AND  the model's own forced pick IS the flagged team
```

Both-flagged games are never touched -- the same clean-case handling
`coach_fade_overlay` / `tank_zone_fade_tilt_overlay` use: no measured
direction when both sides carry the flag. Never flip in any other
situation. The overlay is deliberately one-directional (a fade, never a
back): it never moves a pick TOWARD the flagged side.

Implemented in `src/nfl_ats/third_down_reversion_fade_overlay.py`:
`third_down_over_flag_by_game` derives the flag from a schedule frame and a
raw PBP frame; `apply_third_down_reversion_fade_overlay` applies the
pick-level transform; `overlay_disclosure_note` produces the plain-English
provenance sentence (not currently surfaced anywhere).

## Pregame safety

The trait is a PRIOR-SEASON aggregate -- fully known before Week 1 of the
season being flagged, since it depends only on plays from a season that has
already finished, and the threshold is a frozen constant that reads no data
at all. `tests/test_third_down_reversion_fade_overlay.py` proves this
empirically with two leakage regression tests: mutating a game's own
current-season PBP data never changes its flag, and a later season's PBP
data never changes an earlier season's already-computed flags. Because the
trait is a prior-season aggregate, this rule can and should fire in Week 1
of any season, unlike the tank-zone or coach-fade overlays' in-season
windows.

## Why this construction, not a full retrained-model challenger

Same reasoning the sibling overlay docs give: (1) a full retrained
challenger dual-tracked against whatever model is currently active would
answer a different, confounded question, and (2) the task explicitly named
the pick-level tilt precedents as the pattern to follow -- a pick-level
design touches zero training frames, zero feature profiles, and zero stored
model artifacts.

## What is and is not wired in

- `src/nfl_ats/third_down_reversion_fade_overlay.py`: the transform
  (`apply_third_down_reversion_fade_overlay`), the disclosure sentence
  (`overlay_disclosure_note`, not currently surfaced anywhere), and the
  recorder (`record_third_down_reversion_fade_challenger_decisions`).
- `scripts/record_third_down_reversion_fade_challenger.py`: the standalone
  weekly recording entry point. **Not** `nfl-ats publish-predictions
  --record-decisions` -- see that script's own docstring for why (the CLI
  wiring is a pending follow-up gated by
  `tests/test_cli.py::test_publish_challenger_result_map_covers_live_active_registry`).
- `artifacts/prospective/challengers.json`: registered as
  `third_down_reversion_fade_overlay`, status `ACTIVE_PROSPECTIVE`, `model`
  block a snapshot of the active configuration at registration time (for
  fingerprint-mismatch detection only, mirroring the existing challengers).
- **Not wired anywhere:** there is no switch that applies the fade to the
  published card. Playing this fade for real is a separate owner decision
  this document does not make.
- **Tracked independently against the active model's own card, not stacked
  on the other overlays** in the live prospective ledger. A separate
  stacked-on-production BACK-TEST (below) answers the "does it add anything
  on top of what is actually played" question as a mined-seasons read.

## Stacked-on-production back-test (mined-seasons read, context not a gate)

`scripts/third_down_reversion_fade_stacked_backtest.py` applies the fade on
top of the reconstructed four-member production union
(`overlay_union_coach_division_revenge_player_arrests_spread_gap_v1`) on the
frozen opener archive
(`artifacts/opener_evaluation/20260819T174244Z/per_game.parquet`, 1,537 REG
games 2020-2025, baseline 53.36% on 1,503 scored games), scored under the
production probability rule, with week-blocked (primary) and season-blocked
(secondary) paired bootstrap (20,000 samples, seed 20260901). See the
session report for the measured numbers and the `weak-signals record`
disposition (`redzone_reversion_c3_stacked_on_production` family).

**Per the binding taxonomy**: an interval containing zero is NEVER grounds
to reject this line of work. Only a whole interval on the wrong side of
zero, on BOTH blockings, is treated as a resolved wrong sign for the
STACKED form; every other outcome (including a negative point estimate
whose interval crosses zero) is `unresolved_below_power` and does not
block registering the challenger.

## Both grades, from Week 1

`nfl-ats prospective-score` settles every `ACTIVE_PROSPECTIVE` challenger,
including this one, at both grades automatically once picks are recorded:
`decision_line` (the spread the pick was actually made at -- primary) and
`close_line` (the same pick re-settled at the close -- secondary). No
challenger-specific scoring code is needed; this is the same generic
machinery the other overlay challengers already use
(`docs/prospective_evidence.md`).

## Tests

`tests/test_third_down_reversion_fade_overlay.py`, mirroring
`tests/test_tank_zone_fade_tilt_overlay.py`'s structure:

1. `third_down_over_flag_by_game`: the flag fires on a fixture reproducing
   the trait/centering/threshold convention, fires on both sides of a
   double-flagged game, and is false for teams below the top quartile.
2. The frozen threshold is proven GLOBAL, not a locally recomputed
   quantile, via a dedicated fixture.
3. Two leakage regression tests: a game's own current-season PBP data never
   moves its flag; a later season's PBP data never moves an earlier
   season's flag.
4. Missing prior-season data means `flag=False`, never an exception.
5. `apply_third_down_reversion_fade_overlay`: flips only the clean case
   (exactly one side flagged and the model's pick is on it), never flips a
   both-flagged game, never flips outside the flagged population, leaves
   postseason games untouched, treats a missing schedule row as no signal,
   is a no-op when disabled, and changes only `home_cover_probability` on
   flipped rows.
6. `overlay_disclosure_note`: empty when nothing flipped, states the flip
   count and matchups when something did, and explicitly says "not applied
   to the published card".
7. `record_third_down_reversion_fade_challenger_decisions`: records the
   fade's own arm, is append-only and idempotent, refuses outside the
   recording lock window, refuses a fingerprint mismatch, and refuses an
   inactive registration.

## Decision left to the orchestrator

Nothing here decides anything -- it only starts a free evidence stream.
Once 2026 accrues enough weeks, `nfl-ats prospective-score` reports this
challenger's `probability_positive` at both grades, paired against the
active model, exactly like the other overlay challengers.
