# Bye-edge fade overlay: a no-window-cost prospective challenger

Written 2026-09-01. Follows the `surface_switch_tilt_overlay` /
`interim_hc_first_game_tilt_overlay` precedent
(`docs/surface_switch_tilt_overlay.md`, `docs/interim_coach_screen.md`) for
wiring a pick-level, post-prediction transform into the prospective
challenger ledger at zero rotation-registry window cost.

## The registered cell, exactly as measured

Read from `registry/weak_signals.json` before this module was built:
`bye_overval_fade_full_slate_post2011` -- seasons 2012-2025 restricted to
week-blocks containing at least one strictly off-bye team anywhere in the
league; flag = exactly one of the two teams off a STRICT bye (>=12-day gap to
its own immediately preceding game this season); value column is the
FADE-side cover indicator.

| Field | Value |
|---|---|
| Full-slate effect | **+0.5508134037685816** accuracy points |
| Week-blocked 95% CI | **[-0.5667823486802533, +1.6434387325997184]** |
| `probability_positive` (week-blocked) | **0.8375** |
| Season-blocked secondary | +0.5508134037685816 pts, 95% [-0.4285341983532842, +1.5664703739568875], P+ 0.8603, 14 blocks |
| `n_flag` / `n_total` / `n_complement` | 498 / 2171 / 1673 |
| `fraction_of_slate` | 0.22938737908797788 |
| Classification | `unresolved_below_power` |

The interval crosses zero. Per AGENTS.md, an interval crossing zero at this
evaluator's ~2-point resolution is the EXPECTED shape for a real small
signal, never grounds to close the line or decline building a no-window-cost
prospective challenger. Neither admissible closing ground applies (no
resolved wrong sign, no positive-control bound), so this stays
`unresolved_below_power`; wiring it here is an EV-positive dual-tracked play
(P+ 0.8375 > 0.5), not a claim of a proven edge (AGENTS.md "a promotion bar
is not a decision bar").

## This entry replaced a prior buggy-map entry (disclosed, not buried)

`docs/bye_overvaluation_screen.md`, "Correction 2026-08-22": the original
`build_bye_maps` sorted each team's games ACROSS seasons, so every season
opener inherited a >=12-day gap from the PRIOR season's finale and was
misflagged "off bye". Fixed by computing gaps within `(team, season)`
groups. The corrected instrument is
`artifacts/bye_overvaluation_screen/post_fix_seed20260822/results.json`
(seed 20260822); every number quoted above, and every number this module's
own flag logic reproduces, comes from that corrected artifact/script, never
the buggy cross-season one. This module's own leakage regression tests
(`tests/test_bye_edge_fade_overlay.py`) pin the fix: a future season's
schedule data never changes an earlier season's already-computed flags, and
a season opener specifically is proven never to inherit the prior season's
gap.

## Direction check -- verified, not assumed

The registered cell's `value_column` is `"fade_side_cover"` and its
description ends "Predicted direction POSITIVE". Read directly from the
corrected artifact (`artifacts/bye_overvaluation_screen/post_fix_seed20260822/results.json`,
cell `bye_overval_fade_full_slate_post2011`):

- `subset_cover` (the fade side's own cover rate): **0.5261044176706827**
- `complement_cover`: **0.502092050209205**

The fade side covers MORE than the complement -- fading the bye-holding side
is BOTH the predeclared direction AND the measured direction. This check is
stated explicitly because, in this same build batch, a sibling cell failed
exactly this check and was dropped for it -- passing it is not automatic.

## Instrument sanity: the both-off-bye null control

`bye_overval_both_bye_sanity` (BOTH teams off strict bye, full window
2009-2025, two-sided predeclared null): effect **+0.01237162203848194**
accuracy points, `probability_positive` **0.55985** -- a clean null, exactly
as a working instrument should read when rest cancels on both sides. This
overlay never touches a both-off-bye game (see "The rule" below), consistent
with that null control; it is not a mechanism this overlay claims to trade.

## Era caveat: the pre-2011 control leans the OPPOSITE way

| Entry | Seasons | Effect (accuracy pts) | `probability_positive` |
|---|---|---|---|
| `bye_overval_home_edge_post2011` | 2012-2025 | -0.3304447 | 0.06365 |
| `bye_overval_home_edge_pre2011` | 2009-2011 | +0.2707765 | 0.70695 |

The identical HOME-off-strict-bye construct reads NEGATIVE post-2011 and
POSITIVE pre-2011 -- the two eras lean opposite ways, exactly why the
registered fade cell is restricted to 2012+. **2026 postdates that boundary,
so the overlay applies to ALL of 2026 regardless of week** -- this is a
caveat on the effect's era-stability, reported here, not grounds to restrict
which weeks the overlay is eligible to fire on.

## What this is not

None of the registry entries above spend a rotation-registry window -- the
bye-overvaluation screen is a measure-only, mined 5-cell battery
(`scripts/bye_overvaluation_screen.py`, `docs/bye_overvaluation_screen.md`).
This document and its registration do not change that, and nothing here is
an owner decision to play the fade on the real card. It is dual-tracked
only.

## The construct, exactly as measured (ported, not redesigned)

Ported **verbatim** from `scripts/bye_overvaluation_screen.py`:

- The strict-bye threshold, `POST_BYE_GAP_DAYS = 12`
  (`scripts/bye_overvaluation_screen.py:58`).
- `build_bye_maps` (`scripts/bye_overvaluation_screen.py:79-104`, the
  CORRECTED, post-2026-08-22 version): melt each REG-season game into two
  team-rows, sort each team's rows within its own `(team, season)` group by
  `gameday`, and flag a `>=12`-day gap to the immediately preceding game in
  that group (`scripts/bye_overvaluation_screen.py:92-94`):
  ```python
  long_df["gap_days"] = long_df.groupby(["team", "season"])["gameday_dt"].diff().dt.days
  long_df["post_bye"] = (long_df["gap_days"] >= POST_BYE_GAP_DAYS).fillna(False).astype(bool)
  ```
  A team's first REG-season game has no preceding game in the SAME season,
  so its gap is `NaN` and is folded to `False` -- never a bye.

Implemented in `src/nfl_ats/bye_edge_fade_overlay.py`'s
`bye_edge_flag_by_game`, which reads the newest local schedule snapshot
(`data/raw/<snapshot>/schedules.parquet`) directly, mirroring
`surface_switch_flag_by_game`'s data source.

**One deliberate deviation from a byte-for-byte port, and why.** The screen
script computes this same gap sequence on its own MEASUREMENT population,
which has already dropped push/no-line games (`add_ats_outcomes`, then
`home_cover.notna()`) before `build_bye_maps` ever runs -- acceptable for a
measure-only screen, but wrong for a live, pregame-safe overlay: excluding a
game from the sequence because it happened to push requires knowing its
RESULT, which is not knowable before kickoff for a game still to be played,
and must never be allowed to shift an earlier team's already-computed gap.
`bye_edge_flag_by_game` instead runs the identical gap/threshold logic
against every REG-season row in the RAW schedule snapshot -- a pure
structural fact. Two leakage regression tests
(`tests/test_bye_edge_fade_overlay.py`) prove this empirically: the function
does not even require or read `result`/`spread_line`, and a future season's
schedule data never changes an earlier season's already-computed flags.

Team codes are canonicalized (`TEAM_ABBREVIATION_ALIASES`) before the
`(team, season)` grouping -- a merge-safety measure only, mirroring
`surface_switch_flag_by_game`; it does not change which games count as a
team's own REG-season sequence, so it does not alter the measured construct
itself.

## The rule, exactly as built (parameter-free, frozen)

```
flags = bye_edge_flag_by_game(schedules)   # home_off_bye, away_off_bye per game
home_is_bye_team = home_off_bye AND NOT away_off_bye
away_is_bye_team = away_off_bye AND NOT home_off_bye
model_pick_home  = home_cover_probability >= 0.5

flip when:
    REG season only
    AND (
        (home_is_bye_team AND model_pick_home)
        OR (away_is_bye_team AND NOT model_pick_home)
    )
```

In plain language: **if exactly one of the two teams is off a strict bye
(>=12-day gap to its own immediately preceding game this season) AND the
active model's own forced pick IS that bye-holding team, flip the pick to
the other side.** REG season only (the registered read was scored on
regular-season games); both-off-bye and neither-off-bye games are NEVER
touched -- `bye_overval_both_bye_sanity` is the null control for the
both-bye case, not a mechanism this overlay claims to trade.

Implemented in `src/nfl_ats/bye_edge_fade_overlay.py`: `bye_edge_flag_by_game`
derives the flags; `apply_bye_edge_fade_overlay` applies the rule at pick
level; `overlay_disclosure_note` produces the plain-English provenance
sentence (not currently surfaced anywhere).

## Stacked-on-production back-test (mined-seasons read -- CONTEXT, not a gate)

`scripts/bye_edge_fade_stacked_backtest.py` applies the overlay ALONE on top
of the frozen active-model opener baseline
(`artifacts/opener_evaluation/20260819T174244Z/per_game.parquet`, 1,537
REG-season games 2020-2025, production probability rule, baseline 53.36% on
1,503 scored games) -- the same baseline every sibling overlay's solo read
uses in `scripts/overlay_stack_backtest.py`.

Measured 2026-09-01 (`artifacts/bye_edge_fade_stacked/20260901T194319Z/results.json`,
20,000-sample bootstrap, seed 20260821):

| Blocking | Delta (accuracy pts) | 95% CI | `probability_positive` | Blocks |
|---|---|---|---|---|
| Week-blocked | **+0.5988** | [-0.4670, +1.6880] | 0.8474 | 107 |
| Season-blocked | **+0.5988** | [+0.1266, +1.4175] | 0.9980 | 6 |

`n_flipped` = **68** of 1,537 games (candidate accuracy 53.96% vs. baseline
53.36%). Per-season flip counts: 2020: 12, 2021: 11, 2022: 8, 2023: 9, 2024:
17, 2025: 11.

**Verdict handling (binding, per AGENTS.md/CLAUDE.md).** An interval or CI
that contains zero is NEVER grounds to reject, fail, or close an experiment.
Only two grounds ever close a line of work: (1) refuted mechanism -- a
RESOLVED wrong sign (whole interval on the wrong side of zero) or zero
split-half reliability; (2) bounded by a positive control proven able to
detect an effect that size. Everything else is `unresolved_below_power`;
report `probability_positive`, never "contains zero". Here, the point
estimate is POSITIVE on both blockings and the season-blocked interval sits
ENTIRELY above zero; the week-blocked interval crosses zero but is not
below zero. Neither admissible closing ground applies, so this stacked read
is recorded `unresolved_below_power` in a NEW family
(`bye_overval_fade_stacked_on_production`, registered
2026-09-01) -- **and the challenger registration proceeds regardless**, per
the crossing-zero invariant and the "a promotion bar is not a decision bar"
rule.

## Week 1 2026 preview (dry-run only -- no ledger write)

Applied to the published Week 1 card
(`artifacts/margin_predictions/2026-week-01-20260824T120725Z/recommendations.csv`,
16 games): **0 flips**. This is BY CONSTRUCTION, not a defect -- the strict
bye definition requires a gap to a team's own immediately PRECEDING REG-
season game, and Week 1 is every team's first game of the season, so no team
can be off a strict bye in Week 1. The overlay's first live opportunity to
fire arrives once byes start (historically NFL weeks 5-15).

**Historical per-season firing rate** (measured from the 2026 schedule
snapshot, `data/raw/20260824T115346Z/schedules.parquet`, via
`bye_edge_flag_by_game`): across REG 2009-2026, an average of **10.5%** of
REG games per season have exactly one team off a strict bye (range 8.1%-
12.5% by season; the 2026 season's own already-published full schedule reads
11.0%, 30 of 272 games). Of those FLAGGED games, the model's own pick sits on
the bye-holding side only part of the time -- the stacked back-test above
measured an ACTUAL flip rate of 4.4% of all graded games (68 of 1,537,
2020-2025). Both rates are read/measured, not assumed.

## Why this construction, not a full retrained-model challenger

Same two reasons the sibling overlay docs give for their own tilts: (1) a
full retrained challenger dual-tracked against whatever model is currently
active would answer a different question than the isolated, corrected-
instrument construct measured above, and (2) the task explicitly named the
pick-level tilt precedents as the pattern to follow, and a pick-level design
touches zero training frames, zero feature profiles, and zero stored model
artifacts.

## What is and is not wired in

- `src/nfl_ats/bye_edge_fade_overlay.py`: the transform
  (`apply_bye_edge_fade_overlay`), the signal reader
  (`bye_edge_flag_by_game`), the disclosure sentence
  (`overlay_disclosure_note`, not currently surfaced anywhere), and the
  recorder (`record_bye_edge_fade_challenger_decisions`).
- `scripts/record_bye_edge_fade_challenger.py`: a standalone weekly entry
  point calling the recorder directly and printing its JSON result.
  **`src/nfl_ats/cli.py` is NOT touched by this build** -- wiring a seventh
  `try`/`except` block into `_cmd_publish_predictions` (mirroring the six
  existing overlay blocks) is a separate orchestrator integration pass,
  because `tests/test_cli.py::test_publish_challenger_result_map_covers_live_active_registry`
  asserts `cli.PUBLISH_CHALLENGER_RESULT_KEYS` exactly equals the set of
  live `ACTIVE_PROSPECTIVE` challengers using that command, and six
  concurrent workers editing that one map would collide. Until that pass
  lands, `weekly_recording_command` in the challenger registration names the
  standalone script above, never
  `nfl-ats publish-predictions --record-decisions`.
- `artifacts/prospective/challengers.json`: a PROPOSED entry only (this
  build does not edit that file -- see the orchestrator's atomic
  integration pass), registered as `bye_edge_fade_overlay`, status
  `ACTIVE_PROSPECTIVE`, `model` block a snapshot of the active configuration
  at registration time (for fingerprint-mismatch detection only, mirroring
  the existing challengers).
- **Not wired anywhere:** there is no switch that applies the fade to the
  published card. Playing this fade for real is a separate owner decision
  this document does not make.
- **Each challenger is tracked independently against the active model's own
  card, not stacked on the other overlays.** The stacked back-test above is
  a diagnostic read, not how the challenger is scored week to week --
  `nfl-ats prospective-score` settles this challenger's own recorded picks
  against the active model's own paper ledger, paired game by game, exactly
  like every other overlay challenger.

## Both grades, from Week 1

`nfl-ats prospective-score` settles every `ACTIVE_PROSPECTIVE` challenger,
including this one, at both grades automatically once picks are recorded:
`decision_line` (the spread the pick was actually made at -- primary) and
`close_line` (the same pick re-settled at the close -- secondary). No
challenger-specific scoring code is needed; this is the same generic
machinery the other overlay challengers already use
(`docs/prospective_evidence.md`). Week 1 2026 itself records zero flips (see
above), so the first week this challenger's arm can diverge from the active
model's own picks is whenever 2026's first bye week lands.

## Tests

`tests/test_bye_edge_fade_overlay.py`, mirroring
`tests/test_surface_switch_tilt_overlay.py`'s structure:

1. `bye_edge_flag_by_game`: fires on a strict (>=12-day) gap, does NOT fire
   on an 11-day gap (the boundary case), fires for both teams simultaneously
   when both qualify, is `False` for a team's first game of the season (no
   prior game), raises on missing columns, and two leakage regression tests
   (the flags never depend on any outcome-bearing column, since neither is
   even read; a future season's schedule data never changes an earlier
   season's flags, and a season opener specifically never inherits the
   prior season's gap -- pinning the 2026-08-22 fix).
2. `apply_bye_edge_fade_overlay`: flips a pick on the strict-bye-holding
   side, does NOT flip a both-off-bye game, does NOT flip when neither side
   is off a strict bye (the 11-day case), does NOT flip when the model's
   pick is already on the non-bye side, leaves postseason games untouched,
   treats a missing schedule row as no signal, is a no-op when disabled, and
   changes only `home_cover_probability` on flipped rows (byte-identical
   everywhere else, and zero effect outside the flagged population).
3. `overlay_disclosure_note`: empty when nothing flipped, states the flip
   count and matchups when something did, and explicitly says "not applied
   to the published card".
4. `record_bye_edge_fade_challenger_decisions`: records the fade's own arm
   (which can diverge from the active model's raw pick), is append-only and
   idempotent, refuses outside the recording lock window, refuses a
   fingerprint mismatch (an active-model promotion, or any retuned/foreign
   model config, under the challenger's feet), and refuses an inactive
   registration.

## Decision left to the orchestrator

Nothing here decides anything -- it only starts a free evidence stream, plus
a mined-seasons context read that already leans favorable (P+ 0.8474
week-blocked, 0.9980 season-blocked). Once 2026 accrues enough weeks past
its first bye,`nfl-ats prospective-score` reports this challenger's
`probability_positive` at both grades, paired against the active model,
exactly like the other overlay challengers. That number is new, independent
evidence about the FADE rule specifically -- it neither replicates nor
substitutes for the registry reads above.
