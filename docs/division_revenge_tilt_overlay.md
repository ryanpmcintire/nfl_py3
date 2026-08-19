# Division-revenge tilt overlay: a no-window-cost prospective challenger

Written 2026-08-19. Follows the `injury_value_lost_tilt_overlay` /
`hc_year_one_fade_overlay` precedent (`docs/injury_value_lost_tilt_overlay.md`,
`docs/coach_fade_overlay.md`) for wiring a pick-level, post-prediction
transform into the prospective challenger ledger at zero rotation-registry
window cost.

## Mined lineage (read, not re-derived)

`division_revenge_game` -- "2nd meeting this season vs. same opponent; team
lost the 1st meeting" -- is one of 17 predeclared cells in the NFL
situational/behavioral bias battery (`scripts/nfl_bias_battery_screen.py`,
predeclared before any cell was scored). Two grades of the same construct are
recorded in `registry/weak_signals.json`, both `unresolved_below_power`,
correlated (overlapping windows, same construct, same direction) rather than
independent confirmations of each other:

| Entry | Window | Games | Effect (accuracy pts) | `probability_positive` |
|---|---|---|---|---|
| `bias_battery_division_revenge_game` | close-graded, 2009-2025 | 8,634 team-games | +0.1907 | 0.8825 |
| `bias_battery_division_revenge_game_opener` | opener-graded, 2020-2025 | 3,006 team-games (2,748 complement, 258 flagged) | +0.2911 | 0.8642 |

Both entries **read**, directly from `registry/weak_signals.json`, before
this module was built (2026-08-19). Both intervals cross zero
(`[-0.1146, 0.5053]` close-graded, `[-0.2229, 0.7888]` opener-graded) --
per AGENTS.md, an interval crossing zero at this evaluator's ~2-point
resolution is the EXPECTED shape for a real small signal, never grounds to
close the line. Both grades sit on the same (positive) side, which is the
evidence-check bar this task set: "if both grades lean the same way, build."

## What this is not

Neither `registry/weak_signals.json` entry above spends a rotation-registry
window -- both are bias-battery re-screens (close-graded mined measurement,
opener-graded re-screen through `nfl_ats.experiment_runner`), not an
opener-window confirmation run. This document and its registration do not
change that: nothing here draws from the project's two remaining opener
windows, and nothing here is an owner decision to play the tilt on the real
card (unlike `hc_year_one_fade_overlay`). It is dual-tracked only.

## The construct, exactly as measured (ported, not redesigned)

A game is a "division revenge game" for one specific side when it is the
SECOND (or later) meeting between the same two teams in the same regular
season, and that side LOST the first such meeting. Under the current NFL
scheduling formula, two regular-season meetings between the same two teams
are (with vanishing exception) always division games, so neither the
original bias-battery cell nor this port adds an explicit `div_game` filter
-- the meeting-count logic alone reproduces the "division opponents" framing
both registry descriptions use.

Ported verbatim (same masks, same `meeting_rank >= 1 and first_margin < 0`
logic) from `nfl_ats.experiment_runner._flag_division_revenge_game`, which
itself ports `scripts/nfl_bias_battery_screen.py`'s `revenge_flag` column:

```
for each (team, opponent, season):
    sort that team's meetings against that opponent by gameday
    meeting_rank = 0-indexed occurrence count
    first_margin = team's own score margin in the FIRST such meeting
    revenge_flag = (meeting_rank >= 1) and (first_margin < 0)
```

`first_margin` reads `result` (home_score - away_score, `schedules.parquet`'s
own column) from the team's own perspective (`result` for the home side,
`-result` for the away side, `features.py`'s convention). Implemented in
`src/nfl_ats/division_revenge_tilt_overlay.py`'s
`division_revenge_side_by_game`, which reads the newest local schedule
snapshot (`data/raw/<snapshot>/schedules.parquet`) directly, mirroring
`coach_fade_overlay.year_one_by_game`'s data source.

**Pregame-safe by construction.** A game's revenge flag depends only on the
outcome of a STRICTLY EARLIER meeting between the same two teams in the same
season -- never on the current game's own result, and never on a later
meeting. The first meeting itself always has `meeting_rank == 0` and is
therefore never flagged. Two leakage regression tests
(`tests/test_division_revenge_tilt_overlay.py`) prove this empirically:
mutating a LATER meeting's result never changes an earlier meeting's
already-computed flag, and mutating a future season's schedule never changes
an earlier season's flags.

**The loser of the first meeting is unique.** Score margins are zero-sum
(modulo an exact tie), so it is IMPOSSIBLE for both teams in the same game to
qualify as the revenge side simultaneously. A tied first meeting
(`first_margin == 0` for both sides) simply produces no revenge side for
either team, and the game is left untouched -- there is no "both qualify"
case to adjudicate, unlike the coach-fade overlay's both-year-1 games.

## The rule, exactly as built (parameter-free, frozen)

```
model_pick_home = home_cover_probability >= 0.5

flip when:
    revenge_home  AND  NOT model_pick_home   (model picked AWAY from the home revenge side)
 OR
    revenge_away  AND  model_pick_home       (model picked AWAY from the away revenge side)
```

In plain language: **when the active model's forced pick is AGAINST the
revenge side, flip to the revenge side.** REG season only (the construct's
measurements were both scored on regular-season games:
`_bias_battery_merged_features` restricts the population to
`game_type == "REG"` before any cell is built). No threshold exists anywhere
in this rule -- it is a pure sign-of-disagreement flip, exactly like
`injury_value_tilt_overlay`'s construction, so there is nothing derived from
2018-2025 outcomes and nothing that could have been tuned against them.

Implemented in `src/nfl_ats/division_revenge_tilt_overlay.py`:
`division_revenge_side_by_game` derives the flag;
`apply_division_revenge_tilt_overlay` applies the rule at pick level;
`overlay_disclosure_note` produces the plain-English provenance sentence
(not currently surfaced anywhere).

## Why this construction, not a full retrained-model challenger

Same two reasons `docs/injury_value_lost_tilt_overlay.md` gives for its own
tilt, restated for this construct: (1) a full retrained challenger dual-
tracked against whatever model is currently active would answer a different
question than the isolated bias-battery cell measured, and (2) the task
explicitly named the pick-level tilt precedents
(`hc_year_one_fade_overlay` / `injury_value_lost_tilt_overlay`) as the
pattern to follow, and a pick-level design touches zero training frames,
zero feature profiles, and zero stored model artifacts.

## What is and is not wired in

- `src/nfl_ats/division_revenge_tilt_overlay.py`: the transform
  (`apply_division_revenge_tilt_overlay`), the signal reader
  (`division_revenge_side_by_game`), the disclosure sentence
  (`overlay_disclosure_note`, not currently surfaced anywhere), and the
  recorder (`record_division_revenge_tilt_challenger_decisions`).
- `src/nfl_ats/cli.py`'s `_cmd_publish_predictions`: a **fifth**, purely
  additive, fail-open `try`/`except` block (mirroring the existing four)
  that calls the recorder when `--record-decisions` is passed. This writes
  ONLY to `artifacts/prospective/challenger_decisions.parquet`; it never
  touches `recommendations.csv`, `CURRENT_PREDICTIONS.md`, `README.md`, or
  the public site. **The production pick path
  (`publish_active_predictions`) is untouched by this build.**
- `artifacts/prospective/challengers.json`: registered as
  `division_revenge_tilt_overlay`, status `ACTIVE_PROSPECTIVE`, `model`
  block a snapshot of the active configuration at registration time (for
  fingerprint-mismatch detection only, mirroring the three prior overlay
  challengers).
- **Not wired anywhere:** there is no `OVERLAY_ENABLED`-style switch that
  applies the tilt to the published card. Playing this tilt for real is a
  separate owner decision this document does not make.
- **Each challenger is tracked independently against the active model's own
  card, not stacked on the other overlays.** `_cmd_publish_predictions`
  calls `record_overlay_challenger_decisions`,
  `record_nomination_challenger_decisions`,
  `record_injury_value_tilt_challenger_decisions`,
  `record_division_revenge_tilt_challenger_decisions`, and
  `record_backup_qb_fade_challenger_decisions` in five SEPARATE try/except
  blocks, each reading the SAME un-flipped active-model card and applying
  its own transform independently -- one overlay's flip never feeds another
  overlay's input. This matches the existing pattern exactly.

## Both grades, from Week 1

`nfl-ats prospective-score` settles every `ACTIVE_PROSPECTIVE` challenger,
including this one, at both grades automatically once picks are recorded:
`decision_line` (the spread the pick was actually made at -- primary) and
`close_line` (the same pick re-settled at the close -- secondary). No
challenger-specific scoring code was needed; this is the same generic
machinery the other three overlay challengers already use
(`docs/prospective_evidence.md`).

## Tests

`tests/test_division_revenge_tilt_overlay.py`, mirroring
`tests/test_division_revenge_tilt_overlay.py`'s sibling structure
(`tests/test_coach_fade_overlay.py` / `tests/test_injury_value_tilt_overlay.py`):

1. `division_revenge_side_by_game`: fires on the away side of a rematch,
   fires on the home side of a rematch, never fires on the first meeting,
   is false on both sides after an exact tie, is false for a single
   meeting, raises on missing columns, and two leakage regression tests
   (a later meeting's mutation does not move an earlier meeting's flag; a
   future season's schedule does not move an earlier season's flags).
2. `apply_division_revenge_tilt_overlay`: flips to the away revenge side,
   flips to the home revenge side, does not flip after a tied first
   meeting, does not flip a single meeting, leaves postseason games
   untouched, does not flip when the pick already agrees, treats a missing
   schedule row as no signal, is a no-op when disabled, and changes only
   `home_cover_probability` on flipped rows (byte-identical everywhere
   else).
3. `overlay_disclosure_note`: empty when nothing flipped, states the flip
   count and matchups when something did, and explicitly says "not applied
   to the published card".
4. `record_division_revenge_tilt_challenger_decisions`: records the tilt's
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
replicates nor substitutes for the bias-battery cell's own close- and
opener-graded numbers above.
