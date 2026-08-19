# Backup-QB fade overlay: a no-window-cost prospective challenger

Written 2026-08-19. Follows the `division_revenge_tilt_overlay` /
`injury_value_lost_tilt_overlay` / `hc_year_one_fade_overlay` precedent
(`docs/division_revenge_tilt_overlay.md`, `docs/injury_value_lost_tilt_overlay.md`,
`docs/coach_fade_overlay.md`) for wiring a pick-level, post-prediction
transform into the prospective challenger ledger at zero rotation-registry
window cost.

## Evidence check (read from the registry before building, as the task required)

`backup_qb_start` -- "Starting QB differs from the team's modal QB this
season (>=3 prior starts)" -- is one of 17 predeclared cells in the NFL
situational/behavioral bias battery (`scripts/nfl_bias_battery_screen.py`).
Two grades of the same construct are recorded in `registry/weak_signals.json`,
both `unresolved_below_power`:

| Entry | Window | Games | Effect (accuracy pts) | `probability_positive` |
|---|---|---|---|---|
| `bias_battery_backup_qb_start` | close-graded, 2009-2025 | 7,002 QB-baseline-eligible team-games | -0.2731 | 0.1684 |
| `bias_battery_backup_qb_start_opener` | opener-graded, 2020-2025 | 2,436 team-games (1,893 complement, 543 flagged) | -2.3578 | 0.0982 |

Both entries **read**, directly from `registry/weak_signals.json`, before this
module was built (2026-08-19). **Both grades lean the SAME way**: a negative
effect at both grades means the flagged (backup-starter) side covers LESS
than its complement -- the backup-QB side under-covers, both close- and
opener-graded (`probability_positive` well under 0.5 at both: 0.1684 and
0.0982, i.e. roughly 83% and 90% confidence in the negative direction
respectively). This satisfies this task's own evidence-check gate ("confirm
the close-graded entry's direction; if both grades lean the same way, build;
if they conflict, do not build") -- they do not conflict, so this module was
built. Both intervals also cross zero at various points; per AGENTS.md that
is the EXPECTED shape for a real small signal at this evaluator's resolution,
never grounds to decline building a no-window-cost prospective challenger.

## Important caveat (stated up front, not buried)

The active `weak_stack` model already carries QB-continuity and
injury/availability features (`docs/injury_value_lost.md`, the QB-continuity
family in `nfl_ats.experiment_runner`). **This overlay may therefore be
double-counting information the model already prices** into
`home_cover_probability` -- fading a backup-QB team the model has ALREADY
discounted for that same reason would double-discount it. That is exactly
the open question prospective dual-tracking exists to measure, not something
resolved by assumption here: if the model already fully prices the
backup-QB effect, this overlay's flips should show no edge over the model's
own raw picks; if the model under-prices it (plausible, since the model's
features describe QB *continuity*/*value lost*, not the specific "modal
starter this season" construct the bias battery measured), the overlay's
flips should show a positive edge. `nfl-ats prospective-score` settles this
empirically once 2026 accrues weeks.

## What this is not

Neither `registry/weak_signals.json` entry above spends a rotation-registry
window -- both are bias-battery re-screens, not an opener-window
confirmation run. This document and its registration do not change that, and
nothing here is an owner decision to play the fade on the real card (unlike
`hc_year_one_fade_overlay`). It is dual-tracked only.

## The construct, exactly as measured (ported, not redesigned)

A team's start is a "backup start" when its starting QB differs from that
team's own MODAL starter so far this season, once at least 3 prior starts
this season have been observed (the battery's own frozen eligibility floor
-- kept exactly, not re-derived). Ported verbatim (same running counts, same
`>= 3` threshold) from
`nfl_ats.experiment_runner._bias_battery_qb_backup_flag`, which itself ports
`scripts/nfl_bias_battery_screen.py`'s `_qb_backup_flag`:

```
for each (team, season), in gameday order:
    total_prior = number of this team's starts already counted this season
    if total_prior >= 3:
        modal_qb = the most frequent starter among those prior starts
        flag = (this game's own starter != modal_qb)
    else:
        flag = False   # not yet eligible -- no baseline to compare against
    (only after computing the flag) add this game's own starter to the count
```

Implemented in `src/nfl_ats/backup_qb_fade_overlay.py`'s
`backup_qb_flag_by_game`, which reads the newest local schedule snapshot
(`data/raw/<snapshot>/schedules.parquet`)'s `home_qb_name`/`away_qb_name`
columns directly, mirroring `coach_fade_overlay.year_one_by_game`'s data
source. "Not yet eligible" and "confidently not a backup start" both fold
into `flag = False`: a fade rule that only ever asks "is this side flagged"
cannot distinguish the two, so collapsing them is a design choice (mirroring
how `coach_fade_overlay.year_one_by_game` folds "no observed prior season"
into `False`), not a redesign of the eligibility floor itself, which is
still exactly the `>= 3` the battery measured.

**Pregame-safe by construction.** At the time of any given game, the modal
starter used to evaluate it is computed only from that same team's STRICTLY
EARLIER starts this season. A later start can never change an earlier game's
flag. Two leakage regression tests
(`tests/test_backup_qb_fade_overlay.py`) prove this empirically: mutating a
LATER start never changes an earlier game's already-computed flag, and
mutating a future season's QB history never changes an earlier season's
flags.

**Unlike the division-revenge construct, both sides of a game CAN be
flagged simultaneously** (two teams can each be starting a non-modal QB in
the same game) -- there is no "unique loser" structure here, so the overlay
follows `coach_fade_overlay`'s "clean case" pattern instead: a both-flagged
game is reported (`both_backup_games`) but never flipped, since the measured
direction is a flagged-vs-complement contrast with no measured direction for
flagged-vs-flagged.

## The rule, exactly as built (parameter-free except the frozen eligibility floor)

```
model_pick_home = home_cover_probability >= 0.5
picked_is_backup   = backup_home if model_pick_home else backup_away
opponent_is_backup = backup_away if model_pick_home else backup_home

flip when:
    picked_is_backup  AND  NOT opponent_is_backup   (the "clean case")
```

In plain language: **when the active model's pick is ON a backup-start side
and the opponent is NOT also a backup start, flip away.** REG season only
(the construct's measurements were both scored on regular-season games).
The only number anywhere in this rule is the battery's own frozen `>= 3`
eligibility floor -- there is no threshold on the FLIP decision itself, so
nothing here was derived from or tuned against 2018-2025 outcomes.

Implemented in `src/nfl_ats/backup_qb_fade_overlay.py`:
`backup_qb_flag_by_game` derives the flag;
`apply_backup_qb_fade_overlay` applies the rule at pick level;
`overlay_disclosure_note` produces the plain-English provenance sentence
(not currently surfaced anywhere).

## Why this construction, not a full retrained-model challenger

Same two reasons `docs/injury_value_lost_tilt_overlay.md` and
`docs/division_revenge_tilt_overlay.md` give for their own tilts, restated
for this construct: (1) a full retrained challenger dual-tracked against
whatever model is currently active would answer a different, confounded
question (see the double-counting caveat above) rather than the isolated
bias-battery cell measured, and (2) the task explicitly named the pick-level
tilt precedents as the pattern to follow, and a pick-level design touches
zero training frames, zero feature profiles, and zero stored model
artifacts.

## What is and is not wired in

- `src/nfl_ats/backup_qb_fade_overlay.py`: the transform
  (`apply_backup_qb_fade_overlay`), the signal reader
  (`backup_qb_flag_by_game`), the disclosure sentence
  (`overlay_disclosure_note`, not currently surfaced anywhere), and the
  recorder (`record_backup_qb_fade_challenger_decisions`).
- `src/nfl_ats/cli.py`'s `_cmd_publish_predictions`: a **sixth**, purely
  additive, fail-open `try`/`except` block (mirroring the existing five)
  that calls the recorder when `--record-decisions` is passed. This writes
  ONLY to `artifacts/prospective/challenger_decisions.parquet`; it never
  touches `recommendations.csv`, `CURRENT_PREDICTIONS.md`, `README.md`, or
  the public site. **The production pick path
  (`publish_active_predictions`) is untouched by this build.**
- `artifacts/prospective/challengers.json`: registered as
  `backup_qb_fade_overlay`, status `ACTIVE_PROSPECTIVE`, `model` block a
  snapshot of the active configuration at registration time (for
  fingerprint-mismatch detection only, mirroring the four prior overlay
  challengers).
- **Not wired anywhere:** there is no `OVERLAY_ENABLED`-style switch that
  applies the fade to the published card. Playing this fade for real is a
  separate owner decision this document does not make.
- **Each challenger is tracked independently against the active model's own
  card, not stacked on the other overlays.** `_cmd_publish_predictions`
  calls all five (now six) overlay/nomination recorders in SEPARATE
  try/except blocks, each reading the SAME un-flipped active-model card and
  applying its own transform independently -- one overlay's flip never
  feeds another overlay's input. This matches the existing pattern exactly.

## Both grades, from Week 1

`nfl-ats prospective-score` settles every `ACTIVE_PROSPECTIVE` challenger,
including this one, at both grades automatically once picks are recorded:
`decision_line` (the spread the pick was actually made at -- primary) and
`close_line` (the same pick re-settled at the close -- secondary). No
challenger-specific scoring code was needed; this is the same generic
machinery the other overlay challengers already use
(`docs/prospective_evidence.md`).

## Tests

`tests/test_backup_qb_fade_overlay.py`, mirroring
`tests/test_division_revenge_tilt_overlay.py`'s structure:

1. `backup_qb_flag_by_game`: fires after 3 prior starts with a different QB,
   requires the eligibility floor (no flag before 3 prior starts even with
   a QB change), is false before any start this season, fires on both sides
   when both teams are backup starts, raises on missing columns, and two
   leakage regression tests (a later start's mutation does not move an
   earlier game's flag; a future season's QB history does not move an
   earlier season's flags).
2. `apply_backup_qb_fade_overlay`: flips away from the clean-case backup
   start, does not flip before the eligibility floor, does not touch a
   both-backup game, leaves postseason games untouched, does not flip a
   game with no backup signal, treats a missing schedule row as no signal,
   is a no-op when disabled, and changes only `home_cover_probability` on
   flipped rows (byte-identical everywhere else).
3. `overlay_disclosure_note`: empty when nothing flipped, states the flip
   count and matchups when something did, and explicitly says "not applied
   to the published card".
4. `record_backup_qb_fade_challenger_decisions`: records the fade's own arm
   (which can diverge from the active model's raw pick), is append-only and
   idempotent, refuses outside the recording lock window, refuses a
   fingerprint mismatch (an active-model promotion under the challenger's
   feet), and refuses an inactive registration.

## Decision left to the orchestrator

Nothing here decides anything -- it only starts a free evidence stream. Once
2026 accrues enough weeks, `nfl-ats prospective-score` reports this
challenger's `probability_positive` at both grades, paired against the
active model, exactly like the other overlay challengers. That number
answers BOTH the direction question (does the bias-battery cell's negative
lean replicate prospectively) AND the double-counting question (does the
fade add anything once the model's own QB-continuity features are already in
play) at once -- neither is resolved by assumption in this document.
