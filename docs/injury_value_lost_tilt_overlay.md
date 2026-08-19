# Injury value-lost tilt overlay: a no-window-cost prospective challenger

Written 2026-08-19. Follows `docs/injury_value_lost.md` (`injury_value_lost_narrowed`:
+1.316 accuracy points, `probability_positive` 0.8875 on the already-spent
456-game `[2020, 2021]` opener window; split-half reliability 0.9325; survives
a market-move control and a drop-QB stress test; `unresolved_below_power`),
and the `hc_year_one_fade_overlay` / `best_pick_nomination_v2` precedents
(`docs/coach_fade_overlay.md`, `docs/best_pick_ranker.md` § nomination-v2) for
wiring a pick-level, post-prediction transform into the prospective challenger
ledger at zero rotation-registry window cost.

**What this is not.** `docs/injury_value_lost.md` section 7 freezes a
predeclaration for a REAL NFL opener-window confirmation of
`injury_value_lost_narrowed` (the `player_value`-vs-`player` isolation),
which would draw `[2022, 2023]` -- one of only two opener windows left in the
project -- and its own text says explicitly: **"Do not assign until the 2026
prospective number is in hand"** for `mod07_weak_signal_stack`
(`docs/injury_value_lost.md` lines 380-385). That window is **not** spent by
this document, by the registration below, or by anything else in this
repository as of 2026-08-19; `mod07_weak_signal_stack` has zero games played
this season (2026 Week 1 has not kicked off), so the gating condition plainly
has not been met, and this build does not attempt to satisfy it early.

What this document DOES build is the free path: a **parameter-free pick-level
tilt**, dual-tracked against whatever model is currently active, in the
prospective challenger ledger, exactly like `hc_year_one_fade_overlay`. It
accrues its own independent 2026 evidence in parallel with the
`mod07_weak_signal_stack` stream, at no window cost, and **is not applied to
the published card** -- no owner decision to play it has been made, unlike
the coach-fade overlay.

---

## The rule, exactly as built

Implemented in `src/nfl_ats/injury_value_tilt_overlay.py`.

For every game on the active model's own weekly card, read the same two
pregame-available columns `docs/injury_value_lost.md` section 4 isolated as
the cleanest available construct (also named in
`nfl_ats.surgical_gating.VALUE_LOST_DIFF_COLUMNS`, imported rather than
re-declared so the two modules can never drift):

```
diff_injury_skill_epa_value_lost
diff_injury_defense_disruption_value_lost
```

from `data/processed/game_features_player.parquet` -- the canonical,
FIXED-prior-severity table (manifest `player_feature_version: "v2"`: no
learned-availability table was passed at build time, matching the exact
isolation the registry entry measured), rebuilt every Tuesday by
`weekly-run`'s step 3 (`build-player-features`) regardless of which profile
the card path itself is currently using.

```
value_lost_diff(game) = diff_injury_skill_epa_value_lost + diff_injury_defense_disruption_value_lost
                       = (home_skill_lost - away_skill_lost) + (home_defense_lost - away_defense_lost)
```

`diff_X = home_X - away_X` is `features.py`'s existing convention (confirmed
by reading `src/nfl_ats/features.py:309`), so a positive total means the HOME
team lost strictly more value than the AWAY team.

**The tilt (zero threshold, zero tuning):**

```
model_pick_home = home_cover_probability >= 0.5

flip when:
    model_pick_home  AND  value_lost_diff > 0   (model picked the MORE-hurt home team)
 OR
    NOT model_pick_home  AND  value_lost_diff < 0   (model picked the MORE-hurt away team)
```

In plain language: **tilt toward the side with less injury value lost,
whenever the model's own pick disagrees with that side.** A game where the
differential is exactly zero (the common case -- 29.86% of games have no
listed value-lost differential of either kind, per
`docs/surgical_injury.md` section 1.2) is left untouched, since a zero
differential carries no directional information. A game where the model's
pick already agrees with the healthier side is also left untouched -- there
is nothing to tilt.

**No threshold exists anywhere in this rule.** Unlike
`nfl_ats.surgical_gating.gate_by_value_lost_magnitude` (a DIFFERENT, already-
frozen candidate that defers to a baseline pick when the construct's
*magnitude* is small, using a distribution-derived threshold,
`docs/surgical_injury.md`), this tilt acts purely on the *sign* of the
differential. It requires no derived constant, so there is nothing to derive
from pre-2018 data and nothing that could have been tuned against 2018-2025
outcomes -- satisfying the task's own admissibility bar
("no peeking at 2018-2025 outcomes to pick thresholds; any threshold must be
derived from pre-2018 data or be parameter-free") by having no free parameter
at all.

**Regular-season only.** The construct's split-half reliability and mechanism
screen were both measured on regular-season games only
(`docs/injury_value_lost.md` section 3.1: "4,431 completed REG games"), so
the overlay gates on `game_type == "REG"` when that column is present,
mirroring `coach_fade_overlay`'s convention. Unlike the coach-fade overlay,
there is no week-8 cutoff -- the measured mechanism (`docs/injury_value_lost.md`
section 2-3) spans the full 35-week `[2020, 2021]` opener window, so the tilt
applies in every regular-season week.

---

## Why this construction, not a full retrained-model challenger

The obvious alternative -- register `player_value` as its own margin-predict
feature profile, exactly like `mod07_weak_signal_stack` does for `weak_stack`
-- was considered and rejected for this build. Two reasons:

1. **It would not test the isolated construct.** `docs/injury_value_lost.md`
   section 4's `player_value`-vs-`player` isolation is clean only because
   both arms share every other input. A `player_value`-profile challenger
   dual-tracked against WHATEVER model is currently active (`weak_stack`
   as of 2026-08-18, which already carries value-lost, bias, and
   learned-availability columns) would instead measure "does a leaner,
   fixed-severity, value-lost-only model beat the current production
   model" -- a legitimate question, but a different one from the isolation
   the registry entry actually measured, and one that risks silently
   re-introducing a semantics-shift confound if the active profile changes
   again.
2. **The task asked for a "tilt"**, and named `hc_year_one_fade_overlay` /
   `best_pick_nomination_v2` as the precedents -- both pick-level, post-
   prediction transforms of the active model's own picks, not separately
   trained models. A pick-level tilt is also strictly cheaper to reason
   about: it touches zero training frames, zero feature profiles, and zero
   stored model artifacts, exactly like the coach-fade overlay's own
   "Design choice" rationale (`docs/coach_fade_overlay.md`).

A full `player_value`-profile prospective challenger remains buildable later,
following the exact `mod07_weak_signal_stack` pattern (register a `model`
block with `feature_profile: "player_value"`, `feature_table:
"data/processed/game_features_player.parquet"` -- the FEATURE_SETS machinery
already selects the right column subset; no new code required), if a future
session wants that different, complementary question answered. It is not
built here.

---

## What is and is not wired in

- `src/nfl_ats/injury_value_tilt_overlay.py`: the transform
  (`apply_injury_value_tilt_overlay`), the signal reader
  (`raw_value_lost_diff`), the disclosure sentence
  (`overlay_disclosure_note`, not currently surfaced anywhere), and the
  recorder (`record_injury_value_tilt_challenger_decisions`).
- `src/nfl_ats/cli.py`'s `_cmd_publish_predictions`: a **fourth**, purely
  additive, fail-open `try`/`except` block (mirroring the existing three for
  the paper ledger, the coach-fade overlay, and the v2 nomination) that calls
  the recorder when `--record-decisions` is passed. This writes ONLY to
  `artifacts/prospective/challenger_decisions.parquet`; it never touches
  `recommendations.csv`, `CURRENT_PREDICTIONS.md`, `README.md`, or the public
  site. **The production pick path (`publish_active_predictions`) is
  untouched by this build.**
- `artifacts/prospective/challengers.json`: registered as
  `injury_value_lost_tilt_overlay`, status `ACTIVE_PROSPECTIVE`, `model`
  block a snapshot of the active configuration at registration time (for
  fingerprint-mismatch detection only, mirroring `hc_year_one_fade_overlay`).
- **Not wired anywhere:** there is no `OVERLAY_ENABLED`-style switch that
  applies the tilt to the published card. Playing this tilt for real is a
  separate owner decision this document does not make.

---

## Both grades, from Week 1

`nfl-ats prospective-score` settles every `ACTIVE_PROSPECTIVE` challenger,
including this one, at both grades automatically once picks are recorded:
`decision_line` (the spread the pick was actually made at -- primary) and
`close_line` (the same pick re-settled at the close -- secondary). No
challenger-specific scoring code was needed; this is the same generic
machinery `mod07_weak_signal_stack`, `hc_year_one_fade_overlay`, and
`best_pick_nomination_v2` already use (`docs/prospective_evidence.md`).

---

## Tests

`tests/test_injury_value_tilt_overlay.py`, mirroring
`tests/test_coach_fade_overlay.py`'s structure:

1. `raw_value_lost_diff` reads exactly the two shared columns and nothing
   else, and raises on a missing column.
2. `apply_injury_value_tilt_overlay`: flips the clean-disagreement case in
   both directions (home hurt more / away hurt more), leaves a tie untouched,
   leaves an already-agreeing pick untouched, leaves postseason games
   untouched, treats a missing feature row as a tie, is a no-op when
   disabled, and changes only `home_cover_probability` on flipped rows
   (byte-identical everywhere else).
3. `overlay_disclosure_note`: empty when nothing flipped, states the flip
   count and matchups when something did, and explicitly says "not applied
   to the published card".
4. `record_injury_value_tilt_challenger_decisions`: records the tilt's own
   arm (which can diverge from the active model's raw pick), is append-only
   and idempotent, refuses outside the recording lock window, refuses a
   fingerprint mismatch (an active-model promotion under the challenger's
   feet), refuses an inactive registration, and refuses a missing feature
   table.

`tests/test_cli.py`'s two `publish-predictions` recording tests were extended
with a monkeypatched fourth recorder call, matching the existing pattern for
the paper ledger, the overlay, and the nomination rule.

---

## Decision left to the orchestrator

Nothing here decides anything -- it only starts a free evidence stream.
Once 2026 accrues enough weeks, `nfl-ats prospective-score` reports this
challenger's `probability_positive` at both grades, paired against the
active model, exactly like the other three prospective challengers. That
number is new, independent evidence about the TILT rule specifically -- it
neither replicates nor substitutes for docs/injury_value_lost.md section 7's
own NFL opener-window confirmation, which remains correctly deferred until
the `mod07_weak_signal_stack` prospective number is in hand.
