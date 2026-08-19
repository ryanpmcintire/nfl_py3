# Spread-gap-zone fade overlay: a no-window-cost prospective challenger

Written 2026-08-19. Follows the `surface_switch_tilt_overlay` /
`backup_qb_fade_overlay` / `division_revenge_tilt_overlay` /
`injury_value_lost_tilt_overlay` / `hc_year_one_fade_overlay` precedent
(`docs/surface_switch_tilt_overlay.md`, `docs/backup_qb_fade_overlay.md`,
`docs/division_revenge_tilt_overlay.md`, `docs/injury_value_lost_tilt_overlay.md`,
`docs/coach_fade_overlay.md`) for wiring a pick-level, post-prediction
transform into the prospective challenger ledger at zero rotation-registry
window cost.

## Evidence (read from the registry before building, as the task required)

`registry/weak_signals.json:pick_conditioned_spread_gap_zone_pre2018`,
recorded 2026-08-19, `unresolved_below_power`:

The active model's own forced picks (`home_cover_probability >= 0.5`),
restricted to games where `7.5 < abs(spread_line) <= 10.0`, were **mined**
at 45.96% accuracy on the opener line (2020-2025, n=198) and 47.97% on the
close line (2018-2025, n=271) -- both below the model's own overall
accuracy, same direction. A **predeclared, never-mined replication** on
2011-2017 walk-forward picks (a window predating both mined windows) reads
46.22% accuracy (n_bucket=238 of 1,743 scored games). The registry's own
entry states the bucket bounds (7.5/10.0) were **frozen before this
replication ran** -- not fit to it.

| Read | Window | n (bucket) | Accuracy | `probability_positive` (mined direction) |
|---|---|---|---|---|
| Mined (opener) | 2020-2025 | 198 | 45.96% | -- |
| Mined (close) | 2018-2025 | 271 | 47.97% | -- |
| Predeclared replication | 2011-2017 | 238 | 46.22% | 0.9135 |

Week-blocked 95% interval (hypothesis-signed, positive = replicates the
mined direction): [-2.156, +11.698] accuracy points. **Three windows, all
the same direction** -- the model's own picks underperform specifically
inside this spread-gap zone. The interval crosses zero; per AGENTS.md that
is the EXPECTED shape for a real small signal at this evaluator's
resolution, never grounds to decline building a no-window-cost prospective
challenger.

**Frozen thresholds, not tuned on the replication.** The 7.5/10.0 bucket
bounds were fixed BEFORE the 2011-2017 replication ran (stated in the
registry entry's own description). This module reuses those same two
numbers verbatim and adds no threshold of its own.

## What this is not

The registry entry above spends no rotation-registry window -- it is a
lead-generation replication on a pre-2018 walk-forward population, not an
opener-window confirmation run (the entry's own description notes the
2011-2017 seasons overlap rotation-registry windows spent by OTHER families,
but "windows retire per-family, not globally, and this screen spends no
window of its own"). This document and its registration do not change that,
and nothing here is an owner decision to play the fade on the real card
(unlike `hc_year_one_fade_overlay`). It is dual-tracked only.

## Critical caveat: this is a PICK-CONDITIONED construct, not a market-conditioned one

Stated here because it must not be buried. The measured ~46% figures are the
accuracy of **OUR OWN model's forced picks** when restricted to this
spread-gap zone -- not the accuracy of any fixed market side (e.g. "the
underdog covers 54% of the time in this zone", which would be a
market-level, model-independent claim). That means:

- The flip's expected in-zone accuracy is only the complement of the
  measured number (roughly 54%) **IF** the lean is real **AND IF** the
  active model's own pick-generation process inside this zone stays stable
  going forward.
- A change to the active model's configuration (a promotion, a retrain)
  could change which games and which sides land in the zone and how the
  model picks them, and would not automatically inherit this measurement --
  which is exactly why `record_spread_gap_zone_fade_challenger_decisions`
  refuses to record on a fingerprint mismatch (see below) rather than
  silently continuing under a different base model.
- The 2026 prospective ledger settles the actual in-zone accuracy
  empirically instead of assuming the complement holds.

## Interaction with other overlays on the production card (stated explicitly)

This challenger is tracked **INDEPENDENTLY** against the active model's own
UN-flipped card, exactly like every other overlay challenger in this
repository. `_cmd_publish_predictions` calls every overlay recorder in
SEPARATE try/except blocks, each reading the SAME un-flipped active-model
card and applying its own transform independently -- this overlay never
sees `coach_fade_overlay`'s flips (or any other overlay's), and no other
overlay ever sees this one's.

`coach_fade_overlay` (the year-1-head-coach fade) is currently played for
real on the PUBLISHED card, weeks 1-8 of 2026 (`docs/coach_fade_overlay.md`).
If a game happens to sit in both that overlay's clean-case set AND this
overlay's spread-gap zone, the two rules could disagree about the published
pick -- but that interaction is a property of the PUBLISHED card, which this
module never touches. This challenger's own prospective evidence is always
scored against the un-flipped active model, never against whatever the
published card actually shows.

## The rule, exactly as built (frozen, unconditional within the zone)

```
abs_spread = abs(spread_line)

flip when:
    game_type == "REG"  (when present)
    AND  spread_line is present/numeric
    AND  SPREAD_GAP_LOWER_BOUND (7.5) <= abs_spread <= SPREAD_GAP_UPPER_BOUND (10.0)
```

In plain language: **when the CARD's market line satisfies
`7.5 <= abs(spread_line) <= 10`, flip the active model's pick to the other
side.** REG season only. Unlike every sibling overlay, this rule does NOT
condition on which side the model already picked -- inside the zone, EVERY
forced pick flips, unconditionally, because the measured construct is a
property of the ZONE itself, not of a particular side.

**Same data plumbing as the sibling overlays' recorders.** The flip logic
reads `spread_line` directly off the predictions/card frame -- the identical
decision-line field the sibling overlays' recorders already read for
`decision_home_spread` (`injury_value_tilt_overlay.record_injury_value_tilt_challenger_decisions`
and its siblings) -- so this module uses the SAME spread source the
published card carries, just consumed earlier (at the point of computing the
flip, not only at the point of recording it).

Implemented in `src/nfl_ats/spread_gap_zone_fade_overlay.py`:
`apply_spread_gap_zone_fade_overlay` derives the flag and applies the rule
at pick level in one step (no separate schedule- or feature-table-dependent
signal reader is needed, since the zone is entirely a function of the
card's own `spread_line`); `overlay_disclosure_note` produces the
plain-English provenance sentence (not currently surfaced anywhere).

## Why this construction, not a full retrained-model challenger

Same two reasons the sibling overlay docs give for their own tilts, restated
for this construct: (1) a full retrained challenger dual-tracked against
whatever model is currently active would answer a different, confounded
question (see the pick-conditioned caveat above) rather than the isolated,
thrice-replicated spread-gap-zone construct measured, and (2) the task
explicitly named the pick-level tilt precedents as the pattern to follow,
and a pick-level design touches zero training frames, zero feature
profiles, and zero stored model artifacts.

## What is and is not wired in

- `src/nfl_ats/spread_gap_zone_fade_overlay.py`: the transform
  (`apply_spread_gap_zone_fade_overlay`), the disclosure sentence
  (`overlay_disclosure_note`, not currently surfaced anywhere), and the
  recorder (`record_spread_gap_zone_fade_challenger_decisions`).
- `src/nfl_ats/cli.py`'s `_cmd_publish_predictions`: an **eighth**, purely
  additive, fail-open `try`/`except` block (mirroring the existing seven)
  that calls the recorder when `--record-decisions` is passed. This writes
  ONLY to `artifacts/prospective/challenger_decisions.parquet`; it never
  touches `recommendations.csv`, `CURRENT_PREDICTIONS.md`, `README.md`, or
  the public site. **The production pick path
  (`publish_active_predictions`) is untouched by this build.**
- `artifacts/prospective/challengers.json`: registered as
  `spread_gap_zone_fade_overlay`, status `ACTIVE_PROSPECTIVE`, `model`
  block a snapshot of the active configuration at registration time (for
  fingerprint-mismatch detection only, mirroring the seven existing
  challengers).
- **Not wired anywhere:** there is no `OVERLAY_ENABLED`-style switch that
  applies the fade to the published card. Playing this fade for real is a
  separate owner decision this document does not make.
- **Each challenger is tracked independently against the active model's own
  card, not stacked on the other overlays.** See "Interaction with other
  overlays" above.

## Both grades, from Week 1

`nfl-ats prospective-score` settles every `ACTIVE_PROSPECTIVE` challenger,
including this one, at both grades automatically once picks are recorded:
`decision_line` (the spread the pick was actually made at -- primary) and
`close_line` (the same pick re-settled at the close -- secondary). No
challenger-specific scoring code was needed; this is the same generic
machinery the other overlay challengers already use
(`docs/prospective_evidence.md`).

## Tests

`tests/test_spread_gap_zone_fade_overlay.py`, mirroring
`tests/test_division_revenge_tilt_overlay.py`'s structure (adapted: this
overlay has no separate schedule-derived flag function, since the zone is a
pure function of the card's own `spread_line`):

1. `apply_spread_gap_zone_fade_overlay`: flips a pick at the lower bound
   (7.5) and the upper bound (10.0) inclusive, does not flip just outside
   either bound, flips regardless of which side was originally picked
   (both a home-favorite and an away-favorite in-zone case), leaves
   postseason games untouched, does not flip a missing/non-numeric
   `spread_line`, is a no-op when disabled, and changes only
   `home_cover_probability` on flipped rows (byte-identical everywhere
   else).
2. `overlay_disclosure_note`: empty when nothing flipped, states the flip
   count and matchups when something did, and explicitly says "not applied
   to the published card".
3. `record_spread_gap_zone_fade_challenger_decisions`: records the fade's
   own arm (which can diverge from the active model's raw pick), is
   append-only and idempotent, refuses outside the recording lock window,
   refuses a fingerprint mismatch (an active-model promotion under the
   challenger's feet), and refuses an inactive registration.

## Decision left to the orchestrator

Nothing here decides anything -- it only starts a free evidence stream. Once
2026 accrues enough weeks, `nfl-ats prospective-score` reports this
challenger's `probability_positive` at both grades, paired against the
active model, exactly like the other overlay challengers. That number
answers whether the pick-conditioned zone-underperformance lean actually
holds up out-of-sample against the LIVE, currently-active model -- it is not
resolved by assumption in this document.
