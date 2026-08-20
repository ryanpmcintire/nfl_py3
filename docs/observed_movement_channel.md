# Observed line-movement channel: predeclaration

Written 2026-08-20, before any accuracy sign for this experiment is
computed. Companion to `docs/opener_evaluation.md` (the frozen production
protocol this reuses) and `scripts/odds_microstructure_battery.py`'s H3
cells (a different, already-recorded read of observed movement — see
"Relationship to already-recorded entries" below, read and reconciled
*before* this document was written, not after).

## Why this experiment, and why now

**Owner correction, 2026-08-20 (binding for this document):** the pool's
picks are editable until each game's own kickoff; only the **grading
line** freezes Tuesday noon (revised once Wednesday, then frozen for the
week). `docs/pool_edge_plan.md`'s "pool format" section previously said
picks lock Tuesday — that was the error, now corrected in that file. This
means a late-week pick refresh that uses market information observed
*after* Tuesday, but still graded against the frozen Tuesday line, is a
legitimate, leak-free pool strategy: nothing about it uses information
from after any given game's own kickoff.

This experiment measures what that refresh channel is worth, using
**observed** market movement only (the close, or a later intraday
snapshot, both of which are archived, already-realized numbers with no
forecasting model in between) — as opposed to the `movement_direction_tilt_*`
family recorded 2026-08-19, which flips picks using MKT-06's **predicted**
movement (a fitted, walk-forward direction model). Observed movement is a
strictly different, and strictly less speculative, channel: it requires no
model at all, only reading the board again before submitting.

## Relationship to already-recorded entries (reconciliation, read first)

Two families already touch this ground. Both were opened and read before
any new number below was produced.

1. **`movement_direction_tilt_opener` (+ two variants), recorded
   2026-08-19**, `artifacts/movement_tilt_screen/20260819T160330Z/metadata.json`.
   Read: these flip the opener pick to the side a **fitted, walk-forward
   ridge model's predicted** post-open movement favors (median/no-filter/
   top-quartile confidence gates on the *prediction*), not the side the
   market actually, observably moved. Confirmed: this is the PREDICTED-movement
   channel the task description anticipated finding here. No overlap with
   this document's OBSERVED-movement design; both are legitimate, separate
   channels and are not reconciled further.
2. **`odds_microstructure_H3_*` (3 cells), recorded 2026-08-18**,
   `artifacts/odds_microstructure/20260818T225430Z/metadata.json`, described
   in that script as "reproduce the project's own movement-oracle formula
   exactly ... pick the side the line moves toward, settle at the frozen
   opener." This genuinely **does** use observed movement in pick space,
   and predates this document. It is NOT a duplicate of what follows,
   for three concrete reasons, and both stand as independent, legitimate
   reads:
   - It always plays the movement side and **drops every zero-movement
     game** from its population (`oracle_correct` is `NaN`-masked at
     `open_move == 0`); this document's oracle arm instead keeps those
     games by falling back to the production pick, so it reports a
     genuine **full-slate** accuracy on the whole paired population, not a
     movement-only subset.
   - It reports accuracy **only against a flat 50% baseline** ("delta vs a
     50% baseline"); it never computes a **paired delta against the
     production model's own picks on the same games**, which is what a
     pool decision (replace my pick with the movement pick, yes/no)
     actually needs.
   - It has no threshold overlay (flip only when the move clears a
     magnitude bar) and no Sunday-morning-realism arm using the
     2023-2025 hourly archive; both are new here.
   - **The Tuesday-to-Wednesday partial-window read this task asked to
     find and cite is exactly `odds_microstructure_H3_3_1_tue_to_wed_oracle_2023_2025`**:
     effect **+4.4669 accuracy points** (54.47% point accuracy), interval
     [-1.3353, +10.1108] week-blocked, `probability_positive` 0.9326,
     n=347, 2023-2025, source
     `artifacts/odds_microstructure/20260818T225430Z/cells_summary.csv`.
     [read: registry/weak_signals.json, key
     `odds_microstructure_H3_3_1_tue_to_wed_oracle_2023_2025`]. The
     "full-week" (Tue-to-close) companions in the same family read
     **+5.075 points** (2020-2025, n=1133) and **+5.707 points**
     (2023-2025, n=587) — both already recorded, both already
     `unresolved_below_power` (a positive-leaning interval that still
     contains zero at the tighter subsample is the expected shape at this
     evaluator's resolution, not a rejection).

No recorded entry mislabels predicted movement as observed, or vice
versa; the taxonomy in the task description held. This document proceeds
to measure the two things the H3 family did not: the **paired-vs-production**
read or a movement-based decision, **threshold-gated overlays** on the
production pick, and a **Sunday-morning realism** arm built from the actual
hourly archive rather than a single Wednesday-noon snapshot.

## Population and protocol (frozen before scoring)

Base population: `nfl_ats.clv.opener_pick_evaluation`'s own paired frame —
the identical weekly-refit `weak_stack`/ridge/alpha-10 protocol
`docs/opener_evaluation.md` uses (one model per (season, week), trained
only on strictly-earlier completed games), run against every archived game
with both a `tue_open` consensus and a resolvable close, 2020-2025
(reported n at run time; `docs/opener_evaluation.md` records 1,537 such
games). Production pick = `pick_home_at_open_probability_rule`
(`home_cover_probability_at_open >= 0.5`), the rule `pool.py`/`backtest.py`
actually play (`docs/opener_evaluation.md`, 2026-08-19 addendum) — NOT the
sign rule, since a decision about what to actually submit must be graded
against what production actually submits. Grading line throughout:
`margin_vs_open = result - tue_open_home_spread` (the frozen Tuesday
grading line, per the owner correction above).

### Arm 1 — Oracle (full-slate)

For every paired game: `oracle_pick_home = True` if
`close_home_spread > tue_open_home_spread` (line moved toward home), `False`
if it moved toward away, and **the production pick** if
`close_home_spread == tue_open_home_spread` exactly (no games dropped —
this is the full-slate design difference from the H3 family noted above).
Graded against `margin_vs_open`. Reported: full-slate oracle accuracy,
full-slate production accuracy on the identical population, and the
**paired delta** (oracle-minus-production, per game, week-blocked
bootstrap).

### Arm 2 — Threshold overlays

Two predeclared cells: flip the production pick to the movement side only
when `|close_home_spread - tue_open_home_spread| >= 0.5`, and only when
`>= 1.0`; below threshold (or exactly zero movement) the production pick is
kept unchanged. For each cell, report: full-slate accuracy of the
overlay rule, the paired delta vs production (same bootstrap design as
Arm 1), the count of flip-eligible games (movement clears the threshold),
the count of games where the flip actually changes the pick (movement
side disagrees with production), and the **model-agrees-already
fraction** — of flip-eligible games, the share where the movement side
already matches the production pick (so the overlay is a no-op there).

### Arm 3 — Sunday-morning realism (2023-2025 hourly coverage only)

Same construction as Arm 1 (oracle) and Arm 2 (threshold overlays), with
the close replaced by the last `intraday_hourly` capture at or before a
per-game cutoff: `min(that week's Sunday 13:00 ET, that game's own
kickoff)`. This makes early-week games (Thursday/Friday/Saturday) use the
last capture before their own kickoff, exactly as predeclared, while every
Sunday-or-later game (including Monday night) uses the single
Sunday-morning refresh — the realistic "check the board once before the
early slate, resubmit, done" cadence. Restricted to the 2020-2025
population intersected with seasons 2023-2025 (the archive's only
`intraday_hourly` coverage; [measured] 6,966 manifests, all in
2023/2024/2025). Games with no `intraday_hourly` capture before their own
cutoff are dropped from this arm only, and the drop count is reported.
Sunday-ET cutoff per (season, week) is derived from each game's own
`commence_time_utc` (converted to America/New_York), not from the
`gameday`/`gametime` feature columns, to avoid a second timestamp
convention; documented assumption: the week's Sunday is the first
`America/New_York` Sunday on or after the week's earliest kickoff, which
holds for every normal NFL week and is not re-derived per exception.

### Bootstrap

`nfl_ats.clv.week_blocked_bootstrap`, `samples=20_000`,
**`seed=20260819`** (this document's own seed, distinct from the H3
family's `20260818`), `block="week"` primary and `block="season"`
secondary, on every cell above. Week-blocking already treats within-week
games as non-independent draws at the block level and needs no separate
ICC term — per the project's binding ICC=0 mandate, no such term is
estimated or padded anywhere in this script.

## Reporting contract (binding, AGENTS.md)

Every cell is reported regardless of sign or whether its interval
contains zero. `probability_positive` is reported for every cell; the
phrase "contains zero" is never used as a verdict. Every cell not meeting
an admissible terminal ground (`wrong_sign_resolved` with the WHOLE
interval below zero, `no_split_half_reliability`, or
`positive_control_bound` — no positive control is run in this experiment,
so that ground is not available here) is recorded
`unresolved_below_power` via `nfl-ats weak-signals record`, reporting
`probability_positive`, before any narrative write-up treats it as
settled. Recorded under `observed_movement_*` names (checked against
`registry/weak_signals.json` for collisions before recording — none
found as of this predeclaration).

## Amendment, 2026-08-20 (same day, before any run's numbers were used)

Owner refinement: **picks cannot change after Sunday 16:00 ET.** The true
per-game decision deadline is `min(kickoff, Sunday 16:00 ET)`, not the
13:00 ET guess this document originally used for Arm 3. Consequences,
applied to the script before it was run for the results reported back:

1. **Arm 3 now reports two cutoff variants.** `sunday_1600_realism` is the
   realizable/primary read, built from the last `intraday_hourly` capture
   at or before `min(kickoff, Sunday 16:00 ET)` — for SNF/MNF this
   correctly uses the Sunday-afternoon line, not that game's own close.
   `sunday_1300_conservative` is kept alongside (already built, and
   strictly inside the true deadline for every game) as a deliberately
   conservative comparison point, not the headline.
2. **Arm 1's close-based oracle is disclosed, not corrected.** For
   SNF/MNF/late-Sunday-afternoon games, the close prints AFTER the true
   16:00 ET deadline, so Arm 1 (which grades the close) overstates what
   is actually reachable on those specific games — it remains a genuine
   upper bound, just not a fully playable one for that subset. Arm 1's
   cell reports the count and share of 2023-2025 games that are
   "deadline-bound" (their own kickoff is after Sunday 16:00 ET) as a
   measured [read] quantity, and points to `sunday_1600_realism`'s oracle
   cell as the deadline-respecting variant for exactly that subset.
3. No other predeclared design element changes: population, production
   pick definition, grading line, threshold values, and the bootstrap
   protocol (samples/seed/blocking) are all unchanged from the sections
   above.

## Addendum, 2026-08-20: the archive's real Sunday ceiling is ~10:55 ET, not 13:00/16:00

**[measured]**, after the corrected script ran: the `sunday_1600_realism` and
`sunday_1300_conservative` cells came back numerically IDENTICAL (same
816-game population, same `sunday_home_spread` per game to the decimal,
same accuracy/paired-delta/interval/P+ to six decimal places). Verified
this was not a code bug — `nfl_ats.clv.load_snapshot_manifest_index`,
grouped by `(season, week)`, shows every one of the 54
`intraday_hourly`-covered weeks has its LAST Sunday-local-time capture at
local hour **10** (i.e., the archive stops around 10:55 ET Sunday
morning), with **zero variance across all 54 weeks** — `Monday` rows exist
but only one per week (a single early-morning "week close-out" capture,
not meaningful coverage). Both nominal cutoffs (`13:00`, `16:00` ET) sit
strictly after that real ceiling for every game, so `min(cutoff,
kickoff)` resolves to the same last-available capture (~10:55 ET) under
either nominal hour — hence identical output. This is a genuine data
coverage gap in the archive between ~11:00 ET Sunday and kickoff, not a
defect in the deadline logic (the `deadline_bound` diagnostic, which
counts games whose OWN kickoff is later than the nominal cutoff, correctly
differs between the two hours: 310 games at 16:00 vs. 384 at 13:00 — the
per-game cutoff arithmetic works; the archive simply has nothing to serve
past ~10:55 ET regardless of which later hour is requested).

**Consequence for the registry.** Recording `sunday_1600_realism` and
`sunday_1300_conservative` as two independent signals would silently
double-count one measurement under two names — a pooling hazard this
project's binding commensurability rule exists to prevent. The registry
therefore records ONE merged Sunday-realism family per cell
(`observed_movement_oracle_sunday_am_realism`,
`observed_movement_threshold_0_5_sunday_am_realism`,
`observed_movement_threshold_1_0_sunday_am_realism`), honestly named for
what was actually measured — the archive's real Sunday-morning ceiling,
not either nominal target — carrying this note in full. The
`artifacts/observed_movement_channel/<run_id>/metadata.json` run artifact
itself is left as-generated (both cutoff labels present, byte-identical,
which is itself the evidence of this finding) rather than re-run, since
no result changes — only the registry naming/labeling does.

**Consequence for the EV statement.** The measured Sunday-realism numbers
are therefore a LOWER BOUND on what a true end-of-window (kickoff- or
16:00-ET-respecting) refresh is worth: real Sunday-morning market
movement between ~11:00 ET and kickoff (including same-day injury news,
a well-documented driver of late moves) is not captured by this archive
and is not measured here. The true value of a full 16:00 ET refresh
remains unmeasured and is plausibly higher, not lower, than the number
reported.

## What this experiment is not

- Not a new forecasting model — every input (`tue_open`, `close`, the
  selected `intraday_hourly` snapshot) is a realized, already-observed
  market quote at the time it would be read.
- Not a claim about playability logistics beyond "the information exists
  before the relevant kickoff" — book/pool submission mechanics are out of
  scope here.
- Not a re-selection of the active model or its features; the production
  pick is read from the frozen `weak_stack`/ridge/alpha-10 recipe exactly
  as `docs/opener_evaluation.md` runs it, unmodified.
