# Movement leads battery: predeclaration (LEAD-01, LEAD-06, LEAD-07)

Written 2026-09-05, **frozen before any accuracy sign below is computed.**
This document predeclares a new rotation family, `movement_leads_v1`, that
answers three Phase 12 ROADMAP leads on the local point-in-time odds
archive: LEAD-07 (movement timing decomposition by day-part), LEAD-01
(Wednesday-revision follow, which turns out to be LEAD-07's own cell (a) in
its playable form -- declared as its own registry entry per the mission
brief), and LEAD-06 (rising-total, stable-spread underdog). It reuses the
observed-movement threshold-overlay construction from
`scripts/observed_movement_channel.py` and the rotation-window-governed
scoring shape from `scripts/movement_expansion_battery.py` /
`docs/movement_expansion_battery.md` verbatim (production pick =
`pick_home_at_open_probability_rule`, grading line = `margin_vs_open`, week-
blocked bootstrap primary, within-week permutation null, perfect-foresight
positive control) -- read both first.

## Binding closing-grounds taxonomy (verbatim, AGENTS.md)

> An interval or CI that contains zero is NEVER grounds to reject, fail, or
> close an experiment. At this evaluator's ~2-point resolution, "contains
> zero" is the EXPECTED outcome for a real small signal. Only two grounds
> ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
> (whole interval on the wrong side of zero) or zero split-half reliability;
> (2) bounded by a positive control proven able to detect an effect that
> size. Everything else is `unresolved_below_power`: record it with
> `nfl-ats weak-signals record`, report `probability_positive`, never the
> binary "contains zero". The registry code hard-rejects inadmissible
> closures; if a record command errors, the verdict is wrong, not the
> validator. Verdicts flow only through `nfl-ats weak-signals record` /
> `nfl-ats rotation record`, never through prose.

Also binding and restated: within-week correlation is ZERO -- never
estimated or padded; every claim is labeled measured / read (path:line) /
reported / inferred; cells and thresholds below are frozen before any sign
is seen; the primary cell is named before scoring; decisions are
expected-value decisions (`probability_positive` above 0.5 favours the
candidate), never a 0.90/95% gate.

## Instrument facts, read before any cell was designed

**[read]** `src/nfl_ats/odds_backfill.py:52-59` -- the `historical_backfill`
decision-label archive that covers the full 2020-2025 window defines exactly
six checkpoints: `tue_open` (Tue 09:00 ET), `thu_pre_tnf` (Thu 18:00 ET),
`sat_midday` (Sat 12:00 ET), `sun_early_close` (Sun 12:30 ET),
`sun_late_close` (Sun 16:15 ET), `mon_pre_mnf` (Mon 19:00 ET). **There is no
Wednesday checkpoint anywhere in this label set.** The only place a
Wednesday reading exists is the finer-grained `intraday_hourly` archive.

**[measured]**, `nfl_ats.clv.load_snapshot_manifest_index` grouped by
`decision_label` and, for `intraday_hourly` rows, by ET weekday: the
`intraday_hourly` capture kind covers **only seasons 2023, 2024, 2025** (18
weeks each), with per-weekday manifest counts Tuesday 1,242 / Wednesday
1,296 / Thursday 1,242 / Friday 1,296 / Saturday 1,242 / Sunday 594 / Monday
54. Wednesday coverage is real and substantial, but it exists **only** in
this 2023-2025-only archive -- confirming LEAD-01's own ROADMAP-stated
instrument ("2023-2025 hourly archive, Wed-noon minus Tue-open") is not an
arbitrary scope choice, it is the *only* way to get a Wednesday reading at
all.

**Consequence, disclosed before any cell is built**: LEAD-07's day-part
split names three segments -- (a) Tuesday-open to Wednesday, (b) Wednesday
to Saturday-midday, (c) Saturday to Sunday-morning. The mission brief marks
only (c) as "hourly arm, 2023-2025 only, disclosed" -- but since (a) and (b)
both use Wednesday as an endpoint, and Wednesday only exists in the
2023-2025 `intraday_hourly` archive, **all three day-part cells, plus the
(c)-minus-(a) difference and LEAD-01, are 2023-2025-only populations**, not
just (c). This is measured now, before any accuracy sign, and is treated as
a predeclared scope correction to the ROADMAP row text, not a post-hoc
rationalization.

**[read]** `docs/observed_movement_channel.md:204-225` -- the archive's real
Sunday coverage ceiling is ~10:55 ET regardless of a nominal 13:00/16:00 ET
cutoff request (every one of 54 `intraday_hourly`-covered weeks has its last
Sunday-local capture at local hour 10, zero variance). Cell (c)'s
"Sunday-morning" endpoint is this same ~10:55 ET ceiling, not a literal
16:00 ET reading; `min(kickoff, Sunday 16:00 ET)` is still the requested
cutoff (the owner's binding per-game deadline), it simply resolves to
whatever the archive last captured before that clock time.

Also read: `docs/late_week_refresh.md:225` (same ~10:55 ET ceiling, cited
there for the live weekly-refresh cadence, not re-derived).

## Rotation-registry window (declared and assigned before any cell was scored)

**[measured]**, this session, via the actual CLI (not simulated):

```
nfl-ats rotation declare --name movement_leads_v1 \
  --description "Phase 12 movement-timing/lead battery (LEAD-01/06/07): day-part decomposition of the observed-movement follow rule (Tue-open to Wed-noon, Wed-noon to Sat-midday, Sat-midday to Sunday-morning realism ceiling), the Wednesday-revision follow rule in its playable form, and a rising-total/stable-spread underdog flag. Threshold-overlay on the production probability-rule pick, graded at the frozen Tuesday opener." \
  --grade opener --acknowledge-mined
nfl-ats rotation assign --name movement_leads_v1
```

Declared successfully (`status: "open"`, `acknowledges_mined_2018_2025:
true`). `--acknowledge-mined` is required for the same structural reason
`movement_expansion_v1` needed it: the opener pool (`GRADE_POOLS["opener"] =
(2020, 2025)`) sits entirely inside `MINED_SEASONS = (2018, 2025)`.

**Assigned window: `[2020, 2021]`** -- exactly as `eligible_blocks` predicts
for a brand-new family with no `--inherits` (**[measured]**, previewed
before the real assign call in a sandboxed, unsaved `Registry`:
`eligible_blocks` returns `((2020, 2021), (2021, 2022), (2022, 2023),
(2023, 2024), (2024, 2025))`, and `assign_window` always takes the first).
This matches the same earliest-block outcome `mod07_weak_signal_stack`,
`best_pick_ranker_opener`, and `movement_expansion_v1` all drew -- per-family
retirement (AGENTS.md, `opener-windows-are-not-scarce`), not scarcity.

**Disclosed tension, resolved before scoring, not after**: `[2020, 2021]`
has **zero overlap** with the 2023-2025 seasons every day-part/Wednesday
cell requires (measured above: the `intraday_hourly` archive does not exist
before 2023). `assign_window` always draws the *earliest* eligible block
deterministically -- there is no CLI option to request a later block, and
no `--inherits` chain exists yet that would exclude 2020-2021 for this
brand-new family. Two cells are unaffected by this (see below); four are
not. The resolution, decided here, before any cell is scored:

- **`movement_leads_rising_total_dog`** (LEAD-06) needs only `tue_open` and
  the latest pre-deadline `historical_backfill` checkpoint (`thu_pre_tnf`
  / `sat_midday` / `sun_early_close`), all of which exist across the full
  2020-2025 archive (**[measured]** below: 100% coverage in the assigned
  window). This cell is scored on the assigned `[2020, 2021]` window,
  exactly as `movement_expansion_v1`'s convention requires, and is this
  family's one genuinely window-matched cell.
- **`movement_leads_wed_follow_1_0`, `movement_leads_sat_follow_1_0`,
  `movement_leads_sun_am_follow_1_0`, `movement_leads_sun_vs_wed_per_point`**
  cannot be computed on `[2020, 2021]` at all (zero Wednesday or
  Sunday-morning-hourly readings exist there). They are scored on the full
  2023-2025 `intraday_hourly`-covered population instead -- the *only*
  population LEAD-01's and LEAD-07(c)'s own ROADMAP-declared instruments
  ever named. This is disclosed explicitly in each cell's own
  `weak-signals record --season-start 2023 --season-end 2025` call (not
  `2020`/`2021`, which would misstate what was actually measured) and in
  the family's `rotation record` notes. The consequence for governance,
  stated plainly: the `movement_leads_v1` family's assigned `[2020, 2021]`
  window is spent by this look for the purposes of never re-litigating this
  *family name* on that block again, but it does **not**, strictly, cover
  the 2023-2025 population the day-part cells actually used -- a future
  session proposing a *different* Wednesday/day-part construction on
  2023-2025 data would need its own fresh governance (a family whose grade
  pool naturally lands there, since `opener`'s deterministic earliest-block
  rule cannot be pointed at 2023-2025 directly). This gap is named here
  rather than silently worked around.

## Population (measured this session, before any accuracy sign)

**LEAD-06 window population** (`[2020, 2021]`, base archive
`artifacts/opener_evaluation/20260819T174244Z/per_game.parquet`): **466
games** (2020: 227, 2021: 239 -- identical to `movement_expansion_v1`'s own
reported count for this window, confirming no drift in this reused
archive). Latest pre-deadline checkpoint (`tue_open` <
`thu_pre_tnf` < `sat_midday` < `sun_early_close`, in that priority order,
excluding `sun_late_close`/`mon_pre_mnf` because both sit structurally after
the owner's Sunday 16:00 ET deadline) resolves for **466 of 466 games
(100%)**: `sun_early_close` 424, `thu_pre_tnf` 28 (Thursday games, already
kicked off before any Saturday/Sunday checkpoint), `sat_midday` 11
(early-kickoff Sunday games whose own kickoff preceded 12:30 ET),
`tue_open` 3 (no later checkpoint resolvable at all -- these three
necessarily read zero total/spread movement and are never flagged).

**Day-part population** (2023-2025, base archive restricted to those
seasons): **816 games** (272 per season). Checkpoint coverage
(**[measured]**, `nfl_ats.clv.load_decision_quotes` +
`decision_market_consensus`, per-game `min(kickoff, target ET time)`
cutoff, duplicated from `observed_movement_channel.py`'s Arm-3 cutoff
construction since that helper is module-private there, exactly as that
script's own docstring explains for its own duplicated `_true_week_correct`):

| checkpoint | resolvable | coverage |
|---|---:|---:|
| `wed_noon` (Wed 12:00 ET, `intraday_hourly`) | 816 / 816 | 100.0% |
| `sat_midday` (`historical_backfill`) | 750 / 816 | 91.9% |
| `sun_am` (min(kickoff, Sun 16:00 ET), real ceiling ~10:55 ET, `intraday_hourly`) | 816 / 816\* | 100.0%\* |

\*Coverage of the raw cutoff read is 816/816; cell (c)'s actual population
is 750 because it additionally requires `sat_midday` (see below).

Per-cell population (games with every checkpoint that cell needs):

| cell | population | drop reason |
|---|---:|---|
| (a) Tue-open -> Wed-noon | 816 | none (wed_noon always resolvable) |
| (b) Wed-noon -> Sat-midday | 750 | 66 missing `sat_midday` |
| (c) Sat-midday -> Sun-morning | 750 | 66 missing `sat_midday` |
| (c)-(a) joint (needs all four checkpoints) | 750 | same 66 |

**Move-size facts** (mean absolute incremental move, points, over each
cell's own resolvable population -- population-construction facts, not an
accuracy sign): (a) 0.3392, (b) 0.4957, (c) 0.2927. Threshold-eligible
(|move| >= 1.0) counts: (a) 105/816, (b) 164/750, (c) 72/750 -- all
comfortably populated relative to this archive's ~54 available week-blocks
(18 weeks x 3 seasons), so a week-blocked bootstrap has no structural
sparse-week risk in any of the three arms.

## Construction (frozen before scoring)

For any two home-spread readings `EARLIER` and `LATER` (chosen per cell
below) and the production pick `PROD = pick_home_at_open_probability_rule`:

```
move            = LATER - EARLIER
movement_home   = True  if move > 0   (line moved toward home)
                  False if move < 0   (line moved toward away)
                  undefined if move == 0
threshold pick  = movement_home if |move| >= 1.0 else PROD
```

identical to `observed_movement_channel.py`'s / `movement_expansion_battery.py`'s
own `threshold_pick`/`oracle_pick` helpers (imported unmodified from
`scripts.movement_expansion_battery` rather than re-implemented, since they
are pure functions of two spread series and a threshold -- no change to
that file). Paired flip-value = candidate pick minus production pick, both
graded at `margin_vs_open` (the frozen Tuesday line), per-game correctness
via `nfl_ats.clv.pick_correct`; `accuracy_points` = fraction * 100.

**Day-part endpoints** (chained across the week, each day-part's move is
INCREMENTAL -- the change within that specific segment, not cumulative from
Tuesday -- so each day-part answers "does a fresh move happening
specifically here carry information", independent of what any other
day-part's cell does):

- (a) `EARLIER = tue_open`, `LATER = wed_noon` (Wed 12:00 ET,
  `intraday_hourly`, 2023-2025). This is simultaneously LEAD-07 cell (a)
  and LEAD-01's own predeclared construction ("follow the Tuesday->Wednesday
  move when >= 1.0 point... recomputed pick graded at the frozen Tuesday
  line") -- ONE computation, registered once as
  `movement_leads_wed_follow_1_0`, per the mission brief's own instruction
  that LEAD-01 "is cell (a) of LEAD-07 in its playable form."
- (b) `EARLIER = wed_noon`, `LATER = sat_midday` (`historical_backfill`,
  restricted to the 2023-2025 games with a resolvable `wed_noon`).
- (c) `EARLIER = sat_midday`, `LATER = sun_am` (`min(kickoff, Sunday 16:00
  ET)`, real ceiling ~10:55 ET, `intraday_hourly`).

**LEAD-06 (rising-total, stable-spread dog)**: `EARLIER = tue_open`,
`LATER` = the latest pre-deadline `historical_backfill` checkpoint
(`tue_open` < `thu_pre_tnf` < `sat_midday` < `sun_early_close`, excluding
`sun_late_close`/`mon_pre_mnf` as post-deadline). Flag = `(LATER_total -
EARLIER_total) >= 2.0` AND `|LATER_spread - EARLIER_spread| < 0.5`.
Candidate pick on a flagged game = the underdog at the frozen Tuesday line
(`tue_open_home_spread > 0` => home is the dog => pick home; `< 0` => pick
away; the exact-zero pick'em case, expected to be rare, falls back to the
production pick, matching every other tie-break convention in this
family). Candidate pick elsewhere = production pick. Flagged-game counts
are reported per season (not predeclared as a specific count, since this is
a population fact only visible after running the flag, not an accuracy
sign).

## Cells (5 registered, frozen order and names before scoring)

1. **`movement_leads_wed_follow_1_0`** -- day-part (a) / LEAD-01, threshold
   1.0, `EARLIER=tue_open`, `LATER=wed_noon`. Population 816, 2023-2025.
2. **`movement_leads_sat_follow_1_0`** -- day-part (b), threshold 1.0,
   `EARLIER=wed_noon`, `LATER=sat_midday`. Population 750, 2023-2025.
3. **`movement_leads_sun_am_follow_1_0`** -- day-part (c), threshold 1.0,
   `EARLIER=sat_midday`, `LATER=sun_am`. Population 750, 2023-2025.
4. **`movement_leads_sun_vs_wed_per_point`** -- the paired difference in
   PER-POINT VALUE between (c) and (a): for each arm, restricted to that
   arm's own >=1.0-point-eligible subset of the joint 750-game population,
   `per_point_value = paired_delta_accuracy_points / mean(|move|)` among
   that subset (accuracy points earned per point of observed movement).
   Reported quantity = `per_point_value(c) - per_point_value(a)`, week-
   blocked bootstrapped jointly on the 750-game population so both arms'
   eligible subsets are drawn from the same resampled weeks every
   iteration. **Predeclared direction: positive** (Sunday-morning moves
   carry MORE per-point value than Wednesday moves -- injury/lineup news
   concentrated late in the week versus Tuesday-to-Wednesday's thinner,
   more flow-driven information).
5. **`movement_leads_rising_total_dog`** -- LEAD-06, `[2020, 2021]` window,
   466 games.

**Primary cell, declared before scoring**: `movement_leads_sun_vs_wed_per_point`
(cell 4) is this family's primary cell for the single
`nfl-ats rotation record --name movement_leads_v1` call, per the mission
brief's own proposal. It is the cell that most directly answers LEAD-07's
predeclared hypothesis (day-part value concentration) rather than a single
threshold rule's raw accuracy, and it is the cell every other day-part cell
here exists to feed.

## Positive control (frozen before scoring)

Perfect-foresight control, identical construction to
`movement_expansion_battery.py`'s own (`observed_movement_channel.md`'s
sensitivity-check convention): candidate pick = `margin_vs_open > 0` (the
realized settlement outcome itself, a deliberate total leak), scored on
BOTH the 750-game day-part population and the 466-game LEAD-06 window
population, NOT recorded to `registry/weak_signals.json` (instrument
diagnostic only). Proves the harness -- population, pairing, bootstrap --
can fully resolve a large effect at each of this family's two population
sizes; it is not a size-matched control (RWB-15's calibrated synthetic-
replica detection rates remain the standing reference for what a modest
true effect looks like at comparable n, cited not re-derived here).

## Within-week permutation null (frozen before scoring)

200 draws per cell (`scripts.movement_expansion_battery.null_distribution`,
imported unmodified -- same convention as
`fluview_home_elevated_opener_look.py`'s `NULL_PERMUTATIONS`), `margin_vs_open`
shuffled within each `(season, week)` group, the FIXED candidate/production
picks re-graded under each shuffle. Not centred on zero by design (the
home-tilt null-artifact lesson) -- reported alongside the week-blocked
bootstrap, never in place of it.

## Bootstrap (frozen)

`nfl_ats.clv.week_blocked_bootstrap`, `samples=20_000`, **`seed=20260905`**
(this document's own seed, distinct from every sibling movement document:
`observed_movement_channel.md` 20260819, `movement_attribution.md`
20260820, `movement_composition_eval.md` 20260822,
`movement_expansion_battery.md` 20260831), `block="week"` primary and
`block="season"` secondary. Per the binding ICC=0 mandate, no separate
correlation term is estimated for within-week games.

## Reporting contract (binding, AGENTS.md)

Every cell is reported regardless of sign or whether its interval contains
zero. `probability_positive` is reported for every cell; "contains zero" is
never used as a verdict. A cell may be proposed `refuted_mechanism` /
`wrong_sign_resolved` ONLY if both the week-blocked AND season-blocked 95%
intervals sit entirely below zero; `bounded_by_control` is not available to
any cell here (the perfect-foresight control proves gross sensitivity, not
a size-matched null). Every cell not meeting an admissible terminal ground
is recorded `unresolved_below_power` via `nfl-ats weak-signals record`,
`--family movement_leads_v1`, `--category market`, before any narrative
treats it as settled. The four day-part/Wednesday cells are recorded with
`--season-start 2023 --season-end 2025` (the population actually used);
`movement_leads_rising_total_dog` is recorded with
`--season-start 2020 --season-end 2021` (the assigned window it actually
used). The family's rotation-registry window (`[2020, 2021]`) is recorded
spent via `nfl-ats rotation record --name movement_leads_v1` once every
cell is scored, using cell 4's own numbers as the family headline per the
`mod07_weak_signal_stack` / `best_pick_ranker_opener` precedent, with the
season-mismatch disclosed above restated in `--notes`.

## What this is not

- Not a re-proposal of any `movement_expansion_*`, `movement_attribution_*`,
  or `observed_movement_*` cell -- all are read, not re-run, above; every
  cell here is commensurable with that family (same construction, same
  `accuracy_points` unit) and MUST be disclosed as correlated with it, not
  pooled blind.
- Not a change to the live `movement_rule_composed_v1` challenger, the
  published card, or any challenger registry entry. Measurement and
  recording only; promotion is a separate, later decision.
- Not a claim that `[2020, 2021]` is this family's full answer for the
  day-part cells -- it structurally cannot be (see the disclosed tension
  above); those four cells' honest population is 2023-2025, stated as such
  in their own registry season fields.

## Results (added after the look, 2026-09-05)

All numbers in this section are **[measured]**, read directly from
`artifacts/movement_leads_battery/20260905T042928Z/metadata.json` and
cross-checked against `cells_summary.csv` in the same artifact directory,
and from the five `movement_leads_*` entries plus the `movement_leads_v1`
family in `registry/weak_signals.json` / `registry/rotation_registry.json`.

**Window as the CLI assigned it**: `registry/rotation_registry.json`
`families.movement_leads_v1` -- `grade: "opener"`, `acknowledges_mined_2018_2025:
true`, one window, seasons `[2020, 2021]`, `state: "spent"`, `verdict:
"unresolved"`, `spent_at: "2026-09-05"` -- exactly as predicted above before
any cell was scored.

**Populations (measured)**: day-part base (2023-2025) 816 games; wed_noon
and Sunday-morning-ceiling readings both resolved for **816 of 816 (100%)**
-- better coverage than predeclared population estimates assumed (no
missing-capture drops at all for either hourly cutoff). `sat_midday`
resolved for 750 of 816 (91.9%), so cells (b), (c), and the joint (a)+(c)
population are all 750 games, 733 graded (10 pushes plus games missing a
resolvable checkpoint intersection). LEAD-06 window population: 466 games,
456 graded -- identical counts to `movement_expansion_v1`'s own reported
figures for this window, confirming no archive drift.

**Cells (5, frozen order, `accuracy_points` throughout except cell 4, which
is a per-point-of-movement RATE -- see its own unit caveat below)**:

| Cell | n | Effect | Week-blocked 95% CI | Week P+ | Season-blocked 95% CI | Season P+ | Null percentile |
|---|---:|---:|---|---:|---|---:|---:|
| `movement_leads_wed_follow_1_0` (a / LEAD-01) | 799 | -0.1252 pts | [-1.6519, +1.5019] | 0.4079 | [-1.8797, +1.1236] | 0.4131 | 31st |
| `movement_leads_sat_follow_1_0` (b) | 733 | +1.5007 pts | [-0.4071, +3.4530] | 0.9306 | [+0.4065, +3.2787] | 1.0000 | 92nd |
| `movement_leads_sun_am_follow_1_0` (c) | 733 | +1.0914 pts | [-0.4098, +2.6574] | 0.9096 | [+0.4115, +1.6260] | 1.0000 | 88.5th |
| `movement_leads_sun_vs_wed_per_point` (c minus a, **primary**) | 750 | +0.0941 pts/pt | [-0.0314, +0.2161] | 0.9278 | [+0.0162, +0.2149] | 1.0000 | 91st |
| `movement_leads_rising_total_dog` (LEAD-06) | 456 | -0.4386 pts | [-1.0965, **0.0000**] | 0.0000 | [-0.4545, -0.4237] | 0.0000 | -- |

Cell 4's own components (not separately registered): per-point value on
Wednesday-eligible games (n=94) is **-0.0139** pts/pt; on Sunday-morning-
eligible games (n=72) is **+0.0802** pts/pt. The predeclared direction
(Sunday-morning moves carry MORE per-point value than Wednesday moves)
holds on the point estimate and leans that way in 92.8%/100% of
week-/season-blocked resamples, but the week-blocked interval still crosses
zero -- an EXPECTED below-power reading at this resolution, not a negative.

**Positive controls** (perfect-foresight, not recorded to the signal
registry): on the 733-game day-part population, **+46.9304 accuracy
points**, week-blocked 95% [+43.4066, +50.5435] P+ 1.0, season-blocked
[+44.4444, +50.8197] P+ 1.0. On the 456-game window population, **+46.2719
accuracy points**, week-blocked [+40.7809, +51.8125] P+ 1.0, season-blocked
[+44.9153, +47.7273] P+ 1.0 -- matching `movement_expansion_v1`'s own
reported control on the identical window (+46.2719) to the fourth decimal,
an exact cross-check that this family's window-population construction is
byte-for-byte the same archive slice. Both controls confirm gross harness
sensitivity at both population sizes; neither is a size-matched control for
any individual cell's effect.

**A note on `movement_leads_rising_total_dog`'s week-blocked interval**:
its upper bound is **exactly 0.0**, not a small negative number -- a
discrete-mass artifact of only 6 flagged games existing in the entire
456-game window (5 in 2020, 1 in 2021): many of the 20,000 bootstrap
resamples draw none of the 6 flagged-game weeks and score an exact-zero
paired delta, which is what both the upper quantile and `probability_positive
= 0.0` (literally 0 of 20,000 draws exceeded zero) mechanically reflect.
This is disclosed explicitly rather than rounded away: it is NOT the same
thing as a smoothly resolved negative interval, and per this document's own
predeclared AND-rule (both week- and season-blocked intervals entirely
below zero), the week-blocked side's exact-zero boundary makes
`wrong_sign_resolved` inadmissible even though the season-blocked interval
alone is entirely negative. The cell stays `unresolved_below_power`, and
the flagged-count caveat (6 of 456 games) is the more important fact than
either interval endpoint.

**Registry names** (all five, `family: "movement_leads"`, `classification:
"unresolved_below_power"`, `closing_ground: null`, `recorded_at:
"2026-09-05"`, `category: "market"`, source
`artifacts/movement_leads_battery/20260905T042928Z/metadata.json`):
`movement_leads_wed_follow_1_0` (seasons [2023,2025]),
`movement_leads_sat_follow_1_0` (seasons [2023,2025]),
`movement_leads_sun_am_follow_1_0` (seasons [2023,2025]),
`movement_leads_sun_vs_wed_per_point` (seasons [2023,2025], the per-point
RATE unit caveat is in its own `--notes`), `movement_leads_rising_total_dog`
(seasons [2020,2021], the actual assigned window). Rotation family
`movement_leads_v1` recorded via `nfl-ats rotation record`, verdict
`unresolved`, headline numbers = cell 4's own (the predeclared primary),
full battery detail in `--notes`, per the `mod07_weak_signal_stack` /
`best_pick_ranker_opener` / `movement_expansion_v1` precedent of recording
a family's headline cell with full detail in notes.

**Artifact**: `artifacts/movement_leads_battery/20260905T042928Z/`
(`metadata.json`, `cells_summary.csv`, `per_game_dayparts.parquet`,
`per_game_dayparts_joint.parquet`, `per_game_rising_total.parquet`).

**What this implies for the decision, before what is wrong with it**: this
battery does not add a playable rule today. `probability_positive` ranges
from 0.0 (`rising_total_dog`, a 6-game subgroup, not a stable reading in
either direction) to 0.93 (`sat_follow_1_0`). The two most interesting
reads are (1) `sat_follow_1_0` and `sun_am_follow_1_0` both lean strongly
positive in isolation (P+ 0.93 / 0.91) while `wed_follow_1_0`/LEAD-01 reads
as a coin flip (P+ 0.41) -- consistent with later-week moves (closer to
Friday-final injury news) carrying more value than earlier-week moves; and
(2) the primary cell's direct per-point comparison confirms this ordering
on its point estimate (Sunday-AM +0.0802 pts/pt vs Wednesday -0.0139
pts/pt, diff +0.0941, P+ 0.928) without resolving it. None of the five
cells meets an admissible closing ground, so per AGENTS.md all five stay
`unresolved_below_power`, not closed, and remain available for a future
pooled read alongside the rest of the movement family (disclosed above as
correlated with `observed_movement_*`/`movement_expansion_*`, not pooled
blind here). The rising-total-dog cell's extremely small flagged count
(6 games) is the standing reason it should not be read as evidence against
LEAD-06's mechanism -- it is underpowered by construction, not refuted.
