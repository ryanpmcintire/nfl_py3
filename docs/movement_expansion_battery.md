# Movement expansion battery: predeclaration

Written 2026-08-31, **frozen before any accuracy sign below is computed.**
This is the ONE predeclared expansion shot at the late-week observed-movement
channel the owner asked for this session, on top of an already busy family:
`movement_rule_composed_v1` (the live prospective challenger,
`docs/movement_composition_eval.md`), `docs/observed_movement_channel.md`
(the oracle/threshold-overlay design this document reuses), and
`docs/movement_attribution.md` (the injury/weather/public-alignment
decomposition, whose strongest cell, `movement_attribution_pop_threshold_injury`
at +17.07 pts, P+ 0.976, n=123, is already live inside the composed rule and is
**not** re-proposed here).

## Binding closing-grounds taxonomy (verbatim, AGENTS.md -- restated per the
project's rule for any document, script, or subagent that scores or
adjudicates an experiment)

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
> validator.

Also binding and restated: within-week correlation is ZERO -- never
estimated or padded; every claim is labeled measured / read (path:line) /
reported / inferred; cells and thresholds below are frozen before any sign
is seen; numbers and intervals are reported before verdicts; decisions are
expected-value decisions, never a 0.90/95% gate.

## What already exists (read first, so nothing here is a re-run of a spent look)

- **`movement_rule_composed_v1`** (`artifacts/prospective/challengers.json`):
  follows the market when the current captured line moves >=1.0 pt off the
  frozen Tuesday line, composed onto the live four-member production chain.
  Evidence (`registry/weak_signals.json:movement_rule_composed_chain`):
  +1.5303 accuracy points vs the incumbent (three-member) chain, week-blocked
  95% [-0.8065, +3.8765] P+ 0.8942, season-blocked [-0.4902, +3.3333] P+
  0.9297, n=1,503, 2020-2025. The composition caveat (measured chain is
  three-member, live chain is the four-member OR union) is disclosed there,
  not re-litigated here.
- **`docs/observed_movement_channel.md`**: the frozen oracle/threshold design
  this document reuses verbatim (production pick =
  `pick_home_at_open_probability_rule`, grading line = `margin_vs_open`,
  threshold overlay mechanics). Its own frozen grid tested exactly two
  magnitudes, **0.5 and 1.0**, and exactly two timings, **close** and a
  **Sunday-morning ~10:55 ET realism ceiling** (2023-2025 `intraday_hourly`
  only). Neither a magnitude above 1.0 nor a timing checkpoint earlier than
  Sunday morning has ever been tested. That is this document's gap.
- **`docs/movement_attribution.md`**: decomposes the close-vs-open
  disagreement population by INJURY/WEATHER/PUBLIC cause.
  `movement_attribution_pop_threshold_injury` (+17.07 pts, P+ 0.976, n=123)
  is already live inside the composed rule's evidence chain (per the mission
  brief) and is **not** re-proposed as new here. This document's cells are a
  different cut -- magnitude and timing of the movement itself, not its
  attributed cause -- and are commensurable (same population, same
  construction, same units) but **not identical**: they must be disclosed as
  correlated, not pooled blind, with both `movement_attribution_*` and
  `observed_movement_*`.
- **Public-betting divergence** (`docs/public_betting_battery_predeclaration.md`,
  5 cells recorded 2026-08-20: fade-heavy-public at close/opener, sharp
  bet%/money% divergence, model-interaction against/diff) and the
  "steam vs. public" cut (`movement_attribution_pop_*_public_book_shading_public`
  / `_reverse_line_movement`) are **both already spent** on this same
  archive. Every point estimate in the public-betting battery leans negative
  (P+ 0.10-0.30) on n=44-91; none is re-run here. **Decision: do not re-propose
  a public-betting cell in this battery** -- it would not be a new question,
  it would be the same mined battery re-asked.
- **`odds_microstructure_H3_*` / cross-book dispersion** (`vi_disp_movement_*`,
  `sagarin_battery_*`): the oracle-ceiling and cross-book/cross-model
  divergence cells are read and are **not playable channels** -- the
  `odds_microstructure_*_oracle_*` entries are ceiling controls by the
  mission brief's own instruction, and `vi_disp_movement_top_vs_bottom_tercile`
  found the multi-book-dispersion mechanism dead at this instrument
  (split-half -0.042, ROADMAP 2026-08-23 Wave 4). Neither is reused as a
  candidate direction here.

**What is genuinely new**: the purchased point-in-time archive
(`data/market/raw`, `capture_kind="historical_backfill"`) carries daily
decision-label checkpoints beyond `tue_open` and the close --
`thu_pre_tnf` (Thursday, pre-Thursday-Night-Football) and `sat_midday`
(Saturday midday) -- across the **full 2020-2025 archive**, not just the
2023-2025 `intraday_hourly` subset Arm 3 of `observed_movement_channel.md`
used. No document in this repo has ever built a candidate from either
checkpoint. This is a genuine timing-window gap, and the frozen 0.5/1.0
threshold grid leaves a genuine magnitude gap above 1.0. Both are chosen as
the two candidate directions for this battery, per the mission brief's own
suggested list ("movement-magnitude or timing-window variants not already
inside the composed rule").

## Rotation-registry window (read first: `src/nfl_ats/rotation.py`, `registry/rotation_registry.json`)

**Measured**: no family with "movement" in its name exists in
`registry/rotation_registry.json` (12 families as of this session:
`best_pick_ranker`, `best_pick_ranker_opener`, `cfb_role_continuity`,
`combined_stacker`, `era_weighting_half_life_8`, `fluview_elevated_on_production`,
`fluview_home_elevated_opener`, `graph_off_sack_rate_on_production`,
`graph_ratings_v2_team_stat`, `mod07_weak_signal_stack`, `pbp_drive_bundle`,
`player_qb_continuity`). This is therefore a NEW family, declared and drawn
through the CLI before any cell below is scored:

```
nfl-ats rotation declare --name movement_expansion_v1 \
  --description "Movement-magnitude (>=2.0 pt, untested tier above the frozen 0.5/1.0 grid) and movement-timing (thu_pre_tnf, sat_midday checkpoints, untested timing beyond tue_open/close/Sunday-AM) expansion of the observed-movement channel, threshold-overlay on the production probability-rule pick, graded at the frozen Tuesday opener." \
  --grade opener --acknowledge-mined
nfl-ats rotation assign --name movement_expansion_v1
```

`grade=opener` because every cell needs a resolvable `tue_open` consensus
(`GRADE_POOLS["opener"] = (2020, 2025)`, `src/nfl_ats/rotation.py`).
`--acknowledge-mined` is REQUIRED, not optional: the opener pool (2020-2025)
sits entirely inside `MINED_SEASONS = (2018, 2025)`, so `eligible_blocks`
returns an EMPTY tuple for any opener-graded family that does not acknowledge
it (**measured**, previewed in a sandboxed, unsaved `Registry` object before
committing anything: `R.eligible_blocks(hypothetical_family, size=2)` returns
`()` when `acknowledges_mined_2018_2025=False` and
`((2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025))` when
`True`). This matches AGENTS.md's own "opener windows are not scarce"
memory: windows retire per-family, not globally, and every opener-graded
confirmation in this registry (`mod07_weak_signal_stack`,
`best_pick_ranker_opener`, both spent `[2020, 2021]`) has already
acknowledged the same mined range for the same structural reason.

**Window drawn**: default size (`DEFAULT_WINDOW_SIZE["opener"] = 2`), the
same convention both existing opener-graded families used (neither requested
a larger 3-4 season block). The earliest eligible block for a brand-new
family with no prior windows is deterministically **`[2020, 2021]`** (the
lowest-starting 2-season block in the opener pool that starts at or after
`MIN_ELIGIBLE_START_SEASON=2011` -- trivially satisfied since the opener pool
itself starts at 2020). Per rule 4 (per-family retirement), this is legal
even though `mod07_weak_signal_stack` and `best_pick_ranker_opener` already
spent the identical `[2020, 2021]` block -- retirement is per-family, and
this is a new, independent family.

**Consequence for scoring**: every cell's PRIMARY (confirmation-grade) figure
below is computed on exactly the drawn window's seasons, **2020 and 2021
only** -- not the full 2020-2025 archive the sibling mined-battery documents
used. This is a genuinely smaller, genuinely fresh-to-this-family look, and
it is scored honestly at that size rather than silently widened to the
full archive for more power. `nfl-ats rotation record` marks the window spent
once every cell is scored, regardless of verdict.

## Population (frozen, verified this session before any sign was computed)

Base archive: `artifacts/opener_evaluation/20260819T174244Z/per_game.parquet`
(the exact 1,537-game paired 2020-2025 archive `docs/opener_evaluation.md`,
`docs/movement_composition_eval.md`, and `docs/observed_movement_channel.md`
all reuse read-only). Restricted to `season in {2020, 2021}` (the drawn
window): **[read, measured] 466 games** (2020: 227, 2021: 239 -- matches the
window's own `tue_open` coverage exactly, since the archive already requires
both a `tue_open` consensus and a resolvable close). 10 pushes excluded from
every accuracy cell (`correct_at_open_probability_rule` null), leaving 456
non-push graded games -- the identical count `mod07_weak_signal_stack` and
`best_pick_ranker_opener` both report for this exact window, confirming the
population matches their precedent exactly.

Production pick throughout: `pick_home_at_open_probability_rule` (the rule
`pool.py`/`backtest.py` actually play, not the sign rule). Grading line
throughout: `margin_vs_open = result - tue_open_home_spread` (the frozen
Tuesday line the pool grades against, per AGENTS.md's "grade the decision at
the opener" rule).

**Timing-checkpoint coverage** (`nfl_ats.clv.build_pairing_table`,
`capture_kind="historical_backfill"`, `labels=DECISION_LABELS`,
`seasons=(2020, 2021)`, schedule-joined to REG games, then inner-joined to
the 466-game window on `game_id`) -- **[measured]**:

| checkpoint | games with a resolvable reading | coverage of the 466-game window |
|---|---:|---:|
| `thu_pre_tnf` | 463 | 99.4% |
| `sat_midday` | 435 | 93.3% |
| both | 435 | 93.3% |

Coverage is high and comparable across both seasons (not concentrated in one
season) -- a genuine, well-populated timing dimension, not a sparse backfill
like the public-betting archive. Games missing a checkpoint's own reading are
DROPPED from that checkpoint's cells only (not credited with "kept pick" as
if that were an informed no-movement read); the drop count is reported per
cell in the results table.

## Cells (5, frozen before scoring; effect units `accuracy_points` throughout)

For a "current" home-spread column `CUR` (one of `close_home_spread`,
`thu_pre_tnf_home_spread`, `sat_midday_home_spread`), define, identically to
`observed_movement_channel.md`'s Arm 1/Arm 2 construction:

```
move            = CUR - tue_open_home_spread
movement_home   = True  if move > 0   (line moved toward home)
                  False if move < 0   (line moved toward away)
                  undefined if move == 0
oracle pick     = movement_home if move != 0 else production_pick
threshold pick  = movement_home if |move| >= T else production_pick
```

Paired flip-value = candidate pick minus production pick, both graded at
`margin_vs_open`, per-game correctness via `nfl_ats.clv.pick_correct`, mean
difference = the reported effect (`accuracy_points` = fraction * 100, the
same unit-scaling convention every sibling movement doc uses to avoid the
x100 bug flagged in `odds_microstructure_H3_*`'s own registry notes).

1. **`movement_expansion_window_close_threshold_1_0`** -- reproduction/
   consistency cross-check, NOT a new mechanism: the already-established
   close-timing, 1.0-pt-magnitude threshold overlay
   (`observed_movement_threshold_1_0` / `movement_rule_composed_chain`'s
   underlying construction), scored ONLY on this family's drawn
   `[2020, 2021]` window instead of the full 2020-2025 archive. Answers "does
   the already-known real direction still lean the same way on the specific
   233-week-block slice this family is allowed to look at" -- an instrument
   consistency check, not a fresh hypothesis. Population: all 466 (close is
   always present in this archive by construction).
2. **`movement_expansion_thu_oracle_full_slate`** -- TIMING, ceiling read.
   Oracle at the `thu_pre_tnf` checkpoint (always follow the Thursday-implied
   side when `thu_pre_tnf != tue_open`, else keep production). Answers: how
   much of the eventual close-oracle's ceiling is already visible by
   Thursday, well before the composed rule's typical "current captured line"
   read and well before the Friday-final injury report that
   `movement_attribution_pop_threshold_injury`'s mechanism concentrates in.
   Population: 463 (drops 3 missing `thu_pre_tnf`).
3. **`movement_expansion_thu_threshold_1_0`** -- TIMING, playable rule.
   Same 1.0-pt threshold construction as cell 1, but `CUR = thu_pre_tnf`
   instead of close. Directly answers the decision-relevant question: would
   the SAME frozen threshold rule, read on Thursday instead of waiting for
   the close, still be worth playing. Population: 463.
4. **`movement_expansion_sat_threshold_1_0`** -- TIMING, playable rule, later
   checkpoint. Same construction, `CUR = sat_midday`. Saturday sits after
   most Friday-final injury news lands (the mechanism
   `movement_attribution.md` found carries the value) but still well before
   Sunday kickoff -- the natural "did I already capture most of the
   Friday-injury-driven move by Saturday" checkpoint. Population: 435.
5. **`movement_expansion_close_threshold_2_0`** -- MAGNITUDE, untested tier.
   Same close-timing construction as cell 1, but `T = 2.0` instead of 1.0 --
   the next grid point above the frozen 0.5/1.0 pair, testing whether a
   stricter magnitude bar concentrates value the way
   `movement_attribution.md`'s injury cut did (unfiltered +5.26 pts ->
   threshold-1.0 +9.66 pts -> threshold-injury +17.07 pts: value
   concentrating as the cut sharpens). Population: all 466.

No cell here duplicates `movement_attribution_pop_threshold_injury` or any
`public_betting_battery_*` cell: none of these five is a cause-attribution
cut or a public-betting cut. They are commensurable with the existing
`observed_movement_*` / `movement_rule_composed_chain` family (same
population type, same construction, same `accuracy_points` unit) and MUST be
disclosed, at record time, as correlated with it -- never pooled blind. Every
`--notes` field below states this explicitly; `--family movement_expansion`
is passed explicitly rather than left to name-prefix inference.

## Positive control (frozen before scoring)

**Perfect-foresight control**, scored on the identical `[2020, 2021]`/466-game
population as every cell above, NOT recorded to `registry/weak_signals.json`
(an instrument diagnostic, not a hypothesis): candidate pick =
`home_covers = margin_vs_open > 0` (the REALIZED outcome, a deliberate,
total leak of the settlement itself into the "pick"). This proves the
harness -- the exact bootstrap, population, and pairing machinery this
document's five real cells reuse -- CAN detect and fully resolve a very
large effect (both week- and season-blocked intervals entirely positive, no
"contains zero" ambiguity) at this exact window's n=456-466. It is not a
size-matched control (it does not bound what a modest true effect would look
like at this n the way RWB-15's calibrated 0.5/1/2-point synthetic replicas
do); it is a gross sensitivity check that this script's construction is not
structurally broken. RWB-15's own week-blocked detection rates for calibrated
NFL synthetic effects (0.5 pt: 3/8 replicas resolve; 1 pt: 2/8; 2 pt: 7/8;
`docs/... RWB-15 sensitivity audit`) remain the standing, already-measured
reference for what size effect this evaluator can resolve at typical n --
cited, not re-derived, here.

## Within-week permutation null (frozen before scoring)

For each of the five cells, 200 draws (matching
`scripts/fluview_home_elevated_opener_look.py`'s own `NULL_PERMUTATIONS`
convention): `margin_vs_open` is shuffled WITHIN each (season, week) group
(never across weeks -- this is a re-grading of the two FIXED picks under a
shuffled outcome, not a re-fit, and it costs no extra model fits since
neither pick depends on the outcome), the paired candidate-minus-production
delta is recomputed under each shuffle, and the null's mean, 2.5th/97.5th
percentiles, and the observed delta's percentile rank within that null are
reported. This null is **not centred on zero by design** (the home-tilt
null-artifact lesson, `MEMORY: home-tilt-null-artifact`): the production
pick itself carries a home/away tilt unrelated to any movement mechanism, so
a permutation null built from the SAME fixed picks inherits that tilt. It is
reported ALONGSIDE the week-blocked bootstrap, never in place of it, and
never read as a single yes/no test.

## Bootstrap (frozen)

`nfl_ats.clv.week_blocked_bootstrap`, `samples=20_000`, **`seed=20260831`**
(this document's own seed, distinct from every prior movement document's:
`observed_movement_channel.md` used 20260819, `movement_attribution.md`
20260820, `movement_composition_eval.md` 20260822), `block="week"` primary
and `block="season"` secondary. Week-blocking already treats within-week
games as non-independent draws at the block level; per the project's binding
ICC=0 mandate, no separate ICC term is estimated or padded anywhere in this
design.

## Reporting contract (binding, AGENTS.md)

Every cell is reported regardless of sign or whether its interval contains
zero. `probability_positive` is reported for every cell; "contains zero" is
never used as a verdict. A cell is proposed `refuted_mechanism` /
`wrong_sign_resolved` ONLY if both the week-blocked AND season-blocked 95%
intervals sit entirely below zero; `bounded_by_control` is not available to
any cell here (the perfect-foresight control proves gross sensitivity, not a
size-matched null result). Every cell not meeting an admissible terminal
ground is recorded `unresolved_below_power` via `nfl-ats weak-signals
record`, `--family movement_expansion`, before any narrative treats it as
settled. The rotation family's window is recorded spent via
`nfl-ats rotation record --name movement_expansion_v1` once all five cells
are scored, with verdict `unresolved` unless a cell resolves a terminal
ground (in which case the family's own verdict reflects that cell,
per the `mod07_weak_signal_stack` / `best_pick_ranker_opener` precedent of
recording the family's headline cell's own numbers with full-battery detail
in `--notes`).

## What this is not

- Not a re-proposal of `movement_attribution_pop_threshold_injury` or any
  `public_betting_battery_*` / `movement_attribution_pop_*_public_*` cell --
  all are already spent and are read, not re-run, above.
- Not a change to the live `movement_rule_composed_v1` challenger, the
  published card, or any challenger registry entry. Measurement and
  recording only; promotion is an orchestrator decision.
- Not a claim that the drawn `[2020, 2021]` window is the full archive's
  answer -- it is one rotation-registry-governed look at a smaller,
  genuinely fresh-to-this-family slice, honestly sized rather than widened
  for more power.
