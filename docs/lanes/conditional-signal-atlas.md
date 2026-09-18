# conditional-signal-atlas

## Goal

MOD-19, owner direction 2026-09-13: blanket signals are useful, but every
signal so far (served tilts, challengers, registry weak signals) was measured
alone. Going forward each signal's effect is measured inside conditioning
splits across any dimension we can name, and the card's decision is made from
the conditional reads together. Done when a predeclared split library
exists, a harness computes every registered signal's marginal inside every
split cell and records every cell with probability_positive, the findings
page shows a plain-English conditional profile per signal, and gated paired
challengers fire signals only where the conditional read is
positive-expected, graded at the opener.

## State

- Queued 2026-09-13 with ROADMAP row MOD-19. Worked case LEAD-65
  (`docs/lanes/lead65-protection-window-split.md`): the protection tilt's
  edge sits in weeks 1-4 (probability_positive 0.977) and weeks 5-18 read as
  a probable drag (0.099, unresolved). Blanket read +0.40 was an average of
  a lift and a drag.
- 2026-09-17: split library v1 predeclared at `registry/split_library.json`
  (week_in_season recorded as already-scored; ol_rush_continuity with exact
  cell rule, source columns and leakage note; eight further splits queued by
  name only and must not be scored until specified). First scored split the
  same day (subagent, parent-verified by rerun;
  `tests/scratch/lanes/pbp08_ol_rush_continuity_split_20260917.*`): tilt
  leave-one-out marginal in early low-continuity games +2.64
  [+0.58, +4.65], P+ 0.977, decisive 15-6 on 21 (exact p 0.078); early
  high-continuity cell degenerate (19 games, 0 fires, marginal
  unidentified, recorded without a standard error so it stays out of the
  pool). Both cells in `registry/weak_signals.json`, family
  `pbp08_protection_mismatch__continuity`, unresolved_below_power
  (registry now 6,663 signals).
- 2026-09-17 later: harness v1 built at `scripts/signal_atlas.py`
  (subagent, reviewed in-session; split rules read from the library, only
  week_in_season implemented, anything else errors; leave-one-out of
  signal-unique flips, paired week-blocked bootstrap, seed fixed).
  First harness run: coach fade across the three week cells, opener grade
  on the current evaluation (`opener_evaluation/20260917T161504Z`, model
  f2a706ba), parent-verified by rerun with identical output — weeks_1_4
  0.00 [-3.33, +3.24], P+ 0.462, decisive 23-23 on 46 (exact p 1.0);
  weeks_5_12 -0.31 [-1.28, +0.62], P+ 0.252, decisive 13-15 on 28
  (exact p 0.851); weeks_13_18 degenerate (ineligible past week 8,
  0 fires in 496 games, no standard error). Family
  `coach_fade__week_in_season`, all unresolved_below_power (registry now
  6,666 signals).
- 2026-09-17 later still: all nine served members through the harness,
  opener grade, 27 cells in one batch (`tests/scratch/atlas_week_batch.json`,
  recorded with `--replace`). Headlines, decisive first: pbp08 early +2.50
  P+ 0.977 dec 15-6/21 and late -1.21 P+ 0.055 dec 11-17/28; arrests late
  +0.81 [+0.20, +1.42] P+ 0.988 dec 4-0/4 (exact p 0.125); everything else
  inside ±0.6 with decisive splits near even; 9 of 27 cells degenerate
  (ineligible weeks, recorded without SE). Every cell unresolved_below_power
  — no admissible closing ground exists for a resolved-positive verdict
  anywhere in the set. Families `<signal>__week_in_season`, registry now
  6,690 signals. MOD-19 step 3 (nine served members) is done for the week
  split; the remaining library splits score next.
- 2026-09-18: split library v2 (`spread_band`: short |spread|<=7.0, long
  >=7.5, Tuesday opener, gapless in half-point steps) and the harness
  extended to dispatch it (spread_line verified 100% equal to
  tue_open_home_spread over 1,537 rows). All nine members re-run on the
  current evaluation (`opener_evaluation/20260918T161301Z`, model
  0d7f451b) and recorded in one batch (`tests/scratch/atlas_spread_batch.json`,
  18 cells, families `<signal>__spread_band`, registry now 6,708).
  Headlines, decisive first: bye long -1.53 [-2.79, -0.29] P+ 0.012
  dec 2-7/9 and tank short -0.34 [-0.60, -0.09] P+ 0.0 dec 1-5/6 both sit
  wholly negative but on 9 and 6 decisive games — held unresolved (the
  ground permits closure, nothing requires it on a thin decisive basis),
  noted not closed; pbp08 long -1.53 P+ 0.087 dec 7-12/19 crosses zero;
  arrests short +0.42 P+ 0.927 dec 8-3/11 crosses zero; rest near even.
  Batch-builder hiccup in-session (stale glob + leftover coach-append wrote
  a 21-cell mix; caught by count check, rebuilt clean 18) — scratch only,
  registry untouched until the clean batch.

## Tried

- Nothing beyond LEAD-65 yet.

## Next

1. Split library, predeclared and versioned in `registry/split_library.json`
   before any cell is scored. The list below is a starting point and is not
   exhaustive: any dimension the owner or a session can name and source
   without leakage gets added to the library and scored, and the library is
   expected to keep growing. Starting list: time in season (weeks 1-4 / 5-12 / 13-18),
   spread size band and side (favourite / dog), home / away, division game,
   rest and travel (short week, bye, long trip), weather and roof, primetime,
   era (2009-2013 / 2014-2019 / 2020+), market movement since open, public
   share, quarterback change since the window, head-coach change, offensive-
   line and pass-rush snap continuity (lagged player snaps, `players.py`),
   injury load, and whatever comes next. Each split names its source column
   and a leakage note.
2. Harness `nfl-ats signal-atlas --signal <id>` built on the leave-one-out
   convention in `tests/scratch/lanes/pbp08_early_season_split_20260913.py`
   (paired week-blocked bootstrap, seed fixed): one row per (signal, split,
   cell) with n, flips, delta of the whole card, 90% interval,
   probability_positive; recorded with `weak-signals record` under family
   `<signal>__<split>`. Never drop a cell for containing zero.
3. Run it for the nine served members first, then the active challengers,
   then the registry's pooled weak signals.
4. Findings page: one conditional profile per served signal, plain English,
   numbers read from the atlas artifact.
5. Gated paired challengers: `<signal>_gated_v1` fires only in cells with
   probability_positive above 0.5, registered against the served signal and
   graded at the opener. The challenger decides what is served, not the atlas.

## Open

- Whether to serve a gate (LEAD-65's week gate first) before a season of
  paired tracking (owner).
