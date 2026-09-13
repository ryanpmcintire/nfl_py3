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

## Tried

- Nothing beyond LEAD-65 yet.

## Next

1. Split library, predeclared and versioned in `registry/split_library.json`
   before any cell is scored: time in season (weeks 1-4 / 5-12 / 13-18),
   spread size band and side (favourite / dog), home / away, division game,
   rest and travel (short week, bye, long trip), weather and roof, primetime,
   era (2009-2013 / 2014-2019 / 2020+), market movement since open, public
   share, quarterback change since the window, head-coach change, offensive-
   line and pass-rush snap continuity (lagged player snaps, `players.py`),
   injury load. Each split names its source column and a leakage note.
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
