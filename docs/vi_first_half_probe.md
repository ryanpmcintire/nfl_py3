# First-half line source probe (LEAD-60) — GO

**Measured 2026-09-04, zero new fetches.** The question was whether the
2005–2016 VegasInsider backfill cache carries first-half lines, which
would unblock LEAD-02 (1H/full-game script disagreement).

## Coverage (measured, `Select-String` over the local cache)

- `data/raw/vegasinsider/20260822T033952Z/line_movement/` — **165/165
  files** contain `1st Half` sections, spanning capture-years 2006–2017
  (seasons 2005–2016, every year represented).
- `data/raw/vegasinsider/20260822T033952Z/snapshots/` — **176/189
  board files** contain `1st Half` sections.

## Shape (measured, one movement file read in full)

Each movement snapshot carries three column groups — full game, **1st
Half**, **2nd Half** — each with Fav/Dog spread cells and Over/Under
total cells plus prices (e.g. `IND-18.5 -110` under a 1H Fav cell).
So the cache holds 1H **and** 2H spread/total movement trails, not just
1H snapshots.

## What this unblocks

- LEAD-02 as designed (1H-implied vs full-game-implied script
  disagreement), plus a free 2H sibling nobody proposed: halftime
  adjustment traits (LEAD-27's 3Q mechanism has a 2H-line cousin).
- Join keys are the same (capture, matchup, book) as the completed
  backfill; the tidy schema needs four new nullable columns
  (`half1_spread_line`, `half1_total_line`, `half2_spread_line`,
  `half2_total_line`) or a companion half-line table. Parser work only.

## What was NOT done

No parser was written, no artifact built, no ATS screen run, no window
spent. The 13 snapshot files without a 1H section are unexamined (layout
variant or genuinely absent — the builder will count them).
