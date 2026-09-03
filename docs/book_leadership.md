# Book leadership: who moves first (descriptive measurement)

**Status:** design frozen 2026-09-03 before the leadership table was
computed (numbers appended in Results the same session from the frozen
script). This is a DESCRIPTIVE measurement, not an experiment: no
candidate, no baseline, no ATS outcome, no registry verdict, no window.

**Lane:** SKY-04 (market microstructure: quote arrivals, book leadership,
latency, information diffusion). Files: this document,
`scripts/book_leadership.py`, `tests/test_book_leadership.py`,
`artifacts/book_leadership/`.

## Frozen design

- **Population:** `spreads` market rows from `data/market/raw/*/quotes.parquet`,
  seasons 2023–2025, REG season games matched to an `nflverse_game_id`.
- **Move event:** for one game, consecutive snapshots by `observed_at_utc`
  where at least one book's `home_spread_line` differs from its own value in
  the previous snapshot. The opening snapshot never counts (no prior).
- **First-mover credit:** among books that changed in a move event, the book(s)
  with the earliest `bookmaker_last_update_utc` split one credit evenly.
  A book that did not change earns nothing for that event, even if its
  timestamp is old.
- **Staleness profile:** per book, median `observed_at_utc minus
  bookmaker_last_update_utc` over all its rows (how stale the feed reads at
  capture time).
- **Outputs:** per-book move participations, first-move credits, leadership
  share (credits / participations), staleness median; plus the totals that
  let a reader re-derive them.
- **Out of scope (disclosed):** prices are ignored (leadership is scored on
  line moves only); alternate lines do not exist in the game-spreads market;
  provider timestamps are trusted as given (no independent clock audit);
  nothing here says which book is "sharp" — first is not right.

## Results (measured 2026-09-03)

Run by `scripts/book_leadership.py` in one pass (artifact
`artifacts/book_leadership/20260903T201202Z/results.json`): 855 games,
7,424 snapshots, 27,548 move events, 2023–2025 spreads.

| Book | Participations | First-move credits | Leadership share | Median staleness (s) |
|---|---|---|---|---|
| bovada | 8,222 | 5,044.3 | 0.6135 | 48 |
| williamhill_us | 5,973 | 3,825.6 | 0.6405 | 100 |
| mybookieag | 5,795 | 3,636.7 | 0.6276 | 73 |
| draftkings | 6,164 | 2,759.9 | 0.4477 | 46 |
| betus | 4,647 | 1,875.2 | 0.4035 | 46 |
| betrivers | 5,250 | 1,870.8 | 0.3563 | 44 |
| pointsbetus | 2,752 | 1,485.2 | 0.5397 | 65 |
| fanatics | 2,465 | 1,372.8 | 0.5569 | 36 |
| lowvig | 4,248 | 1,225.6 | 0.2885 | 44 |
| fanduel | 3,083 | 938.8 | 0.3045 | 35 |
| betmgm | 3,062 | 926.5 | 0.3026 | 48 |
| betonlineag | 3,964 | 872.0 | 0.2200 | 44 |

Smaller books trail below (unibet_us 0.22, superbook 0.24, twinspires 0.17;
wynnbet 0.44 and barstool 0.38 on thin participation).

What this implies, before what is wrong with it: line moves are not
synchronized — offshore-leaning books (Bovada, WH, MyBookie) post the new
number first roughly three-fifths of the times they move, while the big
US retail books (DK, FD, MGM, Caesars-side) mostly follow at shares
0.30–0.45. If the late-week refresh ever weights captures by source, this
table is the empirical input. What is wrong with it: (1) provider clocks
are trusted as given — a book whose feed timestamps lag will look like a
leader and one that stamps aggressively will look slow, so shares confound
clock skew with speed; (2) capture cadence is coarse (scheduled snapshots,
not a wire feed), so within-window sequencing is only as fine as provider
stamps allow; (3) first is not right — nothing here scores whether the
early move pointed the correct way. Descriptive only: no registry entry
(no candidate, no baseline), no window, no model change.
