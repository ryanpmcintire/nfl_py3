# Week 1 covers settled (2026-09-15)

## Goal

Grade 2026 Week 1 against the final scores and make every reader surface show
the card that was actually played. Done when the board shows all sixteen
covers and the record strip reads the served card's record, not the Tuesday
ledger's.

## State

- Fresh nflverse snapshot `data/raw/20260915T102254Z`; `nfl-ats settle --season
  2026 --week 1 --write-graded` wrote `artifacts/settlement/game_results.parquet`
  and `graded_decisions.parquet` (1,056 graded rows, 79 arms, 0 pending).
- **The played card went 9-7** (measured, matches the owner's own count).
  Losses: ARI at LAC, BAL at IND, BUF at HOU, DAL at NYG, MIA at LV (the Best
  Pick), WAS at PHI, DEN at KC.
- **The card's record is 9-7 on every surface.** Three places had been grading
  the Tuesday lock instead of the sides actually served, so the site showed
  10-6 beside rows that graded 9-6:
  - the board's season-record strip (`This week` / `Season to date`),
  - the prospective scoreboard on the board and the model page,
  - the history page's "The card's picks" and every challenger's
    "against the card" delta.
  `served_paper_decisions` in `clv.py` loads the paper ledger and overlays the
  latest pick-revision side; `board_content.py` and `board_site_content.py`
  now read the card through it. "so far" drops once no game in the week is
  pending. Strip reads **This week: 9-7 / Season to date: 9-7 / Best Pick:
  0-1**; history reads **The card's picks 9-7, 56% right**; model page's
  played-card row reads **2026 so far: 9-7**.
- The prospective scoreboard now compares like with like -- "the card as
  played 9-7 vs. the former rule chain 8-8" -- instead of pairing two Tuesday
  arms under a label a reader would take for the played card.
- Every remaining `10-6` on the site belongs to a named challenger rule that
  made those picks (rules that never fired in Week 1 hold the Tuesday sides,
  so 10-6 is genuinely theirs) or to a finished week of an earlier season.
- `totals.load_population` / `load_population_wave2` now join against the
  schedules snapshot their feature table was built from
  (`feature_source_schedules_path`, reading the feature manifest) instead of
  whatever raw snapshot is newest. Ingesting Week 1 finals had put the newest
  snapshot 99 total-lines ahead of `game_features.parquet`, which is only
  realigned by the Tuesday rebuild.
- `nfl-ats card-ledger-check`: no disagreements. `publish-board` and
  `publish-predictions` re-run; card sides unchanged.

## Tried

- `nfl-ats build-pbp-features` first, to realign the wave-2 table: no effect,
  because it inherits `game_features.parquet`'s pinned snapshot
  (`20260908T162105Z`). Rebuilding the canonical table instead would stale the
  waterfall feed's `feature_table_sha256` and fail-close `publish-board`, so
  the snapshot resolver was the right layer.

## Next

- Tuesday 2026-09-16: `weekly-run --record-decisions` for Week 2; that rebuild
  moves both feature tables onto a current snapshot.

## Open

- The late-week refresh went 1-5 on its six Week 1 flips (`pick_revisions`,
  before_refresh 5-1). One week is one week, but it is the first live evidence
  on rules the owner has already turned most of off.
