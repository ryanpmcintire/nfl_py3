# market-move-decomposition

## Goal

Decompose the served market-move term in the four-term pick probability
(`src/nfl_ats/pick_probability_fit.py`) to find what construction carries the
edge: the served fit scores 57.15% held out on 2020-2025 with the move term,
versus ~53% on 2011-2019 without move data (**reported**, from the task
brief). Test alternative move constructions against the served one, paired,
before any serving change.

## Served move construction (measured, read)

- `_market_move_table` (`src/nfl_ats/pick_probability_fit.py:122-153`) reads
  `MARKET_MOVE_ARTIFACT_ROOT = "sharp_weighted_follow"` (line 51), newest
  `per_game.parquet`, column `MARKET_MOVE_COLUMN = "leader_median_net"`
  (line 52), for feature version `MARKET_MOVE_FEATURE_LEGACY =
  "leader_median_pre_sunday_v1"` (`pick_probability.py:45`).
- `sharp_book_movement_features` (`src/nfl_ats/sharp_book_movement_features.py:43-167`)
  builds `leader_median_net_move`: for each game, per book net home-spread-line
  move summed from eligible quotes between the Wednesday of the game's week
  (`_wednesday = sunday - 4d`) and `cutoff_utc = min(kickoff, Sunday 16:00 ET)`
  (excludes Sunday afternoon moves by default, `include_sunday=False`), keeping
  only quotes with `bookmaker_last_update_utc <= observed_at_utc` (sanity) and
  `observed_at_utc >= Monday` of that week. `leader_median_net` = **median**
  (not leadership-weighted) of summed per-book moves across
  `LEADER_BOOKS = (bovada, williamhill_us, mybookieag)` (line 24). This is
  `MOVE_COLUMN = "market_move_toward_home"` in the fit, entered raw (fillna 0)
  alongside `MOVE_AVAILABLE_COLUMN` in `FIT_FEATURES = (model_logit,
  composition_flag_sum, market_move_toward_home, market_move_available)`
  (`pick_probability_fit.py:46`).
- Local archive: `artifacts/sharp_book_weighted_movement/spread_quotes.parquet`
  (12 books, `historical_backfill` captures, ~3.8M rows, columns include
  `observed_at_utc`, `bookmaker_last_update_utc`, `commence_time_utc`).
  Reference served per-game table:
  `artifacts/sharp_weighted_follow/20260909T233611Z/per_game.parquet`
  (816 games, has `leader_median_net`).

## Predeclared alternative constructions (before fitting)

1. **a_all_books_median** — same Wednesday-to-cutoff window, median move
   across all 12 archived books (`LEADERSHIP_WEIGHTS` keys) instead of the 3
   leader books only.
2. **b_early_window_tue_noon_thu** / **b_late_window_thu_cutoff** — same
   3 leader books, split the window at Thursday 00:00 ET: early = Tuesday
   12:00 ET (pool freeze) to Thursday; late = Thursday to `cutoff_utc`. Two
   looks under one family, each independently replacing the served window.
3. **c_key_number_crossing_count** — same window/books as served, but the
   move value is the signed count of key numbers (3, 7, 10, 14, 17, and
   negatives) strictly crossed between the line at Wednesday and the line at
   cutoff, median across leader books, instead of raw points.
4. **d_move_across_3_and_7_separate** — two-term replacement of the single
   move column: `move_across_3` (signed 0/1 crossing of +-3) and
   `move_across_7` (signed 0/1 crossing of any of +-7/10/14/17), both entered
   into the same fitted logit (5 terms total, still one calibrated
   probability, no side-only flip).

Total looks this family: 5 arms (a, b_early, b_late, c, d) x 3 metrics
(accuracy, Brier, log loss) + decisive-game record, each vs the served arm,
LOSO by season 2020-2025, week-blocked bootstrap (2000 draws,
`probability_positive`). Family name for `weak-signals`:
`market_move_decomposition`.

## State

Script `scripts/market_move_decomposition.py` written and run once
(single run, no re-fitting after seeing numbers). It: (1) rebuilds
Wednesday-cutoff leader-median points from `spread_quotes.parquet` via
as-of book lines at Tuesday-noon/Wednesday/Thursday/cutoff boundaries
(`pd.merge_asof` per game/book) and checks parity against the archived
`leader_median_net` column; (2) builds each alternative move series; (3) for
each arm, swaps only the move column(s) into a copy of
`build_fit_population(...)`'s output (legacy/served feature version) and
reuses `_design/_fit_logit/_standardisers/_predict/_natural_coefficients`
from `pick_probability_fit.py` unmodified, monkeypatching only the
module-level `FIT_FEATURES` tuple at call time (5-term for arm d) — no
`src/` edits; (4) LOSO over `OUTER_SEASONS = 2020..2025`, trained on all other
seasons in the same population (matches the pattern in
`scripts/base_model_recency.py::_loso_predict`); (5) reports accuracy with
plain and decisive record (arm pick != served pick), Brier, log loss, and a
week-blocked bootstrap improvement (2000 draws, seed 20260923) for each vs the
served four-term fit's own LOSO predictions on the identical population/folds.
Artifacts under `artifacts/market_move_decomposition/<run_id>/` (metadata.json
with parity check, provenance, per-arm fold coefficients; asof_lines.parquet).

Run launched in background from this session (`uv run --no-sync python
scripts/market_move_decomposition.py`) but did not confirm completion before
this subagent's turn cap; the background process does not survive past this
session. The script is idempotent and self-contained (single run, no manual
steps) — a fresh session/agent should simply (re)run it in the foreground and
read the newest `artifacts/market_move_decomposition/<run_id>/metadata.json`.
Expect a few minutes: 5 arms x 3 metrics x week-blocked bootstrap (2000 draws
each) plus one `_asof_lines` reconstruction over the ~3.8M-row quote cache.

## Tried

Single run of `scripts/market_move_decomposition.py`, no reruns, no
parameter search after seeing results (this lane predeclares the 4
constructions above before the script executed).

## Next

0. Run `/f/Repos/nfl_py3/.tools/uv.exe run --no-sync python
   scripts/market_move_decomposition.py` (foreground; a previous background
   attempt this session did not confirm completion before hand-off). It reads
   only existing artifacts (`build_fit_population`, the opener evaluation,
   `spread_quotes.parquet`, `sharp_weighted_follow/.../per_game.parquet`) and
   writes to `artifacts/market_move_decomposition/<new run_id>/`; no `src/`
   edits, no network calls.
1. Read `artifacts/market_move_decomposition/<newest run_id>/metadata.json`
   for the parity check (`parity_check.max_abs_diff_points` should be ~0
   against the archived `leader_median_net`; a nonzero value means the
   Wednesday/cutoff reconstruction diverges from the served artifact and the
   arm numbers need re-checking before trusting them) and the 5 `arms`
   entries (accuracy/brier/logloss vs served, decisive record, bootstrap
   `probability_positive`).
2. For every arm, record with (fill `<...>` from metadata.json; use
   `probability_positive` from `accuracy_points_bootstrap` for the effect
   estimate unless the owner wants Brier/logloss as the primary metric
   instead — record whichever is the decision-relevant one, or all three):

```
uv run --no-sync nfl-ats weak-signals record \
  --family market_move_decomposition \
  --signal-id market_move_decomposition_<arm_name> \
  --category market \
  --effect-units accuracy_points \
  --estimate <accuracy_points_bootstrap.estimate> \
  --interval-low <accuracy_points_bootstrap.lower> \
  --interval-high <accuracy_points_bootstrap.upper> \
  --probability-positive <accuracy_points_bootstrap.probability_positive> \
  --n-games <n_games> \
  --n-decisive <n_decisive> \
  --decisive-record "<arm_decisive_record> vs served <served_decisive_record>" \
  --classification unresolved_below_power \
  --plain-summary "<arm_name> move construction vs the served leader-median move term in the four-term pick probability, LOSO 2020-2025." \
  --source artifacts/market_move_decomposition/<run_id>/metadata.json
```

   (Adjust the exact flag names to the current `weak-signals record` CLI
   before running; check `nfl-ats weak-signals record --help` first — this
   lane does not re-verify the CLI surface.) Do not mark any arm
   `wrong_sign_resolved` unless the whole bootstrap interval sits on the wrong
   side of zero for that arm's primary metric; otherwise every arm keeps
   `unresolved_below_power` per AGENTS.md.
3. Only after every arm is recorded, decide whether any construction is a
   serving candidate — that is a root/owner decision, not this lane's.

## Open

- The as-of reconstruction assumes every archived quote satisfies
  `bookmaker_last_update_utc <= observed_at_utc`, matching the served
  eligibility filter; not independently re-derived beyond the parity check.
- Arm b splits the window at a fixed Thursday-00:00-ET boundary and arm c/d
  use the served Wednesday-cutoff window with symmetric key numbers
  (+-3/7/10/14/17); these are this session's operational definitions of the
  task's prose, not the only possible ones.
- No `weak-signals record` command has been run by this subagent (task scope:
  draft commands only, root runs them).
