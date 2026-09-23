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
  (`_wednesday = sunday - 4d`) and `cutoff_utc = min(kickoff, Sunday 16:00 ET)`,
  keeping only quotes with `bookmaker_last_update_utc <= observed_at_utc`
  (sanity), `observed_at_utc >= Monday`, AND **`observed_at_utc < Sunday 00:00
  ET`** (`include_sunday=False`; this is stronger than "excludes Sunday
  afternoon" — it excludes every Sunday-day quote, so the true eligible window
  for an ordinary Sunday game ends at Sunday midnight, not at kickoff/4pm).
  `leader_median_net` = **median** (not leadership-weighted) of summed
  per-book moves across `LEADER_BOOKS = (bovada, williamhill_us, mybookieag)`
  (line 24). This is `MOVE_COLUMN = "market_move_toward_home"` in the fit,
  entered raw (fillna 0) alongside `MOVE_AVAILABLE_COLUMN` in `FIT_FEATURES =
  (model_logit, composition_flag_sum, market_move_toward_home,
  market_move_available)` (`pick_probability_fit.py:46`).
- **The archived reference `artifacts/sharp_weighted_follow/20260909T233611Z/per_game.parquet`
  (816 games, 2023-2025 only) was built by a since-removed
  `scripts/sharp_weighted_follow.py`** (recovered from
  `.claude/worktrees/agent-*/scripts/sharp_weighted_follow.py`, all identical
  copies) from the **frozen** cache
  `artifacts/experiments/sharp_book_movement/{quotes.parquet,kickoff.parquet}`
  (sha256 recorded in `docs/sharp_weighted_follow.md`), calling
  `nfl_ats.sharp_book_movement_features.sharp_book_movement_features(quotes,
  games)` **verbatim**, where `games = kickoff.rename(columns={"nflverse_game_id":
  "game_id"})` — i.e. `commence_time_utc`/`week_first_commence_utc` come
  straight from the frozen `kickoff.parquet` columns, not derived from the
  quotes cache. Games outside 2023-2025 legitimately have
  `MOVE_AVAILABLE_COLUMN = 0` in the served fit (left-join + fillna 0 in
  `build_fit_population`); this is expected, not a bug.

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

## State (mid-fix — script edited but NOT rerun; do not trust the 20260923T212214Z
numbers below, they are the pre-fix run kept only for reference)

**Root cause of the 4.5-point parity gap, found this session (measured, read
code — not yet reverified by a rerun):**

1. `scripts/market_move_decomposition.py::_game_anchors` derived
   `commence_time_utc` by grouping the WRONG, much larger quotes cache
   (`artifacts/sharp_book_weighted_movement/spread_quotes.parquet`, all
   historical_backfill captures, many seasons) instead of reading the frozen
   `artifacts/experiments/sharp_book_movement/kickoff.parquet` (816 games,
   2023-2025) that the real reference builder used verbatim. **FIXED**: now
   reads `KICKOFF_CACHE` directly and uses its `commence_time_utc`/
   `week_first_commence_utc` columns as-is.
2. `_eligible_quotes` was missing the Sunday exclusion
   (`observed_at_utc < _sunday`, Sunday 00:00 ET) that
   `sharp_book_movement_features` always applies when `include_sunday=False`
   (the default and the served setting) — see
   `src/nfl_ats/sharp_book_movement_features.py:117-132`. The script only
   bounded quotes by `cutoff_utc` (kickoff or Sunday 16:00 ET), so it wrongly
   included Sunday-morning quote updates for any game with a later cutoff.
   **FIXED**: `_eligible_quotes` now merges in `_sunday` and adds
   `q.observed_at_utc.lt(q._sunday)` to the filter.
3. `QUOTES_CACHE` switched from
   `artifacts/sharp_book_weighted_movement/spread_quotes.parquet` to the
   frozen `artifacts/experiments/sharp_book_movement/quotes.parquet` (the
   same file the archived reference was built from).
4. Added a second, independent parity check in `main()`: calls the real
   `nfl_ats.sharp_book_movement_features.sharp_book_movement_features(quotes,
   games)` verbatim (already imported) and compares ITS `leader_median_net_move`
   against the archived `leader_median_net` too, written to metadata as
   `true_function_parity_check` alongside the existing `parity_check` (the
   script's own `merge_asof`-based reconstruction, needed because the
   challenger arms use custom windows/books the production function doesn't
   expose). `parity_check` also now reports `mean_abs_diff_points` and
   `n_games_off_gt_0_25_points` (task ask).

**Verification status: NOT DONE.** The script has not been executed since
these edits. `ruff check` was ALSO not fully clean as of the last successful
edit — 2 findings likely remain (see Next item 1). The pre-fix run
(`artifacts/market_move_decomposition/20260923T212214Z/metadata.json`,
`max_abs_diff_points=4.5` over 799 games) is superseded and should not be
quoted as current; leaving its old results table out of this lane version
deliberately so nobody records stale numbers.

## Tried

- Read `src/nfl_ats/sharp_book_movement_features.py` (43-167, 188-262),
  `src/nfl_ats/pick_probability_fit.py` (122-283), `scripts/sharp_book_weighted_movement.py`
  (full), and the archived `scripts/sharp_weighted_follow.py` recovered from
  a `.claude/worktrees/agent-*/` copy (not in the live tree) to trace exactly
  what built the archived reference file. Confirmed frozen source files still
  exist: `artifacts/experiments/sharp_book_movement/{quotes.parquet
  (2,387,404 rows, 17 bookmaker_keys, superset of LEADERSHIP_WEIGHTS),
  kickoff.parquet (816 rows, columns nflverse_game_id/commence_time_utc/
  season/week/week_first_commence_utc)}`.
- Edited `scripts/market_move_decomposition.py`: `_game_anchors` signature
  changed from `(quotes, population)` to `(kickoff, population)`;
  `_eligible_quotes` Sunday-exclusion added; `QUOTES_CACHE`/new
  `KICKOFF_CACHE` constants; `main()` now reads `kickoff.parquet` and calls
  the real `sharp_book_movement_features` for a second cross-check.
- Ran `ruff check --fix` (safe fixes applied: import sort, `dict()` rewrite,
  unnecessary `int()` removal) plus manual line-wrap edits for remaining
  E501s. Confirmed via `git show HEAD:scripts/market_move_decomposition.py |
  ruff check -` that 13 of the original errors predate this session (not
  introduced by this fix). Last edit (wrapping `true_function_parity_check`'s
  long lines) was interrupted by the 50-tool-call subagent cap before it
  applied.

## Next

1. **Finish ruff cleanup first.** Run
   `uv run --no-sync ruff check scripts/market_move_decomposition.py` (or
   `.tools/uv.exe run --no-sync ruff check scripts/market_move_decomposition.py`
   on Windows). As of this handoff it likely still flags: the
   `"true_function_parity_check": {...}` dict (3 long lines: n_games_compared/
   max_abs_diff_points/mean_abs_diff_points/note — wrap the same way the
   `parity_check` dict just above it was already wrapped, ternaries in
   parens, note string split into a tuple of literals) and the
   `(out_dir / "metadata.json").write_text(json.dumps(payload, indent=2,
   default=str), encoding="utf-8")` line (~104 chars, split the call across
   lines). Iterate `ruff check --fix` + manual wraps to 0 errors. No `#`
   comments anywhere (repo-wide rule).
2. **Run the script once**: `.tools/uv.exe run --no-sync python
   scripts/market_move_decomposition.py` (Windows) from `F:\Repos\nfl_py3`.
   Read the new `artifacts/market_move_decomposition/<run_id>/metadata.json`.
   Check `parity_check.max_abs_diff_points`, `.mean_abs_diff_points`,
   `.n_games_off_gt_0_25_points`, and `true_function_parity_check.max_abs_diff_points`
   (should be exactly 0 or within float noise, since it calls the production
   function verbatim on the frozen cache — if this one alone is not ~0, the
   archived `sharp_weighted_follow/20260909T233611Z/per_game.parquet` was
   built from different source files than the currently-committed frozen
   cache; compare against the sha256 values recorded in
   `docs/sharp_weighted_follow.md` before concluding anything).
3. **If both parity checks are ~0** (expected): pull the 5 arms' `arm_accuracy`,
   `served_accuracy`, `arm_brier`/`served_brier`, `arm_logloss`/`served_logloss`,
   `n_decisive`, `arm_decisive_record`/`served_decisive_record`, and the three
   `*_bootstrap` blocks (`estimate`/`lower`/`upper`/`probability_positive`)
   from the new metadata.json, and REPLACE the results table and all five
   `nfl-ats weak-signals record` commands below with the new numbers (same
   format as the deleted pre-fix table: effect units `accuracy_points` in
   points not fractions, `--category market`, `--plain-summary`, decisive
   record in `--description`; real flags per last session's `--help` check:
   `--name --description --source --effect --effect-units --classification
   --interval-low --interval-high --probability-positive --sample-games
   --reliability --family --classification-evidence --plain-summary
   --category --notes` — no `--sample-blocks`, `--signal-id`,
   `--n-decisive`, or `--decisive-record` flags exist). Classification stays
   `unresolved_below_power` unless an interval now sits entirely on one side
   of zero (check freshly — do not assume the old verdict carries over
   unchanged). Do not rerun the arms a second time after seeing numbers
   (predeclared, single-run discipline, matches this family's existing
   convention).
4. **If parity is not ~0 after the fix**: name exactly which of the two
   checks failed and why (see Next item 2's diagnostic branch), and mark all
   five arms not recordable — do not record approximate numbers, per the
   original task's fallback instruction.
5. Only after every arm is (re-)recorded, note for the root/owner: given
   every arm underperformed the served construction on accuracy and decisive
   record in the pre-fix run, the working expectation is the fix moves point
   estimates but is unlikely to flip the direction — confirm this expectation
   against the new numbers rather than assuming it.

## Open

- No `weak-signals record` command has been run yet by any subagent in this
  lane (task scope: draft/verify commands, root runs them).
  `--reliability` (split-half) is unmeasured for all five arms; AGENTS.md
  marks it decisive for later adjudication — the root should add it or record
  `--notes` saying it is missing.
- The diff-then-sum-per-quote-update method production actually uses (see
  `sharp_book_movement_features` lines 133-146: `home_spread_line.diff()`
  summed over qualifying updates) is not bit-identical to this script's
  `_asof_lines`/`_median_points` "value at boundary minus value at boundary"
  approach at one edge case: a quote landing at the exact boundary timestamp
  to the second. Expected negligible with real API timestamps; the new
  `true_function_parity_check` (calling the real function directly) is the
  way to confirm this doesn't matter in practice — if it's ~0 while the
  script's own `parity_check` is not, that edge case (or something like it)
  is the residual, not the anchor/Sunday bugs.
- Arm b splits the window at a fixed Thursday-00:00-ET boundary and arm c/d
  use the served Wednesday-cutoff window with symmetric key numbers
  (+-3/7/10/14/17); these are this session's operational definitions of the
  task's prose, not the only possible ones.
- The frozen `artifacts/experiments/sharp_book_movement/` cache only covers
  2023-2025 (816 games); the fit population spans 2020-2025 (1,503 games).
  Games outside 2023-2025 will have `MOVE_AVAILABLE_COLUMN = 0` for every arm,
  same as the served fit — this is correct/expected, not a gap to fix.
