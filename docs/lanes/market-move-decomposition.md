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

## Unit 3 (this session, 2026-09-23): all-books median on the ACTIVE
Sunday-inclusive window

**Note on the stale text below:** the "mid-fix" narrative in the old State
section (and the pre-fix 20260923T212214Z numbers) is leftover prose from
before commit `6533e24`; registry/weak_signals.json confirms **Unit 1's five
arms ARE already recorded** (`unresolved_below_power`, all-books-median arm
effect +0.399 pts, `probability_positive` 0.8985 — matches the +0.40 [-0.13,
+1.00] P+0.90 figure this lane's brief already cites). Not rewriting that
section; treat the registry as ground truth over the prose below it.

**Scope**: the served fit's active pick probability
(`artifacts/active_pick_probability.json`) uses
`market_move_feature_version = leader_median_through_sunday_prekick_v1`
(`MARKET_MOVE_FEATURE_SUNDAY`), which Unit 1 did not test — Unit 1 compared
against the LEGACY (`leader_median_pre_sunday_v1`, `include_sunday=False`)
construction only. `_market_move_table`
(`pick_probability_fit.py:122-153`) does not compute the Sunday version live;
for `MARKET_MOVE_FEATURE_SUNDAY` it reads the frozen
`artifacts/sunday_market_probability/20260920_fixed/market_move.parquet`
verbatim (sha256-checked). That file was built by
`scripts/sunday_market_probability_eval.py` from
`artifacts/sharp_book_weighted_movement/spread_quotes.parquet`
(`decision_label == "intraday_hourly"`, seasons 2023-2025 only, 799 games),
via a bespoke `sunday_move()` window — LEADER_BOOKS only, Wednesday to
`min(kickoff, Sunday 12:45pm ET)` — used ONLY where the game was already
`market_move_available` under the legacy construction AND the legacy
reconstruction matched the population's stored legacy value within 1e-9
(`eligible`); otherwise the legacy value is carried through unchanged. This
hybrid, not a clean `sharp_book_movement_features(..., include_sunday=True)`
call, is what `build_fit_population(market_move_feature_version=
leader_median_through_sunday_prekick_v1)` actually returns today. (Calling
the real `sharp_book_movement_features(..., include_sunday=True)` on the
OTHER frozen cache from Unit 1, `artifacts/experiments/sharp_book_movement/
quotes.parquet`, would return all-zero moves for every game: that cache has
no `snapshot_timestamp_utc`, and `include_sunday=True` makes
`snapshot_timestamp_utc < cutoff_utc` a mandatory filter term — NaT compares
False everywhere, so eligible quotes drop to zero. This is a live
train/serve note, not this unit's concern.)

**Script**: `scripts/market_move_all_books_active.py` (new, imports shared
`_score_arm`/`_loso_predict`/`_week_blocked_bootstrap` from
`market_move_decomposition.py` as `mmd`). `ruff check` clean (no `--fix`
used after the initial import-order pass). Ran once in the foreground:
`.tools/uv.exe run --no-sync python scripts/market_move_all_books_active.py`
→ `artifacts/market_move_decomposition/20260923T214756Z/metadata.json`.

**Parity (measured)**: reconstructed the active version from raw quotes
(replicating `sunday_move()` plus the legacy-eligibility blend) and compared
against `build_fit_population(market_move_feature_version=
leader_median_through_sunday_prekick_v1)`'s `market_move_toward_home` column,
799 exposed games: **max_abs_diff_points = 0.0, mean_abs_diff_points = 0.0**
(exact). Reconstruction confirmed correct before grading, per the task's
requirement.

**Arm `e_all_books_median_active_window`** (12-book median, same
Wednesday-to-`min(kickoff, Sun 12:45pm ET)` window/eligibility as the active
version, `MOVE_AVAILABLE` = has-a-value under this window for any game, not
gated by legacy exposure) vs served (active, 3-leader-book), four-term fit,
LOSO 2020-2025, 1503 games, week-blocked bootstrap (2000 draws):
- Accuracy: arm 57.29% (861-642) vs served 57.42% (863-640); paired delta
  **-0.133 accuracy points, 95% CI [-0.59, +0.33], probability_positive =
  0.2215**.
- Decisive games (14 where arm and served disagree): arm 6-8, served 8-6.
- Brier: served lower (arm-minus-served comparison via `served-arm` = -0.00016
  [-0.00061, +0.00028], P+ 0.248, i.e. served favored).
- Log loss: served lower likewise (-0.00032 [-0.00125, +0.00059], P+ 0.2575).

**Implication for the served move**: on the window the model already serves
(Sunday-inclusive), switching from 3 leader books to all 12 books does NOT
help — direction is opposite Unit 1's legacy-window all-books result
(+0.40 pts, P+0.90 there vs -0.13 pts, P+0.22 here). The interval spans zero
both ways (not wrong-sign-resolved), so this closes nothing per AGENTS.md;
classification is `unresolved_below_power`. Read together with Unit 1: the
all-books-median edge Unit 1 found appears specific to the legacy
(Sunday-excluded) window, not a property of "more books" in general — worth
naming as a caveat if Unit 1's finding is ever proposed for serving.

**Draft record command (root runs)**:
```
nfl-ats weak-signals record \
  --family market_move_decomposition \
  --name e_all_books_median_active_window \
  --description "All 12 tracked books' median spread-line move over the ACTIVE Sunday-inclusive window (leader_median_through_sunday_prekick_v1: Wed-open through min(kickoff, Sun 12:45pm ET), same window/eligibility build_fit_population uses for the served active version) vs the served 3-leader-book median, four-term fit LOSO 2020-2025, 1503 games. Reconstruction parity vs build_fit_population(market_move_feature_version=leader_median_through_sunday_prekick_v1): max/mean abs diff 0.0 over 799 exposed games. Decisive record: arm 6-8 vs served 8-6 on 14 decisive games." \
  --source artifacts/market_move_decomposition/20260923T214756Z/metadata.json \
  --effect -0.13 \
  --effect-units accuracy_points \
  --interval-low -0.59 \
  --interval-high 0.33 \
  --probability-positive 0.2215 \
  --sample-games 1503 \
  --classification unresolved_below_power \
  --classification-evidence "interval crosses zero [-0.59,+0.33]; not wrong-sign-resolved; no positive control run at this effect size" \
  --category market \
  --plain-summary "Using all 12 tracked sportsbooks instead of just the 3 leader books, on the same Sunday-inclusive window the active model already serves, did not beat the leader-book version - the estimate is slightly negative and the interval straddles zero." \
  --notes "Split-half reliability not measured (same gap as Unit 1's five arms). Brier improvement -0.00016 [-0.00061,+0.00028] P+ 0.248; log-loss improvement -0.00032 [-0.00125,+0.00059] P+ 0.2575, both favoring served. Direction opposite Unit 1's legacy-window all-books result (+0.40 pts, P+0.90) - the all-books edge looks specific to the Sunday-excluded window."
```

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

1. **[SUPERSEDED — done]** Items 1-5 below this line (ruff cleanup, rerun,
   pull numbers, record 5 arms) describe Unit 1 and are already complete:
   `registry/weak_signals.json` holds all 5 `market_move_decomposition` arms
   (`unresolved_below_power`, all-books-median effect +0.399 pts P+ 0.8985,
   matching this lane's own Goal/brief figures), confirmed by this session
   via direct registry read. Ignore items 2-5's instructions; they are kept
   only as a record of what Unit 1 did. Do not rerun Unit 1's arms.
2. **Root: run the Unit 3 record command** printed in the "Unit 3" section
   above (`e_all_books_median_active_window`, effect -0.13 accuracy points
   [-0.59, +0.33], P+ 0.2215, source
   `artifacts/market_move_decomposition/20260923T214756Z/metadata.json`).
   That is the only outstanding `weak-signals record` call in this lane.
3. Optional hygiene, non-blocking: the old "State"/"Tried" prose below (the
   "mid-fix... NOT DONE" text, pre-fix 20260923T212214Z run) is stale —
   superseded by the Unit 3 section's note and by the registry. Could be
   trimmed in a future pass; not required before closing this lane.
4. If a future unit wants to reconcile the train/serve mismatch noted in
   Unit 3 (the active feature version's frozen training data uses a
   Wed-to-Sun-12:45pm-ET window with a legacy-exposure fallback, while live
   serving via `market_move_toward_home()` calls
   `sharp_book_movement_features(..., include_sunday=True)` with cutoff =
   kickoff and a `snapshot_timestamp_utc` requirement) — flag it as a new,
   separate predeclared look; it is a distinct question from Unit 3's
   all-books-vs-leader-books comparison.

## Open

- Unit 3's `e_all_books_median_active_window` arm has not been recorded yet
  (command is in Next item 2; root runs it). Unit 1's five arms ARE
  recorded — see Next item 1.
  `--reliability` (split-half) is unmeasured for all five Unit-1 arms AND
  for the Unit 3 arm; AGENTS.md
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
