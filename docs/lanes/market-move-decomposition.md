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

Run completed: `artifacts/market_move_decomposition/20260923T212214Z/metadata.json`
(1,503-game fit population, all six seasons 2020-2025 represented in each
LOSO fold).

**Parity check**: comparing the recomputed served construction (leader books,
Wednesday-cutoff window, via `merge_asof` on-the-record lines) against the
archived `leader_median_net` gives `max_abs_diff_points = 4.5` over 799
compared games — not the ~0 expected for a faithful reconstruction. The
`served_p` baseline used for every arm comparison is the true archived
artifact (read directly through `build_fit_population`, not the
reconstruction), so the served side of every comparison is correct; only the
four challenger constructions (a, b_early, b_late, c, d — all built from the
`merge_asof` lines) inherit whatever reconstruction error drives that 4.5
outlier. Likely cause (**inferred**, not run down): this script derives each
game's `commence_time_utc`/`week_first_commence_utc` anchor from the quotes
cache's own `commence_time_utc` column (min per game), not from the schedule
snapshot the production artifact build used, so a handful of games with a
corrected/rescheduled kickoff in the odds-API record get a different
Wednesday/cutoff boundary. Treat the four challenger arms' point estimates as
approximate until this is resolved; the directional pattern (below) is
unanimous across all five arms, which is some protection against one bad
game driving the conclusion, but not a substitute for fixing the anchor.

**Results (measured, `metadata.json`), n=1,503 per arm, LOSO by season
2020-2025, served accuracy 0.57152, served Brier 0.245388, served log loss
0.683952 in every row (fixed baseline)**:

| arm | accuracy | vs served (pp, 95% CI, P+) | decisive (arm-served) | Brier improvement (P+) | logloss improvement (P+) |
|---|---|---|---|---|---|
| a_all_books_median | 0.56687 | -0.466 [-2.13, +1.26], P+=0.293 | 84-91 of 175 | +0.000569 [-0.0015,+0.0026], P+=0.728 | +0.00134 [-0.0030,+0.0056], P+=0.757 |
| b_early_window (Tue noon-Thu) | 0.56620 | -0.532 [-2.10,+1.06], P+=0.242 | 77-85 of 162 | -0.000527, P+=0.268 | -0.00101, P+=0.293 |
| b_late_window (Thu-cutoff) | 0.55888 | -1.264 [-3.11,+0.72], P+=0.099 | 91-110 of 201 | -0.0000016, P+=0.506 | +0.000181, P+=0.544 |
| c_key_number_crossing_count | 0.55755 | -1.397 [-3.11,+0.40], P+=0.054 | 83-104 of 187 | -0.000505, P+=0.312 | -0.000918, P+=0.337 |
| d_move_across_3_and_7_separate | 0.55755 | -1.397 [-3.11,+0.40], P+=0.059 | 80-101 of 181 | -0.000652, P+=0.256 | -0.00135, P+=0.261 |

No arm's interval sits entirely on one side of zero on any metric, so all
five are `unresolved_below_power` per AGENTS.md (zero-crossing never closes a
signal). Every arm's accuracy point estimate is negative (a challenger
construction that underperforms the served raw-points/leader-books/Wednesday
window), and on decisive games (where the arm and served disagree) the
served side wins more often in all five arms — consistent directional signal
that the served construction is at least not dominated by any of these four
alternatives, though not powered to call any of them refuted. `a_all_books_median`
is the one arm with a Brier/logloss point estimate favoring the challenger
(better calibrated, still worse accuracy) — worth a second look with a larger
sample rather than promotion.

## Tried

Single run of `scripts/market_move_decomposition.py` (completed
2026-09-23T21:22:14Z, exit 0), no reruns, no parameter search after seeing
results (this lane predeclared the 4 constructions above before the script
executed). Results in `artifacts/market_move_decomposition/20260923T212214Z/`.

## Next

1. (Optional, recommended before recording) Fix the anchor discrepancy behind
   `parity_check.max_abs_diff_points = 4.5` in
   `artifacts/market_move_decomposition/20260923T212214Z/metadata.json` —
   derive `commence_time_utc`/`week_first_commence_utc` in
   `scripts/market_move_decomposition.py::_game_anchors` from the schedule
   snapshot (as `find_matching_opener_evaluation` / `load_snapshot` already
   do elsewhere) instead of the quotes cache's own `commence_time_utc`
   column, then rerun once and confirm `max_abs_diff_points` is ~0 before
   trusting the four challenger point estimates precisely (the directional
   read — all four underperform served on accuracy and decisive record — is
   unlikely to flip, but exact numbers may move).
2. `--help` on `nfl-ats weak-signals record` was checked this session; real
   flags are `--name --description --source --effect --effect-units
   --classification --interval-low --interval-high --probability-positive
   --sample-games --reliability --family --classification-evidence
   --plain-summary --category --notes` (no `--signal-id`, `--n-decisive`, or
   `--decisive-record` flags exist — put the decisive record in
   `--description`/`--notes`). `--reliability` (split-half) was not computed
   this session; AGENTS.md calls it decisive for later adjudication, so the
   root should add it or record `--notes` saying it is missing. Record each
   arm (numbers already filled from the completed run; repeat with
   `--effect-units brier_improvement --effect <brier estimate>` and
   `--effect-units log_loss_improvement --effect <logloss estimate>` as
   sibling cells in the same family if the owner wants those metrics
   recorded too, not just accuracy):

```
uv run --no-sync nfl-ats weak-signals record --league nfl \
  --family market_move_decomposition \
  --name market_move_decomposition_a_all_books_median \
  --category market --effect-units accuracy_points --effect -0.4657351962741184 \
  --interval-low -2.134899297413437 --interval-high 1.25936432175152 \
  --probability-positive 0.2925 --sample-games 1503 \
  --classification unresolved_below_power \
  --description "All-12-books median move (vs 3 leader books), same Wednesday-cutoff window, replacing the served move term in the four-term pick probability. Decisive games 84-91 (arm) vs 91-84 (served) of 175." \
  --plain-summary "Using every book's line move instead of just the three sharpest books did not beat the current model, LOSO 2020-2025." \
  --source artifacts/market_move_decomposition/20260923T212214Z/metadata.json

uv run --no-sync nfl-ats weak-signals record --league nfl \
  --family market_move_decomposition \
  --name market_move_decomposition_b_early_window_tue_noon_thu \
  --category market --effect-units accuracy_points --effect -0.5322687957418496 \
  --interval-low -2.099771999937959 --interval-high 1.064590818363273 \
  --probability-positive 0.2415 --sample-games 1503 \
  --classification unresolved_below_power \
  --description "Leader-books median move measured Tuesday noon to Thursday only (early window), replacing the served Wednesday-cutoff window. Decisive games 77-85 (arm) vs 85-77 (served) of 162." \
  --plain-summary "Looking only at the earliest-week line move instead of the full week did not beat the current model, LOSO 2020-2025." \
  --source artifacts/market_move_decomposition/20260923T212214Z/metadata.json

uv run --no-sync nfl-ats weak-signals record --league nfl \
  --family market_move_decomposition \
  --name market_move_decomposition_b_late_window_thu_cutoff \
  --category market --effect-units accuracy_points --effect -1.2641383898868928 \
  --interval-low -3.114692603188457 --interval-high 0.7204021832732596 \
  --probability-positive 0.099 --sample-games 1503 \
  --classification unresolved_below_power \
  --description "Leader-books median move measured Thursday to decision time only (late window), replacing the served Wednesday-cutoff window. Decisive games 91-110 (arm) vs 110-91 (served) of 201." \
  --plain-summary "Looking only at the late-week line move instead of the full week did not beat the current model, LOSO 2020-2025." \
  --source artifacts/market_move_decomposition/20260923T212214Z/metadata.json

uv run --no-sync nfl-ats weak-signals record --league nfl \
  --family market_move_decomposition \
  --name market_move_decomposition_c_key_number_crossing_count \
  --category market --effect-units accuracy_points --effect -1.3972055888223553 \
  --interval-low -3.112734313615445 --interval-high 0.39737076864258175 \
  --probability-positive 0.0535 --sample-games 1503 \
  --classification unresolved_below_power \
  --description "Signed count of key numbers (3/7/10/14/17) crossed between the Wednesday and cutoff line, leader books, replacing the raw-points served move term. Decisive games 83-104 (arm) vs 104-83 (served) of 187." \
  --plain-summary "Counting which key numbers the line crossed instead of the raw point move did not beat the current model, LOSO 2020-2025." \
  --source artifacts/market_move_decomposition/20260923T212214Z/metadata.json

uv run --no-sync nfl-ats weak-signals record --league nfl \
  --family market_move_decomposition \
  --name market_move_decomposition_d_move_across_3_and_7_separate \
  --category market --effect-units accuracy_points --effect -1.3972055888223553 \
  --interval-low -3.106513896987196 --interval-high 0.3997870586833602 \
  --probability-positive 0.059 --sample-games 1503 \
  --classification unresolved_below_power \
  --description "Two-term replacement of the move column: signed crossing of 3 and signed crossing of 7/10/14/17 entered separately in the same fitted logit (5-term fit). Decisive games 80-101 (arm) vs 101-80 (served) of 181." \
  --plain-summary "Splitting the line move into a crossed-3 signal and a crossed-7-or-more signal did not beat the current model, LOSO 2020-2025." \
  --source artifacts/market_move_decomposition/20260923T212214Z/metadata.json
```

   Do not mark any arm `wrong_sign_resolved` — no interval sits entirely on
   the adverse side for any arm on any metric (checked this session); all
   five stay `unresolved_below_power` per AGENTS.md.
3. Only after every arm is recorded, decide whether any construction is a
   serving candidate — that is a root/owner decision, not this lane's. Given
   every arm underperforms the served construction on accuracy and decisive
   record, the working read is that the served leader-median/Wednesday-cutoff
   construction is not obviously improved on by any of these four
   alternatives, though power is too low to call the alternatives refuted.

## Open

- **Parity gap (needs a decision before promotion, not before recording
  unresolved results)**: `max_abs_diff_points = 4.5` between the recomputed
  served construction and the archived `leader_median_net`, 799 games
  compared. Suspected cause and fix are in `Next` item 1. Recording the five
  `unresolved_below_power` cells now is still valid — the served side of
  every comparison used the true archived artifact, not the reconstruction —
  but do not treat the four challengers' exact point estimates as final until
  the anchor is fixed and the parity check reruns near 0.
- The as-of reconstruction assumes every archived quote satisfies
  `bookmaker_last_update_utc <= observed_at_utc`, matching the served
  eligibility filter; not independently re-derived beyond the parity check.
- Arm b splits the window at a fixed Thursday-00:00-ET boundary and arm c/d
  use the served Wednesday-cutoff window with symmetric key numbers
  (+-3/7/10/14/17); these are this session's operational definitions of the
  task's prose, not the only possible ones.
- No `weak-signals record` command has been run by this subagent (task scope:
  draft commands only, root runs them). `--reliability` (split-half) is
  unmeasured for all five arms; AGENTS.md marks it decisive for later
  adjudication.
