# SIM-04 held-out grade, what-if runs and dashboard

## Goal

Finish SIM-04 after the engine unit (`scripts/sim04_engine.py`, lane
`sim04-play-simulator.md`). Steps: add pregame team conditioning; grade
leave-one-season-out at the opener against the served discrete read and record
the result in the registry; run SIM-05 what-if runs; add a simulated-margin
chart to the dashboard game page, then publish and push.

## State

- CLOSED 2026-09-26. Final results are in `docs/sim04_unit_log.md` ("Unit LOSO grade result", "SIM-05 and dashboard") and ROADMAP SIM-04/SIM-05. Anything below is history. (2026-09-25, this subagent, STOPPED AT 50-TOOL-CALL CAP MID-EDIT)

Team conditioning IS implemented and IS verified correct in
`scripts/sim04_engine.py`. `scripts/sim04_loso.py` has the team-conditioned
grading wired in, EXCEPT the last edit (fixing the two
`record_command_draft_*` strings to the real `weak-signals record` CLI flags)
was rejected by the tool-call-cap hook before it applied. **The file on disk
still has the OLD, WRONG draft-command block** (uses nonexistent
`--signal`/`--artifact` flags copied from a stale draft). See "Next step 0"
below for the exact fix, already fully composed.

### Engine changes (`scripts/sim04_engine.py`, done, verified)

- `load_team_ratings()` (lru_cache): reads `game_features_pbp.parquet` REG
  rows, `game_id` + `home_/away_off_epa_per_play` + `home_/away_def_epa_per_play`.
- `build_transition_frame(pbp, team_ratings=None)`: when given, merges those 4
  columns by `game_id`, computes `is_home_off = posteam==home_team`,
  `off_row` (home col if is_home_off else away col), `def_row` (away col if
  is_home_off else home col — the row's own defense is the OTHER team), fills
  NaN (early 2009) with the training window's own nanmean. Adds
  `off_row`/`def_row`/`is_home_off` columns to `out` always (NaN/-1 sentinel
  when `team_ratings is None`, so the unconditioned path is untouched).
- `build_tables(seasons, condition_on_team=False)`: when True, loads ratings,
  computes `team_kernel_h = 0.5 * nanstd(training-window-only off EPA spread)`,
  `league_off_mean`/`league_def_mean` (also training-window-only, no leakage),
  and `nn_trees_cond` (see below). Stores `off_row`/`def_row`/`is_home_off`
  arrays and `nn_cache_cond: {}` regardless (unused when unconditioned).
- `build_neighbor_index_scipy` / `pick_index_nn_conditioned`: a SEPARATE
  per-(down,phase) `scipy.spatial.cKDTree` index (`nn_trees_cond`), queried at
  k_state=200 (`K_STATE`), cached per rounded state key in `nn_cache_cond`
  (same `round_state_key`, same rounding constants as the unconditioned path —
  untouched). Kernel weight
  `w = exp(-((off_row-off_sim)^2+(def_row-def_sim)^2)/(2h^2)) *
  (1.5 if is_home_row==is_home_sim else 1)` (`TEAM_KERNEL_LAMBDA=1.5`,
  predeclared, not tuned). Resample via manual CDF (`cumsum` + `searchsorted`
  on `rng.random()*sum`), NOT `rng.choice(p=...)` — the latter plus sklearn's
  `KDTree.query` validation overhead (`check_array`/narwhals) was measured
  the dominant cost (cProfile: ~8s/11.8s in one 300-game conditioned run);
  switching the CONDITIONED-only tree to `scipy.spatial.cKDTree` and the
  resample to manual CDF raised throughput from ~48 to ~110 games/sec at
  n=20000 warm-cache. The UNCONDITIONED `pick_index_nn`/`nn_trees`/`nn_cache`
  path (sklearn `KDTree`, `rng.integers`) is **completely untouched** —
  this is why the reproduction below is byte-for-byte.
- `run_one_game(..., home_ratings=None, away_ratings=None)`: branches
  `conditioned = home_ratings is not None and away_ratings is not None and
  tables["team_kernel_h"] is not None`; when True picks
  `off_sim`/`def_sim`/`is_home_sim` from whichever team is on offense and
  calls `pick_index_nn_conditioned`; else calls the OLD unchanged
  `pick_index_nn`. `simulate(..., home_ratings=None, away_ratings=None)`
  passes through positionally; `simulate_from_states` untouched (its call to
  `run_one_game` still has only 7 positional args, valid since the two new
  params default to `None`).

**Sanity check 1 (measured, unconditioned mode unchanged):**
`F:/Repos/nfl_py3/.venv/Scripts/python scripts/sim04_engine.py
--n-games-per-season 3334` ->
`artifacts/sim04_engine/20260926T030954Z/report.json`: hits 2/5, log-loss
delta **+0.0044510** (unit 3d: +0.0045), SD ratio 1.08618 (unit 3d: 1.086),
points/game 41.774 (unit 3d: 41.8). Matches unit 3d exactly -> conditioning
being off truly reproduces old behavior.

**Sanity check 2 (measured, conditioned mode, look 1, one run, not tuned):**
script run ad hoc (not saved as a repo artifact — rerun from
`build_tables(TRAIN_SEASONS, condition_on_team=True)` if needed), 768 real
2015-2017 REG games, N=60 reps/game, actual pregame ratings both sides:
- correlation(sim_mean_margin, actual_margin) = **0.288** (positive, real
  team-quality signal reaches the simulated margin).
- implied home-field edge (both teams set to league-mean off/def rating,
  N=2000 games) = **+0.56 pts**, vs the real training-window (2009-2014)
  raw average home margin of **+2.57 pts**. Read: the conditioned engine's
  home_flag/lambda mechanism captures well under half of the real aggregate
  home-field effect at equal team quality — worth naming as a limitation,
  not a rejection ground (no sign flip, not tested for reliability).

### `scripts/sim04_loso.py` changes (mostly done)

- New constants: `TEAM_COND_N_REPS_REQUESTED=1000`, `TEAM_COND_N_REPS=200`,
  `RATING_COLS`.
- New `build_team_conditioned_hists(frame, game_features, n_reps)`: per
  graded season, `build_tables(train_seasons, condition_on_team=True)` (one
  build per season, reused for every game in that season — this is why the
  state-neighbor cache warms across games), then per game looks up
  home/away off/def EPA from `game_features_pbp.parquet` (fallback to that
  season's own `league_off_mean`/`league_def_mean` on NaN), calls
  `simulate(n_reps, rng, tables, home_ratings=..., away_ratings=...)`,
  and from the SAME margins array builds (a) the **raw** histogram
  (primary) and (b) `recenter_hist(shape_dev, predicted_margin_at_open)`
  (secondary). Order-checked against `frame.index` before returning.
- `main()` now also computes `tc_primary_summary`/`tc_secondary_summary` via
  the existing `summarize_candidate` (labels `"team_conditioned_raw"` /
  `"team_conditioned_recentered"`), adds them plus push-calibration rows to
  `report.json`, adds 10 new `per_game` columns
  (`team_conditioned_{raw,recentered}_{cover,push,loss,log_loss,brier}`),
  and a new `report["team_conditioning"]` block documenting the mechanism,
  the sanity numbers above, and the N-reps reduction rationale.
- **NOT YET APPLIED (the edit the cap blocked):** `report["record_command_draft_primary"]`
  and `report["record_command_draft_secondary"]` currently still read (WRONG,
  on disk right now):
  ```
  "record_command_draft_primary": (
      "F:/Repos/nfl_py3/.venv/Scripts/python -m nfl_ats weak-signals record "
      "--effect-units log_loss_improvement "
      f"--probability-positive {tc_primary_summary['week_blocked_bootstrap']['log_loss_probability_positive']} "
      "--signal sim04_engine_team_conditioned_raw_vs_discrete_read "
      f"--artifact artifacts/sim04_loso/{timestamp}/report.json "
      "--classification unresolved_below_power"
  ),
  "record_command_draft_secondary": ( ... same pattern with --signal sim04_engine_team_conditioned_recentered_vs_discrete_read ... ),
  ```
  This is WRONG: `weak-signals record`'s real flags (confirmed by reading
  `src/nfl_ats/cli_commands/registry.py` around line 652-790 and the worked
  examples in `docs/edge_audit_redteam.md:171-183`) are `--name`, `--league`,
  `--effect-units` (valid values incl. `log_loss_improvement`, confirmed in
  `src/nfl_ats/weak_signals.py:56-65`), `--effect`, `--interval-low`,
  `--interval-high`, `--probability-positive`, `--sample-games`,
  `--sample-blocks`, `--season-start`, `--season-end`, `--classification`,
  `--source`, `--description`, `--classification-evidence`, `--notes`. There
  is no `--signal` or `--artifact` flag.

## Tried

- Confirmed pbp already carries `home_team`/`away_team` (no merge collision
  risk) and `game_features_pbp.parquet` has the 4 EPA rating columns with
  NaNs only in early 2009 (48 rows).
- Profiled the conditioned draw path (cProfile) twice; found and fixed the
  real bottleneck (sklearn `KDTree.query` validation + `rng.choice(p=...)`),
  not the kernel-weight math itself.
- Measured conditioned-mode throughput at three scales (2000/20000 games,
  fixed ratings): ~48 -> ~69 (after scipy+CDF fix) -> ~110 games/sec at
  n=20000 warm cache. At 110 games/sec, 1537 games x 1000 reps/game (the
  orchestrator's literal ask) would take **~3.9 hours**, breaching the ~1hr
  budget, so `TEAM_COND_N_REPS` was set to 200 (not 1000) BEFORE running or
  viewing any grade — documented in `report["team_conditioning"]
  ["n_reps_reduction_reason"]`. At 200 reps x 1537 games (~307k total
  sim-games) expect roughly 45-50 min wall clock, but this has **not been
  run to completion yet** (see Next).

## Next

1. **Apply the blocked edit** to `scripts/sim04_loso.py`: replace the
   `record_command_draft_primary`/`record_command_draft_secondary` block
   (currently the wrong `--signal`/`--artifact` version quoted above, near
   the end of `main()`, right after the `push_probability_calibration_at_key_numbers`
   / `engine_training_windows` keys) with real-flag versions. Full text to
   use (verified against the registry parser and edge_audit_redteam.md
   examples) — the intent, ready to paste as the replacement for the
   quoted-above wrong block:
   ```
   "record_command_draft_primary": (
       "nfl-ats weak-signals record --name sim04_engine_team_conditioned_raw_vs_discrete_read "
       "--league nfl --effect-units log_loss_improvement "
       f"--effect {tc_primary_summary['pooled_log_loss_delta_vs_baseline']} "
       f"--interval-low {tc_primary_summary['week_blocked_bootstrap']['log_loss_delta_ci95'][0]} "
       f"--interval-high {tc_primary_summary['week_blocked_bootstrap']['log_loss_delta_ci95'][1]} "
       f"--probability-positive {tc_primary_summary['week_blocked_bootstrap']['log_loss_probability_positive']} "
       f"--sample-games {tc_primary_summary['n_games']} "
       f"--sample-blocks {tc_primary_summary['week_blocked_bootstrap']['n_blocks']} "
       "--season-start 2020 --season-end 2025 --classification unresolved_below_power "
       f"--source artifacts/sim04_loso/{timestamp}/report.json "
       "--description \"LOSO 2020-2025 Tuesday-opener grade of the raw team-conditioned play-level "
       "simulator histogram (kernel-weighted on pregame off/def EPA, k_state=200, h/lambda "
       "predeclared) vs the served discrete three-way read at the opening spread; three-way log "
       "loss, week-blocked bootstrap\" "
       "--classification-evidence \"Interval crossing zero never closes a signal (AGENTS.md); "
       "no resolved wrong sign and no positive-control bound measured for this candidate, so "
       "unresolved_below_power is the only admissible classification\" "
       f"--notes \"N reps/game reduced 1000->{TEAM_COND_N_REPS} for the ~1hr budget (see "
       "team_conditioning.n_reps_reduction_reason in this artifact); raw (non-recentered) "
       "histogram, so it also carries the engine's own margin-mean bias unlike the recentered "
       "secondary row\""
   ),
   "record_command_draft_secondary": (
       "nfl-ats weak-signals record --name sim04_engine_team_conditioned_recentered_vs_discrete_read "
       "--league nfl --effect-units log_loss_improvement "
       f"--effect {tc_secondary_summary['pooled_log_loss_delta_vs_baseline']} "
       f"--interval-low {tc_secondary_summary['week_blocked_bootstrap']['log_loss_delta_ci95'][0]} "
       f"--interval-high {tc_secondary_summary['week_blocked_bootstrap']['log_loss_delta_ci95'][1]} "
       f"--probability-positive {tc_secondary_summary['week_blocked_bootstrap']['log_loss_probability_positive']} "
       f"--sample-games {tc_secondary_summary['n_games']} "
       f"--sample-blocks {tc_secondary_summary['week_blocked_bootstrap']['n_blocks']} "
       "--season-start 2020 --season-end 2025 --classification unresolved_below_power "
       f"--source artifacts/sim04_loso/{timestamp}/report.json "
       "--description \"LOSO 2020-2025 Tuesday-opener grade of the team-conditioned simulator "
       "histogram shape re-centered on the served predicted_margin_at_open vs the served discrete "
       "three-way read; three-way log loss, week-blocked bootstrap\" "
       "--classification-evidence \"Interval crossing zero never closes a signal (AGENTS.md); "
       "no resolved wrong sign and no positive-control bound measured for this candidate, so "
       "unresolved_below_power is the only admissible classification\" "
       f"--notes \"N reps/game reduced 1000->{TEAM_COND_N_REPS} for the ~1hr budget (see "
       "team_conditioning.n_reps_reduction_reason in this artifact); shares draws with the raw "
       "primary row, correlated, never pool as independent\""
   ),
   ```
2. **Syntax-check** `scripts/sim04_loso.py` after the edit (e.g.
   `F:/Repos/nfl_py3/.venv/Scripts/python -c "import ast; ast.parse(open('scripts/sim04_loso.py').read())"`).
3. **Run the grade**: `F:/Repos/nfl_py3/.venv/Scripts/python scripts/sim04_loso.py`
   in the background (expected ~45-60 min at TEAM_COND_N_REPS=200; if it
   is running much slower than the ~110 games/sec benchmark, that's a real
   finding to log, not a reason to silently raise N or kill it early — let
   it finish or stop it and record what ran). This produces a NEW
   `artifacts/sim04_loso/<timestamp>/report.json` + `per_game.parquet`
   (supersedes `20260926T022317Z`, which only had the league-average
   provisional candidate).
4. Pull `report["team_conditioned_raw"]` and `report["team_conditioned_recentered"]`
   (pooled log-loss/Brier deltas, per-season deltas, week-blocked CI,
   `log_loss_probability_positive`), plus `report["push_probability_calibration_at_key_numbers"]`
   rows for the two new candidates, plus the reliability-decile tables — that
   is the material for the "Unit LOSO grade" section this subtask still owes
   `docs/sim04_unit_log.md`, and for finishing this lane's State section.
5. Run BOTH `report["record_command_draft_primary"]` and
   `["record_command_draft_secondary"]` verbatim (prefix with
   `F:/Repos/nfl_py3/.venv/Scripts/python -m` if `nfl-ats` isn't on PATH in
   the shell used — check `nfl-ats --help` first). If either command errors,
   the verdict is wrong; do not weaken the classification to make it accept
   — report the error and pick the correct classification/flags instead.
6. Update this lane's State (replace this "STOPPED AT CAP" state with the
   finished grade numbers + registry confirmation) and add the "Unit LOSO
   grade" section to `docs/sim04_unit_log.md` (mechanism, sanity numbers,
   pooled/per-season deltas for both candidates, CI, probability_positive,
   reliability table summary, push calibration at key numbers, the N-reps
   deviation and why, and the two registry commands run with their exact
   output).
7. Only after 1-6: SIM-05 (4th-down policy swap, QB-out what-if) and the
   dashboard chart are still open, per the Goal section — separate clearing
   units, not part of this one.

## Open

- Whether TEAM_COND_N_REPS=200 gives push-probability-at-key-numbers
  estimates fine enough to be useful (min increment 0.005) is itself worth
  one line in the writeup — coarser than the baseline's continuous
  lattice-derived push probabilities. Not a blocker, just note it.
- The implied home-field edge (+0.56 sim vs +2.57 real at equal rating) is a
  named, measured gap in the conditioning mechanism, not yet explained or
  fixed. Do not silently patch `TEAM_KERNEL_LAMBDA` to close this gap without
  predeclaring a new value first and re-running sanity check 1 to confirm
  unconditioned mode is still untouched.
