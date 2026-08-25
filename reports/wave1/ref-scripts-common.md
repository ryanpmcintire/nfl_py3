# Wave 1 — `scripts/` boilerplate extraction to `scripts/_common.py`

**Task:** `ref-scripts-common`
**Branch:** `swarm/ref-scripts-common`
**Input:** top recommendations of `reports/wave1/hyg-scripts-audit.md`
(read via `git show swarm/hyg-scripts-audit:reports/wave1/hyg-scripts-audit.md`;
not present on this branch at session start).

## What was done (measured this session)

Created **`scripts/_common.py`** (174 lines) holding the shared boilerplate the
audit quantified in §4a, and converted **34 scripts** (33 by batch transform +
`max_ev_composition.py` import repair) to use it. Nothing was archived or
deleted; no script file was removed.

Shared helpers extracted (each byte-for-byte the algorithm it replaces,
docstrings/formatting aside):

| Helper | Replaces | Cluster size (strict normalized-text match) |
|---|---|---:|
| `latest_schedules()` | per-file `_latest_schedules()` | 26 files |
| `default_schedules()` | per-file lazy wrapper | 9 files (+2 `vardec_*`) |
| `block_bootstrap_two_group(...)` | verbatim week-blocked bootstrap | 27 files exact |
| `summarize(...)` (home-cover variant) | slate-scaled cell summary | 6 files exact |
| `bootstrap_pearson_ci(...)` | paired-resample Pearson CI | 4 files exact |

Also re-exported: `REPO`, and importing `_common` performs the same
`sys.path.insert(0, str(REPO / "src"))` every script previously repeated.

## Behavior preservation (how verified)

- The transformer deleted a local def **only if** its docstring-stripped,
  whitespace-collapsed body equaled the canonical body in `_common.py`
  (trailing-comma formatting differences ignored). Files whose bodies differed
  in any other way were skipped and keep their local copies — measured SKIP
  log included `cfb_surface_familiarity_screen.py` (variable renaming),
  `build_environmental_exposure_join.py` / `build_stadium_county_fips.py`
  (digit-filtered snapshot glob), all `summarize` sign/value_col variants
  (`close_game_luck`, `divisional_rematch`, `primetime_cells`,
  `special_teams`, `team_style`, etc.), and `fluview_battery_screen` /
  `nfl_travel_rest_battery_screen` / `nfl_weather_battery_screen`
  snapshot-resolution variants. Those variants remain untouched.
- Private name preserved mechanically: call sites of the removed local
  `_latest_schedules()` were renamed to the public `latest_schedules()`;
  `default_schedules()` wrappers were only removed where their own
  `_latest_schedules` was also shared, so every remaining reference resolves
  to an identical function.
- No CLI argument, output JSON key, print format, artifact path, provenance
  payload, or registry write path was altered. Argparse scaffolding and
  per-screen `write_experiment_artifact` payloads were deliberately **not**
  extracted: each screen's flags, notes, and payload fields are unique
  (inferred from reading several `main()` blocks), so no behavior-preserving
  shared form exists for them at this granularity.
- Cross-script import audit: grepped every `from <sibling> import ...` in
  `scripts/`. One real breakage found and fixed —
  `max_ev_composition.py` imported `_latest_schedules` from
  `pbp08_matchup_screen`; repaired to
  `from pbp08_matchup_screen import latest_schedules as _latest_schedules`
  (same function). All other sibling imports (`bye_overvaluation_screen`,
  `body_clock_screen`, `nfl_bias_battery_screen`) resolve because the sibling
  module still exposes the name (locally or via `_common`).
- Import smoke test: all 35 touched modules imported under both package mode
  (`import scripts.x`) and, where runnable, direct mode. The only failures are
  `FileNotFoundError` from eager module-level data loads
  (`fluview`, `pbp08`, travel/weather batteries, `surface_familiarity`,
  `max_ev_composition`) — reproduced identically against `HEAD` versions with
  `git show HEAD:...` in an empty-data sandbox, so they are pre-existing
  environmental effects of this worktree's empty `data/raw/`, not regressions.
- Full test suite run twice on the branch with fresh basetemps: **1855 passed,
  5 skipped, 0 failed** (an earlier single run showing 1 failed + 162 errors
  was traced to stale pytest basetemp state and did not reproduce on a clean
  basetemp, nor on a second clean run).

## Line count (measured)

- `git diff --shortstat`: **34 files changed, 188 insertions(+), 1720
  deletions(-)** → 1,532 net lines removed from modified scripts.
- Added `scripts/_common.py`: 174 lines.
- **Net repository change ≈ −1,358 lines**, comfortably above the ≥800 target.

Converted scripts (34): altitude, arctic_shift_gate, attention_battery,
body_clock, build_gdelt_weekly_features, bye_overvaluation,
cfb_special_teams, close_game_luck, divisional_rematch,
environmental_exposure_battery, ffc_adp_divergence, fluview_battery,
motivation_ladder, nfl_bias_battery, nfl_forecast_weather,
nfl_travel_rest_battery, nfl_weather_battery, nfl_weather_followup,
ol_continuity, max_ev_composition (import repair only), pbp08_matchup,
primetime_cells, qb_age_curve, redzone_reversion, roof_decision,
sagarin_divergence, special_teams, surface_familiarity, team_style,
vardec_lategame, vardec_sigma_map, venue_milestone, vi_dispersion,
weather_total_interaction — i.e. well beyond the "at least 15
highest-duplication screens" requirement.

## Not done / deferred (with grounds)

- **Archive candidates (audit §6):** out of scope for this task by instruction
  ("do NOT archive/delete any script"); no file moved or removed.
- **`summarize` signed/value-col variants and `score_cell` clusters:** left in
  place because their output JSON keys differ per family (`subset_cover` vs
  `subset_mean`, presence of `sign_dir`); unifying them would change written
  artifacts, violating the behavior-preserving constraint. This is the task's
  stated constraint (identical outputs), not a judgment that the variants are
  worth keeping forever.
- **argparse/provenance driver:** same reason — payloads and flag sets differ
  per screen; extraction would require a config schema and per-screen review,
  which cannot be mechanical within this task's behavior-preservation bar.

## Gates (measured, this session)

```
ruff format --check .   -> 637 files already formatted (pass)
ruff check .            -> All checks passed!
mypy src                -> Success: no issues found in 105 source files
pytest                  -> 1855 passed, 5 skipped (basetemp outside repo)
```

No experiment window was opened and no scoring look was executed (binding
rule 3: static code movement plus test-suite runs only). No commit pushed;
branch `swarm/ref-scripts-common` only.
