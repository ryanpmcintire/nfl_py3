# ref-tests-kit — overlay/tilt shared test scaffold extraction (wave 1)

**Task.** Execute `reports/wave1/hyg-tests-audit.md` recommendation 1 ONLY:
extract the shared overlay/tilt fixture scaffold into
`tests/_overlay_test_kit.py` and convert the duplicated files to use it.
Branch: `swarm/ref-tests-kit`.

## What was done

Created `tests/_overlay_test_kit.py` (**measured**, 129 lines) with three
shared writers:

- `write_challenger_registry(artifacts, *, challenger_id, model_config,
  status="ACTIVE_PROSPECTIVE")` — the `prospective/challengers.json` payload.
- `write_active_model_and_card(artifacts, *, season, week, created_at_utc,
  recommendations, ...)` — the weekly forecast card (`metadata.json` +
  `recommendations.csv`) and `active_ats_model.json`. Optional parameters
  cover every measured variant: `forecast_dir`, `feature_profile`,
  `probability_method` (used only by the ECDF/era pair), `min_train_games`,
  `min_edge`, `feature_table_path`.
- `write_registry_root(tmp_path, *, stadium_station_map_csv)` — the reference
  stadium/station map used by the three weather-KN suites.

Each converted file keeps a thin local wrapper with its original name and
signature, so **every call site is untouched**; only the duplicated bodies
were replaced.

## Files converted (14 of the audit's 18)

| File | Helpers converted |
|---|---|
| test_backup_qb_fade_overlay.py | registry + active-model card |
| test_coach_fade_overlay.py | registry (`_write_overlay_registry`) + active-model card |
| test_division_revenge_tilt_overlay.py | registry + card |
| test_ecdf_mapping_incumbent_overlay.py | challenger registry + feature-table card writer |
| test_era_weighted_half_life_8_overlay.py | same as ecdf |
| test_forecast_cold_visitor_tilt_overlay.py | registry + card + registry root |
| test_forecast_weather_kn_precip_high_total_tilt_overlay.py | registry + card + registry root |
| test_forecast_weather_kn_warm_team_cold_late_tilt_overlay.py | registry + card + registry root |
| test_injury_value_tilt_overlay.py | tilt registry + card |
| test_interim_hc_first_game_tilt_overlay.py | registry + card |
| test_player_arrests_back_side_overlay.py | registry only (see "left unconverted") |
| test_smooth_cdf_mapping_overlay.py | challenger registry + feature-table card writer |
| test_spread_gap_zone_fade_overlay.py | registry + card |
| test_surface_switch_tilt_overlay.py | registry + card |

**Files NOT converted (4 of 18)** — no shared scaffold present to extract:

- `test_four_overlay_composition.py`, `test_four_overlay_incumbent.py`,
  `test_nflcom_refresh_overlay.py`, `test_injury_signal_refresh_tilt.py` —
  **measured** via AST hash comparison of all private helpers across all 18
  overlay/tilt files: these four share zero byte-identical or constants-only
  helper bodies with the cluster (they use different recorder/card shapes).
  Converting them would mean inventing an abstraction over non-duplicated
  code. Left untouched.
- `test_player_arrests_back_side_overlay.py` is partial by design: its
  `_write_active_model_and_card` writes a structurally different metadata
  payload (no `season`/`week` keys, inline card frame, hardcoded alpha).
  Routing it through the shared writer would change bytes on disk inside a
  research-screen fixture for zero duplication removed, so only its registry
  writer was converted.

## Measured line delta vs the audit's estimate

**Measured** (`git diff --stat tests/`, this session): 14 files changed,
259 insertions(+), 741 deletions(−); plus the new 129-line kit → net
≈ −353 lines. The audit's ~1,010-line estimate assumed `_recorder_predictions`
and `_write_data_root` were extractable; they are not without obfuscation:

- `_recorder_predictions()` exists in 9 files but every body encodes a
  *different* scenario (game ids, teams, spreads are per-overlay fixtures),
  so it is scenario data, not scaffold.
- `_write_data_root()` / `_write_repo_root()` wrap each file's own schedule
  fixture via `write_snapshot(...)` — nothing shared beyond one call.

The duplicated boilerplate that WAS real: `_write_registry` (8 byte-identical
copies, md5-verified), the simple `_write_active_model_and_card` (13 copies
differing only in season/week/created-at constants, diff-verified), the trio's
`_write_challenger_registry` (3 byte-identical copies), and
`_write_registry_root` (3 copies differing only in stadium CSV rows).

## Verification

- **Test IDs unchanged**: AST-extracted full signatures (name + args) of every
  `test_*` function in all 18 files from `HEAD` vs working tree —
  **measured**: 356 before, 356 after, multisets identical; kit adds 0 tests.
- No test bodies were weakened: no `_is_leak_safe_*`, `refuses_*`,
  `requires_its_*_columns`, fingerprint-mismatch, lock-window, or
  inactive-registration assertion was edited — only the private fixture-writer
  helpers were. These are research-screen suites, and none of the 18 files is
  safety/canary/contract-release-blocking per the audit's own classification;
  regardless, every leakage/contract assert survives verbatim (verified by the
  signature diff above and the suite run below).
- **measured** pytest on the 14 converted files: **308 passed, 1 skipped**
  (skip is environmental: "local nflverse snapshot not present").
- Full gates (**measured**):
  - `ruff format --check .` → 637 files already formatted
  - `ruff check .` → All checks passed!
  - `mypy src` → Success: no issues found in 105 source files
  - `pytest -q` → **1855 passed, 5 skipped** (all skips environmental:
    local snapshots/artifacts absent from the checkout)

## Notes

- The kit imports `ACTIVE_ATS_MODEL_VERSION` from `nfl_ats.active_model`,
  exactly as each converted file did, so written payloads keep the same
  version string and JSON key order (probability_method inserted after
  calibration_method / feature_profile, matching the originals).
- No experiment window was opened: this is a pure test refactor; nothing was
  scored, recorded, or adjudicated.
