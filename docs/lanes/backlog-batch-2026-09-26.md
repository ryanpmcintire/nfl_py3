# Backlog batch 2026-09-26

## Goal
Work remaining backlog 2026-09-26: triage the 16 research lanes named by the
owner, close every one whose Next/Open showed nothing left to run, and keep
open only lanes with a real remaining action.

## State
Closed 16 lanes to `docs/lanes/done/` (each got a "Closed 2026-09-26" note at
the top of its State section, verified against `registry/weak_signals.json`
and git history, not just the lane's own text):
lead59-archive-battery, line-move-regrade-legacy, market-move-decomposition,
mod18-spread-regime, total-conditioned-lattice, opener-population-backfill,
pooled-signal-model, conditional-signal-atlas, scheduler-once-timeout,
lane-replay-dream-rsi, prospective-leads-2026-09-23, opener-error-transfer,
surface-switch-fitted-term, lead65-protection-window-split,
clv-metric-everywhere, every-metric-every-experiment.

Verification done, not just trusted: registry grep confirmed all named cells
present (e.g. `market_move_decomposition_e_all_books_median_active_window`,
`total_conditioned_key_number_lattice_{log_loss,accuracy}`,
`mod18_spread_regime_{served_2020_2025,extended_2011_2025}`,
`pooled_signal_sixth_fit_vs_{four_term,model_only}`, `opener_error_transfer_v5`
family, `surface_switch_fitted_term` family, `special_teams_return_top_quartile`,
`hc_year_one_fade`); git log/ROADMAP confirmed conditional-signal-atlas's
commit+publish and scheduler-once-timeout's fix (`786a569`) already landed;
lead59's two type-trait cells confirmed recorded and committed (`93ba72d`)
and ROADMAP LEAD-59 already carries the finding (only
`docs/officials_archive_battery.md`'s tail note is still unwritten — optional
hygiene, not blocking).

`docs/lanes/README.md` Active section and Done section both updated to match.

## Tried
Read each lane's Next/Open; where a lane's own reconciliation note said
"nothing further to run" the underlying registry/git state was spot-checked
rather than taken on faith, since several lanes carry stale pre-reconciliation
"Next" text (draft record commands, "script not yet run") below a newer
dated note that supersedes it.

## Next
Nothing queued by this batch. The one open policy question surfaced but not
decided: whether every-metric-every-experiment's backfilled cells count as
new "looks" for AGENTS.md's look-counting rule (proposal on record: same
family, flagged `backfilled` in notes, not counted) — an owner call, not a
command to run.

## Open
- every-metric-every-experiment's backfilled-look accounting question above.
- lead59's `docs/officials_archive_battery.md` tail note (cosmetic, listed
  above, never blocking).
- No lanes outside the named 16 were touched; `sim08-simulator-rebuild`,
  `backlog-batch-2026-09-25`, and other Active lanes retain their own
  real remaining actions untouched by this batch.

## Decided 2026-09-26 (root)

- Backfilled cells (a new metric scored on predictions already preserved from
  a counted fit) stay in the fit's family, flagged backfilled, and do not add
  a look: no new fit, band, or cut was chosen. Each metric is still reported
  beside the others, so no best metric is picked. This closes the question.
- Also closed: inactives-capture-empty (live Thursday capture returned 11 real
  rows), backlog-batch-2026-09-25. Fixed: Friday odds window (`b94e59d`),
  Bovada error evidence and card note plural (`ebf7e02`).
- SKY-07 publication decisions made from the rules (docs/open_benchmark_suite.md, Decisions 2026-09-26); next: per-dataset nflverse license check and first local export.

## SKY-07 result 2026-09-26 (steps 1 and 6 of the blocker list)

Finding first: `src/nfl_ats/open_benchmark.py` and `tests/test_open_benchmark.py`
were deleted whole by `b7ed31d` ("Repository cut... -54%"), but ROADMAP.md
SKY-07 and `docs/open_benchmark_suite.md` still read as if the foundation is
implemented. **measured** (`git show --stat b7ed31d` shows -674/-238 lines;
`git show HEAD:src/nfl_ats/open_benchmark.py` errors "does not exist"). I
restored the library from `b7ed31d^` into `src/nfl_ats/open_benchmark.py`,
stripped its 8 docstrings to match the current no-comments/no-docstrings
policy (logic untouched), and did NOT restore the deleted test file (test
moratorium: no new tests). **This restoration is on disk, uncommitted** per
this task's "no commit" constraint — the orchestrator must decide whether to
commit it and whether to backfill test coverage.

Step 1 (license check, **measured** via `gh api`): the only dataset needed for
core columns is nflverse `schedules`/`games`
(https://github.com/nflverse/nflverse-data/releases/tag/schedules, asset
`games.csv`). `nflverse-data` repo `LICENSE.md` decodes to CC BY 4.0
("Attribution 4.0 International"). Release notes say data is maintained
upstream in `nflverse/nfldata` (`games.rds`); that repo carries no separate
LICENSE file or conflicting license text in its README, so nothing overrides
the nflverse-data CC-BY-4.0 terms actually governing the fetched release
asset. Kept: nflverse schedules. No other dataset was needed (release has zero
extra feature columns, so team_stats/other nflverse files were out of scope
and untouched). Confirms `config/source_policies.json`'s existing "green,
allowed_with_attribution" entry.

Step 6 (export + independent verify, **measured**): built from the existing
local snapshot `data/raw/20260923T005026Z/schedules.parquet` (fetched
2026-09-23, seasons 2009-2026, no new fetch needed — respects "light memory
only"). `kickoff_utc` = `gameday`+`gametime` localized `America/New_York` (per
nflreadr data dictionary: gametime is always Eastern) then converted to UTC;
`decision_time_utc` = kickoff - 1h; `inputs_observed_through_utc` = same.
Split: train = seasons 2009-2024, validation = season 2025 (both "completed"),
test = season-2026 rows with `kickoff_utc >= now` AND a posted `spread_line`
(excludes both future weeks with no line yet and the one stale 9/24 game whose
score isn't in this snapshot — neither mislabeled, both just left out of the
release). Exported to
`F:\Repos\nfl_py3\tests\scratch\open_benchmark_v1\` (observations.csv +
manifest.json): **4661 rows total — train 4345, validation 285, test 31**.
`license_spdx=CC-BY-4.0`, `source_urls` = the 3 URLs above, `public_url=None`
(placeholder).

Independent verification, all passed: recomputed SHA-256 of observations.csv
by hand (`ab5d84e5af...31681`) matches `manifest.json`'s
`files[0].sha256`/`dataset_content_sha256` field and byte count; 0 of 31 test
rows carry `ats_margin`/`cover_side`; game_id unique across all 4661 rows;
chronology strict (max train kickoff 2025-02-09 < min validation kickoff
2025-09-05 < max validation kickoff 2026-02-08 < min test kickoff 2026-09-27);
0 rows violate `inputs_observed_through_utc <= decision_time_utc < kickoff_utc`.
`manifest.json` publication block: `{"ready": false, "blockers":
["external hosting location is not configured"]}` — exactly and only the
unset public URL, as required.

Remaining blockers before publication: (1) tag a GitHub release and pin its
asset URL into `public_url` (owner/orchestrator action, not run here); (2)
decide whether to commit the restored `src/nfl_ats/open_benchmark.py` and
whether to recreate `tests/test_open_benchmark.py` under the moratorium's
rules; (3) ROADMAP.md/docs still describe a foundation that was silently
deleted — needs a correction pass independent of this export.
- SKY-07 retired (root, 2026-09-26): the library was cut on 2026-09-10 as unimported; not restored, since a benchmark changes no pick, grade, or page. ROADMAP row and doc updated.
