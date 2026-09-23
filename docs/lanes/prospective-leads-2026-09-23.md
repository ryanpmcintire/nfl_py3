# Prospective leads 2026-09-23

## Goal

Register the market-move-decomposition, opener-error-transfer, and
total-conditioned-lattice research candidates as prospective paired
challengers of the served four-term probability, recorded at each weekly
lock without changing the served card, where each has a genuine
point-in-time live input path.

## State

**Registration mechanism (read/measured):** `artifacts/prospective/challengers.json`
(`nfl_ats.prospective_scoring.challenger_registry_path`,
`src/nfl_ats/prospective_scoring.py:290-358`) lists challengers with
`status: ACTIVE_PROSPECTIVE`; `active_challenger_ids()` drives what
`scripts/lockday_verify.py` audits as wired vs `unwired` against
`PUBLISH_CHALLENGER_RESULT_KEYS`/`REFRESH_CHALLENGER_RESULT_KEYS`
(`lockday_verify.py:260-330`), which check that
`weekly_recording_command` corresponds to a real recorder called from
`nfl-ats publish-predictions --record-decisions` or `refresh-picks
--record-decisions`. `docs/lanes/week3-research-recorders.md` is the live
precedent (recorders wired this way, audited weekly).

**Root correction (mid-task):** the all-books median market move was
re-graded on the active Sunday-inclusive window and does NOT beat the
served move (-0.13 pts [-0.59, +0.33], P+ 0.22; `docs/lanes/market-move-decomposition.md`
Unit 3). Dropped from registration per root's instruction, though it did
have a live input path (odds-ingest already captures all
`LEADERSHIP_WEIGHTS` books multiple times daily via the scheduled
`odds_capture.ps1` -> `nfl-ats odds-ingest --markets spreads,h2h,totals` job,
`scripts/capture_scheduler.py:41,404` on).

**Registered (measured, this session):** `total_conditioned_key_number_lattice_v1`
added to `artifacts/prospective/challengers.json` (69 entries now;
validated via `active_challenger_ids`/`find_challenger`). Live-dry-run
verified: a standalone script reusing `total_conditioned_read()` and
`prior_pool_for_week()` (`scripts/total_conditioned_lattice.py`,
`nfl_ats.mass_preserving_lattice`) against the live feature table produced
16/16 Week 3 2026 per-game probability rows (`total_line`/`spread_line`
already live pregame columns, `src/nfl_ats/features.py:679`; pool cutoff
2026-09-24, all games ungraded/`result` NaN, eligible pool 1,391 prior
games -- point-in-time correct). Sample:
`2026_03_CIN_PIT` served_cover=0.5209 vs challenger_cover=0.4965 (mid
total_band); `2026_03_SEA_WAS` served=0.5254 vs challenger=0.5125 (low
band). **Status: registered but NOT wired** -- no
`record_total_conditioned_lattice_decisions()` writer exists yet, and it is
not in `PUBLISH_CHALLENGER_RESULT_KEYS`, so this week's real lock run will
show it `unwired` in the audit until root builds the writer.

**Not registered -- no live input path:** college-trained opener-error term
(`docs/lanes/opener-error-transfer.md`). Confirmed via
`grep -ic cfb scripts/capture_scheduler.py` = 0: no scheduled job pulls CFB
lines. `data/cfb/lines/raw/20260816T143907Z/manifest.json` is a single
historical snapshot from LEAD-53-era research, not a recurring capture.
`scripts/opener_error_transfer_unit3.py` is a manual/offline research
script, not wired to any weekly job. Nothing to register until a weekly
CFBD pull (CFBD_API_KEY already in user env) is added to
`capture_scheduler.py` and a live point-in-time CFB-to-NFL transfer scorer
is built.

## Tried

- Full read of `prospective_scoring.py`, `cli_commands/prospective.py`,
  `lockday_verify.py` challenger dicts, and an existing full registry entry
  (`pbp08_protection_mismatch_early_window_v1`) as the format precedent.
- Live dry run of total-conditioned-lattice against Week 3 2026 (script at
  `C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3\...\scratchpad\total_conditioned_lattice_live_dry_run.py`,
  not in repo -- scratchpad only).
- Confirmed odds-ingest already captures all `LEADERSHIP_WEIGHTS` books
  (moot now given the market-move drop).

## Next (root)

1. Build `record_total_conditioned_lattice_decisions()` (pattern after
   `served_total_challenger.py` / `lattice_centre_challenger.py`): at lock,
   for each current-week game compute `total_conditioned_read()` against the
   point-in-time eligible pool (same construction as the dry-run script),
   write rows to a new `prospective/total_conditioned_lattice_decisions.parquet`
   ledger with the `CHALLENGER_DECISION_COLUMNS` schema.
2. Call that writer from `publish-predictions --record-decisions`; add its
   result key to `PUBLISH_CHALLENGER_RESULT_KEYS` in `lockday_verify.py`.
3. Re-run `scripts/lockday_verify.py --season 2026 --week 3 --json` and
   confirm `total_conditioned_key_number_lattice_v1` moves from unwired to
   recorded.
4. If college opener-error is still wanted: schedule a weekly CFBD pull in
   `capture_scheduler.py`, then build a live scorer before registering.

## Open

- The registered entry currently has no working recorder -- it will show as
  `unwired` in the next real audit until Next item 1-2 land. This is the
  same gap class already tracked in `week3-research-recorders.md`, not a
  new failure mode.
- No served-path change, no commit/publish, no registry entries touched
  besides the one addition above.
