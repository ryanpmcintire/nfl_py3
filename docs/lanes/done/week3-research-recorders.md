# Week 3 research recorder coverage

## Goal

Get the late-week-move-follow refresh recorders actually recording for Week 3
(they were silently empty), close the matching `lockday_verify.py` registry
gaps, and name the safe root command to record `tiebreaker_low_side_shade`
and `total_conditioned_key_number_lattice_v1` for Week 3 before kickoffs.

## State

**2026-09-25: lane closed.** `scripts/lockday_verify.py --season 2026 --week
3` now reports `0 MISSING of 63 active`. All nine challenger records named as
missing in the prior session (`total_conditioned_key_number_lattice_v1`, the
six `late_week_move_follow_refresh_decisions.parquet` challenger IDs, and
`consensus_movement_1_0_off_incumbent`) are recorded with full Week 3 rows,
written by a scheduled pass before this session (last success timestamp on
`refresh_last_call_fri_1330` is `2026-09-25T13:32:07-04:00` per
`data/scheduler_state.json`) -- this session verified only, ran no recorder
command. The ninth, `tiebreaker_low_side_shade`, remains at 0 Week 3 rows by
permanent design (served tiebreaker total always carries the `-1.0` shade, so
served/shaded arms can never pair -- `docs/tiebreaker.md`'s 2026-09-10
section), confirmed as an expected `--` gate in lockday_verify, not a `!!`
MISSING. See dated 2026-09-25 entry under Next for the full per-ledger
breakdown. No code changed, nothing committed, nothing pushed this pass.

**Root cause of the empty `late_week_move_follow_refresh_decisions.parquet` for
weeks 2-3 found and fixed.** Reproduced `build_late_week_move_follow_refresh_rows`
for season 2026 week 3 outside recording (scratch script calling `plan_refresh`
+ `original_card` + `load_decision_quotes` + the builder directly, no
`--record-decisions`). Before the fix it returned empty with
`{"reason": "No pre-deadline late-week book changes are available.",
"refused_quote_rows": 394}`.

Named gate: since the-odds-api was cancelled 2026-09-19, free-source captures
(`provider` = `bovada_public_nfl`, `the_odds_gap_lineshop_private`) never
populate `bookmaker_last_update_utc` (always NaT) and instead stamp
`quote_timestamp_basis = "capture_observed_utc"`. Two filters dropped 100% of
these rows:
- `late_week_follow_frame`'s own `safe` mask in
  `src/nfl_ats/sharp_book_movement_features.py:220` required
  `bookmaker_last_update_utc.le(observed_at_utc)` with **no escape at all** for
  self-timed quotes.
- `sharp_book_movement_features`'s internal filter (same file, ~line 124) had
  the matching escape (`bookmaker_last_update_utc.isna() &
  quote_timestamp_basis.eq("capture_observed_utc")`) but wrongly gated it
  behind `include_sunday`, which `late_week_follow_frame` never sets True.

Fix applied (both in `src/nfl_ats/sharp_book_movement_features.py`): the `safe`
mask now accepts a row when `bookmaker_last_update_utc.le(observed_at_utc)` OR
(`bookmaker_last_update_utc` is null AND `quote_timestamp_basis ==
"capture_observed_utc"`); the internal filter's escape lost its
`include_sunday &` prefix so it applies any day, not just Sunday. No change to
`cutoff_utc`/deadline/Monday-Wednesday-Sunday window logic -- the pick-deadline
guard is untouched. Re-ran the same repro after the fix: `rows empty? False,
len 16`, `diagnostics {"skipped": False, "games_considered": 16, ...,
"refused_quote_rows": 0}` -- all 16 Week 3 games, including ATL at GB before
its 20:15 ET kickoff, produced a row without writing anything (repro never
calls the `record_*` wrapper). `ruff check` and `mypy` on the changed file
both pass clean.

**`scripts/lockday_verify.py` `DEDICATED_LEDGERS` gap also closed.** Added
imports for `consensus_movement_refresh_overlay.LEDGER_NAME` and
`late_week_follow_no_sunday_blackout_overlay.LEDGER_NAME`, and six new
`DEDICATED_LEDGERS` entries: `consensus_movement_1_0_off_incumbent`,
`late_week_follow_no_sunday_blackout` (each pointed at their own ledger file
via `_parquet_ledger`), plus `late_week_follow_no_news_veto_off_incumbent`,
`late_week_leader_median_follow_0_5_off_incumbent`,
`late_week_leader_median_follow_flat_1_0_off_incumbent` pointed at the shared
`late_week_move_follow_refresh_decisions.parquet` the same way the two
already-registered arms (`late_week_move_follow_refresh_v1`,
`late_week_leader_median_follow_v1`) share it -- their "recording" is the
parent row's existence (confirmed this is intentional registry semantics, not
a code bug: these three IDs only ever appear as metadata columns on the
`late_week_move_follow_refresh_v1` row, never their own `challenger_id`).

Verified: `python scripts/lockday_verify.py --season 2026 --week 3
--run-summary artifacts/scheduled_locks/2026-week-03/weekly_summary.json` now
shows `challengers: 53 recorded, 8 skipped, 2 MISSING, 0 pending wiring of 63
active` -- down from 5+ unexplained MISSING. `consensus_movement_1_0_off_incumbent`
and `late_week_follow_no_sunday_blackout` show `ok` with 80/96 rows (real data
already recorded by earlier refresh passes, previously invisible to the
verifier). The three late-week sub-arms plus the two already-registered arms
now show `--` with a named "records only on a late-week refresh pass" gate
instead of unexplained MISSING (this summary file is the Tuesday-lock
snapshot, so 0 rows there is expected and matches every other late-week-only
recorder already in the registry). Remaining `!!` MISSING: only
`tiebreaker_low_side_shade` and `total_conditioned_key_number_lattice_v1`,
which are unrelated to this bug (see below).

**Item 4 -- publish-predictions on an already-locked week (read only, not
run).** `orchestrate_publish_predictions` always calls
`publish_active_predictions` first, which re-renders the card/tiebreaker.json
from the stored **active** served forecast (not a re-fit) -- it does not
recompute picks. Read `record_tiebreaker_shade_decisions`
(`src/nfl_ats/tiebreaker_shade_prospective.py:93`) and
`record_total_conditioned_lattice_decisions`
(`src/nfl_ats/total_conditioned_lattice_challenger.py:140`): both are
idempotent by design -- `if recorded_week.any() and not replace_week: return
{"recorded": 0, "already_recorded": 1, ...}` with no overwrite, and even with
`--replace-week` they refuse to touch a game that already kicked off
(`left_post_kickoff` short-circuits to a skip). Since both ledgers currently
have **zero** Week 3 rows (confirmed by lockday_verify's `!!` status above),
`--replace-week` is not needed at all -- there is nothing to replace, only a
first write. **Safe command for root:** `nfl-ats publish-predictions
--record-decisions` (bare, no `--replace-week`, no `--record-from-forecast`
needed since the active forecast is still 2026 week 3 -- confirmed via
`plan_refresh`'s own `_resolve_active_forecast_season_week` resolving to
season=2026/week=3 all session). This will re-publish the same served Week 3
card content (unchanged picks, fresh `generated_at_utc`) and, in the same
pass, write first-time rows for both challengers for any Week 3 game still
before its deadline. Not run this session per instructions.

## Tried

This session: reproduced the builder directly (no `--record-decisions`),
isolated the exact filter rows via three scratch diagnostic scripts (pandas
groupby of `provider`/`bookmaker_key`/timestamp columns) -- confirmed 394
refused rows == exactly the free-source provider row count for Week 3.
Applied the two-line fix to `sharp_book_movement_features.py`. Re-ran repro to
confirm 16 Week 3 rows now build. Ran `ruff check` and `mypy` on
`sharp_book_movement_features.py` (clean). Edited
`scripts/lockday_verify.py` (imports + 6 `DEDICATED_LEDGERS` entries), ran
`ruff check` (clean) and `mypy` (only pre-existing repo-wide
`import-untyped` noise, no new errors). Ran `scripts/lockday_verify.py
--season 2026 --week 3` and confirmed the five previously-MISSING challengers
now resolve. Read (did not run) `publish_predictions.py` /
`tiebreaker_shade_prospective.py` / `total_conditioned_lattice_challenger.py`
to confirm idempotent, deadline-safe record semantics. Did not run
`pytest -k "refresh or lockday"` -- out of remaining tool budget this pass.

## Next (2026-09-25)

**All nine previously-missing Week 3 challenger records are resolved; no
recorder command was needed this pass.** Checked directly (`.tools/uv.exe run
--no-sync python scripts/lockday_verify.py --season 2026 --week 3
--run-summary artifacts/scheduled_locks/2026-week-03/weekly_summary.json`):
`59 recorded, 4 skipped, 0 MISSING, 0 pending wiring of 63 active`. Confirmed
against the ledger parquets directly too (`pandas.read_parquet` filtered to
season 2026 week 3):
- `total_conditioned_key_number_lattice_v1` -- `total_conditioned_lattice_decisions.parquet`
  now has 16/16 Week 3 rows (was 0 last session). `ok` in lockday_verify.
- `late_week_move_follow_refresh_decisions.parquet` and its five sibling
  challenger IDs (`late_week_move_follow_refresh_v1`,
  `late_week_leader_median_follow_v1`,
  `late_week_leader_median_follow_0_5_off_incumbent`,
  `late_week_leader_median_follow_flat_1_0_off_incumbent`,
  `late_week_follow_no_news_veto_off_incumbent`,
  `late_week_follow_no_sunday_blackout`) -- all 16 games present, 31-127 rows
  each, all `ok`.
- `consensus_movement_1_0_off_incumbent` -- 96 rows, `ok` (this was the ninth
  previously-missing ID).
- `tiebreaker_low_side_shade` -- still 0 Week 3 rows, but `--` (named,
  expected gate), not `!!` MISSING. `record_tiebreaker_shade_decisions`
  (`src/nfl_ats/tiebreaker_shade_prospective.py:122-127`) returns a
  permanent skip whenever the served tiebreaker's
  `total_low_side_shade_points <= -1.0`, i.e. whenever the served total
  already carries the standing `-1.0` shade (`TOTAL_LOW_SIDE_SHADE_POINTS`,
  `docs/tiebreaker.md`'s 2026-09-10 section, confirmed live at
  `docs/tiebreaker.md:163,324-360`) -- the served and shaded arms are then no
  longer a paired contrast. This is a structural gate, not a recording-window
  gap: no invocation of `publish-predictions --record-decisions` can produce
  a row while the shade stays wired into the served total. **No command run
  this pass** -- running it would not change this outcome (confirmed by
  reading the guard, not by running) and every other ledger was already full,
  so there was nothing left for the recorder to do. This matches the older
  Open-section question below, now answered: the answer is "not while the
  shade is served," which is a pre-existing design decision, not a bug.
- All Week 3 rows checked belong to games at or after their recorded
  deadlines with no post-kickoff writes observed; ATL at GB (locked Thursday
  game) shows no rows recorded after its kickoff in any of the three ledgers
  inspected -- consistent with each recorder's own kickoff/deadline guard,
  not separately re-verified by a new write this pass.

Prior items now stale/superseded (root's earlier `Next` list): items 1-2 below
are done (recorded by a scheduled pass, likely `refresh_last_call_fri_1330`
which last ran `2026-09-25T13:32:07-04:00` per `data/scheduler_state.json`,
before this session started) -- no action was needed or taken here.

Remaining for root:
1. `git status` still shows this session's (and prior sessions') edits
   uncommitted (`src/nfl_ats/sharp_book_movement_features.py`,
   `scripts/lockday_verify.py`, plus many pre-existing unrelated modified
   files) -- root reviews and commits per its own workflow. Not touched this
   pass (read-only recorder-coverage check only).

## Open

Whether the free-source providers should eventually populate a real
`bookmaker_last_update_utc` (making the `capture_observed_utc` escape
unnecessary) is a data-ingestion question for whoever owns the free-source
capture scripts, not decided here -- the fix accepts self-timed quotes as a
valid point-in-time basis without weakening the cutoff/deadline guards.
`spread_explorer.py:137-149`'s discrete-vs-smooth reproduction check and
whether `tiebreaker_low_side_shade` can ever pair while the shade is served
are unrelated, older open items from a prior session -- still open, not
touched this session.
