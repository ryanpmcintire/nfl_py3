# lead61-second-half-channel

## Goal

ROADMAP LEAD-61 second-half-line channel: measure the `half_line_2h_underdog_refresh_v1`
ledger since the Odds API cancellation (2026-09-19), find whether any current free
source carries second-half spreads, and stop the recorder from silently reusing a
stale snapshot. If no free source exists, give a named skip gate and disable the
dead halves scheduler jobs.

## State (measured this session, 2026-09-24)

- Ledger `artifacts/prospective/half_line_refresh_decisions.parquet`: 507 rows,
  season 2026 week1=202, week2=209, week3=96. Every week-3 row has
  `overlay_status == "game_absent_from_matched_quotes"`, `matched_book_count == 0`,
  and reuses `halves_snapshot_id = "20260915T160610Z-halves"`
  (`halves_observed_at_utc = 2026-09-15T16:06:10Z`) — 9 days stale at measurement
  time. Root cause: `_cmd_odds_ingest_halves`
  (`src/nfl_ats/cli_commands/market.py:79`) hard-requires `THE_ODDS_API_KEY`
  (cancelled 2026-09-19), so every `odds_*_halves` scheduler job fails and the
  halves snapshot on disk never advances; `freshest_snapshot_before` in
  `src/nfl_ats/half_line_refresh_overlay.py` had no staleness bound, so it kept
  silently reusing the dead Sept-15 snapshot and reporting a false
  "no matched quote" per game instead of naming the real cause.
- Source check (measured from raw captures, no live fetch): Bovada
  (`data/market/raw/20260923T211819Z/response.json`, provider `bovada_public_nfl`,
  the `marketFilterId=def&preMatchOnly=true` endpoint scripts/capture_bovada_private.py
  hits): every market's `period` is `('Game', True)` only — zero half/quarter
  markets present in the payload at all. The Odds Gap
  (`data/market/raw/20260924T145032Z-odds-gap-private/response.json`,
  `scripts/capture_odds_gap_private.py`): each game object has a `period_data` key,
  but it is `null` for all 32 NFL games and all games of any sport in the capture —
  the public/free tier the script scrapes does not populate it.
  **Conclusion: no currently integrated free source carries second-half lines.**
  Wiring a free 2H source is not possible with today's capture code; a
  named-skip-gate fix (not a new source) is the right unit.
- Code fix applied in `src/nfl_ats/half_line_refresh_overlay.py` (not on the
  forbidden-file list):
  - Added `OVERLAY_STATUS_NO_CURRENT_HALVES_SOURCE =
    "no_second_half_source_since_paid_feed_cancelled"` and
    `HALVES_SNAPSHOT_STALE_BOUND = pd.Timedelta(days=7)`.
  - In `build_half_line_refresh_rows`, right after the freshest halves snapshot is
    located (and before `matched_book_disagreement` runs), added: if
    `pass_instant - halves.observed_at_utc > HALVES_SNAPSHOT_STALE_BOUND`, return
    the empty frame with `skipped: True`, `reason:
    OVERLAY_STATUS_NO_CURRENT_HALVES_SOURCE`, a `reason_detail` sentence, and the
    stale snapshot's id/observed time — instead of falling through to
    per-game `game_absent_from_matched_quotes` silence.
  - Both new names added to `__all__`.
  - No other file touched. Historical ledger rows (507) were NOT rewritten —
    only future passes get the new named skip.

## Tried

- Ran a **read-only preview** (no `--record-decisions`, allowed):
  `.tools/uv.exe run --no-sync python -m nfl_ats.cli refresh-picks --season 2026
  --week 3 > F:\Repos\nfl_py3\_tmp_refresh_w3.json 2>&1` (447 lines). Tool-call
  cap hit before the `half_line_refresh_overlay` block in that output was
  inspected — **not yet confirmed** that the new skip reason actually appears in
  the live dry-run JSON. `_tmp_refresh_w3.json` is an untracked scratch file at
  the repo root; delete it (not part of the fix, would look like repo clutter if
  committed).

## Next

1. Re-run or grep the already-produced `F:\Repos\nfl_py3\_tmp_refresh_w3.json`
   (or rerun the same preview command) for the `half_line_refresh_overlay` block
   and confirm `reason == "no_second_half_source_since_paid_feed_cancelled"` and
   `skipped: true` for week 3 (2026-09-24 pass, halves still frozen at 2026-09-15).
   Delete `_tmp_refresh_w3.json` afterward.
2. Disable the dead halves scheduler jobs in `scripts/capture_scheduler.py`
   (`odds_tue_open_halves` ~L618, `odds_sat_halves` ~L633, `odds_wed_opener_halves`
   ~L648, `odds_thu_tnf_halves` ~L664, `odds_sun_close_halves` ~L679,
   `odds_mon_mnf_halves` ~L694 — line numbers approximate, grep `_halves"` in
   that file). Each `Job(...)` call's positional args were not fully inspected
   this session — **first read the `Job` dataclass definition** in that same file
   (or wherever it's imported from) to find the correct enable/disable field
   (do not guess; the second positional arg after `_cli(...)` may or may not be
   an `enabled` flag — confirm before changing). Set them disabled and extend
   each job's description with the reason: "Disabled 2026-09-24: no free source
   supplies second-half lines since the paid Odds API feed was cancelled
   2026-09-19 (measured: Bovada default endpoint returns Game-period markets
   only, Odds Gap's period_data is null); see
   docs/lanes/lead61-second-half-channel.md." Keep the Job entries (history/
   rationale) rather than deleting them, matching repo style elsewhere.
3. Run verification: `.tools/uv.exe run --no-sync ruff check
   src/nfl_ats/half_line_refresh_overlay.py scripts/capture_scheduler.py`,
   `.tools/uv.exe run --no-sync mypy src/nfl_ats/half_line_refresh_overlay.py`,
   and `.tools/uv.exe run --no-sync pytest -k half` (repo-wide, no new tests
   added per moratorium). Fix any failures the two edits caused.
4. Re-run the same read-only `refresh-picks --season 2026 --week 3` preview once
   more after the scheduler edit to reconfirm nothing regressed (still no
   `--record-decisions`).
5. Report back to the orchestrator; do not commit/push/publish (out of scope for
   this subagent per task constraints).

## Open

- `HALVES_SNAPSHOT_STALE_BOUND = 7 days` was chosen because the halves jobs run
  on a weekly cadence (Tue/Wed/Thu/Sat/Sun/Mon); any successful weekly capture
  would land inside 7 days. Not owner-confirmed; reasonable but arbitrary — flag
  if a reviewer wants a different bound.
- If the owner later finds/adds a free 2H source, wire it into
  `market_data_halves.py` / the `odds-ingest-halves` CLI path (currently hard-
  wired to The Odds API only) as a separate follow-up; out of scope here.

## Update 2026-11-03 (Bovada period probe)

- Measured: `capture_odds_tape._fetch()` on the `marketFilterId=def` URL (2026-11-03T13:00:20Z, HTTP ok, 15 events) returned only `Game` period markets (Moneyline, Point Spread, Total).
- Measured: per-event endpoint `services/sports/event/coupon/events/A/description<event link>?lang=en` (JAX @ BAL, 2026-11-03T13:00:28Z, 220 KB) lists period descriptions Game 112, First Half 21, 1st Quarter 16, Regulation Time 2. Pregame First Half Point Spread (main 1H line plus alternates) and 1st Quarter Point Spread exist. No "2nd Half" period exists pregame; no 2H spread market.
- Not tested: live halftime listing (no live game Tuesday). Next live check: Thursday 2026-11-05 JAX @ BAL halftime, one per-event request, look for a Second Half period.
- No 2H capture built; no Job entries written (no 2H market to time). Block guard untouched, no marker written.
- Inferred, not built: a 2H line could be approximated as full-game spread minus 1H spread from the pregame per-event feed; it is a derived quantity, not a quoted 2H market, and needs its own declaration before use.

## Update 2026-11-03 (live 2H probe built)

- Script: `scripts/probe_bovada_live_periods.py` (reuses `capture_odds_tape._fetch`, now `_fetch(url=URL)`, its block marker `data/market/bovada_public_nfl_blocked.json`, `TapeStop`, `_iso_ms`; no second HTTP client). Each poll: one list request (the `def` URL without `preMatchOnly`, so live events appear), then one per-event request for every NFL event that is live or in the window. Writes `data/raw/odds_2h_probe/<UTC stamp>/probe.jsonl` + `manifest.json`; rows carry request/response/list timestamps, provider `lastModified`, `live`, period-to-market-count map, and any market whose period or description mentions 2nd half / second half / 2H. A block writes the marker and stops cleanly (exit 1); an existing marker exits 2 without a request.
- Window derivation (measured 2026-11-03, nflreadpy `load_pbp` 2018-2025, `time_of_day` per play vs kickoff from `data/raw/20261027T220410Z/schedules.parquet` gameday+gametime in ET; filtered 5<halftime<60 min and 60<first-half-end<150 min): minutes after kickoff of the last 1H play, prime time (Thu, Mon, Sun night; n=440) p05 73.3, median 85.7, p95 98.6, MAD 5.4 (Thursday only n=141: 74.4 / 86.4 / 101.4). Halftime length (last 1H play to first 2H play): prime time p05 13.8, median 16.9, p95 37.6 min. Window = kickoff+73 to kickoff+136 min (p05 first-half end to p95 first-half end + p95 halftime). For JAX @ BAL (20:15 ET Thu) that is 21:28 to 22:31 ET.
- Interval: SAMPLES_PER_HALFTIME = 2 (same two-samples rule as the tape's SAMPLES_PER_LAG) over the shortest halftime p05 13.8 min = 414 s, the default. A listed 2H market that exists at all lives at least the shortest halftime, so this yields at least two looks inside any halftime. Each look costs one list plus one 220 KB event request, so it is deliberately not tightened to the 17 s provider lag.
- Verified (13:04Z, `--repeat-minutes 0.5 --interval-seconds 60 --pregame-hours 64`): 1 poll, 1 row, JAX @ BAL pregame, periods {1st Quarter 16, First Half 21, Game 112, Regulation Time 2}, second_half_markets [], status complete, no marker. Output `data/raw/odds_2h_probe/20261103T130428Z/`.
- Scheduler Job for the root to add to `scripts/capture_scheduler.py` (times are ET; 21:25 starts 3 min before the window, 70 min runs to 22:35; one-off, remove after 2026-11-05):

```
Job(
    "odds_2h_probe_thu_20261105",
    "thu",
    "21:25",
    90,
    [
        str(UV),
        "run",
        "--no-sync",
        "python",
        str(REPO / "scripts" / "probe_bovada_live_periods.py"),
        "--repeat-minutes",
        "70",
    ],
    True,
    "One-off LEAD-61 probe: does Bovada list a 2nd Half period at halftime of JAX at BAL (kickoff 20:15 ET); window and interval derived in docs/lanes/done/lead61-second-half-channel.md.",
    season_guarded=False,
    added_on="2026-11-03",
),
```
- Next: after Thursday, read `probe.jsonl` for rows with `live: true` and non-empty `second_half_markets` or a period key containing Half other than First Half. Then record the verdict here.
