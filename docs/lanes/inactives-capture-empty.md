# Inactives capture returns 0 rows; source-freshness reported it complete

## Goal
Fix why every `data/players/inactives/<ts>/inactives.parquet` (52 dirs, incl.
Sun 2026-09-13 and 2026-09-20 game-day captures after real inactives were
posted) has 0 rows, and stop source-freshness from calling that "complete".

## State
Fixed. Root cause was two bugs, both fixed and verified this session.

1. **NFL.com's primary source is now permanently broken.** `measured`, live
   fetch today (`GET https://www.nfl.com/inactives/`): the page is a static
   promo shell (`class="blank-placeholder"` / "Custom Promo" CTA) containing
   the literal string `PRIMARY_PLACEHOLDER_TEXT` unconditionally, with zero
   `nfl-o-inactive*` class occurrences anywhere. Confirmed the same in the
   saved `primary.html` of the 2026-09-20 game-day captures
   (`data/players/inactives/20260920T153527Z/primary.html`,
   `.../20260920T184005Z/primary.html`) — the site was redesigned and no
   longer server-renders real tables, so the parser was never going to find
   rows there again, on any week.
2. **The fallback (RotoWire) was never attempted.** `src/nfl_ats/inactives_capture.py`
   had `need_fallback = not rows and empty_reason != EMPTY_REASON_OFFSEASON_PLACEHOLDER`
   — since bug 1 makes NFL.com show that placeholder text on every single
   capture regardless of season state, this permanently short-circuited the
   fallback. Fixed to `need_fallback = not rows`
   (`src/nfl_ats/inactives_capture.py:339`). Verified via
   `.tools\uv.exe run --no-sync python scripts\capture_scheduler.py --run-job inactives_sun_early --dry`
   → `data/players/inactives/20260924T184617Z/manifest.json` now has a
   populated `"fallback"` block (previously always `null` when primary
   showed the placeholder text).
3. **Source freshness never looked at `row_count`.** `_evaluate_one` in
   `src/nfl_ats/source_freshness_policy.py` only checked whether a snapshot
   *directory* existed within budget, so an empty snapshot was reported
   `complete`. Added `SourceObservation.row_count`, a new
   `newest_snapshot_manifest_row_count()` reader in
   `src/nfl_ats/capture_freshness.py`, wired it for the `"inactives"` source
   id in `observe_from_disk`, and added a fail-closed branch in
   `_evaluate_one` (`source_freshness_policy.py:520-542`): 0 rows before the
   week's T-90 due instant stays `not_due` (unchanged, legitimate); 0 rows
   at/after due now returns `policy.on_absent` (`degraded`), never
   `complete`.

## Tried
- Live-fetched `nfl.com/inactives/` and `rotowire.com/football/inactives.php`
  this session (`measured`) — NFL.com's page is a dead placeholder shell,
  RotoWire currently also shows its own real "not posted yet" placeholder
  (expected: today is 2026-09-24, ATL at GB kicks off 20:15 ET tonight, T-90
  not reached).
- Could not re-fetch real historical Week 2 (2026-09-20) inactives from a
  live source — both sites only carry the current week; the 2026-09-20
  captures' saved `primary.html` was inspected instead to confirm the root
  cause held on that actual game day.
- Ran the real daemon argv:
  `.tools\uv.exe run --no-sync python scripts\capture_scheduler.py --run-job inactives_sun_early --dry`
  → wrote a real snapshot `data/players/inactives/20260924T184617Z`
  (`ok=True`, `row_count=0`, `fallback` populated, `season=2026 week=3`) —
  correct for right now (Week 3 inactives not due).
- `report_for_publication(...)` right now: `inactives` state = `not_due`,
  `due_at_utc=2026-09-24T22:45:00+00:00` (T-90 of tonight's 20:15 ET
  kickoff) — correct, not falsely `complete`.
- Simulated `now` 6h later with the same 0-row observation via
  `evaluate_sources(obs, now, first_kickoff=...)`: state flips to
  `degraded`, reason "latest snapshot has 0 rows past the T-90 window ...
  not a complete one" — confirms the fail-closed branch fires once a window
  has genuinely opened.
- `ruff format` / `ruff check` on the 3 touched files: clean.
  `mypy src/nfl_ats/inactives_capture.py src/nfl_ats/capture_freshness.py
  src/nfl_ats/source_freshness_policy.py`: `Success: no issues found in 3
  source files`.
  `pytest -k inactives`: 23 passed. `pytest -k "source_freshness or
  capture_freshness"`: 70 passed (2 pre-existing, unrelated
  `pbp08_protection_mismatch_tilt_overlay` warnings).

## Next
- Nothing pending for this bug. The next real-world proof is passive: watch
  the first `inactives_sun_early`/`inactives_thu_*` capture whose T-90 window
  has actually opened (e.g. tonight's ATL at GB game, T-90 ≈ 18:45 ET) and
  confirm it either gets real RotoWire rows now that the fallback actually
  runs, or — if RotoWire's markup doesn't match the shared
  `_parse_shared_design_system` regexes either — correctly reports `degraded`
  instead of `complete`.
- If RotoWire also turns out to be structurally unparseable by the shared
  regex (never proven with real in-season data — `docs/inactives_channel.md`
  flagged this as inferred-by-analogy, not measured), the next fix is a
  RotoWire-specific parser. Out of scope for this bounded task; the
  fail-closed freshness fix means that failure mode will now surface as
  `degraded`, not hide as `complete`.
- `docs/lanes/lead64-friday-designations.md` step 5 named this exact bug as
  a blocker for its own "out" ground-truth join; it can now retry once real
  rows exist.

## Open
- None. Both bugs are fixed, minimally, with no test files added
  (moratorium respected) and no comments added to touched `.py` files.

## Root follow-up 2026-09-24 15:10 ET
- Measured: live `report_for_publication` at 15:08 ET reads inactives `not_due` (due 22:45 UTC), but the card written by the 15:00 refresh_thu still printed "inactives complete". The card's freshness line comes from a different evaluation path or instant; check it after the 19:15 refresh_thu_inactives_primetime run (expect complete with rows > 0, or degraded if RotoWire's markup does not parse).
- Scheduler restarted 15:03 ET (pid 28808) to load the BEST-PICK-LEDGER logging.

## RotoWire parser proven and fixed 2026-09-24 ~19:45 ET
`_parse_shared_design_system` (NFL.com's `d3-o-table`/`nfl-o-inactive-report__unit`
markup) was never going to match RotoWire — confirmed **measured**: RotoWire uses
none of those NFL.com Deacon3 class names anywhere. Wayback (`web.archive.org`)
was globally down for most of this session (CDX API and most direct snapshot
fetches returned an "Internet Archive: Temporarily Offline" page); the
`https://archive.org/wayback/available?url=www.rotowire.com/...` lightweight API
still worked and found one real in-season snapshot:
`http://web.archive.org/web/20251116200055/https://www.rotowire.com/football/inactives.php`
(Week 11 2025, Sunday early-game window, saved to
`tests/scratch/rotowire/wayback_20251116.html`, gitignored). Real markup is a
`<div class="col border-r">` grid, one `<div class="bg-concrete border-tb">`
team header (full team name text + team-logo `<img>`, no table) per team,
followed by `<ul class="list is-small ...">` of `<li><span>POS</span><a>Player
Name</a></li>` — no status/reason column at all (every listed player is by
definition inactive).

Fixed in `src/nfl_ats/inactives_capture.py`: added `_parse_rotowire_grid`
(new regexes `_ROTOWIRE_TEAM_HEADER`/`_ROTOWIRE_TEAM_LIST`/`_ROTOWIRE_LIST_ITEM`/
`_ROTOWIRE_POSITION`/`_ROTOWIRE_PLAYER_LINK`, ~line 145) reusing the existing
`NICKNAME_TO_CODE`/`strip_html` imports — team code comes from the last word of
the full team name (e.g. "Los Angeles Rams" -> "rams" -> `LA`, matching the
repo's schema, not RotoWire's own `LAR` logo-code, which would have silently
broken the `_schedule_lookup` join). Wired into `run_capture`'s fallback branch
in place of `_parse_shared_design_system`. Also fixed a stale/misleading
manifest warning ("primary source parsed 0 rows ... guessed markup structure")
that previously fired unconditionally whenever the fallback found rows, even in
the normal case where primary legitimately showed its own placeholder; now
gated on `primary_html is not None and not primary_showed_placeholder`.

Verified this session (**measured**):
- `_parse_rotowire_grid` on the real Week 11 snapshot: 150 rows, 24 teams, 0
  warnings, correct codes (`ARI`, `LA`, `SF` for 49ers, etc.), position + name
  populated, `status="Inactive"`.
- `_parse_rotowire_grid` on the live current-week placeholder
  (`data/players/inactives/20260924T184617Z/fallback.html`, contains
  `FALLBACK_PLACEHOLDER_TEXT`): 0 rows, 0 warnings (as expected; `run_capture`
  never even calls the parser here since `fallback_showed_placeholder` short-
  circuits first).
- Full `run_capture(season=2025, week=11, slot="sun_early", ...)` with a fake
  fetch function returning the dead NFL.com placeholder for primary and the
  real Week 11 RotoWire HTML for fallback: `ok=True`, `row_count=150`,
  `source_used="fallback"`, `teams_seen` has 24 entries, `warnings=[]`.
- `ruff format` / `ruff check src/nfl_ats/inactives_capture.py`: clean (one
  `B905 zip() strict=` fix applied: `zip(headers, lists, strict=False)`, since
  a header/list count mismatch is a warned condition, not an error).
  `mypy src/nfl_ats/inactives_capture.py`: `Success: no issues found in 1
  source file`. `pytest -k inactives`: 23 passed, no regressions.
- Did not touch `source_freshness_policy.py` (out of scope per batch rule).

### Next
- Tonight's real capture (ATL at GB, T-90 ≈ 18:45 ET / 22:45 UTC) is the first
  live proof against RotoWire's *current*-week markup, not an archived one.
  Structure should match (same site template), but watch
  `data/players/inactives/<tonight's stamp>/manifest.json` for `row_count > 0`,
  `source_used="fallback"`, and empty `warnings`. If RotoWire changed their
  grid classes since Nov 2025, `_parse_rotowire_grid` will return 0 rows and
  `source_freshness_policy`'s fail-closed branch (already fixed, see above)
  reports `degraded` rather than falsely `complete` — check `empty_reason` in
  that manifest first before re-diagnosing from scratch.
- Fetched artifacts kept for reference under `tests/scratch/rotowire/`
  (gitignored): `wayback_20251116.html` (real populated page),
  `live_20260924.html` (today's live placeholder, JS-light static HTML with no
  table markup at all pre-T-90), `run_capture_out/` (this session's
  synthetic end-to-end run).
