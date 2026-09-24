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
