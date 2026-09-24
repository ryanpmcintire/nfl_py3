# publish-predictions preservation (late-week section, deactivated recorder, tiebreak gate)

## Goal

Fix three bugs surfaced by root running `nfl-ats publish-predictions
--record-decisions` on 2026 Week 3 (2026-09-24 ~16:30 ET):
1. Republish wipes the `<!-- LATE_WEEK_REFRESH:START -->...:END -->` section.
2. `backup_qb_fade_overlay`'s recorder raises (registered
   `DEACTIVATED_STRUCTURAL_NO_OP`) instead of skipping, and isn't tracked in
   `PUBLISH_CHALLENGER_RESULT_KEYS`.
3. `tiebreaker_low_side_shade` records 0 rows with no named gate in
   `lockday_verify`.

Constraints: no code comments/docstrings in .py; no new tests (edit existing
only if legitimately broken); never `cd` in shell calls; never run
`--record-decisions` for real -- verify via pure function calls on scratch
copies. No commits/pushes.

## State — all three fixes implemented in code; NOT YET VERIFIED (ruff/mypy/pytest not run this session; hit 50-tool-call cap mid-verification)

### Item 1 — DONE (code written, not yet run/verified)

`src/nfl_ats/publishing.py`:
- Import block (~line 53) now imports `LATE_WEEK_REFRESH_START,
  LATE_WEEK_REFRESH_END` from `nfl_ats.pick_refresh` alongside
  `load_pick_revisions, served_best_pick`.
- Immediately before the `atomic_text(detail, destination)` call (was line
  660, now a bit later after the inserted block): if `destination.is_file()`,
  read its existing text; if it contains the H1 marker
  `f"# NFL ATS predictions: {metadata['season']} Week {metadata['week']}"`
  (i.e. same season/week as about to be published) AND exactly one
  `LATE_WEEK_REFRESH_START`/`END` pair, slice out that block verbatim and
  append it to `detail` before writing. A different season/week (new H1
  text) means the block is silently dropped -- correct per spec ("guard
  against carrying a stale block into a new week").

### Item 2 — DONE (code written, not yet run/verified)

`src/nfl_ats/cli_commands/publishing.py`:
- Added imports: `from nfl_ats.data import DataContractError` and
  `from nfl_ats.prospective_scoring import ACTIVE_CHALLENGER_STATUS,
  find_challenger`.
- Added `"backup_qb_fade_overlay": "backup_qb_fade_challenger_ledger"` to
  `PUBLISH_CHALLENGER_RESULT_KEYS`.
- Added `reclassify_inactive_challengers(result, result_keys,
  artifacts_root)` right after `collect_failed_recorders` (generic: walks
  every `(challenger_id, result_key)` pair, and for any entry with an
  `"error"`, looks up the real registry status via `find_challenger`
  (swallowing `FileNotFoundError, DataContractError, ValueError, KeyError` as
  "leave alone"); if status != `ACTIVE_CHALLENGER_STATUS`, replaces the
  entry with `{"recorded": 0, "skipped": True, "reason": f"challenger
  registered as {status!r}; only {ACTIVE_CHALLENGER_STATUS} challengers have
  picks recorded"}` — matches the shape already used elsewhere in this file
  for `--record-decisions`-not-passed skips.
- Called once before each existing `collect_failed_recorders(...)` call: in
  `orchestrate_publish_predictions` (with `PUBLISH_CHALLENGER_RESULT_KEYS`)
  and in `_cmd_refresh_picks` (with `REFRESH_CHALLENGER_RESULT_KEYS`).

**Test edits in `tests/test_cli.py` (legitimate, required by the above):**
- `test_publish_challenger_result_map_covers_live_active_registry` (~line
  676): **measured** that the real registry has 5 non-`ACTIVE_PROSPECTIVE`
  challengers whose `weekly_recording_command` mentions
  `nfl-ats publish-predictions --record-decisions` (`backup_qb_fade_overlay`
  plus 4 `SUPERSEDED_BY_PROMOTION` ids) — none previously in
  `PUBLISH_CHALLENGER_RESULT_KEYS`; adding `backup_qb_fade_overlay` makes the
  old strict `==` assertion false. Changed to `expected <=
  set(cli.PUBLISH_CHALLENGER_RESULT_KEYS)` (every live-active challenger must
  still be covered; extra entries like the now-tracked deactivated one are
  allowed) and kept the no-duplicate-values check but against the dict's own
  length instead of `len(expected)`.
- `test_publish_predictions_records_cleanly_when_a_challenger_is_deactivated`
  (~line 1244): added a scratch `challengers.json` registry file written
  under `tmp_path / "artifacts" / "prospective"` (before `fake_publish` is
  defined) with two entries (`backup_qb_fade_overlay`,
  `gaussian_mean_mapping_incumbent`, both `DEACTIVATED_STRUCTURAL_NO_OP`) so
  `find_challenger` actually resolves inside the test's isolated
  `NFL_ATS_ARTIFACTS_DIR` and `reclassify_inactive_challengers` fires.
  Updated assertions (~line 1340+): now checks
  `payload["backup_qb_fade_challenger_ledger"]["skipped"] is True` and
  `"DEACTIVATED_STRUCTURAL_NO_OP" in [...]["reason"]` (was `["error"]`
  before); same swap for the `mean_refuses` branch on
  `gaussian_mean_mapping_incumbent_challenger_ledger`.
- **Not checked yet:** whether `failed_recorders` (if asserted anywhere else
  in this test or others) still expects `backup_qb_fade_overlay` to appear
  there — it now won't, since its entry has `skipped` not `error`. Grep
  `tests/test_cli.py` for `failed_recorders` before running pytest.
- Exit-255 mechanism (root's literal observation): **explicitly deprioritized
  by the orchestrator this round** — root now believes it came from the
  PowerShell pipeline, not the command. Not chased further.

### Item 3 — DONE (lockday_verify.py only; recorder already correct)

- **Measured (read-only):** `record_tiebreaker_shade_decisions`
  (`src/nfl_ats/tiebreaker_shade_prospective.py:93`, gate at lines 122-127)
  **already** returns `skip("the served tiebreaker total already carries the
  low-side shade...")` whenever `total_low_side_shade_points <= -1.0 +
  1e-9`. `TOTAL_LOW_SIDE_SHADE_POINTS = -1.0` is a fixed constant
  (`tiebreaker.py:49`) always applied to the served guess, so this gate now
  **always** fires — the recorder-side fix the task asked for was already
  shipped in an earlier session; no change needed there.
- Real bug was purely in `scripts/lockday_verify.py`: its classifier for
  `DEDICATED_LEDGERS` entries (~line 505) only shows `"skipped"` when the
  static dict has a `legitimately_empty` key; `tiebreaker_low_side_shade`'s
  entry (~line 73-79) didn't have one, so it fell through to `"MISSING",
  "no rows and no gate explaining why"`.
- Fix: added `"legitimately_empty": "the served tiebreaker total permanently
  carries the -1.0 low-side shade (TOTAL_LOW_SIDE_SHADE_POINTS in
  tiebreaker.py), so this challenger's served and shaded arms can never pair
  -- see docs/tiebreaker.md's 2026-09-10 section"` to that dict entry,
  following the exact pattern of the `specialist_absence_fade_refresh_v1`
  entry a few lines below it.

## Tried

- Read `pick_refresh.py` markers/splice logic and `publishing.py`'s
  `publish_active_predictions` in full (again, to re-confirm before editing).
- Read `cli_commands/publishing.py`'s `PUBLISH_CHALLENGER_RESULT_KEYS`,
  `REFRESH_CHALLENGER_RESULT_KEYS`, `collect_failed_recorders`, both call
  sites, and the backup_qb_fade call site (confirmed result key name
  `backup_qb_fade_challenger_ledger`).
- Read `prospective_scoring.py`'s `find_challenger`/`load_challenger_registry`
  /`ACTIVE_CHALLENGER_STATUS` to get exact signatures and exception types.
- Ran a read-only `python -c` snippet against the real
  `artifacts/prospective/challengers.json` to enumerate every
  non-`ACTIVE_PROSPECTIVE` challenger whose `weekly_recording_command`
  mentions the publish command (5 found) — this is what justified the
  `test_publish_challenger_result_map_covers_live_active_registry` rewrite.
- Read `tiebreaker_shade_prospective.py` in full through line 160 and
  `tiebreaker.py` around `TOTAL_LOW_SIDE_SHADE_POINTS` — confirmed the
  recorder-side gate already exists and is permanent.
- Read `scripts/lockday_verify.py`'s `DEDICATED_LEDGERS` dict and its
  classifier loop (~lines 415-510) to confirm the exact missing-key
  mechanism.
- Made all edits listed above via `Edit`. **Did not run ruff, mypy, or
  pytest yet this round** — hit the 50-tool-call cap while locating
  `_publish_with_fresh_empty_arrest` in `tests/test_publishing.py` to build a
  scratch verification script for item 1.

## Next

1. **Verification commands (none run yet this round):**
   - `.tools\uv.exe run --no-sync ruff format src/nfl_ats/publishing.py
     src/nfl_ats/cli_commands/publishing.py scripts/lockday_verify.py
     tests/test_cli.py`
   - `.tools\uv.exe run --no-sync ruff check` (same files)
   - `.tools\uv.exe run --no-sync mypy src`
   - `.tools\uv.exe run --no-sync pytest -k "publish or card or cli or
     lockday or tiebreak"`
   - Before running the CLI test, grep `tests/test_cli.py` for
     `failed_recorders` to check the deactivated-challenger test doesn't
     also assert `backup_qb_fade_overlay` appears in `failed_recorders`
     (it now won't, by design — its entry is `skipped`, not `error`).
2. **Item 1 direct-function verification (not done yet):** in
   `tests/test_publishing.py`, `_write_active_publication_fixture(root)`
   (~line 43) plus a helper named `_publish_with_fresh_empty_arrest` (its
   definition was not located before the cap hit — grep
   `def _publish_with_fresh_empty_arrest` in that file) build a full
   synthetic fixture usable outside pytest (any `Path` works, not just
   `tmp_path`). Plan: under `tests/scratch/` (gitignored), call
   `_write_active_publication_fixture(root)`, publish once via
   `_publish_with_fresh_empty_arrest(root, destination=..., readme_path=...,
   published_at=instant)`, hand-inject a
   `<!-- LATE_WEEK_REFRESH:START -->...END -->` block into the destination
   file, republish with the **same** `published_at`/season/week and assert
   the block survives verbatim; then edit the fixture's
   `forecast/metadata.json` + `active_ats_model.json` week fields to a new
   week and republish again, asserting the block is now absent.
3. If pytest surfaces any other assertion tied to the two changed tests
   (e.g. `failed_recorders` list contents, or other tests reading
   `PUBLISH_CHALLENGER_RESULT_KEYS` by exact set), fix those too — same
   "legitimate change" allowance.
4. Once verified, this lane's three fixes are complete; no owner-facing
   dashboard/publish work is implied (this is harness/recorder-plumbing
   maintenance, not a served-card change), so no publish-board/push step is
   needed per AGENTS.md ("read-only audits, harness maintenance... do not
   trigger operational jobs").

## Open

- Item 1: guard uses the H1 title text as the season/week fingerprint
  (simplest option from the prior plan); a re-render-from-ledger alternative
  was explicitly deferred as a bigger lift and is still not needed.
- Item 2: exit-255 mechanism is now explicitly out of scope per the
  orchestrator (root traced it to the PowerShell pipeline).
- Item 3: no further code changes anticipated; only verification remains.
- Nothing has been run (ruff/mypy/pytest) — treat all three fixes as
  unverified until the Next-section commands are executed.
