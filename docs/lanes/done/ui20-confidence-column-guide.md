# UI-20: Confidence column gets a tooltip and a column-guide entry

## Goal

Ship one visible reader-facing dashboard improvement from ROADMAP.md UI-20's
queue, verified not already shipped, regenerated locally, diffed, and tested.

## State

Code change made and lint/type clean. Test run and `publish-board`
regeneration/diff and the ROADMAP UI-20 dated sentence are NOT done yet (ran
out of tool-call budget mid-verification). No commit/push made (root's job).

Change: `src/nfl_ats/board_terminal.py`, in the This Week board table builder
(the function containing `market_help`/`probability_help`/`flip_help` locals,
~line 909-937). Added a `confidence_help` string ("Slight, Lean or Strong: how
firmly the cover chance clears this pick's own three-way split. The note below
the table gives the exact cutoffs.") and:
- wrapped the `<th>Confidence</th>` header in an `<abbr title=...>` like the
  three other data columns already have (Books now / Cover chance / Flips at)
- added a matching `<dt>Confidence</dt><dd>...</dd>` row to the collapsible
  "Column guide" `<dl>` (previously had only 3 of 4 data-column entries)

Reader-visible effect: hovering "Confidence" in the This Week board header, or
opening "Column guide", now explains what Slight/Lean/Strong means and points
to the existing legend paragraph below the table for the exact cutoff
percentages (that legend text itself — "Slight, Lean and Strong split the
chances into thirds..." — was already shipped 2026-09-13, commit 6631aa0; it
renders today in `docs/index.html` under `class="policy-note pick-lock-note"`,
confirmed via `grep -c "thirds of the card" docs/index.html` = 1 per table).

## Tried

- Read ROADMAP.md UI-20 row in full (line 768, very long). Its queue items
  (a)-(h) plus the 2026-09-12 "Survey candidates left: a legend line for the
  confidence words" are ALL already shipped in code and rendering on the
  current `docs/index.html`/`history.html` — verified directly:
  - confidence-words legend: `git log -S "_confidence_legend_text"` ->
    commit 6631aa0 (2026-09-13); text present in rendered `docs/index.html`.
  - history opener-vs-close side by side: shipped 2026-09-05 lane Y /
    reconfirmed 2026-09-11 lane 2.
  - (a)/(b)/(c)/(f)/(g) explicitly noted shipped in the row's own history.
  - (d) season-record strip: `grep -n "season-record-strip"
    src/nfl_ats/board_terminal.py` -> line 1488, exists.
  - (e) challenger/rival-rules Week 1 previews: fixed 2026-09-11 lane 3
    (policy-id filter bug), tests exist
    (`test_rival_rules_section_*` in `tests/test_board_terminal.py`).
  So the task's own suggested examples were dead ends — do NOT reimplement
  either; found a genuinely unshipped gap instead: the "Confidence" board
  column was the only one of the four data columns with no `<abbr>` tooltip
  and no `<dt>` entry in the Column guide, confirmed via
  `grep -o '<dt>[^<]*</dt>' docs/index.html` (only Books now / Cover chance /
  Flips at, 3 occurrences each, no Confidence) and
  `grep -o '<th>Confidence</th>' docs/index.html` (plain, no abbr, 3 occurrences).
- Checked no existing test pins the exact header/column-guide markup I
  changed: `grep` in `tests/test_board_terminal.py` for
  `market_help|probability_help|flip_help|column_guide|<th>Confidence` ->
  no matches. The one hit for `<th>Confidence` project-wide
  (`tests/test_site_theme_invariants.py:44`) is the cover-curve table twin in
  `dashboard/viz.py` — a different table, unaffected.
- `ruff format --check src/nfl_ats/board_terminal.py` -> "1 file already
  formatted". `ruff check src/nfl_ats/board_terminal.py` -> "All checks
  passed!". `mypy src/nfl_ats/board_terminal.py` -> "Success: no issues found
  in 1 source file".
- Attempted
  `pytest tests/test_board_terminal.py tests/test_board_content.py tests/test_public_board.py tests/test_board_humanised.py -q`
  twice: both times exited 5 ("no tests collected") with only a
  `PytestCacheWarning: could not create cache path ...\.pytest_cache\v\cache:
  [WinError 5] Access is denied` in the output — looks like default `-n auto`
  xdist plus a cache-permission issue in this sandbox is eating the run
  silently rather than a real collection failure: `pytest
  tests/test_board_terminal.py -p no:cacheprovider --collect-only -q`
  DID work and found 88 tests. Was mid-retry with
  `-p no:cacheprovider -p no:xdist -q` (no output captured yet) when the
  tool-call cap stopped the session.

## Next

1. Re-run in a fresh shell:
   `/f/Repos/nfl_py3/.tools/uv.exe run --no-sync pytest tests/test_board_terminal.py tests/test_board_content.py tests/test_public_board.py tests/test_board_humanised.py -p no:cacheprovider -p no:xdist -q`
   and confirm all pass (expect ~88+ in test_board_terminal.py alone, no
   failures related to the column-guide/header change).
2. Regenerate locally: check `nfl-ats publish-board --help` for a
   local/dry/site-destination flag (prior UI-20 lanes used
   `--site-destination <scratch>`), run it, and diff the rendered
   `index.html` (and any other page using the same board table renderer)
   against `docs/index.html` to confirm only the Confidence `<th>`/`<dl>`
   change landed (plus whatever routine timestamp/number drift is expected).
3. Update ROADMAP.md UI-20 row with one dated sentence (2026-09-23) noting:
   the Confidence column tooltip/guide-entry shipped, AND that the
   2026-09-12 "Survey candidates left: a legend line for the confidence
   words" item was found already shipped (2026-09-13, commit 6631aa0) so it
   can stop being listed as an open survey candidate.
4. Root: run `publish-board` for real, review the rendered-page diff, and
   push per AGENTS.md "Dashboard and publication" (not this subagent's job).

## Open

- The four dashboard pages (`docs/index.html`, `history.html`, `model.html`,
  `findings.html`) were already modified/uncommitted at session start (small
  2-4 line diffs each, pre-existing, unrelated to this lane — likely leftover
  from a prior unpublished `publish-board` run). Not touched or investigated
  further here; root should check before publishing so this lane's diff isn't
  conflated with that pre-existing one.
- Untracked `scripts/opener_error_transfer_unit2.py` and
  `scripts/players_on_field_rating_unit2.py` also present at session start;
  out of scope for this lane (scripts/ is off-limits per task instructions).
