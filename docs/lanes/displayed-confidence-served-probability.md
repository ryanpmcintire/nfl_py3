# Lane: displayed confidence -> served probability

## Goal
Audit finding 4 (docs/lanes/statistical-audit-2026-09-15.md): displayed cover
chance came from an in-sample 12-cell lookup in `displayed_confidence.py`
(`PSEUDO_OBSERVATIONS=20.0`, `PICK_SIDE_FLOOR` masking). Task: measure LOSO
calibration of (i) served `four_term_pick_probability_v1` probability of the
served side vs (ii) the 12-cell lookup rebuilt LOSO; if (i) wins, make it the
displayed number and delete the in-sample lookup/constants from the served
path; regenerate board/card locally (no publish); run ruff/mypy/pytest.

## State (mid-refactor, NOT yet verified to compile or run)

**Key discovery (read, verified in code, not yet re-run):** production already
bypasses the in-sample lookup whenever a pick-probability model is active.
`load_pick_probability_model_or_none` (`pick_probability.py:509`) returns None
only if BOTH `active_pick_probability.json` and `active_ats_model.json` are
missing from `artifacts_root` - never true in the real repo. So in
`board_content.py::load_board_content` and `publishing.py::_publication_context`,
the `pick_probability is not None` branch always fires, and
`attach_pick_probability` (`pick_probability.py:820-878`) already sets
`DISPLAYED_PICK_PROBABILITY_COLUMN = CALIBRATED_PICK_PROBABILITY_COLUMN` (the
four-term model's own served-side probability) at line 876-877. The 12-cell
lookup (`fit_production_displayed_confidence`/`attach_displayed_confidence`
old version) was only ever reached as a **dead fallback** for the no-model
case. **This means step 3 (before/after Week 3 numbers) should show NO change**
- confirm this empirically once the code compiles again, do not assume.

**Step 1 (LOSO calibration measurement) has NOT been done yet.** No script has
been run. Still needed before writing a verdict into this lane:
- `pick_probability_fit.build_fit_population(artifacts_root, data_root)` gives
  the graded population (`model_logit`, `composition_flag_sum`,
  `market_move_toward_home`/`market_move_available`, `home_covered`, `season`,
  `tue_open_home_spread`, `model_probability`=`home_cover_probability_at_open`).
- (i): reproduce the LOSO loop already in `fit_pick_probability`
  (`pick_probability_fit.py` ~lines 300-420, read up to line 400 only so far -
  read the rest) to get each game's out-of-season four-term home probability;
  convert to served-side probability; compute Brier/log loss/reliability table,
  overall and per season.
- (ii): the old 12-cell lookup code was **already deleted** from
  `displayed_confidence.py` before this measurement was run (ordering mistake -
  do the measurement in a throwaway script, do not restore it to `src/`).
  Reimplement it standalone in a scratch script: bucket by `|spread|` (<=6.5,
  <7.5, <=10, >10), band by the served four-term probability (<0.55, <0.60,
  >=0.60), blend `(prior_wins + 20*stated)/(prior_games + 20)` using
  leave-one-season-out priors (all OTHER seasons' completed games, not
  walk-forward), report Brier/log loss/reliability per season and paired
  vs (i) with an interval (bootstrap or normal approx on the paired
  difference). Old logic reference: `git show
  eeba647:src/nfl_ats/displayed_confidence.py` (last commit before this
  session's edit) has the original `cell_keys`/`ReliabilityCells`/
  `fit_reliability_cells` code if needed as a reference implementation.
- Write the numbers into this lane's Open/State before code is considered
  final. If (ii) ever beats (i), STOP and re-derive a LOSO lookup instead of
  proceeding (not expected, given (i) is already the served, LOSO-fit model).

## Tried / done so far (code edits, in the working tree, uncommitted)

1. Rewrote `src/nfl_ats/displayed_confidence.py` in full: deleted
   `PSEUDO_OBSERVATIONS`, `DISPLAY_BUCKETS`, `PROBABILITY_BANDS`,
   `display_spread_bucket`, `probability_band`, `cell_keys`, `ReliabilityCells`,
   `fit_reliability_cells`, `archive_display_stream`, `prior_rows_before`,
   `walk_forward_displayed_confidence`, `derive_strength_bands`,
   `ProductionDisplayedConfidence`, `_newest_opener_evaluation`,
   `_evaluation_model_id`, `_empty_production`,
   `fit_production_displayed_confidence`, `served_strength_bands`,
   `DisplayedConfidenceSourceError`, `DISPLAYED_CONFIDENCE_POLICY`,
   `DISPLAYED_CONFIDENCE_SERVED`, `DISPLAYED_CONFIDENCE_FILENAME`,
   `STRENGTH_BAND_QUANTILES`. Kept: `PICK_SIDE_FLOOR` (plain 0.5 constant, still
   legitimately used elsewhere as "a picked side is never shown below 50%"),
   `DISPLAYED_PICK_PROBABILITY_COLUMN`, `DISPLAYED_STRENGTH_WORD_COLUMN`,
   `STRENGTH_WORDS`, `STRENGTH_ROUNDING_PLACES`, `StrengthBands` dataclass
   (unchanged shape), `displayed_pick_probability`, `displayed_strength_word`.
   Changed `attach_displayed_confidence(predictions, calibration)` to
   `attach_displayed_confidence(predictions, bands: StrengthBands | None)` -
   always sets displayed = stated pick-side probability (identity, no lookup),
   attaches strength word from `bands` if given.
2. `board_content.py`: import block now `from nfl_ats.displayed_confidence
   import (PICK_SIDE_FLOOR, StrengthBands, attach_displayed_confidence)`.
   `_played_pick` signature dropped its `calibration` param (was
   `ProductionDisplayedConfidence | None`); body no longer calls
   `.calibrate()`. `load_board_content`: removed the
   `fit_production_displayed_confidence(...)` call and the
   `displayed_confidence=` kwarg into `resolve_card_view`; the
   `if pick_probability is None:` branch now does
   `strength_bands = None; final = attach_displayed_confidence(final, None)`
   (dropped `played_calibration` variable entirely); fixed the
   `_played_pick(row, played_overrides[game_id], strength_bands)` call site
   (dropped the removed arg). **Verified no remaining `played_calibration` or
   `fit_production_displayed_confidence` references in this file** (grepped).
3. `card_view.py`: import line now
   `from nfl_ats.displayed_confidence import StrengthBands,
   attach_displayed_confidence`. `resolve_card_probabilities`'s
   `displayed_confidence` param retyped to `StrengthBands | None` (done). Its
   `attach_displayed_confidence(final_predictions, displayed_confidence)` call
   needs no change (signature-compatible).

## State update (this session, tool-call cap hit mid-verification)

Steps 1-5 of "Next" below are DONE and the tree compiles:
- `card_view.py` line ~521: `displayed_confidence: ProductionDisplayedConfidence | None = None`
  changed to `StrengthBands | None = None`. Confirmed zero remaining
  `ProductionDisplayedConfidence` hits in `card_view.py`.
- `publishing.py`: import block now only
  `from nfl_ats.displayed_confidence import (DISPLAYED_PICK_PROBABILITY_COLUMN,
  attach_displayed_confidence)`. Removed `fit_production_displayed_confidence`
  call in `_publication_context`; removed the `displayed_confidence=` kwarg
  passed to `resolve_card_view`; `served = ... else
  attach_displayed_confidence(view.predictions, None)`; dropped
  `ProductionDisplayedConfidence` from the return-type tuple and the
  `return (...)` tuple; dropped `displayed_confidence` from the
  `publish_active_predictions` unpacking (~line 452-462, now 3 fewer names);
  deleted the `atomic_json(displayed_confidence.to_dict(), forecast_dir /
  DISPLAYED_CONFIDENCE_FILENAME)` line entirely.
- `public_board.py`: import block drops `served_strength_bands`; added
  `PickProbabilitySourceError, load_pick_probability_model_or_none` to the
  `nfl_ats.pick_probability` import. Added helper `_served_strength_bands
  (artifacts_root: Path | None, active: Mapping[str, Any] | None) ->
  StrengthBands | None` right before `confidence_word` (~line 728), body
  exactly as the lane specified. Both call sites
  (`strength_bands = served_strength_bands(...)` ~line 1619 and
  `strength_bands=served_strength_bands(...)` ~line 4048) now call
  `_served_strength_bands`.
- Repo-wide sanity grep for all deleted symbol names across `src` and `tests`:
  every hit is a false positive from an unrelated same-named local symbol in
  another module (`clv.py::prior_rows_before`,
  `home_side_location.py::prior_rows_before`,
  `score_lattice.py::PSEUDO_OBSERVATIONS`,
  `pick_probability.py::STRENGTH_BAND_QUANTILES`, and our own new
  `_served_strength_bands`). Verified `pick_probability.py`'s local
  `from nfl_ats.displayed_confidence import (...)` at line 830 only imports
  the two preserved symbols (`DISPLAYED_PICK_PROBABILITY_COLUMN`,
  `DISPLAYED_STRENGTH_WORD_COLUMN`). No real remaining references anywhere.
- **Measured**: `.tools/uv.exe run python -c "import nfl_ats.cli"` exits 0,
  no output/errors - full import graph compiles.
- **Measured**: `ruff format --check` on the 5 touched files
  (`displayed_confidence.py card_view.py publishing.py public_board.py
  board_content.py`) initially flagged 1 file
  (`displayed_confidence.py:71`, an over-long `bands.words(...)` line from the
  prior session's edit); fixed with `ruff format` (1 file reformatted); `ruff
  check` on the same file then reported "All checks passed!". The cap hit
  before the *combined* re-run across all 5 files could be confirmed, and
  before `board_content.py` was individually confirmed clean (it was not
  touched this session, only in a prior one - worth re-checking once as a
  matter of hygiene).

## Next (do these in order)

0. **Resume verification, cap was hit mid-command**: re-run
   `.tools/uv.exe run ruff format --check src/nfl_ats/displayed_confidence.py
   src/nfl_ats/card_view.py src/nfl_ats/publishing.py
   src/nfl_ats/public_board.py src/nfl_ats/board_content.py` and `ruff check`
   on the same 5 files - expected clean given each file was already checked
   individually or untouched, but not yet confirmed together in one command.
   Then run `.tools/uv.exe run mypy src` (repo-wide per policy) and fix any
   new type errors introduced by this lane's edits only (do not fix
   pre-existing unrelated errors). Then find and run tests: `Glob` for
   `tests/test_*board*`, `tests/test_*card*`, `tests/test_*publish*`,
   `tests/test_public_board.py`, plus `grep -rl
   "displayed_confidence\|board_content\|card_view\|public_board\|publishing"
   tests/` to be sure; run with
   `.tools/uv.exe run pytest -q <paths>`. Fix only legitimately broken
   existing tests (import/signature changes only, e.g. anything importing
   `ProductionDisplayedConfidence`/`served_strength_bands`/
   `fit_production_displayed_confidence` from `nfl_ats.displayed_confidence`
   or `nfl_ats.public_board`) - add no new tests (moratorium).

**Items 1-5 below are DONE (see "State update" above) - do not redo them.**
Remaining: item 0 above, then old items 6-9 below (LOSO measurement, Week 3
regeneration/equality check, full lint+mypy+pytest gate, final report).

1. **FIX COMPILE ERROR FIRST**: `card_view.py`, function `resolve_card_view`
   (~line 521), still reads
   `displayed_confidence: ProductionDisplayedConfidence | None = None` - that
   name is no longer imported. Change to
   `displayed_confidence: StrengthBands | None = None`. Then
   `grep -n "ProductionDisplayedConfidence" src/nfl_ats/card_view.py` to
   confirm zero hits remain.
2. `publishing.py`: import block (~line 30) drop `DISPLAYED_CONFIDENCE_FILENAME`,
   `ProductionDisplayedConfidence`, `fit_production_displayed_confidence`; keep
   `DISPLAYED_PICK_PROBABILITY_COLUMN`, `attach_displayed_confidence`. In
   `_publication_context` (~152-243): delete the
   `displayed_confidence = fit_production_displayed_confidence(...)` call
   (~191-196); delete the `displayed_confidence=displayed_confidence` kwarg
   passed to `resolve_card_view`; change
   `served = view.predictions if view.pick_probability is not None else
   attach_displayed_confidence(view.predictions, displayed_confidence)` to use
   `None` as the second arg; drop `ProductionDisplayedConfidence` from the
   function's return-type tuple annotation and from the `return (...)` tuple.
   Update the sole call-site unpacking in `publish_active_predictions`
   (~line 452-462) to drop `displayed_confidence` from the unpacked names.
   Delete the line
   `atomic_json(displayed_confidence.to_dict(), forecast_dir /
   DISPLAYED_CONFIDENCE_FILENAME)` (~line 670) entirely - that artifact stops
   being written; `scripts/statistical_audit_20260915.py` (unowned, one-off)
   already handles it being absent.
3. `public_board.py`: import block (~line 56) drop `served_strength_bands`,
   keep the rest. Add `load_pick_probability_model_or_none,
   PickProbabilitySourceError` to the existing
   `from nfl_ats.pick_probability import BASE_PROBABILITY_POLICY,
   calibrated_discrete_sweep` line (~line 90). Add a small helper, e.g.:
   ```
   def _served_strength_bands(artifacts_root, active):
       if active is None or artifacts_root is None:
           return None
       try:
           model = load_pick_probability_model_or_none(artifacts_root)
       except PickProbabilitySourceError:
           return None
       if model is None:
           return None
       return StrengthBands(lean_min=model.lean_minimum, strong_min=model.strong_minimum)
   ```
   Replace both call sites (`~line 1600`:
   `strength_bands = served_strength_bands(artifacts_root, active_model)`; and
   `~line 4029`:
   `strength_bands=served_strength_bands(artifacts_root, artifacts.active)`)
   with calls to `_served_strength_bands`.
4. `pick_probability.py`, `card_explanation.py`, `pool_workbench.py` need NO
   changes (verified: they only use symbols preserved unchanged -
   `DISPLAYED_PICK_PROBABILITY_COLUMN`/`DISPLAYED_STRENGTH_WORD_COLUMN`/
   `displayed_pick_probability`/`displayed_strength_word`/`StrengthBands`).
5. Repo-wide sanity grep for anything still referencing deleted symbols:
   `grep -rn "ProductionDisplayedConfidence\|fit_production_displayed_confidence\|ReliabilityCells\|served_strength_bands\|DisplayedConfidenceSourceError\|PSEUDO_OBSERVATIONS\|DISPLAYED_CONFIDENCE_POLICY\|DISPLAYED_CONFIDENCE_SERVED\|DISPLAYED_CONFIDENCE_FILENAME\|derive_strength_bands\|walk_forward_displayed_confidence\|archive_display_stream\|prior_rows_before\|cell_keys\|display_spread_bucket\|DISPLAY_BUCKETS\|PROBABILITY_BANDS\|STRENGTH_BAND_QUANTILES" src tests` -
   fix any remaining hit (`tests/test_public_board.py` only uses `StrengthBands`,
   confirmed safe).
6. Do the Step-1 LOSO measurement described above (build_fit_population +
   scratch script) and write the numbers into this lane.
7. Regenerate the board/card **locally, in-process** (do not run
   `publish-board`/`publish-predictions`): write a throwaway script that calls
   `nfl_ats.board_content.load_board_content(artifacts_root, data_root=...)`
   (or `resolve_card_view` directly) and prints
   `game_id, matchup, pick side, DISPLAYED_PICK_PROBABILITY_COLUMN` for the
   active week's 16 games. Compare against `docs/index.html`'s currently
   committed percentages (`git status` shows it modified from a prior session -
   check `git diff docs/index.html` / `git show HEAD:docs/index.html`) or
   against a `git stash`-based before/after if a true diff is required. Expect
   **zero change** in both pick side and displayed percentage, per the State
   section above - confirm, don't assume; if any side changes, STOP and report
   instead of proceeding.
8. Run, once each, on touched files: `ruff format --check .`,
   `ruff check .`, `mypy src`, and `pytest -q` scoped to related tests (find
   with `Glob` for `tests/test_*board*`, `tests/test_*card*`,
   `tests/test_*publish*`, `tests/test_public_board.py`). Fix any legitimately
   broken existing test minimally (signature/import changes only); add no new
   tests (moratorium).
9. Report to the root: measured calibration comparison numbers, confirmation
   that Week 3 picks/sides and displayed numbers are unchanged (dead-code-only
   change), file:line list of the diff, test/lint results. Root handles
   publish-board/publish-predictions/commit/push.

## Open
- Whether `StrengthBands` cutoffs (lean/strong tertiles) themselves should also
  be chosen out-of-season is a separate, not-yet-raised question - out of
  scope for this lane (task only targeted the displayed cover-chance number
  and its 12-cell lookup, not the strength-word legend thresholds).
- `docs/index.html` was already modified before this session started (per
  git status at session start) - confirm that pre-existing diff is unrelated
  before attributing any docs/index.html change to this work.
