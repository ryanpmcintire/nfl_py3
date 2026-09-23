# Spread explorer: discrete lattice serving

## Goal
Make `src/nfl_ats/spread_explorer.py` serve cover+push probabilities at every
alternative line from the SAME discrete margin distribution conditional on the
line that the card uses (policy `discrete_conditional_non_push_v1`, from
`mass_preserving_lattice.py`), reproducing the card's served
`home_cover_probability` at the posted line, failing closed on mismatch.
Keep legacy (pre-lattice) Gaussian cards working unchanged. Flip line stays
one adverse-direction number. No code comments/docstrings. No new tests.
Do not publish/commit/push (that is the primary orchestrator's job later).

## State (steps 1-5 applied, verification pending)
Applied Next items 1-5 this session:
1. `compute_spread_explorer_distribution`'s Gaussian-only check block replaced
   with the discrete-aware `is_discrete`/`policy_name` version (verbatim edit
   from the old Next item 1) — `src/nfl_ats/spread_explorer.py:383-411`.
2. `__all__` now includes `spread_explorer_home_cover_probability` and
   `spread_explorer_three_way_probability` (`spread_explorer.py:445-459`).
3. `public_board.py`'s `compute_spread_explorer_params(...)` production call
   site (~3949-3961) now passes `artifacts_root=artifacts_root,
   active=artifacts.active`.
4. `_assert_spread_explorer_matches_card` (`public_board.py:790-816`) now
   `continue`s past any `game_id` whose `params[game_id].discrete_reader is
   not None` before doing the Gaussian re-derivation, so discrete games skip
   the stale Gaussian recheck (the real check already ran inside
   `compute_spread_explorer_params`/`_rebuild_discrete_reader`).
5. `board_content.py` imports `spread_explorer_home_cover_probability`
   (~line 109); `_build_cover_curve` (~2591-2634) and `_flip_line`
   (~2637-2690) both branch on `params.discrete_reader is not None` to call
   `spread_explorer_home_cover_probability(params, line)` instead of
   `widget_home_cover_probability(...)` for discrete games; Gaussian branch
   (`discrete_reader is None`) left byte-for-byte identical, including the
   `key_line_pinned` override in `_build_cover_curve` (that pin logic is
   redundant but harmless for discrete since `spread_explorer_three_way_probability`
   already pins internally — discrete branch does NOT duplicate the pin check).

Step 6 verification DONE this session, all green:
- `ruff format --check` initially flagged 1 file (pre-existing unwrapped long
  lines from the earlier agent's session plus my new one); ran `ruff format`
  (whitespace-only) to fix, re-checked clean: "3 files already formatted".
- `ruff check` on the three files: "All checks passed!".
- `mypy src`: "Success: no issues found in 239 source files" (ran in
  background, tailed `mypy_out.txt` in scratchpad).
- `pytest tests/test_spread_explorer.py tests/test_key_line_pick_read.py
  tests/test_discrete_push_read.py -q`: "67 passed, 1 warning in 33.86s".
- Real production run (throwaway script, scratchpad
  `verify_explorer.py`, not committed): loaded live `artifacts/` for
  season=2026 week=3 (`base_probability_policy` = `discrete_conditional_non_push_v1`
  for all 16 games), called `compute_spread_explorer_params` with
  `artifacts_root`/`active` wired — **no `DataContractError`**, 16/16 games
  produced params. For 3 sample games (ARI_SF, ATL_GB, BAL_DAL): posted-line
  `spread_explorer_home_cover_probability` reproduces
  `card_home_cover_probability` exactly (e.g. ARI_SF 0.532961 == 0.532961);
  ±3 offsets move sensibly; push=0 at all three (all posted lines are
  half-point, so no push mass is expected there — correct, not a bug); flip
  lines computed without error (11.5 / 7.0 / -2.0 — note: script did not read
  the real pick side, so flip direction here is illustrative of the mechanism
  working, not a verified pick-consistent flip; production `_flip_line` in
  `board_content.py` does read the real pick side and was not separately
  re-verified beyond the unit tests above).

Lane complete. Nothing published/committed/pushed, per task constraint.

Done in `spread_explorer.py`:
- Imports: added `BASE_PROBABILITY_POLICY`, `fit_production_discrete_push_reader`
  to the existing `from nfl_ats.mass_preserving_lattice import (...)` line.
- `SpreadExplorerGameParams` dataclass: added `discrete_reader:
  DiscretePushReader | None = None` and `discrete_point: float | None = None`
  (both default None, backward compatible with direct-construction call sites
  in tests).
- New helpers added right after the dataclass:
  - `_discrete_group_flag(group: pd.DataFrame) -> bool`: True iff
    `group["base_probability_policy"]` (when the column exists) equals
    `BASE_PROBABILITY_POLICY` for every row; raises `DataContractError` on a
    mixed group; returns False when the column is absent (legacy card).
  - `_rebuild_discrete_reader(features, artifacts_root, active, *, season,
    week) -> DiscretePushReader`: fails closed (DataContractError) if
    `artifacts_root is None`, else calls
    `fit_production_discrete_push_reader(features, artifacts_root,
    dict(active) if active else None, season=season, week=week)` and fails
    closed if `.reader is None`.
- `compute_spread_explorer_params(...)`: added `artifacts_root: Path | None =
  None, active: Mapping[str, Any] | None = None` params. Body now: computes
  `is_discrete = _discrete_group_flag(group)`; if discrete, rebuilds the
  reader, computes `points = centers + residual_location(model.residuals,
  probability_method)` (matches `serve_discrete_three_way`'s point formula
  exactly), checks via `reader.read(line, point).home_cover_probability` per
  game (through `apply_pick_overrides`) instead of the Gaussian
  `smoothed_home_cover_probability`; else unchanged legacy Gaussian check.
  Same `atol=1e-9` tolerance both branches. Per-game `SpreadExplorerGameParams`
  now also sets `discrete_reader=reader` (None for legacy),
  `discrete_point=float(point) if is_discrete else None`.
- New functions after `widget_home_cover_probability`:
  - `spread_explorer_three_way_probability(params, line) -> (cover, push,
    loss)`: if `params.discrete_reader is not None`, calls
    `params.discrete_reader.read(line, params.discrete_point,
    conditioning_line=params.card_line)`; at the exact card line when
    `key_line_pinned`, substitutes `card_home_cover_probability` for cover and
    keeps the discrete push mass (`loss = max(0, 1 - pinned - push)`) —
    mirrors the existing pinned convention in
    `board_content._build_cover_curve`. Else falls back to the Gaussian
    `widget_home_cover_probability` (push=0).
  - `spread_explorer_home_cover_probability(params, line) -> float`:
    `cover/(cover+loss)` from the above (0.5 if both zero).
- `compute_spread_explorer_distribution(...)`: signature updated with the same
  new `artifacts_root`/`active` params (this part landed).

## Tried
Read and mapped the whole chain: `spread_explorer.py` (bug at old lines
137-149), `card_refit.py` (`smooth_reference_probability`,
`SMOOTH_PROBABILITY_COLUMN = "home_cover_probability_smooth"`),
`mass_preserving_lattice.py` (`DiscretePushReader`, `band_read`,
`serve_discrete_three_way`, `serve_discrete_sweep`,
`fit_production_discrete_push_reader`, `BASE_PROBABILITY_POLICY =
"discrete_conditional_non_push_v1"`), `public_board.py` (call site,
`_assert_spread_explorer_matches_card`, `_game_deep_dive` fallback),
`board_content.py` (`_build_cover_curve`, `_flip_line`, `_build_adjuster`,
`SpreadAdjusterParams`). Confirmed via grep that
`compute_spread_explorer_distribution`/`spread_explorer_three_way` are
currently exercised only by tests, not wired into production board building.

Key finding worth preserving: `board_content._build_cover_curve` prefers
`spread_explorer_params` (Gaussian) over the `sweep` DataFrame fallback, and
`sweep` (from `serve_discrete_sweep`/`calibrated_discrete_sweep`) is *already*
fully discrete-correct. Today `compute_spread_explorer_params` crashes for
lattice weeks, so `spread_explorer_params` is always empty for Week 3 and
`_build_cover_curve`/`_flip_line` silently fall back to the correct `sweep`
path — i.e. the real production failure mode is publish-board **hard-crashing**
for lattice weeks, not a wrong-but-live Gaussian chart. Once the crash is
fixed, `spread_explorer_params` becomes non-empty for those games, so
`_build_cover_curve`/`_flip_line` MUST be updated to consult
`params.discrete_reader` too, or they will regress from "correct via sweep
fallback" to "wrong via Gaussian widget". This is not optional polish — see
Next item 5.

`SpreadAdjusterParams`/`_build_adjuster` confirmed dead code today: `GameDive.
adjuster` is built but grep found no `.adjuster` read anywhere in
`board_content.py` rendering or `board_terminal.py` except the direct unit
test `tests/test_key_line_pick_read.py:768-835`
(`test_board_curve_adjuster_and_widget_carry_the_pinned_number`), which
constructs `SpreadExplorerGameParams` directly without discrete fields and
must keep passing unchanged (Gaussian legacy path).

Tests inventoried that touch this surface (none run yet this session):
`tests/test_spread_explorer.py` (all legacy Gaussian, no
`base_probability_policy` column — must stay green untouched),
`tests/test_key_line_pick_read.py` lines 640-836 (pinned-game reproduction,
`_build_cover_curve`/`_build_adjuster`/`board_terminal._adjuster_html`),
`tests/test_discrete_push_read.py` lines 572-596
(`spread_explorer_three_way(distribution, line, discrete_read=reader)` with
`SpreadExplorerGameDistribution` built directly — signature untouched, safe).

## Next
All 6 planned items done this session (see State/Tried below). Nothing left
in this lane; move to `docs/lanes/done/` on the next commit pass. If ever
revisited: `_build_adjuster`/`SpreadAdjusterParams` (Open, below) still needs
the same discrete wiring if it is ever rendered.

## Next (original, all applied — kept for the record)
1. **Finish the interrupted edit** in `compute_spread_explorer_distribution`
   (around the `check = float(smoothed_home_cover_probability(...))` block,
   currently still the OLD unconditional-Gaussian code). Apply this edit
   (old_string -> new_string) in `F:\Repos\nfl_py3\src\nfl_ats\spread_explorer.py`:

   OLD:
   ```
       line = float(target_rows["spread_line"].iloc[0])
       supplied = float(row["home_cover_probability"])

       check = float(
           smoothed_home_cover_probability(
               model.residuals,
               np.array([center]),
               np.array([line]),
               method=probability_method,  # type: ignore[arg-type]
           )[0]
       )
       pinned = bool(pick_overrides and str(game_id) in pick_overrides)
       if pinned:
           check = float(apply_pick_overrides([check], [str(game_id)], pick_overrides)[0])
       if not math.isclose(check, supplied, rel_tol=0.0, abs_tol=1e-9):
           raise DataContractError(
               f"Refit {probability_method!r} probability for season {season} week {week} game "
               f"{game_id!r} does not reproduce the supplied card's home_cover_probability -- the "
               "feature table or configuration has drifted from the one that produced this card; "
               "refusing to answer a spread query that could disagree with the published pick"
           )
   ```

   NEW:
   ```
       line = float(target_rows["spread_line"].iloc[0])
       supplied = float(row["home_cover_probability"])

       is_discrete = str(row.get("base_probability_policy", "")) == BASE_PROBABILITY_POLICY
       if is_discrete:
           reader = _rebuild_discrete_reader(features, artifacts_root, active, season=season, week=week)
           point = center + residual_location(model.residuals, probability_method)
           check = reader.read(line, point).home_cover_probability
           policy_name = BASE_PROBABILITY_POLICY
       else:
           check = float(
               smoothed_home_cover_probability(
                   model.residuals,
                   np.array([center]),
                   np.array([line]),
                   method=probability_method,  # type: ignore[arg-type]
               )[0]
           )
           policy_name = repr(probability_method)
       pinned = bool(pick_overrides and str(game_id) in pick_overrides)
       if pinned:
           check = float(apply_pick_overrides([check], [str(game_id)], pick_overrides)[0])
       if not math.isclose(check, supplied, rel_tol=0.0, abs_tol=1e-9):
           raise DataContractError(
               f"Refit {policy_name} probability for season {season} week {week} game "
               f"{game_id!r} does not reproduce the supplied card's home_cover_probability -- the "
               "feature table or configuration has drifted from the one that produced this card; "
               "refusing to answer a spread query that could disagree with the published pick"
           )
   ```
   Note: `SpreadExplorerGameDistribution` return statement right after is
   unchanged (still sets `card_probability_method=probability_method`, no new
   fields needed there — `spread_explorer_three_way` keeps taking
   `discrete_read` as an explicit caller-supplied arg, untouched).

2. Update the `__all__` list at the bottom of `spread_explorer.py` to add
   `"spread_explorer_three_way_probability"` and
   `"spread_explorer_home_cover_probability"`.

3. Wire the production entry point in `src/nfl_ats/public_board.py` (~line
   3942-3960, inside the function building the board pages — grep
   `spread_explorer_params: dict\[str, SpreadExplorerGameParams\] = \{\}`):
   add `artifacts_root=artifacts_root, active=artifacts.active` kwargs to the
   `compute_spread_explorer_params(...)` call. The surrounding gate
   (`str(artifacts.metadata.get("probability_method")) in ("gaussian",
   "gaussian_median")`) already lets discrete-served weeks through (that
   field tracks the location-smoothing method, not the served policy) — do
   not change the gate itself.

4. Fix `_assert_spread_explorer_matches_card` in `public_board.py` (~line
   790-816): it re-derives a Gaussian check from the JSON-safe payload
   (`spread_explorer_payload`, which only carries center/mean/std/line) and
   will spuriously fail for discrete games even after step 1-3. Skip that
   re-check for discrete games (trust the stronger check already performed
   inside `compute_spread_explorer_params`): iterate `params.items()`
   alongside the payload and `continue` when `params[game_id].discrete_reader
   is not None`.

5. **Not optional** (see State's key finding): update
   `board_content._build_cover_curve` and `board_content._flip_line` (~lines
   2589-2688) to call the new `spread_explorer_home_cover_probability`/
   `spread_explorer_three_way_probability` from `spread_explorer.py` instead
   of `widget_home_cover_probability` directly, whenever
   `params.discrete_reader is not None`; keep the exact existing Gaussian
   call (`widget_home_cover_probability(...)`) for the `discrete_reader is
   None` branch so `tests/test_key_line_pick_read.py:768-835` keeps passing
   byte-for-byte. Import the two new names into `board_content.py`.
   `_build_adjuster`/`SpreadAdjusterParams` is confirmed dead/unrendered code
   — leave it Gaussian, lowest priority, do not expand scope there.

6. Verify (one pass, do not loop):
   - `F:\Repos\nfl_py3\.tools\uv.exe run --no-sync ruff format --check src/nfl_ats/spread_explorer.py src/nfl_ats/public_board.py src/nfl_ats/board_content.py`
   - `... ruff check src/nfl_ats/spread_explorer.py src/nfl_ats/public_board.py src/nfl_ats/board_content.py`
   - `... mypy src` (tail the output; this repo's mypy run is slow — use a
     background run if needed and check the completion notification, don't
     poll)
   - `... pytest tests/test_spread_explorer.py tests/test_key_line_pick_read.py tests/test_discrete_push_read.py -q`
   - One real run building spread-explorer data for 2026 Week 3: find the
     outer function name in `public_board.py` around the call site (step 3)
     and how the `publish-board` CLI invokes it (grep `publish-board` /
     `публиш` in `src/nfl_ats/cli_commands/`), then either call that function
     directly from a throwaway script in the scratchpad dir, or find a
     smaller entry point that only builds `spread_explorer_params` (i.e. call
     `compute_spread_explorer_params` directly with `artifacts.predictions`/
     `explorer_features`/`artifacts_root`/`active` for season=2026 week=3, no
     need to render full HTML) — print, for 3 sample games,
     `spread_explorer_home_cover_probability(params, card_line)`,
     `(..., card_line - 3)`, `(..., card_line + 3)`, and confirm the posted-line
     value equals `params.card_home_cover_probability`. Confirm no
     `DataContractError` is raised (that is literally "no longer refuses").

## Open
- Did not touch `_build_adjuster`/`SpreadAdjusterParams` beyond noting it is
  dead code; if the orchestrator later renders it, it will need the same
  discrete wiring as `_build_cover_curve`.
- Did not run ruff/mypy/pytest/the real Week 3 build yet this session — all
  verification is still pending (step 6).
- `_game_deep_dive`'s Gaussian fallback branch in `public_board.py`
  (~line 1131-1153, used only when `game_sweep` is empty) was intentionally
  left untouched — for discrete weeks `game_sweep` should be non-empty
  (populated from the served sweep artifact), so this is a true fallback path,
  not a primary server of alt-line answers; revisit only if verification shows
  it is actually hit for Week 3.
- No files were published/committed/pushed, per the task constraint.
