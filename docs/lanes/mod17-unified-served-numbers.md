# MOD-17 / POL-12: unified served numbers (tiebreaker lattice centre)

## Goal
Make the tiebreaker final, push probability, and lattice-read cover probability
agree in side with the served four-term pick probability
(`nfl_ats.pick_probability_fit`, `artifacts/active_pick_probability.json`) on
every game, not just the week's last game. Never publish a tiebreaker number
that contradicts the pick.

## State -- fixed, verified this session
Audit (measured, 2026-09-24, `artifacts/margin_predictions/2026-week-03-20260924T161231Z/predictions.csv`
method=`market_residual` vs `CURRENT_PREDICTIONS.md`'s served side/probability):
7 of 16 Week 3 games have `predicted_margin` on the wrong side of the spread
line relative to the served pick (CIN_PIT, HOU_IND, KC_MIA, LAC_BUF, LA_DEN,
NYJ_DET, TEN_NYG). Before the fix, calling `tiebreaker.tiebreaker_report` with
the real served `published_pick_side` for those games produced a
`pick_cover_probability` **below 50% for the served side** (e.g. HOU_IND
0.420, KC_MIA 0.444, LA_DEN 0.463) while the score itself still rendered
"consistent with the pick" -- a live contradiction between the printed cover
chance and the printed pick, exactly the MOD-17 defect the row's 2026-09-11
entries diagnosed and built `lattice_centre_challenger.challenger_centre` for
but never wired in.

**Fix applied** (`src/nfl_ats/tiebreaker.py` `build_report`): the lattice
centre is no longer always `model_view.predicted_margin`. It now calls the
existing `nfl_ats.lattice_centre_challenger.challenger_centre(predicted_margin,
spread_line, pick_side)` (deferred import inside the function to avoid the
existing tiebreaker<->lattice_centre_challenger import cycle) -- if
`predicted_margin` is already strictly on the served pick's side, the centre
is unchanged (byte-identical for all 9 already-agreeing Week 3 games,
confirmed for PHI_CHI: still CHI 21 - PHI 19, centre 0.939); otherwise the
centre snaps to the minimum consistent step off the spread line
(`floor(line)+1` HOME / `ceil(line)-1` AWAY), discrete, no Gaussian. Added
`TiebreakerReport.lattice_centre_margin` (the actual centre used) and pointed
`publishing._tiebreaker_json_payload`'s `lattice_centre_margin` field at it
instead of the raw `predicted_margin`, since that field is read back by
`board_content.py` (reader-facing `implied_margin` fallback) and
`tiebreaker_shade_prospective.py`.

**Re-verified after the fix**, all 7 disagreeing games: `pick_cover_probability`
now >= 0.5 on the served side (CIN_PIT 0.557, HOU_IND 0.521, KC_MIA 0.507,
LAC_BUF 0.547, LA_DEN 0.531, NYJ_DET 0.561, TEN_NYG 0.543), scores still land
on the pick side ("consistent with the ... pick"), push probability 0% on all.
Real CLI `nfl-ats tiebreaker --season 2026 --week 3` still prints CHI 21 - PHI
19, total 40, "consistent with the CHI +3.5 pick" -- byte-identical to the
published card (no regression on the one game that was already consistent).

Gates measured: `pytest tests -k "tiebreak or lattice" -n 2` -- 90 passed, no
test edited or added. `ruff format --check` / `ruff check` on
`tiebreaker.py` + `publishing.py` clean. `mypy src` -- Success: no issues
found in 240 source files.

## Tried
- Considered importing `lattice_centre_challenger` at module top of
  `tiebreaker.py`: rejected, that module imports `last_game_of_week` etc. back
  from `tiebreaker` at its own top level, which fails at partial-module-load
  time. Function-local (deferred) import is safe since by call time
  `nfl_ats.tiebreaker` is always already fully loaded; proven by the live CLI
  run and the direct-call script above.

## Next
- The prospective ledger `artifacts/prospective/lattice_centre_decisions.parquet`
  and `src/nfl_ats/lattice_centre_challenger.record_lattice_centre_decisions`
  were built to compare served-vs-challenger arms paired; now that the
  challenger centre IS the served centre for the pick-side computation, decide
  whether that paired ledger should be retired or repointed (it would now be
  comparing the served number against itself for the flip games). Not touched
  this session -- read-only audit only touched `tiebreaker.py` /
  `publishing.py`.
- Orchestrator: regenerate/republish the Week 3 card if this fix should reach
  `CURRENT_PREDICTIONS.md`/`tiebreaker.json` before lock (PHI_CHI, the current
  last game, is unaffected by the fix, so republishing is not urgent for
  correctness this week, but the `lattice_centre_margin` field in
  `tiebreaker.json` for future weeks with a flip game will now differ).

## Open
- None blocking. This subagent did not commit, publish, or touch ROADMAP.md
  per task scope (read-only audit of the MOD-17 row's tail was performed, not
  edited).
