# Pool tiebreaker: one lattice, one margin, one total

The pool breaks ties on the final score of the week's last game
(`src/nfl_ats/tiebreaker.py`'s module docstring; owner, 2026-09-01). This
document covers the 2026-09-05 consistency fix, prompted by the owner
verbatim: "our project over/under total needs to line up with our spread
prediction. otherwise something is clearly wrong/out of sync" -- and, once
the fix was under way, "if our spread prediction disagrees with the total
prediction, we need to understand why we dont have a unified model."

## The bug

The published Week 1 card picks KC -3 (53.4% cover probability; the active
model's own `predicted_margin` for DEN at KC was +3.19). `nfl-ats
tiebreaker` guessed **KC 23, DEN 20** -- a margin of exactly 3, an exact
**push** against the card's own pick, because the guess's integer score
came from the kernel-weighted historical *neighborhood*'s median actual
margin/total, not from the production model's own margin or the served
total. A margin that happens to round to the market's own number, or to
the wrong side of it, is not a rounding curiosity -- it directly
contradicts the pick the card just made.

## The fix: one lattice, one margin, one total

The project already had the unifying object for this: MOD-05's joint score
lattice (`src/nfl_ats/score_lattice.py`, `docs/score_lattice.md`) -- an
empirical `(margin, total)` residual cloud, recentred on a guess centre and
interpolated onto the integer score grid. `nfl_ats.tiebreaker.build_report`
now uses it directly whenever a production model view exists:

1. **Centre** = `(model_view.predicted_margin, guess_total_line)` -- the
   SAME production margin behind the card's pick, and the SAME served
   total (`docs/totals_model.md`: market total + `TOTALS_RESIDUAL_WEIGHT`
   x residual). Never the blended `MODEL_RESIDUAL_WEIGHT`-weighted margin
   `guess_margin` uses elsewhere on this page -- that field measures a
   different question (the market-anchored score guess), while the
   consistency check is about whether the guess agrees with the SIDE the
   card already picked.
2. **Pick side** = `"HOME"` if `model_view.residual > 0` else `"AWAY"` --
   the same sign the raw model used to pick a side in the first place.
   `pick_spread_line` = `model_view.forecast_line`, the line the model's
   pick was actually measured against.
3. **Projected score** = `nfl_ats.score_lattice.pick_consistent_top_score`:
   candidates are every feasible final whose margin lies STRICTLY on the
   pick side of the spread line -- a push (`margin == spread_line`) or a
   wrong-side final is never a candidate -- AND whose total lies within a
   total-proximity tolerance of the served total (1 point first, widened
   to 2 only when the 1-point window admits nothing). Among THOSE
   candidates, the one chosen is the one GEOMETRICALLY CLOSEST to the
   continuous centre `(model_view.predicted_margin, guess_total_line)` in
   `(margin, total)` space -- a candidate needs no lattice mass at all to
   be picked. Empirical lattice mass only breaks a NEAR-TIE (candidates
   within 0.5 points of each other's distance to the centre); it can never
   pull the choice away from the centre toward a farther, better-populated
   cell, only decide among cells that are already about equally close.
   **Second fix, 2026-09-05** (owner bug report against the real,
   published guess KC 38 - DEN 6, produced by the FIRST fix below): "most
   probable cell within the total tolerance" is still the wrong primary
   rule on a lattice this thin (effective sample size ~150 games spread
   across thousands of feasible score cells) -- a handful of scattered,
   unrelated historical games can concentrate their votes onto one distant
   cell while the real cluster near the centre is fragmented across
   several neighbours, so "most mass" kept finding a tail score even after
   the total window excluded the very first outlier. Making geometric
   closeness the primary criterion, with mass demoted to a near-tie
   breaker only, is the actual fix -- see
   `nfl_ats.score_lattice.pick_consistent_top_score`'s own docstring for
   the full rule and the worked KC 24 - DEN 20 result at the real
   production centre `(3.19, 43.62)`.

   *(First fix, superseded by the above: restricting the candidate set to
   also sit within a total-proximity tolerance BEFORE taking the argmax of
   lattice mass. That closed the original KC 23 - DEN 20 push/26-point-final
   bug but not the thin-lattice mass-concentration failure mode above.)*

   Alongside the chosen score, `pick_cover_probability` (mass on the
   pick's side) and `ScoreLattice.push_probability` report `P(cover)` and
   `P(push)` off the SAME lattice, unaffected by the selection-rule change
   -- so the panel can state "consistent with the KC -3 pick, P(cover)
   42%" without a second computation.
4. **Hard guard, never a tail score**: even the geometrically nearest
   admissible candidate must sit within 3 points of the centre on BOTH the
   margin axis and the total axis. If it does not -- or if no
   side-and-total-admissible candidate exists at any tolerance -- the
   guess degrades: `build_report` raises `TiebreakerConsistencyError` (a
   `ValueError` subclass) instead of ever publishing a tail score.
   Publishing catches it and refuses to write `tiebreaker.json` / the
   card's tiebreaker line for THAT week only -- the pool's card itself
   still publishes regardless, matching the fail-open contract every other
   optional artifact on this path already follows.
5. **The neighbourhood's raw exact-score mode list** (`common_scores`,
   `median_total`, `median_home_margin`) stays exactly as it was, reported
   as a secondary "most common finals" display -- it is not consistency-
   constrained and is never the served guess when a model view exists.

When no production model view prices the game (a historical query, or a
market-only guess), `build_report` keeps its original median-based
behaviour unchanged -- there is no card pick for a market-only guess to be
consistent with.

## Measured Week 1 result

Read live from the checkout at write time (2026-09-05, after both fixes):
`nfl-ats tiebreaker --season 2026 --week 1` guesses **KC 24, DEN 20** --
margin clears the KC -3 line strictly, total sits within a point of the
served ~43.6-point total, and the score is the feasible final geometrically
nearest the centre `(3.19, 43.62)` -- never the KC 23 - DEN 20 push the
original rounding produced, and never the KC 38 - DEN 6 tail score the
first (total-proximity-only) fix produced. The chosen final can still shift
between runs as new games complete and the neighbourhood/centre move; re-run
`nfl-ats tiebreaker` (or read the published `tiebreaker.json`) for the
current number rather than treating a number quoted here as fixed forever.

## Persistence and where the panel/assistant read it

`nfl_ats.publishing.publish_active_predictions` computes ONE
`TiebreakerReport` per publish and reuses it for: the card's tiebreaker
line under the picks table, `tiebreaker.json` (written both beside the
linked forecast artifact and beside the published card), and the existing
`lineage.json` tiebreaker input records -- never three independently timed
computations that could disagree. `tiebreaker.json`'s `implied_margin`
field is deliberately `guess_home - guess_away` (the score's own margin),
never the pre-lattice blended `guess_margin` -- the published score and its
stated margin can never disagree once persisted.

`nfl_ats.board_content.TiebreakerView` (This Week's collapsed panel) and
the board assistant's "tiebreaker" intent both read `tiebreaker.json`
read-only -- see `docs/site_content_pipeline.md`'s 2026-09-05 section --
and render "Tiebreaker not published for this week" when it is absent
(every forecast before this session) or when the consistency gate refused.
