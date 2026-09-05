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
   the lattice's most probable final whose margin lies STRICTLY on the
   pick side of the spread line -- a push (`margin == spread_line`) or a
   wrong-side final is never a candidate, and a cell with zero real mass
   is never returned as an invented guess. Ties are broken by closeness to
   the continuous centre. Alongside it, `pick_cover_probability` (mass on
   the pick's side) and `ScoreLattice.push_probability` report `P(cover)`
   and `P(push)` off the SAME lattice, so the panel can state "consistent
   with the KC -3 pick, P(cover) 42%" without a second computation.
4. **Publish gate**: if no admissible final exists, or the chosen final's
   total drifts more than one point from the served total, `build_report`
   raises `TiebreakerConsistencyError` (a `ValueError` subclass). Publishing
   catches it and refuses to write `tiebreaker.json` / the card's tiebreaker
   line for THAT week only -- the pool's card itself still publishes
   regardless, matching the fail-open contract every other optional
   artifact on this path already follows.
5. **The neighbourhood's raw exact-score mode list** (`common_scores`,
   `median_total`, `median_home_margin`) stays exactly as it was, reported
   as a secondary "most common finals" display -- it is not consistency-
   constrained and is never the served guess when a model view exists.

When no production model view prices the game (a historical query, or a
market-only guess), `build_report` keeps its original median-based
behaviour unchanged -- there is no card pick for a market-only guess to be
consistent with.

## Measured Week 1 result

Read live from the checkout at write time (2026-09-05): the model's own
`predicted_margin` for DEN at KC and the served total combine, through the
lattice, to a score whose margin clears the KC -3 line strictly and whose
total sits within a point of the served ~43-point total -- never the
KC 23 - DEN 20 push the old rounding produced. The exact final the lattice
selects depends on the neighbourhood's live density near that centre at
publish time; re-run `nfl-ats tiebreaker` (or read the published
`tiebreaker.json`) for the current number rather than treating a number
quoted here as fixed.

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
