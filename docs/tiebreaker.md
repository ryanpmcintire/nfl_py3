# Pool tiebreaker: one lattice, one margin, one total

The pool breaks ties on the final score of the week's last game
(`src/nfl_ats/tiebreaker.py`'s module docstring; owner, 2026-09-01). This
document covers the 2026-09-05 consistency fix, prompted by the owner:
the projected over/under total has to line up with the spread prediction,
otherwise something is out of sync -- and, once the fix was under way, the
question of why there is no unified model when the two disagree.

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

## 2026-09-10: what "consistent" means, stated before the audit ran

MOD-17's contract had never been written down as a checkable list, so three
different sessions each wired a new served number (the joint-residual served
total, the -1 low-side total shade, the pick-side floor on the displayed
confidence) against their own reading of it. This section freezes the list
FIRST, then reports what a standing audit script
(`scripts/mod17_served_number_audit.py`) finds when it is run against the
live card. The nine invariants below were written and committed to this file
before the script was executed for the first time.

For every game on the current week's published card:

- **C1 pick side is determinate.** The printed pick names one of the two
  teams, and the printed line is `-spread_line` for a home pick, `+spread_line`
  for an away pick. One pick, one line, no third possibility.
- **C2 the projected margin is strictly on the pick side.** `predicted_margin`
  > `spread_line` for a home pick, `<` for an away pick. Equality -- a push at
  the point estimate -- is a violation, not a rounding curiosity. This is the
  original 2026-09-05 bug, restated as a per-game check.
- **C3 the served cover probability is on the pick side.** The pick-side cover
  probability (`home_cover_probability` for a home pick, its complement for an
  away pick) is at least 50%. A card that picks a side at 48% is telling the
  reader two different things.
- **C4 the displayed confidence never falls below 50% on the picked side.**
  `displayed_pick_probability` >= `PICK_SIDE_FLOOR` (0.5) for every game --
  the floor `src/nfl_ats/displayed_confidence.py` added on 2026-09-10, checked
  on the served card rather than only inside the calibrator.
- **C5 the served total is the method total plus the shade, in that order.**
  `served_total` == (the `SERVED_TOTAL_METHOD` output: market total plus the
  method's weighted residual) + `TOTAL_LOW_SIDE_SHADE_POINTS`. The shade is
  applied AFTER the blend, to the total only, and never to the margin.
- **C6 the projected score is a cell of the same lattice.** `(guess_home,
  guess_away)` are both in the lattice's feasible support, the cell's margin
  is strictly on the pick side of the same `spread_line`, and the cell sits
  within the 3-point hard guard of the centre `(predicted_margin,
  served_total)` on BOTH axes, and within the total tolerance the selection
  actually used (1 point, widened to 2 only when 1 admits nothing).
- **C7 the projected total agrees with the projected score.** Published
  `projected_total` == `guess_home + guess_away` and published
  `implied_margin` == `guess_home - guess_away`. Two fields, one score.
- **C8 the published tiebreaker is the tiebreaker the code serves now.**
  `tiebreaker.json` beside the linked forecast carries the shade fields, and
  its `served_total`, score and pick side equal what the current code produces
  for the same game. A drift here means a published number went stale behind a
  wiring change, which `AGENTS.md`'s "no number on the site may go stale" rule
  already bans.
- **C9 the two prospective totals arms are commensurable.** The
  `totals_served_method` ledger's `served_total_blend_k01` and
  `served_total_joint_residual` are on the SAME shade scale as each other, so
  the paired absolute-error comparison measures the METHOD and nothing else.
  The shade has its own separate paired challenger
  (`tiebreaker_low_side_shade`, `artifacts/prospective/tiebreaker_shade_decisions.parquet`);
  a shade folded into one arm of the method ledger would be measured twice and
  attributed to the wrong policy.

Two things the audit REPORTS but deliberately does not fail on, declared here
so the choice is not made after seeing the numbers:

- **The card's cover probability and the lattice's are two different objects
  today.** The card's comes from the margin model's probability mapping
  (`probability_method`, the discrete push read, the home-side offset); the
  lattice's `pick_cover_probability` comes from the empirical `(margin, total)`
  residual cloud. MOD-17's text asks for one lattice behind both; MOD-18 C2
  owns the mapping. The audit fails only when the two disagree about the SIDE
  -- the part a reader could see contradict itself on one page -- and prints
  the magnitude gap as a diagnostic.
- **Only the week's last game has a served total and a served score.** The
  audit computes the per-game served total and the per-game pick-consistent
  lattice cell for all sixteen games anyway, because that is the cheapest way
  to find out whether the unification would hold everywhere if it were served
  everywhere. Non-tiebreaker rows are reported, and their C5/C6 failures are
  real failures of the invariant; nothing on the site reads them today.

One further check, **C10**, was added AFTER the list above was frozen and is
disclosed as such: the read-only `nfl-ats tiebreaker` rehearsal showed the
operator CLI building its guess for a different side than the card picks, so
the audit now also checks that `nfl_ats.tiebreaker.build_report`'s fallback
side (the sign of `predicted_market_residual`, used whenever no published row
is supplied) agrees with the card's pick.

## 2026-09-11: what the audit found on the live Week 1 card

**Measured** this session: `.\.tools\uv.exe run --no-sync python
scripts/mod17_served_number_audit.py` against active model `d49194e04945a5e5`,
forecast `margin_predictions/2026-week-01-20260910T210852Z`, served total
method `joint_residual`, shade -1.0, displayed-confidence policy
`displayed_confidence_reliability_v2_pick_side_floor`.

`P(cov)` is the card's own cover probability on the picked side; `shown` is
the displayed confidence the reader sees; `margin` is `predicted_margin`;
`method` is the served total before the shade and `served` after it; `cell` is
the pick-consistent lattice cell with its margin `cm`, total `ct` and the
total tolerance `tol` the selection used; `latP` is the lattice's own cover
probability on the picked side.

```
            game  pick   line  P(cov)   shown    word  margin  mkt tot   method   served      cell   cm   ct  tol   latP
         ARI_LAC   ARI   +9.5  0.6405  0.5335  slight   +4.58     47.5   47.487   46.487     25-21   +4   46    1  0.682
         ATL_PIT   PIT   -3.5  0.5578  0.5543    lean   +1.29     41.5   41.526   40.526     22-18   +4   40    1  0.394
         BAL_IND   IND   +3.5  0.5528  0.5540    lean   -5.55     48.0   47.940   46.940     22-25   -3   47    1  0.386
         BUF_HOU   HOU   +1.5  0.5405  0.5699    lean   -0.58     45.0   45.418   44.418     22-23   -1   45    1  0.474
         CHI_CAR   CAR   +2.5  0.5166  0.5690    lean   -2.34     47.0   47.343   46.343     22-24   -2   46    1  0.505
         CLE_JAX   JAX   -8.5  0.5336  0.5933  strong   +9.20     40.0   39.976   38.976     24-15   +9   39    1  0.555
         DAL_NYG   DAL   -2.5  0.5004  0.5685    lean   -2.88     48.0   47.869   46.869     22-25   -3   47    1  0.542
          DEN_KC   DEN   +2.5  0.5181  0.5691    lean   +2.70     43.5   43.724   42.724     22-20   +2   42    1  0.509
          GB_MIN   MIN   -1.5  0.5387  0.5698    lean   +2.36     46.5   46.826   45.826     24-22   +2   46    1  0.530
          MIA_LV   MIA   +3.5  0.5690  0.5548    lean   +0.93     40.5   40.499   39.499     20-19   +1   39    1  0.593
          NE_SEA    NE   +3.5  0.5032  0.5686    lean   +3.03     36.5   36.734   35.734     19-16   +3   35    1  0.571
          NO_DET    NO   +6.5  0.5309  0.5695    lean   +7.11     50.0   50.194   49.194     28-22   +6   50    1  0.454
         NYJ_TEN   NYJ   +1.5  0.5153  0.5690    lean   +0.65     38.5   38.721   37.721     19-18   +1   37    1  0.560
           SF_LA    SF   +3.5  0.5558  0.5542    lean   +1.36     47.5   47.362   46.362     24-23   +1   47    1  0.651
          TB_CIN   CIN   -3.5  0.5063  0.5687    lean   +3.33     50.5   50.322   49.322     27-23   +4   50    1  0.403
         WAS_PHI   WAS   +5.5  0.6323  0.5566    lean   +0.86     44.5   44.088   43.088     22-21   +1   43    1  0.718
```

`NE_SEA` and `SF_LA` are FROZEN (their pick deadlines, 2026-09-10T00:20Z and
2026-09-11T00:35Z, have passed): the card prints the pick and the confidence
that were published at the deadline, and the audit checks the printed number
against the frozen one rather than against the live refit. `NE_SEA`'s printed
50.2% against a live 56.9% is that freeze working, not a drift.

**C1, C3, C4, C5, C6, C7 and C9 hold on all sixteen games.** Every printed
line matches the served spread; no card probability and no displayed
confidence falls below 50% on the picked side; every served total is exactly
the method total plus the -1 shade; every pick-consistent cell is on the
lattice, strictly on the pick side, inside the 3-point hard guard and inside
the 1-point total tolerance; the published `projected_total` and
`implied_margin` agree with the published score.

### Violation 1 (C2, five games): there are three different margins

The projected margin is NOT strictly on the pick side for `ATL_PIT` (+1.29 vs
+3.5), `BAL_IND` (-5.55 vs -3.5), `DEN_KC` (+2.70 vs +2.5), `NO_DET` (+7.11
vs +6.5) and `TB_CIN` (+3.33 vs +3.5). Two separate mechanisms, both measured
this session:

1. **Four are deliberate overlay flips.** The four-overlay composition flipped
   `BAL_IND` (`coach_fade`) and `ATL_PIT` / `DEN_KC` / `NO_DET`
   (`pbp08_protection_mismatch_tilt`). A flip mirrors the cover probability
   and changes the printed pick but does NOT move `predicted_margin`, so the
   served margin is left pointing at the side the card no longer picks. Each
   flip names a mechanism, which is what `AGENTS.md` requires of a flip; what
   it does not do is produce a margin, so MOD-17's "one margin" has nothing to
   re-centre on.
2. **One is a second margin inside the model itself.** `TB_CIN` was not
   flipped. Measured by least squares over all sixteen games (max absolute
   error 2.4e-14): the served cover probability is exactly
   `norm.sf(spread_line - predicted_margin, loc=+0.369057, scale=12.646524)`,
   so the margin that actually decides the pick is `predicted_margin +
   0.369057`, not `predicted_margin`. That is the per-game `point` field
   `discrete_push_read.json` already publishes (ARI_LAC: 4.949287 =
   4.580230 + 0.369057). On `TB_CIN` the two sit on opposite sides of the
   line: +3.33 does not cover -3.5, +3.698 does. A **third** centre exists as
   well -- every game's 50% margin band is centred on `predicted_margin +
   1.129237`, and its half-widths are asymmetric, so the band comes from the
   empirical residual quantiles while the cover probability comes from
   `gaussian_median`.

**This lane did not fix it, and says so rather than narrowing the invariant
after seeing the numbers.** Re-centring the lattice on the pick-deciding
margin would change a served number (the published tiebreaker score) on a rule
this row's own text pins to `predicted_margin`, and re-centring it for an
overlay-flipped game would require inventing a margin the overlay never
produces. Both are modelling changes that need their own predeclared
comparison. What this lane ships instead is the diagnosis, on the record, plus
a standing audit that will print it every week.

One bound worth having: **`predicted_margin` is not rendered on any
reader-facing page.** A grep of `board_content.py`, `board_terminal.py`,
`public_board.py` and `card_explanation.py` returns no hit, so no reader can
currently see the projected margin contradict the pick beside it. The damage
today is internal: `tiebreaker.json`'s `lattice_centre_margin` is published on
the other side of its own `pick_side` for a flipped game, and the lattice the
served score is chosen from is centred somewhere the card does not believe.

### Violation 2 (C10, the same five games): the operator CLI picks its own side

`nfl-ats tiebreaker` with no published row falls back to the sign of
`predicted_market_residual`, which is the point estimate's side, so on those
five games it builds a guess for the side the card does not play and then
prints "consistent with the ... pick" about it. **Measured** by the read-only
rehearsal this session: the CLI guesses **KC 23, DEN 20, "consistent with the
KC -2.5 pick"** while the published card picks **DEN +2.5** and its tiebreaker
is **KC 22, DEN 20**. Not fixed here for the same reason as C2 -- the fallback
cannot see the overlay chain or the probability mapping's offset without
plumbing the whole served card into `build_report`.

### Violation 3 (C8, fixed by republishing, not by this lane)

The published `tiebreaker.json` beside the linked forecast was written before
the -1 low-side shade was wired into `build_report` earlier the same night. It
carries no `total_low_side_shade_points`, a `served_total` of 43.7238 where
the code now serves 42.7238, and the score **KC 23 - DEN 21** where the code
now produces **KC 22 - DEN 20**. This is a stale published number, banned by
`AGENTS.md`; the fix is one `nfl-ats publish-predictions`, which this lane is
not permitted to run because it rewrites `CURRENT_PREDICTIONS.md`. Handed to
the session's publish step.

### Fixed here: the shade was leaking into one arm of a paired ledger

Wiring the shade into `build_report` also moved `TiebreakerReport.served_total`
onto the shaded scale, and `nfl_ats.served_total_challenger` was recording
that value as the `served_total_joint_residual` arm while its
`served_total_blend_k01` arm came from `comparison_total_blend_k01`, which is
computed before the shade. The next weekly recording would therefore have
written **42.7238 against 43.6115** -- a 1.1-point gap of which a full point is
the shade, not the method -- into a paired absolute-error ledger whose one
existing row (2026 week 1, recorded 2026-09-09T23:13:48Z) holds the unshaded
pair 43.7238 / 43.6115. Grading that would have credited the shade's known
low-side edge to the joint model.

The fix is one property and one line. `TiebreakerReport.served_total_before_shade`
returns `guess_total_line - low_side_shade_points`, and the challenger records
it for the joint arm. **Measured** after the change, from the audit's own
recorder-path line: the next recording would write **blend_k01 43.6115
(market+0.1115), joint_residual 43.7238 (market+0.2238)** -- both arms
unshaded, commensurable with each other and with the stored row. The shade
keeps its own separate paired challenger, so it is still measured, once.

`nfl_ats.tiebreaker_shade_prospective` had the mirror of the same problem: its
"served" arm reads `guess_home + guess_away` straight out of the published
`tiebreaker.json`, which is now already shaded, while its "shaded" arm is a
lattice centred at `market_total - 1`. Once production serves the shade, those
two arms are no longer the predeclared contrast -- they are shade-on-blend
against shade-on-market. Rather than silently redefine another row's declared
arms, the recorder now SKIPS with a stated reason when the published payload's
`total_low_side_shade_points` is already at or past -1, so no contaminated
pair is written and LEAD-54's owner decides what that challenger becomes.

### Still open on this row

- The three margins above (C2 / C10). The named next step is to test, as a
  predeclared challenger, centring the lattice on the pick-deciding margin
  (`discrete_push_read.json`'s `point`) instead of `predicted_margin`, graded
  on the tiebreaker's own closest-total and pick-consistency metrics.
- `artifacts/prospective/totals_served_method_decisions.parquet` holds one row
  and its `realised_total` is still NaN: DEN at KC kicks off 2026-09-14. The
  row cannot report a method result until that settles.

## 2026-09-11 (later the same day): the pick-deciding-margin lattice centre

This section is a **predeclaration**. Everything from here to the results
heading was written and saved before a single error number existed; the results
section was appended afterwards. It builds the named next step the section
above left open: a paired challenger that centres the pick-consistent lattice
on the margin that actually decides the pick instead of on `predicted_margin`.

### The two arms

**Served arm (the incumbent).** Exactly what `nfl_ats.tiebreaker.build_report`
does today: `nfl_ats.score_lattice.pick_consistent_top_score` on a lattice
centred at `(predicted_margin, served_total)`, candidates restricted to integer
finals strictly on the pick side of the spread and inside the total tolerance,
geometric closeness primary, lattice mass a near-tie breaker, the 3-point hard
guard on both axes.

**Challenger arm.** Identical in every respect except the margin coordinate of
the centre, which becomes the **pick-deciding margin**:

```
point = predicted_margin + residual_location
```

`residual_location` is the median of the fitted margin-model residuals under
the served `gaussian_median` probability method, and `point` is the per-game
field `discrete_push_read.json` already publishes. It is read from the served
read, never re-derived and never a new constant: the served cover probability
is `norm.sf(spread_line - predicted_margin, loc=residual_location,
scale=residual_std)`, so `point` is by construction the margin at which the
served cover probability crosses 50%. On the live Week 1 card that location is
+0.369057; across the 2020-2025 walk-forward it moves every week (it is a
fitted median, not a constant) and is recovered per week from the archive's own
raw cover probabilities.

### The overlay-flip rule, chosen and stated before scoring

An overlay flip mirrors the served cover probability and changes the printed
pick without producing any margin, so on a flipped game `point` is still on the
side the card no longer picks, and the challenger needs a rule. Two candidates
were named; this section picks one and says why before any number is seen.

> **Rule (predeclared): when `point` is not strictly on the served pick's side
> of the spread, the challenger centres the lattice on the line plus the
> minimum consistent step** -- the nearest integer margin strictly on the pick
> side, `floor(spread_line) + 1` for a HOME pick and `ceil(spread_line) - 1`
> for an AWAY pick.

The rejected alternative was the mirror of the centre across the line,
`2 * spread_line - point`. It was rejected because it manufactures a margin
whose magnitude equals the model's disagreement in the opposite direction: a
game the margin model likes by 5 the wrong way would be centred 5 points the
right way, asserting a conviction the overlay never expressed. A flip asserts a
side and nothing more, so the minimum consistent step is the smallest claim
that honours it. Both branches are free of tuned constants: `floor(x) + 1` is
the smallest integer strictly above `x` and `ceil(x) - 1` the largest strictly
below, for every real `x`, half-point lines included.

The served arm gets no flip rule. Leaving it centred on `predicted_margin` is
what produces the C2 violation above, and the point of the comparison is to
price that.

### Historical read, frozen before it was run

**Archive.** `artifacts/opener_evaluation/<stamp>/per_game.parquet` for the
ACTIVE model -- the walk-forward opener evaluation, 1,537 graded games across
107 week blocks, seasons 2020-2025, one row per game with the Tuesday opener
line, the model's residual against it, and both the raw and the served cover
probability. `predicted_margin` is `tue_open_home_spread + residual_at_open`.
The per-week `residual_location` and `residual_std` are recovered by least
squares from that week's own `(residual_at_open,
home_cover_probability_at_open_raw)` pairs, which is an exact inversion of the
`gaussian_median` mapping rather than a fit with error; the recovery residual
is reported so the reader can check that claim.

**Pick side** is the served one, `home_cover_probability_at_open >= 0.5`, that
is, after the home-side offset. Where that disagrees with the sign of
`residual_at_open`, the game is the historical analogue of a C2 violation; the
offset is this archive's only mechanism that moves a pick without moving a
margin, so it is also what exercises the flip rule. Overlay flips are not
reconstructible from this archive and are disclosed as untested historically --
the flip rule is exercised here by offset flips and prospectively by real
overlay flips.

**Lattice history is walk-forward.** For a target game in `(season, week)` the
lattice is built from every lined final with a completed score whose gameday is
strictly before that week's first kickoff, matching the precedent in
`nfl_ats.tiebreaker_shade_prospective.shaded_score`.

**The total centre is held identical between the arms** and is the game's
market total, unshaded. The challenger changes the margin coordinate only; the
low-side shade is a separate paired challenger whose ledger must not be
contaminated (invariant C9 above), and the served joint-residual total cannot
be reconstructed walk-forward for 2020-2025 without refitting the totals model
107 times. Holding the total axis identical is the same device
`docs/score_lattice.md` section 1.1 used, and it means the measured difference
is the margin centre and nothing else.

**Populations, both declared now.**

- **Primary: every graded game** in the archive where both arms return a
  pick-consistent cell. The construction is per-game; the week's last game is a
  1-in-14 sample of the same object with no separate mechanism, and the audit
  above already computes all sixteen cells every week for exactly this reason.
- **Secondary: the week's last game only** -- the game the pool actually breaks
  ties on, 107 of them, reported with its own interval rather than folded in.

Games where either arm's hard guard refuses a cell are excluded from the pair
and counted; exclusions are never arm-specific.

**Metrics, frozen.** Paired per-game difference, served minus challenger, so
**positive favours the challenger** throughout.

| id | metric | units |
|---|---|---|
| `mod17_lattice_centre_closest_total` | closest-total absolute error, `abs(guess_total - realised_total)`, primary population | `mae_improvement` |
| `mod17_lattice_centre_closest_score` | closest-score error, `abs(guess_home - actual_home) + abs(guess_away - actual_away)`, primary population | `mae_improvement` |
| `mod17_lattice_centre_pick_consistency` | share of games whose lattice centre is strictly on the served pick's side, primary population | `accuracy_points` |
| `mod17_lattice_centre_closest_total_last_game` | closest-total absolute error, week's-last-game population | `mae_improvement` |
| `mod17_lattice_centre_closest_score_last_game` | closest-score error, week's-last-game population | `mae_improvement` |

Intervals are `nfl_ats.clv.week_blocked_bootstrap`, block `week`, 2,000
resamples, the repository's default seed, reporting the estimate, the 95%
interval and `probability_positive`. Within-week correlation is zero for this
project, so `(season, week)` is the resampling unit.

**Positive control.** A third arm, `oracle_margin`, identical to the challenger
except that its centre is the **realised** home margin, put back on the pick
side by the same minimum-step rule when the realised margin lands on the other
side. It exists to prove the instrument can detect a centre move at all, so
that a null on the challenger can be classified honestly instead of by
assumption. Its effect size is reported, and `bounded_by_control` is only ever
claimed for an effect the control was actually shown able to detect.

**Decision rule, frozen.** The pool submits a tiebreaker every week either way,
so this is expected value, not a threshold (`AGENTS.md`: a promotion bar is not
a decision bar). If `probability_positive` exceeds 0.5 on the primary
closest-total metric and the closest-score metric is not resolvably worse, the
challenger is the better card to play and the MOD-17 row says so in those
words. Serving it is a separate step that rewrites a published number and is
not taken in this lane; what this lane ships is the measurement, the registry
rows and a prospective ledger that records both guesses every week. An interval
containing zero closes nothing and classifies as `unresolved_below_power`, per
the binding taxonomy.

**Prospective arm.** `nfl_ats.lattice_centre_challenger`, challenger id
`tiebreaker_lattice_centre`, ledger
`artifacts/prospective/lattice_centre_decisions.parquet`, recorded from
`nfl-ats publish-predictions --record-decisions` beside the shade and
served-total arms. Each week it stores both integer scores and the centre each
came from, and settles both against the realised final. It never writes the
card.

### Results (measured 2026-09-11, after the predeclaration above was saved)

**Measured** this session by `.\.tools\uv.exe run --no-sync python
scripts\mod17_lattice_centre_screen.py`; artifact
`artifacts/mod17_lattice_centre/20260911T043520Z/` (`predictions.csv`, one row
per scored game with both arms' cells, and `summary.json`). Scope: **1,537
graded games, 107 week blocks, seasons 2020-2025**, opener archive
`opener_evaluation/20260910T211255Z` for active model `d49194e04945a5e5`. The
per-week `gaussian_median` mapping inverts exactly: the largest gap between the
archive's own raw cover probability and the one rebuilt from the recovered
`(location, scale)` is **1.1e-16** over all 107 weeks. Every game produced a
pick-consistent cell on every arm -- zero hard-guard refusals, so no exclusions
and nothing arm-specific to disclose.

#### What this implies for the decision

**1. The challenger removes the C2 defect outright, and that part is resolved.**
The served centre sits strictly on the side the card plays in **85.0%** of games;
the challenger does so in **100%**. Paired, that is **+15.03 accuracy points,
95% [+13.01, +16.94], `probability_positive` 1.0000**. In counts: **231 of
1,537 games** (15.0%) were served with the lattice centred on the side the card
does not play. The pick-deciding margin fixes **188** of them by itself; the
remaining **43** are the ones where something moved the pick without moving any
margin -- in this archive the home-side offset, which the archive's own metadata
independently reports as changing 43 opener picks -- and the predeclared
minimum-consistent-step rule catches every one.

**2. It is not cosmetic.** The two arms print a different final on **747 of
1,537 games (48.6%)**.

**3. The cost is not measurable at this sample size.** Closest-total error moves
**-0.0215 points, 95% [-0.0523, +0.0116], `probability_positive` 0.0985**;
closest-score error moves **-0.0267 points, 95% [-0.0615, +0.0091],
`probability_positive` 0.0707**. Both lean to the served arm, neither is
resolved, and neither closes anything.

**4. The predeclared decision rule fires NO, and the served card was not
touched.** The rule hung on `probability_positive` > 0.5 for closest-total; it
is 0.0985. Playing the challenger on the strength of that metric alone would be
taking the 10/90 side. The card, `tiebreaker.json` and the published guess are
unchanged by this lane.

**5. And the positive control says that metric could never have said yes.**
Centring the lattice on the **realised** margin -- perfect knowledge of the
thing the centre is supposed to estimate -- moves closest-total error by
**-0.0221, 95% [-0.0580, +0.0132], `probability_positive` 0.11**: no gain
either. The total axis is pinned by the served total and the one-point
tolerance, so the margin centre barely reaches it. The same oracle moves
closest-score error by **+2.371, 95% [+2.149, +2.605],
`probability_positive` 1.0000**, so the score metric is the one that registers a
centre change at all. The rule is left standing exactly as written and its NO is
reported as its NO; what the control changes is the *next* predeclaration, which
should hang on closest-score, where the instrument demonstrably works.

#### The frozen table

Paired per-game difference, served minus challenger, positive favours the
challenger. `nfl_ats.clv.week_blocked_bootstrap`, block `week`, 2,000 resamples,
seed 20260816.

| metric | games | estimate | 95% low | 95% high | `probability_positive` |
|---|---|---|---|---|---|
| closest-total error (`mae_improvement`) | 1,537 | **-0.02147** | -0.05233 | +0.01163 | **0.0985** |
| closest-score error (`mae_improvement`) | 1,537 | **-0.02668** | -0.06153 | +0.00912 | **0.0707** |
| centre on the pick side (`accuracy_points`) | 1,537 | **+15.029** | +13.007 | +16.939 | **1.0000** |
| closest-total error, week's last game | 101 | -0.01980 | -0.12871 | +0.08911 | 0.3713 |
| closest-score error, week's last game | 101 | -0.01980 | -0.16832 | +0.12871 | 0.3978 |
| centre on the pick side, week's last game | 101 | +16.832 | +9.901 | +24.752 | 1.0000 |
| positive control, closest-total error | 1,537 | -0.02212 | -0.05797 | +0.01321 | 0.1100 |
| positive control, closest-score error | 1,537 | **+2.3709** | +2.1488 | +2.6045 | 1.0000 |

Raw levels: closest-total MAE **10.286 served / 10.308 challenger / 10.308
oracle**; closest-score MAE **14.643 / 14.669 / 12.272**.

Split by which branch of the predeclared centre rule fired:

| branch | games | closest-total | closest-score |
|---|---|---|---|
| `pick_deciding_point` (the served read's own `point`) | 1,494 | -0.0207 | -0.0234 |
| `minimum_consistent_step` (the flip rule) | 43 | -0.0465 | -0.1395 |

The flip branch's 43 games are the only place this comparison is measured on
games whose pick no margin supports, and 43 games carry no useful interval; it
is reported so the branch is not invisible, not as a result.

#### Recorded

Family `lattice_centre_challenger`, league `nfl`, seasons 2020-2025, all seven
rows via `nfl-ats weak-signals record`, all `unresolved_below_power`:
`mod17_lattice_centre_closest_total`, `mod17_lattice_centre_closest_score`,
`mod17_lattice_centre_pick_consistency`,
`mod17_lattice_centre_closest_total_last_game`,
`mod17_lattice_centre_closest_score_last_game`, and the two control rows
`mod17_lattice_centre_control_closest_total` /
`mod17_lattice_centre_control_closest_score` under category `control`.

No closing ground is claimed on any of them, and each one says why in its
`classification_evidence`. The two error metrics cross zero, which closes
nothing. `bounded_by_control` is not available either: on closest-total the
control could not detect even a perfect-information effect, and on closest-score
it was shown able to detect 2.4 points against a measured 0.027, about ninety
times smaller. The pick-consistency row is resolvably **positive**; the
classification enum has no state for that, so it takes the non-terminal option
and its evidence field says so, following the precedent in
`docs/score_lattice.md` section 2.8.

#### Week 1 confirmation (DEN at KC), served card untouched

**Measured** by running `record_lattice_centre_decisions` against the live
Week 1 forecast with the ledger redirected to a scratch directory, so no
repository ledger row was written by this lane -- the real row is written by the
session's own `nfl-ats publish-predictions --record-decisions`, which is also
what republishes the shade-correct `tiebreaker.json` that violation 3 above
needs.

DEN at KC is one of the four overlay-flipped games, so it exercises the flip
branch on its first live week. The card picks **DEN +2.5**; the served centre is
`predicted_margin` **+2.7045** (KC's side, the C2 violation); the pick-deciding
`point` is **+3.0736** (`residual_location` +0.369057), still on KC's side
because an overlay flip produces no margin; the predeclared rule therefore
centres the challenger at `ceil(2.5) - 1` = **+2.0**, strictly on DEN's side.

| served total | served arm | challenger arm |
|---|---|---|
| 43.7238, the currently published pre-shade total | KC 23 - DEN 21 | KC 23 - DEN 21 |
| 42.7238, the total the code serves now | KC 22 - DEN 20 | KC 22 - DEN 20 |

**Both arms agree on this week's guess**, under both totals. The challenger
changes nothing a reader sees in Week 1; what it changes is that the number is
now built around a margin that backs the team the card actually plays, and that
both guesses are on the record before the deadline either way.

#### What is wrong with it

- **The closest-total metric is the wrong instrument and the predeclared
  decision rule was hung on it.** That is a predeclaration error, not a finding,
  and it is left in place rather than swapped after the fact. The control is the
  evidence for the claim.
- **Historical overlay flips are not in this read.** The archive's only
  pick-moving-without-a-margin mechanism is the home-side offset, 43 games. Real
  overlay flips (four of sixteen on the live Week 1 card) are a much larger
  share of the served card than 2.8%, so the flip branch is under-weighted
  historically and is really only tested prospectively.
- **The total axis is the market total, not the served joint-residual total
  plus the shade.** Both arms see the identical number, so the paired contrast
  is unaffected, but the absolute error levels here are not the levels the live
  tiebreaker would post.
- **The historical read grades at the opener line**, which is the project's
  declared primary grade, while the live tiebreaker is built on the served
  spread. Same construction, different line.
- **The week's-last-game population is 101 games and its intervals are wide
  enough to be compatible with almost anything.** It is reported because it is
  the population the pool actually scores, not because it settles anything.
- **`predicted_margin` is still what production centres on.** Nothing in this
  section changed a served number; the ledger accrues from this week's publish
  and the decision is revisited when it has rows.
