# Home-side probability mapping: frozen lane T predeclaration

Family `mod18_home_side_mapping_v1`. Frozen before scoring, 2026-09-07 evening.
Read: `artifacts/active_ats_model.json` names model `a4c757efd2525da6`,
`probability_method` gaussian_median, forecast
`margin_predictions/2026-week-01-20260907T164352Z`. Measured:
`find_matching_opener_evaluation` returns `opener_evaluation/20260907T152026Z`
(1,537 games, 1,503 non-push, 2020-2025). Read: ROADMAP.md row MOD-18 names the
played policy `overlay_union_coach_division_revenge_player_arrests_v2`.

## Why this lane exists

Lane L's Diagnosis D and lane P's home split (read: `docs/hybrid_margin_mapping.md`,
lane P report) locate the served probability's calibration gap by HOME SIDE, not
by spread size alone: on 7.5-10 spreads the home underdog covered 59.4% against a
stated 44.9% (69 games, +14.5 pp [+3.0, +26.3], P+ 0.993) while home favourites
there show no gap (-0.1 pp, 125 games). Every prior candidate (K1-K3, H1, H2, C2)
conditioned on the LINE only. These two arms condition the probability SHAPE on
which side of the line the home team is. The served POINT prediction is not
touched by either arm; the point location is lane S's territory, and this lane
owns only the probability around whatever point the incumbent produces.

## Arms (two; no other variants, no per-bucket pick flipping, no sign selection, no tuning)

Let h be the home side of a game's line (home favourite when the home spread is
positive, home underdog when negative, pick'em when zero) and b the lane-J spread
band from `nfl_ats.spread_regime.spread_bucket` (0-3, 3.5-6.5, 7, 7.5-10, 10.5+;
7.5 belongs to the upper band).

**T1 `smooth_home_side_shift`.** Cover probability = the incumbent gaussian_median
mapping evaluated at (point prediction + s(h, b)). s(h, b) is the shrunken mean of
the incumbent's realised residual against its own mapping centre, `result -
(point + gaussian_median location)`, over prior completed games in that cell:
s = sum(residuals) / (n + 100). The prior weight of 100 games toward zero is
fixed here, before scoring, and is regularisation, not a fitted parameter. A cell
with no prior games shifts by zero. The shift moves only the mapping's centre for
probability purposes; the served point prediction is untouched.

**T2 `lattice_home_side`.** Lane K's K1 conditional integer-margin read
(`nfl_ats.conditional_margin.fit_conditional_margin`, 2.5-point Gaussian kernel
over prior predicted margins, integer margin atoms kept, half-push decision
credit) fitted on the prior games with the SAME home side h as the target, so the
discrete shape near 3, 7 and 10 is learned per side. Fallback to the unconditioned
K1 (all prior games) when the same-side support in the kernel window is under 200
games, where support = the number of same-side prior games whose predicted margin
lies within two bandwidths (5 points) of the target centre. That floor is
declared regularisation, not a significance bar. Pick'em lines never reach 200
same-side games and always fall back.

## Prior history and exclusions (identical to lanes K and L)

For a target week, eligible history = completed games from the five trailing
seasons including prior completed weeks of the target season, excluding the whole
target week, the target game ids, and any game not completed one day before the
target week's first game (`gameday + 1 day < first target gameday`). Both arms
are fitted once per week from that history. Training pairs are lane K's cached
prior-only centre stream (weekly refits, sha256
`560de573b9523834a98f881b73d441653ffd986314151ae2b2c2d0bbe36120c0`, reused with
its digest recorded): `predicted_margin` there is the mapping centre (raw point +
gaussian_median location) and `result` the actual home margin. On archive rows
the archived opener probability is the incumbent read; the cached point differs
from the archive's by at most 0.0173 points on 15 games (measured by lane L).
Pre-2020 history uses the nflverse line the cache carries; this line-vintage
difference is stated, not hidden.

On the archive, T1 is evaluated exactly: the incumbent's per-week sigma is
recovered from the cached stream as (centre - line) / Phi^-1(p) (constant within
each week to 1e-12, measured), and the shifted probability is
Phi(Phi^-1(p_incumbent) + s / sigma), which is the same number as re-reading the
Gaussian at the shifted centre.

## Measurement (identical to lanes K and L)

Isolated `NFL_ATS_ARTIFACTS_DIR` / `NFL_ATS_REGISTRY_DIR` seeded with the active
manifest and copies of both registries; `data/` read-only. Both arms post-process
the archived out-of-time point predictions, so no refit is needed.

- Paired forced-pick opener accuracy versus the active model's own opener picks
  (`pick_home_at_open_probability_rule`) on the 1,537-game archive (1,503
  non-push); week-block bootstrap 20,000 draws, seed 20260817, within-week
  correlation ZERO, no variance inflation; frozen-pick within-week outcome
  permutation null 2,000 draws, same seed, including the maximum over both arms
  (descriptive selection context, never a decision threshold).
- Brier and log loss on non-push rows, improvement = incumbent loss minus
  candidate loss (positive favours the candidate).
- Reliability by spread band and by pick side (favourite / underdog), and the
  table this lane exists to move: by spread band AND home side (home favourite /
  home underdog), the calibration gap (realised home cover rate minus mean stated
  home probability, plus pick accuracy minus stated confidence) for the
  incumbent, K1, T1 and T2, with week-block intervals. The question is whether
  the 7.5-10 home-underdog gap of +14.5 pp shrinks without the 0-3 and 3.5-6.5
  bands getting worse.
- Per-season deltas; push-rate check for T2 at |line| = 3 and 7 (predicted push
  probability versus the realised push rate, MOD-05's recentring constraint; T1
  carries no push mass and reports zero there).
- Through the PLAYED three-member card: `nfl-ats overlay-composition` on a scratch
  opener evaluation per arm; composed picks compared with the incumbent's composed
  picks on the same archive; no subset search.
- Week 1 2026: read-only scratch `margin-predict` (gaussian_median) into the
  isolated artifacts root, checked equal to the active forecast, then both
  mappings applied to that output with the same prior-only history; model-side
  differences reported. Not a production CLI integration.

Rotation: declare the family and assign the first block before scoring; score the
frozen arms on 2020-21, 2022-23, 2024-25 in order, recording each look before the
next assignment; record every arm, window, season, loss, composed-card and
bucket/home-side cell through `nfl-ats weak-signals record` with a pool-player
plain summary; every unsupported cell is `unresolved_below_power`. Calibration
gaps themselves are signed diagnostics with no admissible registry unit and are
published in the report and artifact, not recorded as effects (lane L's
correction); per-cell candidate-versus-incumbent Brier improvement and accuracy
are the recorded cell effects.

Mined-archive discount, stated plainly: the 2020-2025 opener pool has been spent
by the spread-regime, conditional-margin, hybrid and home-dog-location families;
the active model, the H/K/L/P diagnosis that motivated the home-side split, the
lattice, and the overlay subset were all selected on this archive. This is
descriptive reused evidence with a stated discount, not independent
confirmation; no invented numerical discount is applied.

## Decision rule

Expected value at the opener through the played card: P+ above 0.5 favours
playing an arm. Ranking by composed-card point delta, ties T1 then T2. No
promotion in this lane; the active model, weekly.py, the served probability
method, the played profile and the card are not changed.

## Closing-grounds taxonomy (verbatim, binding)

An interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval on the wrong
side of zero) or zero split-half reliability; (2) bounded by a positive control
proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". The registry code
hard-rejects inadmissible closures; if a record command errors, the verdict is
wrong, not the validator. Never use 95%, 0.90, or any threshold as a DECISION
bar; decide on expected value (`probability_positive` above 0.5 favours playing
it), thresholds only govern what docs may CLAIM. Grade at the OPENER (the pool's
grade); a close-graded number may never veto a play. Within-week game
correlation is ZERO by owner mandate: never estimate or pad it. Never say
something "needs N more games": the data is fixed and the project is
model-limited.
