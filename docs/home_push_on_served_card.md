# Is the served home-side push still the right setting on the nine-member card?

MOD-18 lane AW. Predeclared 2026-09-09 before any arm was scored.

## The situation

The served model applies a walk-forward **home-side offset** to the point
forecast on spreads of 7 or more before the cover probability is formed
(`HOME_SIDE_OFFSET_BUCKETS = ("7", "7.5-10", "10.5+")`, policy
`home_side_offset_big_spreads_v2`, `src/nfl_ats/home_side_location.py:35,154`;
the served read in opener evaluation is `src/nfl_ats/clv.py:2176-2203`, gated by
`evaluate_at_opener(..., home_side_offset=...)` which defaults to the policy
flag `HOME_SIDE_OFFSET_SERVED = True`). Each week's per-bucket offset is the
shrunken mean of `result - point_incumbent` over the prior five seasons'
already-scored games, 100 pseudo-games toward zero, target week excluded.

It was promoted on numbers taken against the **three-member** card: +0.80
accuracy points standalone, week-blocked [-0.20, +1.81], `probability_positive`
0.934, and +0.33 through the played card, [-0.60, +1.29], P+ 0.736 (read:
`docs/home_side_offset_promotion.md:224,226`). Two things changed since.

1. The played card is now the **nine-member** composition
   `overlay_union_coach_division_arrests_bye_coldvisitor_protection_interim_tank_precip_v3`
   (read: `src/nfl_ats/four_overlay_composition.py:1,16-26`). Every member
   conditions on the model's own probability, so the push changes which games
   the card flips, and the card can undo a push flip.
2. Two lanes finished today on the same archive and both say the big-spread
   edge is a *home* effect. The home team covers **57.7%** of the 130 non-push
   10.5+ games 2020-2025, and the arms that gain there gain by tilting home:
   home-pick share at 10.5+ rises from the served **59.23%** to
   60.77 / 66.92 / 70.00 / **73.08%** across the shrinkage arms (read:
   `docs/line_size_shrinkage.md:341-342`), while at 10.5+ the served model is
   50.77% and the offset itself is worth about +3.1 points to the incumbent
   there (read: `docs/big_spread_signal.md:72`).

So the question is not "does a home tilt help on big spreads" — two lanes say
it does — but whether **this particular carrier, at these particular buckets,
is still the honest one now that six more members sit on top of it.**

## The question

> On the card that is actually played, which home-side push setting has the
> higher expected forced-pick accuracy at the Tuesday opener, and by how much?

## Arms (frozen before any number is read)

All four share one walk-forward. The offset is always fitted from the arm's own
**raw** out-of-time stream (`point_incumbent = opener line + raw residual`), so
the fitted per-bucket values are identical across arms; the arms differ only in
which buckets are allowed to carry the fitted value.

| arm | setting | code equivalent |
|---|---|---|
| **H0 (served)** | push on for `7`, `7.5-10`, `10.5+` | `fit_home_side_offsets(prior)` |
| **H1** | push off everywhere | `evaluate_at_opener(..., home_side_offset=False)` |
| **H2** | push on for every bucket | `fit_home_side_offsets(prior, all_buckets=True)` (lane S's S2) |
| **H3** | push on for `10.5+` only | offsets masked to the one bucket |

H0 is the baseline of every paired comparison. H1/H2/H3 are the candidates.

## Surfaces

- **standalone** — the model's own opener pick, no card.
- **card** — the nine-member OR union **recomputed on each arm**
  (`scripts/spread_hole_arms.py --card served` semantics, i.e.
  `nfl_ats.unserved_tilt_marginals.served_card_flip_set`), never inherited from
  the served arm. The card is what is played, so it is the deciding surface.

## Population and grade

`artifacts/opener_evaluation/20260910T005854Z` (active model
`2e8c616b476dd0d2`, weak_stack market-residual ridge alpha 10,
`gaussian_median`), 1,537 archived games 2020-2025, **1,503 non-push at the
opener**. Graded at the **Tuesday opener** by the production probability rule
(`home_cover_probability >= 0.5`). The replay must reproduce the archive before
any arm is read: maximum residual gap and maximum probability gap against
`per_game.parquet` are reported first, and the target is 0.0 on the residual.

## What gets reported

Paired accuracy-point deltas (candidate minus H0) on the games both arms score,
**week-blocked and season-blocked** bootstraps, 20,000 samples, seed 20260821,
using `nfl_ats.overlay_composition.blocked_bootstrap_matrix` — the project's own
resampler. Within-week correlation is zero by mandate; the week block is the
unit either way.

Cells, on both surfaces:

- **overall** and by bucket `0-6.5`, `7`, `7.5-10`, `10.5+`
  (`nfl_ats.spread_regime.spread_bucket`, coarsened as
  `scripts/spread_hole_arms.py` coarsens it);
- by era **2020-21 / 2022-23 / 2024-25** (magnitude per era, never presence);
- alongside every arm: **home-pick share**, **favourite-pick share**, **Brier**
  and **log loss**, and picks changed.

Plus two things the promotion never showed:

- **Overlap.** How many of the push's flips (games where H0's pick differs from
  H1's) are also flipped by one of the nine members, per member, and whether
  the card's final pick on those games ends up on the push's side or back on
  the no-push side. A push whose flips are mostly re-flipped by the card is
  redundant with it, whatever its standalone number says.
- **Positive control.** Perfect foresight on the push's own flip set: take
  H1's card and pick the winning side on exactly the games the push touches.
  That is the ceiling any push setting could reach on that flip set, and it is
  the only thing that can bound a null here.

## Decision rule, declared now

The pool is forced picks and the decision bar is **expected value on the played
card**, not a promotion threshold. The setting with the higher expected opener
accuracy on the **card** surface is the one that should be served, and its
`probability_positive` is reported as the strength of that preference. An
interval containing zero is not a result and never closes an arm.

If the winner is not H0, the report names exactly what changing it takes
(`HOME_SIDE_OFFSET_BUCKETS`, the policy id `home_side_offset_big_spreads_v2`,
the Model page's home-side push table, then opener-evaluation + composition +
publish) and which Week 1 picks move — read-only, in the scratchpad. **This
lane does not change served behaviour.**

## Closing-grounds taxonomy (binding, verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (a) refuted mechanism — a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (b) bounded by a positive
control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it, report `probability_positive`, never
"contains zero".

Every cell is recorded with `nfl-ats weak-signals record` under the name
`home_push_on_served_card_<arm>_<bucket|overall>_<window>`.

---

# Result

Measured 2026-09-09,
`artifacts/home_push_on_served_card/20260910T033757Z/`. Nothing served was
changed by this lane.

## The replay is exact

Joined game for game against
`artifacts/opener_evaluation/20260910T005854Z/per_game.parquet` on all **1,537**
archived games: maximum line gap **0.0**, maximum residual gap **0.0**, maximum
home-side-offset gap **0.0**, maximum probability gap **0.0**. The H0 arm is the
served archive, not an approximation of it.

## Decision — keep the served setting

On the played card, 1,503 non-push games, week-blocked bootstrap, 20,000
samples, seed 20260821:

| setting | card accuracy | delta vs served | week 95% | P+ candidate | P+ served |
|---|---:|---:|---|---:|---:|
| **H0 served** (7 / 7.5-10 / 10.5+) | **56.89%** | — | — | — | — |
| H1 push off | 56.22% | -0.665 | [-1.211, -0.133] | 0.0068 | **0.993** |
| H2 every bucket | 56.89% | +0.000 | [-0.741, +0.736] | 0.5024 | 0.498 |
| H3 10.5+ only | 56.62% | -0.266 | [-0.658, +0.067] | 0.0725 | **0.928** |

Standalone, the same ordering with wider bands: H1 -0.599 [-1.494, +0.266]
P+ 0.091, H2 -0.798 [-1.799, +0.201] P+ 0.061, H3 -0.333 [-1.053, +0.331]
P+ 0.166.

One cell closes on an admissible ground:
`home_push_on_served_card_h1_push_off_card_overall_2020_2025` is
`refuted_mechanism` / `wrong_sign_resolved` — the week-blocked
[-1.211, -0.133] and the season-blocked [-0.913, -0.351] interval both sit
entirely below zero. Every other recorded cell is `unresolved_below_power`.

## The push is a signed per-bucket bias correction, not a home tilt

Mean served offset by bucket and season:

| season | 0-6.5 | 7 | 7.5-10 | 10.5+ |
|---|---:|---:|---:|---:|
| 2020 | 0.000 | -0.213 | +0.367 | -0.022 |
| 2021 | 0.000 | -0.133 | +0.080 | +0.488 |
| 2022 | 0.000 | -0.624 | +0.584 | +1.178 |
| 2023 | 0.000 | -0.671 | +0.549 | +1.391 |
| 2024 | 0.000 | -0.399 | +0.332 | +1.471 |
| 2025 | 0.000 | -0.155 | +0.694 | +1.529 |

At 10.5+ the home team covers 58.02% of the 131 scored games and the push takes
the model's home-pick share from 41.98% to 58.78% — essentially onto the base
rate — lifting card accuracy there from 46.56% to 51.15%. At exactly 7 the
offset is NEGATIVE, an away push, against a 47.30% home cover rate on 77 games.
The policy follows the bucket's measured bias in both directions.

Split-half reliability of that trait, at the level the policy operates (one
offset per bucket): odd/even-week halves of 2020-2025 give r = +0.508
(Spearman-Brown +0.673), week-blocked 95% [-0.713, +0.947], P(r>0) 0.72; the
2020-22 vs 2023-25 split gives r = +0.710 (SB +0.831). Within one season the
same split is at the noise floor (r = -0.029 over 30 season-by-bucket cells,
95% [-0.357, +0.371]) on ~11 games a cell. No `no_split_half_reliability`
closure is available against this family.

## Overlap with the six new members

The push changes the model's own pick on **43** of 1,503 games (24 at 10.5+, 12
at 7.5-10, 7 at 7). **16** of those are also flipped by a card member
(`division_revenge_tilt` 8, `bye_edge_fade` 4, `coach_fade` 3,
`tank_zone_fade_tilt` 2, `pbp08_protection_mismatch_tilt` 1). After the card is
recomposed on each arm only **20 of the 43** still change the played pick; on
the other 23 the card lands on the same side either way. On those 43 games the
played card is right 69.77% with the push and 46.51% without. Union flip counts
barely move (H0 487, H1 490, H2 489, H3 488), so no arm destabilises the
composition.

## By bucket and by era, on the played card

| bucket | n | with push | without | push worth | week 95% |
|---|---:|---:|---:|---:|---|
| 0-6.5 | 1104 | 58.79% | 58.79% | 0.00 | push is zero there |
| 7 | 74 | 50.00% | 45.95% | +4.05 | [-9.459, 0.000] |
| 7.5-10 | 194 | 52.58% | 52.06% | +0.52 | [-2.874, +1.604] |
| 10.5+ | 131 | 51.15% | 46.56% | +4.58 | [-9.565, 0.000] |

| era | n | with push | without | push worth | week 95% |
|---|---:|---:|---:|---:|---|
| 2020-21 | 456 | 57.24% | 57.02% | +0.22 | [-0.664, 0.000] |
| 2022-23 | 514 | 59.14% | 58.17% | +0.97 | [-1.789, -0.198] |
| 2024-25 | 533 | 54.41% | 53.66% | +0.75 | [-2.030, +0.557] |

Magnitude varies, sign does not. Era 1 is a warm-up artifact rather than a weak
era: the offset stream starts empty in 2020, so the push flips 1 pick in 2020
and 3 in 2021 against 10 / 10 / 8 / 11 in 2022-2025.

## Calibration

All lines, 1,503 games: model Brier 0.25158 (H0) / 0.25174 (H1) / 0.25231 (H2)
/ 0.25155 (H3); card Brier 0.24696 / 0.24682 / 0.24680 / 0.24705. The arms sit
within 0.0008 of each other and the card ordering is the reverse of the
accuracy ordering — the push moves the side without improving the stated
probability, which is what a point shift under a smooth Gaussian read does. The
place to fix the 10.5+ probabilities is the discrete lattice read, not a bigger
point shift.

## Positive control

Perfect foresight on exactly the 43 games the push flips: +0.865 accuracy points
over the served card, week [+0.462, +1.327], P+ 1.0000; +5.344 at 10.5+,
[+1.709, +9.375], P+ 0.9997. The instrument resolves an effect of that size on
this flip set, and there was no null to bound — the served push captures 0.665
of the 1.530 points available against push-off, 43% of the ceiling. No cell in
this lane is `bounded_by_control`; the control is recorded so a future candidate
on the same flip set can be.

## Week 1 2026, read-only

Only 2 of 16 games sit in a served push bucket (ARI@LAC 9.5 and CLE@JAX 8.5,
both 7.5-10, offset +0.785) and neither changes side. Re-deriving every setting
from the served sidecar's own recovered `gaussian_median` location and scale
(the 16 published probabilities reproduce to 8.1e-12): H1 and H3 change **no**
Week 1 pick; H2 changes **one**, `2026_01_NE_SEA` from NE to SEA (0.4963 ->
0.5084).

## Recorded

102 cells, family `mod18_home_push_on_served_card_v1`, argv lists in
`record_commands.json`; verified in the registry as 101
`unresolved_below_power` plus the one `refuted_mechanism` /
`wrong_sign_resolved` cell above. Twelve of them first failed the validator
with `non-positive standard_error` — era-by-bucket subsets where the two arms
made identical picks, so every bootstrap draw is exactly zero — and were re-run
with `--standard-error` omitted rather than with a fabricated one; a dead heat
records as effect 0.0 at `probability_positive` 0.5. 48 further (arm, bucket)
combinations were not recorded and are listed in `record_skipped.json`: they are
structural identities where both arms carry the same offset by definition, so a
row there would be a definition entered as a measurement.

