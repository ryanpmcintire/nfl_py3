# Best Pick: rank by the card's own decision score (lane BP-RANK)

Written 2026-09-14, before any number in **Results** was computed. Owner
prompt the same day: unconvinced by the Best Pick methodology after the
Week 1 nominee (MIA +3.5 at LV) lost.

## Why this exists

The served nominator ranks eligible games by `candidate_dist`, the distance
from 0.5 of a **separate** `market_residual` refit at `ridge_alpha = 2000`
(`nfl_ats.best_pick_nomination.NOMINATION_RIDGE_ALPHA`). The card the reader
sees is the alpha=10 `weak_stack` model, and its own per-game number is the
`Decision score` column. The two are different models, so the star can land
on a game the card does not consider its most confident eligible pick.

That is exactly what happened on the 2026 Week 1 card (read,
`git show HEAD:CURRENT_PREDICTIONS.md`): the nominee MIA +3.5 carried a
Decision score of **55.0%** while seven other games on the same card showed
**57.0-57.1%**. A reader is told the star is the week's best pick and the
board simultaneously tells them six or seven other picks are more likely to
land. Whatever the archive says, that is incoherent on its face.

The archive already says the ranker itself is the weak part (read,
`docs/best_pick_composed_rule.md` § Results): unfiltered, `candidate_dist`
ranks at **50.96% (53/104)** and the unfiltered dispersion tie-break at
**50.00% (52/104)**, against a 52.66% all-pick week average. All the measured
lift sits in the two SCREENS — below-median cross-book dispersion (chooser 6
alone, 57.28%) and spreads of 6.5 or less (B2, 58.82%, read
`docs/best_pick_bucket_confidence.md` § Results). Nothing has ever scored the
obvious ranker: the card's own served probability.

This lane changes a **RANKING only**, never a side. Every forced pick stays
as published; only which single game carries the star can move.

## Binding closing-grounds taxonomy, verbatim

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close
a line of work: (a) refuted mechanism — a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (b) bounded by a
positive control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it, report `probability_positive`, never
"contains zero". Within-week correlation is ZERO; week-blocked bootstrap.
Decide on expected value: the pool forces one nomination a week, so the arm
with the higher expected Best-Pick accuracy is served; 0.90-style thresholds
govern only what the docs may claim.

## Population

The frozen 2020-2025 opener archive, no new fitting:

- `artifacts/ridge_alpha_promotion/20260818T221459Z/opener_paired.parquet`
  — 1,537 games, 107 REG weeks, supplies `candidate_prob_open` (the alpha=2000
  ranking score the incumbent uses) and the frozen `tue_open_home_spread`.
- `artifacts/odds_microstructure/20260818T225430Z/spread_novig_tue_open.parquet`
  — cross-book Tuesday opener `spread_std`, fed through production's own
  `nfl_ats.best_pick_nomination.dispersion_pool_from_frame`.
- `artifacts/opener_evaluation/20260913T134234Z/per_game.parquet` — the ACTIVE
  model's served opener stream, supplying both the served decision score
  (`home_cover_probability_at_open`, the bucket-adjusted probability rule the
  card prints) and the served grade (`correct_at_open_probability_rule`).

**Multiplicity, stated plainly: this is the SIXTH reuse of this ~107-week
opener population for the Best Pick family** (ridge_alpha promotion look, the
odds-microstructure battery, the 2026-08-18 ranker screen, the 2026-08-19 v3
audit, the POL-09 composed-rule scoring, the 2026-09-09 bucket-confidence
lane). Every number below carries that compounding look-reuse discount. No
rotation window is assigned or spent: this scores a ranking inside an
already-played rule, it is not a new-signal look.

## Predeclared arms (exactly three plus a control)

Every arm draws from the SAME eligible pool the production rule builds:
below-median cross-book `spread_std` for the week (production's
`dispersion_pool_from_frame`, including its documented fallbacks), then the
served small-spread screen `abs(tue_open_home_spread) < 7.0`, falling back to
the unscreened pool when the screen empties it. Arms differ ONLY in the score
they rank that pool by, descending, tie-broken by lower `spread_std` then
`game_id` — production's own tie-break.

- **A0 — incumbent, as served today.** Rank by
  `abs(candidate_prob_open - 0.5)`. This is v2 plus the B2 small-spread
  screen, the rule live since 2026-09-09.
- **A1 — served decision score.** Rank by
  `abs(home_cover_probability_at_open - 0.5)`: the card's own number, the one
  the reader is shown. This is the arm the owner's objection names.
- **A2 — served decision score, dispersion screen removed.** A1 ranked inside
  the small-spread screen alone. Separates "does the card's number rank
  better" from "does the dispersion screen still earn its place once the
  ranker is the card's own number".
- **Control — perfect foresight** inside the A0 pool, to show the instrument
  can see an effect at all on ~100 weeks.

## Predeclared grade and metric

PRIMARY: top-1 weekly hit rate on the **served** grade
(`correct_at_open_probability_rule`); a week whose nominee pushed is dropped
from that pair only. Effect = accuracy points, candidate minus A0, paired
within week, `clv.week_blocked_bootstrap`, week blocks, 20,000 draws, seed
20260914. Report the effect, the 95% interval and `probability_positive`
before any one-word verdict.

Reported alongside, never gated on: the per-season split, the count of weeks
where the nominee differs, and the descriptive "star versus the card's most
confident eligible pick" disagreement rate that motivated the lane.

## Decision rule, fixed in advance

The pool forces one nomination a week. The arm with the higher expected
Best-Pick accuracy is served, `probability_positive > 0.5` being the whole
bar. If A1 leads A0, A1 is served and the card's star becomes the card's own
most confident eligible pick. If A0 leads, the incumbent stands and the
**board must stop implying otherwise**: the coherence defect is then fixed in
the reader-facing text, not in the ranking.

## Results

Measured 2026-09-14. Artifact
`artifacts/best_pick_served_score_ranker/20260914T163425Z/`
(`summary.json`, `weekly.csv`, `weekly.parquet`). Command:
`uv run python scripts/best_pick_served_score_ranker_eval.py`.
107 weeks built, 1,537 games, nothing refit.

**Decision: A1 -- rank by the card's own decision score -- has the higher
expected Best-Pick accuracy and is served.** It scores **62.14% (64/103)**
against the incumbent's **55.88% (57/102)**: **+5.94 accuracy points, 95%
week-blocked [-1.98, +13.86], `probability_positive` 0.934** on 101 paired
weeks. Season-blocked: [-2.02, +14.85], `probability_positive` 0.915
(measured 2026-09-14 audit).

**The evidence in games, stated plainly (audit, measured from `weekly.csv`):
the two rankers graded differently in only 16 of the 101 paired weeks, and
A1 went 11-5 on those.** Every other week the two stars either coincided or
both won or both lost together. The +5.94 headline is that 11-5 spread over
101 weeks. Leave-one-season-out keeps the sign in every fold (+2.38 to +8.33),
but 2024 and 2025 are exact ties, so the two most recent seasons carry none
of the edge. **The +5.94 is not evidence for the ranker**: it is the best of
three arms on a population this family has scored six times, with no
held-out season, and 16 decisive weeks cannot separate rankers at this
evaluator's resolution. The owner rejected it as overfit on 2026-09-14. The
only reason the card's own most confident pick carries the star is that a
star must be placed every week and this ranking has no fitted parameter and
is the one the board can state truthfully; the alpha=2000 ranker keeps
recording as the paired challenger and neither has out-of-sample support.

| Arm | Weeks won | Hit rate | Effect (acc. pts) | 95% week-blocked | `probability_positive` | Nominee differs |
|---|---:|---:|---:|---|---:|---:|
| **A0 incumbent (alpha=2000 distance)** | 57/102 | 55.88% | -- | -- | -- | -- |
| **A1 card's own decision score** | **64/103** | **62.14%** | **+5.94** | [-1.98, +13.86] | **0.934** | 37 |
| A2 A1 without the dispersion screen | 65/105 | 61.90% | +5.88 | [-3.92, +15.69] | 0.879 | 60 |
| Positive control (perfect foresight) | 104/107 | 97.20% | +41.18 | [+31.37, +50.98] | 1.000 | 82 |

A0 reads 55.88% here against the 58.82% the 2026-09-09 bucket-confidence lane
recorded for the same rule, because that lane graded against the
`20260909T183120Z` opener stream of model `c657058903f3232b` and this one
grades against `20260913T134234Z`. The comparison that decides anything is
paired inside one run, and both arms here share the same grade.

### The coherence defect, measured

In **37 of 101** graded weeks the incumbent's star sits on a game that is NOT
the card's most confident eligible pick -- a third of all weeks where the
board tells the reader one thing and the star says another. On those 37 weeks
the incumbent hit **48.65%** and the card's own top pick hit **64.86%**. When
the two agree the incumbent hits 59.38%. This is not independent
corroboration of the headline: a paired difference can only come from weeks
where the two stars differ, so the 37-week subset restates the same 16
decisive weeks at higher variance. It is kept as the description of the
display defect (a star on a game the card ranks below others), nothing more.

### Season split (never gated on)

| season | A0 | A1 |
|---|---:|---:|
| 2020 | 47.06% | 70.59% |
| 2021 | 55.56% | 72.22% |
| 2022 | 52.94% | 55.56% |
| 2023 | 47.06% | 41.18% |
| 2024 | 75.00% | 75.00% |
| 2025 | 58.82% | 58.82% |

A1 ahead in three seasons, level in two, behind in one.

### Registry

Recorded before this write-up, family `bp_rank_served_score_v1`:
`bp_rank_served_score_vs_alpha2000` (+5.9406, P+ 0.9336),
`bp_rank_served_score_no_dispersion_vs_alpha2000` (+5.8824, P+ 0.8787),
`bp_rank_foresight_control_2020_2025` (+41.1765, P+ 1.0). All
`unresolved_below_power`: no resolved wrong sign, and the foresight control
sits an order of magnitude above these effects, so it bounds nothing here.

### What was shipped

- `best_pick_nomination._nominate` now ranks the eligible pool by
  `ranking_score`, which is the served displayed pick probability's distance
  from 0.5 when the card carries one, and the alpha=2000 `candidate_dist`
  otherwise. The alpha=2000 refit still runs and still feeds the
  `best_pick_nomination_v2` / `v3` challenger ledgers, so the old rule keeps
  recording as the paired OFF arm.
- The nomination now runs **after** the production overlays and on the frame
  the card actually serves. It previously ran on the pre-overlay predictions,
  so an overlay could flip the starred pick's side after the star was placed.
- The star cannot move once its game is past the pick deadline
  (`published_picks.locked_best_pick`, `card_view.apply_locked_best_pick`).
  Week 1's star therefore stays on MIA +3.5, which is what the pool graded,
  and the card says so in plain words.
- Reader-facing text on the card, the board and the assistant glossary now
  says the star is the pick the card is most confident in among the close
  games the books agree on, because that is now what it is.

### Week 1, honestly

The 2026 Week 1 star MIA +3.5 lost (LV by 14). Had A1 been live, the star
would have been IND +3.5 at 57.1%, and BAL won by 18, so it would have lost
too. One week decides nothing either way; the 101-week paired read is the
evidence.
