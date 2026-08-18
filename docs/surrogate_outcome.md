# Does opener-to-close line movement work as a surrogate outcome?

Investigated 2026-08-18. The project's binding constraint is statistical
power: forced-pick ATS accuracy is a noisy label (margins scatter around any
pregame expectation with sd ~13 points), so ~130-150 tests have mostly
produced "unresolved" verdicts. The idea tested here: use the market's own
opener-to-close movement as a second, quieter label on the same games, since
movement has sd on the order of 1-2 points. Before quantifying any power
gain, this document establishes whether the surrogate is valid at all --
whether a model that is good at anticipating where the line moves is
actually good at picking games, or just good at reproducing the market's own
information a few days early.

## Verdict

**The movement label is a real but weak surrogate, and it fails the
efficiency test the idea needed to pass.** It carries a probability-positive
(not proven) signal about which candidate models are better (Step 1), and it
is demonstrably gameable by market-imitation, confirmed directly with an
adversarial model built for the purpose (Step 2). The close is measurably
sharper than the opener, so movement is not pure noise (Step 3). But once
the surrogate's own noise is honestly measured -- not assumed -- resolving
a 1.3-LINE-point effect would need **~3,300 opener-paired games under the
movement label against ~590 under direct forced-pick accuracy**, on an
archive that has 1,537 -- so the movement channel is roughly **5.6x less
efficient than the label it was proposed to replace**. (See the unit
correction below: 1.3 *line* points is not the injury lead's size. The
injury lead is 1.3 *accuracy* points, which **neither** channel resolves on
data in hand -- the direct label needs ~2,200-3,300 paired games.)
**Do not adopt it as a power lever or an efficiency multiplier.** It may
serve only as a narrow, always-secondary sanity check, per the rule at the
end of this document -- never as a promotion or rejection criterion on its
own.

## Data and method

All three questions and the adversarial control were coded before any of
them were run; the code and constants below are exactly what produced every
number in this document, one pass, nothing retuned.

**Population**: every game in the purchased snapshot archive with a paired
`tue_open` consensus and a resolvable close, 2020-2025, the same 1,537-game
set `docs/opener_evaluation.md` uses (confirmed by cross-check: this
document's `player`-profile numbers reproduce that doc's 52.50%
opener / 51.09% close exactly, and the promoted `weak_stack` reproduces
52.83% exactly). **That is the entire usable sample; opener coverage does
not exist before 2020.** All three questions and the efficiency arithmetic
run on this same population; nothing here draws a fresh rotation window
(read-only re-measurement of an already-fixed pick-stream generator, exactly
as `docs/opener_evaluation.md` frames its own admissibility).

**Surrogate-outcome label**: `movement_agreement` -- did a model's forced
pick direction match the direction the closing line actually moved?
Directionally identical to the existing movement-oracle convention in
`nfl_ats.clv.opener_pick_evaluation` (`open_move > 0` <=> "pick home"), NaN
(excluded) when the line never moved. Implemented in
`src/nfl_ats/surrogate_outcome.py::movement_agreement`.

**Step 1 candidates**: 13 existing, already-built feature profiles spanning
a real quality range -- `base`, `pbp`, `player_qb`, `player_injuries`,
`player_continuity`, `player_qb_injuries`, `player_qb_continuity`,
`player_injuries_continuity`, `player`, `player_injury_value`,
`player_value`, `player_participation`, `weak_stack` -- each run through the
unmodified `opener_pick_evaluation` (ridge, alpha 10, `market_residual`
target, `min_train_games=500`), all on the identical 1,537 games. No new
model was built for this step; nothing about the active model or any pick
changed.

**Step 2 adversarial control**: `fit_movement_target_model` (new, in
`src/nfl_ats/surrogate_outcome.py`) -- the same leak-safe weekly-refit
recipe as `opener_pick_evaluation`, same `player` feature columns, but fit
DIRECTLY to `open_move` instead of `market_residual`. This is deliberately
"good at tracking the close" by construction and is never a candidate ATS
model. `min_train_games=150` (not 500): it can only train on games that
themselves carry a resolved movement label (2020+), unlike the real model,
which also sees the full pre-2020 archive.

**Step 3**: direct comparison of `|result - opener|` vs `|result - close|`
on the same 1,537 games (model-independent; these residuals do not depend
on which candidate scored the game).

All uncertainty is a week- or season-blocked bootstrap (3,000 resamples,
seed 20260818), reported as `probability_positive` per the binding project
convention -- an interval crossing zero is evidence to weigh, not grounds
for rejection.

## Step 1: does surrogate skill predict outcome skill?

| Candidate | Opener accuracy (outcome) | Movement agreement (surrogate) |
|---|---|---|
| player_qb_injuries | 50.50% | 53.71% |
| player_injuries | 50.57% | 54.14% |
| base | 51.03% | 53.88% |
| player_continuity | 51.03% | 54.31% |
| pbp | 51.36% | 49.40% |
| player_injuries_continuity | 51.43% | 53.79% |
| player_injury_value | 51.43% | 53.71% |
| player_qb | 51.56% | 53.28% |
| player_value | 52.43% | 55.43% |
| player | 52.50% | 55.26% |
| player_qb_continuity | 52.56% | 53.62% |
| weak_stack | 52.83% | 55.52% |
| player_participation | **52.96%** | **55.60%** |

Sorted by opener accuracy ascending. `pbp` is the clearest outlier: middling
real accuracy but the single WORST movement agreement (49.40%, below a coin
flip) -- a reminder that the relationship is a lean, not a rule, for any one
candidate.

Across all 13 candidates on the identical 1,537 games: **Pearson r =
+0.448** (game-level paired bootstrap 95% [-0.319, +0.786],
`probability_positive` **0.848**); Spearman rho = +0.407 (95% [-0.262,
+0.803], `probability_positive` **0.890**). Three names --
`player_participation`, `weak_stack`, and `player` -- sit in the top four of
BOTH rankings (top four by opener accuracy: `player_participation`,
`weak_stack`, `player_qb_continuity`, `player`; top four by movement
agreement: `player_participation`, `weak_stack`, `player_value`, `player`).
The sign is right and the majority evidence (85-89%) favors a real positive
relationship -- per the project's binding rule, the crossing-zero interval
is not grounds to call this a null. **But it is not proof, and n=13
candidates is a thin basis for anything stronger than a lean.**

## Step 2: the market-imitation adversarial control

The purpose-built movement-imitator (fit directly to `open_move`, same
features as the real `player` baseline) vs. the real `player` baseline, head
-to-head on the 1,380 games both could score (paired, week-blocked
bootstrap, 95 week-blocks, 3,000 resamples):

| Metric | Real `player` baseline | Adversarial movement-fit | Gap | 95% CI | P(direction) |
|---|---|---|---|---|---|
| Movement agreement (surrogate) | 55.29% | **63.63%** | adversarial +8.28 pts | [+4.13, +12.42] | **1.00** adversarial higher |
| Opener accuracy (real outcome) | **53.11%** | 51.78% | real +1.34 pts | [-2.47, +5.07] | **0.756** real better |

**The surrogate rewards the imitator, decisively.** The adversarial model's
63.63% movement agreement beats every one of the 13 legitimate candidates in
Step 1 (best: 55.60%) by 8+ points, essentially certain (`probability_positive`
1.00) -- while its real accuracy trails the plain, untuned baseline it was
built to imitate, with the evidence leaning that direction at 76%. **If the
surrogate were used naively to rank or screen candidates, it would have
promoted the movement-imitator over every genuine football model in this
document.** This is the confound the idea itself warned about, confirmed
directly rather than argued: the surrogate is invalid as a standalone
screen whenever a candidate's construction has any exposure to reproducing
market/consensus information rather than football information. Reported
plainly per instruction, not hedged: this is a real failure mode, not an
edge case.

## Step 3: is the close actually sharper than the opener?

Yes, measurably, on the same 1,537 games (paired bootstrap, 3,000 resamples):

| | vs. opener | vs. close | Improvement (close - open error) | 95% CI | P(close sharper) |
|---|---|---|---|---|---|
| MAE | 9.912 | 9.815 | 0.096 pts | [0.025, 0.171] | **0.996** |
| RMSE | 12.827 | 12.690 | 0.138 pts | [0.064, 0.216] | -- |
| sd(result - line) | 12.830 | 12.693 | -- | -- | -- |

(`sd(result - opener) = 12.830` reproduces `docs/pool_edge_plan.md`'s
independently-measured 12.83 exactly -- a direct cross-check that this
pipeline is scoring the same population the rest of the project uses.)

The close is a resolvably better predictor of the true final margin than the
opener -- small in absolute terms (~1% MAE reduction) but clearly not zero.
**Movement is not pure noise; the week's information does sharpen the
number.** This validates the premise that movement carries *some*
information. It does not by itself say that information is efficiently
recoverable as a screening signal for a specific candidate model -- that is
what Steps 1, 2, and the arithmetic below settle.

## Quantifying the prize: does the efficiency gain survive?

The project's own established exchange rate: 1 point of true margin
improvement moves forced-pick accuracy by `Phi(1/sigma) - 0.5`, with
`sigma = 12.83` (measured here, matches the ledgered 12.83) -> **+3.11
accuracy points per true point** at the opener.

The same construction requires a transfer coefficient for movement: how
many points does the close actually move, on average, per point of a
model's own predicted residual? Measured directly (OLS, `open_move ~
residual_at_open`, `player` model, n=1,537):

- slope (beta) = **0.0486** (95% CI [0.0068, 0.0876], `probability_positive`
  0.990 -- genuinely positive, the market does drift toward a model's
  signal)
- **R-squared = 0.375%** -- a model's own residual explains well under half
  a percent of a game's realized movement. The other 99.6% is exogenous:
  injury news, public money, weather, other bettors' models -- exactly the
  channel the adversarial control in Step 2 exploited.
- conditional noise sd = 1.474 (matches the unconditional sd(open_move) =
  1.476 almost exactly -- conditioning on the residual barely reduces the
  noise, because it explains almost none of it)

That is the mechanism behind Step 2's result, restated as a number: a
model's genuine signal is a small, weak driver of movement next to
everything else moving the line, so a surrogate built on movement mostly
measures those other things.

Using the MEASURED transfer (not an assumed 1-for-1 incorporation), the
effect of a 1-point true improvement on movement agreement is `Phi(0.0486 *
1.0 / 1.474) - 0.5` = **+1.31 accuracy-equivalent points** -- smaller than
the direct channel's +3.11, not larger. Games needed for a 95% two-sided
test of a given true effect size (`N = 0.25 * (1.96 / effect)^2`, both
channels treated as ~50% Bernoulli labels, consistent with how the
project's own 12.83 exchange rate was derived):

| True effect (points of LINE improvement) | Direct outcome-accuracy channel | Movement channel, MEASURED transfer | Movement channel, idealized 1-for-1 transfer (not supported by data) |
|---|---|---|---|
| 1.0 pt | 995 games | 5,564 games | 15 games |
| 1.3 pt | **590 games** | **3,293 games** | 10 games |

> **UNIT CORRECTION (2026-08-18, after review).** The row above was
> originally annotated "the parked injury-lead size". That is wrong and the
> annotation is removed. The `True effect` column is **points of true LINE
> improvement**, which the ~3x exchange rate turns into ~3.11 accuracy
> points; 1.3 line points is therefore a ~4.0-accuracy-point effect, which
> is enormous. **The injury lead is +1.316 ACCURACY points** (registry
> `injury_value_lost_gradient`, `effect_units: accuracy_points`) -- about
> **0.42** line points, roughly a third the size the row assumed.
>
> Redone in the right units, a 1.316-accuracy-point effect needs **~5,545
> games** unpaired, or **~2,200-3,300 paired** depending on how often the
> two arms actually make different picks (10-15%). The archive holds
> **1,537**.
>
> **So the direct channel canNOT resolve the injury lead on data in hand
> either** -- the reverse of what this section originally concluded. The
> surrogate verdict is untouched: the adversarial market-imitation test
> killed it independently of any sample-size arithmetic, and the movement
> channel remains the worse of the two by the same ~5.6x ratio. What
> changes is that "just use the direct label, it already resolves this" is
> not available as an answer, which is precisely why the power work
> (`docs/evaluator_power.md`, `docs/variance_reduction.md`,
> `docs/anytime_valid.md`, `docs/purged_cv.md`) is the live path.
The "idealized" column exists only to show where the idea's original appeal
came from -- if a model's residual fully and immediately moved the close
1-for-1, the surrogate would indeed be enormously cheap (~15 games). That
premise is measured and rejected (beta = 0.049, not 1.0). **The efficiency
gain the idea hoped for does not survive contact with the transfer
coefficient; if anything the direct label is the cheaper one.**

This also explains, rather than contradicts, Step 1's positive
cross-candidate correlation: comparing whole, substantially different
candidate families averages over enough games that some real signal shows
through in aggregate (P+ 85-89%), but that aggregate correlation is itself
partly built from the same market-imitation variance Step 2 isolated --
consistent with a real but weak and confounded relationship, not a strong,
clean one.

## Screen-then-confirm rule

1. **Never use movement agreement alone to accept or reject a candidate.**
   Per the arithmetic above, on the current archive it cannot resolve the
   project's decision-relevant effect sizes (~1.3 points), and per Step 2 it
   can be gamed upward by market-imitation without any real accuracy gain.
2. **Never apply it to a candidate whose construction has any exposure to
   market/consensus information** (features built from betting percentages,
   steam-chasing, or anything trained toward predicting the close or its
   movement rather than the true result). Step 2 shows this exact
   construction can beat every legitimate candidate's surrogate score while
   being worse in practice. If a candidate's edge over baseline looks
   unusually large on movement agreement specifically, treat that as a
   reason for MORE scrutiny of its construction, not promotion.
3. **A movement-graded number may never promote a candidate any more than a
   close-graded number may veto one** (extending the existing binding rule
   in `AGENTS.md`: "Grade the decision at the OPENER"). Every promotion
   decision must still be made, and reported, at real forced-pick opener
   accuracy.
4. **Where it may still help, narrowly**: as a free, same-data, first-pass
   sanity check across a large sweep of candidate variants (e.g., dozens of
   ablations or hyperparameter settings) before committing full opener
   -accuracy evaluation to a shortlist -- since Step 1's aggregate
   correlation, while weak, still beats a coin flip (P+ 85-89%) for
   substantially different candidate families. Any shortlist survivor still
   requires the real opener-accuracy grade before anything is claimed or
   promoted; the surrogate buys a cheaper first cut, never a conclusion.

## What this does and does not change

Zero picks, models, feature profiles, or `artifacts/active_ats_model.json`
moved as part of this investigation. `src/nfl_ats/surrogate_outcome.py`
(`movement_agreement`, `movement_agreement_rate`,
`fit_movement_target_model`) is the only production code added, and none of
it is wired into weekly scoring, the active model, or any candidate feature
profile -- it exists to support this validity question and any future
re-check of it, per the rule above.
