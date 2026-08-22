# Ceiling attack: market error vs noise floor in opener ATS residuals

Question: of total ATS-residual MSE against the frozen Tuesday opener, how much
is MARKET error that a better model could in principle capture, and how much is
the irreducible execution-noise floor (~12.7 pts sd from `vardec_noisefloor`)?

Status: measure-only MSE-decomposition milestone, first run 2026-08-22
(`artifacts/ceiling_error_split/20260822T221446Z/results.json` is the final
run; two earlier same-day runs are superseded debugging runs with identical
numbers; all six scoped gates pass). Nothing here selects wagers; nothing was
recorded to `weak_signals`.

## Population and method

Population: the predeclared opener-evaluation archive
`artifacts/opener_evaluation/20260819T174244Z/per_game.parquet` — 1,537 paired
2020–2025 REG games graded against Tuesday openers, `weak_stack` composed card
(**measured**). 34 pushes (`margin_vs_open == 0`) are dropped for the primary
stats, leaving 1,503 (**measured**).

Target: `margin_vs_open`. Predictors:

1. **market line** — predicts 0, so its squared error IS total MSE;
2. **our composed card** — `residual_at_open`; honest-treatment note: this
   embeds market information by construction (its training target IS the
   market residual), so its measured capture is read directly against the
   market predictor rather than claimed as independent football signal;
3. **oracle blend** — in-sample OLS on [1, our pred]. Because the market
   predictor is constant, this spans every linear blend of market and card;
   it is an UPPER bound on what shrinkage could extract from the current card;
4. **perfect movement foresight** — OLS on [1, `open_move`], the
   late-information channel measured separately (upper bound for that channel);
5. **joint oracle** — [1, our pred, `open_move`].

Theoretical floor: execution-noise sd **12.704 points**, taken from the latest
production vardec-noisefloor run
(`registry/experiments/vardec-noisefloor/20260822T213001Z.json`,
sqrt(14.244^2 − 6.441^2)) (**measured** by that lane, consumed here). The
theoretical better-team-model slice is market MSE minus execution variance:
the part a PERFECT pregame strength model could still remove if the market
priced none of it — an upper bound, since the market prices most matchup
structure already.

Accuracy translation: flat exchange rate 3 accuracy points per point of RMS
improvement at sigma ≈ 13 (`docs/pool_edge_plan.md`). Because forced picks
grade sign, not magnitude, flat-exchange translations UNDERSTATE sign-channel
value; direct measured accuracy deltas are always printed beside them and the
two are never mixed. Uncertainty: week-blocked bootstrap, 2,000 resamples,
seed 20260822.

## MSE results (all measured, artifact above)

| Quantity | Value |
|---|---|
| Games scored / pushes dropped | 1,503 / 34 |
| Market RMS error (opener) | **12.972 points** (MSE 168.27 pts²) |
| Card raw RMS error | 13.025 points (unshrunk card ADDS variance) |
| Oracle-blend RMS | 12.953 points (slope 0.335 — optimal shrinkage keeps 1/3) |
| corr(outcome, card) | 0.0511 |
| corr(market error, card error) | 0.9884 |
| Capturable share, current card | **0.29%** of MSE |
| Capturable share, perfect movement foresight | **2.28%** of MSE |
| Capturable share, joint oracle | 2.43% of MSE |
| Execution-noise share (theory) | **95.92%** of MSE |
| Unmatched-matchup share (theory ceiling) | **4.08%** of MSE (6.87 pts²) |

Bootstrap week-blocked 95% intervals (**measured**): current-card capturable
share 0.04%–1.12%; late-information share 1.12%–4.07%.

The headline reading: our errors are 98.8% correlated with market errors — we
are, in squared-error terms, a very slightly perturbed market — and the
unshrunk card's RMS is WORSE than predicting zero. Yet the same card wins
53.36% of forced picks at the opener (**measured**,
`correct_at_open_probability_rule`). Both facts are true because picks grade
only the sign/threshold: a prediction with sd ~2 points and correlation 0.05
removes almost no variance while still flipping picks above 50%. Never quote
one metric as if it were the other.

## The gap table: banked → wall (accuracy points)

Anchor honesty note: the tasking said "banked 53.8%", but THIS archive
recomputes to **53.36%** probability-rule / **52.83%** sign-rule opener
accuracy (**measured**, matches `metadata.json` to 1e-12); the nearest 53.8%
in the repo is the 2024 season row (`docs/opener_evaluation.md`, **read**).
The table anchors on the archive number; add ~0.45 to every row if anchoring
on 53.8%.

| Row | Accuracy points | Basis |
|---|---|---|
| Banked anchor (this archive) | 53.36 | measured |
| **Total gap → omniscient-practical wall (57–58%)** | **+3.64 … +4.64** | wall band from `docs/pool_edge_plan.md` |
| ├ Capturable by better team model | ≤ +0.80 [0.00, +2.28] | theoretical unmatched-matchup ceiling at flat exchange; realized capture by the CURRENT card is +0.06 [+0.01, +0.22] |
| ├ Capturable by late information | +1.72 measured direct (movement oracle 55.08% − banked) | movement channel measured separately; flat-exchange equivalent only +0.45 [+0.22, +0.80] |
| └ Irreducible remainder (execution noise) | not expressible in acc points | 95.9% of market MSE = execution-noise floor 12.70 pts; capturable by NO pregame model |
| Remainder beyond both slices vs wall | +1.12 … +2.12 | omniscience beyond perfect movement foresight (private/aggregated info the line never sees) plus wall-band width |

Cross-checks against `docs/pool_edge_plan.md` gap accounting (**measured**
here vs **reported** there):

- Plan's midweek channel "~2.6 points": this archive gives 55.08 − 52.83 =
  +2.25 on the sign-rule baseline or 55.08 − 53.36 = +1.72 on today's banked
  anchor — same channel, different anchors; consistent.
- Plan's "measuring teams better is bounded near zero": confirmed hard — the
  entire variance-capturable space above the market line is ≤ +0.80 accuracy
  points even under a perfect-strength upper bound.
- Plan's sigma ≈ 13.1 / rate ≈ 3: this subset's residual sd is 12.97
  (opener grade, 2020–2025) — consistent.
- Plan's guardrail ("60%+ would be a leak"): supported — 95.9% of opener MSE
  is execution noise no model can touch.

## Gates (scoped, all PASS)

population_matches_archive_metadata · reproduces_archive_accuracies (both
opener accuracies match `metadata.json` to 1e-12) ·
execution_noise_source_in_band (12.70 ∈ [12.0, 13.5]) ·
nested_oracles_monotone (MSE: joint ≤ min(blend, move); blend ≤ min(market,
card raw)) · theory_shares_sum_to_one · bootstrap_reproducible_same_seed.

## Caveats

- Oracles are IN-SAMPLE upper bounds; achievable gains are smaller.
- The team-model ceiling assumes the market prices nothing beyond the line it
  posts and a perfect model captures ALL unmatched matchup variance — both
  generous to the candidate; the realized +0.06 is the honest current number.
- The execution floor is single-window (2021–2025 sim calibration, see
  `docs/vardec_noisefloor.md` caveats); era stability untested there.
- Flat-exchange translations near these tiny deltas carry key-number
  curvature risk; direct accuracy deltas are the safer read wherever both
  exist.
