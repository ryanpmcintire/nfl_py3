# Turnover variance decomposition in NFL margins, 2009–2025 REG

Measured 2026-08-22 by `scripts/vardec_turnovers.py`; every number below is
**measured** from that run's output (`artifacts/vardec_to/results.json`,
PBP snapshot `20260817T184927Z`) unless tagged **reported** or **inferred**.
This is a mechanism measurement, not a candidate feature and not a rotation
verdict; nothing in `src/` changed and no window was spent. Companion to
`docs/vardec_penalties.md` (same machinery, same bootstrap conventions).

## Question

How much of the variance in the final margin do turnovers explain — the folk
"turnover luck" channel — and how does it split between persistent team trait,
single-game luck, and fumble-recovery luck specifically?

## Constructs

- **Turnover-EPA swing** (per team-game): sum of `epa` over snapshot rows with
  `interception == 1` or `fumble_lost == 1`, `play == 1`, EPA present,
  `posteam == team`. The expected-points cost of that team's giveaways on its
  own offensive snaps.
- **Return touchdowns are inside the construct (measured):** nflverse EPA on
  these rows embeds the defensive score — 761 of 7,517 kept interceptions are
  pick-sixes averaging **-8.119 EPA** versus -3.982 for non-scoring picks;
  378 of 4,393 kept lost fumbles became defensive scores averaging **-8.045**
  versus -4.534. The folk "pick-six / scoop-and-score" channel is therefore
  included by construction, not approximated.
- **Exclusions (explicit):** of 781,712 regular-season plays, 12,524 carry a
  turnover flag; **644 are no-play rows excluded** (dead-ball/offsetting,
  no EPA by construction). Zero play==1 turnover rows lacked EPA or posteam;
  11,880 rows entered aggregates over 4,431 games.

## Headline decomposition (4,431 games, 2009–2025 REG)

| Quantity | Value |
|---|---|
| SD of game turnover-EPA differential | **8.938 pts** (mean absolute 6.885 pts) |
| SD of final margin | 14.618 pts |
| **Gross variance share** Var(swing diff)/Var(margin) | **37.38%**, week-blocked bootstrap 95% [35.37%, 39.41%], P+ > 0.999 |
| Correlation with margin | +0.576 |
| Regression variance share (R²) | **33.13%**, bootstrap [31.09%, 35.28%] |
| League-mean swap delta-SD (swap differential → 0) | **8.938 pts** [8.72, 9.14] |
| Same vs ATS margin (margins around the line) | corr +0.569, gross share **46.34%**, R² 32.37% |

For scale, penalties measured 8.67% gross / 3.29% R² on identical machinery:
turnovers are roughly four times the margin-variance channel penalties are.

### Component split

| Component | SD | corr w/ margin | R² [95%] | Gross share |
|---|---|---|---|---|
| Interception EPA diff | 6.899 pts | +0.509 | 25.88% [23.65, 28.04] | 22.28% |
| Lost-fumble EPA diff | 5.165 pts | +0.316 | 10.00% [8.38, 11.73] | 12.48% |
| Turnover count diff | 1.820 TOs | −0.583 | 33.94% [31.86, 35.94] | 1.55% |

Unlike penalties (where raw counts carried only 0.14% R²), the bare turnover
count differential carries essentially the full regression share of the EPA
swing (**inferred** reading of the measured near-tie: for turnovers, face
value suffices; situation adds almost nothing).

## Fumble-recovery luck slice

Method limit first: the narrowed snapshot has no all-fumbles column and no
recovery attribution, so a literal "treat recoveries as 50/50 assigned" EP
recomputation is impossible here. Proxy used: fit OLS of lost-fumble EPA on
pre-play situation (down, yardline_100, ydstogo, quarter, score state; in-sample
R² only 8.48%) across all 4,393 kept lost fumbles, recompute every team-game
fumble swing from fitted situational expectations, and take realized-minus-
fitted as the play-realization residual — of which recovery assignment and
return outcome are the dominant pieces (**inferred**; the fit cannot separate
them from other play-level noise).

| Quantity | Value |
|---|---|
| SD of situation-neutralized fumble diff | 4.918 pts (vs 5.165 realized) |
| SD of realization residual diff | 1.627 pts |
| Residual gross share / corr with margin | 1.24% [1.12%, 1.36%] / +0.087 |
| R²(INT + neutral fumble diff) | 32.79% |
| R²(INT + realized fumble diff) | 33.51% |
| **Recovery/return-luck slice** | **+0.72 R² points**, bootstrap [0.32, 1.16], P+ > 0.999 |
| Fumble component's total increment over INT-alone | +6.91 R² points |

So of the fumble component's ~6.9-point regression contribution, only about
0.72 points (~10%, **inferred** division of two measured numbers) is
play-level realization luck; most of it is which team loses how many fumbles
where on the field. Recovery luck is a real but thin slice of a big channel.

## Forecastable fraction (luck vs trait)

Season-lagged swap (prior-season giveaway-EPA-per-snap × this game's snaps,
contiguous prior season; 4,175 of 4,431 games covered):

- Naive lagged-rate swap delta-SD: 9.143 pts — **larger** than doing nothing.
  Unshrunk lagged rates are mis-scaled at game level: the OLS-optimal slope of
  realized swing on lagged prediction is **0.229** (regression to the mean),
  so swapping in raw prior-season rates adds variance instead of removing it.
  Measured, and the reason the naive swap must never be quoted as a ceiling.
- Fitted (in-sample OLS) swap delta-SD: 8.903 pts; forecastable fraction of
  swing variance **0.48%** (correlation +0.069); carries in-sample optimism,
  so an upper bound at this resolution.
- Forecastable share of TOTAL margin variance ≈ 0.18% (0.0048 × 37.38%)
  (**inferred** product of two measured numbers).

The trait-vs-luck verdict is stark: turnovers move margins by ±8.9 points per
game, but essentially none of that movement is ex-ante predictable from
season-lagged team rates — the game-level swing is single-game luck, not
persistent skill.

## Reliability anchors

| Trait | YoY reliability | Pairs | Provenance |
|---|---|---|---|
| Team net turnover margin per game | **+0.1463** | 512 | measured; matches the known ~+0.13 anchor (**reported**, task brief) — **inferred**: the folk number refers to net differential, not giveaway rate |
| Team giveaway rate per game | **+0.2550** | 512 | measured |
| Team giveaway rate per snap | +0.2502 | 512 | measured |
| Team turnover-EPA per snap | +0.1892 | 512 | measured |
| Team fumbles-lost per game | **+0.0902** | 512 | measured |

The giveaway-rate reliability (+0.26) is nearly double the net-margin
reliability (+0.15): takeaway generation is close to noise year to year, which
is why even the well-measured trait side collapses at game level (**inferred**
synthesis of the measured table above).

## Reading

Turnovers are the largest single real-time variance channel this lane has
measured — a ±8.9-point one-game differential, 37% of total margin variance,
46% around the line — yet under half a percent of it is forecastable from
season-lagged traits, and the fumble-recovery slice the folk explanation leans
on is ~0.7 R² points of margin variance. The folk story is right that
turnovers decide games weekly and wrong if heard as a team skill story:
game-level turnover differentials are mostly luck wearing a trait costume
(**inferred** synthesis; all component numbers measured).

Per the standing invariant, nothing here was closed or rejected; all headline
intervals exclude zero with P+ > 0.999 and signs are positive throughout. No
`weak-signals record` entry was written: the registry lives outside this
task's declared file ownership, so recording is left as an explicit follow-up
for the owner rather than silently skipped.

## Limitations

1. No all-fumbles or recovery columns survive ingestion locally; the 50/50
   recovery counterfactual is proxied by situational-expectation residuals and
   cannot separate recovery assignment from return-yardage noise.
2. The swing is a posteam-side construct; defensive takeaways enter through
   their effect on the opponent's giveaways, symmetrically by design.
3. The fitted-swap forecastable fraction carries in-sample OLS optimism; a
   true out-of-sample version would shrink rates first and score at or below it.
4. EPA return-TD embedding means the swing mixes possession value with
   scoring value; the count-differential row shows the mixture changes little
   relative to face value.
