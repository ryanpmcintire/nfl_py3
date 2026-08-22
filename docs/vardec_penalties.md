# Penalty variance decomposition in NFL margins, 2009–2025 REG

Measured 2026-08-22 by `scripts/vardec_penalties.py`; every number below is
**measured** from that run's output (`artifacts/vardec_pen/results.json`,
PBP snapshot `20260817T184927Z`) unless tagged **reported** or **inferred**.
This is a mechanism measurement, not a candidate feature and not a rotation
verdict; nothing in `src/` changed and no window was spent.

## Question

How much of the variance in the final margin do penalties explain (count,
yards, EPA swing, type mix), and what fraction of that is forecastable ex ante
from season-lagged team rates?

## Constructs and exclusions

- **Penalty-EPA swing** (per team-game): sum of `epa` over snapshot rows with
  `penalty == 1`, `play == 1`, EPA present, `posteam == team`. It is the net
  expected-points swing penalties produced on that team's offensive snaps
  (own offensive fouls negative, opponent defensive fouls positive). Same
  mixed-signal caveat as the `penalty_discipline` reconstruction: the narrowed
  snapshot carries no `penalty_team`, so the *committing* side is not locally
  knowable; the benefiting side is.
- **Exclusions (explicit).** Of 781,712 regular-season plays, 55,584 carry the
  penalty flag; **6,472 are no-play fouls and were excluded** — dead-ball
  fouls and offsetting double fouls replaying the down, which carry no EPA in
  nflverse and cannot enter an EPA swing by construction (the local snapshot's
  stored columns cannot separate offsetting from dead-ball within that bucket;
  both are inside the excluded 6,472). Zero play==1 penalty rows lacked EPA or
  posteam, so 49,112 rows entered the aggregates over 4,431 games.
- **Declined fouls are NOT separable** in the narrowed snapshot (no play
  description or accepted flag survives ingestion): they remain inside the
  kept rows contributing their actual play's EPA. This inflates swing
  magnitude somewhat and biases toward zero differently on the two sides of
  any given game; direction of net bias unknown (**inferred** limitation, not
  a measured quantity).
- **Type mix** uses the widened officials snapshot
  (`data/raw/officials/20260820T112517Z/game_penalty_types.parquet`, joined to
  home/away attribution from `game_penalties.parquet`): 2,895 REG games,
  2015–2025 only. Its own manifest records offsetting/no-play handling upstream
  (aggregated per `(game_id, penalty_type)` from full-column PBP).

## Headline decomposition (4,431 games, 2009–2025 REG)

| Quantity | Value |
|---|---|
| SD of game penalty-EPA differential | **4.303 pts** (mean absolute 3.325 pts) |
| SD of final margin | 14.618 pts |
| **Gross variance share** Var(swing diff)/Var(margin) | **8.67%**, week-blocked bootstrap 95% [8.08%, 9.24%], P+ > 0.999 |
| Correlation with margin | +0.181 |
| Regression variance share (R²) | **3.29%**, bootstrap [2.29%, 4.44%] |
| League-mean swap delta-SD (swap differential → 0) | **4.300 pts** [4.19, 4.41] |
| Same vs ATS margin (margins around the line) | corr +0.181, gross share 10.74%, R² 3.26% |

Secondary differentials, same sample: count differential SD 3.67 penalties,
corr with margin +0.037 (R² 0.14%, bootstrap [0.01%, 0.40%]); yardage
differential SD 35.1 yards, corr +0.039 (R² 0.15%). Raw counts and yards are
far weaker margin correlates than the EPA swing — the *situation* of a foul,
not its face value, carries the margin variance (**inferred** reading of the
measured gap between 3.3% and 0.14%).

## Forecastable fraction

Counterfactual swap with season-lagged team rates (prior-season
swing-per-offensive-snap × current-game snaps, contiguous prior season only;
4,175 of 4,431 games covered):

- Naive lagged-rate swap delta-SD: 4.396 pts — **larger** than doing nothing.
  Unshrunk lagged rates are mis-scaled at game level: the OLS-optimal slope of
  realized swing on lagged prediction is **0.260** (regression to the mean),
  so swapping in raw prior-season rates adds variance instead of removing it.
  Measured, and the reason the naive swap must never be quoted as a ceiling.
- Fitted (in-sample OLS) swap: forecastable fraction of swing variance
  **0.54%** (correlation +0.0734); carries in-sample optimism, so it is an
  upper bound at this resolution.
- Forecastable share of TOTAL margin variance ≈ 0.05% (0.0054 × 8.67%)
  (**inferred** product of two measured numbers).

## Reliability anchors

| Trait | YoY reliability | Pairs | Provenance |
|---|---|---|---|
| Team overall penalty rate (precedent method) | **+0.2604** | 512 | measured; reproduces the known ~+0.261 anchor (**reported**, ROADMAP/registry via `scripts/penalty_discipline_interval.py`) |
| Team penalty-EPA swing per snap | **+0.1609** | 512 | measured |
| Team Offensive Holding rate per snap | **+0.1474** | 320 (2015–2025) | measured |
| Team Defensive Holding rate per snap | **+0.1179** | 320 (2015–2025) | measured |
| Referee-crew Offensive Holding rate | +0.3226 | 158 | **reported**, `docs/penalty_crew_tendencies.md` §3 |
| Referee-crew Defensive Holding rate | +0.2702 | 158 | **reported**, same |

The crew-vs-team gap is expected arithmetic, not a contradiction: crews
observe both teams' fouls across a full season while team-season holding
counts sit on far fewer calls, so the same underlying tendency resolves much
better at the crew level (**inferred**).

## Reading

Penalties are a large real-time variance channel — a ±4.3-point one-game
differential, 8.7% of margin variance, 10.7% of variance around the line —
but essentially none of it is ex-ante forecastable from season-lagged team
rates (upper bound ~0.5% of swing variance, ~0.05% of margin variance)
(**inferred** synthesis; all component numbers measured). The game-level
swing is situational and officiating-crew noise, not a persistent team trait:
the swing construct's own reliability (+0.16) sits well below even the modest
count-rate reliability (+0.26).

Per the standing invariant, the near-zero forecastable fraction is **not** a
rejection: probability-positive framing is correlation +0.073 with the naive
prediction and +0.0734's R² bounded below by the bootstrap lower bounds above;
no admissible closing ground exists (signs are positive throughout; no
positive control was run). This is category 3, unresolved — a crew-level
forecastability variant (the +0.32/+0.27 traits) remains untested and is the
obvious next look if anyone wants it. No `weak-signals record` entry was
written: the registry lives outside this task's declared file ownership, so
recording is left as an explicit follow-up for the owner rather than silently
skipped.

## Limitations

1. Declined fouls inseparable locally (above).
2. Swing is a posteam-side mixed signal, not committing-team attribution.
3. Type-mix legs cover 2015–2025 only (widened snapshot's window).
4. The fitted-swap forecastable fraction carries in-sample OLS optimism; a
   true out-of-sample version would shrink rates first and would score at or
   below it.
