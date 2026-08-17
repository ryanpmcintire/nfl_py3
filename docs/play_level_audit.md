# Play-level data: what 166x more rows actually buys

Written 2026-08-17. The standing intuition behind MOD-09 and PBP-09 was that
game-level data is capped by reality — about 285 games a season, 4,703 in the
whole feature table — while play-level data holds 781,712 rows, roughly 166
times more, and is untouched. If every game is next-token prediction, that
ratio looks like the largest unexploited lever in the project.

This document records the audit of that claim. The short version: **the 166x is
real as a row count and mostly a mirage as an information gain**, the "untouched"
half of the premise is wrong in both directions, and exactly one genuine gap
survived the audit.

All figures below are descriptive diagnostics on 2009–2012, which no family has
reserved. No rotation-registry window was spent and no candidate was graded.

## The premise is wrong in both directions

**Play-derived data is not untouched.** The active model
(`market_residual` / `player` / ridge / alpha 10) reads an explicit 79-column
allowlist. Of those, roughly 44 are already play-derived: the team-form columns
(`off_epa_per_play`, `off_cpoe`, `def_epa_per_play`, `off_sack_rate`, and the
rest) are per-play quantities pre-aggregated upstream by nflverse, and the five
QB columns are computed here directly from the raw play snapshot
(`quarterbacks.py` filters `qb_dropback == 1` and aggregates per-dropback EPA
and CPOE).

**What is genuinely unused was tested and rejected.** Of the 103 pbp/drive
columns sitting in `game_features_pbp.parquet`, the active model uses zero — not
by oversight but by outcome. The raw-PBP/drive bundle was screened
(nested 2018–2025: 50.36% ATS, no edge), the drive layer worsened Brier
(0.250803 → 0.250891), opponent adjustment as an addition worsened every error
metric, and the replication spent `[2013, 2017]` to close the family at
**−0.08 points** on 1,247 games.

## Why more rows is not more information

The intuitive objection to play-level data is that plays within a game are
massively correlated, so the effective sample is far smaller than the row count.
That objection is **false as stated, and the truth is worse for the premise.**

Measured ICC of play-level EPA is **0.0131** — a design effect of 1.67x, so 52.5
competitive plays per team-game carry the information of about 31.4 independent
ones. Plays are nearly independent. But that is precisely what makes the sample
mean a sufficient statistic: if the rows are close to iid draws, then the
per-game average already extracts everything they say about a team's mean, and
exposing the 781,712 rows individually adds nothing to an estimate you have
already computed.

Crucially, the 166x does not multiply the training set. The ridge fits on ~4,700
**games**, and that is the number of learnable units. Play count multiplies
feature *resolution*, not sample size — and resolution is already near its
ceiling. A full team-season of 824 competitive plays yields EPA/play with
reliability 0.765; the remaining ~24% is sampling noise that no amount of
per-play granularity removes, because it is irreducible without more games.

**Direct test.** Predicting rest-of-season point differential from the first K
games, with and without per-play information:

| K games | R² (net EPA + point diff) | R² (point diff only) | gain from per-play |
|---|---|---|---|
| 2 | 0.196 | 0.171 | **+0.025** |
| 3 | 0.209 | 0.192 | +0.017 |
| 4 | 0.243 | 0.243 | +0.000 |
| 6 | 0.276 | 0.274 | +0.002 |
| 8 | 0.248 | 0.235 | +0.013 |
| 12 | 0.181 | 0.181 | +0.000 |

Per-play data earns its keep in weeks 1–3 and essentially nothing after week 4.
That is the entire noise-reduction dividend — and the model already collects it
through the nflverse EPA columns it reads.

## The scale any of this has to clear

sd of the ATS residual is 13.13 points, and the market already explains 19.6% of
margin variance. Converting hit rates into signal strength:

| ATS accuracy | corr(prediction, residual) | R² |
|---|---|---|
| 52.05% (current, close) | 0.0644 | 0.0041 |
| 52.50% (current, opener) | 0.0785 | 0.0062 |
| 54.00% | 0.1253 | 0.0157 |

The entire existing edge lives inside **0.6% of residual variance**, and moving
52.5% → 54% means finding roughly another 1% that the market missed. A
measurement refinement worth +0.02 R² on team strength — a quantity the market
also observes, and prices — has no plausible route to that.

## Two candidates closed on measurement

**Pace and possession forecasting (PBP-09) is dead.** Game play volume has
almost no variance to forecast: 125.5 plays per game with sd 8.7, a coefficient
of variation of **0.070**. Forecasting it from both teams' prior pace reaches
R² 0.041, shaving about 2% off the residual sd. And the link to the outcome runs
backwards: `corr(plays, |margin|) = −0.20`, because blowouts kill clock and end
in kneel-downs, while `corr(drives, |margin|) = +0.004` is simply zero. There is
no pregame pace signal here to harvest. Reproduced independently before the
roadmap row was changed.

**Distribution-shape variance is dead.** Split-half reliability of a team's
play-EPA *standard deviation* is **r = +0.014** — dispersion is not a team
property at all — and `corr(pregame dispersion, |ATS residual|) = −0.028`. The
p90 tail looks reliable (0.512) but correlates +0.877 with the mean, so it is
the mean wearing a hat. Only skew shows any daylight (0.297) and it is weak.
This is distinct from MOD-16, which tested pace-conditioned scale rather than
distribution shape; both are now closed.

## The one genuine gap

**Defensive per-play measurement is roughly half noise, and the fix was never
tested properly.** Split-half reliability over 2009–2012, stepped up to
full-season with Spearman-Brown:

| quantity | full-season reliability |
|---|---|
| offense EPA/play | ~0.80 |
| defense EPA/play allowed | ~0.46 |

(Two independent measurements agreed on the gap; they differed on the second
decimal — 0.739/0.405 and 0.795/0.457 — because of differing split
definitions. The gap itself is not sensitive to that choice.)

Opponent adjustment is the textbook correction for exactly this, and PBP-05
built it. But every test of it **added** the adjusted columns alongside the raw
ones, taking the design from 58 columns to 106, 124, and 142 on ~4,700 rows at
alpha 10. At that ratio, adding 45–66 collinear columns to a ridge is a variance
disaster whether or not the adjustment carries signal — so PBP-05's negative
result does not answer the question it appears to answer.

The **dimension-neutral substitution** — identical column count, adjusted
defensive rates swapped in for raw ones — has never been run. It is cheap, the
columns already exist for NFL, and it should be screened on CFB first, where
registry rule 8 makes iteration free and the clean core is 8,933 games, an order
of magnitude more resolving power than any three-season NFL window. CFB has no
adjusted columns yet, so the screen requires porting the estimator; that port is
the honest cost of the recommendation.

## The ceiling on the whole family, measured

The substitution screen was run on CFB (2026-08-17,
`docs/cfb_opponent_adjustment.md`) and **closed**: the dimension-neutral swap
improved clean-core margin MAE by −0.0003 points, `probability_positive`
0.463 on 9,093 games. Isolating the opponent block from a time-weighting
change bundled with it recovers +0.0005 MAE — operationally zero.

The valuable part is the positive control. Fitting the same adjustment **once
on the entire 2006–2025 history**, so the columns can see the future, moves
MAE by only **+0.0129 points** (`probability_positive` 0.984). The instrument
detects a leak of that size, so the null is measured rather than underpowered
— and that number is a **ceiling on the entire family**: perfect future
knowledge of team quality is worth about a hundredth of a point of margin
error on these games.

That single measurement retroactively explains a long trail of negative
results — the drive bundle, PBP-05, the variance model, role continuity. The
target is the **residual from the market line**, and the market already prices
team quality. Measuring team strength more precisely refines a quantity the
spread has already accounted for, so the achievable gain is bounded near zero
no matter how good the measurement gets. A revisit needs a different *target*,
not a better adjustment.

## Standing conclusion

Play-level data is already the substrate of most of what the model reads. The
remaining value is **not** in re-screening the 48-column bundle, not in sequence
models over drives, and — now measured — not in reducing measurement noise at
constant dimension either. That last route was the one candidate this audit
rated worth a screen, and the screen closed it along with a ceiling that
explains the rest.

The generalisable lesson: **anything that only measures team quality better is
bounded near zero, because the spread already prices team quality and our target
is the residual from it.** Effort belongs where the market prices *badly*, not
where our own estimate is noisy. The one axis where plays genuinely multiply
learnable units rather than resolution is as an instrument for **player**-level
quantities — roughly 2,000 players observed week over week, rather than 4,703
games — which is also where the only measured lean in the MOD-07 stack came
from (availability, `probability_positive` 0.899).
