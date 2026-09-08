# MOD-18 Lane R: how much of the model's disagreement to trust, by line size

## Predeclaration (frozen before computation, 2026-09-08)

Everything in this section was written and saved BEFORE any candidate number
was computed. The replay stage records this file's SHA-256 at that moment
(`artifacts/research/laneR/reproduction.json`, field
`predeclaration_sha256`), so the measured section appended below cannot have
influenced it.

### What motivates the arm

Read: `docs/big_spread_diagnosis.md` (lane L, section 2, the
`residual_information` table) measures the slope of the actual home margin
against the opener on the model's served residual, per lane-J spread bucket,
on the S3 served evaluation (1,503 non-push opener games 2020-2025):
`0-3` +0.19 [-0.34, +0.71]; `3.5-6.5` +0.53 [-0.03, +1.06] P+ 0.97;
`7` +0.20 [-1.31, +1.72]; `7.5-10` +0.47 [-0.33, +1.23] P+ 0.87;
`10.5+` **-0.99** [-1.98, +0.02] P+ 0.03. Slope 1 means the residual is right
in size, 0 that it says nothing, negative that it points the wrong way. Read:
the same document locates the accuracy hole in the road picks at `10.5+`
(37.0% on 54 games, home beating the served point by +4.63 [+1.29, +7.89]).

This is a MECHANISM, not a threshold: the informativeness of the model's
disagreement with the line depends on the size of the line, so the weight the
served point puts on that disagreement should depend on it too. It is not a
pick flip bolted onto a bucket (AGENTS.md, "No unexplained threshold flips on
the played card"): the same estimator runs in every bucket, is fitted only on
prior completed games, and reduces to the incumbent wherever the data say the
residual is calibrated.

Read: `docs/home_side_offset_promotion.md`, section "S3 played", defines the
incumbent: the walk-forward home-side offset served only in the `7`,
`7.5-10` and `10.5+` buckets, policy `home_side_offset_big_spreads_v2`. Read:
`src/nfl_ats/home_side_location.py:239` (`prior_rows_before`) defines the
exclusions the incumbent fits under: strictly earlier weeks, seasons at or
after the target season minus five, rows with a recorded result and a valid
raw point only.

### R1 (the arm)

For a target week (season `S`, week `W`) and lane-J spread bucket `b`:

1. Prior rows `P = prior_rows_before(archive_stream, S, W)` -- the SAME
   exclusions the served S3 offset uses. Nothing from the target week or any
   later week, nothing before `S - 5`, results only.
2. For every prior row `i` whose archived Tuesday opener falls in bucket `b`:
   * `x_i` = the model's RAW out-of-time residual at that opener (its
     disagreement with the line, in points);
   * `y_i` = actual home margin minus that opener line.
3. Ordinary least-squares slope WITH an intercept -- the same estimator lane
   L's diagnosis table used (read: `scripts/big_spread_diagnosis.py:429`,
   `week_blocked_slope`):

   ```
   beta_hat_b = (sum(x*y) - sum(x)*sum(y)/n) / (sum(x*x) - sum(x)*sum(x)/n)
   ```

   with `n = n_b`, the number of such rows. If `n_b < 3` or the denominator is
   not strictly positive (no variation in `x`), `beta_hat_b := 1.0` -- no
   information to move off the incumbent.
4. Shrunk toward the incumbent weight 1.0 with a 100-game prior, the same
   `PRIOR_WEIGHT_GAMES = 100` the served offset uses (read:
   `src/nfl_ats/home_side_location.py:34`):

   ```
   beta_b = (n_b * beta_hat_b + 100 * 1.0) / (n_b + 100)
   ```

5. Served point for a game in bucket `b`:

   ```
   point_R1 = line + beta_b * residual_raw + offset_b
   ```

   where `offset_b` is the UNCHANGED S3 served offset
   (`fit_home_side_offsets`, zero outside `7` / `7.5-10` / `10.5+`). So **S3
   is exactly R1 with every beta = 1**, and the arm is a strict generalisation
   of the incumbent, not a competing recipe.

Declared, because the two layers are fitted on the same prior rows: the offset
is NOT refitted under the new slope. It stays the incumbent's offset, so any
measured difference is attributable to the slope alone.

### The centre shift (algebra, pinned in a test)

`MarginModel.predict` forms `predicted_margin = line + residual_raw` for the
`market_residual` target and then adds `center_offset` to both the margin and
the reported market residual (read: `src/nfl_ats/margin.py:995-1000`).
Therefore the shift that serves `point_R1` is

```
center_offset_R1 = (beta_b - 1) * residual_raw + offset_b
```

because `line + residual_raw + [(beta_b - 1) * residual_raw + offset_b]
= line + beta_b * residual_raw + offset_b`. The incumbent is the same
expression at `beta_b = 1`: `center_offset_S3 = offset_b`. A row without a
resolvable bucket (missing line) keeps `beta = 1`, i.e. the incumbent.

The ridge, its feature profile, the out-of-time residual sample, the
`gaussian_median` probability mapping and the played three-member overlay
union are all unchanged. Probabilities come only from
`MarginModel.predict(center_offset=...)`.

### R1b (predeclared sibling)

Identical to R1 except that the three big buckets `7`, `7.5-10` and `10.5+`
share ONE pooled slope, fitted by the same formula on the union of their prior
rows (so `n` is the pooled count and the shrinkage weight is still 100). The
two small buckets keep their own R1 slopes. This is a predeclared sensitivity
on how finely the slope may vary, not a search over tunings.

### Positive control (predeclared)

The walk-forward `beta_hat` for the `3.5-6.5` bucket must stay clearly
positive, reported with a whole-week block bootstrap interval on the prior
rows used for the Week 1 2026 fit. If it does not, the instrument cannot see
slopes at all and no arm's reading is interpretable; that would be stated
plainly rather than treated as evidence about the mechanism.

### How it is graded

1. Replay S3 EXACTLY first on the active-matched opener archive. Stop before
   any candidate scoring if any served probability differs by more than 1e-9.
2. Paired accuracy at the OPENER (AGENTS.md: "Grade the decision at the
   OPENER"), non-push games only.
3. Whole-week block bootstrap, 20,000 draws, seed 20260817, within-week
   correlation ZERO -- never estimated, never padded (binding: games in a week
   are independent).
4. Standalone (model alone) AND through the played three-member card, via
   `overlay-composition --per-game-artifact` on research `per_game` artifacts
   carrying `research_arm` and `active_model_id` `research_laneR_R1` /
   `research_laneR_R1b`, never the active identity.
5. Reported per season, per bucket, by pick side (home picks / road picks --
   where lane L located the hole) and per bucket x pick side. The pick-side
   grouping uses the INCUMBENT's standalone opener pick, so the cells do not
   move with the arm.
6. Brier and log loss beside accuracy.
7. Week 1 2026 sides that would change, read from the linked forecast's
   `home_side_offset.json` sidecar and `predictions.csv`. No forecast is
   regenerated and no pick is changed by this lane.

Family `mod18_home_side_location_v1`; cell names
`mod18_home_side_location_v1_r1_*`; classification `unresolved_below_power`
unless an admissible closing ground applies.

### Disclosure

This is a POST-HOC mechanism found on the same mined 1,537-game archive that
S2 and S3 were selected on, and the `10.5+` slope estimate that motivated it
came from those same games. Walk-forward fitting removes look-ahead inside
each game, but it does not undo the archive-level selection: the reading here
is a correlated sensitivity on a reused window, not independent confirmation.

### Closing-grounds taxonomy (binding, verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval
on the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". The registry code
hard-rejects inadmissible closures; if a record command errors, the verdict is
wrong, not the validator. Never state that anything "needs more games"; decide
on expected value (P+ above 0.5 favours playing), and state what the numbers
imply for the DECISION (play R1 instead of S3?) before what is wrong with them.

### Decision rule, declared before the numbers

`probability_positive` above 0.5 through the played card favours PLAYING the
arm instead of S3; the pool submits 285 forced cards either way, so declining
a candidate that is more likely than not better is taking the other side of
that bet. The sensitivity R1b is disclosed beside R1, never selected on its
result.

## Measured results (appended after computation, 2026-09-08)

The digest stamped at replay time,
`2cdf77a1123bd801da1a668005e99bd01bc2286229c4daf51a02e816a62e8de1`, is the
SHA-256 of this file with the predeclaration section ONLY; appending this
section changes the file's digest, which is why the frozen value is recorded
in `artifacts/research/laneR/reproduction.json` rather than re-derived here.

### The decision first

**Keep S3; do not play R1 or R1b.** Through the played three-member card R1
scores 55.156% against S3's 55.888% -- **-0.732 accuracy points, 95% [-1.932,
+0.404], `probability_positive` 0.099** on 1,503 non-push opener games across
107 weeks; R1b scores 54.691%, **-1.198 [-2.451, +0.000], P+ 0.024**. On the
expected-value rule that is a 90/10 and a 98/2 bet against the arm, so the
decision is not close.

**The mechanism is not refuted by that decision, and nothing here is closed.**
In the bucket it was diagnosed in the arm does what lane L predicted: at
`10.5+` R1 is **+2.290 [-3.200, +8.276] P+ 0.745** standalone (49.62% against
47.33%), and on the road picks that carry the whole hole it is **+7.407
[-5.660, +21.569] P+ 0.834** (54 games). The loss is paid somewhere else --
`7.5-10`, where the residual is informative and the arm down-weighted it
anyway. Every cell is `unresolved_below_power`; no admissible closing ground
applies (no resolved wrong sign on the arm, no split-half reliability read, no
positive control that bounds an effect this size).

### Positive control (predeclared): the instrument can see slopes

Walk-forward slopes at the Week 1 2026 fit (1,310 prior archive rows, whole
weeks resampled, 20,000 draws, seed 20260817); `beta` is the same estimate
shrunk toward 1.0 with the 100-game prior:

| bucket | prior games | weeks | slope | 95% | P(slope > 0) | served beta |
|---|---:|---:|---:|---|---:|---:|
| 0-3 | 504 | 90 | +0.0498 | [-0.4919, +0.5911] | 0.576 | +0.207 |
| **3.5-6.5 (the control)** | 468 | 90 | **+0.5653** | [-0.1309, +1.2193] | **0.944** | +0.642 |
| 7 | 65 | 48 | +0.4413 | [-1.3763, +2.2933] | 0.693 | +0.780 |
| 7.5-10 | 160 | 76 | +0.9133 | [+0.0200, +1.8041] | 0.977 | +0.947 |
| 10.5+ | 113 | 64 | **-1.1745** | [-2.2255, -0.1138] | 0.015 | -0.154 |
| 7+ pooled (R1b) | 338 | 90 | +0.2065 | [-0.4291, +0.8501] | 0.739 | +0.388 |

The control holds: the `3.5-6.5` slope is clearly positive, so the estimator
can see a slope where one exists, and the negative overall reading below is a
reading of the ARM, not an instrument failure. Lane L's headline is also
reproduced out-of-time: on prior-only rows the `10.5+` slope is -1.17 with the
whole 95% interval below zero, so "the model's disagreement points the wrong
way on the biggest lines" is the best-supported part of this lane.

### Replay

Measured (`artifacts/research/laneR/reproduction.json`): archive
`artifacts/opener_evaluation/20260908T115957Z` (the active model
`a4c757efd2525da6`'s matched evaluation, 1,537 games), feature digest
`457aafb7...` equal to the active manifest. S3's served probabilities replay to
a maximum gap of **3.77e-15**, the served offsets to 6.66e-16 and the raw
residual to 1.40e-13 -- all far inside the 1e-9 stop rule, so candidate
scoring ran.

### Decision table (S3 versus R1 versus R1b)

1,503 non-push opener games, 107 weeks, week-blocked bootstrap 20,000 draws,
seed 20260817, within-week correlation zero.

| read | S3 | R1 | R1 delta, 95%, P+ | R1b | R1b delta, 95%, P+ |
|---|---:|---:|---|---:|---|
| model alone, opener | 54.558% | 53.759% | -0.798 [-2.177, +0.536] P+ 0.114 (88 flips) | 53.360% | -1.198 [-2.705, +0.326] P+ 0.053 (116 flips) |
| THROUGH the played card | **55.888%** | 55.156% | **-0.732 [-1.932, +0.404] P+ 0.099** (63 flips) | 54.691% | **-1.198 [-2.451, +0.000] P+ 0.024** (80 flips) |
| Brier (improvement) | 0.251584 | 0.250543 | +0.001041 [-0.000220, +0.002309] P+ 0.945 | 0.250302 | +0.001282 [-0.000121, +0.002704] P+ 0.964 |
| log loss (improvement) | 0.696624 | 0.694378 | +0.002246 [-0.000373, +0.004875] P+ 0.954 | 0.693862 | +0.002762 [-0.000124, +0.005696] P+ 0.969 |
| Week 1 2026 sides changed | reference | 3 of 16 | ARI at LAC, NE at SEA, NYJ at TEN | 3 of 16 | the same three games |

Both arms make the stated chances BETTER (Brier and log loss improve at P+
0.945-0.969) while making the forced pick WORSE. That is coherent rather than
contradictory: shrinking the residual pulls the served probability toward
50%, the right direction for a model that states about 56% and is right about
54%, but near the 50% threshold it also flips picks, and those flips lose.

### By pick side (where lane L located the hole)

Cells cut by the INCUMBENT's standalone opener pick, so the grouping does not
move with the arm.

| cell | n | R1 standalone | R1 card | R1b card |
|---|---:|---|---|---|
| picked the home team | 651 | -0.768 [-2.462, +0.818] P+ 0.153 | -0.768 [-2.294, +0.703] P+ 0.132 | -0.922 [-2.572, +0.736] P+ 0.115 |
| picked the road team | 852 | -0.822 [-2.945, +1.225] P+ 0.200 | -0.704 [-2.494, +0.974] P+ 0.192 | -1.408 [-3.222, +0.245] P+ 0.045 |
| 10.5+ and picked the road team | 54 | **+7.407 [-5.660, +21.569] P+ 0.834** | +1.852 [-9.091, +13.208] P+ 0.568 | +1.852 [-9.091, +13.208] P+ 0.568 |
| 10.5+ and picked the home team | 77 | -1.299 [-4.110, +0.000] P+ 0.000 | -1.299 [-4.110, +0.000] P+ 0.000 | -3.896 [-8.642, +0.000] P+ 0.000 |
| 7.5-10 and picked the road team | 112 | **-6.250 [-13.043, +0.000] P+ 0.011** | -5.357 [-10.680, -0.862] P+ 0.005 | -8.929 [-16.505, -2.586] P+ 0.001 |
| 7.5-10 and picked the home team | 82 | +0.000 [-4.938, +4.880] P+ 0.396 | -1.220 [-5.682, +2.597] P+ 0.185 | +0.000 [-3.659, +3.614] P+ 0.350 |

### By bucket

| bucket | n | S3 right | R1 standalone | R1 card | R1b card |
|---|---:|---:|---|---|---|
| 0-3 | 552 | 56.16% | -0.906 [-2.737, +0.945] P+ 0.143 | +0.181 [-1.357, +1.748] P+ 0.532 | +0.181 [-1.357, +1.748] P+ 0.532 |
| 3.5-6.5 | 552 | 56.16% | -1.268 [-3.104, +0.546] P+ 0.070 | -0.906 [-2.708, +0.904] P+ 0.129 | -0.906 [-2.708, +0.904] P+ 0.129 |
| 7 | 74 | 51.35% | +5.405 [+1.266, +11.250] P+ 0.985 | +0.000 [+0.000, +0.000] P+ 0.000 | -2.703 [-9.231, +4.000] P+ 0.148 |
| 7.5-10 | 194 | 51.55% | **-3.608 [-7.921, +0.490] P+ 0.027** | -3.608 [-7.104, -0.500] P+ 0.007 | -5.155 [-9.444, -1.053] P+ 0.003 |
| 10.5+ | 131 | 47.33% | **+2.290 [-3.200, +8.276] P+ 0.745** | +0.000 [-4.688, +4.878] P+ 0.447 | -1.527 [-6.667, +3.817] P+ 0.242 |

### By season (through the played card)

| season | n | R1 | R1b |
|---|---:|---|---|
| 2020 | 220 | -1.364 [-3.211, +0.465] P+ 0.045 | -1.364 [-3.167, +0.472] P+ 0.048 |
| 2021 | 236 | +0.000 [-2.620, +2.532] P+ 0.447 | -0.424 [-3.333, +2.146] P+ 0.332 |
| 2022 | 248 | -0.806 [-3.600, +1.961] P+ 0.240 | -1.210 [-4.065, +1.606] P+ 0.165 |
| 2023 | 266 | -0.752 [-3.371, +1.812] P+ 0.232 | -1.128 [-4.264, +1.916] P+ 0.204 |
| 2024 | 266 | -0.752 [-4.511, +2.682] P+ 0.313 | -1.128 [-5.344, +2.682] P+ 0.271 |
| 2025 | 267 | -0.749 [-3.759, +2.299] P+ 0.274 | -1.873 [-4.151, +0.375] P+ 0.033 |

### Week 1 2026 (read only, no pick changed)

The linked card `margin_predictions/2026-week-01-20260908T124514Z` replays
from its own sidecar to 9.99e-16. Three of sixteen served sides would change
under R1, the same three under R1b: ARI at LAC (line 10.5, raw residual
-5.81, home cover 0.390 to 0.599 under R1 and 0.501 under R1b), NE at SEA
(3.5, 0.4976 to 0.5026) and NYJ at TEN (1.5, 0.4843 to 0.5060). Two of those
three sit within a quarter-point of the 50% threshold under S3.

### Why it lost, and what to fit differently (inferred)

The served slope is walk-forward, so early weeks fit it on a thin window, and
the shrinkage weight of 100 games was borrowed from an estimator of a MEAN.
For a SLOPE that prior is far too weak. Measured season means of the unshrunk
walk-forward slope (`artifacts/research/laneR/slopes.parquet`): `7.5-10` reads
-1.89 in 2020, -0.50 in 2021, then +0.07, +0.07, -0.00, +0.45, so the served
`beta` for that bucket sat between 0.41 and 0.67 for most of the archive --
even though the full prior-only window at Week 1 2026 puts that slope at
**+0.91 [+0.02, +1.80]**. The arm therefore spent six seasons down-weighting
the one big bucket whose residual is informative, and paid -3.6 points there.
The same weakness runs the other way in `0-3`: the slope estimate is noise
around zero (+0.05 [-0.49, +0.59]) yet 504 prior games drag `beta` to 0.21, a
large change to the biggest bucket bought with no evidence.

Two concrete, predeclarable follow-ups, both disclosed as further post-hoc
work on this same archive:

1. Derive the shrinkage weight from the slope's OWN sampling variance
   (empirical Bayes on the between-bucket variance of the slope) instead of
   borrowing the offset's 100-game count prior. The owner's rule that an
   underived constant gating an irreversible decision is a defect applies to
   the 100 here exactly as it applies to `PRIOR_WEIGHT_GAMES` in the offset.
2. Serve the rescale ONLY where the mechanism was diagnosed (`10.5+`), the way
   S3 restricted the offset to the big buckets. On this archive that cell
   reads +2.29 P+ 0.745 standalone and +7.41 P+ 0.834 on its road picks. That
   is bucket selection AFTER seeing the split, on the archive that produced
   the split, so it must be predeclared and graded as its own arm, never
   quoted from this table.

### Provenance and scope

Measured this session by
`.\.tools\uv.exe run --no-sync python scripts\residual_slope_opener_eval.py
--stage replay|score|week1|record`. Artifacts (all provenance-stamped) under
`artifacts/research/laneR/`: `replay.parquet`, `slopes.parquet`,
`scored.parquet`, `cells.json`, `week1.parquet`, `week1.json`,
`reproduction.json`, `record_commands.ps1`, the research
`opener_evaluation/R1` and `opener_evaluation/R1b` per-game artifacts (model
identities `research_laneR_R1` / `research_laneR_R1b`, never the active
identity) with their `overlay_subset_composition` runs, and `commands.json`
with the exact successful argv. The Week 1 2026 card was READ only; no
forecast was regenerated and no pick was changed by this lane.

192 `weak-signals record` commands (96 cells x 2 arms) are written to
`artifacts/research/laneR/record_commands.ps1` for the coordinator to run
serially -- this lane never touches the shared registry. Every row is
`unresolved_below_power`, carries the disclosure above in `--notes` and a
plain-English `--plain-summary`, and all 192 were verified to parse against
the real CLI parser without executing.
