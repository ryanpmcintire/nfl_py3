# MOD-18 lane S: a home-side location correction on big spreads

Predeclared 2026-09-07 (lane S) BEFORE any score was computed. Family
`mod18_home_side_location_v1`. Grade: opener. The scoring section at the end
is appended after the frozen procedure ran and is labelled as such.

## What is known (read before this was designed)

Lane Q (`docs/home_dog_location.md`, its scored section; read) re-ran lane
L's Diagnosis D on the TRUE home split of the active model's opener
evaluation (`artifacts/opener_evaluation/20260907T152026Z`, 1,537 games,
2020-2025). Home point error = actual home margin minus the incumbent's point
forecast (`tue_open_home_spread + residual_at_open`):

- 10.5+ home favourites: +2.41 points [+0.11, +4.68], probability_positive
  0.980 (102 games).
- 10.5+ home underdogs: +4.50 [-0.66, +9.40], probability_positive 0.955
  (32 games).
- 7.5-10 home underdogs: +1.13 [-1.54, +3.90], 0.799 (69); home favourites
  +1.05 [-1.64, +3.64], 0.778 (126).
- 0-3 and 3.5-6.5: no such error on either side.

So the model under-locates the HOME team on big spreads on BOTH sides. Lane
Q put a home-underdog-only term into the ridge (`home_dog_points`, and a
hinge above 7): -0.33 / -0.27 accuracy points at the opener, and the home
error did not shrink, because the ridge weights the term under 0.1 point per
point with a sign that flips across training epochs. Lane Q's own follow-ups
are (a) a symmetric home-side hinge and (b) a correction OUTSIDE the ridge.
This lane runs exactly those two, nothing else.

nflverse convention throughout: a POSITIVE `spread_line` means the HOME team
is favoured; `|spread_line|` is the spread size regardless of side.

**Inferred mechanism (stated before scoring, not a threshold flip):** the
ridge carries one constant home-field term, while the realised home
advantage on big spreads is larger than that constant on either side
(candidate football reasons: a road favourite sitting on a lead compresses
big road wins; a home underdog's crowd and garbage-time scoring keep it
closer). Whatever the reason, the residual is a function of "how big is the
spread" that is common to both home sides, so the fix is a location term
that grows with the spread size, learned from prior games.

## Frozen candidates (no tuning, no sign selection, nothing else)

- **S1 `home_side_hinge_7`** (inside the ridge): one added feature
  `home_side_hinge_7` = max(0, |spread_line| - 7), a function of the row's
  own line only, computed in `src/nfl_ats/home_side_location.py`. With
  `spread_line` already in weak_stack, this is the only non-collinear
  symmetric shape (|s| itself is s + 2 max(0, -s), an exact combination of
  `spread_line` and lane Q's column, and was not run). Profile
  `weak_stack_home_side_hinge_7` = weak_stack + that one column; ridge alpha
  10; target `market_residual`; mapping `gaussian_median`, unchanged.
- **S2 `home_offset_by_size_walkforward`** (outside the ridge): the incumbent
  produces its point forecast for a target week exactly as archived; then a
  home-side offset c(b) is ADDED to the point forecast, where b is the
  row's spread bucket at the line it is scored at, buckets {0-3, 3.5-6.5,
  7-7.5, 7.5-10, 10.5+} (`spread_bucket` in `src/nfl_ats/spread_regime.py`,
  reused verbatim), and

  c(b) = sum over prior games in b of (actual home margin - incumbent point
  forecast) / (n_b + 100).

  "Prior games" are the archived opener-evaluation rows (the incumbent's
  OUT-OF-TIME points: `tue_open_home_spread + residual_at_open`) whose
  gameday plus one day is strictly before the earliest gameday of the target
  week (one-day completion allowance), in seasons no earlier than the target
  season minus five (five trailing seasons plus the target season's prior
  weeks), with the whole target week excluded. The prior weight of 100 games
  is fixed regularisation toward zero, declared here, not a tuned bar. The
  cover probability is the UNCHANGED gaussian_median mapping (the incumbent
  weekly fit's residual median and standard deviation) evaluated at the
  corrected point; the probability shape is not this lane's (lane T owns
  it). Training rows exist only from 2020 (the archive), so 2020's early
  weeks are shrunk almost entirely to zero by construction.

Neither arm touches the active model, `weekly.py`, the played profile or
the card. Pregame safety: S1 is row-local; S2 uses only prior completed
games. `tests/test_home_side_location.py` pins both (a future game or a
mutated later result cannot change an earlier row's feature or offset).

## Frozen measurement (identical to lanes H/K/L/Q)

Harness `scripts/home_side_location_opener_eval.py`, isolated
`NFL_ATS_ARTIFACTS_DIR` / `NFL_ATS_REGISTRY_DIR` scratch roots seeded with
the active manifest and copies of the live registries; `data/` read-only;
BLAS/OpenMP threads held to 1.

1. S1: chronological weekly refit exactly as `clv.evaluate_opener_per_game`
   (training = completed regular-season games with `gameday` strictly
   before the earliest archived game of the week; `spread_line` overridden
   to the archived Tuesday opener before the column is attached). The
   baseline profile's picks and probabilities must replay the archive
   exactly before any candidate number is read; the stored weekly
   gaussian_median parameters must reproduce the archived probabilities.
   S2: post-processing of the archived out-of-time points; no refit.
2. Primary: paired forced-pick accuracy at the OPENER versus the active
   model's own probability-rule picks on the 1,537-game archive (1,503
   non-push), accuracy points; week-blocked bootstrap 20,000 draws, seed
   20260817, within-week correlation ZERO; frozen-pick within-week
   permutation null (2,000 draws); per-season deltas.
3. Secondary: Brier and log-loss improvement (own units); reliability by
   the lane-J spread buckets and by home favourite / home underdog; the
   home point-error table by bucket and home side BEFORE and AFTER for both
   arms (the number this lane exists to move: do the 10.5+ home-side errors
   shrink toward zero?), with MAE improvement cells.
4. THROUGH THE PLAYED CARD: `nfl-ats overlay-composition` on a scratch
   opener evaluation per arm; candidate three-member card (coach fade +
   division revenge + player arrests) versus the incumbent three-member
   card.
5. Week 1 2026: read-only scratch prediction per arm versus the forecast
   named in the active manifest; list the sides that change.
6. Windows: `nfl-ats rotation declare --grade opener --acknowledge-mined`
   then `rotation assign` BEFORE scoring; each block recorded with
   `rotation record` before the next is assigned; every arm, window,
   season, loss, bucket, point-error and card cell recorded through
   `nfl-ats weak-signals record` with a `--plain-summary`. Names carry the
   prefix `mod18_home_side_location_v1_`.

**Reuse discount, stated plainly:** the 2020-2025 opener archive has been
read by lanes H (5 arms), K (3), L (2), Q (2), the zone-fade bounds, the
127-subset union and the ~130-150-look 2018-2025 ledger, and BOTH arms here
were motivated by a diagnosis read on this same archive (S2 additionally
trains on it walk-forward). Every number below is a descriptive read on
reused data, not an independent confirmation. S1 is additionally scored on
2011-2019 at the nflverse (close-proxy) grade as a cheaper replication; S2
has no pre-2020 training stream and is not replicated.

**Decision rule:** expected value at the opener through the played card;
`probability_positive` above 0.5 favours playing it. No interval crossing
zero closes anything; only a resolved wrong sign (whole interval on the
wrong side of zero) or a positive-control bound could, and neither is
available here. Everything else is recorded `unresolved_below_power`.

## Scored 2026-09-07 (appended AFTER the frozen procedure ran; nothing above was edited)

All numbers below are **measured** this session by
`scripts/home_side_location_opener_eval.py --out <scratch>/laneS --stage
build|score|losses|diagnosis|composition|week1|replay` (scratch root
`C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3\dcbb74c0-77a9-470f-8e6e-713bc3331924\scratchpad\laneS`),
against the active model `a4c757efd2525da6` and its matched archive
`artifacts/opener_evaluation/20260907T152026Z` (1,537 games, 1,503 non-push,
2020-2025). The baseline profile replayed the archive exactly before any
candidate number was opened (max probability difference 3.8e-15, 0 pick
disagreements); the stored weekly gaussian_median parameters reproduce the
archived probabilities to 0.0 and the base stream replays the archived
points to 1.4e-13. Every cell is recorded `unresolved_below_power` under the
prefix `mod18_home_side_location_v1_` (88 weak-signal rows, 3 spent opener
windows, all `unresolved`).

**Decision read (opener, paired vs the active model's own picks, 1,503 non-push):**

| arm | delta (pts) | 95% week-blocked | probability_positive | picks changed | through the played three-member card |
|---|---|---|---|---|---|
| S1 `home_side_hinge_7` (in the ridge) | -0.13 (53.83% vs 53.96%) | [-0.85, +0.59] | 0.324 | 26 | 0.00 [-0.53, +0.59], P+ 0.448 (16 flips) |
| S2 `home_offset_by_size_walkforward` (outside the ridge) | -0.20 (53.76%) | [-1.60, +1.20] | 0.368 | 107 | **+0.33 (55.56% vs 55.22%) [-0.80, +1.46], P+ 0.695** (71 flips) |

Frozen-pick within-week permutation null (2,000 draws): S1 at the 38.2nd
percentile of its own null [-0.53, +0.67], S2 at the 35.3rd of [-1.26,
+1.40]; family-max null [-0.47, +1.40].

**Decision rule applied:** expected value at the opener through the played
card, P+ above 0.5 favours playing. S1 is below 0.5 on both reads: not
played. S2 is 0.368 standalone but **0.695 through the played card**, which
is the declared decision read, so the rule favours playing S2 as a lean --
with the swing between the two reads stated plainly: standalone the 107
changed picks net -3 correct; after the three-member overlays only 71 picks
differ and they net +5 correct, so the whole difference between "do not
play" and "play" is 8 games of 1,503. Nothing is closed: no full-archive
interval sits wholly on the wrong side of zero and no positive control
bounds an effect this size. The lane does not change the card; the
coordinator decides.

Windows (opener pool, earliest-eligible blocks, all recorded `unresolved`):
2020-21 S1 -0.22 [-1.50, +1.10] P+ 0.298 / S2 +0.44 [-1.55, +2.47] P+ 0.624
(n 456); 2022-23 S1 -0.58 [-1.91, +0.59] P+ 0.137 / S2 -1.36 [-3.89, +1.17]
P+ 0.135 (n 514); 2024-25 S1 +0.38 [-0.75, +1.52] P+ 0.676 / S2 +0.38
[-1.93, +2.86] P+ 0.580 (n 533).
Per season (S1 / S2): 2020 0.00 / 0.00; 2021 -0.42 / +0.85; 2022 +0.81 /
-0.81; 2023 -1.88 [-3.80, -0.37] P+ 0.000 / -1.88 [-5.28, +1.87] P+ 0.132;
2024 -0.38 / -2.26 [-4.76, +0.38] P+ 0.030; 2025 +1.12 [-0.73, +3.03] P+
0.841 / +3.00 [-0.73, +6.96] P+ 0.934.

**Brier / log loss** (improvement, positive favours the arm): S1 Brier
+0.000006 [-0.00027, +0.00027] P+ 0.525, log loss +0.000007 P+ 0.517; S2
Brier -0.00057 [-0.00149, +0.00034] P+ 0.111, log loss -0.00124 [-0.00315,
+0.00064] P+ 0.100 (baseline Brier 0.25174, log loss 0.69694). S2's
corrected probabilities are less calibrated overall even where its picks
gain.

**S1 replication at the nflverse (close-proxy) grade:** 2011-2017 -0.12
[-0.92, +0.69] P+ 0.363 (n 1,743); 2018-2019 0.00 [-1.01, +1.01] P+ 0.420
(n 493). Not opener evidence; S2 has no pre-2020 training stream.

### Reliability by lane-J bucket (accuracy; stated confidence 55-57% in every cell for every arm, the mapping being unchanged)

| bucket | n | incumbent | S1 | S2 |
|---|---|---|---|---|
| 0-3 | 552 | 56.2% | 56.5% | 54.2% |
| 3.5-6.5 | 552 | 56.2% | 56.0% | 56.0% |
| 7 | 74 | 52.7% | 54.1% | 51.4% |
| 7.5-10 | 194 | 48.5% | 48.5% | 51.5% |
| 10.5+ | 131 | 44.3% | 41.2% | 47.3% |

By TRUE home side (paired accuracy-point cells vs the incumbent,
`bucket_cells.json`): S2 7.5-10 home underdogs **+7.25 [+1.49, +13.89] P+
0.995** (n 69; S2 picks the home side 27.5% vs 20.3%); S2 7.5-10 all +3.09
[-0.53, +6.99] P+ 0.937; S2 10.5+ home favourites +7.07 [-1.00, +15.31] P+
0.948 (n 99); S2 10.5+ home underdogs -9.38 [-24.14, +3.03] P+ 0.044 (n 32);
S2 0-3 all **-1.99 [-4.23, +0.18] P+ 0.028** (n 552; the 0-3 offset was
negative, -0.4 to -0.9 points, in the 2020-2023 fits and S2 picks the home
side 35% there against the incumbent's 41%). S1 0-3 home favourites +1.02
[0.00, +2.29] P+ 0.952 (n 294); S1 10.5+ home underdogs -9.38 [-20.69, 0.00]
P+ 0.000 (n 32); S1 3.5-6.5 home underdogs -0.85 P+ 0.000 (n 235).

### Did the home point error shrink? S2 by about a third at 10.5+; S1 barely (`diagnosis.parquet`)

Home point error = actual home margin minus the arm's point forecast
(incumbent: `tue_open_home_spread + residual_at_open`; S1: opener line +
its own residual; S2: incumbent point + the walk-forward offset). Pushes
included.

| bucket / side | n | incumbent | S1 | S2 |
|---|---|---|---|---|
| 10.5+ all | 134 | +2.91 [+0.83, +5.01] P+ 0.997 | +2.78 [+0.71, +4.88] P+ 0.996 | **+1.97 [-0.12, +4.08] P+ 0.968** |
| 10.5+ HOME FAVOURITE | 102 | +2.41 [+0.11, +4.68] P+ 0.980 | +2.32 [+0.02, +4.59] P+ 0.976 | **+1.48 [-0.83, +3.74] P+ 0.896** |
| 10.5+ HOME UNDERDOG | 32 | +4.50 [-0.66, +9.40] P+ 0.955 | +4.26 [-0.92, +9.17] P+ 0.944 | **+3.53 [-1.66, +8.46] P+ 0.907** |
| 10.5+ incumbent picked the underdog | 65 | +5.18 [+2.10, +8.20] P+ 0.999 | +5.05 | +4.27 [+1.14, +7.33] P+ 0.996 |
| 7.5-10 all | 195 | +1.08 [-1.01, +3.20] P+ 0.839 | +1.06 | +0.66 [-1.45, +2.79] P+ 0.724 |
| 7.5-10 HOME UNDERDOG | 69 | +1.13 [-1.54, +3.90] P+ 0.799 | +1.06 | +0.71 [-1.95, +3.48] P+ 0.703 |
| 7.5-10 HOME FAVOURITE | 126 | +1.05 [-1.64, +3.64] P+ 0.777 | +1.05 | +0.63 [-2.08, +3.24] P+ 0.672 |
| 0-3 all | 573 | -0.10 [-1.16, +0.96] P+ 0.419 | -0.07 | +0.34 [-0.73, +1.40] P+ 0.732 |

MAE improvement (points, positive favours the arm, `mae_cells.json`): S2
10.5+ home underdogs +0.22 [-0.15, +0.57] P+ 0.881 (n 32), 10.5+ all +0.10
[-0.08, +0.27] P+ 0.857, 7.5-10 home underdogs +0.085 [-0.02, +0.19] P+
0.943, overall -0.002 [-0.033, +0.028] P+ 0.433 (n 1,537); S1 overall -0.002
[-0.010, +0.006] P+ 0.329.

Why S2 removes only a third of the 10.5+ error (measured,
`offsets_by_season.parquet`, `week1.json`): the offsets are walk-forward and
shrunk with a 100-game prior, so at 10.5+ they are about 0 in 2020, +0.5 in
2021, +1.2 in 2022, +1.4 in 2023, +1.5 in 2024-2025 and +1.92 for Week 1
2026 (113 prior games; the raw mean is about +2.9, halved by the prior
weight at that sample size). At 7.5-10 they run +0.3 to +0.7 (+0.78 for
2026); at 3.5-6.5 +0.38 for 2026; at 0-3 negative through 2023 and +0.03 for
2026; at exactly 7, -0.17. What S1's ridge learned (`coefficients.parquet`,
per point above seven, unscaled, mean over each season's weekly fits):
-0.12 and -0.05 in the 2020 and 2021 fits, +0.10 to +0.12 in 2022-2026
(2013-2017 fits +0.06 to +0.19, 2018-2019 about 0 to -0.09) -- a tenth of a
point per point, sign-flipping across epochs, the same failure lane Q saw:
the ridge cannot hold a location term of this size.

### Week 1 2026

Forecast read from the active manifest via `active_artifact_path(artifacts,
active, "weekly_forecast")` =
`artifacts/margin_predictions/2026-week-01-20260907T232720Z`. Lines
identical to the feature table; harness baseline reproduces the forecast to
8.3e-16; `nfl-ats margin-predict ... --feature-profile
weak_stack_home_side_hinge_7` (scratch artifacts root) reproduces the S1
harness to 5.6e-17; S2 recomputed from the live forecast's own point and
the five-trailing-season offsets matches the walk-forward stream to 7.8e-16.
**S1 changes 0 of 16 sides. S2 changes one: NE +3.5 at SEA, forecast NE ->
S2 SEA (home cover probability 0.4976 -> 0.5098)** -- the same game and
direction lane Q's Q2 moved. No live forecast, ledger or card was touched.
