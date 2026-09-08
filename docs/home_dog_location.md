# MOD-18 lane Q: let the ridge learn the home-underdog location error

Predeclared 2026-09-07 (lane Q) BEFORE any score was computed. Family
`mod18_home_dog_location_v1`. Grade: opener. The scoring sections below were
appended after the frozen procedure ran and are labelled as such.

## What lane L found (read, `docs/hybrid_margin_mapping.md`, "Diagnosis D")

The active model's POINT forecast (`tue_open_home_spread + residual_at_open`)
is fine where the home team is favoured and wrong where the home team is a big
underdog. Read from that table (reported by lane L, re-measured below on the
candidate and the incumbent alike):

- 10.5+ spreads, all 134 games: actual home margin minus the point forecast
  +2.91 points [+0.83, +5.01], probability_positive 0.997.
- 10.5+ spreads, the 65 games where the incumbent picked the underdog:
  +5.18 [+2.10, +8.20], probability_positive 0.999; the 69 favourite picks
  +0.77 [-2.46, +3.82], probability_positive 0.68.
- 7.5-10 spreads, the 84 underdog picks: +2.57 [-0.74, +5.84], probability_
  positive 0.936; the 111 favourite picks -0.05, probability_positive 0.49.
- 0-3 and 3.5-6.5: no such error on either side.

nflverse convention throughout: a POSITIVE `spread_line` means the HOME team
is favoured; the home team is the underdog when `spread_line < 0`.

**Inferred mechanism (stated before scoring, not a threshold flip):** the
ridge carries one constant home-field term while the market prices home
field as spread-dependent; when the home team is a big underdog the model
under-locates it by several points (candidate football reasons: garbage-time
and backdoor dynamics that compress big road wins, a road favourite's
tendency to sit on a lead). Whatever the reason, the error is a smooth
function of "how big a home underdog is this", so the right fix is a feature
the ridge can weight from prior games, not a rule that flips picks at 7.5.

## Frozen candidates (no tuning, no sign selection, nothing else)

Both are functions of the row's own `spread_line` at the line the row is
scored at (the archived Tuesday opener in the paired archive, the nflverse
line in training), computed in `src/nfl_ats/home_dog_location.py`:

- **Q1 `home_dog_points`** = max(0, -spread_line): points by which the home
  team is the underdog; 0 when the home team is favoured or pick'em.
  Profile `weak_stack_home_dog_points` = weak_stack + that one column.
- **Q2 `home_dog_hinge_7`** = Q1 plus max(0, -spread_line - 7): the same
  quantity above seven points, so the fitted correction may bend where lane L
  located the error. Profile `weak_stack_home_dog_hinge_7` = weak_stack + both
  columns.

Ridge alpha stays 10; the probability mapping stays `gaussian_median`; the
target stays `market_residual`; the active model, `weekly.py`, the played
profile and the card are untouched. Pregame safety: the columns are row-local,
so a future game or a mutated result cannot move an earlier row's feature;
`tests/test_home_dog_location.py` pins that.

### Correction received before scoring (2026-09-07, coordinator via lane P)

Read before any candidate number existed (the walk-forward build was still
running; no `p_Q1`/`p_Q2` value had been opened): lane L's "underdog" rows are
stratified by the INCUMBENT'S PICK SIDE, not by which team is the underdog.
Lane P's true home/away split of the same opener evaluation (reported,
unverified here until Diagnosis D below re-runs it): 7.5-10 home underdogs
(69 games) home covered 59.4% against the model's 44.9%; home favourites
(125) -0.1 pp; 10.5+ home underdogs (32) 59.4% vs 47.0%; 10.5+ home
favourites (99) 57.6% vs 49.4%, model right 39.4%. So at 10.5+ the home
team over-covers on BOTH sides; the home-underdog concentration holds at
7.5-10.

Decision, stated before scoring: Q1 and Q2 stay exactly as declared. The
proposed third arm `home_spread_size` = |spread_line| is NOT added, because
`spread_line` is already a weak_stack feature and |s| = s + 2 * max(0, -s)
is an exact linear combination of `spread_line` and Q1's `home_dog_points`
(read: `margin_feature_columns("market_residual", "weak_stack")` lists
`spread_line`); it would add no function the ridge cannot already fit. Q1
already gives the ridge separate home-location slopes on the two sides
(the `spread_line` coefficient on the favourite side, that plus the
`home_dog_points` coefficient on the underdog side). A genuinely new
symmetric term would be a hinge in |s| (max(0, |s| - 7)); it is proposed as
follow-up text, not added after the fact. Diagnosis D below is run on the
TRUE home-favourite / home-underdog split as well as the pick-side split.

## Frozen measurement (identical to lanes H/K/L)

Harness `scripts/home_dog_location_opener_eval.py`, isolated
`NFL_ATS_ARTIFACTS_DIR` / `NFL_ATS_REGISTRY_DIR` scratch roots seeded with the
active manifest and the live registries; `data/` read-only; BLAS/OpenMP
threads held to 1.

1. Chronological weekly refit exactly as `clv.evaluate_opener_per_game`:
   training = completed regular-season games with `gameday` strictly before
   the earliest archived game of the week; `spread_line` overridden to the
   archived Tuesday opener before the candidate columns are attached; the
   baseline profile's picks and probabilities must replay the archive
   (`artifacts/opener_evaluation/20260907T152026Z`, matched to the active
   model by `find_matching_opener_evaluation`) exactly before any candidate
   number is read.
2. Primary metric: paired forced-pick accuracy at the OPENER versus the active
   model's own probability-rule picks on the 1,537-game archive (1,503
   non-push), in accuracy points; week-blocked bootstrap 20,000 draws, seed
   20260817, within-week correlation ZERO; frozen-pick within-week
   permutation null (2,000 draws, same seed); per-season deltas.
3. Secondary: Brier and log-loss improvement (own units); reliability
   (accuracy vs stated confidence) by the lane-J spread buckets and by
   home-favourite / home-underdog; Diagnosis D re-run on the candidate point
   forecasts so the +5.18 home-underdog error at 10.5+ can be seen to shrink
   or not.
4. THROUGH THE PLAYED CARD: `nfl-ats overlay-composition` on the scratch
   opener evaluation for each arm; candidate three-member card (coach fade +
   division revenge + player arrests) versus the incumbent three-member card.
5. Week 1 2026: read-only scratch prediction for each arm versus the forecast
   named in the active manifest (`weekly_forecast.artifact`); list the sides
   that change.
6. Windows: `nfl-ats rotation declare --grade opener --acknowledge-mined`
   then `rotation assign` (earliest eligible two-season opener block) BEFORE
   scoring; the assigned block is recorded with `rotation record`, the next
   block assigned only after the previous is recorded; every arm, window,
   loss, bucket and card cell is recorded through `nfl-ats weak-signals
   record` with `--plain-summary`. Names carry the prefix
   `mod18_home_dog_location_v1_`.

**Reuse discount, stated plainly:** the 2020-2025 opener archive has been
mined by lane H (5 arms), lane K (3), lane L (2), the zone-fade bounds and
the 127-subset union, and by the ~130-150-look 2018-2025 ledger; the feature
itself was motivated by a diagnosis read on this same archive. Any number
below is a descriptive read on reused data, not an independent confirmation.
The 2011-2019 seasons are additionally scored at the nflverse (close-proxy)
grade as a cheaper replication that the diagnosis did not touch.

**Decision rule:** expected value at the opener through the played card;
`probability_positive` above 0.5 favours playing it; no interval crossing zero
closes anything; only a resolved wrong sign or a positive-control bound could,
and neither is available here. Everything else is recorded
`unresolved_below_power`.

## Scored 2026-09-07 (appended AFTER the frozen procedure ran; nothing above was edited)

All numbers below are **measured** this session by
`scripts/home_dog_location_opener_eval.py --out <scratch>/laneQ --stage
build|score|losses|diagnosis|composition|week1|replay` (scratch root
`C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3\dcbb74c0-77a9-470f-8e6e-713bc3331924\scratchpad\laneQ`),
against the active model `a4c757efd2525da6` and its matched archive
`artifacts/opener_evaluation/20260907T152026Z` (1,537 games, 1,503 non-push,
2020-2025). The baseline profile replayed the archive exactly before any
candidate number was opened (max probability difference 3.8e-15, 0 pick
disagreements). Every cell is recorded `unresolved_below_power` under the
prefix `mod18_home_dog_location_v1_` (88 weak-signal rows, 3 spent opener
windows, all `unresolved`).

**Decision read (opener, paired vs the active model's own picks):**

| arm | delta (pts) | 95% week-blocked | probability_positive | picks changed | through the played three-member card |
|---|---|---|---|---|---|
| Q1 `home_dog_points` | -0.33 (53.63% vs 53.96%) | [-1.06, +0.39] | 0.155 | 27 of 1,503 | 0.00 [-0.54, +0.54], P+ 0.462 |
| Q2 `+ home_dog_hinge_7` | -0.27 (53.69%) | [-1.26, +0.73] | 0.273 | 62 | -0.27 [-1.07, +0.59], P+ 0.235 |

Frozen-pick within-week null (2,000 draws): Q1 sits at the 13th percentile of
its own null, Q2 at the 37th. Windows: 2020-21 Q1 +0.22 P+ 0.574 / Q2 0.00
P+ 0.449; 2022-23 Q1 -1.36 [-2.71, -0.19] P+ 0.011 / Q2 -0.78 P+ 0.175;
2024-25 Q1 +0.19 P+ 0.563 / Q2 0.00 P+ 0.460. Seasons (Q1 / Q2): 2020 0.00 /
-1.36; 2021 +0.42 / +1.27; 2022 0.00 / -0.40; 2023 -2.63 [-4.56, -0.79] /
-1.13; 2024 -0.38 / +0.38; 2025 +0.75 / -0.37. Brier improvement Q1
+0.000005 P+ 0.522, Q2 -0.00002 P+ 0.477; log-loss improvement Q1 -0.00001
P+ 0.493, Q2 -0.00009 P+ 0.441. Nflverse-graded 2011-2017 (not in the mined
ledger, n 1,743): Q1 +0.23 [-0.58, +1.04] P+ 0.683; Q2 +0.80 [-0.17, +1.81]
P+ 0.938; 2018-2019 (n 493): Q1 -0.20 P+ 0.307, Q2 -0.41 P+ 0.289.

**Expected value says: do not play either arm; the incumbent stays.** Both
are below P+ 0.5 at the opener standalone and through the played card.
Nothing is closed: no whole interval sits on the wrong side of zero on the
full archive, and no positive control bounds a feature of this size.

**Did the home point error shrink? Barely.** Diagnosis D on the TRUE home
side (home point error = actual home margin minus the arm's point forecast):

| bucket / side | n | incumbent | Q1 | Q2 |
|---|---|---|---|---|
| 10.5+ all | 134 | +2.91 [+0.83, +5.01] P+ 0.997 | +2.86 | +2.62 [+0.54, +4.72] P+ 0.993 |
| 10.5+ home favourite | 102 | +2.41 [+0.11, +4.68] P+ 0.980 | +2.37 | +2.48 |
| 10.5+ home underdog | 32 | +4.50 [-0.66, +9.40] P+ 0.955 | +4.40 | +3.04 [-2.12, +7.99] P+ 0.873 |
| 10.5+ incumbent picked the underdog (lane L's row) | 65 | +5.18 [+2.10, +8.20] P+ 0.999 | +5.11 | +5.00 |
| 7.5-10 home underdog | 69 | +1.13 [-1.54, +3.90] P+ 0.799 | +1.09 | +1.21 |
| 7.5-10 home favourite | 126 | +1.05 P+ 0.778 | +1.03 | +1.05 |

Lane P's correction is confirmed here: at 10.5+ the home team beats the
point forecast on BOTH sides (favourites +2.41, P+ 0.98 on 102 games), so a
home-underdog-only feature addressed at most a third of the games carrying
the error. Absolute point error (MAE improvement, positive favours the arm):
Q1 -0.002 points overall P+ 0.325; Q2 -0.002 P+ 0.414; Q2 at 10.5+ home
underdogs +0.34 [-0.26, +0.89] P+ 0.868 (n 32), paid for at 0-3 (-0.012, P+
0.047, n 573).

**What the ridge learned** (`coefficients.parquet`, per point, unscaled): Q1's
weight on `home_dog_points` in the 2020-2025 fits runs -0.11, -0.06, +0.07,
+0.11, +0.06, +0.06 -- a sign that flips across training epochs and a size of
under a tenth of a point per point of underdog. Q2 does bend: `home_dog_points`
about -0.12 to -0.24 with `home_dog_hinge_7` +0.35 to +0.60, so a 14-point
home dog gets roughly +2.4 points added back (2024 fit). That bend moved the
10.5+ home-underdog point error from +4.50 to +3.04 on 32 games and cost
accuracy there (56.3% vs 59.4%, picks home 53% vs 31%).

**Reliability (accuracy / stated confidence), incumbent -> Q1 / Q2:** 0-3
56.2% -> 55.3% / 55.4%; 3.5-6.5 56.2% -> 56.5% / 56.7%; 7 52.7% unchanged;
7.5-10 48.5% -> 49.0% / 47.9%; 10.5+ 44.3% -> 42.0% / 42.8%; stated
confidence 55-57% in every bucket for every arm (the mapping is unchanged by
construction). By home side the only cell above P+ 0.9 is Q1 on 3.5-6.5 home
favourites +0.95 [0.00, +2.14] P+ 0.953 (n 317), offset by 0-3 home underdogs
-2.03 [-4.58, 0.00] P+ 0.019 (n 246) and 10.5+ home favourites -3.03 P+ 0.045
(n 99).

**Week 1 2026** (forecast `artifacts/margin_predictions/2026-week-01-20260907T232720Z`,
read from the active manifest; lines identical; baseline reproduced to 8e-16;
`nfl-ats margin-predict` with each profile reproduces the harness to 9e-16):
Q1 changes 0 of 16 sides; Q2 changes one, NE +3.5 at SEA from NE to SEA
(home cover probability 0.4976 -> 0.5009).

**Reading (inferred):** the ridge, shrunk at alpha 10 and trained on nflverse
close lines since 2009, does not find a stable per-point home-underdog
correction; the opener-graded error at 10.5+ (home over-cover on both sides)
is either an opener-vs-close phenomenon the training lines cannot see, a
probability-shape problem rather than a location one (lane K's K1 fixed the
7.5-10 band from the distribution side), or too small per game to survive
shrinkage. Proposed follow-up, not run: a symmetric hinge max(0, |s| - 7)
(the "big spread either side" term |s| itself is collinear with `spread_line`
+ Q1), and a training-side check of the same location error measured against
opener lines where the archive has them.
