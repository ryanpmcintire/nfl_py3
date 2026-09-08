# Hybrid margin mapping: frozen lane L predeclaration

Family `mod18_hybrid_margin_v1`. Frozen before scoring, 2026-09-07.
Read: active manifest artifacts/active_ats_model.json identifies a4c757efd2525da6,
gaussian_median; forecast margin_predictions/2026-week-01-20260907T164352Z.
Measured: find_matching_opener_evaluation returns opener_evaluation/20260907T152026Z;
the requested 20260907T164523Z metadata is absent. Use the matching run.
Read: ROADMAP.md:761 names played policy overlay_union_coach_division_revenge_player_arrests_v2.

H1 hybrid_key_distance: p = w*p_K1+(1-w)*p_gaussian_median,
w = w0*exp(-d/tau); d = min distance of absolute line to {3,7,10,14}.
Grid w0={0,.25,.5,.75,1}, tau={.5,1,2,4}; minimize mean Brier
on prior non-push games, ties in the listed grid order. Half-point distance=.5.
Use five trailing seasons including prior completed weeks of target season.
Require 200 prior scored games, otherwise w0=0,tau=.5. This is a declared
cold-start fallback, not a tuned finding. K1 is lane K's 2.5-point kernel over
prior OOS predicted margins, counting actual integer margin atoms unchanged.
Training K1 probabilities must themselves be out of time, not refit in sample.
K1 warmup begins after one prior season in the cached lane K center stream.

H2 hybrid_key_distance_by_size: same grid independently within small |line|<=3,
mid 3<|line|<=7, large |line|>7. Extra parameters allowed only when EACH cell
has >=200 prior non-push games spanning >=3 seasons. Otherwise use H1 for
ALL cells. This support rule is fixed regularization, not a significance bar.
No accuracy-based threshold flip or retrospective weight selection.

Weights and K1 history exclude the whole target week, target ids, and dates
not completed before the first game of the target week (date plus one day).
Use archived incumbent probabilities on overlapping archive rows, and the
reproduced gaussian_median probabilities outside it. The cached center stream
is reused with its digest recorded; archive points are line+residual_at_open.
K1 conditioning center includes prior residual median, as in lane K.
D reports both actual POINT error and the median-adjusted conditioning error.

Measurement: paired opener accuracy on all 1537 archive games (1503 non-push),
week-block bootstrap 20000 draws seed 20260817, within-week correlation ZERO,
no added variance inflation. Frozen-pick within-week outcome permutation null
2000 draws same seed, including maximum over both arms. Brier and log loss
use non-push rows and half-push decision credit, positive loss improvement
means incumbent loss minus hybrid loss. Reliability: buckets 0-3,3.5-6.5,7,
7.5-10,10.5+, with 7.5 in the upper bucket; favourite/underdog PICK sides.
D: signed actual home margin minus point forecast, also favourite-oriented
error sign(line)*error, by bucket and incumbent favourite/underdog pick;
add home-favourite/home-underdog strata to expose cancellation. Calibration
gap is realized home cover minus probability, plus pick accuracy minus confidence.
Intervals use the same bootstrap, and calibration excludes pushes.
Run overlay-composition on both scratch evaluations, compare through the
unchanged three-member union; no subset search. Week 1 read-only scratch
margin-predict incumbent plus both hybrid mappings of that output/history.

Declare and assign rotation before scoring. Score frozen arms on successive
2020-21,2022-23,2024-25 windows, recording each predecessor before assignment.
Record all accuracy arms/windows/full archive, losses, composed comparisons,
reliability and diagnosis cells with distinct units and pool-player summaries.
Rank by composed-card point delta, ties H1 then H2; no promotion in this lane.
Mined-archive discount: inherited active-model, H/K spread diagnosis and lattice
selection plus overlay subset selection; two correlated hybrids and shared
windows. Descriptive reused evidence, not independent confirmation; no invented
numerical discount. All unsupported closures remain unresolved_below_power.

Integration scope: additive central methods accept explicitly prior histories;
margin.py is outside the permitted files, so research script owns historical
state and scratch candidate outputs. Do not claim production CLI integration.

## Closing-grounds taxonomy (verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or close an experiment. At this
evaluator's ~2-point resolution, "contains zero" is the EXPECTED outcome for a real small signal. Only two
grounds ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (2) bounded by a positive control proven able to
detect an effect that size. Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the binary "contains zero". The
registry code hard-rejects inadmissible closures; if a record command errors, the verdict is wrong, not
the validator. Never use 95%, 0.90, or any threshold as a DECISION bar; decide on expected value
(`probability_positive` above 0.5 favours playing it), thresholds only govern what docs may CLAIM.
Grade at the OPENER (the pool's grade); a close-graded number may never veto a play. Within-week game
correlation is ZERO by owner mandate: never estimate or pad it. Never say something "needs N more
games": the data is fixed and the project is model-limited.



## Measured results and diagnosis (added after the frozen declaration)

# Lane L: hybrid margin mapping

**Inferred ? decision:** retain the incumbent over these two frozen hybrids on the opener and through the played card. The measurements below favour it; neither arm closes the discrete-margin mechanism. No active model, weekly policy, or card was changed by lane L.

**Measured ? `C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3\b05b3398-3142-4412-ab83-4ff679c1ad88\scratchpad\laneL/results.json` and `composition.json`:** paired results against the active model's own opener picks; 1,537 archive games, 1,503 non-push, 107 week blocks. Intervals are 95% week-block bootstrap, 20,000 draws, seed 20260817, within-week correlation ZERO. Incumbent standalone accuracy is 53.9587%; its three-member card is 55.2229%.

| arm   | opener delta points / interval / probability_positive   |    n | accuracy   | played-card delta points / interval / probability_positive   | played accuracy   |
|:------|:--------------------------------------------------------|-----:|:-----------|:-------------------------------------------------------------|:------------------|
| H1    | -0.5988 [-2.0639, +0.8609]; probability_positive 0.1956 | 1503 | 53.360%    | -0.3327 [-1.6502, +0.9365]; probability_positive 0.2927      | 54.890%           |
| H2    | -0.7984 [-2.6210, +1.0014]; probability_positive 0.1803 | 1503 | 53.160%    | -0.1996 [-1.7139, +1.3141]; probability_positive 0.3809      | 55.023%           |

**Measured ? `laneL/losses.json`:** Brier/log-loss improvements below are incumbent minus hybrid, so positive favours the hybrid; all use the same 1,503 non-push games. H1 has tiny positive loss improvements despite fewer correct forced picks. The higher composed-card point estimate among these two arms belongs to H2; neither exceeds the incumbent.

| arm   | metric   |   incumbent |   hybrid | improvement / interval / probability_positive           |
|:------|:---------|------------:|---------:|:--------------------------------------------------------|
| H1    | brier    |    0.251740 | 0.251689 | +0.0001 [-0.0013, +0.0014]; probability_positive 0.5359 |
| H1    | log_loss |    0.696941 | 0.696770 | +0.0002 [-0.0026, +0.0029]; probability_positive 0.5543 |
| H2    | brier    |    0.251740 | 0.251996 | -0.0003 [-0.0019, +0.0014]; probability_positive 0.3869 |
| H2    | log_loss |    0.696941 | 0.697378 | -0.0004 [-0.0039, +0.0030]; probability_positive 0.4066 |

**Measured ? `laneL/reliability.parquet`:** opener accuracy, intervals, and mean stated pick confidence in percent; 7.5 is assigned to the upper band. Full favourite/underdog cells and the alternative 7.5 allocation remain in the artifact.

| arm       | bucket   |   n |   accuracy |   lower |   upper |   confidence |
|:----------|:---------|----:|-----------:|--------:|--------:|-------------:|
| incumbent | 0-3      | 552 |      56.16 |   51.80 |   60.41 |        55.40 |
| incumbent | 10.5+    | 131 |      44.27 |   36.57 |   52.00 |        55.46 |
| incumbent | 3.5-6.5  | 552 |      56.16 |   51.22 |   61.15 |        55.82 |
| incumbent | 7        |  74 |      52.70 |   41.56 |   63.51 |        56.58 |
| incumbent | 7.5-10   | 194 |      48.45 |   41.08 |   55.49 |        56.25 |
| H1        | 0-3      | 552 |      55.43 |   51.00 |   59.85 |        54.99 |
| H1        | 10.5+    | 131 |      43.51 |   35.77 |   51.59 |        55.81 |
| H1        | 3.5-6.5  | 552 |      54.17 |   49.56 |   58.82 |        55.37 |
| H1        | 7        |  74 |      54.05 |   43.48 |   64.79 |        57.06 |
| H1        | 7.5-10   | 194 |      51.55 |   44.90 |   58.13 |        56.21 |
| H2        | 0-3      | 552 |      55.43 |   51.09 |   59.72 |        54.96 |
| H2        | 10.5+    | 131 |      44.27 |   36.36 |   52.59 |        55.88 |
| H2        | 3.5-6.5  | 552 |      53.80 |   49.38 |   58.20 |        55.34 |
| H2        | 7        |  74 |      54.05 |   43.48 |   64.86 |        57.21 |
| H2        | 7.5-10   | 194 |      50.52 |   43.92 |   57.14 |        56.31 |

**Measured ? `laneL/week1_cli_check.json`:** read-only scratch `margin-predict --probability-method gaussian_median` reproduced all 16 current probabilities exactly (maximum difference 0). **Measured ? `laneL/week1_diff.parquet`:** both hybrids change three model sides: JAX to CLE at 7.5, NYG to DAL at 2.5, DET to NO at 7. These compare the model sides in the forecast, not a republished overlay card; the played card already has DAL. Candidate probability tables are in `laneL/artifacts/margin_predictions/H1` and `H2`. They are generated by the scratch incumbent CLI followed by the prior-only hybrid mapper, not by an integrated candidate production CLI.

| game_id         |   spread_line |   home_cover_probability |     p_H1 |     p_H2 | changed_H1   | changed_H2   |
|:----------------|--------------:|-------------------------:|---------:|---------:|:-------------|:-------------|
| 2026_01_NE_SEA  |      3.500000 |                 0.497618 | 0.494027 | 0.496121 | False        | False        |
| 2026_01_SF_LA   |      3.500000 |                 0.444186 | 0.436725 | 0.441076 | False        | False        |
| 2026_01_ARI_LAC |     10.500000 |                 0.333599 | 0.321442 | 0.321442 | False        | False        |
| 2026_01_ATL_PIT |      3.500000 |                 0.440741 | 0.432877 | 0.437463 | False        | False        |
| 2026_01_BAL_IND |     -3.500000 |                 0.445068 | 0.473103 | 0.456755 | False        | False        |
| 2026_01_BUF_HOU |     -1.500000 |                 0.541818 | 0.533869 | 0.540666 | False        | False        |
| 2026_01_CHI_CAR |     -2.500000 |                 0.516354 | 0.500187 | 0.502875 | False        | False        |
| 2026_01_CLE_JAX |      7.500000 |                 0.510735 | 0.489260 | 0.489260 | True         | True         |
| 2026_01_DAL_NYG |     -2.500000 |                 0.501887 | 0.485521 | 0.488242 | True         | True         |
| 2026_01_GB_MIN  |      1.500000 |                 0.540859 | 0.552645 | 0.542566 | False        | False        |
| 2026_01_MIA_LV  |      3.500000 |                 0.439710 | 0.431722 | 0.436380 | False        | False        |
| 2026_01_NO_DET  |      7.000000 |                 0.521132 | 0.499897 | 0.499897 | True         | True         |
| 2026_01_NYJ_TEN |      1.500000 |                 0.484287 | 0.493003 | 0.485549 | False        | False        |
| 2026_01_TB_CIN  |      3.500000 |                 0.508925 | 0.505612 | 0.507544 | False        | False        |
| 2026_01_WAS_PHI |      5.500000 |                 0.367097 | 0.363995 | 0.366872 | False        | False        |
| 2026_01_DEN_KC  |      3.000000 |                 0.517801 | 0.525337 | 0.532873 | False        | False        |

## Diagnosis D: where the point forecast errs

**Inferred ? plain answer:** the 7.5?10 weakness is principally a distribution/calibration problem in this measurement, not evidence of a general point forecast that overstates big favourites' winning margins. It is not possible to assign all causality to one stage from group means: asymmetric score distributions, home/away composition, and the choice of pick also matter. The larger 10.5+ band has a point-location error as well.

**Measured ? `laneL/diagnosis.parquet`:** on 7.5?10, actual minus POINT predicted home margin is +1.081 points [-1.013, +3.204], probability_positive .8392 (195 games including pushes). Orienting the same error toward the favourite gives +0.281 [-1.550, +2.058], probability_positive .6191; negative would mean favourites were overrated. Within the 111 incumbent favourite picks the favourite-oriented error is -0.472 [-2.664, +1.584], probability_positive .3255. These numbers do not establish a broad favourite-overprediction mechanism.

**Measured ? `laneL/diagnosis.parquet`:** within that band, the smooth mapping overstates its selected-side success by 7.798 percentage points (accuracy minus confidence -7.798 [-15.273, -.655], probability_positive .01565), while K1's gap is -4.120 [-11.126, +3.091], probability_positive .13495. A sharper localization: among 69 HOME UNDERDOG games, realized home cover minus smooth probability is +14.548 points [+3.006, +26.286], probability_positive .99315; K1 gives -.380 [-11.934, +11.387], probability_positive .48285. Among 125 non-push HOME FAVOURITE games, the smooth home-cover gap is -.0955 points [-9.813, +9.238], probability_positive .49115. The home/away distinction explains information hidden by a single spread-band average.

**Measured ? `laneL/diagnosis.parquet`:** at 10.5+, actual home margin exceeds the point forecast by +2.907 points [+0.832, +5.006], probability_positive .99695 (134 games); the median-adjusted conditioning center error is +3.531 [+1.427, +5.647], probability_positive .99955. **Inferred:** improve home-oriented point location there as well as probability shape; a universal big-favourite fade does not follow from these measurements.

**Measured ? `laneL/diagnosis.parquet`:** the following point errors use the archive's exact point prediction `tue_open_home_spread + residual_at_open`, not K1's median-adjusted conditioning center. Positive nflverse home spread denotes HOME favoured. Favourite/underdog below describes the incumbent's selected team. Pushes remain in point-error means and are excluded from probability calibration.

| bucket   | incumbent pick side   |   n incl pushes | home point error / interval / probability_positive       | favourite-oriented error / interval / probability_positive   |
|:---------|:----------------------|----------------:|:---------------------------------------------------------|:-------------------------------------------------------------|
| 0-3      | all                   |             573 | -0.1022 [-1.1569, +0.9616]; probability_positive 0.4195  | -0.7359 [-1.7955, +0.2922]; probability_positive 0.0793      |
| 0-3      | favourite             |             278 | +0.8906 [-0.6354, +2.4471]; probability_positive 0.8666  | -0.9552 [-2.5069, +0.5671]; probability_positive 0.1078      |
| 0-3      | underdog              |             283 | -0.9995 [-2.5282, +0.4700]; probability_positive 0.0947  | -0.5517 [-1.9433, +0.8982]; probability_positive 0.2177      |
| 0-3      | pickem                |              12 | -1.9407 [-12.0877, +6.3940]; probability_positive 0.3649 | +0.0000 [+0.0000, +0.0000]; probability_positive 0.0000      |
| 3.5-6.5  | all                   |             558 | +0.1063 [-0.8688, +1.1309]; probability_positive 0.5849  | +0.3024 [-0.8000, +1.4095]; probability_positive 0.7057      |
| 3.5-6.5  | favourite             |             301 | +0.9873 [-0.3454, +2.3813]; probability_positive 0.9276  | +0.2948 [-1.2161, +1.7748]; probability_positive 0.6570      |
| 3.5-6.5  | underdog              |             257 | -0.9255 [-2.3173, +0.5069]; probability_positive 0.0989  | +0.3114 [-1.2762, +1.9854]; probability_positive 0.6401      |
| 7        | all                   |              77 | -0.9101 [-4.1040, +2.2467]; probability_positive 0.2864  | +0.4679 [-2.7454, +3.6615]; probability_positive 0.6127      |
| 7        | favourite             |              39 | -1.3468 [-5.6626, +2.8795]; probability_positive 0.2813  | -0.7656 [-5.1007, +3.5111]; probability_positive 0.3808      |
| 7        | underdog              |              38 | -0.4620 [-5.1039, +4.2280]; probability_positive 0.4133  | +1.7339 [-2.8909, +6.3902]; probability_positive 0.7615      |
| 7.5-10   | all                   |             195 | +1.0809 [-1.0126, +3.2044]; probability_positive 0.8392  | +0.2806 [-1.5498, +2.0576]; probability_positive 0.6191      |
| 7.5-10   | favourite             |             111 | -0.0464 [-2.1818, +2.2057]; probability_positive 0.4874  | -0.4719 [-2.6636, +1.5839]; probability_positive 0.3255      |
| 7.5-10   | underdog              |              84 | +2.5706 [-0.7426, +5.8398]; probability_positive 0.9361  | +1.2750 [-1.8568, +4.3766]; probability_positive 0.7854      |
| 10.5+    | all                   |             134 | +2.9069 [+0.8324, +5.0057]; probability_positive 0.9970  | +0.7594 [-1.4081, +2.9593]; probability_positive 0.7552      |
| 10.5+    | favourite             |              69 | +0.7665 [-2.4550, +3.8174]; probability_positive 0.6778  | -2.0709 [-5.1945, +1.1571]; probability_positive 0.1038      |
| 10.5+    | underdog              |              65 | +5.1790 [+2.0962, +8.1974]; probability_positive 0.9992  | +3.7639 [+1.0903, +6.5269]; probability_positive 0.9969      |

**Measured ? `laneL/diagnosis.parquet`:** calibration gaps below are percentage points, realized minus stated. `home cover gap` fixes the home side; `pick confidence gap` uses each mapping's own selected side. Rows are stratified by the INCUMBENT pick side, keeping the population paired even when K1 changes picks.

| bucket   | incumbent pick side   | smooth home cover gap pp / interval / probability_positive   | smooth pick confidence gap pp / interval / probability_positive   | lattice home cover gap pp / interval / probability_positive   | lattice pick confidence gap pp / interval / probability_positive   |
|:---------|:----------------------|:-------------------------------------------------------------|:------------------------------------------------------------------|:--------------------------------------------------------------|:-------------------------------------------------------------------|
| 0-3      | all                   | -0.2663 [-4.4726, +3.9244]; probability_positive 0.4488      | +0.7621 [-3.6809, +5.0554]; probability_positive 0.6391           | -2.0283 [-6.2462, +2.1816]; probability_positive 0.1721       | -0.4940 [-4.6964, +3.6950]; probability_positive 0.4128            |
| 0-3      | favourite             | +2.2584 [-3.7947, +8.2740]; probability_positive 0.7601      | +0.7860 [-5.9493, +7.3544]; probability_positive 0.5952           | +0.2704 [-5.7442, +6.2355]; probability_positive 0.5287       | -0.6977 [-7.4554, +5.8448]; probability_positive 0.4217            |
| 0-3      | underdog              | -2.7905 [-8.1060, +2.4046]; probability_positive 0.1489      | +0.5193 [-5.3229, +6.3017]; probability_positive 0.5740           | -4.3525 [-9.7642, +0.8725]; probability_positive 0.0524       | +0.1866 [-4.8704, +5.3188]; probability_positive 0.5361            |
| 0-3      | pickem                | +0.9848 [-25.2461, +27.3685]; probability_positive 0.5814    | +5.7546 [-20.5180, +31.5481]; probability_positive 0.6164         | -0.2995 [-27.2526, +26.5877]; probability_positive 0.4693     | -11.4443 [-37.3462, +14.5913]; probability_positive 0.1852         |
| 3.5-6.5  | all                   | +0.1648 [-4.0237, +4.5636]; probability_positive 0.5302      | +0.3391 [-4.5875, +5.2918]; probability_positive 0.5599           | -2.4471 [-6.6932, +1.9554]; probability_positive 0.1394       | -5.4200 [-9.8386, -1.1145]; probability_positive 0.0073            |
| 3.5-6.5  | favourite             | +3.1835 [-1.8904, +8.3466]; probability_positive 0.8905      | -0.7653 [-7.3178, +5.8146]; probability_positive 0.4109           | -0.4936 [-5.6753, +4.7880]; probability_positive 0.4315       | -10.0419 [-15.9400, -4.0692]; probability_positive 0.0008          |
| 3.5-6.5  | underdog              | -3.4289 [-9.9323, +3.2995]; probability_positive 0.1558      | +1.6538 [-4.8774, +8.1720]; probability_positive 0.6936           | -4.7727 [-11.2650, +2.0100]; probability_positive 0.0817      | +0.0823 [-6.5011, +6.7238]; probability_positive 0.5111            |
| 7        | all                   | -1.4274 [-12.6865, +9.7166]; probability_positive 0.3946     | -3.8806 [-15.0056, +6.9300]; probability_positive 0.2417          | -3.4259 [-14.5923, +7.5808]; probability_positive 0.2686      | -6.5212 [-16.3892, +3.7406]; probability_positive 0.1046           |
| 7        | favourite             | -4.3214 [-19.3828, +11.0100]; probability_positive 0.2901    | -8.0333 [-22.8779, +7.2024]; probability_positive 0.1542          | -6.4397 [-21.3694, +8.6737]; probability_positive 0.2084      | -8.7262 [-23.7176, +6.7605]; probability_positive 0.1362           |
| 7        | underdog              | +1.7973 [-14.6826, +17.0949]; probability_positive 0.5803    | +0.7467 [-14.7424, +16.7081]; probability_positive 0.5362         | -0.0676 [-16.3373, +14.8075]; probability_positive 0.4924     | -4.0641 [-19.6021, +11.5486]; probability_positive 0.3046          |
| 7.5-10   | all                   | +5.1128 [-2.6272, +12.6461]; probability_positive 0.9034     | -7.7981 [-15.2733, -0.6547]; probability_positive 0.0157          | +3.5338 [-4.2301, +11.2024]; probability_positive 0.8126      | -4.1201 [-11.1258, +3.0909]; probability_positive 0.1349           |
| 7.5-10   | favourite             | +3.9960 [-4.9010, +12.9133]; probability_positive 0.8108     | -11.3902 [-20.9176, -2.0473]; probability_positive 0.0084         | -0.7485 [-9.5315, +8.0156]; probability_positive 0.4401       | +0.3729 [-8.6742, +9.6641]; probability_positive 0.5404            |
| 7.5-10   | underdog              | +6.6064 [-5.6421, +18.3701]; probability_positive 0.8590     | -2.9944 [-14.2076, +8.3900]; probability_positive 0.3016          | +9.2606 [-3.2387, +21.3113]; probability_positive 0.9294      | -10.1289 [-21.3833, +1.2557]; probability_positive 0.0402          |
| 10.5+    | all                   | +9.1994 [+1.3355, +16.7377]; probability_positive 0.9887     | -11.1894 [-19.1251, -3.2553]; probability_positive 0.0032         | +11.2919 [+3.1807, +19.1035]; probability_positive 0.9961     | -12.4528 [-20.5237, -4.5280]; probability_positive 0.0014          |
| 10.5+    | favourite             | -3.1638 [-16.3224, +9.0840]; probability_positive 0.3064     | -7.8472 [-19.2869, +3.5772]; probability_positive 0.0892          | -2.5653 [-15.6138, +9.5138]; probability_positive 0.3404      | -3.2126 [-14.5578, +7.8957]; probability_positive 0.2826           |
| 10.5+    | underdog              | +22.1422 [+10.9311, +32.7572]; probability_positive 1.0000   | -14.6882 [-25.1935, -3.9021]; probability_positive 0.0047         | +25.7986 [+14.3569, +36.5680]; probability_positive 1.0000    | -22.1263 [-32.5423, -11.2170]; probability_positive 0.0001         |

## Learned behaviour, stability and limitations

**Read ? `docs/hybrid_margin_mapping.md:1` and `src/nfl_ats/hybrid_margin.py:1`:** H1 minimizes prior non-push Brier over w0=(0,.25,.5,.75,1), tau=(.5,1,2,4), using w0*exp(-distance/tau). H2 fits the same grid within |line|<=3, 3<|line|<=7, and >7 only when every cell has at least 200 prior non-push games across three seasons; otherwise it uses H1 for all cells. Ties choose the listed grid order. Five trailing seasons, whole-week exclusion and a one-day completion allowance prevent target outcomes entering a fit. No bucket-based pick-flip rule was fitted.

**Measured ? `laneL/stream.parquet`:** H2's support condition is met on every archive and Week 1 target. Mean lattice weights by season follow; these are descriptive, not newly selected settings.

|    season |   hybrid_weight_H1 |   hybrid_weight_H2 |   hybrid_size_supported_H2 |
|----------:|-------------------:|-------------------:|---------------------------:|
| 2020.0000 |             0.3698 |             0.3709 |                     1.0000 |
| 2021.0000 |             0.3818 |             0.4088 |                     1.0000 |
| 2022.0000 |             0.4926 |             0.5654 |                     1.0000 |
| 2023.0000 |             0.4159 |             0.4893 |                     1.0000 |
| 2024.0000 |             0.4337 |             0.4486 |                     1.0000 |
| 2025.0000 |             0.1870 |             0.2994 |                     1.0000 |
| 2026.0000 |             0.4242 |             0.2748 |                     1.0000 |

**Measured ? `laneL/results.json`:** opener accuracy changes by season, in percentage points:

|   season |      H1 |      H2 |
|---------:|--------:|--------:|
|     2020 |  0.4545 |  0.9091 |
|     2021 |  2.9661 |  4.2373 |
|     2022 | -2.4194 | -2.8226 |
|     2023 | -1.8797 | -2.6316 |
|     2024 | -2.6316 | -2.2556 |
|     2025 |  0.3745 | -1.4981 |

**Measured ? `laneL/results.json`:** the frozen-pick within-week outcome-permutation null uses 2,000 draws, seed 20260817. H1's null interval is [-1.66333998669328, 1.2641383898868928], observed percentile 0.3555; H2's is [-1.7298735861610113, 1.4637391882900865], observed percentile 0.2325. Maximum-over-two-arm null interval is [-1.4637391882900865, 1.4637391882900865]. This is descriptive selection context, not a decision threshold.

**Read ? `docs/hybrid_margin_mapping.md:1`:** this is a mined archive with inherited active-model selection, lane H/K spread diagnosis and lattice selection, and overlay subset selection; two correlated hybrids and overlapping cells add reuse. No independent-confirmation claim or invented numerical discount is made. **Inferred:** the small loss improvement for H1 is worth preserving as unresolved evidence, while the composed-card comparison does not favour playing either hybrid.

**Measured ? `laneL/inputs.json`:** reused lane K's prior-only center stream, with source path and SHA256 recorded and feature digest checked against the active manifest. The cached strict-week point stream differs from the archived residual on 15 games, maximum .01728533 points; its Gaussian probabilities differ by at most .00245581 with zero pick disagreements (`laneL/results.json`). The comparator and weight-training probabilities on archive rows use the archive itself; K1 retains the strict-week centers. Earlier component history uses available nflverse lines, whereas the archive uses openers. This line-vintage difference is explicit.

**Read ? `src/nfl_ats/calibration.py:394`:** the additive hybrid methods require explicit prior prediction pairs, including out-of-time component probabilities; they never substitute residual-only history. **Read ? fleet file scope and the allowed-files list:** `margin.py` production state plumbing is outside lane L's permitted edits. **Inferred ? integration limit:** candidate production `margin-predict`, alternative-line sweeps and shared cover/push/loss state still need that plumbing and corresponding tests; registering parser choices is not end-to-end activation readiness. The hybrid retains an explicitly requested smooth component and is not a wholly discrete served distribution. No production promotion is claimed.

## Registry audit and verification

**Measured ? live `registry/weak_signals.json`:** 58 ACTIVE cells beginning `mod18_hybrid_margin_v1_`: eight arm/window/full-archive accuracy comparisons, two composed-card accuracy comparisons, 32 bucket/side accuracy comparisons, 12 season accuracy comparisons, two Brier and two log-loss improvements. Every active cell is `unresolved_below_power` and has a pool-player `plain_summary`. **Measured ? `registry/rotation_registry.json`:** 2020?21, 2022?23 and 2024?25 each recorded with verdict `unresolved`; initial declaration/assignment preceded scoring. Exact CLI argv and results are in `laneL/commands.json`, `live_replay/commands.json` and `season_recording_live/commands.json`. Bulk entries execute the actual `nfl_ats.cli.main` parser and handlers in process, not registry helper functions or JSON edits.

**Measured ? registry correction:** I initially recorded 78 signed point-error diagnostic cells as `ats_points`, then invalidated all 78 via `weak-signals invalidate` after reading the registry's positive-favours-candidate contract (`src/nfl_ats/weak_signals.py:117`). A signed bias is not a paired candidate improvement. Diagnosis D remains valid, read-only data in `diagnosis.parquet`; no signed diagnostic enters the effect pool. The CLI also rejects a `probability_gap` unit. I asked about extending it, then withdrew the need because a new unit would not solve the candidate-comparison contract. All calibration-gap cells are published above and in the artifact, not falsely recorded as accuracy effects. This is an explicit exception to the predeclaration's overly broad recording sentence, not a claim that every D cell is in the registry. These invalidations correct measurement semantics and do not close any mechanism.

**Measured ? final gates:** 94 passed, five deliberate residual-only skips, 26.63 seconds. The skips cover three conditional and two hybrid methods that require paired histories; their own tests ran. `ruff format --check` and `ruff check` are clean on all five touched Python files; `mypy src` is clean across 253 files. An earlier mypy run found five errors in the concurrently edited `coordinator_changes.py`; the final run is clean. The leakage tests verify that future and same-week changes cannot move an earlier hybrid weight or probability; lattice history leakage is also covered by `tests/test_conditional_margin.py`.

**Measured ? commands run (all with locked uv, no sync):**

```powershell
.\.tools\uv.exe run --no-sync python scripts/hybrid_margin_opener_eval.py --out <scratchpad>/laneL --centers <scratchpad>/laneK/centers.parquet --stage build
# Then --stage score, losses, composition, diagnosis, cells, week1, replay.
# Initial rotation declare / assign were executed before build and score.
# NFL_ATS_ARTIFACTS_DIR=<scratchpad>/laneL/artifacts
# NFL_ATS_REGISTRY_DIR=<scratchpad>/laneL/registry for measurements.
.\.tools\uv.exe run --no-sync nfl-ats margin-predict --season 2026 --week 1 --features data/processed/game_features_weak_stack.parquet --feature-profile weak_stack --ridge-alpha 10 --probability-method gaussian_median --no-line-sweep
.\.tools\uv.exe run --no-sync python scripts/cli_contract_snapshot.py tests/fixtures/cli_contract.json --normalize-years
.\.tools\uv.exe run --no-sync ruff format --check src/nfl_ats/hybrid_margin.py src/nfl_ats/calibration.py scripts/hybrid_margin_opener_eval.py tests/test_hybrid_margin.py tests/test_calibration_ecdf_smoothing.py
.\.tools\uv.exe run --no-sync ruff check src/nfl_ats/hybrid_margin.py src/nfl_ats/calibration.py scripts/hybrid_margin_opener_eval.py tests/test_hybrid_margin.py tests/test_calibration_ecdf_smoothing.py
.\.tools\uv.exe run --no-sync mypy src
.\.tools\uv.exe run --no-sync pytest -q -n 2 -p no:cacheprovider --basetemp=C:\Users\Ryan\AppData\Local\Temp\laneL-pytest-final tests/test_hybrid_margin.py tests/test_conditional_margin.py tests/test_calibration.py tests/test_calibration_ecdf_smoothing.py tests/test_gaussian_median.py tests/test_cli_contract.py tests/test_experiment_registry.py tests/test_board_humanised.py
```

**Measured ? artifact identity:** active model `a4c757efd2525da6`, `probability_method gaussian_median`; forecast `artifacts/margin_predictions/2026-week-01-20260907T164352Z`. `find_matching_opener_evaluation(Path('artifacts'), active)` returns `artifacts/opener_evaluation/20260907T152026Z`; the user-specified `20260907T164523Z/metadata.json` does not exist. **Read ? `ROADMAP.md:761`:** played policy is `overlay_union_coach_division_revenge_player_arrests_v2`; all composed comparisons use its three members.

**Read ? fleet scope:** coordinator owns session-wide scheduler, dashboard, handoff and publication actions; lane L made no such writes, no commit/push, no automated wagering, no web.archive.org access. **Measured ? lane L changes:** new `src/nfl_ats/hybrid_margin.py`, `scripts/hybrid_margin_opener_eval.py`, `docs/hybrid_margin_mapping.md`, `tests/test_hybrid_margin.py`; additive changes to calibration, residual-only test skip list and CLI fixture; registry writes only via CLIs. Shared working tree has concurrent unrelated changes; final status is attached below.

## Proposed ROADMAP MOD-18 text (not applied)

**Inferred ? proposed update from these measured results:** Lane L completed predeclared H1/H2 key-distance hybrids, fitting monotone lattice weights on prior non-push Brier only. H1 opener -0.599 pts [-2.064,+0.861], probability_positive .196; H2 -0.798 [-2.621,+1.001], .180 (n=1503). Through the three-member card: H1 -0.333 [-1.650,+0.936], .293; H2 -0.200 [-1.714,+1.314], .381. H1's Brier/log-loss improvements are +.000051/+.000171, probability_positive .536/.554; H2's are -.000256/-.000437. Both change three Week 1 model sides. Keep the incumbent over these frozen hybrids; all comparative cells remain unresolved. Diagnosis localizes the 7.5?10 issue to probability calibration especially HOME UNDERDOG games (+14.55-point realized-minus-smooth gap, K1 -.38), while aggregate favourite-oriented point error is +.28 points, not a measured universal big-favourite overprediction. At 10.5+, home-oriented point error +2.91 points also calls for location work. Preserve the absolute integer lattice; investigate home/away-conditioned probability shape and point-location errors as a NEW predeclared family, not a threshold flip. 58 active comparison cells and three looks recorded; 78 diagnostic mis-recordings invalidated. Production history/state integration remains outside lane L's scope.
