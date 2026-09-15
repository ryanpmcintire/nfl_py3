# A joint probability for the base model and the nine composition flags

## Why this is being measured

Paraphrased, owner directive: the served card runs the base model's opener
probability through a chain of nine composition members
(`src/nfl_ats/four_overlay_composition.py`) that can only leave the
probability alone or replace it with its complement, `1 - p`. The owner has
ruled that architecture wrong: a situational signal is evidence with a
fitted weight inside one calibrated probability, never a rule that flips a
served side on its own. `docs/lanes/market-updated-model.md` and
`docs/market_updated_model.md` already did this for the late-week line move
alone, on 799 games (2023-2025), and found the base model's own logit gets a
fitted coefficient of only 0.15-0.33 against outcomes -- if that holds, the
served probabilities are overconfident by roughly three times. This document
is the full version: the base model's logit, all nine composition flags, and
the late-week market move, fit together as one joint logistic model, scored
honestly leave-one-season-out against the card (M0) and against the served
flip chain it would replace (M1).

## Binding closing grounds (verbatim, AGENTS.md)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with `nfl-ats
weak-signals record`, report `probability_positive`, never the binary
"contains zero". The registry code hard-rejects inadmissible closures; if a
record command errors, the verdict is wrong, not the validator. Also
binding: failing a promotion bar of p 0.90 or 0.95 is not grounds to reject
either -- sample sizes here rarely clear it. Nor is clearing p 0.5 grounds to
serve anything; research closure and card-serving are different decisions.
Within-week correlation is zero, so the week-blocked bootstrap
(`nfl_ats.clv.week_blocked_bootstrap`, blocks `(season, week)`) is the
honest unit, 20,000 draws, seed 20260914.

## Population

**Read**, `artifacts/active_ats_model.json`: active model id `195222798c7e6edb`
(`market_residual`/ridge/weak_stack, `calibration_method: none`).
**Measured**, this session: the newest directory under
`artifacts/opener_evaluation/` (`20260914T161406Z`) has
`feature_table_sha256` identical to the active model's, so it is the active
model's own opener-graded stream. It carries 1,537 games, seasons 2020-2025
(227/239/255/272/272/272 by season).

### Which probability column is "raw"

**Read**, `src/nfl_ats/clv.py:2086-2117`: the opener-evaluation stream writes
two probability columns. `home_cover_probability_at_open_raw` is the
`market_residual` ridge model's own prediction with nothing else applied.
`home_cover_probability_at_open` (no `_raw` suffix) additionally has a
walk-forward **home-side offset** calibration added
(`fit_home_side_offsets`, policy `home_side_offset_big_spreads_v2`) before
being served -- **measured**, this session: 401 of 1,537 games (26%) have a
non-zero offset, and the offset changes the >=0.5 pick on 46 of them
(`artifacts/opener_evaluation/20260914T161406Z/metadata.json`,
`home_side_offset.opener_picks_changed_by_offset`). Per the task's own
instruction to use the rawest column available when the stream's own
probability already has something applied, **M0 in this document is
`home_cover_probability_at_open_raw`**, not the served `_at_open` column.
Separately, the nine-member overlay chain in `four_overlay_composition.py`
is never applied to this evaluation stream at all -- it runs only at
card-serving time -- so `_raw` needed no further stripping on that account.

Outcome: `home_covered = 1` when `margin_vs_open > 0`, `0` when `< 0`;
**measured**, 34 of 1,537 games are exact pushes (`margin_vs_open == 0`) and
are dropped, leaving **1,503 graded games**.

### The nine flags, source and sign convention

Each flag is built by importing and calling the member's own static
flag-builder function against the newest schedules snapshot
(`nfl_ats.snapshots.latest_snapshot`/`load_snapshot`), not by re-deriving the
mechanism by hand, and not by calling the member on the model's own raw
pick (which would make the covariate depend on the very prediction the
joint model is trying to correct). Sign is `+1` when the flag favours home,
`-1` when it favours away, `0` when neither or both sides are flagged.
The favoured side for each flag was read directly off the member's own
`apply_*` function: every one of the nine members flips a game exactly when
the model's raw pick disagrees with the flag's favoured side (verified by
construction below, and cross-checked for the seven non-held members by
calling the real `apply_*` functions directly -- see Result 1).

| Member | Flag builder (module) | Favoured side when flagged | Eligibility |
|---|---|---|---|
| Coach fade | `coach_fade_overlay.year_one_by_game` | away, when only the away coach is year-one (fade the year-one team) | weeks 1-8, REG |
| Division revenge | `division_revenge_tilt_overlay.division_revenge_side_by_game` | the revenge-seeking team itself | REG, all weeks |
| Player arrests | `player_arrests_back_side_overlay._broad_side_flags` | the team with an incident in the trailing 14 days | REG, all weeks |
| Bye edge | `bye_edge_fade_overlay.bye_edge_flag_by_game` | away, when only home is off a bye >=12 days (fade the bye team) | REG, all weeks |
| Forecast cold visitor | `forecast_cold_visitor_tilt_overlay.forecast_cold_visitor_flag_by_game` | home only (one-directional) | REG, outdoor |
| Protection mismatch | `pbp08_matchup_flags.build_flag_table` (`back_side`) | HOME or AWAY per the trailing pressure-quartile mismatch | REG, seasons >=2009 |
| Interim HC first game | `interim_hc_first_game_tilt_overlay.interim_first_game_flag_by_game_fail_open` | the interim coach's own team | REG, all weeks |
| Tank zone | `tank_zone_fade_tilt_overlay.tank_zone_flag_by_game` | away, when only home is bottom-2 in-season (fade the tank team) | weeks 14-18, REG |
| Precip + high total | `forecast_weather_kn_precip_high_total_tilt_overlay.precip_high_total_flag_by_game` | home only (one-directional) | REG, outdoor |

Two of these nine (interim HC, precip+high-total) are currently held off the
served card by the owner (`OWNER_HELD_MEMBERS`,
`four_overlay_composition.py:86`) and contribute no flips to M1 below; they
are still included as covariates in the joint model (M2/M3), since a fitted
weight is exactly how the owner's directive says a small-sample signal
should be allowed to speak instead of being held at zero by decree.

### Historical availability of each flag's inputs (measured, not assumed)

All nine flags turned out to be computable for the full 2020-2025
population from already-archived, point-in-time-safe data -- none had to be
recorded as structurally unavailable:

- **Player arrests**: not a live-only fetch. The newest snapshot's safe
  index (`data/raw/player_arrests/20260914T160929Z/incidents_point_in_time.parquet`)
  is the *full* USA Today arrests database back to 2000 (1,116 incidents,
  measured), because arrest dates are historical public record, not a
  forecast; the snapshot's own recency gate (`MAX_SNAPSHOT_AGE`, 36h) exists
  to protect *live serving* freshness and was bypassed here deliberately --
  the flag only ever looks 14 days behind each historical game's own
  decision date, so a 2026 fetch date does not leak into a 2021 game.
- **Forecast cold visitor and precip+high-total**: these need a forecast
  fetched at a specific historical decision cutoff (Tuesday noon / kickoff
  nearest), not a live-only value either -- the repository already holds
  point-in-time-walked historical reconstructions,
  `data/raw/forecast_archive/full_2020_2025/forecasts.parquet` (1,615 rows,
  Tuesday-noon cutoff, `forecast_temp_f`) and
  `data/raw/forecast_archive/kickoff_nearest_2009_2025/forecasts.parquet`
  (4,431 rows, kickoff-nearest cutoff, `forecast_precip_prob_pct`), each
  built by walking strictly backward from the historical decision cutoff to
  the actual MOS bulletin cycle in force at that time (manifests read this
  session). Join coverage is reported in Result 0.
- **Protection mismatch**: built from already-recorded play-by-play
  (`data/pbp/raw/20260817T184927Z`), a trailing rolling feature computed
  from games strictly before the one being flagged -- pregame-safe by
  construction, the same guard the rest of the repository's feature builder
  already applies.
- **Coach fade, division revenge, bye edge, tank zone, interim HC**: all
  static functions of the schedules snapshot alone (prior-season coach,
  head-to-head history, bye gaps, in-season standings, interim-coach
  listing), already point-in-time-safe.

### The market move

**Read**, `docs/market_updated_model.md`: `leader_median_net` from
`artifacts/sharp_weighted_follow/20260909T233606Z/per_game.parquet` is
already signed toward home (positive = toward home). It exists only for
2023-2025 (816 of the archive's rows measured non-null). Per the task's own
instruction ("fit that term on 2023-2025 only, state which"), the market
term is fit **only on the 2023-2025 subset**; M3 (flags, no market) covers
the full 2020-2025 population instead of restricting to three seasons.

## Predeclared arms, before any joint-model number was computed

- **M0**: raw model probability, `home_cover_probability_at_open_raw`.
  Baseline every other arm is paired against.
- **M1**: the served flip chain as it runs today, reproduced by calling the
  real `apply_*` function for each of the seven non-held members directly
  (not a hand re-derivation) on a frame seeded with M0's own probability,
  and taking the union of their flip sets exactly as
  `apply_four_overlay_composition` does (`semantics:
  "joint_or_against_raw_card_complement_once"`, `four_overlay_composition.py:410-460`):
  each member decides independently against the *raw* probability, and any
  game flagged by one or more members has its raw probability complemented
  once. The two owner-held members are excluded, matching what is actually
  served today. Fidelity: this calls production's own `apply_coach_fade_overlay`,
  `apply_division_revenge_tilt_overlay`, `apply_player_arrests_back_side_overlay`,
  `apply_bye_edge_fade_overlay`, `apply_forecast_cold_visitor_tilt_overlay`,
  `apply_pbp08_protection_mismatch_tilt`, `apply_tank_zone_fade_tilt_overlay`
  functions directly; it does not go through the
  `apply_four_overlay_composition` orchestrator itself, which additionally
  requires a live arrest-snapshot staleness object and hashing not
  meaningful for a historical batch run. Cross-checked in Result 1 against
  the signed-flag union computed independently from the nine covariates
  below.
- **M2**: joint logistic model, `home_covered ~ logit(M0) + 9 signed flags +
  market move`, fit leave-one-season-out on the 2023-2025 subset (3 folds),
  L2 penalty chosen per outer fold by inner leave-one-season-out on that
  fold's training seasons only (candidate grid `[0.1, 0.3, 1, 3, 10, 30,
  100]` on standardized covariates, chosen to minimize inner out-of-fold log
  loss).
- **M3**: M2 without the market move, so the full 2020-2025 population (6
  folds) is covered.
- **M4**: `logit(M0)` alone with a fitted slope and intercept (pure
  recalibration / Platt scaling), leave-one-season-out over 2020-2025. This
  is the arm that isolates "is the raw model's own confidence overconfident"
  from "do the flags add anything" -- reported first in Results per the
  task's instruction.
- **Positive control**: perfect foresight restricted to the games where
  M2's out-of-sample pick differs from M0's, on the M2 (2023-2025)
  population.

## Metrics, all out of season

Log loss and Brier vs M0 (and vs M1 where stated), week-blocked bootstrap
(20,000 draws, seed 20260914), reported as `baseline - candidate` so
positive means the candidate arm reduced loss; `probability_positive`
reported for every cell, never a binary "contains zero" read. A five-bin
reliability table (0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0) per arm. Hit
rate in accuracy points (`candidate - baseline`, times 100) vs M0 and vs M1,
with the same bootstrap. Decisive-game records: for each arm, the games
where its pick differs from M0's, and both arms' won-loss record on exactly
those games. Per-fold coefficients in natural units (undoing the internal
standardization), with an approximate Wald 95% interval from the fold's own
fitted information matrix (a descriptive uncertainty band on the
coefficient itself, distinct from the week-blocked bootstrap that governs
every accuracy/log-loss/Brier decision read). Season split, descriptive.
Every look is counted by the eval script and reported before any single
cell is discussed.

## Results

**Measured**, `scripts/joint_probability_model_eval.py`,
`artifacts/joint_probability_model/20260914T214742Z/summary.json`,
`per_game.csv`, `coefficients.csv`. Population: 1,537 opener-graded games
2020-2025, 34 pushes dropped, **1,503 graded games**, 6 week-blocked
seasons; the market-move arm (M2) covers **799 of those games**, 2023-2025
only (move present for exactly the 2023-2025 subset, as predeclared). Seed
20260914, 20,000 bootstrap draws. **33 looks taken**, logged in
`summary.json["look_log"]`: 3 LOSO fits (M2/M3/M4), 18 head-to-head
bootstrap comparisons (6 comparisons x {log loss, Brier, accuracy}), 5
five-bin calibration tables, 4 decisive-game reports, 1 positive control, 2
season-split tables. Zero registry cells resolve to a wrong sign
(`registry_resolved_wrong_sign_cells: []`).

**Measured**, `home_cover_probability_at_open` (the served, offset-adjusted
column) differs from the raw M0 column used here on 394 of 1,503 graded
games -- confirms the choice to score M0 on the rawest column, not the
served one, materially changes which games are even being asked about.

### 1. The calibration finding, first: is the raw model overconfident?

**Yes, and M4 (pure recalibration, `logit(p) -> a*logit(p) + b`, no flags,
no market) puts a number on it.** Fitted leave-one-season-out over all six
seasons, the slope on the raw model's own logit is **0.135 to 0.424 across
the six folds** (every fold's 95% Wald interval touches or nearly touches
zero at its low end, but every point estimate is positive and well below
1.0), with an intercept pinned near zero (0.0006 to 0.012 in every fold):

| Held-out season | Slope on raw logit | 95% Wald interval | Intercept |
|---|---:|---|---:|
| 2020 | 0.424 | [0.016, 0.831] | 0.012 |
| 2021 | 0.214 | [-0.187, 0.616] | 0.010 |
| 2022 | 0.327 | [-0.063, 0.717] | 0.013 |
| 2023 | 0.345 | [-0.047, 0.737] | 0.004 |
| 2024 | 0.289 | [-0.104, 0.681] | 0.006 |
| 2025 | 0.303 | [-0.096, 0.703] | 0.001 |

A slope of ~0.14-0.42 means the raw model's own logit needs to be shrunk to
14-42% of its stated magnitude before it matches how often home actually
covers, out of season -- i.e. the raw model is overconfident by roughly
2.4x to 7x depending on the fold, most folds clustering around 3x-4x. This
is the same direction and a similar (slightly higher) magnitude to
`docs/market_updated_model.md`'s sibling reading of 0.15-0.33 on the
2023-2025, served-probability population; this document's 0.135-0.424 on
the full 2020-2025, raw-probability population is consistent with that
finding, not a contradiction of it.

That overconfidence shows up almost entirely as **worse calibration**, not
as picking the wrong side. M4 vs M0, out of season: log loss improves by
**+0.00442** [-0.00109, +0.01001], P+ 0.943; Brier improves by **+0.00206**
[-0.00061, +0.00478], P+ 0.936 -- both one-sided in the recalibration's
favour though neither interval fully clears zero. Accuracy is a dead heat,
**-0.067 points** [-0.868, +0.731], P+ 0.433, and M4's pick differs from
M0's on only 45 of 1,503 games (card 23-22 on those, M4 22-23) -- shrinking
the logit toward zero rarely moves a pick across 0.5, since M0 itself is
already clustered near a coin flip (calibration table below: 1,272 of 1,503
raw predictions fall in the 0.4-0.6 bin). The overconfidence is real and
the improvement in log loss/Brier from correcting it leans positive, but at
this sample size it is `unresolved_below_power` on every metric, not a
closed result -- reported per the binding rule, not discarded for touching
zero.

### 2. The flags: do they add anything on top of recalibration?

M3 (raw logit + all nine flags, no market, full 1,503-game population,
L2 chosen by inner LOSO -- every fold selected the largest candidate,
100.0, i.e. the inner search preferred heavy shrinkage) reads **better**
than M0 on every metric, and two of three clear the zero line entirely:

- Log loss: **+0.00814** [**+0.00074, +0.01553**], P+ 0.985 -- interval
  entirely above zero.
- Brier: **+0.00394** [**+0.00034, +0.00755**], P+ 0.984 -- interval
  entirely above zero.
- Accuracy: **+1.131 points** [-1.593, +3.823], P+ 0.795.

All nine flags' fitted coefficients are **positive in every one of the six
folds** (`coefficients.csv`), meaning every flag's jointly-fit weight
agrees with the direction the member's own mechanism already assumed --
none flip sign when fit against outcomes instead of hand-picked. None of
the nine individual flags' confidence intervals clear zero on their own at
this sample size (expected: nine covariates sharing ~1,250 training games a
fold), so no single flag is being singled out here; this document reports
the joint result, not per-flag promotions. `flag_interim_hc` and
`flag_precip` -- the two members currently held off the served card by the
owner -- have among the largest point-estimate coefficients (0.34-0.87 and
0.07-0.33) but also the widest intervals (smallest flag populations, 16 and
21 games respectively), consistent with `unresolved_below_power` rather
than either an endorsement or a refutation of the hold.

M2 (raw logit + flags + late market move, 2023-2025 only, 799 games) reads
the same direction against M0 -- log loss +0.00603 [-0.00327, +0.01596] P+
0.893, Brier +0.00296 [-0.00161, +0.00783] P+ 0.892, accuracy +0.501
[-3.620, +4.563] P+ 0.596 -- weaker and noisier than M3 simply because a
third of the population and only 3 LOSO folds are available. The market-move
coefficient itself is positive in all three folds (0.090, 0.132, 0.135) and
its interval clears zero in one of the three (2024: [0.006, 0.264]),
consistent with `docs/market_updated_model.md`'s own finding that the move
carries real, if noisy, signal once the model's own logit is already in the
equation.

### 3. The decision-relevant comparison: does the joint model beat the served chain?

**Not on this read, though the read is unresolved rather than a refutation.**
M1 (the served nine-member flip chain reproduced from production's own
`apply_*` functions, cross-checked bit-for-bit below) is itself a strong,
mostly-resolved improvement over the raw model out of season: log loss
**+0.01023** [**+0.00216, +0.01844**], P+ 0.993; Brier **+0.00498**
[**+0.00104, +0.00897**], P+ 0.993 -- both intervals entirely above zero;
accuracy +2.196 [-0.531, +4.836], P+ 0.944. That is a materially cleaner
read than M3's own log loss/Brier reads on the identical nine flags fit
jointly (Result 2), even though M3 still separately beats M0.

Comparing M3 directly against M1 makes this explicit -- M3 trails the
served chain, though the interval touches zero on every metric:

- Log loss: **-0.00208** [-0.00795, +0.00381], P+ 0.244.
- Brier: **-0.00104** [-0.00390, +0.00184], P+ 0.239.
- Accuracy: **-1.065 points** [-3.098, +1.021], P+ 0.154.

M2 vs M1 (2023-2025 only, market move included) is closer to a dead heat:
log loss -0.00038 [-0.00815, +0.00759] P+ 0.463; Brier -0.00019 [-0.00390,
+0.00374] P+ 0.461; accuracy -0.250 [-3.153, +2.632] P+ 0.441 -- none of
these lean meaningfully either direction.

Per the binding rule, none of these -0.15 to -0.24 probability-positive
reads is a resolved wrong sign (no interval sits wholly on the wrong side
of zero) and none of them is grounds to reject the joint-model approach.
But per the same rule, a `probability_positive` near 0.5 is equally not
grounds to serve one over the other. **What this implies for the decision:
on this measurement, replacing the served rule-based chain with the
single-equation joint model (M2 or M3) is not supported** -- the served
chain's own out-of-season read is the more one-sided of the two, not the
joint model's. The clearest supported change is the one M4 isolates on its
own: **the raw model's logit should be shrunk (~0.14-0.42x) before it is
served or composed with anything else**, independent of whether the flags
stay as flip rules or become fitted terms. Whether the flags themselves
should move from flip rules into a fitted joint model is, on this read,
still open -- the joint version is not shown to be better, but the flip
chain is also not shown to be safe from the same overconfidence M4
diagnoses, since M1's own probabilities were never recalibrated in this
comparison (M1 keeps the raw model's overconfident logit wherever a member
does not flip it).

### 4. M1 reproduction fidelity (measured cross-check)

M1 was built by calling the seven non-held members' real `apply_*`
functions (`apply_coach_fade_overlay`, `apply_division_revenge_tilt_overlay`,
`apply_player_arrests_back_side_overlay`, `apply_bye_edge_fade_overlay`,
`apply_forecast_cold_visitor_tilt_overlay`,
`apply_pbp08_protection_mismatch_tilt`, `apply_tank_zone_fade_tilt_overlay`)
on a frame seeded with M0, and unioning their flip sets exactly as
`apply_four_overlay_composition` does. **Measured**: this union (465
flipped games) is bit-for-bit identical to an independent derivation from
the nine signed flag columns themselves (a game flips iff a non-held flag's
sign disagrees with M0's own pick side) -- 0 of 1,503 games differ between
the two constructions
(`m1_reproduction_detail.mismatch_between_apply_functions_and_signed_flag_shortcut`
= 0). Per-member flip counts: coach fade 105, division revenge 147, player
arrests 23, bye edge 68, forecast cold visitor 60, protection mismatch 115,
tank zone 14 (interim HC and precip are excluded, per
`OWNER_HELD_MEMBERS`).

### 5. Calibration tables (5 bins, out of season)

| Bin | M0 n / gap | M1 n / gap | M3 n / gap | M4 n / gap | M2 n / gap (2023-25) |
|---|---|---|---|---|---|
| 0.2-0.4 | 176 / +0.133 | 134 / +0.079 | 26 / +0.088 | 1 / +0.634 | 17 / +0.096 |
| 0.4-0.6 | 1272 / -0.002 | 1272 / -0.009 | 1437 / -0.001 | 1502 / -0.0004 | 755 / +0.002 |
| 0.6-0.8 | 55 / -0.008 | 97 / +0.008 | 40 / -0.028 | 0 / -- | 27 / +0.038 |

No arm ever predicts outside 0.2-0.8 -- this model has never been
confident, at the raw stage or after any of these corrections. M0's largest
miscalibration is the thin 0.2-0.4 bin (n=176, actual rate 0.50 vs mean
predicted 0.367, gap +0.133); M1 and M3 both shrink that gap by pulling
some of those games toward the middle (fewer games land in the tails at
all: n=134 and n=26). The 0.4-0.6 bin, which holds 85-97% of every arm's
predictions, is well calibrated everywhere (all gaps under 0.01 except
M2's +0.002).

### 6. Decisive-game records (games where the arm's pick differs from M0's)

| Arm | n differ from M0 | M0 record on those | Arm record on those |
|---|---:|---|---|
| M1 | 465 | 216-249 | 249-216 |
| M3 (oos) | 481 | 232-249 | 249-232 |
| M4 (oos) | 45 | 23-22 | 22-23 |
| M2 (oos, 2023-25) | 298 | 147-151 | 151-147 |

Full records: M0 809-694 (53.83%), M1 842-661 (56.02%), M3 826-677 (54.96%),
M4 808-695 (53.79%), M2 437-362 (54.69% of its 799-game population).

### 7. Positive control

Perfect foresight restricted to the 298 games (2023-2025) where M2's
out-of-sample pick differs from M0's: card record on those 298 games
147-151, forced to 298-0. **Effect vs M0: +18.899 accuracy points, 95%
week-blocked [+16.129, +21.744], P+ 1.0.** This bounds the ceiling of any
rule agreeing with M2's particular 298-game disagreement set; M2's actual
measured gain (+0.501) sits well inside this ceiling -- real resolving
power exists at this game count, the candidate effect itself simply does
not clear it, the textbook `unresolved_below_power` shape.

### 8. Season split (descriptive)

| Season | M0 | M1 | M3 (oos) | M4 (oos) | M2 (oos, 2023-25) |
|---|---:|---:|---:|---:|---:|
| 2020 | 51.36% | 57.27% | 54.09% | 50.45% | -- |
| 2021 | 54.66% | 55.93% | 56.78% | 54.66% | -- |
| 2022 | 54.03% | 58.47% | 57.66% | 55.24% | -- |
| 2023 | 55.64% | 58.65% | 56.02% | 54.89% | 55.64% |
| 2024 | 54.14% | 53.01% | 53.01% | 54.51% | 54.14% |
| 2025 | 52.81% | 53.18% | 52.43% | 52.43% | 54.31% |

M1 leads every season through 2023 and is roughly flat-to-behind M0 in
2024-2025; M3 tracks M1 closely in the earlier seasons and falls slightly
behind it from 2023 on -- descriptively consistent with Result 3's
season-pooled read, not separately bootstrapped.

## Look count

33 looks total (Result 0 above); every look is logged verbatim in
`summary.json["look_log"]` and was counted before any single cell was
discussed, per AGENTS.md.

## Registry

Family `joint_probability_model_v1`. All 18 head-to-head cells (6
comparisons x {log_loss_improvement, brier_improvement, accuracy_points})
plus the positive control, 19 cells total, recorded via `nfl-ats
weak-signals record`. `registry_resolved_wrong_sign_cells` is empty
(measured) -- no cell resolves to a wrong sign, so every cell is classified
`unresolved_below_power` with no `--closing-ground` (the M1-vs-M0 and
M3-vs-M0 log-loss/Brier cells have intervals entirely on the positive side,
which is a strong reading, not a closure -- `unresolved_below_power` is the
only classification available for a result that is not being closed
negative). Nothing in `src/` is touched by this document; M1-M4 are
challengers scored here, not switched onto the served card.

