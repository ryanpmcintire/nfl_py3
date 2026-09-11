# LEAD-63: the pick as a mixture over "starter finishes" and "starter exits"

Owner's idea, 2026-09-10, after the Week 1 opener: the Seattle starter left in
the first minutes and Seattle won by 3, half a point short of covering. The
pick was right; the question is whether the served cover probability should be
a mixture over two worlds -- the starting quarterback finishes, and the
starting quarterback leaves early -- weighted by a probability computed before
the pick deadline, instead of a single starter-plays-all-game forecast.

Everything above the "Results" heading was written before the window was
assigned and before the screen ran.

## Closing grounds that apply to this lane (verbatim, AGENTS.md)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval on
the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". The registry code
hard-rejects inadmissible closures; if a record command errors, the verdict is
wrong, not the validator.

Verdicts flow only through `nfl-ats weak-signals record` and
`nfl-ats rotation record`, never through prose in this file.

## Step 1 -- the label

`scripts/qb_exit_mixture_screen.py label`, over every season in the newest
play-by-play snapshot. A "dropback" here is a play with `qb_dropback == 1` and
a non-null `passer_player_id`, i.e. pass attempts and sacks; quarterback
scrambles carry no passer id in nflverse (measured: every null-passer dropback
in 2021 has `play_type == 'run'`), so they are excluded from the identification
rather than misattributed.

For each team-game:

- the **starting QB** is the passer on the team's first dropback;
- **starter share** is the starter's dropbacks over the team's dropbacks;
- an **early exit** is flagged when the starter's last dropback falls in
  quarters 1-3, the team drops back again with a different passer, and the
  score is within 14 points at the starter's last dropback. The margin
  condition excludes blowout rests, which are a coaching decision rather than
  an availability event.

Team-games with fewer than 10 dropbacks are ineligible (there is no meaningful
starter share to compute). Two robustness variants are written alongside the
primary flag: `early_exit_strict` additionally requires at least 5 relief
dropbacks, and `early_exit_any_margin` drops the 14-point condition.

Output: `data/processed/qb_starter_exits.parquet`.

## Step 2 -- the pregame exit probability

`scripts/qb_exit_mixture_screen.py features` then `... fit`. Nine columns, all
visible before kickoff:

1. `on_injury_report` -- the starter appears on that week's team injury report;
2. `inj_report_rank` -- Out 4, Doubtful 3, Questionable 2, Probable 1, absent 0;
3. `inj_practice_rank` -- DNP 2, Limited 1, Full/absent 0;
4. `experience_years` -- seasons since the quarterback's first NFL pass. No
   birth date exists anywhere under `data/`, so career experience stands in for
   age; that substitution is a limitation of the snapshot, not a design choice;
5. `log_career_starts_prior`;
6. `prior_missed_starts` -- his primary team's games last season minus his
   starts for them;
7. `has_prior_starts`;
8. `qb_prior_exit_rate` -- his own exit rate over strictly earlier seasons,
   shrunk to the league base rate with a 20-start prior;
9. `opp_sack_rate` -- the opponent defense's sack rate over strictly earlier
   games this season plus last season, shrunk to the league rate with a
   200-dropback prior.

Injury rows are joined on (season, week, team, gsis id). Where nflverse carries
a `date_modified`, any row stamped at or after kickoff is dropped; kickoff is
built from the newest schedule snapshot's `gameday` and `gametime` in Eastern
time.

Fit: plain logistic regression on standardised features, trained only on
seasons strictly before the season being scored, requiring four training
seasons and 20 training exits. Reported: split-half reliability of the flag
(odd versus even seasons, per quarterback and per team, Spearman-Brown
corrected, bootstrap interval) and calibration of the fitted probability
(decile table, Brier, Brier skill against the base rate).

**Disclosed approximation.** The starter's identity comes from the game that
was actually played, not from a Tuesday depth chart. This follows the existing
backup-QB overlay, which reads the realised `home_qb_name`/`away_qb_name` from
the schedule snapshot (`docs/backup_qb_fade_overlay.md`). Every *other* input
is strictly pre-kickoff. A production version would have to take the expected
starter from the depth chart instead.

## Step 3 -- the mixture, and how it is graded

Let `p` be production's served home cover probability at the opener
(`home_cover_probability_at_open` from `nfl_ats.clv.opener_pick_evaluation`,
which is the model's own probability with the served walk-forward home-side
offset applied). Let `h` and `a` be the pregame exit probabilities of the home
and away starters, and let `d` be the cover-probability penalty a team pays for
having a backup at quarterback.

Taking the two exits as independent, and noting that a game in which both
starters leave has no net side:

```
P(home covers) = p
               + d * h * (1 - a)      home's starter leaves, away's does not
               - d * a * (1 - h)      away's starter leaves, home's does not
               = p + d * (h - a)
```

which is exactly the ROADMAP formula
`P(finishes) x P(cover | finishes) + P(exits) x P(cover | exits)` written for
both sides at once. The pick is `P(home covers) >= 0.5`, the same rule
production uses.

`d` is **not** fitted here. It is read from the existing registry entry
`bias_battery_backup_qb_start_opener`, whose
`classification_evidence` records `raw_gap=-2.9095 pts (unscaled)`: opener
graded, 2020-2025, the backup-start side covers 2.9095 percentage points less
often than its complement. So `d = -0.029095`.

**Stated discount for a reused window.** That entry was measured on 2020-2025,
which contains this lane's 2020-2021 window. `d` is a constant imported from an
earlier measurement rather than tuned on this window, but the overlap is real
and the screen's interval should be read with it in mind. AGENTS.md permits a
reused window with a stated discount; this is the statement.

Arms, all graded at the opener on the rotation-assigned window with the
repository's week-blocked block bootstrap (20,000 resamples, seed 20260910):

- **screen** -- the single outcome look: `d = -0.029095`, `h` and `a` from the
  fitted pregame model.
- **snap_share_scaled** -- secondary. A starter who exits is replaced for only
  part of the game, so this arm scales `d` by the measured mean relief dropback
  share on exit games. Strictly smaller than the screen, reported for shape.
- **positive_control_oracle_exit** -- replaces the fitted probability with the
  realised exit flag at the same `d`. This asks whether the instrument could
  see the effect even with perfect foreknowledge of who leaves.
- **positive_control_oracle_exit_large_delta** -- realised exit flag at
  `d = -0.50`, which forces a flip on every one-sided exit game. A deliberately
  leaky sensitivity check on the harness itself.
- **permutation null** -- margins permuted within week, 200 permutations.

Endpoints: forced-pick accuracy points at the opener (primary, predeclared in
ROADMAP LEAD-63) and Brier improvement of the mixture probability against the
opener cover outcome (secondary, recorded because a probability nudge this
small can move the probability in the right direction without moving a pick).
Also reported separately, as ROADMAP LEAD-63 asked: the asymmetric cell, games
where exactly one of the two starters was on that week's injury report and
still started.

**Expectation stated before the signs were seen** (ROADMAP LEAD-63): the base
rate is a few percent and roughly symmetric, so the mixture moves only picks
already near 50%; the value, if any, is in the asymmetric cell.

## Results

Appended after the window was assigned and the arms were run. Nothing above
this heading was edited afterwards.

### Step 1 -- how often a starter leaves early

**Measured** (`.\.tools\uv.exe run --no-sync python scripts/qb_exit_mixture_screen.py label`,
play-by-play snapshot `20260817T184927Z`, written to
`data/processed/qb_starter_exits.parquet`): 9,260 team-games labelled, 8,860 of
them regular-season and eligible, 3 team-games ineligible for fewer than 10
dropbacks.

**The base rate is 3.804% of team-games** -- 337 early exits in 8,860 -- and the
starter takes 95.98% of his team's dropbacks on an average team-game. Two
robustness variants on the same rows: requiring at least 5 relief dropbacks
gives 3.736%, and dropping the 14-point blowout condition gives 5.959%, so
roughly a third of pre-fourth-quarter handovers are blowout rests rather than
availability events.

| Season | Team-games | Exits | Rate | Strict (>=5 relief) | No margin filter |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2009 | 512 | 23 | 4.492% | 4.297% | 8.008% |
| 2010 | 512 | 30 | 5.859% | 5.469% | 7.617% |
| 2011 | 511 | 19 | 3.718% | 3.718% | 5.675% |
| 2012 | 512 | 16 | 3.125% | 3.125% | 4.883% |
| 2013 | 512 | 19 | 3.711% | 3.711% | 5.859% |
| 2014 | 512 | 17 | 3.320% | 3.125% | 6.250% |
| 2015 | 512 | 18 | 3.516% | 3.516% | 5.664% |
| 2016 | 512 | 17 | 3.320% | 3.320% | 5.273% |
| 2017 | 512 | 17 | 3.320% | 3.320% | 5.469% |
| 2018 | 512 | 13 | 2.539% | 2.539% | 4.297% |
| 2019 | 512 | 18 | 3.516% | 3.516% | 5.078% |
| 2020 | 512 | 16 | 3.125% | 3.125% | 5.664% |
| 2021 | 543 | 18 | 3.315% | 3.315% | 5.525% |
| 2022 | 542 | 23 | 4.244% | 4.059% | 5.904% |
| 2023 | 544 | 22 | 4.044% | 4.044% | 6.802% |
| 2024 | 544 | 25 | 4.596% | 4.412% | 6.802% |
| 2025 | 544 | 26 | 4.779% | 4.779% | 6.434% |

Range 2.539% (2018) to 5.859% (2010); no trend, a rise since 2018. The
owner's predeclared expectation that "the base rate is a few percent" holds.

**Measured** (same artifact): when a starter does leave early, his replacement
takes 59.69% of the team's dropbacks on average, and the exit lands in the
second quarter 137 times, the third 119, the first 86.

### Step 2 -- the pregame exit probability

**Measured** (`... features`, injuries snapshot `20260910T203021Z`, schedules
`data/raw/20260908T162105Z/schedules.parquet`, written to
`data/processed/qb_exit_pregame_features.parquet`): 18.49% of starters appear
somewhere on their own team's injury report that week. **Zero** injury rows
were stamped at or after kickoff and dropped; `date_modified` is present on
100% of quarterback rows for 2010-2024 (so on 100% of the 2020-2021 window) and
absent for 2009, 2025 and 2026, where the week-keyed final report is used as
published.

Exit rate by cell, all 8,860 eligible team-games:

| Cell | Team-games | Exits | Rate |
| --- | ---: | ---: | ---: |
| Starter anywhere on the injury report | 1,638 | 76 | 4.640% |
| Starter not on the injury report | 7,222 | 261 | 3.614% |
| Game status Questionable | 199 | 12 | 6.030% |
| Game status Probable | 527 | 20 | 3.795% |
| Did not practice | 27 | 3 | 11.111% |
| Rookie season | 1,667 | -- | 5.639% |
| 1-2 years experience | 2,488 | -- | 4.220% |
| 3-5 years | 2,290 | -- | 3.188% |
| 6-10 years | 1,909 | -- | 2.724% |
| 11+ years | 506 | -- | 2.569% |

Experience is the cleanest gradient in the table -- monotone from 5.64% for
rookies to 2.57% for eleven-year veterans. Opponent sack rate is the weakest:
by quartile the exit rate runs 3.93%, 4.38%, 2.89%, 4.02%, which is no
ordering at all.

#### Split-half reliability of the exit flag

**Measured** (same command, odd versus even seasons, Spearman-Brown corrected,
4,000-draw bootstrap over units, seed 20260910):

| Unit | Units | Pearson r | 95% | Spearman-Brown | `probability_positive` |
| --- | ---: | ---: | --- | ---: | ---: |
| Starting QB, >=8 starts per half | 93 | 0.3217 | [0.1109, 0.5276] | 0.4868 | 0.997 |
| Starting QB, >=16 starts per half | 71 | 0.2170 | [-0.0647, 0.5318] | 0.3567 | 0.921 |
| Team, >=40 team-games per half | 32 | 0.5236 | [0.2897, 0.7087] | 0.6873 | 1.000 |

The flag carries a stable carrier at both the quarterback and the team level.
`NO_SPLIT_HALF_RELIABILITY_MAX` in `src/nfl_ats/weak_signals.py` is 0.10;
every reading here is several times that, so **`no_split_half_reliability` is
not an available closing ground for this lane** and nothing downstream may use
it.

#### Calibration of the fitted probability

**Measured** (`... fit`, chronological, trained only on strictly earlier
seasons, first scored season 2013): 6,813 out-of-sample team-games,
2013-2025. Mean predicted 3.459% against an observed 3.655%; Brier 0.035137
against the base-rate Brier 0.035212, a Brier skill score of **+0.0021**; AUC
**0.6426**.

On the 2020-2021 window itself: 1,055 team-games, 34 exits (3.223%), mean
predicted 3.321%, Brier skill **-0.0054**, AUC **0.5757**. The fitted
probability spans 1.32% to 13.79% there, with the median at 2.57%.

**Inferred:** the model is close to unbiased in the mean and ranks better than
a coin flip over the full out-of-sample span, but on this particular two-season
window its ranking is close to flat. I think the honest reading is that the
pregame signal is real but thin, and that most of its content is the
experience gradient plus the Questionable flag, not the opponent sack rate.

### Step 3 -- the mixture against production at the opener

**Measured** (`nfl-ats rotation declare` then `rotation assign`): the assigned
opener confirmation window is **2020-2021**, the mined-window acknowledgment
recorded, drawn fresh -- this family inherits nothing and the block was the
earliest eligible one, the same block LEAD-62 drew.

**Measured** (`.\.tools\uv.exe run --no-sync python scripts/qb_exit_mixture_screen.py score`,
artifact `artifacts/experiments/qb_exit_mixture/results.json`, paired rows in
`paired_predictions.csv`): the production arm is
`nfl_ats.clv.opener_pick_evaluation` re-run on the rotation split, active model
`d49194e04945a5e5`, profile `weak_stack`, `gaussian_median`, ridge alpha 10,
500 minimum training games.

**Verified before anything was graded**: the re-run reproduces the saved
production card `artifacts/opener_evaluation/20260910T211255Z/per_game.parquet`
on the window to 5.6e-17 on `home_cover_probability_at_open`, and that saved
card scores 53.2895% on 456 non-push 2020-2021 opener games -- the same
production baseline LEAD-62 reported.

The mixture population is one game smaller: **465 paired, 455 non-push, 35
weeks** (220 in 2020, 235 in 2021). The missing game is `2021_13_NE_BUF`, the
December wind game in Buffalo, where New England recorded 3 dropbacks and so
falls under the 10-dropback eligibility floor from step 1. Production's
accuracy on the 455 that remain is 53.1868%.

#### The primary screen

| Arm | Picks changed | Candidate | Production | Accuracy points | Week-blocked 95% | `probability_positive` |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| **screen** (d = -0.029095, fitted exit probability) | **3** | 52.9670% | 53.1868% | **-0.21978** | [-0.90703, +0.44944] | **0.29240** |
| snap-share-scaled (d = -0.017367) | 2 | 53.1868% | 53.1868% | 0.00000 | [-0.66225, +0.66372] | 0.50605 |
| control: oracle exit, same d | 2 | 53.1868% | 53.1868% | 0.00000 | [-0.65359, +0.66667] | 0.49970 |
| control: oracle exit, d = -0.50 | 11 | 53.4066% | 53.1868% | +0.21978 | [-1.29032, +1.56951] | 0.63103 |

Season-blocked for the screen: [-0.45455, 0.00000], `probability_positive`
0.12363. Per season, week-blocked: 2020 n=220, **-0.45455** points, [-1.43541,
0.00000], `probability_positive` 0.17883; 2021 n=235, **0.00000** points,
[-1.25000, +1.31579], `probability_positive` 0.50203.

**Measured** (permutation null, margins permuted within week, 200
permutations): null mean -0.03077 points, standard deviation 0.38292, central
95% [-0.65934, +0.65934]. The observed -0.21978 sits at the 15th percentile of
that null, i.e. comfortably inside it.

#### Why only three picks moved

**Measured** (`paired_predictions.csv`): the mixture's mean absolute shift to
production's cover probability is **0.000554**, and its largest shift anywhere
on the window is **0.003546**. Production's opener probabilities on this window
have a standard deviation of 0.0683 around a mean of 0.4467, so a shift that
small can only move a pick that was already a coin flip: all three flips had a
production probability inside 0.0008 of 0.500, and two of the three turned a
winner into a loser.

This is arithmetic, not bad luck. The shift is `d * (h - a)`; `d` is 0.0291 and
the two starters' fitted exit probabilities differ by 0.019 on average, so the
typical nudge is six ten-thousandths. The band in which any flip is even
possible at this `d` -- games with a production probability within 0.029095 of
0.500 -- holds **105 of the 465 games**, so the mechanism is not structurally
locked out of the card; the fitted exit probabilities simply never differ
enough to reach it.

#### The asymmetric cell

ROADMAP LEAD-63 predeclared this as the place the value would be if anywhere:
games where exactly one of the two starters was on that week's injury report
and still started.

**Measured**: **129 of the 455 graded games** are in that cell. Production
scores 55.8140% there, better than its 53.1868% over the whole window. The
mixture **changed zero picks** in the cell, so its effect there is exactly
**0.00000 accuracy points**, `probability_positive` 0.50000 -- a dead heat by
construction, not a measured null. Under the oracle-exit control at the same
`d` the cell moved 2 picks and still scored 0.00000 points.

#### What the controls say

**Measured**: with the realised exit flag substituted for the fitted
probability at the same `d`, the mixture's **Brier improves by +0.00033**,
week-blocked [-0.00027, +0.00094], `probability_positive` **0.86500**
(season-blocked `probability_positive` 1.00000). **Inferred:** I think that is
the mechanism's direction showing up cleanly -- when you actually know which
starter left, moving the cover probability toward the other side is the right
move; it is just far too small a move to change which side gets picked.

**Measured**: at `d = -0.50`, which forces a flip on essentially every
one-sided exit game, the oracle control changes 11 picks and scores **+0.21978
accuracy points**, week-blocked [-1.29032, +1.56951], `probability_positive`
0.63103. **This control is itself unresolved.** It does not establish that the
instrument can detect a candidate-sized effect and it therefore does **not**
supply a `positive_control_bound` closing ground; per AGENTS.md a positive
control has to be *proven able* to detect an effect that size, and an
11-flip oracle arm on 455 games is not.

#### Classification

Every graded arm is recorded `unresolved_below_power`. The three closing
grounds were checked and none is available:

- **`wrong_sign_resolved`** -- the screen's week-blocked interval is
  [-0.90703, +0.44944], which does not sit entirely below zero, and the
  season-blocked interval [-0.45455, 0.00000] has zero on its boundary. Not a
  resolved wrong sign.
- **`no_split_half_reliability`** -- the exit flag's split-half reliability is
  0.4868 at the quarterback level and 0.6873 at the team level, both far above
  the 0.10 floor. Unavailable.
- **`positive_control_bound`** -- the only control that moved the card is
  itself unresolved at `probability_positive` 0.63103, as set out above.
  Unavailable.

#### What this lane concluded, and what it did not

**Inferred, stated as a decision rather than a criticism** (AGENTS.md: state
what a result implies for the decision before stating what is wrong with it):
at the size `d` currently carries, this mixture is not a card change -- it
moves three picks in two seasons and `probability_positive` is 0.29240, so
playing it is taking the short side of a roughly 71/29 bet on a change that
barely exists. The right next move is not to play the mixture as specified.

**What is NOT concluded.** The mechanism is not refuted. The exit flag is
reliable, its base rate is a stable 3-5% a season with a clean experience
gradient, and under oracle knowledge the probability moves the right way at
`probability_positive` 0.86500 on Brier. What is bounded here is one specific
`d`, borrowed from a full-game backup-start cell, applied to a partial-game
event. Three things this measurement points at, none of them settled:

1. `d` is the whole story. A cover-probability penalty of 0.029 cannot move a
   card whose probabilities sit 0.067 wide around 0.447. Measuring a penalty
   specific to *in-game* handovers -- the backup's actual snap share, the
   relief quarterback's own quality, whether the exit is an injury or a
   benching -- is a different measurement than the one this lane borrowed.
2. The exit probability is thin where it matters. AUC 0.6426 over 2013-2025
   collapses to 0.5757 on 2020-2021, and the opponent sack rate contributes no
   ordering at all. Most of the pregame content is experience plus the
   Questionable flag.
3. The mixture as written moves the pick side only. Its natural home is the
   served cover probability and the projected margin -- the discrete key-number
   lattice conditional on the line -- where a small probability shift is
   readable even when it flips nothing. That is where the oracle control's
   Brier gain showed up.

**Stated discount, repeated:** `d` was read from a cell measured on 2020-2025,
which contains this window. It was imported as a constant and not tuned here,
but the overlap is real.

### Files, and what was recorded where

Created or changed by this lane:

- `scripts/qb_exit_mixture_screen.py` -- subcommands `label`, `features`,
  `fit`, `score`.
- `data/processed/qb_starter_exits.parquet` -- 9,260 labelled team-games.
- `data/processed/qb_exit_pregame_features.parquet` -- 8,860 rows, nine
  pregame columns plus the fitted `p_exit`.
- `artifacts/experiments/qb_exit_mixture/results.json` and
  `paired_predictions.csv`.
- `docs/qb_exit_mixture.md` (this file) and ROADMAP.md's LEAD-63 row.

Recorded through the CLIs, never through this prose. Seven weak signals under
family `qb_exit_mixture`, all `unresolved_below_power`, league `nfl`, seasons
2020-2021, source `docs/qb_exit_mixture.md`:

| Signal | Units | Effect | Interval | `probability_positive` |
| --- | --- | ---: | --- | ---: |
| `qb_exit_mixture_on_production_opener` | accuracy_points | -0.21978 | [-0.90703, +0.44944] | 0.29240 |
| `qb_exit_mixture_on_production_opener_asymmetric_cell` | accuracy_points | 0.00000 | [0, 0] | 0.50000 |
| `qb_exit_mixture_on_production_opener_snap_share_scaled` | accuracy_points | 0.00000 | [-0.66225, +0.66372] | 0.50605 |
| `qb_exit_mixture_on_production_opener_brier` | brier_improvement | -0.0000384 | [-0.000104, +0.0000325] | 0.13590 |
| `qb_exit_mixture_positive_control_oracle_exit` | accuracy_points | 0.00000 | [-0.65359, +0.66667] | 0.49970 |
| `qb_exit_mixture_positive_control_oracle_exit_brier` | brier_improvement | +0.000335 | [-0.000266, +0.000941] | 0.86500 |
| `qb_exit_mixture_positive_control_oracle_exit_large_delta` | accuracy_points | +0.21978 | [-1.29032, +1.56951] | 0.63103 |

The three control rows carry category `control`; the four candidate rows carry
`health`. The registry warns that these seven overlap on the same 2020-2021
seasons and are correlated decompositions of the same football -- they are, and
they must not be pooled or sign-counted as independent votes.

Rotation: `nfl-ats rotation record --name qb_exit_mixture_on_production
--verdict unresolved --probability-positive 0.2924`, window [2020, 2021] spent
2026-09-11.

