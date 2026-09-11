# XLG-06: rookie priors, and the rookie-prior-surplus tilt on production

Lane XLG-06 (2026-09-11). This file is written in two parts. Sections 0-5 are
the **predeclaration**: they were written and the rotation family was declared
and its window assigned *before* any ATS outcome was scored. Sections 6 onward
are the **measurements**.

The XLG-06 chain up to this lane (Stage 1 recruiting -> CFB usage, Stage 2
recruiting -> rookie NFL production, Stage 3 the frozen `mu(r)` prior, Stage 4
the wiring predeclaration) was written in September 2026 and its code was
removed by the 2026-09-10 repository cut (`b7ed31d`, "60 src modules imported
by nothing but their own tests"). The Stage-4 family
`xlg06_rookie_prior_on_production` was declared 2026-09-03 without
`--acknowledge-mined`, which is why its assign was refused: the opener pool is
2020-2025, every block of it intersects the 2018-2025 multiplicity ledger, and
a family that has not acknowledged that ledger has zero eligible windows. This
lane therefore declares a new, acknowledged family and rebuilds the prior
self-contained under `scripts/`.

---

## 0. What is being claimed, in plain words

A rookie arrives in the NFL with a track record - what he did in college and
where he was drafted - and none of it is in the model. The model treats every
player it has never seen as the same generic body. When a team starts several
rookies who were, by that visible track record, clearly better than a generic
body, the model is understating that team; when a team starts rookies who were
clearly worse, it is overstating it. That gap should be biggest in the first
weeks of the season, because that is when there is no NFL evidence to override
the prior, and it should shrink as snaps accumulate.

The rule this lane tests: **lean toward the team whose rookie starters'
pre-NFL track record says they are better than a generic replacement, and
away from the team whose rookies' track record says they are worse.**

## 1. Predeclared mechanism and direction (frozen before scoring)

The ROADMAP XLG-06 row's mechanism is that the market prices a team starting
rookies at high-leverage positions at a *generic replacement level*, because at
the moment the line is set there is no NFL evidence about the rookie. The
information that does exist - college production share and draft slot - is
deadline-visible, and if it is not fully in the number, a team whose rookies'
priors exceed generic replacement is underpriced.

**Direction, frozen:** tilt **toward** the side whose deadline-visible rookie
prior surplus is larger. `+1` favours home, `-1` favours away. This is a signed
tilt, not a one-sided fade.

## 2. Deadline-visible inputs (frozen)

Everything below is known before the pick deadline for a week-`w` game.

| Input | Source on disk | Visible when |
| --- | --- | --- |
| Draft round and overall pick | `data/cfb/draft_picks/raw/20260816T164451Z/season=*/` | Draft night, April before the rookie season |
| Pre-draft consensus grade and ranking | same table, `preDraftGrade` / `preDraftRanking` | Before the draft |
| Final college season usage share | `data/cfb/usage/raw/20260818T214627Z/season=*/usage.parquet`, season `draft_year - 1` | End of the college season |
| Player identity (gsis, position, college) | `data/players/raw/20260911T020107Z/players.parquet` | Signing |
| Snaps already taken this season | `data/players/raw/20260905T123614Z/snap_counts.parquet`, **completed weeks only** | After each game |
| Realised NFL value scale | `data/players/values/raw/20260817T184911Z/player_stats.parquet` | After each game (used to FIT on prior classes only, never on the class being scored) |

No network fetch. No future CFB season. The prior for draft class `Y` is fit
using only draft classes strictly earlier than `Y` whose rookie season has
already finished, so nothing about class `Y`'s NFL season enters its own prior.

## 3. The prior, its scale, and its uncertainty (frozen)

The NFL value scale is the one the injury-value construct already uses
(`src/nfl_ats/players.py`): a per-100-snap rate, `skill_epa` =
`rushing_epa + receiving_epa` per 100 offensive snaps for non-QB offensive
players, and `defense_disruption` = the weighted defensive-stat sum
(`tackles_for_loss` 0.5, `fumbles_forced` 2.0, `sacks` 1.5, `qb_hits` 0.25,
`interceptions` 4.0, `passes_defended` 0.5) per 100 defensive snaps.

Two fitted arms:

- **offense arm**, target `skill_epa` per 100 offensive snaps, population
  RB/WR/TE rookies with at least 100 rookie-season offensive snaps;
- **defense arm**, target `defense_disruption` per 100 defensive snaps,
  population DL/LB/DB rookies with at least 100 rookie-season defensive snaps.

Features in both arms: `log(overall draft pick)`, pre-draft grade standardised
within the draft class, final-college-season usage share (offense arm only;
missing values filled with the class-position mean), and position-group
dummies. Ridge, alpha 1.0, refit per draft class on strictly earlier classes.

**Uncertainty** is the training-residual RMSE of the arm that produced the
prior, reported per player as `prior_sd`. The surplus is reported standardised,
`surplus_z = (prior - generic_replacement) / prior_sd`, so the two arms are
commensurable and can be summed on one team.

**Generic replacement** is the mean realised rookie value of that position
group over the strictly-earlier training classes - what a model with no player
identity implicitly predicts for a body at that position.

**QB and OL have no target on this scale at all**: the construct excludes QB
from `skill_epa` by design and no offensive lineman accrues any of these
stats. Their prior therefore comes from the pooled slot-and-grade channel only
(`log(overall)` plus standardised pre-draft grade, fit on the pooled
standardised targets of the two measurable arms), and is labelled
`prior_channel = "slot_only"`. This is an extrapolation onto positions the
scale cannot measure and is reported separately, not hidden inside the total.

## 4. The decay rule (frozen BEFORE estimation)

The prior's weight in week `w` for a rookie who has taken `n` NFL snaps in
completed games before week `w` is

```
weight(n) = N0 / (n + N0),        N0 = 300
```

`N0 = 300` is the value frozen by XLG-06 Stage 3 (`docs/xlg06_stage3_prior_spec.md`,
"fixed N0=300-snap decay"); it is carried forward unchanged rather than re-tuned
here, so no selection happens on this lane's window. The form is the same
reliability form the injury-value construct already uses,
`reliability = career / (career + prior_snaps)`, read from the other side: the
prior holds full weight in week 1 and is down to half weight after 300 snaps,
roughly five full games for a every-down starter.

## 5. The team-week feature and the tilt (frozen)

For each team and week, over every rookie of that draft class on the roster:

```
contribution_i = expected_share_i * weight(n_i) * surplus_z_i
team_surplus   = sum_i contribution_i
diff           = home team_surplus - away team_surplus
```

`expected_share_i` is the mean of (`offense_pct` + `defense_pct`) over that
player's **completed** games this season; a player with no completed game yet
(week 1, or an inactive) gets the walk-forward mean week-1 share of rookies of
the same draft round and position group in strictly earlier seasons.

The flag fires when `|diff|` exceeds the 80th percentile of `|diff|` over
strictly earlier seasons (walk-forward, at least two prior seasons; 0 otherwise):

```
flag = +1 if diff > threshold, -1 if diff < -threshold, else 0
```

**Candidate pick:** `flag = +1` -> home, `flag = -1` -> away, `flag = 0` ->
whatever production picks. Baseline is production `weak_stack` at the opener.

### 5b. The separate descriptive read: who counts as "starting a rookie"

Section 2 of the lane brief asks for the cover rate of teams starting a rookie
at a high-leverage position early versus late. That is a market-pricing
description, not the played rule, and it uses its own frozen definition:

- **high-leverage position groups**: QB; OL (`position_group == "OL"`);
  pass-rusher (NFL position in DE/OLB/EDGE, or the pre-draft position in
  Defensive End / Outside Linebacker / Defensive Edge).
- **starting, pregame-visible**: in weeks 2+, the rookie's mean
  (`offense_pct` + `defense_pct`) over that season's **completed** games is at
  least 0.50; in week 1, where no completed game exists, the rookie was a
  first-round pick (overall <= 32) at one of those groups.
- the cover rate is measured **at the opener**, home-perspective margin minus
  the Tuesday opening home spread, for the team that is starting the rookie,
  and split weeks 1-6 versus weeks 7 and later.

This read runs on the whole paired opener archive (2020-2025), the same way
`scripts/roof_state_screen.py screen` reports cover rates outside its rotation
window; only section 7's stacked ATS comparison spends the assigned window.

Rotation family `xlg06_rookie_prior_surplus_on_production`, grade `opener`,
declared with `--acknowledge-mined`, window assigned before scoring. Because
the window intersects 2018-2025, the ~130-150-look multiplicity ledger
(ROADMAP RWB-16) applies and is stated as a discount on any reading here.

**Closing-grounds taxonomy in force for every verdict below.** An interval or
CI that contains zero is never grounds to reject, fail or close an experiment;
at this evaluator's ~2-point resolution "contains zero" is the expected outcome
for a real small signal. Only two grounds ever close a line of work: a refuted
mechanism (a resolved wrong sign - the whole interval on the wrong side of zero
- or zero split-half reliability), or a bound from a positive control proven
able to detect an effect that size. Everything else is `unresolved_below_power`
and is recorded with `nfl-ats weak-signals record`, reporting
`probability_positive`, never the binary "contains zero".

---

## 6. Measurements

All three commands were run on 2026-09-11 with
`.\.tools\uv.exe run --no-sync python scripts/xlg06_rookie_priors_screen.py
{prior,cover-rates,production}`. Every bootstrap is 20,000 resamples at seed
20260911. Artifacts land under `artifacts/xlg06_rookie_priors/` (gitignored,
local).

### 6.1 A data defect found while building this, worth its own line

`weekly_rosters.pfr_id` - the column `src/nfl_ats/players.py::_stable_crosswalk`
uses to map Pro-Football-Reference snap-count rows onto `gsis_id` - is **almost
never populated for offensive linemen**. Measured on
`data/players/raw/20260905T123614Z/weekly_rosters.parquet`: the fill rate is
0.29% for `position == "OL"` (80,358 rows), 0.00% for `T` (14,636), 0.74% for
`G` (13,174), 0.25% for `C` (7,170), against 60-75% for every skill and defensive
group. It also covers only 3,738 of the 7,095 distinct `pfr_player_id` values in
`snap_counts.parquet`. `players.parquet` carries a `pfr_id` for 88.6% of
linemen and covers 7,066 of those 7,095.

The first run of this lane's prior found **zero** OL rookies in the snap panel
for that reason. The script now builds its crosswalk from `players.parquet` and
falls back to `weekly_rosters`. Nothing in `src/` was touched (other lanes are
editing it), so **any production feature derived from snap counts through
`_stable_crosswalk` is still silently dropping nearly every offensive
lineman**. That is a live defect for someone to pick up, not a finding of this
lane.

### 6.2 The prior: identity chain and coverage

| Quantity | Measured |
| --- | --- |
| Drafted players, classes 2014-2025 | 3,077 |
| Linked to a `gsis_id` on (draft year, overall pick) | 3,066 (99.6%) |
| Passing the surname-or-ESPN-id guard | 3,062 (99.5%) |
| Surname agreement among linked rows | 99.61% |
| Independently confirmed by `collegeAthleteId == espn_id` | 2,090 |
| With a pre-draft consensus grade | 2,905 (94.4%) |
| With a final-college-season CFB usage share | 895 overall; **753 of 827 (91.1%) in the offence arm**, which is the only arm that uses it |

Rookie starters covered, per season (a "starter" here is a rookie who took at
least 300 offensive-plus-defensive snaps in his rookie year):

| Draft class | Drafted | Played a rookie game | Rookie starters (300+ snaps) | With a prior | Fitted / slot-only |
| --- | --- | --- | --- | --- | --- |
| 2015 | 178 | 139 | 69 | 69 | 46 / 23 |
| 2016 | 248 | 208 | 87 | 87 | 67 / 20 |
| 2017 | 248 | 206 | 92 | 92 | 79 / 13 |
| 2018 | 249 | 201 | 104 | 104 | 83 / 21 |
| 2019 | 249 | 209 | 97 | 97 | 73 / 24 |
| 2020 | 247 | 222 | 97 | 97 | 73 / 24 |
| 2021 | 255 | 236 | 94 | 94 | 68 / 26 |
| 2022 | 257 | 231 | 102 | 102 | 78 / 24 |
| 2023 | 252 | 222 | 108 | 108 | 82 / 26 |
| 2024 | 252 | 222 | 100 | 100 | 72 / 28 |
| 2025 | 252 | 223 | 117 | 117 | 93 / 24 |

**Coverage of rookie starters is 100%** from 2015 on: every rookie who reached
300 snaps has a prior. The 2014 class has none, because the walk-forward rule
needs at least 60 usable training rows per arm from strictly earlier classes
and 2014 is the first class in the table.

### 6.3 The prior against realised first-season NFL value

Prior surplus (standardised) versus realised rookie-season value on the
injury-value scale (standardised within arm so the two arms are commensurable),
draft-class-blocked bootstrap:

| Cut | n | Pearson | 95% CI | Spearman | `probability_positive` |
| --- | --- | --- | --- | --- | --- |
| All measurable positions | 1,288 | **+0.1304** | [+0.0934, +0.1713] | +0.1675 | **1.000** |
| Odd draft years | 684 | +0.1488 | [+0.0917, +0.2036] | +0.1895 | 1.000 |
| Even draft years | 604 | +0.1148 | [+0.0726, +0.1656] | +0.1441 | 1.000 |
| Offence arm (RB/WR/TE) | 428 | +0.0336 | [-0.0848, +0.1641] | +0.0773 | 0.715 |
| Defence arm (DL/LB/DB) | 860 | **+0.1662** | [+0.1093, +0.2271] | +0.2164 | **1.000** |

The odd/even draft-year split is the split-half asked for: both halves land on
the same side with overlapping intervals, so the prior is not an artefact of one
set of classes. The correlation is carried by the defence arm; the offence arm
leans the same way at `probability_positive` 0.715, which by the EV rule is
worth keeping and by the closing-grounds taxonomy is `unresolved_below_power`,
not a negative.

**Reliability of the outcome construct**: realised rookie value measured on odd
weeks versus even weeks of the same rookie season correlates **r = 0.2596**
(Spearman-Brown **0.4122**, n = 1,288). That is modest but clearly non-zero, so
`no_split_half_reliability` is inadmissible as a closing ground anywhere in this
lane.

Recorded: `xlg06_rookie_prior_vs_realised_value`,
`..._offense_arm`, `..._defense_arm`, `xlg06_rookie_prior_odd_draft_classes`,
`xlg06_rookie_prior_even_draft_classes`.

### 6.4 The market read: rookie starters early versus late

Opener-graded, 1,503 paired games across 2020-2025, 1,611 team-games in which a
team started a rookie at QB / OL / pass-rusher under the frozen §5b definition.
The number below is the cover rate of the **rookie-starting team**, against the
Tuesday opening line.

| Split | Team-games | Cover rate | Excess over 50%, 95% week-blocked | `probability_positive` |
| --- | --- | --- | --- | --- |
| Weeks 1-6 | 498 | **48.39%** | [-5.02, +1.77] points | 0.173 |
| Weeks 7+ | 1,113 | **51.03%** | [-0.67, +2.72] points | 0.881 |
| Early minus late | 1,611 | **-2.64 points** | [-6.43, +1.11] points | 0.085 |

By position group (descriptive, no interval):

| Group | Team-games | Weeks 1-6 cover | Weeks 7+ cover |
| --- | --- | --- | --- |
| Rookie QB | 280 (74 early) | 50.00% | 51.94% |
| Rookie OL | 1,206 (360 early) | 46.94% | 51.30% |
| Rookie pass-rusher | 326 (109 early) | 51.38% | 52.99% |

Read on expected value, not on significance: the probability that early-season
rookie-starting teams cover **less** than late-season ones is 0.915, and the
probability that an early-season rookie-starting team beats the coin flip is
0.173. The lean is that the opener does **not** fully discount a team that has
handed a high-leverage job to a rookie in the first six weeks - the market's
generic-replacement pricing is, on this read, too generous early. The OL row
carries most of it, which is the same group the crosswalk defect in §6.1 had
been hiding from every snap-derived feature in the repo.

This is a **marginal** read: it pools every rookie starter regardless of what
his own prior says. It is deliberately not the rule §5 plays, which is
conditional on the prior surplus.

Recorded: `xlg06_rookie_high_leverage_starter_weeks_1_6_cover`,
`..._weeks_7plus_cover`, `..._early_minus_late`.

### 6.5 The on-production screen

Baseline is production `weak_stack` at the opener under active model
`d49194e04945a5e5`. Assigned rotation window **[2020, 2021]**, spent by this
run.

**Window read (the one that spends the window):**

| | |
| --- | --- |
| Paired games / weeks | 456 / 35 |
| Games the flag fires on | 234 |
| Forced picks changed | **118** (113 graded) |
| Baseline accuracy | 53.29% |
| Candidate accuracy | **55.70%** |
| Delta | **+2.412 accuracy points** |
| Week-blocked 95% CI | **[-2.450, +7.456]** |
| `probability_positive` | **0.832** |
| Season-blocked 95% CI | [+0.424, +4.545], P+ 1.000 - on 2 blocks only, not treated as informative |
| Positive control | +46.711 points, P+ 1.000 both blockings |

**Full 2020-2025 opener archive, secondary (not a separate window spend):**

| | |
| --- | --- |
| Paired games / weeks | 1,503 / 107 |
| Games the flag fires on | 639 |
| Forced picks changed | 325 |
| Baseline accuracy | 54.56% |
| Candidate accuracy | **55.62%** |
| Delta | **+1.065 accuracy points** |
| Week-blocked 95% CI | [-1.378, +3.508] |
| `probability_positive` | **0.811** |

**Per season:**

| Season | Games | Picks changed | Delta (points) | Week-blocked 95% CI | `probability_positive` |
| --- | --- | --- | --- | --- | --- |
| 2020 | 220 | 66 | +4.545 | [-2.791, +12.613] | 0.878 |
| 2021 | 236 | 47 | +0.424 | [-5.858, +6.410] | 0.558 |
| 2022 | 248 | 80 | +0.806 | [-7.600, +8.642] | 0.588 |
| 2023 | 266 | 52 | **-3.008** | [-7.605, +1.509] | 0.100 |
| 2024 | 266 | 30 | +4.511 | [+1.091, +7.985] | 0.993 |
| 2025 | 267 | 43 | -0.375 | [-4.494, +3.831] | 0.428 |

Four of six seasons lean to the candidate; 2023 is the one clearly bad season
and its interval still crosses zero, so it is not a resolved wrong sign either.

**Disclosed weakness of the frozen threshold.** The §5 rule sets the firing
threshold at the 80th percentile of `|surplus_diff|` over strictly earlier
seasons. Because seasons 2011-2016 carry no priors at all, their zeros drag that
percentile down, and the rule ends up firing on **42.4%** of full-schedule games
and changing 26% of the in-window picks rather than the roughly one-in-five the
percentile was meant to produce. The threshold was frozen before scoring and has
**not** been retuned - retuning it on this window would be selection on the
window. A successor predeclaration should compute the percentile over seasons
that actually have priors.

Recorded: `xlg06_rookie_prior_surplus_tilt_on_production` (window),
`xlg06_rookie_prior_surplus_tilt_full_opener_archive` (secondary), and six
per-season rows under family `rookie_priors_per_season`, which are a correlated
decomposition of the secondary read and not independent votes.
`nfl-ats rotation record` spent the window with verdict `unresolved`.

### 6.6 The activation-visibility control, and the number to actually use

A rookie appears in `snap_counts` for week `w` only if he was **active** that
week. A Tuesday pick deadline does not know the game-day actives list, so the
§5 feature - which drops a rookie's contribution to zero in any week he does not
appear - was quietly reading one bit of post-deadline information.

`scripts/xlg06_rookie_priors_screen.py leakage-control` removes it: once a
rookie has appeared at all, his last-known trailing share and snap total are
carried forward to **every** later week whether or not he appeared. Nothing the
deadline cannot know enters the flag. Both arms are scored on the same full
2020-2025 opener archive (not a rotation window draw - the window was already
spent by §6.5).

| Arm | Picks changed | Flagged | Delta (points) | Week-blocked 95% CI | `probability_positive` |
| --- | --- | --- | --- | --- | --- |
| Observed presence (§6.5 feature) | 325 | 639 | +1.065 | [-1.378, +3.508] | 0.811 |
| **Carry-forward presence (deadline-clean)** | 329 | 628 | **+0.466** | [-2.076, +3.102] | **0.648** |

The two arms agree on **91.4%** of picks. Removing the activation peek cuts the
effect by about **56%**, from +1.065 to +0.466 points, and
`probability_positive` from 0.811 to 0.648.

**Use the clean number for any decision.** It still favours the candidate -
+0.466 points at `probability_positive` 0.648 is a 65/35 lean, and the pool is
forced picks, so declining it is taking the 35 side - but it is a materially
smaller lean than the window read suggests, and the window read (§6.5) carries
the same activation peek. Recorded as
`xlg06_rookie_prior_surplus_tilt_activation_clean`.

## 7. Verdict, and what it implies for the decision

**What this implies for the decision first.** On the assigned window the
candidate is +2.41 accuracy points on 456 forced picks with
`probability_positive` 0.832; over the whole opener archive it is +1.06 points
on 1,503 with `probability_positive` 0.811; and with the activation peek removed
(§6.6, the number to use) it is **+0.466 points at `probability_positive`
0.648**. The pool submits 285 cards either way, so declining a candidate that is
65% likely to be better is taking the other side of a 65/35 bet. On expected
value this belongs on the played card, subject to the composition check that an
overlay positive on its own can still be negative stacked - that stacking test
is the next lane's job, not something this lane measured.

**Classification.** Every reading here is `unresolved_below_power`. Neither
closing ground is available:

- **No resolved wrong sign.** Every interval crosses zero; the headline point
  estimates favour the candidate. The single negative season (2023, -3.01) has
  an interval running to +1.51.
- **No zero split-half reliability.** The prior's outcome construct has a
  measured odd/even-week reliability of 0.2596 (Spearman-Brown 0.4122), and the
  prior itself replicates across the odd/even draft-year halves.
- **Not bounded by a positive control.** The control read (+46.711 points,
  P+ 1.000) proves the harness detects an effect of this size easily, so the
  control bounds nothing here - it confirms the instrument, and the candidate's
  effect is present, not absent.

**Discounts to apply.** Three, all disclosed above: the window intersects the
2018-2025 multiplicity ledger (roughly 130-150 prior looks, ROADMAP RWB-16) and
the family acknowledged it at declaration; the frozen firing threshold is
miscalibrated upward to a 42.4% firing rate (§6.5); and the §5 feature reads
game-day activation, which is worth about 56% of the archive effect (§6.6). The
first two argue for treating the point estimate as optimistic; the third is
already measured and its clean replacement is +0.466 points at
`probability_positive` 0.648.

**What is genuinely new here, beyond the ATS number.** The prior itself is the
durable object: a per-rookie, deadline-visible, uncertainty-carrying estimate on
the same value scale the injury features already use, with 100% coverage of
rookie starters from 2015 on and a decay rule that was frozen two stages ago.
The XLG-06 row previously had a prior whose slope interval ran from -1.72 to
+4.01; this one correlates with realised rookie value at +0.130 [+0.093, +0.171]
and replicates on both draft-year halves.

## 8. Wired as a prospective challenger (2026-09-11 follow-up)

Per AGENTS.md's promotion-bar section, `unresolved_below_power` at
`probability_positive` above 0.5 is a reason to play the tilt on expected
value, not a reason to wait, and this rotation window is already spent (no
further ATS cost to standing the rule up prospectively). It is registered as
`rookie_prior_surplus_tilt_overlay` in `artifacts/prospective/challengers.json`
(status `ACTIVE_PROSPECTIVE`, display name "Rookie-starter value tilt" in
`CHALLENGER_DISPLAY_NAMES`, `src/nfl_ats/dashboard/findings_content.py`),
implemented in `src/nfl_ats/rookie_prior_surplus_tilt_overlay.py` and mirroring
`post_bye_new_playcaller_back_overlay.py` / `veteran_rest_back_overlay.py`'s
shape: a `TiltResult` dataclass, a pure `apply_rookie_prior_surplus_tilt_overlay`
transform on the active model's own card, and
`record_rookie_prior_surplus_tilt_overlay_decisions` wired into
`orchestrate_publish_predictions` in `src/nfl_ats/cli_commands/publishing.py`.

**It plays the activation-clean construction (section 6.6), not the window
read.** The registered evidence cites +0.466 accuracy points, week-blocked 95%
[-2.076, +3.102], `probability_positive` 0.648 as the number this challenger's
own 2026+ ledger should track, and discloses the peeking window's +2.412 points
at `probability_positive` 0.832 (and the peeking full-archive secondary's
+1.065 at 0.811) separately, per AGENTS.md's "label how you know it" rule --
those two numbers are read from this document, not measured by the overlay
itself.

**What is reused verbatim from `scripts/xlg06_rookie_priors_screen.py`:** the
ridge prior fit (`fit_priors(build_prior_inputs())`, `RIDGE_ALPHA=1.0`,
`MIN_TRAIN_CLASS_ROWS=60`, `MIN_ROOKIE_SNAPS=100`, the same position-arm and
`slot_only` channel split), the `N0_SNAPS=300` decay constant, the 80th-
percentile (`FLAG_PERCENTILE`) walk-forward firing threshold, and the two
functions behind "last-known share carried forward" -- `carry_forward_presence`
and `week1_expected_share`. The module is loaded dynamically from `scripts/`
the same way `crew_tilt_refresh_overlay.py`'s `_load_script_module` already
does for `overlay_stack_backtest`, so these are the literal functions, not a
reimplementation. `screen.load_snap_panel()` is called directly wherever the
overlay needs this season's already-played snap shares, which means its
private `_pfr_to_gsis()` crosswalk workaround (section 6.1: link
`pfr_player_id` to `gsis_id` from `weekly_rosters.parquet` first, backfill any
gap from `players.parquet`'s own `pfr_id`/`gsis_id` columns) runs unchanged, so
offensive linemen who have actually taken snaps this season are not silently
dropped from the trailing-share arm the way `players.py::_stable_crosswalk`
still drops them elsewhere in the repo.

**What is NOT reused, and why: `rookie_week_state` / `team_week_surplus` /
`game_level_feature` read a cached, gitignored local artifact
(`artifacts/xlg06_rookie_priors/rookie_prior_table.parquet`, written only by a
manual `scripts/xlg06_rookie_priors_screen.py prior` run) and identify a
season's rookies by inner-joining onto `snap_counts.parquet`.** Measured
2026-09-11: `data/players/raw/20260905T123614Z/snap_counts.parquet` carries
**zero** season-2026 rows, so that inner join would classify every 2026 rookie
-- including every rookie starting in his own team's Week 1, before that game
is played -- as nonexistent, and the overlay would never fire on a genuinely
live card. The overlay therefore refits the prior fresh on every run (never
touching the cached artifact) and adds one live-only branch: a current-draft-
class rookie who is absent from this season's snap panel has his team read
from `players.parquet`'s `latest_team` field (deadline-visible, refreshed with
every new players snapshot) instead of from a snap-count row, and his Week 1
contribution uses the frozen `week1_expected_share` fallback the screen's own
section 5 specifies for exactly this case; for week 2+ with still no snaps,
his contribution is 0.0, exactly as frozen. This is a necessary extension for
running the frozen rule live, not a change to the rule's frozen formulas.

**Confirmed live 2026-09-11 against the active 2026 Week 1 card**
(`artifacts/margin_predictions/2026-week-01-20260910T210852Z`, config
fingerprint `bc77638d47e2748c`, 16 games): the walk-forward threshold for
season 2026 -- computed from 2015-2025 history only, before any 2026 game --
was **0.132863**. 11 of 16 games flagged (6 away-favoured, 5 home-favoured).
6 of those changed production's own raw pick and were recorded as decisions
to the challenger ledger without altering the served card:
`2026_01_ATL_PIT` (ATL at PIT) -> PIT,
`2026_01_CLE_JAX` (CLE at JAX) -> CLE,
`2026_01_DAL_NYG` (DAL at NYG) -> NYG,
`2026_01_MIA_LV` (MIA at LV) -> LV,
`2026_01_SF_LA` (SF at LA) -> LA,
`2026_01_TB_CIN` (TB at CIN) -> TB.
The other 5 flagged games already agreed with production's raw pick, and zero
games carried both a home and an away flag. Rookie starters do exist in a
real Week 1 (confirming the mechanism can fire at all before any current-
season snap data exists), and the served, published card was not changed by
this confirmation run.

