# When should the pick be refreshed? (MKT-08)

Lane MKT-08, 2026-09-11. The served late-week follow rule is held **fixed**;
only the **instants at which it is evaluated** vary. Script:
`scripts/refresh_timing_policy.py`. Artifact:
`artifacts/refresh_timing_policy/20260911T000000Z/` (`results.json`,
`per_game.parquet`, `decay_curve.csv`, `capture_cadence.json`).

## Binding closing-grounds taxonomy (AGENTS.md), verbatim

An interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (1) refuted mechanism — a RESOLVED wrong sign (whole interval on the wrong
side of zero) or zero split-half reliability; (2) bounded by a positive control
proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". The pool is FORCED
PICKS, so the decision is expected value; a promotion bar governs what this
document may claim, never which card is played. Within-week game correlation is
ZERO by owner mandate; the bootstrap is week-blocked and no ICC term is estimated.

## The answer in three lines

1. **Refresh each game once more, as late as the pool allows.** Evaluating the
   same rule at each game's own deadline instead of at the last of the twelve
   scheduled passes is **+0.1996 accuracy points, week-blocked 95%
   [+0.0000, +0.4620], `probability_positive` 0.9756** over 2020-2025, and it
   costs **3.6 refreshes a week instead of 12**. It changed only 3 picks in six
   seasons and all 3 went the candidate's way. It should be played.
2. **Refreshing at all is an era question, not a settled one.** Against the
   Tuesday card the fixed-pass refresh reads **-3.7281 points
   [-6.9264, -0.6508], P+ 0.0101** on 2020-2021 and **+2.2514 points
   [-0.5758, +5.2142], P+ 0.9336** on 2024-2025. On the picks the rule actually
   changes it goes 36.9% right in 2020-2021, 45.2% in 2022-2023 and 58.8% in
   2024-2025. Nothing here is closed.
3. **News-triggered refreshing is not cheaper in the modern archive and is not
   better.** At a 0.5-point trigger it needs 9.9 runs a week in 2024-2025 (vs 12
   fixed) and reads **-1.3133 points against the fixed schedule
   [-2.8143, +0.0000], P+ 0.0357** there.

## What is held fixed

The only timing-sensitive step of the served refresh chain that is replayable on
a historical archive is the **late-week leader-median follow rule**
(`nfl_ats.sharp_book_movement_features.late_week_follow_frame`, threshold from
`leader_follow_threshold`: 1.0 points, or 0.5 when the frozen line is 10.5 or
larger). Every policy below evaluates *that* function; only its `cutoff_utc`
changes.

- **Baseline card.** The served Tuesday card: the active model's opener pick
  (`home_cover_probability_at_open` >= 0.5, offset applied) complemented once by
  the nine-member composition union (`served_card_flip_set(card="served")`).
- **Grading.** Opener-graded forced picks at `margin_vs_open`, from
  `artifacts/opener_evaluation/20260910T211255Z/per_game.parquet` (weak_stack /
  ridge / gaussian_median, `home_side_offset_big_spreads_v2` served). 1,537
  archive games, 34 opener pushes dropped, **1,503 scored**.
- **Deadline.** `nfl_ats.nfl_week.pool_decision_cutoff`, i.e. min(own kickoff,
  that week's Sunday 16:00 ET). No policy reads a quote observed at or after a
  game's own deadline.
- **Uncertainty.** `nfl_ats.overlay_composition.blocked_bootstrap_matrix`,
  week-blocked on `(season, week)`, 20,000 samples, seed 20260911, paired on the
  same games. `probability_positive` uses the repo's shared
  `P(>0) + 0.5*P(==0)` convention.

### Steps deliberately held out, and why

The other served steps are **not timing-sensitive**, so they would contribute an
identical constant to every policy's paired delta: the rookie-crew refit (the
officials archive carries no crew-publication time to replay), the heavy-handle
follow (the Wayback split backfill covers 133 of 816 games and only on weekend
passes), and the small-spread Best Pick nominator. The 1.0-point consensus
movement rule was retired from the played card on 2026-09-10
(`docs/late_week_refresh.md`) and is not replayed. The model's own recompute on
fresh injury/feature data cannot be replayed at all: point-in-time feature tables
for 2020-2025 mid-weeks do not exist. **So this lane measures the timing of the
market-follow half of the refresh, not of the whole refresh.**

## Capture cadence, measured per season

[measured: `artifacts/refresh_timing_policy/20260911T000000Z/capture_cadence.json`,
NFL snapshots only (`request.sport == americanfootball_nfl`), regular and post
season weeks]

| Season | Snapshots | Median per week | Median gap | Wed | Fri | Sun captures 11:00-16:00 ET | Latest Sunday clock before 16:00 ET |
|---|---|---|---|---|---|---|---|
| 2020 | 147 | 7 | 34.3 h | 0 | 0 | 21 | 12:25 |
| 2021 | 154 | 7 | 34.3 h | 0 | 0 | 22 | 12:25 |
| 2022 | 154 | 7 | 34.4 h | 0 | 0 | 22 | 12:25 |
| 2023 | 2,475 | 136 | 60 min | 432 | 432 | 22 | 12:25 |
| 2024 | 2,476 | 136 | 60 min | 432 | 432 | 22 | 12:25 |
| 2025 | 2,476 | 136 | 60 min | 432 | 432 | 22 | 12:25 |

Read this carefully, because it bounds everything below.

- **2020-2022 is a seven-capture week** at fixed decision labels: `true_open`
  Mon 08:55 ET, `tue_open` Tue 08:55, `thu_pre_tnf` Thu 17:55, `sat_midday`
  Sat 11:55, `sun_early_close` Sun 12:25, `sun_late_close` Sun 16:15,
  `mon_pre_mnf` Mon 18:55. There is **no Wednesday and no Friday capture at all**
  in those seasons. The follow rule counts movement from Wednesday onward, so on
  those seasons it has exactly **two** informative readings a week (Thursday
  evening and Saturday midday), both measured against the Tuesday-morning base.
- **2023-2025 adds the `intraday_hourly` backfill**: hourly from Tuesday 00:00 ET
  through **Sunday ~10:55 ET**, then nothing until the single `sun_early_close`
  snapshot at 12:25 ET. [measured: the manifest index, hour-of-day distribution
  per weekday.] In every season, including the hourly ones, there is exactly
  **one** capture per week between 11:00 and 16:00 ET on Sunday.
- Consequence, stated before any result: the archive **cannot** distinguish a
  Sunday 10:00 pass from a Sunday 15:00 pass for any game except through that one
  12:25 snapshot. Every "later is better" reading below is a **lower bound** on
  what real Sunday-afternoon movement could add.
- Second consequence: the frozen follow rule ignores every quote observed at or
  after Sunday 00:00 ET (`observed_at_utc < _sunday` in
  `sharp_book_movement_features`). Three of the twelve served passes
  (Sun 10:00, 11:55, 15:00) therefore **cannot move a follow-driven pick at all**.
  An arm with that blackout removed is reported alongside.

## The four policies, predeclared

Fixed in `scripts/refresh_timing_policy.py` before the first run; the only later
edit added head-to-head and changed-pick reporting, which re-read the same picks
rather than adding an arm.

| Policy | Refresh instants |
|---|---|
| **(d) `tuesday_only`** | None. The Tuesday card. This is the baseline. |
| **(a) `fixed_passes`** | The twelve `refresh-picks` jobs now in `SCHEDULE`: Wed 18:15, Wed 19:15, Thu 11:55, Thu 15:00, Thu 15:25, Thu 19:15, Sat 10:30, Sat 15:50, Sat 19:15, Sun 10:00, Sun 11:55, Sun 15:00 ET. A game is decided by the last pass strictly before its own deadline. |
| **(b) `as_late_as_allowed`** | One pass per game, at its own deadline — the last capture the pool allows. |
| **(c) `news_triggered_0_5` / `news_triggered_1_0`** | Walk every capture instant from Wednesday 00:00 ET to the deadline; refresh whenever the **leader-book median line** has moved 0.5 (resp. 1.0) or more since the last refresh's line, starting from the frozen Tuesday line. The last trigger before the deadline decides the pick; no trigger means the Tuesday pick stands. Both thresholds predeclared. |
| **(b') `as_late_as_allowed_incl_sunday`** | (b), with the rule's Sunday blackout removed. A faithful replica, verified to reproduce the frozen function exactly when the blackout is restored (`replica_max_gap_vs_frozen_rule` = 0.0). |

## Results

Every cell is the policy's opener-graded forced-pick accuracy minus the Tuesday
card's, on the same games, week-blocked.

### Assigned confirmation window, 2020-2021 (456 scored games, 35 week blocks)

| Policy | Picks changed | Accuracy | Delta | 95% week-blocked | P+ | Refreshes/week |
|---|---|---|---|---|---|---|
| `tuesday_only` | 0 | 57.237% | — | — | — | 0 |
| `fixed_passes` | 65 | 53.509% | -3.7281 | [-6.9264, -0.6508] | 0.0101 | 12 |
| `as_late_as_allowed` | 65 | 53.509% | -3.7281 | [-6.9264, -0.6508] | 0.0101 | 3.2 |
| `as_late_as_allowed_incl_sunday` | 91 | 53.070% | -4.1667 | [-8.9286, +0.6682] | 0.0465 | 3.2 |
| `news_triggered_0_5` | 56 | 54.167% | -3.0702 | [-5.4945, -0.8547] | 0.0044 | 2.8 |
| `news_triggered_1_0` | 24 | 55.482% | -1.7544 | [-3.4934, +0.0000] | 0.0214 | 2.8 |

### 2022-2023 (514 scored games, 36 week blocks)

| Policy | Picks changed | Accuracy | Delta | 95% week-blocked | P+ | Refreshes/week |
|---|---|---|---|---|---|---|
| `tuesday_only` | 0 | 59.144% | — | — | — | 0 |
| `fixed_passes` | 73 | 57.782% | -1.3619 | [-4.1584, +1.5595] | 0.1757 | 12 |
| `as_late_as_allowed` | 73 | 58.171% | -0.9728 | [-3.6810, +1.9157] | 0.2483 | 3.8 |
| `as_late_as_allowed_incl_sunday` | 85 | 59.339% | +0.1946 | [-2.9703, +3.4026] | 0.5450 | 3.8 |
| `news_triggered_0_5` | 53 | 58.560% | -0.5837 | [-3.2505, +2.2000] | 0.3387 | 6.4 |
| `news_triggered_1_0` | 26 | 58.755% | -0.3891 | [-1.9569, +1.1928] | 0.3184 | 5.2 |

### 2024-2025 (533 scored games, 36 week blocks)

| Policy | Picks changed | Accuracy | Delta | 95% week-blocked | P+ | Refreshes/week |
|---|---|---|---|---|---|---|
| `tuesday_only` | 0 | 54.409% | — | — | — | 0 |
| `fixed_passes` | 68 | 56.660% | +2.2514 | [-0.5758, +5.2142] | 0.9336 | 12 |
| `as_late_as_allowed` | 69 | 56.848% | +2.4390 | [-0.5629, +5.4945] | 0.9428 | 3.9 |
| `as_late_as_allowed_incl_sunday` | 83 | 56.848% | +2.4390 | [+0.0000, +5.0373] | 0.9764 | 3.9 |
| `news_triggered_0_5` | 55 | 55.347% | +0.9381 | [-1.8762, +3.8889] | 0.7352 | 9.9 |
| `news_triggered_1_0` | 36 | 54.784% | +0.3752 | [-1.7308, +2.7728] | 0.6207 | 6.3 |

### Pooled 2020-2025 (1,503 scored games, 107 week blocks)

| Policy | Picks changed | Accuracy | Delta | 95% week-blocked | P+ | Refreshes/week |
|---|---|---|---|---|---|---|
| `tuesday_only` | 0 | 56.886% | — | — | — | 0 |
| `fixed_passes` | 206 | 56.088% | -0.7984 | [-2.5555, +0.9927] | 0.1917 | 12 |
| `as_late_as_allowed` | 207 | 56.287% | -0.5988 | [-2.3697, +1.1992] | 0.2556 | 3.6 |
| `as_late_as_allowed_incl_sunday` | 259 | 56.554% | -0.3327 | [-2.4275, +1.7659] | 0.3793 | 3.6 |
| `news_triggered_0_5` | 164 | 56.088% | -0.7984 | [-2.4096, +0.8542] | 0.1648 | 6.4 |
| `news_triggered_1_0` | 86 | 56.354% | -0.5323 | [-1.6151, +0.5988] | 0.1731 | 4.8 |

### Per season

Accuracy / delta vs Tuesday / picks changed.

| Season | n | `tuesday_only` | `fixed_passes` | `as_late_as_allowed` | `incl_sunday` | `trigger_0_5` | `trigger_1_0` |
|---|---|---|---|---|---|---|---|
| 2020 | 220 | 59.09% | 53.64 / -5.45 / 26 | 53.64 / -5.45 / 26 | 50.91 / -8.18 / 40 | 55.45 / -3.64 / 24 | 59.55 / +0.45 / 9 |
| 2021 | 236 | 55.51% | 53.39 / -2.12 / 39 | 53.39 / -2.12 / 39 | 55.08 / -0.42 / 51 | 52.97 / -2.54 / 32 | 51.69 / -3.81 / 15 |
| 2022 | 248 | 58.87% | 56.45 / -2.42 / 34 | 56.85 / -2.02 / 33 | 57.66 / -1.21 / 41 | 58.47 / -0.40 / 19 | 58.06 / -0.81 / 8 |
| 2023 | 266 | 59.40% | 59.02 / -0.38 / 39 | 59.40 / +0.00 / 40 | 60.90 / +1.50 / 44 | 58.65 / -0.75 / 34 | 59.40 / +0.00 / 18 |
| 2024 | 266 | 54.14% | 56.02 / +1.88 / 39 | 56.39 / +2.26 / 40 | 58.27 / +4.14 / 41 | 54.89 / +0.75 / 30 | 53.38 / -0.75 / 20 |
| 2025 | 267 | 54.68% | 57.30 / +2.62 / 29 | 57.30 / +2.62 / 29 | 55.43 / +0.75 / 42 | 55.81 / +1.12 / 25 | 56.18 / +1.50 / 16 |

## Head to head against the schedule that is actually served

Each row is the candidate minus `fixed_passes` on the same games. Positive
favours the candidate.

| Window | `as_late_as_allowed` | `incl_sunday` | `news_triggered_0_5` | `news_triggered_1_0` |
|---|---|---|---|---|
| 2020-2021 | +0.0000 [+0.0000, +0.0000] P+ 0.5000, 0 picks differ | -0.4386 [-4.2463, +3.5011] P+ 0.4101 | +0.6579 [-1.3101, +2.6032] P+ 0.7488 | +1.9737 [-1.0799, +5.0218] P+ 0.9032 |
| 2022-2023 | +0.3891 [+0.0000, +0.9728] P+ 0.9381, 2 picks differ | +1.5564 [-0.5906, +3.5573] P+ 0.9243 | +0.7782 [-1.2001, +2.9644] P+ 0.7566 | +0.9728 [-1.3725, +3.5156] P+ 0.7759 |
| 2024-2025 | +0.1876 [+0.0000, +0.5682] P+ 0.8175, 1 pick differs | +0.1876 [-1.8382, +2.2181] P+ 0.5752 | -1.3133 [-2.8143, +0.0000] P+ 0.0357 | -1.8762 [-3.7807, +0.0000] P+ 0.0270 |
| **2020-2025** | **+0.1996 [+0.0000, +0.4620] P+ 0.9756, 3 picks differ** | +0.4657 [-1.0695, +2.0107] P+ 0.7218 | +0.0000 [-1.0774, +1.1148] P+ 0.4965 | +0.2661 [-1.1394, +1.7497] P+ 0.6381 |

The `as_late_as_allowed` row is the clean one. Because the scheduled passes are a
subset of "every capture before the deadline", the late pass sees strictly more
market and only three games in six seasons came out differently — 2022 Week 16
NYG at MIN, 2023 Week 4 DET at GB, 2024 Week 17 BAL at HOU — and the late pass
was right on all three. It is a small, cheap, almost-never-negative gain.

## Where the schedule is actually late

[measured: `results.json` -> `served_pass_gap_to_deadline`] Hours between a
game's last scheduled pass and its own deadline, across all 1,537 archive games:
median **1.08 h**, mean 1.37 h, max 25.0 h. The schedule is already tight for the
ordinary slots and wide open for four of them:

| Kickoff slot | Games | Gap to deadline |
|---|---|---|
| Fri 15:00-20:15 ET | 6 | 19.75-25.0 h |
| Wed 13:00 / 16:30 ET (Christmas) | 2 | 13.0-16.5 h — **no scheduled pass reaches these games at all**; the two Wednesday passes are at 18:15 and 19:15 ET, after both kickoffs |
| Sun 09:30 ET (London) | 21 | 14.25 h — last look is Sat 19:15 ET |
| Sat 13:00-14:00 ET | 13 | 2.5-3.5 h |
| Everything else (Thu, Sun 13:00 and later, Mon) | 1,495 | 1.00-1.17 h |

## The decay curve

The same rule evaluated with its cutoff set H hours before each game's own
deadline. Delta is against the Tuesday card; week-blocked.
[`decay_curve.csv`; served-rule arm.]

| H (hours before deadline) | 2020-2021 | 2022-2023 | 2024-2025 | 2020-2025 | 2020-2025 picks changed |
|---|---|---|---|---|---|
| 0 | -3.73 (P+ 0.010) | -0.97 (0.248) | **+2.44 (0.943)** | -0.60 (0.256) | 207 |
| 3 | -3.51 (0.012) | -0.78 (0.290) | +2.25 (0.928) | -0.53 (0.281) | 196 |
| 6 | -3.07 (0.019) | -0.97 (0.251) | +1.88 (0.894) | -0.60 (0.252) | 193 |
| 12 | -3.07 (0.019) | -0.58 (0.342) | +2.06 (0.920) | -0.40 (0.328) | 190 |
| 18 | -3.07 (0.019) | -0.58 (0.342) | +1.69 (0.860) | -0.53 (0.277) | 188 |
| 24 | -3.07 (0.019) | +0.19 (0.556) | +0.38 (0.608) | -0.73 (0.196) | 177 |
| 36 | -1.75 (0.114) | -0.19 (0.445) | +0.94 (0.749) | -0.27 (0.370) | 168 |
| 48 | -1.54 (0.144) | +0.39 (0.657) | +1.13 (0.806) | +0.07 (0.538) | 155 |
| 60 | -1.54 (0.144) | -0.39 (0.351) | +0.75 (0.732) | -0.33 (0.320) | 145 |
| 72 | 0.00 (0.500) | -0.19 (0.428) | +0.38 (0.627) | +0.07 (0.558) | 55 |
| 96 | 0.00 (0.500) | 0.00 (0.498) | +0.75 (0.844) | +0.27 (0.803) | 22 |
| 120 | 0.00 (0.500) | 0.00 (0.500) | 0.00 (0.500) | 0.00 (0.500) | 0 |

**Later is not monotonically better, and it is not flat either — it has a sign
that depends on the era.** In 2024-2025 the curve rises steadily toward the
deadline: +0.38 at 24 hours out, +1.69 at 18, +2.06 at 12, **+2.44 at the
deadline**, with `probability_positive` climbing 0.61 -> 0.86 -> 0.92 -> 0.94.
There is no plateau inside the last 24 hours; the last twelve hours are where the
gain is. In 2020-2021 the same curve runs the other way: 0.00 at 72 hours out
(nothing has moved yet), -1.54 at 48, -3.07 from 24 down to 6, **-3.73 at the
deadline**. Pooled, the two eras cancel and the curve is flat within noise.

The `incl_sunday` arm is where the archive's ceiling shows. On 2024-2025 it reads
**+3.00 points [+0.1848, +5.9259], P+ 0.9789 at H = 3** and +2.44
[+0.0000, +5.0373], P+ 0.9764 at H = 0 — better than the blackout arm at the same
instants. But the only Sunday reading it can add is the single 12:25 ET snapshot,
so this is a floor on what removing the blackout would be worth, not a measurement
of it.

## Mechanism

On the picks each policy actually changes, how often is the new side right?

| Window | `fixed_passes` | `as_late_as_allowed` | `incl_sunday` | `trigger_0_5` | `trigger_1_0` |
|---|---|---|---|---|---|
| 2020-2021 | 36.9% (65) | 36.9% (65) | 39.6% (91) | 37.5% (56) | 33.3% (24) |
| 2022-2023 | 45.2% (73) | 46.6% (73) | 50.6% (85) | 47.2% (53) | 46.2% (26) |
| 2024-2025 | 58.8% (68) | 59.4% (69) | 57.8% (83) | 54.5% (55) | 52.8% (36) |
| 2020-2025 | 47.1% (206) | 47.8% (207) | 49.0% (259) | 46.3% (164) | 45.3% (86) |

The era trend is monotone across all five policies and it is large: the same
rule, on the same kind of games, goes from losing two picks in three to winning
three in five. Lane AJ's independent 2023-2025 replay of the same rule scored its
changed picks 38-26 (59.4%) [read: `docs/late_week_refresh.md`], which matches the
2024-2025 column here.

**The trait itself is highly reliable, so `no_split_half_reliability` is
inadmissible as a closing ground.** [measured: split-half by leader book —
`bovada` against `williamhill_us` + `mybookieag`, per-game net move at each game's
deadline] Pearson between halves is 0.899 (2020-2021), 0.901 (2022-2023), 0.851
(2024-2025), 0.885 pooled; Spearman-Brown corrected, 0.947 / 0.948 / 0.920 /
0.939 on 1,434 games with both halves present. The books agree about what moved.
What changed between eras is what that agreed move was worth against this card.

**No positive control was run for this channel**, so no cell can be classified
`bounded_by_control`. No cell has a whole interval on the wrong side of zero
across the family — 2020-2021's `fixed_passes` cell does, but the identical rule
reads +2.25 on 2024-2025, so this is a magnitude that changed sign between eras,
not a refuted mechanism, and the owner's era directive governs: weaker-era
readings are never absence. Every cell is recorded `unresolved_below_power` with
a null closing ground. Nothing is closed.

## The decision

Per AGENTS.md's expected-value rule, and stating the decision before the caveats:

1. **Add a per-game last-call refresh and play it.** `as_late_as_allowed` beats
   the twelve served passes at P+ 0.9756 pooled, is never behind in any window
   (0.5000 / 0.9381 / 0.8175), and needs a third of the runs. Declining a change
   that is ~98% likely to help is taking the 2/98 side of the bet.

   **The scheduler change it needs** (not made in this lane; the served schedule
   is unchanged here):
   - a **Sunday ~08:30 ET** pass so the 09:30 ET London games stop being decided
     at Saturday 19:15, 14.25 hours early;
   - a **Wednesday ~11:30 ET** pass, because a Wednesday-afternoon game
     (Christmas) currently has **no** `refresh-picks` job before its kickoff —
     both Wednesday jobs fire at 18:15 and 19:15 ET;
   - a **Friday pass roughly two hours before a Friday kickoff**, where the last
     look is now Thursday 19:15 (19.75-25.0 hours early);
   - a **Saturday ~11:00 ET** pass for Saturday 13:00 kickoffs (now 2.5 hours
     early).

   Those four holes cover 42 of the 1,537 archive games. For every other slot the
   schedule is already within ~1 hour of the deadline and nothing needs to change.

2. **Do not replace the fixed schedule with a news trigger.** Neither threshold
   beats the fixed passes pooled (P+ 0.4965 at 0.5, 0.6381 at 1.0) and both are
   behind in the most recent window (P+ 0.0357 and 0.0270). The cost saving is
   also not what it looks like: against an hourly feed a 0.5-point trigger fires
   9.9 times a week in 2024-2025, barely below the twelve fixed passes. The
   1.0-point trigger at 4.8 runs a week is the only version that is genuinely
   cheaper, and it is 64% likely better than the fixed schedule pooled — worth
   keeping as a fallback if daemon load ever matters, not worth swapping in now.

3. **The Sunday blackout inside the follow rule is worth re-opening as a
   challenger.** Removing it reads +0.4657 pooled (P+ 0.7218) and +1.5564 on
   2022-2023 (P+ 0.9243), on an archive that can only offer it one extra snapshot
   per week. Prospectively the live captures run several times each Sunday
   morning, so its real reach is larger than anything measured here. This lane
   does not change the rule.

4. **The honest tension.** Pooled over 2020-2025 the whole refresh channel is
   0.81 likely to be *costing* accuracy against simply leaving Tuesday's card
   alone. That is not a reason to stop refreshing — the era trend is monotone
   across five independent policies, the trait is reliable at 0.94, and the two
   most recent seasons favour refreshing at P+ 0.934 — but it is the number the
   owner should see next to any claim that the late-week channel is established.

## Registry

Rotation family `refresh_timing_policy` (grade `opener`, mined-era acknowledged,
declared 2026-09-11). Windows spent in order: [2020, 2021], [2022, 2023],
[2024, 2025]; each recorded `unresolved` with a null closing ground. Weak-signal
cells are recorded under family `refresh_timing_policy`, category `market`, units
`accuracy_points`, one per policy per window against both baselines plus the
pooled decay curve. The cells overlap heavily by construction — they are the same
games re-decided at different instants — so they are correlated decompositions of
one window, not independent votes, and every row says so in its notes.

## Sunday-blackout challenger wired, not served (2026-09-11 follow-up)

The decision above (item 3) is now wired as a prospective challenger,
`late_week_follow_no_sunday_blackout`
(`src/nfl_ats/late_week_follow_no_sunday_blackout_overlay.py`), registered
`ACTIVE_PROSPECTIVE` in `artifacts/prospective/challengers.json` with this
document's own evidence (+3.00 accuracy points, week-blocked 95%
[+0.1848, +5.9259], `probability_positive` 0.9789 at H=3 on 2024-2025;
+0.4657 pooled, 95% [-1.0695, +2.0107], `probability_positive` 0.7218 over
2020-2025; leader-book net-move reliability 0.885 raw / 0.939 Spearman-Brown).
Display name "Follow the line on Sunday too" added to `CHALLENGER_DISPLAY_NAMES`
in `src/nfl_ats/dashboard/findings_content.py`.

**What it does not touch.** Neither `pick_refresh.py` nor
`sharp_book_movement_features.py` was modified. The served late-week follow
rule (`late_week_follow_frame`) keeps excluding Sunday quotes exactly as
before; the challenger is a separate module that independently reads the
same live intraday odds store (`nfl_ats.clv.load_decision_quotes`,
`capture_kind="live"` — the identical local read the served rule already
makes on every pass, at zero extra capture-window cost) and recomputes the
leader-median net move over the same three leading books at the same
`leader_follow_threshold` gate, only without the served rule's
`observed_at_utc < that week's Sunday 00:00 ET` filter. Both arms use the
original Tuesday card and the frozen Tuesday grading line.

**Wired into every `refresh-picks` pass.** `record_late_week_follow_no_sunday_blackout_overlay`
is called unconditionally from `_cmd_refresh_picks`
(`src/nfl_ats/cli_commands/publishing.py`), computing and reporting the
challenger's per-game decision on every pass — including a plain rehearsal
with no `--record-decisions` — and only gating the ledger write on that flag
(a deliberate divergence from the other refresh-challenger recorders, which
skip computation entirely without the flag, so that this challenger's
decision is visible on a dry run too). Recorded rows land in
`prospective/late_week_follow_no_sunday_blackout_decisions.parquet`, one row
per eligible game per pass, alongside the Tuesday pick, the served pick, and
the no-blackout arm's own net move, threshold and pick side.

**Confirmed on the live 2026 Week 1 card**, Thursday 2026-09-10
(`nfl-ats refresh-picks --note lane_rehearsal`, no recording flags, before
any Sunday quote exists for this week): 14 games were still eligible. The
Sunday-inclusive follow fired on the same 2 games the served rule already
follows (ATL_PIT, WAS_PHI, net move +1.0 in both — identical to the served
arm, as expected, since no Sunday-morning quotes exist yet this week to
differ on). The challenger's pick differed from the served card on 1 of the
14 games (DAL_NYG); that difference traces to the model's own recompute
drifting from Tuesday's pick on a game the follow rule never governs, not to
any Sunday-movement effect. The served rule and the played card are
unchanged by this lane.
