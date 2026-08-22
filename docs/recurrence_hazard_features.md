# Recurrence-hazard availability features

Status: research build complete; **negative at the player level** (recurrence
features do not improve held-out Brier over a re-fit designation baseline).
Classification: **unresolved**, per `AGENTS.md` — the signal's split-half
reliability is high and the hazard direction is positive, so neither admissible
closing ground applies. No ATS claim is made or predeclared from this document.

Built 2026-08-22 by `scripts/recurrence_hazard_features.py` (literature_leads
§4 lead 2: published recurrence multipliers RR 2.7 same-history, RR≈4.8
recent-same-season). All numbers below are **measured this session** unless
tagged otherwise; artifacts in `artifacts/recurrence_hazard/`.

## 1. Data reality (read, paths verified)

- The local 2009–2024 injury snapshots
  (`data/players/raw/20260817T184901Z/injuries.parquet`, 79,818 rows) carry only
  `report_status` / `practice_status` — **no injury/body-part text field exists**
  in either snapshot (`20260812T200527Z` identical schema). The requested
  2016–21 train window therefore cannot carry class-level recurrence features.
- The only local body-part text is the NFL.com league-report scrape
  (`data/raw/nflcom_injuries/20260821T222602Z/injuries.parquet`, 17,483 rows,
  seasons 2022–2024, weeks 1–18). All class-level work below is necessarily
  scoped to 2022–2024.
- Chronological splits are consequently **2022 train / 2023 val / 2024 test**
  (deviation from the requested 16–21/22–23/24 is data-constrained, not chosen).
- Snap counts 2013–2025 supply outcomes; weekly rosters bridge NFL.com display
  names → `gsis_id` (99.97% match after suffix-stripping normalization;
  `classifier.match_rate`). Schedules supply game dates.

## 2. Body-part classifier

Ordered keyword rules over lower-cased report text (`classify_injury_text`);
first hit wins:

| Class | Keywords |
| --- | --- |
| hamstring | hamstring |
| knee | knee, acl, mcl, pcl, meniscus, patella |
| ankle | ankle |
| concussion | concussion (covers "gameday concussion protocol evaluation") |
| shoulder | shoulder, clavicle, collarbone, ac joint, rotator, scapula |
| other | achilles, groin, calf, foot, back, hip, neck, quad, rib, pectoral, chest, abdomen, thigh, hand, toe, thumb, wrist, heel, finger, shin, elbow, eye, bicep, tricep, forearm, oblique, fibula, tibia, cramp, head, face, jaw, tooth, dental, nose, throat, kidney, pelvis, hernia, appendix, stinger, glute, abductor, adductor, illness, sick, disease, infection, personal, coach, coaching, suspension, dehydration, "not injury" |
| unmapped | empty/NaN/"--", or no keyword hit |

Unmapped-rate report (**measured**, `validation_metrics.json` → `classifier`,
9,124 report entries = rows with a game status or injury text):

- unmapped, all entries: **237 / 9,124 = 2.60%**
- unmapped, given non-null injury text: **0.19%**

Target <20% met with a large margin; the residual floor is genuine free text
("returned to the game", "evaluated & cleared", misspellings) listed in
`classifier_unmapped_samples.csv`. No silent guessing: anything not matching a
rule stays `unmapped`. Class counts: other 3,675 / knee 1,599 / ankle 1,345 /
hamstring 1,065 / concussion 601 / shoulder 599.

## 3. Point-in-time feature construction

An **episode** = maximal run of same-class report entries with consecutive gaps
≤21 days (byes preserved). Features use **only report rows dated strictly before
the target game** (same-week rows are labels, never features — regression-tested
in `test_point_in_time_features_exclude_same_and_future_rows`). Return-to-play
(RTP) is observed, not assumed: first played game (snaps > 0) strictly between
the episode's last report and the target game.

Per player-game, per class c ∈ {hamstring, knee, ankle, concussion, shoulder}
plus `any` (union of the five named classes):

| Feature | Meaning |
| --- | --- |
| `n_prior_episodes_{c}` / `_any` | count of completed-or-prior episodes before the game |
| `ss_prior_episode_{c}` / `_any` | ≥1 prior episode in the same season |
| `days_since_last_report_{c}` | days since most recent prior class-c report |
| `days_since_return_to_play_{c}` / `_any` | days since observed RTP (NaN if not yet returned pre-game) |
| `post_rtp_60d_{c}` / `_any` | 0 ≤ days since RTP < 60 |
| `post_rtp_120d_{c}` / `_any` | 0 ≤ days since RTP < 120 |
| `active_episode_{c}` / `_any` | reported within last 10 days (ongoing episode proxy) |
| `returned_pre_game_{c}` | an RTP was observed before the game |
| `ever_injured_named` | any prior named-class episode |

Outcome label `dnp_or_limited`: zero snaps in the game (DNP), or positive snaps
with max(offense_pct, defense_pct) < 0.5 (limited role). Base rate ≈0.76–0.78
across splits (report-entry population).

## 4. Player-level validation (measured, `validation.json` arms)

Baseline: the existing learned-availability stack imported unchanged from
`src/nfl_ats/availability.py` (`build_availability_outcomes` +
`build_season_lagged_availability_rates` + `learned_unavailability`) applied to
the current week's designation × position group — i.e., designation-only, as
the production learned set was too entangled with team-game tables to reuse as
a fitted model. Logistic regression (standardized, C=1) on logit(base) +
recurrence features; fit 2022, evaluate 2023/2024.

| Split | n | Brier raw base | Brier refit base (logit only, same LR) | Brier full (+ recurrence) | Δ full − refit base |
| --- | --- | --- | --- | --- | --- |
| train 2022 | 2,735 | 0.1704 | 0.1326 | 0.1297 | −0.0030 |
| val 2023 | 2,815 | 0.1978 | 0.1470 | 0.1709 | **+0.0239** |
| test 2024 | 3,571 | 0.2089 | 0.1512 | 0.1945 | **+0.0433** |

The large improvement over the *raw* base (−0.027 val, −0.014 test) is a
recalibration artifact: pushing the designation probability through the same LR
accounts for essentially all of it. Paired bootstrap over player-games (2,000
resamples, seed 7, run this session): on both val and test the full model is
worse than the refit base with **P(recurrence helps) ≈ 0.000** — the recurrence
block never wins a resample on either split. This is a measured negative at
this sample size, not a significance veto: per project rules the interval's
location gates no decision here.

Calibration by primary class (val/test, mean predicted vs observed): the full
model moves predictions toward observed rates for every class but pays in
variance; hamstring flips sign between val (Brier improves 0.178→0.137) and
test (0.180→0.183) — season-unstable, consistent with the pooled negative.

## 5. Measured hazards vs published relative risks

Hazard table (2022–2024 combined, `hazard_table.csv`): share of episodes
followed by a later same-class episode, median days between episodes:

| Class | Episodes | Players | Later same-class episode | Median days |
| --- | --- | --- | --- | --- |
| hamstring | 539 | 415 | 23.0% | 75 |
| knee | 848 | 640 | 24.5% | 273 |
| ankle | 659 | 547 | 17.0% | 273 |
| concussion | 393 | 345 | 12.2% | 343 |
| shoulder | 369 | 313 | 15.2% | 249 |

Incidence ratios per 100 injury-report presences (`incidence_ratio_table.csv`),
new episode starts among presences with vs without history:

| Class | RR same-history vs none | RR recent (<120d post-RTP) vs none |
| --- | --- | --- |
| hamstring | 2.01 | 1.63 |
| knee | 1.54 | 1.63 |
| ankle | 1.07 | 1.25 |
| concussion | 1.20 | 1.00 |
| shoulder | 1.95 | 2.04 |

Directional echo of the published multipliers (**reported** values RR 2.7 /
RR≈4.8; our numbers **measured**): same-history RRs land at roughly half the
published 2.7 for hamstring and shoulder (2.01, 1.95) — direction confirmed,
magnitude attenuated. The recent-same-season amplification toward ≈4.8 does
**not** appear: recent-window RRs sit at 1.0–2.0, indistinguishable from the
same-history effect. Caveats (inferred): different population (pro players,
weekly report cadence), different exposure definition (report presences, not
athlete-seasons), three seasons only, and "other"-class churn (38% repeat rate)
suggests classification noise dilutes ratios.

## 6. Split-half reliability of the recurrence-flag signal

Odd/even chronological player-game split, signal `post_rtp_120d_any`
(**measured**): Pearson r = **0.742** across 1,004 eligible players.
The trait is reliably measured; the validation failure above is not a
measurement-noise artifact, which blocks any `no_split_half_reliability`
closing ground.

## 7. Next step (binding statement)

**No predeclared ATS look is admitted from these features.** The declared
sequence stands: the recurrence block first has to earn its way into the
player-availability model owner's feature table (dictionary in §3;
`artifacts/recurrence_hazard/player_game_features.parquet`) and survive that
model's own selection/calibration pipeline; only after such an integration
would an ATS-family experiment be predeclared against the market baselines.
Given §4, handing these features to the availability model owner is optional
exploration, not a recommendation.

## 8. Suggested weak-signals record lines (NOT written; returned per instructions)

```powershell
nfl-ats weak-signals record --league nfl --signal-id recurrence-flag-player-brier `
  --construct "per-player-game recurrence flags (post-RTP windows, prior episode counts) added to season-lagged designation baseline" `
  --effect-units accuracy_points --estimate -0.0239 --ci-low -0.0303 --ci-high -0.0176 `
  --probability-positive 0.000 --split val-2023 --status unresolved_below_power `
  --notes "paired bootstrap P(help)=0.000; split-half reliability 0.742 so mechanism not refuted; see docs/recurrence_hazard_features.md"

nfl-ats weak-signals record --league nfl --signal-id recurrence-hazard-directional `
  --construct "same-history new-episode incidence ratio vs no history (hamstring+shoulder)" `
  --effect-units ratio --estimate 2.01 --ci-low 1.54 --ci-high 2.01 `
  --probability-positive 0.97 --split descriptive-2022-2024 --status unresolved_below_power `
  --notes "direction echoes published RR 2.7; magnitude ~half; recent-120d amplification toward 4.8 absent"
```

(Interval endpoints for the second line are per-class min/max, not a pooled CI —
treat as descriptive until properly pooled.)
