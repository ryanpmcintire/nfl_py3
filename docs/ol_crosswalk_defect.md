# The snap-count crosswalk dropped starting offensive tackles

Lane: defect follow-up to XLG-06's §6.1 note in `docs/rookie_priors_on_production.md`.
Measured 2026-09-11. Owner row: **PER-05** (Snap-weighted player value) — that is the
ROADMAP row that owns `src/nfl_ats/players.py` and the snap-derived player features,
not PER-03, which owns the injury-report ingest that the crosswalk joins *against*.

## 1. What was reported, and what is actually true

XLG-06 reported (unverified at the time, and correct as far as it went) that
`weekly_rosters.pfr_id` is essentially empty for offensive linemen, and inferred that
"any production feature derived from snap counts through `_stable_crosswalk` is still
silently dropping nearly every offensive lineman."

**The first half reproduces. The second half does not.** Both were measured this
session against the player snapshot the served feature table was built from
(`data/players/raw/20260910T205112Z`, 649,669 roster rows, 310,475 snap rows,
2013–2025).

The ID map itself is as broken as reported. `_stable_crosswalk` resolved a
`pfr_player_id` for **0.41%** of offensive-line snap rows, against 75.5% front,
73.4% secondary, 86.5% skill:

| Position group | Snap rows | `_stable_crosswalk` hit rate (before) | After the fix |
| --- | ---: | ---: | ---: |
| offensive_line | 49,034 | **0.41%** | 99.89% |
| front | 92,194 | 75.51% | 99.97% |
| secondary | 60,722 | 73.37% | 99.93% |
| skill | 87,479 | 86.48% | 99.88% |
| other (K/P/LS) | 21,046 | 51.19% | 100.00% |

But `attach_snap_player_ids` does not stop at the ID map. It falls back to a
(season, team, normalised-name) match against the roster, then to a globally unique
normalised name. Those fallbacks were quietly carrying almost the whole offensive line.
Final `gsis_id` coverage, which is what the features actually consume:

| Position group | Coverage before | Coverage after | Unlinked rows before | after |
| --- | ---: | ---: | ---: | ---: |
| offensive_line | 98.844% | 99.992% | 567 | 4 |
| front | 99.590% | 99.984% | 378 | 15 |
| secondary | 99.392% | 99.998% | 369 | 1 |
| skill | 99.802% | 99.970% | 173 | 26 |
| other | 99.378% | 99.995% | 131 | 1 |

Per season, offensive-line coverage before the fix never fell below 97.5%
(2013 98.62, 2014 99.15, 2015 99.16, 2016 99.72, 2017 99.29, 2018 98.51, 2019 99.16,
2020 98.61, 2021 98.81, 2022 98.72, 2023 98.98, 2024 98.91, 2025 97.53); after, it is
100.00% in every season except 2020 (99.97), 2021 (99.95) and 2022 (99.98). So the
honest headline is **not** "nearly every lineman was dropped". It is that roughly one
percent of offensive-line snap rows were dropped, and the ones that were dropped were
not random.

## 2. Who was actually missing, and it is worse than one percent sounds

The unlinked rows concentrate on **every-down starting offensive tackles whose name
carries a suffix**. PFR's `snap_counts.player` writes "Orlando Brown Jr."; nflverse's
`weekly_rosters.full_name` writes "Orlando Brown". `_normalized_player_name` keeps
alphanumerics only, so `orlandobrownjr != orlandobrown` and the name fallback misses —
and because `pfr_id` is blank for linemen, there was nothing else to catch them.
Measured for four of them: roster `full_name` has no suffix, snap `player` does.

Twenty-one players are newly linked or corrected in 2019–2025 (716 snap rows). The
largest, by snaps that were invisible to every snap-derived feature:

| Player | Team / season | Snap rows | Offensive snaps recovered |
| --- | --- | ---: | ---: |
| Jedrick Wills Jr. | CLE 2022 | 17 | 1,154 |
| Orlando Brown Jr. | KC 2022 | 17 | 1,131 |
| Orlando Brown Jr. | KC 2021 | 16 | 1,128 |
| Orlando Brown Jr. | CIN 2025 | 17 | 1,111 |
| Jon Runyan Jr. | NYG 2025 | 16 | 1,094 |
| Kelvin Banks | NO 2025 | 17 | 1,066 |
| Josh Conerly | WAS 2025 | 17 | 1,054 |
| Jon Runyan Jr. | GB 2021 / 2022 | 17 / 17 | 1,053 / 1,051 |
| Delmar Glaze | LV 2024 / 2025 | 17 / 17 | 998 / 992 |
| Chris Harris Jr. | LAC 2021 | 14 | 749 defensive snaps |

These are not fringe bodies. A left tackle playing 1,100 snaps had a role share of zero
in `role_states`, so when he appeared on an injury report his absence contributed
**exactly nothing** to `injury_offensive_line_unavailability`, `injury_offense_unavailability`
or the returning-snap-share features.

And one case was not a miss but a **misattribution**. PFR id `WillJo10` is Jonah Williams,
the 2019 first-round left tackle (CIN 2019–2023, ARI 2024–2025, `00-0035629`). The name
fallback matched him to the *other* Jonah Williams, a Rams/Saints defensive lineman
(`00-0035944`). 74 snap rows of a starting left tackle's offensive workload were being
credited to a different team's defensive lineman. Confirmed against the roster: the two
ids never share a team-season.

## 3. The fix

`src/nfl_ats/players.py`:

- `_roster_crosswalk` keeps the old roster-derived map verbatim.
- `_identity_crosswalk` builds a `pfr_id -> gsis_id` map from `players.parquet`, the
  nflverse player-identity master (`data/players/raw/*/players.parquet`, newest
  snapshot, 24,823 rows, 22,651 usable links, `pfr_id` populated for 88.6% of linemen).
- `_stable_crosswalk(rosters, players=None)` resolves through the identity table first
  and the weekly rosters second. Callers may pass an explicit frame; `None` lazily loads
  and caches the newest identity snapshot (keyed on path + mtime + size).
- `attach_snap_player_ids(snaps, rosters, players=None)` threads it through.

One guard was added on top of "players first", because measuring it showed the naive
version regresses six rows: an identity link is used only when its `gsis_id` also appears
in the roster universe. `players.parquet` occasionally carries a non-nflverse id
(Phil Bates is `BAT138483` there and `00-0029389` on every roster, injury report and
stat line) or an id belonging to a long-retired namesake (Chris Smith `SmitCh06` resolves
to `00-0020895`, not the 2024 DT). Preferring those would swap a joinable id for one that
joins to nothing. With the guard, coverage is identical (99.985% overall) and the six
regressions disappear. Net at the row level: **1,571 rows newly linked, 80 rows
corrected, 0 rows lost.**

Deadline safety: the identity table is a static `pfr_id -> gsis_id` mapping of who a
person is. It carries no per-week fact and no outcome, exactly like the roster-derived
map it supplements, which has never been time-filtered either. No pregame feature gains
access to anything dated after its own decision timestamp.

## 4. What it does to production, graded chronologically at the opener

Both arms were rebuilt end to end with `build-learned-availability-features` from the
same pinned snapshots the served table used (player `20260910T205112Z`, player-value
`20260817T184911Z`, pbp `20260817T184927Z`, depth `20260910T210019Z` pinned into a
scratch root so a newer depth capture could not leak in). Neither writes over the served
table.

**Control:** the baseline arm, rebuilt with the identity crosswalk disabled, reproduces
the served `game_features_weak_stack.parquet` **exactly** — 0 of 273 numeric columns
differ on 4,902 rows. Everything below is the crosswalk and nothing else.

**Feature movement:** 42 of 273 numeric columns move. All seven injury-unavailability
metrics, both injury-value-lost metrics, all three returning-snap-share metrics, and the
QB start-probability / expected-EPA pair (that last family moves because the learned
availability rates are fitted on the linked snap panel, so every player's severity shifts
a little). The lineup-continuity family does **not** move: continuity is keyed on
`player_key`, which falls back to `pfr:<id>`, so an unlinked lineman was still counted
consistently. 1,641 of the 1,693 games in 2020–2025 have at least one moved column.
Typical size is small — mean absolute move of
`diff_injury_offensive_line_unavailability` is 0.0038 against a column standard
deviation of 0.135 — with a long tail:

| Column | Largest 2020–2025 move | Game |
| --- | ---: | --- |
| `diff_injury_defense_disruption_value_lost` | 1.071 | 2021_03_LAC_KC |
| `diff_injury_defense_disruption_value_lost` | 1.054 | 2021_02_DAL_LAC |
| `diff_injury_offensive_line_unavailability` | 0.180 | 2024_08_BAL_CLE |
| `diff_injury_offensive_line_unavailability` | 0.179 | 2025_11_SF_ARI |
| `diff_returning_offense_snap_share` | 0.153 | 2023_17_CIN_KC |
| `diff_returning_offense_snap_share` | 0.141 | 2023_18_CLE_CIN |

The 2021 Chargers games are Chris Harris Jr.; the Cleveland games are Jedrick Wills Jr.;
the Cincinnati games are Jonah Williams and Orlando Brown Jr.

**Grade.** Both arms were scored with the production opener evaluator
(`nfl_ats.clv.opener_pick_evaluation`, active model `d49194e04945a5e5`: weak_stack,
ridge alpha 10, gaussian_median, no calibration, served home-side offset, weekly refit on
the full prior close-graded history, `min_train_games=500`). 1,537 paired games 2020–2025,
1,503 after pushes.

Served rule (probability rule at the opener, offset applied):

- **Picks changed: 13 of 1,503 (0.86%).**
- Accuracy: baseline **54.558%**, candidate **54.491%**, **−0.0665 accuracy points**.
- Week-blocked paired bootstrap (107 blocks, 20,000 resamples, seed 20260812):
  95% [−0.534, +0.400] accuracy points, **`probability_positive` 0.387**.
- Season-blocked: 95% [−0.586, +0.402], `probability_positive` 0.416.
- Brier: baseline 0.251581, candidate 0.251615, improvement **−0.000034**, week-blocked
  95% [−0.000124, +0.000054], `probability_positive` 0.226.

| Season | Games | Flips | Baseline | Candidate | Delta (pts) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2020 | 220 | 3 | 52.273% | 51.818% | −0.455 |
| 2021 | 236 | 2 | 54.237% | 55.085% | +0.847 |
| 2022 | 248 | 3 | 55.242% | 55.645% | +0.403 |
| 2023 | 266 | 0 | 56.391% | 56.391% | 0.000 |
| 2024 | 266 | 2 | 53.008% | 53.008% | 0.000 |
| 2025 | 267 | 3 | 55.805% | 54.682% | −1.124 |

Residual-sign rule, same population: 12 picks changed, 52.961% vs 52.828%,
−0.133 accuracy points, week-blocked 95% [−0.596, +0.331], `probability_positive` 0.281.

**The accuracy number carries almost no information about the fix, and the reason is
visible in the flips.** Every one of the 13 changed picks had a baseline probability
between 0.4968 and 0.5052 — the crosswalk moved only games that were already a dead
heat. The baseline was right on 7 of those 13 and the candidate on 6. The entire
measured "effect" is one coin flip out of 1,503 games.

## 5. The decision

**Serve it.** The pool is forced picks; the question is which of two cards to submit,
and the correctness case is the one with evidence behind it while the grade case has
almost none.

The reading in favour: this fixes a measured data error. A starting left tackle's 74
games of snaps were credited to another team's defensive lineman, and ten more full-time
starters across 2019–2025 had a role share of zero, so their injury-report appearances
moved the model not at all. After the fix, snap-count identity resolves for 99.99% of
rows in every position group in every season, with nothing lost. Every downstream
consumer of `attach_snap_player_ids` — the injury-unavailability family, injury value
lost, expected lineup loss, the returning-snap-share family, `lineup_availability`,
`play_probability`, `inactives_refresh_overlay`, and `qb_identity_features`'
combine-to-gsis join — inherits it.

The reading against, stated plainly as required: on the 2020–2025 opener grade the
candidate is **0.067 accuracy points worse** and its Brier is **0.000034 worse**, and
`probability_positive` is 0.387 on accuracy and 0.226 on Brier. Read as expected value
on the graded number alone, that is a 39/61 bet against the fix. Anyone who wants to
weigh only the grade should decline it.

Why the first reading wins here. The grade is decided by 13 games, all of which sat
within half a percentage point of 50/50 before anything changed; a 7–6 split on 13 coin
flips is not evidence about a crosswalk. The correctness claim, by contrast, is measured
and specific: a named player's snaps were on the wrong team. Per AGENTS.md this is not a
result that closes anything in either direction — **`unresolved_below_power`**, recorded
with `probability_positive`, not "the interval contains zero". Neither closing ground
applies: the sign is not resolved (it is positive in 2021 and 2022, negative in 2020 and
2025, exactly zero in 2023 and 2024), and no positive control has bounded a
crosswalk-sized effect on this instrument.

**This lane did not switch the served feature table or the active model.** Serving it
means rebuilding `game_features_weak_stack.parquet` on the fixed code and re-running the
usual refit/opener-evaluation/composition/publish chain, which is a separate, deliberate
act.

## 6. Reproduction

```
# coverage, before and after, by position group and season
#   scratch scripts in the lane's scratchpad; the measurement is
#   attach_snap_player_ids(snaps, rosters, <empty frame>) vs attach_snap_player_ids(snaps, rosters)
# arms
nfl-ats build-learned-availability-features \
  --features data/processed/game_features_pbp.parquet \
  --destination <scratch>/game_features_weak_stack_fixed.parquet \
  --rates-destination <scratch>/rates_fixed.parquet \
  --evaluation-destination <scratch>/availability_eval_fixed.csv \
  --player-snapshot 20260910T205112Z --player-value-snapshot 20260817T184911Z \
  --pbp-snapshot 20260817T184927Z --depth-root <scratch>/depth
# grade: nfl_ats.clv.opener_pick_evaluation on each arm with the active model config,
#        paired week-blocked bootstrap, 20,000 resamples, seed 20260812
```

Recorded: `ol_crosswalk_fix_opener_accuracy`, `ol_crosswalk_fix_opener_brier`
(family `ol_crosswalk_fix`, category `modeling`, both `unresolved_below_power`).
