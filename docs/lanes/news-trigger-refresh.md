# News-triggered refresh

## Goal

Dispatch prospective, pregame paper-pick refreshes from game-keyed news events while preserving the served model's single fitted probability and side selection.

## State

The MKT-08 bridge accepts lineup changes, posted inactives, and line moves with explicit game provenance. A durable activation watermark prevents historical backfill, and a successful child refresh writes a completion receipt used for deduplication. The exact UTC scan time reaches the pick-revision path as trigger provenance. Evidence rows and dry runs never count as successful dispatches. The Sunday scheduler entry runs at 12:40 Eastern with catch-up disabled; an already running scheduler must restart to load the module-level schedule.

## Tried

Injury-news dispatch was excluded because the retained source identifies capture time but not the affected game. Failed child refreshes remain pending for retry. A first live invocation establishes the watermark before scanning, so retained historical events are not dispatched.

## Next

Measured: the direct dry command passed without writes or dispatching historical events; 139 focused existing tests passed. The root restarted the verified idle scheduler hidden to load the new schedule. Let future scheduled scans create prospective evidence; score the fixed-clock and news-triggered arms only after real decision rows accrue.

## Open

The first live run intentionally dispatches no older rows. One Sunday scan does not cover later inactive postings. Injury news remains blocked until the source provides reliable game identity. No prospective comparison sample exists yet.

## Unit: historical grade of a Friday-designation resolution refresh (2026-09-23, predeclared before running)

Rather than wait on prospective MKT-08 events, grade the same idea
retrospectively: does a player's status resolving against its Friday
designation, once visible before the pick deadline, sharpen the served
four-term probability when added as a fifth fitted term?

Predictor construction (predeclared): for each team-game, `fixed_unavailability`
(from `data/processed/injury_play_outcomes.parquet`, values 0.35 questionable /
0.85 doubtful / 1.0 out / 0.05 probable / practice-fallback, identical mapping
to `nfl_ats.availability.fixed_unavailability`) is the pregame Friday-designation
severity already baked into `home_injury_skill_epa_value_lost` +
`home_injury_defense_disruption_value_lost` (and away) in
`data/processed/game_features_player_value.parquet` via
`players._injury_value_features` (`severity * role_share * value_rate`, summed
over all visible injured players). Define per-team-game `v = team_total_value_lost
/ sum(fixed_unavailability over all that team's injury-report rows)` (a
"value per severity unit" constant, algebraically exact since
`_injury_value_features` sums `severity_i * role_share_i * value_rate_i` and
role_share/value_rate are unobserved per-player here, so `v` implicitly stands
in for their severity-weighted average — the same allocation-by-severity
approximation Unit 3 already used and disclosed). Only players with
`report_category == "questionable"` are resolved: their pregame severity
(0.35) is replaced with the actual outcome (`unavailable` column, 0.0/1.0,
from actual snap participation). Team shift = `v * sum(unavailable_actual -
0.35)` over that team's questionable players (0 if none). Candidate feature
`news_trigger_value_shift = shift_home - shift_away`, added onto the existing
pregame `value_lost_diff` sign convention (home - away). Reported for scale
only (not re-fit): margin points = `-0.320 * news_trigger_value_shift`, the
Unit 2 pooled slope from `docs/lanes/injury-scenario-producer.md`.

Visibility gate (predeclared, conservative per the task's own instruction):
a team's questionable-resolution is treated as visible before
`min(kickoff, Sunday 4 PM ET)` only if the game is NOT Monday and NOT a
Sunday game with kickoff at/after 16:00 ET; all other games (Thursday,
Saturday, Sunday kickoff <16:00 ET, and any other early weekday/window) use
their own kickoff as the deadline, so the ~90-minute-pre-kickoff inactives
report is visible by construction. Games failing the gate keep
`news_trigger_value_shift = 0.0` (no refresh available at deadline), not
dropped from the population.

Scope: same point-in-time and opener-evaluation constraints Unit 2 already
measured (`home_injury_observed_at` populated; `build_fit_population`'s
opener-evaluation window) intersect to **seasons 2020-2024**, not 2020-2025 —
2025 has 0% `home_injury_observed_at` attestation per Unit 2, a measured
constraint, not a new choice; disclosed deviation from the "2020-2025 or
wider" ask.

Fit: baseline = served 4-term LOSO logistic (`model_logit`,
`composition_flag_sum`, `market_move_toward_home`, `market_move_available`,
exactly `FIT_FEATURES`); candidate = same 4 terms +
`news_trigger_value_shift` as a 5th fitted term (per AGENTS.md, no bare flip).
Paired accuracy/Brier/log-loss/decisive-game record on the same held-out
games, via `nfl_ats.signal_atlas._cell`, identical method to Unit 2 part B.
Separately reported (identification step): count of games with >=1
questionable-designation player pregame, how many pass the visibility gate,
and how many have a nonzero resolved shift.

Script: `scripts/news_trigger_historical.py`. Output:
`artifacts/news_trigger_historical/<ts>/`. Run once, no `src/` changes.

## Unit result (2026-09-23, run once for real)

Ran `.tools/uv.exe run python scripts/news_trigger_historical.py`. Output:
`artifacts/news_trigger_historical/20260923T212553Z/` (`summary.json`,
`scoped_population.parquet`, `paired_probability_population.parquet`).
`ruff check` clean.

Identification: 1226 scoped games (seasons 2020-2024, point-in-time only,
same population Unit 2 used). 1194/1226 (97%) had at least one
`questionable`-designated player pregame on either side. Only 747/1226 (61%)
pass the visibility gate (not Monday, not Sunday kickoff >=16:00 ET); 464
games had a pregame questionable player but were gated out (late-Sunday or
Monday games, resolution not visible before the deadline). 726 games have a
nonzero resolved shift. Mean absolute margin-point shift when nonzero: 0.240
points (using the -0.320 Unit-2 slope) — small, consistent with Unit 2/3's
prior finding that injury value-lost magnitudes are small.

Cover-probability paired look (candidate = served 4-term LOSO logistic +
`news_trigger_value_shift` as a 5th term, vs. the served 4-term baseline,
`nfl_ats.signal_atlas._cell`): accuracy delta **-1.223 points** (candidate
worse), probability candidate is better = **0.094**, interval
**[-3.179, 0.488]** (crosses zero, upper bound is barely positive — not the
whole interval on the wrong side, so this is NOT `wrong_sign_resolved`).
79 decisive games: candidate wins 32, baseline wins 47, exact binomial
p=0.115. Brier improvement -0.000127 (probability_positive=0.506, coin
flip). Log loss improvement -0.000236 (probability_positive=0.512, coin
flip). Classification: **unresolved_below_power** — accuracy leans against
the candidate (9.4% probability of being better) but the interval still
touches the positive side and Brier/log-loss show no signal at all, so this
is not a refuted mechanism (AGENTS.md requires the whole interval on the
wrong side) and no positive control was run.

Decision-relevant conclusion: refreshing the injury value-lost differential
with the actual Friday-questionable resolution (posted-inactive-list-visible
subset only) does not clearly help the served probability over 2020-2024,
and the accuracy signal leans toward hurting it, though not resolved to a
refutation. This closes nothing but argues against folding this specific
5th term into the served path without more power or a design change (e.g.
per-player role-share attribution instead of the severity-only allocation
this unit used).

Record command for the root (not run — registry write is out of scope here):
```
nfl-ats weak-signals record --name news_trigger_resolution_cover_probability --description "news_trigger_value_shift (Friday-questionable resolution to actual play/inactive outcome, gated to resolutions visible before min(kickoff, Sunday 4pm ET), value-weighted via Unit 2's -0.320 points-per-unit slope) added as a 5th fitted term to the served 4-term LOSO logistic (model_logit, composition_flag_sum, market_move_toward_home, market_move_available), seasons 2020-2024, point-in-time injury data only, paired against the served 4-term baseline on the same held-out games." --source artifacts/news_trigger_historical/20260923T212553Z/summary.json --effect -1.2234910277324633 --effect-units accuracy_points --classification unresolved_below_power --league nfl --season-start 2020 --season-end 2024 --standard-error 0.9459525447539588 --interval-low -3.1785864940022033 --interval-high 0.4883019134920306 --probability-positive 0.09425 --sample-games 1226 --sample-blocks 89 --classification-evidence "Bootstrap probability_positive=0.09425 for candidate beating baseline on accuracy (2000 draws, season/week blocks), interval [-3.179,0.488] touches the positive side so not wholly on the wrong side; exact_null_p=0.115 on 79 decisive games (32 candidate wins vs 47 baseline wins); Brier and log-loss improvements are near zero with probability_positive 0.506/0.512 (no signal). Sign not reversed with the whole interval on the wrong side (not refuted), no positive control run, so unresolved_below_power." --category health --plain-summary "Updating the injury picture once a questionable player's status is confirmed (played or sat) did not clearly help or hurt picks over 2020-2024; if anything it leaned toward hurting slightly, but not enough to call it settled."
```

## Next
1. `weak-signals record` above needs the root to run it before any write-up
   calls this settled.
2. If revisited: the severity-only allocation (no per-player role-share
   weighting, precedent from Unit 3) is the weakest link — a per-player
   value decomposition (needs role shares at decision time, not just
   post-hoc snap_share) would sharpen `news_trigger_value_shift` and could
   change the result. 464/1226 games are gated out by the conservative
   Sunday-4pm/Monday rule; a precise per-game inactive-report-post-time
   (instead of the blanket >=16:00 ET rule) would recover some of those.
3. MKT-08's prospective dispatch path (top of this lane) is unaffected by
   this historical unit and still waits on real scheduled scans.
