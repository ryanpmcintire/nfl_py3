# News-triggered refresh

## Goal

Dispatch prospective, pregame paper-pick refreshes from game-keyed news events while preserving the served model's single fitted probability and side selection.

## State

The MKT-08 bridge accepts lineup changes, posted inactives, and line moves with explicit game provenance. A durable activation watermark prevents historical backfill, and a successful child refresh writes a completion receipt used for deduplication. The exact UTC scan time reaches the pick-revision path as trigger provenance. Evidence rows and dry runs never count as successful dispatches. The Sunday scheduler entry runs at 12:40 Eastern with catch-up disabled; an already running scheduler must restart to load the module-level schedule.

## Tried

Injury-news dispatch was excluded because the retained source identifies capture time but not the affected game. Failed child refreshes remain pending for retry. A first live invocation establishes the watermark before scanning, so retained historical events are not dispatched.

## Next

- 2026-09-25 root: unit 1 (already recorded) and the three role-share-unit cells recorded (registry 7,261-7,263), all unresolved_below_power. Per-player role share did not rescue the term (accuracy -0.98 [-2.32,+0.09] P+ 0.04, 12-24 decisive). Reopen only with a decision-time inactive post time or a new mechanism.


Measured: the direct dry command passed without writes or dispatching historical events; 139 focused existing tests passed. The root restarted the verified idle scheduler hidden to load the new schedule. Let future scheduled scans create prospective evidence; score the fixed-clock and news-triggered arms only after real decision rows accrue.

## Open

The first live run intentionally dispatches no older rows. One Sunday scan does not cover later inactive postings. Injury news remains blocked until the source provides reliable game identity. No prospective comparison sample exists yet.

## Unit: historical grade of a Friday-designation resolution refresh (2026-09-23, predeclared before running)

Retrospective grade: does a player's status resolving against its Friday
designation, once visible before the pick deadline, sharpen the served
four-term probability as a fifth fitted term? Predictor: team-level
`v = total_value_lost / severity_sum` (severity-only allocation, no role
share) applied to each team's resolved-questionable-player severity delta;
`news_trigger_value_shift = shift_home - shift_away`. Visibility gate: not
Monday, not Sunday kickoff >=16:00 ET (predeclared conservative proxy for
"before min(kickoff, Sunday 4pm ET)"). Scope: seasons 2020-2024 (2025 has 0%
`home_injury_observed_at` attestation, a measured constraint). Fit: baseline
= served `FIT_FEATURES`; candidate = same + 5th term, LOSO-by-season
logistic, graded via `nfl_ats.signal_atlas._cell`. Script:
`scripts/news_trigger_historical.py`.

## Unit result (2026-09-23, run once for real)

Ran `.tools/uv.exe run python scripts/news_trigger_historical.py`. Output:
`artifacts/news_trigger_historical/20260923T212553Z/`. `ruff check` clean.

1226 scoped games; 97% had >=1 questionable player pregame; 61% (747/1226)
pass the visibility gate; 726 have a nonzero resolved shift. Mean abs shift
when nonzero: 0.240 margin points (small, consistent with prior injury-value
findings).

Cover-probability paired look: accuracy delta **-1.223 points** (candidate
worse), probability candidate is better = **0.094**, interval
**[-3.179, 0.488]** (crosses zero, upper bound barely positive — not the
whole interval on the wrong side, so NOT `wrong_sign_resolved`).
79 decisive games: candidate wins 32, baseline wins 47, exact binomial
p=0.115. Brier improvement -0.000127 (probability_positive=0.506, coin
flip). Log loss improvement -0.000236 (probability_positive=0.512, coin
flip). Classification: **unresolved_below_power** — accuracy leans against
the candidate (9.4% probability of being better) but the interval still
touches the positive side and Brier/log-loss show no signal at all, so this
is not a refuted mechanism (AGENTS.md requires the whole interval on the
wrong side) and no positive control was run.

Decision-relevant conclusion: the severity-only refresh does not clearly help
the served probability over 2020-2024 (leans toward hurting, not resolved to
a refutation). Closes nothing; argues against folding this specific 5th term
into the served path without a design change (see the role-share unit below).

Record command for the root (not run — registry write is out of scope here):
```
nfl-ats weak-signals record --name news_trigger_resolution_cover_probability --description "news_trigger_value_shift (Friday-questionable resolution to actual play/inactive outcome, gated to resolutions visible before min(kickoff, Sunday 4pm ET), value-weighted via Unit 2's -0.320 points-per-unit slope) added as a 5th fitted term to the served 4-term LOSO logistic (model_logit, composition_flag_sum, market_move_toward_home, market_move_available), seasons 2020-2024, point-in-time injury data only, paired against the served 4-term baseline on the same held-out games." --source artifacts/news_trigger_historical/20260923T212553Z/summary.json --effect -1.2234910277324633 --effect-units accuracy_points --classification unresolved_below_power --league nfl --season-start 2020 --season-end 2024 --standard-error 0.9459525447539588 --interval-low -3.1785864940022033 --interval-high 0.4883019134920306 --probability-positive 0.09425 --sample-games 1226 --sample-blocks 89 --classification-evidence "Bootstrap probability_positive=0.09425 for candidate beating baseline on accuracy (2000 draws, season/week blocks), interval [-3.179,0.488] touches the positive side so not wholly on the wrong side; exact_null_p=0.115 on 79 decisive games (32 candidate wins vs 47 baseline wins); Brier and log-loss improvements are near zero with probability_positive 0.506/0.512 (no signal). Sign not reversed with the whole interval on the wrong side (not refuted), no positive control run, so unresolved_below_power." --category health --plain-summary "Updating the injury picture once a questionable player's status is confirmed (played or sat) did not clearly help or hurt picks over 2020-2024; if anything it leaned toward hurting slightly, but not enough to call it settled."
```

## Unit: per-player role-share value decomposition (2026-09-25, predeclared before running)

Fixes the named weakness: Unit 1's `v` was a team-average value-per-severity
constant applied uniformly to every resolved player, ignoring each player's
own role. This unit computes a genuine per-player allocation.

Trailing role share (decision-time, not post-hoc): build the full
per-player-per-game snap table via `attach_snap_player_ids(canonicalize_snaps(snaps),
canonicalize_rosters(rosters))` from `latest_player_snapshot(data/players/raw)`
(same helpers `scripts/injury_outcomes_table.py` uses to build
`injury_play_outcomes.parquet`, but over every game, not just injury rows).
Per (team, gsis_id), sorted chronologically by (season, week, game_id):
`trailing_offense_pct`/`trailing_defense_pct` = `.shift(1).ewm(alpha=2/(8+1),
adjust=False).mean()` — `role_span=8` matches `players.enrich_with_player_features`'s
production default, and reading the shifted series before the current game
mirrors production's `role_states[team]` timing (read, then updated after).
A player's first game for a team has no prior value, filled 0.0, matching
production's empty-role default.

Per-team-game value-per-role-share constants (channel-split, from
`data/processed/game_features_player_value.parquet`): `v_role_skill =
home_injury_skill_epa_value_lost / sum(fixed_unavailability_i *
trailing_offense_pct_i)` over all that team-game's injury-report rows (0 if
denominator <=0); `v_role_defense` symmetric with
`home_injury_defense_disruption_value_lost` and `trailing_defense_pct`.
Per resolved questionable player: `player_shift_i = (v_role_skill *
trailing_offense_pct_i + v_role_defense * trailing_defense_pct_i) *
(unavailable_actual_i - 0.35)`. Team shift = sum over that team's
questionable players (0 if none). `news_trigger_value_shift_v2 = shift_home -
shift_away`, same sign convention as Unit 1.

Visibility gate: kept unchanged (not Monday, not Sunday kickoff >=16:00 ET).
Checked before fitting: all 55 `data/players/inactives/*/inactives.parquet`
snapshots — 54/55 are empty; the one nonempty snapshot (11 rows,
`captured_at_utc=2026-09-24T22:50:37Z`) carries only capture time, not an
official per-game post time — the same limitation already documented for
the excluded injury-news source. No reliable per-game inactive-report post
time exists, so the blanket gate stays.

Scope: identical to Unit 1 — seasons 2020-2024, `build_fit_population`'s
point-in-time opener-evaluation population.

Fit: baseline = `FIT_FEATURES` (4 terms). Candidate A = baseline +
`news_trigger_value_shift` (Unit 1's severity-only term, recomputed here for
a same-run paired comparison). Candidate B = baseline +
`news_trigger_value_shift_v2` (this unit). Both LOSO-by-season logistic,
`nfl_ats.pick_probability_fit._fit_logit`, ridge=`FIT_RIDGE`.

Grading, each candidate vs the same baseline: (a) accuracy/Brier/log-loss
paired cell via `nfl_ats.signal_atlas._cell`, identical method to Unit 1. (b)
Line movement toward the pick: `lm = (1 if pick_home else -1) *
market_move_toward_home`, restricted to rows with `market_move_available==1`;
`diff_line_move = lm_candidate - lm_baseline`; graded via
`line_move_yardstick_paired_eval.cell_stats` (season+week block bootstrap),
same convention `scripts/opener_error_transfer_unit4.py` already uses.
Decisive-game record reported first for both families. Bootstrap draws=2000,
seed=20260925 (new look, disclosed). 2 candidates x 2 grading families = 4
looks graded against baseline in this unit, on top of Unit 1's 3 (accuracy,
Brier, log-loss).

Script: `scripts/news_trigger_player_value.py` (new; reuses
`nfl_ats.pick_probability_fit`, `nfl_ats.signal_atlas._cell`,
`line_move_yardstick_paired_eval.cell_stats`, and `nfl_ats.players` snap
helpers). Output: `artifacts/news_trigger_player_value/<ts>/`. Run once, no
`src/` changes.

## Unit result (2026-09-25, run once for real)

Ran `.tools/uv.exe run --no-sync python scripts/news_trigger_player_value.py`.
No runtime errors; script ran unchanged. Output:
`artifacts/news_trigger_player_value/20260925T192146Z/`. `ruff format --check`
and `ruff check` both pass on the script (no fixes needed).

Leakage spot check (measured): for player `ARI 00-0022921`, week-2 row's
`trailing_offense_pct=1.000000` equals week-1's own `offense_pct=1.00`, not
week-2's own `offense_pct=0.65`; week-1 (first game for the team) is
correctly `0.0` (no prior). Matches manual `shift(1).ewm(...)` recompute
exactly (`max_abs_diff=0.0`) across 3 spot-checked players. Trailing role
shares exclude the current game.

Identification: 1226 scoped games, 97% (1194) had >=1 questionable player
pregame, 61% (747) pass the visibility gate, 663 have a nonzero resolved
`_v2` shift; mean abs shift when nonzero: 0.604 margin points (larger than
Unit 1's 0.240, expected since per-player role weighting concentrates value
on higher-share players instead of spreading it as a flat team average).

**Candidate B (per-player role-share, `news_trigger_value_shift_v2`) vs
baseline:**
- Accuracy: decisive 36 games, baseline wins 24, candidate wins 12
  (exact_null_p=0.0652). Effect -0.9788 accuracy points, se 0.6416, interval
  [-2.3239, 0.0852], probability_positive **0.042**. Brier improvement
  -0.000283 (P+ 0.182), log-loss improvement -0.000583 (P+ 0.184).
  Classification: **unresolved_below_power** (interval crosses zero, upper
  bound positive).
- Line movement toward the pick: decisive 16 games, tied 8-8. Effect
  +0.00189 pts/game (season-block), interval [-0.0152, 0.0189],
  probability_positive **0.7435**. Classification: **unresolved_below_power**
  (crosses zero, decisive record a coin flip).

**Candidate A (severity-only, recomputed this run for a same-run paired
comparison) vs baseline:**
- Accuracy: decisive 79 games, baseline wins 47, candidate wins 32
  (exact_null_p=0.1147). Effect -1.2235 accuracy points (identical point
  estimate to Unit 1, deterministic given the same features/folds), se
  0.9754, interval [-3.2235, 0.5738], probability_positive **0.10125**
  (Unit 1: 0.09425, different bootstrap seed, same conclusion). Same look
  as Unit 1's still-unrecorded command below, not a new cell, not
  separately drafted to avoid double-counting.
- Line movement toward the pick (new cell, not run in Unit 1): decisive 19
  games, baseline wins 12, candidate wins 7. Effect -0.0227 pts/game
  (season-block), interval [-0.0455, 0.0], probability_positive **0.0**.
  Classification drafted **unresolved_below_power**: the interval's upper
  edge sits exactly at zero (not strictly negative) and only 19 games are
  decisive; per the `line-move-regrade-legacy` precedent, a one-sided
  interval alone does not self-classify as refuted, that call is for the
  root since no mechanism is named (AGENTS.md).

Per-fold betas (both candidates, all 5 seasons): `news_trigger_value_shift`
(candidate A) is negative in every fold (-0.071 to -0.196, mean ~-0.12).
`news_trigger_value_shift_v2` (candidate B) is small and sign-unstable
across folds: 2020 -0.0049, 2021 +0.0119, 2022 +0.0099, 2023 +0.0235, 2024
+0.0360, flips sign after 2020 and drifts upward, unlike candidate A's
consistently negative fold coefficients.

4 looks this unit (2 candidates x {accuracy, line-move}), on top of Unit 1's
3 (accuracy, Brier, log-loss).

Decision-relevant conclusion: neither the severity-only nor the per-player
role-share term shows a resolved improvement over the served 4-term
baseline on accuracy or on line movement, 2020-2024. Candidate B's
fold-unstable coefficient argues against the role-share design being ready
to fold into the served path even if a future sample resolved the accuracy
question.

Drafted `weak-signals record` commands (NOT run, registry write is out of
scope here):
```
nfl-ats weak-signals record --name news_trigger_resolution_line_move --description "news_trigger_value_shift (Unit 1 severity-only term) graded on line movement toward the pick (market_move_toward_home, restricted to market_move_available==1 rows) instead of accuracy, same LOSO 4-term baseline, seasons 2020-2024." --source artifacts/news_trigger_player_value/20260925T192146Z/summary.json --effect -0.022727272727272728 --effect-units ats_points --classification unresolved_below_power --league nfl --season-start 2020 --season-end 2024 --interval-low -0.045454545454545456 --interval-high 0.0 --probability-positive 0.0 --sample-games 528 --sample-blocks 2 --classification-evidence "Season-block bootstrap interval [-0.0455,0.0000], upper edge exactly zero not strictly negative; only 19 decisive games (7 candidate vs 12 baseline). Not self-classified as wrong_sign_resolved: no mechanism named per AGENTS.md, and a wholly-one-sided small-sample interval alone is not an admissible closing ground per the line-move-regrade-legacy precedent (drafted unresolved_below_power there too under the same pattern)." --category health --plain-summary "Once a questionable player's status resolves, the market barely moves at all, and on this small sample it moves slightly against the pick more often than with it, too few decisive games to call this settled."

nfl-ats weak-signals record --name news_trigger_role_share_cover_probability --description "news_trigger_value_shift_v2 (per-player trailing role-share value, channel-split skill/defense value-per-role-share over each resolved questionable player's own EWM offense/defense snap share, role_span=8) added as a 5th fitted term to the served 4-term LOSO logistic, seasons 2020-2024, paired against the same 4-term baseline used in Unit 1, same games." --source artifacts/news_trigger_player_value/20260925T192146Z/summary.json --effect -0.9787928221859706 --effect-units accuracy_points --classification unresolved_below_power --league nfl --season-start 2020 --season-end 2024 --standard-error 0.6416186486652954 --interval-low -2.3239453732503885 --interval-high 0.08518250954991828 --probability-positive 0.042 --sample-games 1226 --sample-blocks 89 --classification-evidence "Bootstrap probability_positive=0.042 for candidate beating baseline on accuracy (2000 draws, season/week blocks), interval [-2.324,0.085] touches the positive side so not wholly on the wrong side; exact_null_p=0.0652 on 36 decisive games (12 candidate wins vs 24 baseline wins). Brier/log-loss improvements near zero (P+ 0.182/0.184, no signal). Per-fold coefficient sign-unstable (negative 2020, positive 2021-2024), unlike candidate A's consistently negative folds. Sign not reversed with the whole interval on the wrong side, no positive control run, so unresolved_below_power." --category health --plain-summary "Weighting each hurt player's news by their own role on the team, instead of a flat team-average, still does not clearly help or hurt picks over 2020-2024, and the effect flips direction season to season."

nfl-ats weak-signals record --name news_trigger_role_share_line_move --description "news_trigger_value_shift_v2 (per-player role-share term) graded on line movement toward the pick instead of accuracy, same 4-term LOSO baseline, seasons 2020-2024." --source artifacts/news_trigger_player_value/20260925T192146Z/summary.json --effect 0.001893939393939394 --effect-units ats_points --classification unresolved_below_power --league nfl --season-start 2020 --season-end 2024 --interval-low -0.015151515151515152 --interval-high 0.01893939393939394 --probability-positive 0.7435 --sample-games 528 --sample-blocks 2 --classification-evidence "Season-block bootstrap interval [-0.0152,0.0189] crosses zero, probability_positive 0.7435; decisive record tied 8-8 on 16 games, indistinguishable from no effect." --category health --plain-summary "Weighting hurt players by their own role does not move the betting line toward or away from the pick in any detectable direction."
```

## Next
1. Unit 1's accuracy record command (above, `news_trigger_resolution_cover_probability`)
   and the 3 commands just above (candidate A line-move, candidate B
   accuracy, candidate B line-move) all still need the root to run them
   before any write-up calls these results settled. Candidate A's accuracy
   look is intentionally not re-drafted under a second name (same construct
   as Unit 1, would double-count one look under two registry names).
2. Neither candidate is ready to fold into the served path: both
   unresolved_below_power on accuracy, both coin-flip-or-worse on line
   movement, and candidate B's coefficient sign-flips across folds. No
   design change is indicated without a larger sample or a named mechanism.
3. MKT-08's prospective dispatch path (top of this lane) is unaffected by
   these historical units and still waits on real scheduled scans.
