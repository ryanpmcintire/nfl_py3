# Line-move regrade of legacy (pre-line-move-yardstick) registry families

## Goal
Re-grade Tuesday-knowable legacy registry families on the line-move yardstick
(finer than accuracy per docs/lanes/positive-control-power.md).

## State (2026-09-23, session 6 - batch 4 added and run, ruff clean, record
commands drafted, NOT run)

**Session 6 summary**: added `TERM_DECLARATIONS_BATCH4` (`--batch 4`) for
the 2 families batch 3 flagged but skipped -- `special_teams_return_top_quartile`
and `hc_year_one_fade`. `ruff check` (no --fix) passes clean. Ran `--batch 4`
twice (first run hit a `data_root` path bug in the special_teams builder,
fixed, second run clean, both terms `error: null`, 1503 paired_games each).
Results in Batch 4 results below. `hc_year_one_fade` correctly uses the
week<=8 restriction the registry family requires (batch 3's `coach_fade`
term did not have this restriction, so it is a different, non-duplicate
construction). Record commands drafted, NOT executed (no registry-write
authorization given).

## State (2026-09-23, session 5 - all code written, ruff clean, both runs
executed successfully, record commands drafted, NOT run)

**Session 5 summary**: finished the 3 pending imports (rain_on_grass_dog_challenger,
schedule_flag_features DOME_SHOOTOUT_COLUMN/default_opener_lines/
derive_dome_shootout_favorite_features, transaction_flag_features
SUSPENSION_RETURN_RUST_COLUMN/attach_suspension_return_rust_features), added
the 4 batch-3 builder functions, `SELECTION_RULE_BATCH3`/`RATIO_TABLE_BATCH3`,
`TERM_DECLARATIONS_BATCH3`, `--batch` now `choices=(1,2,3)`, added `--only`
filter. `ruff check` (no --fix) passes clean. Ran
`--batch 2 --only division_revenge_tilt` (confirms the session-4 merge fix —
`error: null`) and `--batch 3` (all 4 terms `error: null`, 1503 paired_games
each, matching base population). Results below. Record commands drafted at
end of Next, NOT executed (no registry-write authorization given).

**division_revenge_tilt bug FOUND, FIXED, AND CONFIRMED (session 5).** Root
cause: `division_revenge_side_by_game(schedule)` returns its own `season`
column; `add_division_revenge_term` merged the WHOLE flags frame (incl.
`season`) onto `population` (which already has `season`), so pandas silently
created `season_x`/`season_y` instead of erroring at merge time. Fix: select
only `["game_id", "revenge_home", "revenge_away"]` before merging, same
pattern every other builder in the file uses. `--batch 2 --only
division_revenge_tilt` now runs with `error: null` — results in Results below.

**Batch 3 candidate research DONE (re-derived full ranking from
`registry/weak_signals.json`), only 4 of 8 confirmed live-buildable within
budget — all 4 coded and run in session 5.** Full reasoning, ranks, and
exclusions below in Tried. Do not re-derive.

## Tried
Unit 1/2 (roof_state, 8-term batch 1) and batch-1/2 selection derivations:
preserved in git history of this file (`git log -- docs/lanes/line-move-regrade-legacy.md`,
commits before this session) — not restated, nothing there needs redoing.

**Batch-3 ranking re-derivation (session 4)**: re-ran the SAME rule text as
`SELECTION_RULE_BATCH2` (classification=unresolved_below_power,
effect_units=accuracy_points, category in
schedule/environment/health/offfield/onfield, one entry per family by max
ratio) via a scratch script (not a repo file), but this time computed
`standard_error` with the documented fallback `(interval_high-interval_low)/
(2*1.96)` when the field is null — the earlier sessions' 81-family list
apparently only used entries with a real non-null `standard_error`; applying
the fallback yields **388** distinct families, a much bigger and differently
ordered list. `division_revenge_tilt__week_in_season` (batch 2's own pick) is
rank **253** at ratio **0.573** in this fuller list.

**Open discrepancy, NOT resolved, flag for orchestrator**: several
high-ratio families rank well above anything batch 1 (topped at 2.218) or
batch 2 (topped at 0.573) actually selected —
`special_teams_return_top_quartile` (ratio 1.694, onfield, confirmed this
session to have a real live builder
`special_teams_return_flag_by_game_fail_open` in
`special_teams_return_tilt_overlay.py`, no LOO-ablation composition-member
entry found) and `hc_year_one_fade` (ratio 1.495, offfield — confirmed this
session to be built by the SAME module as the `coach_fade` backup,
`coach_fade_overlay.py`, `CHALLENGER_ID = "hc_year_one_fade_overlay"`,
function `year_one_by_game`). Neither was picked by batch 1/2 for a reason
not re-derivable this session (possibly the narrower raw list, possibly an
unrecorded exclusion). This session deliberately did NOT include either —
used the `coach_fade` identity/rank already fixed by the prior session's own
backup designation (ratio ~0.52) rather than swapping in the higher-ranked
`hc_year_one_fade` entry for the same module, and left
`special_teams_return_top_quartile` out entirely pending review. Do not
silently resolve this either way without orchestrator input.

Walking down from rank 253 (ratio 0.573) applying the SAME exclusion classes
as batch 1/2's `SELECTION_RULE_BATCH2` text: referee/crew (`referee_battery_*`,
`crew_second_meeting_*`, `penalty_crew_*`/`penalty_discipline`/`penalty_rate_*`);
health category excluded as a block (consistent with zero health entries in
either prior batch's actual `RATIO_TABLE` despite several qualifying by
ratio); composite/pooled-atlas clusters (`*_battery`, `weather_interactions_*`,
`weather_followup_*`, `forecast_weather_kn_*`) excluded under "not one
rebuildable column" — this also explains why those clusters' much-higher
ratios never appeared in batch 1/2 despite being available; fitted
team-rating/team-style pipelines (`team_style_*`, `graph_*`, `apm_unit_*`,
same class as the already-excluded `apm_unit_feature.py`); era-scope
mismatches (`*_pre2011`, `*_pre2018` — population is ~2020-2025, n=1503,
these have near-zero overlap); and **no confirmed live builder** — a large
fraction of remaining high-ratio legacy families have NO matching `def` or
module anywhere in current `src/nfl_ats` because their source files were
deleted in the "Repository cut" commit `b7ed31d` (469,660 -> 216,083 Python
lines). Confirmed via `git log -- src/nfl_ats/<file>.py` showing
history-only files, e.g. `backup_tenure_flag_features.py` (backs
`backup_tenure_gap_on_production`, ratio would've been high) is gone.
Same "no builder found this session" outcome for: `kicker_change_underdog`,
`divisional_rematch_blowout_winner_fade`, `venue_milestone_new_stadium_debut`
(NOT the same as `derive_new_stadium_home_features`, which exists but has a
different name/semantics — did not substitute), `ol_acute_overhaul_fade`,
`qb_age_rookie_late_improvement`, `redzone_reversion_c2_rz_under_rebound`,
`surface_familiarity_*`, `altitude_deficit_4000ft_era_2018_2025`,
`travel_rest_eastbound_multizone`, `pick_conditioned_rest_mismatch_pre2018`,
`bye_overvaluation` (all cuts), `special_teams_punt_net_bottom_quartile` /
`special_teams_composite_edge_top_quartile` (different metrics than the one
confirmed module covers).

**The 4 confirmed batch-3 candidates, in rank order, each has a live
Tuesday-safe standalone builder already in `src/nfl_ats`:**
1. `suspension_return_rust_on_production` 0.572 offfield —
   `transaction_flag_features.attach_suspension_return_rust_features(features,
   schedule=schedule)` -> `SUSPENSION_RETURN_RUST_COLUMN`, already signed via
   `_attach_qualifying_sides` (-1.0 home_qualifies / +1.0 away_qualifies /
   0.0 else). Use the SAME `try/except DataContractError -> fill 0.0` pattern
   already in `add_deadline_drag_term` (transaction data may not cover every
   span).
2. `rain_on_grass_dog_on_production` 0.567 environment —
   `rain_on_grass_dog_challenger.rain_on_grass_flag_by_game(schedule,
   forecasts)` (forecasts = `pd.read_parquet(FORECAST_ARCHIVE)`, same archive
   precip/forecast_cold_visitor already use) returns one boolean
   `rain_on_grass_flag`, NOT pre-split by side. Build the term the same way
   batch 2's `add_low_total_div_home_dog_term` used `schedule["spread_line"]`
   directly: +1.0 when flag true AND home is dog (spread_line<0), -1.0 when
   flag true AND away is dog (spread_line>0), 0.0 else, `game_type=="REG"`
   only. **Not verified against `apply_rain_on_grass_dog_tilt_overlay`'s own
   flip-direction code** (this session read only its eligibility setup, not
   the final sign assignment past line ~140 of
   `rain_on_grass_dog_challenger.py`) — LOSO fit absorbs a wrong sign, but if
   the fitted coefficient looks implausible, re-check that function before
   trusting it.
3. `dome_shootout_favorite_on_production` 0.525 schedule —
   `schedule_flag_features.derive_dome_shootout_favorite_features(schedule,
   default_opener_lines(schedule))` -> `dome_shootout_favorite_flag`, already
   signed (+1.0 home favorite / -1.0 away favorite / 0.0 else) by
   `oracle_derive_dome_shootout_favorite_features` under the hood (Tuesday-
   safe via `decision_time_roof_schedule`, not the oracle roof). **Smoke-
   tested live this session** standalone: ran in ~15s, 4902 rows, 31
   home-favorite / 40 away-favorite / 4831 zero — works end to end.
4. `coach_fade` 0.520/0.513 onfield — `coach_fade_overlay.year_one_by_game(
   schedule)` returns `game_id, season, year_one_home, year_one_away` — MUST
   select only `["game_id","year_one_home","year_one_away"]` before merging
   (it also returns `season`; same collision bug as division_revenge_tilt
   would reappear if not handled). Term: +1.0 home flagged / -1.0 away
   flagged / 0.0 else (same pattern as tank_zone/bye_edge/division_revenge).

Only 4 of the requested 8 confirmed within budget; ranks below coach_fade
(<0.513) have not been examined at all this session.

**Code state on disk right now**: all of the above is APPLIED and confirmed
by a passing `ruff check` and two successful runs (session 5) — nothing
pending from session 4's interrupted edit.

## Batch 4 (session 6, predeclared BEFORE running)

The 2 families the batch-3 ranking flagged but skipped:
`special_teams_return_top_quartile` (ratio 1.694, onfield) and
`hc_year_one_fade` (ratio 1.495, offfield). Both have confirmed live
Tuesday-safe standalone builders in `src/nfl_ats`, used as-is (read, not
re-derived):

1. `special_teams_return_top_quartile` ->
   `special_teams_return_tilt_overlay.special_teams_return_flag_by_game_fail_open(REPO, schedule)`
   (local data confirmed present:
   `data/raw/special_teams/20260819T232400Z/team_season.parquet`) ->
   `home_return_top_quartile`/`away_return_top_quartile`. Term: +1.0 home
   flagged (and not both flagged), -1.0 away flagged (and not both), 0.0
   else -- mirrors `apply_special_teams_return_tilt_overlay`'s own
   both-flagged exclusion.
2. `hc_year_one_fade` -> SAME builder batch 3's `coach_fade_on_production`
   used, `coach_fade_overlay.year_one_by_game(schedule)` (`CHALLENGER_ID =
   "hc_year_one_fade_overlay"` in that module -- confirmed this is the
   correct source). **Read (`registry/weak_signals.json` line 37580) shows
   the registered family is restricted to "weeks 1-8"** (`docs/hc_year_one_fade.md`,
   effect 0.7528 accuracy_points, se 0.5036) and `year_one_by_game` itself
   carries no week filter -- the week<=8 cutoff lives only in
   `apply_coach_fade_overlay`'s `eligible` mask (`OVERLAY_WEEK_MAX = 8`).
   Batch 3's `add_coach_fade_term` applied the flag to ALL weeks, no cutoff
   -- so batch 3 did NOT actually regrade the registered `hc_year_one_fade`
   family, it graded an unrestricted variant of the same underlying flag.
   This resolves part of the open ranking discrepancy: batch 4's
   `hc_year_one_fade_on_production` term applies the same week<=8 filter as
   the registry entry (and as `apply_coach_fade_overlay`), so it is NOT a
   duplicate of batch 3's term despite sharing a builder function.

Both added as `TERM_DECLARATIONS_BATCH4` behind `--batch 4` in
`scripts/line_move_regrade_legacy.py`; batches 1-3 unchanged.

## Results (session 5, measured this session)

`division_revenge_tilt` (batch 2, merge-fix run,
`artifacts/line_move_regrade_legacy/20260923T224035Z/results.json`, 1503 paired games):
line-move mean **-0.0539 pts**, season-block interval **[-0.1149, -0.0118]**
P+ **0.00**, week-block interval **[-0.1005, -0.0200]** P+ **0.00** — both
block types entirely negative (accuracy companion -0.004 pts, P+ 0.1435).

Batch 3 (`artifacts/line_move_regrade_legacy/20260923T224045Z/results.json`,
1503 paired games each, all `error: null`):
- `suspension_return_rust_on_production`: mean **+0.0077**, season
  **[-0.0140, 0.0380]** P+ 0.648, week **[-0.0076, 0.0248]** P+ 0.808.
- `rain_on_grass_dog_on_production`: mean **+0.0057**, season
  **[-0.0038, 0.0159]** P+ 0.865, week **[-0.0027, 0.0168]** P+ 0.877.
- `dome_shootout_favorite_on_production`: mean **-0.0060**, season
  **[-0.0139, 0.0000]** P+ 0.0105, week **[-0.0162, 0.0020]** P+ 0.0755.
- `coach_fade_on_production`: mean **-0.0156**, season **[-0.0201, -0.0119]**
  P+ 0.00, week **[-0.0372, 0.0043]** P+ 0.0635 — season block wholly
  negative but week block crosses zero, so this one does NOT meet the
  wrong-sign-resolved bar on both block types; stays unresolved.

## Batch 4 results (session 6, measured this session)

`artifacts/line_move_regrade_legacy/20260923T224601Z/results.json`, both
`error: null`, 1503 paired games each (matches base population; first run
hit a `data_root` path bug in the special_teams builder -- passed `REPO`
instead of `REPO / "data"`, fixed and confirmed via a clean re-run, `ruff
check` still passes):

- `special_teams_return_top_quartile_on_production`: mean **-0.0642
  pts**, season-block **[-0.0913, -0.0359]** P+ **0.00**, week-block
  **[-0.1086, -0.0225]** P+ **0.002** -- both block types entirely
  negative (accuracy companion -0.0027 pts, P+ 0.2965). decisive record
  35-56 (91 decisive games, mean -1.060 pts); term fires on 36.8% of
  games. **This is the OPPOSITE sign from the registered family's own
  prior read** (registry `special_teams_return_top_quartile`: P+ 0.9547,
  "back them" -- predicted positive on team_covered/accuracy). On the
  line-move yardstick the market moves AWAY from the flagged side, not
  toward it.
- `hc_year_one_fade_on_production` (week<=8 only, matching the
  registry's own restriction -- NOT a duplicate of batch 3's unrestricted
  `coach_fade_on_production`): mean **-0.0043 pts**, season-block
  **[-0.0214, 0.0153]** P+ 0.32, week-block **[-0.0240, 0.0147]** P+
  0.3245 -- both intervals cross zero, unresolved. decisive record 9-14
  (23 decisive games, mean -0.283 pts); term fires on 17.6% of games.

## Next

- 2026-09-23 root: batch 4 recorded (registry 7,059). special_teams_return_top_quartile is a resolved wrong sign on line movement for this Tuesday-base variant (-0.064 [-0.091,-0.036], week block also negative, decisive 35-56), opposite its accuracy prior; not a served member (only in unserved_tilt_marginals). hc_year_one_fade unresolved. Further batches only for families with a live builder; the ranking discrepancy is documented in Open.

- 2026-09-23 root: division revenge (b2) and batch 3 recorded (registry 7,057). Division revenge is a resolved wrong sign on both blockings (-0.054 [-0.115,-0.012]). coach_fade stays unresolved (season block negative, week block crosses zero). Batch 4 (special_teams_return_top_quartile, hc_year_one_fade) run this session -- see Batch 4 results above. special_teams_return_top_quartile is a candidate wrong-sign-resolved case (both blockings wholly negative, P+ 0.00/0.002) but OPPOSITE-signed from its own registry prior (which was positive, P+ 0.9547) -- naming the mechanism is an orchestrator research call, not made this session. hc_year_one_fade (week<=8) stays unresolved (both blockings cross zero). Neither recorded yet (no registry-write authorization this session). Draft commands below are history except where noted.
1. **Orchestrator decision needed on `division_revenge_tilt`**: both season-
   and week-block intervals are wholly negative (P+ 0.00 both). Per
   AGENTS.md this is the shape of an admissible `wrong_sign_resolved` closing
   ground, but naming the mechanism is a research call this session did not
   make — draft below uses `unresolved_below_power` per the orchestrator's
   given flag template; re-draft as `wrong_sign_resolved` with a named
   mechanism and `--closing-ground` only if the orchestrator confirms.
2. Draft `nfl-ats weak-signals record` commands (NOT run, no registry-write
   authorization given this session):

```
nfl-ats weak-signals record --effect-units ats_points --classification unresolved_below_power --league nfl --season-start 2020 --season-end 2025 --sample-blocks 6 --family line_move_regrade_legacy_v2 --category onfield --plain-summary "Division-revenge tilt line-move regrade (merge-bug fixed this session): mean line move -0.054 pts/game toward the pick; season-block interval [-0.115,-0.012] P+ 0.00; week-block interval [-0.100,-0.020] P+ 0.00 -- both intervals fall entirely on the negative side."

nfl-ats weak-signals record --effect-units ats_points --classification unresolved_below_power --league nfl --season-start 2020 --season-end 2025 --sample-blocks 6 --family line_move_regrade_legacy_v3 --category offfield --plain-summary "Suspension-return-rust line-move regrade: mean +0.008 pts/game toward the pick; season-block interval [-0.014,0.038] P+ 0.648; week-block interval [-0.008,0.025] P+ 0.808."

nfl-ats weak-signals record --effect-units ats_points --classification unresolved_below_power --league nfl --season-start 2020 --season-end 2025 --sample-blocks 6 --family line_move_regrade_legacy_v3 --category environment --plain-summary "Rain-on-grass-dog line-move regrade: mean +0.006 pts/game toward the pick; season-block interval [-0.004,0.016] P+ 0.865; week-block interval [-0.003,0.017] P+ 0.877."

nfl-ats weak-signals record --effect-units ats_points --classification unresolved_below_power --league nfl --season-start 2020 --season-end 2025 --sample-blocks 6 --family line_move_regrade_legacy_v3 --category schedule --plain-summary "Dome-shootout-favorite line-move regrade: mean -0.006 pts/game toward the pick; season-block interval [-0.014,0.000] P+ 0.011; week-block interval [-0.016,0.002] P+ 0.076."

nfl-ats weak-signals record --effect-units ats_points --classification unresolved_below_power --league nfl --season-start 2020 --season-end 2025 --sample-blocks 6 --family line_move_regrade_legacy_v3 --category onfield --plain-summary "Coach-fade (year-one HC) line-move regrade: mean -0.016 pts/game toward the pick; season-block interval [-0.020,-0.012] P+ 0.00; week-block interval [-0.037,0.004] P+ 0.064 -- season block alone is wholly negative, week block crosses zero."

nfl-ats weak-signals record --effect-units ats_points --classification unresolved_below_power --league nfl --season-start 2020 --season-end 2025 --sample-blocks 6 --family line_move_regrade_legacy_v4 --category onfield --plain-summary "Special-teams-return-top-quartile line-move regrade: mean -0.064 pts/game toward the pick; season-block interval [-0.091,-0.036] P+ 0.00; week-block interval [-0.109,-0.023] P+ 0.002 -- both intervals wholly negative, OPPOSITE sign from the registered family's own prior (P+ 0.9547 positive). Drafted as unresolved_below_power pending an orchestrator call on whether this is an admissible wrong_sign_resolved closing ground with a named mechanism."

nfl-ats weak-signals record --effect-units ats_points --classification unresolved_below_power --league nfl --season-start 2020 --season-end 2025 --sample-blocks 6 --family line_move_regrade_legacy_v4 --category offfield --plain-summary "HC-year-one-fade (weeks 1-8 only, matching the registered family) line-move regrade: mean -0.004 pts/game toward the pick; season-block interval [-0.021,0.015] P+ 0.32; week-block interval [-0.024,0.015] P+ 0.32 -- both intervals cross zero."
```

3. Report the open discrepancy (special_teams_return_top_quartile /
   hc_year_one_fade ranking above 0.573 unexplained, see Tried/Open below)
   to the orchestrator as an explicit open item — do not resolve it
   unilaterally.
4. Batch 2's 7 successful terms (rookie_priors, low_total_div_home_dog,
   interim_playcaller, forecast_cold_visitor, ats_streak_regress,
   post_bye_new_oc, home_thursday) still need record commands drafted from
   `artifacts/line_move_regrade_legacy/20260923T221920Z/results.json` — not
   done in any session yet, unrelated to this session's scope.

**Units 1 and 2 record commands (still unexecuted, orchestrator-authorized
only)** — unchanged from before this session, preserved in git history of
this file if needed; re-fetch via `git log -p -- docs/lanes/line-move-regrade-legacy.md`
(the two `nfl-ats weak-signals record` blocks for
`roof_state_predicted_open_line_move_regrade_legacy` and
`roof_state_predicted_open_line_move_replication_2011_2019`) rather than
restating here to keep this file under one page.

## Open
- Batch-4 ranking discrepancy RESOLVED for `hc_year_one_fade` this session
  (batch 3's coach_fade term was an unrestricted variant, not the
  registered weeks-1-8 family; batch 4 built the correct restricted
  version, both blockings cross zero). `special_teams_return_top_quartile`
  now needs a DIFFERENT orchestrator call: both blockings are wholly
  negative on the line-move yardstick, opposite-signed from the family's
  own registry prior -- naming a mechanism (or declining to) is a research
  decision this session did not make.
- If a future session wants more than 4 batch-3 terms, ranks below
  coach_fade (<0.513) are completely unexamined.
- Split-half reliability check on the roof_state replication (AGENTS.md's
  second admissible closing ground) still not started, needs an 18th data
  source or a within-2011-2019 split.
