# Line-move regrade of legacy (pre-line-move-yardstick) registry families

## Goal
Re-grade Tuesday-knowable legacy registry families on the line-move yardstick
(finer than accuracy per docs/lanes/positive-control-power.md).

## State (2026-09-23, session 4 - hit 50-tool-call cap mid-edit, batch 2 fix
DONE but unverified, batch 3 selection DONE but code NOT written)

**division_revenge_tilt bug FOUND AND FIXED (root cause, not yet re-run).**
`division_revenge_side_by_game(schedule)` returns its own `season` column.
`add_division_revenge_term` merged the WHOLE flags frame (incl. `season`) onto
`population` (which already has `season`), so pandas silently created
`season_x`/`season_y` instead of erroring at merge time; the `KeyError:
'season'` fired later inside `loso()`/`variant_report` when they referenced
`frame["season"]`. Fix (APPLIED, on disk now) in
`scripts/line_move_regrade_legacy.py`'s `add_division_revenge_term`: changed
`flags = flags.drop_duplicates(subset="game_id")` to
`flags = flags.drop_duplicates(subset="game_id")[["game_id", "revenge_home", "revenge_away"]]`
— select only needed columns before merge, same pattern every other builder
in the file already uses. **Not yet run to confirm** — no mechanism exists
yet to grade it alone (see Next step 5).

**Batch 3 candidate research DONE (re-derived full ranking from
`registry/weak_signals.json`), only 4 of 8 confirmed live-buildable within
this session's budget.** Full reasoning, ranks, and exclusions below in
Tried. Do not re-derive; resume at Next.

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

**Code state on disk right now (verify by reading before continuing):**
- `add_division_revenge_term` fix: APPLIED.
- `from nfl_ats.coach_fade_overlay import year_one_by_game` import: APPLIED
  (inserted after the `bye_edge_fade_overlay` import, before
  `from nfl_ats.data import DataContractError`).
- A second Edit — adding `from nfl_ats.rain_on_grass_dog_challenger import
  rain_on_grass_flag_by_game`; adding `default_opener_lines` and
  `derive_dome_shootout_favorite_features` to the existing
  `schedule_flag_features` import; adding `SUSPENSION_RETURN_RUST_COLUMN`
  and `attach_suspension_return_rust_features` to the existing
  `transaction_flag_features` import — was submitted but the tool-call cap's
  PreToolUse hook blocked it before it ran. **Almost certainly NOT applied.**
  Read the file first; do not blindly re-submit (risk of duplicate/malformed
  edit if it partially landed).
- NOT done at all yet: the 4 new `add_*_term` builder functions (bodies
  fully specified above); `SELECTION_RULE_BATCH3` / `RATIO_TABLE_BATCH3`
  constants; `TERM_DECLARATIONS_BATCH3` tuple; `--batch` argparse choices
  still `(1, 2)` not `(1, 2, 3)`; no `--only <label>` filter flag exists yet
  (needed to grade division_revenge_tilt alone within batch 2). Nothing run
  this session — no new artifacts dir under
  `artifacts/line_move_regrade_legacy/` from session 4.

## Next
1. Read `scripts/line_move_regrade_legacy.py` lines ~1-65 to see exactly
   what landed from the interrupted second import Edit; apply whichever of
   the three additions (rain_on_grass import; schedule_flag_features
   `default_opener_lines`+`derive_dome_shootout_favorite_features`;
   transaction_flag_features `SUSPENSION_RETURN_RUST_COLUMN`+
   `attach_suspension_return_rust_features`) are missing — exact names given
   above, alphabetical placement matches existing style.
2. Add the 4 builder functions (specs above) after `add_division_revenge_term`,
   before `def loso`.
3. Add `SELECTION_RULE_BATCH3` (state the rule + every exclusion actually
   applied, per Tried above) and:
   `RATIO_TABLE_BATCH3 = (("suspension_return_rust_on_production", 0.572, "offfield"), ("rain_on_grass_dog_on_production", 0.567, "environment"), ("dome_shootout_favorite_on_production", 0.525, "schedule"), ("coach_fade_on_production", 0.520, "onfield"))`
   (4 entries only — document why not 8) after `RATIO_TABLE_BATCH2`.
4. Add `TERM_DECLARATIONS_BATCH3` (4 dicts: label/term_columns/builder)
   after `TERM_DECLARATIONS_BATCH2`.
5. Add a reusable way to grade one term alone: `parser.add_argument("--batch",
   type=int, choices=(1,2,3), default=1)`; `parser.add_argument("--only",
   type=str, default=None)`; after selecting `term_declarations` for the
   batch, if `args.only`: `term_declarations = tuple(d for d in
   term_declarations if d["label"] == args.only)`. Include `args.only` in
   the `results["command"]` string.
6. `.tools/uv.exe run ruff check scripts/line_move_regrade_legacy.py` (NO
   --fix) until 0 errors — watch E501 on new multi-arg def lines (wrap onto
   3 lines like existing batch-2 style) and I001 import order.
7. Run division_revenge_tilt alone first (cheap, validates the fix):
   `.tools/uv.exe run python scripts/line_move_regrade_legacy.py --batch 2 --only division_revenge_tilt`
   foreground, timeout. Confirm `error` is null; read
   `line_move_toward_pick_cell` / `accuracy_companion_cell`.
8. Run batch 3: `.tools/uv.exe run python scripts/line_move_regrade_legacy.py --batch 3`
   foreground, timeout (none of the 4 builders loop per-season like
   `rookie_priors` did, should be comparable to batch 1's runtime; background
   only if it actually exceeds ~2 minutes).
9. Read both results.json. Report each term's `line_move_toward_pick_cell`
   (mean_points, season_block_interval, season/week P+) and
   `accuracy_companion_cell` decisive record. Draft (do not run)
   `nfl-ats weak-signals record` commands per term (flag pattern: `--effect-units
   ats_points --classification unresolved_below_power --league nfl
   --season-start 2020 --season-end 2025 --sample-blocks 6 --family
   line_move_regrade_legacy_v3 --category <per RATIO_TABLE_BATCH3>
   --plain-summary "..."`). Do not run without orchestrator authorization.
10. Report the open discrepancy (special_teams_return_top_quartile /
    hc_year_one_fade ranking above 0.573 unexplained) to the orchestrator as
    an explicit open item — do not resolve it unilaterally.

**Units 1 and 2 record commands (still unexecuted, orchestrator-authorized
only)** — unchanged from before this session, preserved in git history of
this file if needed; re-fetch via `git log -p -- docs/lanes/line-move-regrade-legacy.md`
(the two `nfl-ats weak-signals record` blocks for
`roof_state_predicted_open_line_move_regrade_legacy` and
`roof_state_predicted_open_line_move_replication_2011_2019`) rather than
restating here to keep this file under one page.

Batch 2's 7 successful terms (rookie_priors, low_total_div_home_dog resolved
wrong-sign negative; interim_playcaller/forecast_cold_visitor/ats_streak_
regress/post_bye_new_oc/home_thursday unresolved) also still need their
record commands drafted from `artifacts/line_move_regrade_legacy/20260923T221920Z/results.json`
— not done in any session yet.

## Open
- The ranking discrepancy above (special_teams_return_top_quartile,
  hc_year_one_fade) needs orchestrator adjudication before any future batch
  reuses this session's ranked-list methodology.
- If a future session wants more than 4 batch-3 terms, ranks below
  coach_fade (<0.513) are completely unexamined.
- Split-half reliability check on the roof_state replication (AGENTS.md's
  second admissible closing ground) still not started, needs an 18th data
  source or a within-2011-2019 split.
