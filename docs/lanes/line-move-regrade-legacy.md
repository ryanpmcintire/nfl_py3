# Line-move regrade of legacy (pre-line-move-yardstick) registry families

## State (2026-09-23, session 3 - BATCH 2 IN PROGRESS, hit 50-tool-call cap mid-edit)
Batch 2 selection is DONE and predeclared (8 terms, same rule as batch 1,
continued down the same ranked list). Code changes to
`scripts/line_move_regrade_legacy.py` are MOSTLY done but **NOT YET
ruff-clean and NOT YET RUN**. Do not re-derive the selection; do not
re-read the whole registry; resume exactly at "Next" below.

### The 8 batch-2 terms (selected, ratios from the SAME prior accuracy-points
ranking rule as batch 1's RATIO_TABLE; full derivation and every exclusion
reason in `Tried` below)
1. `interim_playcaller_first_game_back_on_production` 0.979 offfield/coach —
   `interim_playcaller_first_game_back_overlay.games_after_playcaller_change_flag_by_game`
   + `load_counted_playcaller_change_events(REPO/"data")`
2. `xlg06_rookie_prior_surplus_tilt_on_production` 0.955 onfield/roster —
   `rookie_prior_surplus_tilt_overlay.rookie_prior_surplus_flags(REPO, games)`
   (loads `scripts/xlg06_rookie_priors_screen.py` as a screen module, same
   pattern as roof_state; loops per season/week — WATCH RUNTIME, this is the
   heaviest of the 8, may need backgrounding)
3. `forecast_cold_visitor_tilt_on_production` 0.847 environment/weather —
   `forecast_cold_visitor_tilt_overlay.forecast_cold_visitor_flag_by_game`
   (reuses the same `FORECAST_ARCHIVE` parquet as precip; also needs
   `schedule["temp"]` for team climate history — NOT explicitly confirmed
   present in `default_schedule()` this session, check on first run)
4. `ats_streak_regress_on_production` 0.835 schedule —
   `schedule_flag_features.derive_ats_streak_regress_features`
5. `post_bye_new_playcaller_back_on_production` 0.835 offfield/coach —
   `post_bye_new_playcaller_back_overlay.post_bye_new_oc_flag_by_game` +
   `load_coordinator_history(REPO/"data")`
6. `home_thursday_on_production` 0.719 schedule —
   `schedule_flag_features.derive_home_thursday_features`
7. `low_total_div_home_dog_on_production` 0.712 schedule — built inline in
   the new `add_low_total_div_home_dog_term` from schedule's own
   `div_game`/`total_line`/`spread_line`/`game_type` (mirrors
   `low_total_div_home_dog_challenger.py`'s eligibility condition; term is
   one-directional -1.0 when eligible, matching that challenger always
   fading toward away, never toward home)
8. `division_revenge_tilt_on_production` 0.573 onfield —
   `division_revenge_tilt_overlay.division_revenge_side_by_game`

Sign convention for the 4 home/away-signed terms (playcaller_change,
post_bye_new_oc, division_revenge — all +1 home_flagged/-1 away_flagged/0
else) was READ directly off each overlay's own merge logic except
playcaller_change and post_bye_new_oc, where the +1/-1 assignment was
INFERRED from the `BackFlip`/`backed_team` dataclass naming pattern (not
read from the `apply_*_overlay` flip-mask lines themselves) — if either
term's `variant_fold_coefficients` come out with an implausible sign,
re-derive from `apply_interim_playcaller_first_game_back_overlay` /
`apply_post_bye_new_playcaller_back_overlay`'s flip_mask logic before
trusting the number.

## Goal
Re-grade Tuesday-knowable legacy registry families on the line-move yardstick
(finer than accuracy per docs/lanes/positive-control-power.md), then, for the
one standout (roof_state_predicted_open), run an out-of-sample replication on
2011-2019 to check whether the 2020-2025 reading holds up.

## State (2026-09-23, session 2 - BOTH UNITS COMPLETE)
**Unit 1** (8 legacy terms, 2020-2025, n=1503 each): ran successfully.
Results: `artifacts/line_move_regrade_legacy/20260923T220145Z/results.json`.
All 8 terms' `error` field is null (no failed builders). Summary (line-move
mean pts, season-block CI, season P+ / week P+, accuracy decisive record):
- tank_zone_fade_tilt: -0.0143, [-0.0310,+0.0024], P+ .042/.256, 56-44
- bye_edge_fade: -0.0086, [-0.0362,+0.0142], P+ .271/.267, 33-33
- **roof_state_predicted_open: +0.0150, [+0.0029,+0.0312], P+ .9975/.827,
  17-26** (only term with a wholly-positive season-block interval; week-block
  crosses zero; accuracy companion decisive record is net negative)
- precip_high_total_tilt: -0.0103, [-0.0306,+0.0069], P+ .151/.143, 12-15
- week1_dog: +0.0040, [-0.0046,+0.0126], P+ .796/.775, 9-11
- interim_hc_first_game_tilt: +0.0017, [-0.0050,+0.0114], P+ .603/.647, 4-5
- division_dog: -0.0093, [-0.0560,+0.0326], P+ .354/.306, 48-49
- deadline_integration_drag: -0.0073, [-0.0186,0.0000], P+ .000/.020, 5-7

**Unit 2** (roof_state_predicted_open OOS replication, 2011-2019, SBR proxy
open/close lines): ran successfully. New script
`scripts/roof_state_line_move_replication.py` (ruff-clean, no --fix).
Results: `artifacts/roof_state_line_move_replication/20260923T220802Z/results.json`.

Key facts established before fitting:
- 2011-2019 is exactly the SBR-proxy-warm-up-scorable window (500-game floor;
  2009-2010 score zero weeks) — confirmed in docs/proxy_opener_replication.md,
  reused here rather than re-derived.
- `artifacts/extended_fit_population/20260923T205910Z/population.parquet`'s
  2011-2019 rows already carry `opener_source=sbr_proxy_discrete`: their
  `model_logit`/`home_covered` are already built and settled against the SBR
  proxy open, so this replication's line-move grade (built from
  `data/processed/sbr_odds.parquet` `close_home_spread - open_home_spread`)
  and its accuracy companion are scored against the same instrument.
- **Forecast-archive pre-2020 check (measured)**: the Tuesday-noon-cutoff
  weather forecast archive does NOT exist pre-2020 (docs/forecast_archive_build.md:
  the `tuesday_noon` cutoff's MOS model archive start measured at 2020-07-12,
  confirmed absent 2015-09-01 and 2009-09-01). The `pool_decision` cutoff
  archive (`data/raw/forecast_archive/pool_decision_2009_2025/forecasts.parquet`,
  cutoff = min(kickoff, Sunday 16:00 ET)) DOES cover 2011-2019 (fetch_status
  'ok' for ~250/season) and is what `roof_state_screen.build_prediction_table()`
  already used for every season including the original 2020-2025 result, so
  this replication reuses that same function unchanged, per the fallback
  instruction. **Caveat that applies to BOTH the original result and this
  replication equally, not newly introduced**: pool_decision is a near-kickoff
  cutoff, not Tuesday-noon, so `predicted_open` is not demonstrated
  Tuesday-actionable in either measurement.

**Replication result**: paired_games=2231, roof_state_term_nonzero_rate=1.21%
(vs original 1.86%). Line-move toward pick: mean **-0.00224** pts, season-block
95% **[-0.0185, +0.0139]**, season P+ **0.386**; week-block 95%
[-0.0300, +0.0250], week P+ 0.4285. **Sign flips negative and both intervals
cross zero** — does not replicate the original's positive season-block
reading. Accuracy companion: mean -0.0009, decisive record **30-32** (near
coin flip, net negative), vs original's 17-26 (also net negative).

**Multiplicity-adjusted reading of the ORIGINAL 8-look result** (computed in
the same script, `multiplicity_adjusted_original_result` block): the original
season-block P+ 0.9975 implies a two-sided p=0.005; Bonferroni-adjusted across
8 looks = **0.040**, Sidak-adjusted = **0.0393** — both barely under 0.05, i.e.
marginal even before the failed replication. The original's OWN week-block
companion (P+ 0.827, implied p=0.346) is fully washed out by multiplicity:
Bonferroni = 1.0, Sidak = 0.967.

**Implication (inferred, stated plainly)**: the original roof_state
season-block reading was the best of 8 predeclared looks, survives Bonferroni/
Sidak only marginally (~0.04), fails entirely on its own week-block companion
even before adjustment, and does not replicate out-of-sample on an independent
9-season window with an independently-sourced line archive (sign flips,
P+ drops from .9975 to .386, accuracy record stays net negative). Per AGENTS.md
neither admissible closing ground applies to either result (neither interval
sits wholly on the wrong side of zero, no positive control was run) — the
correct classification for BOTH remains `unresolved_below_power`, not
`wrong_sign_resolved` and not a promotion. This is a below-power negative
signal, not evidence to serve or to declare a mechanism.

## Tried
Full Unit-1 selection derivation (ranking method, exclusions, builder
provenance) is preserved in git history of this file
(`git log -- docs/lanes/line-move-regrade-legacy.md`, commit before this
session) — not restated here to keep this file under one page; nothing there
needs to be redone.

**Batch-2 selection derivation (session 3)**: re-ran the same ranking
(`registry/weak_signals.json`, classification=unresolved_below_power,
effect_units=accuracy_points, category in
schedule/environment/health/offfield/onfield, one entry per family, ranked
by max(|effect|/standard_error)) via scratch scripts in the scratchpad dir
(not repo files, regenerate if needed — query logic: same filter as batch
1's documented rule). Confirmed batch 1's 8 ranks (8, 11, 17, 29, 36, 38,
41, 53 in the raw ranked list) match exactly. Walked further down applying
the SAME documented exclusions (CFB-only, referee/crew, in-week-injury
health, composite/pooled-atlas, ablations of served composition members)
plus one clarification found this session: `player_arrests_back_side_policy`
(ratio 2.163/1.566, ranks 9/30) is itself a LIVE served composition member
(confirmed via its own `overlay_leave_one_out_2026_08_26` LOO-ablation
registry entry existing), not a legacy accuracy-only family — excluded on
that basis, consistent with the "ablations of composition members" rule.
`apm_unit_feature.py` (ratio 1.590, rank 27) was inspected directly
(`fit_unit_ratings` runs a play-by-play Ridge regression fitting adjusted
plus-minus team ratings) and excluded as a fitted team-rating pipeline, same
exclusion class as `graph_ratings_v2_team_stat`. `rookie_priors`/
`rookie_priors_cover_rate`/`rookie_priors_per_season` (xlg06 family, ranks
5/51/73) use a "screen"-module pattern
(`scripts/xlg06_rookie_priors_screen.py`) structurally identical to
`roof_state_screen` (already accepted in batch 1) — accepted, using the
full-2020-2025-window variant (rank 73, ratio 0.955) rather than the
2024-only cut (rank 5) or the differently-constructed `_cover_rate` variant
(rank 51, no confirmed matching builder). `fluview_*` families skipped
again (Tuesday-safety still unconfirmed, same caveat as batch 1's Open
section). Confirmed via `grep ^def` that each of the 8 selected terms has an
existing, real builder function in `src/nfl_ats` (no fresh construction);
confirmed `rain_on_grass_dog_on_production` (ratio 0.567) and `coach_fade`
(ratio 0.520/0.513) also have confirmed builders
(`rain_on_grass_dog_challenger.py`, `coach_fade_overlay.py`) and are the
next-best deprioritized backups if any of the 8 fails to rebuild — do not
substitute silently, per the task's instruction; record the failure and
report 7 instead if one of the 8 errors and there is no time to validate a
backup.

Code changes made to `scripts/line_move_regrade_legacy.py` this session
(all present in the file on disk right now):
- Added `import argparse`.
- Added imports for all 8 new builders (division_revenge_tilt_overlay,
  forecast_cold_visitor_tilt_overlay, interim_playcaller_first_game_back_overlay,
  post_bye_new_playcaller_back_overlay, rookie_prior_surplus_tilt_overlay,
  plus 4 more names pulled into the existing schedule_flag_features import).
  Import block was reordered/merged once already to fix ruff's I001
  (un-sorted imports) — this fix IS applied and should be ruff-clean now.
- Added `LOW_TOTAL_MAX = 42.0` module constant (mirrors
  `low_total_div_home_dog_challenger.LOW_TOTAL_MAX`, not imported directly
  since that module's flag function needs a full predictions frame with
  `home_cover_probability`, which this script doesn't build until variant
  scoring — term is built inline from schedule instead).
- Added `SELECTION_RULE_BATCH2` and `RATIO_TABLE_BATCH2` constants (after
  `RATIO_TABLE`).
- Added all 8 `add_*_term` builder functions (after `add_deadline_drag_term`,
  before `loso`): `add_playcaller_change_term`, `add_rookie_priors_term`,
  `add_forecast_cold_visitor_term`, `add_ats_streak_regress_term`,
  `add_post_bye_new_oc_term`, `add_home_thursday_term`,
  `add_low_total_div_home_dog_term`, `add_division_revenge_term`.
- Added `TERM_DECLARATIONS_BATCH2` tuple (after `TERM_DECLARATIONS`, batch 1
  definitions untouched).
- Added `--batch {1,2}` argparse flag to `main()`; selects
  `term_declarations`/`selection_rule`/`ratio_table` accordingly; `results`
  dict now includes a `"batch"` key; default (`--batch` omitted) still runs
  batch 1 unchanged — batch-1 behavior is preserved.

**Ruff status at cap time**: `ruff check scripts/line_move_regrade_legacy.py`
(no --fix) was run once after the import-block edits and returned 3 errors:
1 x I001 (import order) — FIXED by the import-block rewrite above, should
now be clean; 2 x E501 (line too long, >100 chars) on the `def
add_forecast_cold_visitor_term(...)` and `def
add_low_total_div_home_dog_term(...)` signature lines (both were written as
single-line signatures exceeding 100 cols). A fix for the FIRST one
(wrapping `add_forecast_cold_visitor_term`'s signature onto 3 lines) was
IN FLIGHT when the tool cap hit — the Edit call may or may not have applied;
**check the file before re-editing** to avoid a duplicate/malformed edit.
The SECOND (`add_low_total_div_home_dog_term`) signature has NOT been
touched yet.

## Next

- 2026-09-23 root: batch 2 ran (artifacts/line_move_regrade_legacy/20260923T221920Z), 7 of 8 recorded (registry 7,052). Resolved wrong signs on line movement, season- and week-block intervals wholly negative: rookie_priors -0.046 [-0.062,-0.027] (decisive 32-56) and low_total_div_home_dog -0.048 [-0.075,-0.020] (24-38); the latter is also a live prospective challenger (src/nfl_ats/low_total_div_home_dog_challenger.py), so its paired tracking should be read with this historical wrong sign. division_revenge_tilt failed to rebuild (KeyError season); fix its builder join and grade it alone. Others unresolved.
**Batch 2 (do this first, fresh agent, new 50-call budget):**
1. Read `scripts/line_move_regrade_legacy.py` around `add_forecast_cold_visitor_term`
   and `add_low_total_div_home_dog_term` to see current state (the first
   signature fix may already be applied).
2. Fix both E501s by wrapping each `def add_..._term(population: pd.DataFrame,
   schedule: pd.DataFrame) -> pd.DataFrame:` onto 3 lines (open paren, two
   params each on own line, closing paren + return type on its own line) —
   same style already used elsewhere in the file.
3. Run `.tools/uv.exe run ruff check scripts/line_move_regrade_legacy.py`
   (NO --fix) until clean (0 errors). Fix anything else it reports; do not
   change batch-1 code paths.
4. Run once in the foreground with a timeout (expect longer than batch 1 —
   the rookie_priors term loops per season/week; if it looks like it will
   exceed a couple minutes, background it per the harness's normal handling,
   do not kill it):
   `.tools/uv.exe run python scripts/line_move_regrade_legacy.py --batch 2`
5. Read the resulting `artifacts/line_move_regrade_legacy/<ts>/results.json`.
   Check every term's `error` key FIRST. For any term with an error: record
   the exact error, do NOT substitute a 9th term, and consider whether
   `rain_on_grass_dog_on_production` (ratio 0.567,
   `rain_on_grass_dog_challenger.py`) or `coach_fade` (ratio 0.520,
   `coach_fade_overlay.py`) — both confirmed-builder backups named in
   `Tried` above — should replace it (orchestrator adjudicates, do not
   auto-substitute).
6. For clean terms, pull `line_move_toward_pick_cell` (mean_points,
   season_block_interval_low/high, season_block_probability_positive,
   week_block_probability_positive) and `accuracy_companion_cell`'s decisive
   record, same fields batch 1 used.
7. Report the 8 (or fewer, if any failed) line-move deltas with season-block
   interval and P+ to the orchestrator, plus draft
   `nfl-ats weak-signals record` commands for each (same flag pattern as the
   two commands already below: `--effect-units ats_points --classification
   unresolved_below_power --league nfl --season-start 2020 --season-end 2025
   --sample-blocks 6 --category <see per-term category in RATIO_TABLE_BATCH2>
   --plain-summary "..."`). Do NOT run them without orchestrator
   authorization.
8. Once reported: this lane stays open (do not move to done/) until the
   orchestrator has reviewed batch 2's numbers, since unit 1/unit 2's own
   record commands (below) are also still unexecuted.

**Units 1 and 2 record commands** (orchestrator-authorized only; not run this
session):

1. Original (8-look) roof term, informational only if not already recorded
   elsewhere — skip if this exact number was already recorded by a prior
   session:
```
nfl-ats weak-signals record --name roof_state_predicted_open_line_move_regrade_legacy \
  --description "Roof-state-predicted-open term added to Tuesday-knowable base, line-move-toward-pick grade, 2020-2025 LOSO" \
  --source artifacts/line_move_regrade_legacy/20260923T220145Z/results.json \
  --effect 0.01497 --effect-units ats_points --classification unresolved_below_power \
  --league nfl --season-start 2020 --season-end 2025 \
  --interval-low 0.002903 --interval-high 0.031162 --probability-positive 0.9975 \
  --sample-games 1503 --sample-blocks 6 --category environment \
  --plain-summary "Best of 8 legacy looks on line-move toward pick; season-block interval wholly positive but week-block crosses zero (P+ 0.827) and accuracy companion decisive record is 17-26."
```

2. Replication (2011-2019 SBR-proxy OOS), the primary new result:
```
nfl-ats weak-signals record --name roof_state_predicted_open_line_move_replication_2011_2019 \
  --description "OOS replication of roof-state-predicted-open on 2011-2019 using SBR proxy open/close lines, line-move-toward-pick grade" \
  --source artifacts/roof_state_line_move_replication/20260923T220802Z/results.json \
  --effect -0.002241 --effect-units ats_points --classification unresolved_below_power \
  --league nfl --season-start 2011 --season-end 2019 \
  --interval-low -0.018494 --interval-high 0.013883 --probability-positive 0.386 \
  --sample-games 2231 --sample-blocks 9 --category environment \
  --plain-summary "Out-of-sample replication on an independent 9-season window with an independent (SBR proxy) line archive; sign flips negative, both season- and week-block intervals cross zero, accuracy companion decisive record 30-32. Does not replicate the 2020-2025 reading; multiplicity-adjusted original season-block p is only marginal (Bonferroni/Sidak ~0.04) and its own week-block companion is fully washed out (Bonferroni/Sidak ~0.97-1.0)."
```

Both commands' every numeric flag is read directly from the two results.json
artifacts above (no hand-typed derived numbers beyond the source values).
Orchestrator should verify by reading both JSON files before running.

Once recorded: move this lane to `docs/lanes/done/`.

## Open
None outstanding for this lane's scope. If a future session wants a
split-half reliability check on the replication (per AGENTS.md's second
admissible closing ground), that would need an 18th data source or a
within-2011-2019 split and is not started here.
