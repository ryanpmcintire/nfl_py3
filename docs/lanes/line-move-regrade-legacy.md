# Line-move regrade of legacy (pre-line-move-yardstick) registry families

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

## Next
Two record commands to run (orchestrator-authorized only; not run this
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
