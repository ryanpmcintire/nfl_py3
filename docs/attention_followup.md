# Attention both_cold follow-up — predeclaration

Written **before** `scripts/attention_followup_screen.py` scores anything, per
AGENTS.md ("the family must be declared before the signs are seen"). Frozen
copy also written to
`<scratchpad>/agent_attention_followup/predeclaration.md`.

## What this follows up

The 2026-08-19 attention pilot
(`scripts/attention_battery_screen.py`, artifact
`artifacts/attention_battery/20260819T155949Z/results.json`, registry entries
`attention_battery_*`) screened 6 predeclared cells built on a Wikipedia
pageview attention-z construct. The strongest cell, **read** from that
artifact: `attention_battery_both_cold` (both teams' trailing attention
z-score `<= -0.5`), week-blocked primary, full-slate effect **+0.5221
accuracy points**, 95% CI **[-0.4408, +1.5040]**, **P+ 0.8568**, n=2,246
games, n_blocks=155 (season\*100+week). Season-blocked secondary CI
**[-0.0044, +1.0604]**, P+ 0.9735 — nearly excludes zero. Split-half
reliability of the underlying attention-z trait (odd/even week, team-level
mean, n=32 teams): **r=0.1315** (read from the same artifact) — low but
nonzero; per AGENTS.md a low split-half reliability is not on its own a
closing ground unless it is *zero* (refuted), so this stays an open,
worth-deepening lead, not a refutation.

Mechanism frame (**inferred**, restated from the task brief): the opening
line is softest where public attention is low or lopsided, because fewer
eyes and less market-making liquidity go into pricing a game nobody is
talking about. `both_cold` found the opposite of what a naive "overlooked
team gets undervalued and should be backed" story would predict: in the
subset, **home_cover was LOWER**, not higher (raw_gap sign convention:
`sign=-1`, subset_cover 0.4751 < complement_cover 0.4994, so the resolved
direction favors the away/road side when both teams are quiet). This
follow-up treats that direction as fixed context, not something to
re-litigate — every cell below reuses `sign=-1` for `home_cover`-scored
cells unless stated otherwise.

## Data source and point-in-time discipline (unchanged from the parent battery)

- Same Wikimedia Pageviews REST API raw files, same 32-team alias mapping,
  same daily granularity, same window rule: attention window is the 7 days
  ending the **Tuesday of the game's own week** (never gameday or later) —
  point-in-time safe for a Tuesday-opener pool decision. Raw JSON reused
  verbatim from `<scratchpad>/agent_attention/raw/*.json` (prior session's
  scratch dir, confirmed still present on disk, **read** its
  `predeclaration.md` and `fetch_manifest.json` this session).
- Same trailing-baseline z-score construction: up to 8 prior in-season games
  (min 2), attention_z = (this window's summed views - trailing mean) /
  trailing std.
- Same population: REG season, 2016-2025 (2015 excluded — partial pageview
  history), non-null `spread_line`, non-null `home_cover` (pushes dropped via
  `add_ats_outcomes`), newest `data/raw/*/schedules.parquet` snapshot.
- Reuses `attention_battery_screen.py`'s loading/construction functions by
  **import**, not by copy-paste or edit, to guarantee bit-identical
  attention_z values to the parent battery. That script is not modified.

## Method

Same `block_bootstrap_two_group` joint week-blocked bootstrap as the parent
battery and `scripts/nfl_bias_battery_screen.py`/`nfl_weather_battery_screen.py`
lineage: 20,000 samples, seed `20260819` (same seed as the parent, for direct
comparability), full-slate-scaled effect (`raw_gap_pts * fraction_of_slate`),
week-blocked (`season*100+week`) primary interval registered, season-blocked
(`season`) secondary reported as a robustness check only. Units:
**accuracy_points**, `home_cover` response unless stated. Cell 4 uses a
different statistic (a regression slope, not a subset-mean gap) and its
units are called out explicitly as non-fungible with the other three — see
its entry.

## The 4 predeclared cells (exact definitions, frozen before scoring)

1. **`attention_followup_both_cold_small_market`** — Decomposes "both teams
   quiet AND persistently a small draw" from "both teams quiet AND merely
   having an average-attention week." Flag = parent `both_cold`
   (`home_z <= -0.5 AND away_z <= -0.5`) **AND** `market_size_proxy <=
   median(market_size_proxy)` over the eligible population, where
   `market_size_proxy = (home_trailing_mean + away_trailing_mean) / 2` — the
   two teams' own trailing raw pageview-volume baselines (NOT the z-score;
   the same `trailing_mean` field the parent script already computes before
   normalizing), averaged. This is a within-instrument market-size stand-in:
   a team with a low baseline volume is a persistently smaller draw
   regardless of any single week's z. Same eligibility as parent
   (`both_baseline`), value_col `home_cover`, `sign=-1`. **Predicted:
   same-signed, LARGER magnitude than the parent's +0.52pt** (a persistently
   small audience should soften a line more than a temporary quiet week for
   a normally-large draw).

2. **`attention_followup_both_cold_non_primetime`** — Tests whether the
   effect survives, or is diluted by, the league's own exposure boost.
   Flag = parent `both_cold` **AND NOT** primetime, where primetime =
   `weekday in {"Thursday","Saturday","Monday"} OR (weekday == "Sunday" AND
   gametime >= "20:00")` (schedules.parquet `weekday`/`gametime` columns;
   this is a documented approximation of TNF/SNF/MNF/Saturday-window
   coverage, not a play-by-play broadcast-slot table — flagged as such, not
   hidden). Same eligibility, value_col, sign as parent. **Predicted:
   same-signed, LARGER magnitude in the non-primetime subset** (a marquee
   time slot gets attention from the schedule itself, which should dilute a
   "nobody is watching this game" mechanism).

3. **`attention_followup_cold_visitor_only`** — Decomposes which side drives
   the parent effect. Flag = `away_z <= -0.5` **only** (home team's z is
   unconstrained). Eligibility relaxes to `away_has_baseline` only (a
   deliberately different, broader eligible population than the parent's
   `both_baseline` — this is the point of the decomposition, not an error).
   value_col `home_cover`, `sign=-1` (same predicted direction as parent).
   **Predicted: same-signed, SMALLER magnitude than the parent** (a broader,
   less-selective population diluted relative to requiring both sides cold).

4. **`attention_followup_deep_cold_tilt`** — The continuous version, using
   the whole `both_baseline`-eligible population (not a threshold subset).
   Statistic: block-bootstrapped OLS slope of `home_cover` on
   `combined_z = home_z + away_z`, using the identical week-blocked joint
   multinomial resampling scheme as the subset-vs-complement cells, but
   recombining per-block sufficient statistics (`n, Sx, Sy, Sxy, Sxx`) into a
   slope each draw instead of a group-mean gap. **Predicted sign: positive**
   (higher combined_z, i.e. less-cold/more attention, associates with higher
   `home_cover`, matching the parent's resolved direction). **Units note —
   NOT directly fungible with cells 1-3 or the parent battery's pts**: the
   registered `effect` for this cell is a **model-implied point estimate**,
   not a raw subset-vs-complement full-slate gap: `slope * 100 *
   (mean(combined_z | both_cold subset) - mean(combined_z | full eligible
   population))`, i.e. "the continuous model's predicted home_cover-pts gap
   for a game at the both_cold subset's typical depth, relative to an
   average-attention game." This anchors the continuous estimate at the
   parent cell's own empirical depth (no arbitrary constant chosen after
   seeing data) so it is comparable *in spirit* to the parent's +0.52pt, but
   it is a different statistic (regression-implied, not empirical subset
   mean) and must not be pooled arithmetically with cells 1-3 without that
   caveat repeated. The raw slope (pts of home_cover per unit combined_z)
   and its own CI are also reported in the artifact for anyone who wants the
   un-anchored number.

## Blocking and reporting

- **Primary interval** (recorded to the registry): week-blocked
  (`season*100+week`).
- **Secondary** (robustness, console/notes only): season-blocked.
- Every cell reports `probability_positive`, never a binary "contains zero."

## Recording commitment (binding, AGENTS.md)

**Every cell above records to `registry/weak_signals.json` regardless of
sign or interval shape.** This is a 4-cell mined/exploratory follow-up
battery on a single already-promising pilot cell; per AGENTS.md an interval
crossing zero is the EXPECTED shape for a real small signal, not grounds to
reject, and the only two admissible closing grounds (RESOLVED wrong sign —
whole interval on the wrong side of the predicted direction — or a
positive-control bound) are not established by this screen for any cell
here. Default classification for all 4 cells: `unresolved_below_power`, no
`closing_ground`. Recording happens via a script
(`scripts/record_attention_followup.py`) that reads the computed
`results.json` and passes every numeric field through unmodified — no
hand-typed numbers — then the registry file is read back to confirm the
entries landed, with `--replace` re-run if a concurrent writer clobbered
them (this repo has multiple agents writing the same registry file this
session).

## Closing-grounds taxonomy (restated verbatim per AGENTS.md, for any
subagent that touches this file)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism — a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability;
(2) bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero." The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator.
