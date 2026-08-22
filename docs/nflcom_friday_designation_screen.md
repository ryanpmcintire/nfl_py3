# NFL.com Friday-designation screen

Status: **predeclared and frozen 2026-08-21, before any scoring run.** The cell
definitions, starter proxy, and direction below were written down before
`scripts/nflcom_friday_designation_screen.py` ever executed; the script is an
implementation of this document, not a source of it.

## Source and ingest

`scripts/ingest_nflcom_injuries.py` ingests the league's own final weekly
injury reports (`https://www.nfl.com/injuries/league/{season}/reg{week}`),
scout-verified as candidate rank B1 in `docs/data_source_scout_v5.md`
Section B. robots.txt was fetched and read before any page fetch (measured
2026-08-21: nothing under `/injuries/` disallowed, no Crawl-delay directive);
the script re-checks robots at runtime and fails closed, and enforces a >= 2s
(2.5s default) delay between fetches.

Snapshot (measured, `data/raw/nflcom_injuries/20260821T222602Z/manifest.json`):

- 54 / 54 pages fetched OK, 0 failures (REG weeks 1-18, seasons 2022 / 2023 /
  2024), one sha256 per page in the manifest.
- 17,483 parsed player-week rows (2022: 5,452; 2023: 5,439; 2024: 6,592).

## Stage 1b agreement vs the local nflverse injuries feed

Join documented normalization: casefold, ASCII-fold accents, strip punctuation,
drop suffix tokens (jr/sr/ii/iii/iv/v); primary key season+week+team+
normalized full name, fallback first-initial+last-name where that key is unique
on both sides within the same season+week+team. nflverse names come from
resolving `gsis_id` through the same snapshot's `weekly_rosters.parquet`.

Measured agreement (artifacts/nflcom_injuries/20260821T222602Z/agreement.json):

| Metric | Value |
|---|---|
| NFL.com player-week rows | 17,483 |
| nflverse REG rows in scope (2022-24) | 16,855 |
| nflverse rows carrying a report status | 8,094 |
| Match rate (NFL.com rows found in nflverse) | 99.63% (16,510 exact + 283 initial-last of 16,855) |
| Game-status exact agreement (both sides designate) | 7,911 / 8,293 = 95.39% |
| Dominant mismatches | nflverse missing entirely: 374 (289 Q, 76 Out, 9 D); `Note` rows: 6; status flips Q<->Out: 2 |

Read: the league's own final Friday/Saturday designations agree with the local
nflverse feed on ~95% of jointly designated player-weeks, with residual
disagreement dominated by nflverse rows that never carried a designation at all.
This gates IN, not out: the source reproduces the dead-after-2024 feed's game
statuses closely enough to serve as its post-2024 replacement candidate, and it
is the only one of the two that still publishes.

## Predeclared screen

Population: REG team-games, seasons 2022-2024, from the newest
`data/raw/*/schedules.parquet`, ATS outcomes via `nfl_ats.features.add_ats_outcomes`,
pushes dropped. One row per TEAM-game (two per game); value = team cover.

Battery pattern copied from `scripts/nfl_weather_battery_screen.py`: joint
week-blocked bootstrap (block = season*100+week), 20,000 draws, seed 20260821,
full-slate scaled effect in accuracy_points (raw gap x fraction of slate),
`probability_positive` as reported; season-blocked bootstrap as secondary only.

### Starter proxy (disclosed, frozen)

A player counts as STARTER-CALIBER for week W if, in their most recent prior
REG game of the same season (snap_counts.parquet, snapshot
`data/players/raw/20260817T184901Z`), they played at least 50% of offensive OR
defensive snaps (`max(offense_pct, defense_pct) >= 0.50`). This is a proxy, not
an official depth chart; it under-counts special-teamers and rotation linemen.
Week 1 games have no prior-week snaps, so cell (a) cannot flag them (counted as
missing-required-data, forced false, reported). Name join to snap counts uses
the same normalization as Stage 1b.

### Cells (direction frozen BEFORE scoring)

Mechanism for all three directional cells: these are FINAL designations
published Friday/Saturday after the Tuesday grading-line freeze but before
kickoff. A visible late-week downgrade marks availability loss that a
Tuesday-anchored price has had least time to absorb, so flagged teams'
realized results should fall short of the spread more often than the
complement. **Frozen direction: negative** (flagged teams' cover rate BELOW
complement) for every scored cell.

- **(a) `q_or_worse_starter_caliber`** — team has >= 1 designation of
  Questionable/Doubtful/Out on a starter-caliber player (proxy above).
  Direction: negative.
- **(b) `out_count_ge2`** — team has >= 2 Out designations (any players).
  Direction: negative.
- **(c) `new_saturday_designation`** — team has >= 1 Questionable-or-worse
  designation ABSENT from the Tuesday snapshot, operationalized against the
  local nflverse feed: no matched nflverse row for that season+week+team+player
  bearing a non-null report status dated on or before the Tuesday preceding the
  game (unmatched players count as new). This is the genuinely late information
  channel. Direction: negative.
- **(d)** era stability split of whichever of (a)-(c) has the largest
  absolute full-slate effect: per-season (2022 / 2023 / 2024) raw gap, interval,
  P+. Descriptive robustness reporting only; recorded in notes, not as separate
  registry entries.

### Leakage statement

Every flag derives solely from pages fetched-as-of their own week: each page is
that week's FINAL injury report, published Friday/Saturday and therefore fully
predating kickoff — no future information enters any flag. The starter proxy
consumes prior-week snap shares only (itself pregame-safe). Cell (c)'s
Tuesday-cutoff reconstruction consumes nflverse `date_modified` timestamps,
which are feed-update metadata used to reconstruct historical visibility, never
game outcomes.

### Multiplicity and recording discipline

This is a mined lead-generation family: three scored cells plus descriptive era
splits, NO multiplicity correction. Per AGENTS.md, every cell is predeclared to
record `unresolved_below_power` unless a terminal classification is admissible:
refuted mechanism (whole interval strictly below zero would be
`wrong_sign_resolved`; no split-half reliability) or bounded by positive
control. An interval crossing zero is NOT grounds for rejection and will not be
treated as one. Registry writes happen ONLY via explicit
`nfl-ats weak-signals record` commands returned for central recording; this
script writes an experiment provenance stamp under
`registry/experiments/nflcom-friday-designation-screen/` and never touches the
registry JSON itself.

---

## Measured results (2026-08-21 run)

Artifact: `artifacts/nflcom_friday_designation_screen/20260821T224931Z/results.json`
(registry stamp `registry/experiments/nflcom-friday-designation-screen/20260821T224931Z.json`).
Population: 815 REG 2022-2024 games, 28 pushes dropped, 787 games = 1,574
team-games; week-blocked primary (54 blocks), 20,000 draws, seed 20260821.
All figures below are measured from that artifact.

| Cell | n_flag | Raw gap | Full-slate effect | Week-blocked 95% | P+ | Season-blocked 95% |
|---|---|---|---|---|---|---|
| (a) q_or_worse_starter_caliber | 1,124 (96 wk-1 missing) | +0.934 pts | +0.667 pts | [-2.578, +4.024] | 0.642 | [-0.969, +3.522] P+ 0.706 |
| (b) out_count_ge2 | 942 | -4.495 pts | -2.690 pts | [-5.401, 0.000] | 0.024 | [-3.730, -1.607] P+ 0.000 |
| (c) new_saturday_designation | 1,542 | +0.000 pts | +0.000 pts | [-16.835, +16.651] | 0.463 | [-7.190, +7.090] P+ 0.298 |

Reading, per the frozen directions:

- **(b) is the lead.** Teams carrying >=2 Out designations cover ~4.5 points
  less often raw (-2.69 full-slate accuracy points), `probability_positive`
  0.024 — i.e. ~97.6% likely in the FROZEN negative direction — and the
  season-blocked secondary sits entirely below zero. Every era split keeps the
  sign: 2022 -2.686, 2023 -6.232, 2024 -5.293 raw pts. The primary interval's
  upper edge touches zero at this evaluator's resolution, which per AGENTS.md
  is the expected shape for a real-but-small signal, not grounds for anything.
  Classification: category 3 `unresolved_below_power` (mined family,
  uncorrected multiplicity, split-half reliability not yet computed).
- **(a)'s sign is opposite the frozen mechanism** (+0.667 pts, flagged
  starter-caliber designations cover slightly MORE), but the interval spans it;
  under the taxonomy that is unresolved, never a refutation.
- **(c) flags 98% of team-games** and is disclosed as near-degenerate: the same
  measurement that motivated it (nflverse feed is Friday-heavy, only 85 Tue
  rows across three seasons) means virtually every final designation IS new vs
  Tuesday (9,057 of 9,075 QA rows). The cell measured "the final report differs
  from a Tuesday snapshot" and found that is almost always true; it carries no
  discriminating information as defined and needs a sharper contrast (e.g.
  NFL.com-vs-nflverse DIVERGENCE) before any further look spends on it.

Registry writes happen via the exact `nfl-ats weak-signals record` lines
returned for central recording; nothing was written to any registry JSON by
this session beyond the automatic provenance stamps named above.
