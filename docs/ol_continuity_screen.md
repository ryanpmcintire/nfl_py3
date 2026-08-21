# Offensive-line continuity screen

Family: offensive-line continuity. Status: **predeclaration frozen before any
cell was scored** (this section was written before
`scripts/ol_continuity_screen.py` was run against any cover outcome); measured
results are appended at the bottom, tagged per the AGENTS.md
label-how-you-know-it rule.

## Feasibility verdict (measured this session)

TRUE OL-start continuity is computable locally; the declared PROXY path
(sack-rate stability) is NOT needed and is not used.

- `data/players/raw/20260817T184901Z/snap_counts.parquet` carries per-player
  `position` and offense snaps per team-game, seasons 2013-2025 REG
  (measured: 324,611 rows; OL-position rows 51,281).
- `data/players/participation/raw/` play-level tables carry no position labels,
  but are unnecessary given snap counts.
- `data/raw/20260817T235649Z/schedules.parquet` has zero missing close-grade
  spreads for REG seasons 2013-2025 (measured).

No NFL window was spent; everything loads from local snapshots.

## Prior-work overlap check

- `docs/` grep for offensive-line / continuity screens: no OL-family screen
  found in `docs/*.md` this session (measured via directory listing of the
  screen docs written alongside this one). `qb_continuity_replication` and
  `cfb_value_weighted_continuity` artifacts exist under `artifacts/` but target
  QB continuity and CFB roster value weighting — different constructs
  (read, artifact directory names only; contents not audited this session).
- `registry/weak_signals.json` was NOT modified or read cell-by-cell this
  session; if a duplicate OL signal exists there it would surface at record
  time. Flagged as **unverified** rather than claimed clean.

## Mechanism

Offensive-line play is coordination-heavy: five players executing shared calls.
A disrupted line degrades pass protection and short-yardage rush conversion
faster than individual talent replacements can restore it, and the market may
under-react to line churn that is visible in snap participation but not in
injury-report headline names. Prediction: teams fielding a less continuous OL
unit than their peers cover LESS over the following stretch; acutely
overhauled lines (most recent game started <=2 of the same five) fade hardest;
chronically stable lines cover MORE.

## Constructs (built from local snap-count + schedule snapshots)

Sources: `snap_counts.parquet` snapshot `20260817T184901Z`,
`schedules.parquet` snapshot `data/raw/20260817T235649Z/`. Franchise aliases
via `TEAM_ABBREVIATION_ALIASES`; REG games only, seasons 2013-2025.

Per team-game:

- **OL starter**: an OL-position player (`C, G, T, OG, OT, LG, RG, LT, RT,
  OL`, including slash-listings such as `C/G` where any component matches)
  whose `offense_pct >= 0.50` that game (played at least half the team's
  offensive snaps).
  *Amendment before scoring (no cover outcome had been touched):* the
  originally drafted rule keyed on `offense_snaps >= 10%` of the team's summed
  offense snaps; inspection of the snapshot showed the team-wide sum
  multi-counts every play (~11 players on the field), making that denominator
  meaningless (~730). The dataset's own `offense_pct` share replaces it. This
  amendment happened before the script was run against any cover outcome.
- **`ol_continuity`** (team-game): (# starters this week who were also starters
  in the team's previous REG game) / 5. Week 1 of each season has no in-season
  prior game -> NaN (missing, reported, never silently dropped into either
  arm).
- **`ol_trailing2`**: mean of `ol_continuity` over the team's two most recent
  REG games with non-NaN continuity; requires >=1 non-NaN game.
- **Season trait** for reliability and prior-season joins: team-season mean of
  `ol_continuity` over weeks 2+, centered by that season's league mean
  (`ol_continuity_centered`). Prior-season value joined one season forward;
  2013 games carry no prior trait and count as missing.

All flags use only information available before the flagged game's kickoff:
previous REG games' snap participation and prior-season aggregates. No
same-game outcome enters any flag (leakage self-check runs inside the script
and its result is written to the artifact).

## Reliability protocol (run BEFORE cells were scored)

1. YoY Pearson r and Spearman rho between centered season traits t and t+1,
   pooled consecutive-franchise pairs 2013-2025, 95% CI from 20,000 pair-level
   bootstrap resamples (seed 20260821).
2. Within-season split-half: Pearson r between each team-season's odd-week
   mean and even-week mean of `ol_continuity` (weeks 2+), same bootstrap CI.
3. Exclusion rule (the ONE admissible input exclusion): a trait whose YoY
   Pearson 95% CI sits entirely at or below 0 is excluded as a cell input on
   `no_split_half_reliability` grounds. An interval crossing zero is NOT an
   exclusion (binding taxonomy).

## Predeclared cells (4, frozen before scoring)

Population: NFL REG close-grade slate 2013-2025, team-perspective long table
(one row per team-game, `team_covered`). Thresholds are quantiles of the
pooled 2013-2025 distribution of the named trailing statistic across all
team-games where it is defined, computed once before scoring.

| # | name | flag | value | sign |
|---|------|------|-------|------|
| C1 | `ol_low_continuity_fade` | `ol_trailing2` <= pooled Q25 | `team_covered` | -1 |
| C2 | `ol_high_continuity_back` | `ol_trailing2` >= pooled Q75 | `team_covered` | +1 |
| C3 | `ol_acute_overhaul_fade` | most recent game `ol_continuity` <= 0.4 (<=2 of 5 returning) | `team_covered` | -1 |
| C4 | `ol_prior_season_weak_early_fade` | weeks 1-8 ONLY AND prior-season `ol_continuity_centered` <= pooled Q25 | `team_covered` | -1 |

Sign convention: `sign` is the PREDICTED direction; `probability_positive`
(P+) is the bootstrap probability the prediction holds. Positive
`full_slate_effect_pts` = prediction confirmed, accuracy points scaled to the
full slate (`nfl_ats.experiment_runner.scale_subset_effect`).

## Method

Week-blocked joint multinomial block bootstrap (primary), season-blocked
secondary, algorithm-identical to `scripts/redzone_reversion_screen.py` /
`scripts/team_style_screen.py`. 20,000 samples, seed 20260821,
accuracy_points full-slate units. Every cell is recorded regardless of sign
or interval shape; an interval crossing zero is never a closing ground
(binding taxonomy). Terminal classifications require an admissible
`--closing-ground`; everything else is `unresolved_below_power`.

## Measured results (2026-08-21 run)

All numbers below are **measured** this session: artifact
`artifacts/ol_continuity_screen/20260821T184636Z/results.json`, produced by
`scripts/ol_continuity_screen.py` against snap-counts snapshot
`data/players/raw/20260817T184901Z/snap_counts.parquet` and schedules
`data/raw/20260817T235649Z/schedules.parquet` (3,322 REG close-graded games;
6,644 team-game rows; traits computed on the full REG universe including
push-graded games, scored on the graded subset). Leakage self-check PASSED:
250 sampled team-games had trailing-2 continuity recomputed from strictly
prior games only, 0 mismatches (measured, same artifact).

### Reliability (measured before scoring)

| trait | YoY Pearson r | 95% CI | Spearman | split-half Pearson | 95% CI |
|-------|--------------|--------|----------|--------------------|--------|
| `ol_continuity` (season mean) | +0.075 | [-0.025, +0.173] | +0.071 | +0.479 | [+0.398, +0.555] |

n = 384 YoY pairs; n = 416 team-seasons for split-half (odd-week vs even-week
means, weeks 2+). No exclusion fired: neither CI sits entirely at or below
zero. **Read (measured numbers, inferred interpretation):** the WITHIN-season
trait is strongly reliable (split-half ~0.48) but its season-level identity
barely persists across years (YoY ~+0.08) — OL continuity is a stable
within-season team property that re-randomizes each offseason, which is the
opposite of the redzone-screen trait pattern and consistent with roster churn
resetting the unit annually.

### Cell results (measured; week-blocked primary, accuracy_points full-slate units, P+ = probability_positive)

| # | cell | n_flag | effect pts | 95% CI | P+ |
|---|------|--------|-----------|--------|-----|
| C1 | ol_low_continuity_fade | 1,785 | -0.031 | [-0.738, +0.677] | 0.459 |
| C2 | ol_high_continuity_back | 2,247 | -0.580 | [-1.416, +0.262] | 0.085 |
| C3 | ol_acute_overhaul_fade | 39 | -0.023 | [-0.110, +0.059] | 0.260 |
| C4 | ol_prior_season_weak_early_fade | 722 | +0.127 | [-0.857, +1.060] | 0.598 |

Season-blocked secondary intervals agree in sign and rough width for every
cell (measured, same artifact).

### Classification

Every cell is category 3, `unresolved_below_power`: no week-blocked interval
sits wholly below zero (C2's P+ 0.085 is a leaning against the stability-back
prediction — upper bound +0.262 — not a resolved wrong sign), no reliability
exclusion fired, and there is no positive control. Per the binding taxonomy
these are recorded, not closed. Notable reads (**inferred**, my reasoning, not
evidence): C2 leaning negative while C1 sits at coin-flip is internally
tense — if taken at face value it would say HIGH continuity teams under-cover
rather than low-continuity teams fading — but one cell at P+ 0.085 with a
wide interval does not resolve the mechanism's direction. C3's n=39 makes it
uninformative at this evaluator's resolution by construction, not by result.
The family stays open; the within-season reliability justifies continued
work on acute-disruption constructs with larger flag populations.

### Registry

No registry JSON was written by this session (per instruction); the four
cells are recorded via the exact `nfl-ats weak-signals record` command lines
returned in the session summary, all as `unresolved_below_power`.
