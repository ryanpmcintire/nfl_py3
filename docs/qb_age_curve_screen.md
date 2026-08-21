# Quarterback age/experience trajectory screen

Family: quarterback age/experience trajectory (career-stage curves against the
market). Status: **predeclaration frozen before any cell was scored** — this
file was written before `scripts/qb_age_curve_screen.py` was run against any
cover outcome; measured results are appended at the bottom, tagged per the
AGENTS.md label-how-you-know-it rule.

## Prior-work overlap check and adjacency disclosure

- **PER-02 quarterback state** (`ROADMAP.md` line 203, read): starter
  probability, per-QB EPA/CPOE/sack/INT performance states toward the league
  mean (`src/nfl_ats/quarterbacks.py`). That family models *performance
  level*; this screen's cells are *career-stage trajectories* (rookie,
  second-year, very-high-experience) interacting with week-of-season. Distinct.
- **`backup_qb_start` / backup-QB fade overlay** (`docs/backup_qb_fade_overlay.md`,
  read): flags games where the starting QB differs from the team's modal QB —
  a *continuity/disruption* flag with no career-stage axis. Adjacent subject
  matter (who starts), different construct. Disclosed, not reused.
- **`player_qb_continuity`** (`registry/weak_signals.json`, reclassified
  `unresolved_below_power` per `docs/revisit_list.md`): feature-family presence
  of the same QB across training windows — again continuity, not career stage.
- **XLG-06 rookie priors** (`scripts/xlg06_rookie_prior_cfb_screen.py`, read):
  CFB-only recruiting-pedigree → freshman usage; no NFL data, no market
  outcome. Same broad "young player" theme, different league, construct, and
  target. No window overlap: this screen spends no CFB data.
- `hc_year_one_fade` is coach year-one, not quarterback. Grep of
  `registry/weak_signals.json` for age/career-stage QB cells: no hit.

Conclusion: the four cells below are uncovered; built fresh.

## Mechanism

Young quarterbacks improve WITHIN a season as film and rep accumulation
compounds, and very-high-experience quarterbacks fade late in the season;
betting markets are alleged to anchor on season-long team-quality priors and
under-adjust within-season trajectory. Predicted signs: first-year starters
cover more often late in the season than early (improvement); very-high-
experience starters cover less often in weeks 13+ (fade); first-year starters
against pressure-heavy defenses under-cover (the interaction the market is
slowest to price); second-year starters outperform market priors all season
(the year-2 jump).

## Data availability disclosure (binding for interpretation)

- **Age is NOT available locally** (measured: the only depth-chart archive,
  `data/quarterbacks/depth/raw/20260812T145333Z/quarterbacks.parquet`, carries
  `team/player_name/gsis_id/pos_abb/pos_rank/espn_id/pos_slot/observed_at_utc`
  — no birth date, and it is a single 2025 snapshot, not historical). The
  experience axis throughout is **career start count**, not age. Every cell
  conclusion therefore speaks to EXPERIENCE trajectories only; any age reading
  would be an unlicensed extrapolation.
- **Blitz rate is NOT available locally** (measured: local PBP snapshots carry
  no `number_of_pass_rushers`/blitz column). Cell C3 uses prior-season
  defensive **pressure rate** — sacks plus QB hits allowed per dropback — as
  the declared proxy for blitz heaviness. Conclusions for C3 speak to
  pressure, not strictly to blitzing.
- Starting-QB identification: per team-game, the passer with the most
  `pass_attempt == 1` plays for that `posteam` in that game (ties broken by
  earliest play). A "career start" = a team-game where a QB is his team's
  primary passer. This approximates official starts and is disclosed as such;
  it uses only completed games strictly before the season in question, so it
  is pregame-safe by construction.

## Reliability protocol (run BEFORE cells were scored)

1. Pressure-proxy persistence: year-over-year Pearson r between a defense's
   season-centered pressure rate allowed in season t and t+1, pooled
   consecutive franchise pairs 2009-2025, 20,000-sample bootstrap CI (seed
   20260821). Exclusion rule (the ONE admissible input exclusion): a CI
   entirely at or below 0 excludes the trait as a cell input on
   `no_split_half_reliability` grounds. An interval crossing zero is NOT an
   exclusion (binding taxonomy).
2. Young-QB performance-level split-half: among first-year-starter team-games,
   per team-season mean CPOE split by odd vs even game index within the
   season, Pearson r + bootstrap CI. This tests whether "young QB current-form
   level" is a stable trait at all; a wholly non-positive CI would be the one
   admissible refutation ground for the improvement mechanism.

## Predeclared cells (4, frozen before scoring)

Population: NFL REG close-grade slate 2009-2025. Value column `team_covered`
(team-perspective long table, one row per team-game). Sign is the PREDICTED
direction; positive full-slate effect points = prediction confirmed, scaled by
`n_flag / n_full_slate` via `scale_subset_effect`. Cells C1-C3 restrict the
comparison population as stated; the bootstrap contrasts flag vs complement
WITHIN that population while scaling to the full slate.

| # | name | population | flag | sign |
|---|------|-----------|------|------|
| C1 | `rookie_late_improvement` | starting QB <=6 career starts entering season | week >= 9 | +1 |
| C2 | `veteran_late_fade` | starting QB >=200 career starts entering season | week >= 13 | -1 |
| C3 | `rookie_vs_pressure` | starting QB <=6 career starts entering season | opponent prior-season pressure-rate centered >= Q75 of panel | -1 |
| C4 | `second_year_jump` | full slate | starting QB's FIRST career start occurred in the previous season (>=1 start last season, none earlier) | +1 |

Method: week-blocked joint multinomial block bootstrap (primary), season-blocked
secondary, algorithm-identical to `scripts/redzone_reversion_screen.py`.
20,000 samples, seed 20260821, accuracy_points full-slate units. Every cell is
recorded regardless of sign or interval shape; an interval crossing zero is
never a closing ground (binding taxonomy). Terminal classifications require an
admissible `--closing-ground`; everything else is `unresolved_below_power`,
recorded with `probability_positive`.

## Measured results (2026-08-21 run)

All numbers below are **measured** this session: artifact
`artifacts/qb_age_curve_screen/20260821T182715Z/results.json`, produced by
`scripts/qb_age_curve_screen.py` against PBP snapshot `20260817T184927Z` and
schedules `data/raw/20260817T235649Z/schedules.parquet` (4,317 REG close-graded
games; 8,862 start-identified team-games in the PBP panel; 544 defense-seasons).

### Reliability (measured before scoring)

| construct | statistic | 95% CI | n | excluded |
|-----------|-----------|--------|---|----------|
| pressure rate allowed, YoY Pearson | +0.267 | [+0.178, +0.351] | 512 defense-season pairs | no |
| first-year-starter game CPOE, odd/even split-half Pearson | +0.383 | [+0.257, +0.503] | 173 team-seasons (>=4 flagged games each) | no |

Neither CI sits at or below zero, so no cell input is excluded on reliability
grounds (measured). The young-QB form trait IS split-half reliable — the
improvement mechanism is not refutable on `no_split_half_reliability` grounds.

### Cell results (measured; week-blocked primary, accuracy_points full-slate units, P+ = probability_positive)

| # | cell | population | n_flag | effect pts | 95% CI | P+ |
|---|------|-----------|--------|-----------|--------|-----|
| C1 | rookie_late_improvement | 2,137 | 1,313 | -0.408 | [-0.917, +0.086] | 0.053 |
| C2 | veteran_late_fade | 94 | 31 | -0.037 | [-0.124, +0.054] | 0.207 |
| C3 | rookie_vs_pressure | 2,137 (496 missing prior) | 415 | -0.033 | [-0.263, +0.196] | 0.387 |
| C4 | second_year_jump | 8,634 | 1,412 | +0.235 | [-0.195, +0.667] | 0.855 |

Season-blocked secondary intervals agree in sign for every cell; C4's secondary
P+ is 0.903 (measured, same artifact).

### Classification

Every cell is category 3, `unresolved_below_power`: no interval sits wholly on
the wrong side of zero (C1's P+ 0.053 is a leaning AGAINST the predeclared
direction, not a resolved wrong sign — its upper bound is +0.086), both
reliability gates passed, and there is no positive control. Per the binding
taxonomy these are recorded, not closed.

Population notes worth carrying forward (**measured**): C2's veteran cell is
tiny — only 94 team-games since 2009 have a >=200-start QB, 31 of them weeks
13+ — so it is below any reasonable detection power and says almost nothing.
C3 lost 496 first-year-starter team-games to missing prior-season pressure
(2009 has no prior season; expansion/alias gaps), leaving 415 flagged rows.

The strongest read (**inferred**, my reasoning, not evidence): the two
young-QB cells lean in OPPOSITE directions to the mechanism's naive story —
late-season first-year starters covered LESS (C1, P+ 0.053 against) while
year-2 starters covered MORE all season (C4, P+ 0.855). If anything survives
pooling across this family it is the year-2 jump, not within-season rookie
improvement; but each cell alone is unresolved and C1/C4 share the young-QB
population, so they are correlated decompositions, not independent
confirmations.
