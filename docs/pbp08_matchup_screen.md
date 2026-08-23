# PBP-08 scheme/matchup interaction screen — predeclaration

Written **before** `scripts/pbp08_matchup_screen.py` scored any cover-rate
outcome. Only population/feasibility counts were examined pre-freeze
(**measured** this session: `pass_oe` non-null on 18.5k-20.8k dropbacks in
every season 2009-2025 of snapshot `data/pbp/raw/20260817T184927Z`; free-rush
dropbacks 16.2k-18.0k per season) — no cover rate, gap, interval, or
probability_positive for any cell was computed beforehand. This is the one
lane the ceiling analysis leaves open (`docs/pool_edge_plan.md`): coarse
pricing of CONDITIONAL propensities — the INTERACTION between offensive shape
and defensive shape, not either level alone.

## Binding closing-grounds taxonomy (verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) refuted mechanism — a RESOLVED wrong sign (whole interval on
the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". The registry code
hard-rejects inadmissible closures; if a record command errors, the verdict is
wrong, not the validator. Every cell below is recorded regardless of sign.
No registry JSON is written by this screen; exact record lines are returned in
the session summary for the owner to apply.

## Overlap check

Checked **before** designing cells (**read** this session,
`registry/weak_signals.json`, 439 signals):

- `team_style_short_game_vs_pressure_defense` is the closest neighbour:
  AWAY prior-season top-quartile `short_pass_share` x HOME defense
  prior-season top-quartile `shotgun_rate_faced`. Different traits (air-yard
  share / shotgun rate, prior-season team-stats window), different source
  table (`game_features_player.parquet`-adjacent team styles, not PBP), and a
  one-sided-role population. Thematically adjacent — **never pool as
  independent**; disclosed here.
- `team_style_pace_mismatch_dog_cover`, `team_style_short_game_identity`,
  `team_style_distinct_identity`: style levels or pace, not pass-efficiency x
  coverage-shape or protection x pressure mismatches. Not overlapping.
- No existing signal name contains `pbp08`, `matchup`, `proe`, `pass_oe`,
  `blitz`, `coverage_epa`, or `pressure_allowed` (**measured**, key grep this
  session). `pbp_drive_bundle` is a market-residual model bundle, not a cell
  screen.

## Data source and leakage posture

- Schedules: newest snapshot `data/raw/20260817T235649Z/schedules.parquet`
  (**read** this session).
- Play-by-play: snapshot `data/pbp/raw/20260817T184927Z` via
  `nfl_ats.pbp.load_pbp_snapshot` + the documented v1 efficiency filter
  `analysis_plays` (competitive-play subset for aggregation).
- Population: REG 2009-2025, pushes/missing-spread dropped via
  `nfl_ats.features.add_ats_outcomes`, one row per team-game (long table),
  canonicalized with `TEAM_ABBREVIATION_ALIASES`.
- All four traits are built ONLY from games strictly earlier than the current
  kickoff: windows are each team's most recent up to **4 completed REG games**
  strictly before this game by `gameday` sort within team (an early-season
  window may reach into the prior season's final weeks — still strictly
  prior; frozen deliberately so weeks 1-3 stay measurable), requiring >= 3
  observations with data; rows with an incomplete window are INELIGIBLE from
  every arm (never defaulted into the complement).

## Derived quantities (frozen)

Per game per team on v1-filtered dropbacks (`qb_dropback == 1`;
pressure = sack OR qb_hit):

- `off_pass_oe_g`: mean nflverse `pass_oe` over the offense's dropbacks.
- `off_press_allow_g`: pressures allowed / dropbacks.
- `cov_epa_free_g`: mean EPA allowed by the defense on opponent dropbacks
  WITHOUT pressure (the coverage-EPA-allowed leg; higher = worse coverage).
- `press_gen_g`: pressures generated / dropbacks faced.

**Blitz-proxy disclosure:** the ingested nflverse snapshot omits the per-play
`blitz` flag (`PBP_SNAPSHOT_COLUMNS`, **read** this session), so "blitz-heavy"
is proxied exactly by `press_gen_g`; "coverage-EPA allowed split by pressure
generated" is computed as specified (EPA on no-pressure dropbacks only).

Window features = mean of the per-game values over the 4-game strictly-prior
window (min 3): `off_pass_oe_w`, `off_press_allow_w`, `cov_epa_free_w`,
`press_gen_w`.

**Quartile assignment is expanding and strictly prior:** for each week-block
(`season*100 + week`) the q25/q75 thresholds come from ALL assigned window
values in STRICTLY EARLIER blocks only (>= 200 pooled observations required;
rows before that are ineligible). No future information touches any
threshold. Flags: bottom = value <= q25, top = value >= q75.

## Split-half reliability anchors (computed FIRST)

Before any cell cover rate, for every trait: among rows whose 4-game window is
complete, correlate the recent-half aggregate (prior games 1-2) against the
older-half aggregate (prior games 3-4), Pearson r across all such team-game
rows. These anchors gate nothing but are reported first and supply the
`--reliability` field of each record line (conservative minimum of the two
legs' reliabilities for an interaction cell). A near-zero anchor would make
the affected family `no_split_half_reliability` — admissible terminal ground —
and nothing else does.

## Cells (directions frozen BEFORE scoring)

| name | flag (both legs at their quartile) | predicted sign |
|---|---|---|
| `pbp08_pass_mismatch` | `off_pass_oe_w` top-quartile AND opponent `cov_epa_free_w` top-quartile (worst coverage) | +1 back the passing side |
| `pbp08_protection_mismatch` | `off_press_allow_w` top-quartile AND opponent `press_gen_w` top-quartile | -1 back the defense side |
| `pbp08_pass_mirror_null` | `off_pass_oe_w` BOTTOM-quartile AND opponent `cov_epa_free_w` BOTTOM-quartile | expected NULL (bottom-vs-bottom control); read two-sided, recorded whatever the sign |
| `pbp08_protection_mirror_null` | `off_press_allow_w` BOTTOM-quartile AND opponent `press_gen_w` BOTTOM-quartile | expected NULL (bottom-vs-bottom control); read two-sided |

(d) Era split of the strongest primary cell by |full-slate effect| among the
two candidate cells only (mirrors are controls): `2009_2017` vs `2018_2025`,
same battery parameters.

## Standard battery (frozen)

Week-blocked bootstrap PRIMARY: 20,000 resamples, seed **20260824**, blocks =
`season*100+week`, two-group gap bootstrap resampling whole week-blocks;
effect reported full-slate-scaled in accuracy points
(sign x raw gap x fraction_of_slate). Season-blocked SECONDARY (same draws,
season blocks). REG 2009-2025. Complement = all eligible team-games not in the
cell (primetime-screen convention).

## Mined-family disclosure

Scheme/matchup interactions are a MINED family (this lane was selected because
the ceiling analysis left it open); quartile-threshold cells are coarse
instrumentation, not theory-free but threshold-chosen. Four cells + two era
splits, correlated by construction (shared windows, shared legs), uncorrected
multiplicity. Adjacent registered signals listed above are never poolable with
these as independent inputs. Measure-only: no registry JSON written by the
script; record lines returned to the owner verbatim.

---

## Results (post-scoring addendum, 2026-08-23)

Run: `artifacts/pbp08_matchup/20260823T000758Z/results.json` (**measured** this
session; registry stamp `registry/experiments/pbp08-matchup-screen/20260823T000758Z.json`).
Population: 4,431 REG games, 114 pushes/missing dropped, 8,634 team-game rows,
8,538 with complete windows, **8,324 eligible** after expanding-quartile
assignment (283 week-blocks).

### Split-half reliability anchors (computed and printed FIRST)

| trait | r (2-vs-2-game split) | n |
|---|---|---|
| off_pass_oe_g | 0.2433 | 8,498 |
| off_press_allow_g | 0.2444 | 8,498 |
| cov_epa_free_g | 0.0631 | 8,498 |
| press_gen_g | 0.0631 | 8,498 |

Both defense-side anchors are weak (~0.06; measured twice independently, the
two values differ in the 5th decimal). On a 2-vs-2 split this understates the
4-game window composite roughly two-fold (Spearman-Brown), but the
defense-side legs are the fragile ones � flagged, not hidden. No anchor is
zero, so `no_split_half_reliability` is NOT admissible as a closing ground.

### Cell table (week-blocked primary, full-slate accuracy points)

| cell | n_flag | raw gap | effect | 95% CI | P+ | season-blocked P+ |
|---|---|---|---|---|---|---|
| pbp08_pass_mismatch (+1) | 381 | +2.34 | +0.107 | [-0.136, +0.353] | 0.8060 | 0.8659 |
| pbp08_protection_mismatch (-1) | 733 | -3.82 | +0.336 | [+0.014, +0.658] | 0.9785 | 0.9797 |
| pbp08_pass_mirror_null (null exp.) | 633 | +0.43 | +0.033 | [-0.296, +0.354] | 0.5712 | 0.5835 |
| pbp08_protection_mirror_null (null exp.) | 393 | +0.40 | +0.019 | [-0.218, +0.263] | 0.5550 | 0.5740 |

Era splits of the strongest candidate (`pbp08_protection_mismatch`):
2009-2017 +0.445 [+0.026, +0.863], P+ 0.9812 (n_flag 354); 2018-2025 +0.225
[-0.255, +0.705], P+ 0.8129 (n_flag 379).

### Reading

- Both mirror controls land where a null should (~+0.02/+0.03 pts, P+
  ~0.56-0.57), so the instrument is not picking up generic quartile-selection
  drift.
- `pbp08_protection_mismatch` is positive on BOTH blockings with both
  intervals excluding zero (P+ 0.979/0.980), direction as predeclared, era
  split consistent in sign. This is a mined-family cell with uncorrected
  multiplicity: it justifies a predeclared confirmation look, not a claim.
- `pbp08_pass_mismatch` leans positive (P+ 0.81) with an interval crossing
  zero � the EXPECTED shape for a real small signal at this evaluator's
  ~2-point resolution; recorded, never rejected for that reason.

All four cells classify `unresolved_below_power`: no resolvably wrong sign,
no zero anchor, no positive-control bound was run.
