# Graph-rating input screen — predeclaration

Written before any Gate 2 (incremental-value) result is produced, per the
binding rule this task was given: "Predeclare... BEFORE producing any
ranking." Gate 1 reuses an existing measurement
(`artifacts/reliability_map/20260826T112507Z/results.json`) rather than
deriving a new one, so reading it first is not a selection step.

## Why this exists

The owner is reopening PageRank/HITS-style graph ratings, redesigned so each
team statistic is scored separately before being fed to the graph engine,
rather than assuming every available statistic belongs. This document is the
predeclared screen that produces that input list. Another agent builds the
engine; this screen only decides what goes in.

## Closing-grounds taxonomy (binding, restated per AGENTS.md so this document
stands on its own)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism — a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero/negative split-half reliability
(`reliability <= 0.10`, the registry's own admissibility ceiling for
`no_split_half_reliability`); (2) bounded by a positive control proven able
to detect an effect that size. Everything else is `unresolved_below_power`:
record it, report `probability_positive`, never the binary "contains zero".

## Gate 1 — does it repeat?

Reused unchanged from `artifacts/reliability_map/20260826T112507Z/results.json`
(83 team-week families swept, all 83 finite — 0 skipped as constant). Not
re-derived here. A family closes at Gate 1 only when its
`pearson_r_ci95` (the untransformed correlation's CI — monotonically related
to the reported Spearman-Brown reliability for `r > -1`, so a resolved-below-
zero `pearson_r_ci95` implies a resolved-below-zero Spearman-Brown reliability
too) sits entirely at or below zero AND the point-estimate Spearman-Brown
reliability is `<= 0.10` (the registry's `NO_SPLIT_HALF_RELIABILITY_MAX`
admissibility ceiling for the `no_split_half_reliability` closing ground).

Measured (read from that artifact before writing this predeclaration,
disclosed since Gate 1 is reuse, not derivation): 3 of 83 families have a
point-estimate Spearman-Brown reliability `<= 0`, but only one,
`gap_sandwich_spot` (SB reliability −0.626, `pearson_r_ci95`
[−0.292, −0.182], entirely below zero), is RESOLVED below zero. The other
two (`bias_prior_week_ats`, `gap_division_revenge`) have a `pearson_r_ci95`
upper bound above zero and are NOT closed by Gate 1 — they proceed to Gate 2
exactly like every other family, per the taxonomy above.

Gate 1 therefore closes exactly one of 83 families
(`gap_sandwich_spot` → `refuted_mechanism` / `no_split_half_reliability`).
Every other family, regardless of point-estimate magnitude or sign,
proceeds to Gate 2 — a low reliability whose interval crosses zero is
`unresolved_below_power`, not a closure.

## Gate 2 — does it add anything the market does not already have?

**Design.** For every family, build one candidate feature: the standardized
home-minus-away differential (`diff_<family> = home_<x> − away_<x>`), the
natural "matchup" reduction already used throughout this project's other
differenced features (`elo_diff`, `graph_pagerank_diff`, the `gap_*`
constructs). Compare two weekly-refit ridge models predicting
`ats_margin` (`result − spread_line`, the project's `market_residual`
target):

- **Baseline** — `nfl_ats.margin.fit_market_baseline`, reused unmodified: zero
  team features, cover probability read directly from the game's own
  no-vig-implied spread odds (`home_spread_odds`/`away_spread_odds`). This
  is "trust the market" with no team-specific information at all — spread-
  odds coverage is ≈100% for every season this screen touches (measured this
  session from `game_features_weak_stack_v4.parquet`, 2013–2025 all ≥99.6%).
- **Candidate** — the same modeling primitives the project already uses
  (`nfl_ats.margin.make_margin_estimator`: median imputation with a
  missingness indicator, standardization, `Ridge(alpha=10.0)`) fit on the
  single `diff_<family>` column, predicting `ats_margin`, with the standard
  20%-holdout out-of-time residual distribution (`distribution_fraction=0.20`,
  `min_distribution_rows=10`) exactly as `nfl_ats.margin.fit_margin_model`
  builds it — reimplemented as a thin wrapper that accepts an explicit
  feature-column list instead of a registered `MarginFeatureProfile`, so
  nothing in `margin.py`/`constants.py`/`clv.py` is touched (concurrency
  boundary respected).

Both models are graded with production's own rule,
`home_cover_probability >= 0.5` (`pool.py`/`backtest.py`'s pick rule, not the
sign rule), settled against the same line for both arms in a given window so
the comparison is paired game-for-game.

**Split (predeclared, chosen before any incremental number exists).**
Team/injury/participation families in this table are ~100%-populated only
from the 2013 season onward (measured this session:
`home_injury_skill_epa_value_lost`/`home_active_roster_continuity` etc. are
0%-covered in 2009–2012, ≥99% from 2013 on). The archived Tuesday-opener
odds snapshot store only covers 2020–2025 (the same 1,503–1,537-game archive
`docs/opener_evaluation.md` and `docs/overlay_subset_holdout_v2.md` already
use as this project's standing outer-test era).

- **SELECTION window**: regular-season games in seasons **2013–2019**
  (≈1,750 games), graded at the **closing** `spread_line` already in the
  feature table — no odds-archive dependency, larger sample, used only to
  screen/rank.
- **HOLDOUT window**: regular-season games in seasons **2020–2025**
  (≈1,500 paired games), graded at the archived **Tuesday opener** via
  `nfl_ats.clv.build_pairing_table`/`close_reference_table` — reused
  unmodified, the same pairing machinery `opener_pick_evaluation` uses.
- Training for every graded week always uses **all** completed regular-season
  games strictly before that week's earliest kickoff, from the full
  2009-onward history (rows from 2009–2012 simply carry an imputed,
  uninformative candidate feature — they still contribute the baseline's
  market information).
- Per rule 3 (grade the decision at the opener — a close-graded number may
  never veto a play), **the HOLDOUT window's opener-graded delta is the
  decision-relevant number for every survivor.** The SELECTION window's
  close-graded delta is reported for transparency and used only to describe
  the ranking; it never overrides or vetoes the holdout figure, and no
  candidate is dropped from the reported list because of what either window
  showed — dropping only happens via the two admissible closing grounds
  above.

**Uncertainty.** Paired per-game `candidate_correct − baseline_correct`,
week-blocked and season-blocked bootstrap (`nfl_ats.clv.week_blocked_bootstrap`,
reused unmodified), **1,000 resamples** (reduced from the project's usual
2,000 for tractability across 83 candidates × 2 windows — declared, not
hidden), seed `20260826`. `probability_positive` is the week-blocked
fraction of resamples with a positive delta; the season-blocked figure is
reported alongside, never substituted as the headline.

**Declared simplifications**, stated up front:
- The single-feature ridge is fit fresh via a local wrapper
  (`fit_single_feature_market_residual_model` in
  `scripts/graph_input_screen.py`) that mirrors `fit_margin_model` exactly
  except for accepting an explicit column list; it is not a different
  recipe, just a different entry point.
- Every candidate uses the SAME ridge alpha (10.0) and the SAME
  distribution-fraction convention — no per-family tuning, so no candidate
  gets an unfair modeling advantage.
- This screen does not test two-feature (home+away separately) or
  nonlinear representations of a family; it tests the natural differenced
  form only, consistent with how this project already differences almost
  every comparable feature.

## Gate 3 — redundancy

Pairwise Pearson correlation of every surviving family's standardized
`diff_<family>` column, computed across the full REG-season archive with
available (non-missing) pairs (feature-only, no outcome label involved, so
this is not a target-informed selection step and may safely use the full
history rather than a window split). Hierarchical agglomerative clustering,
average linkage, on distance `1 − |r|`, cut at `|r| >= 0.6` (families
correlated at or above that threshold co-cluster; below it they are treated
as independent). Threshold declared here, before the correlation matrix is
computed.

**Representative selection per cluster**: the member with the highest
HOLDOUT (opener) week-blocked `probability_positive`; ties broken by higher
Gate-1 reliability, then alphabetically by family name. Every cluster is
reported in full (not just its representative) so the map itself is a
usable artifact, per the task's own instruction.

## Deliverable

`artifacts/graph_input_screen/<UTC timestamp>/results.json`, written via
`write_experiment_artifact`, containing for every one of the 83 families:
Gate 1 reliability and its resolution status, Gate 2 selection- and
holdout-window effects (estimate, week- and season-blocked intervals,
`probability_positive`), Gate 3 cluster id and representative flag, and a
final `status` of `survivor` (passed Gate 1, has a Gate 2 measurement,
reported at holdout `probability_positive`) or `closed_gate1` (the one
`gap_sandwich_spot` case). This project draws no promotion threshold on
`probability_positive` — a screen output is a ranked list with its evidence
attached, not a pass/fail gate on anything but the two admissible closing
grounds.

## Recording

Every one of the 83 families gets one `nfl-ats weak-signals record` entry
(via the same validated `record_signal` function the CLI calls, invoked
directly from `scripts/graph_input_screen.py` in one process rather than 83
subprocess calls — same code path, same validation, not a bypass).
`effect`/`effect_units` = the HOLDOUT window's `delta_accuracy` in
`accuracy_points` (×100); `interval` = its week-blocked 95% CI (×100);
`probability_positive` = its week-blocked value; `reliability` = the Gate 1
Spearman-Brown figure; `family="graph_input_screen"`; `category` inferred
per-family from its name (health for `injury_*`, onfield for
`pbp_*`/offense/defense/qb state metrics, schedule for `schedule_rating`/
continuity constructs, control/modeling where nothing else fits — mapped
explicitly in the script, not left to a default). `classification` is
`refuted_mechanism`/`no_split_half_reliability` for `gap_sandwich_spot`
only; every other family is `unresolved_below_power` UNLESS its own HOLDOUT
(opener-graded) interval is resolved entirely below zero, in which case it is
`refuted_mechanism`/`wrong_sign_resolved` — decided by the opener grade only,
never the selection-window close grade, per rule 3. `--plain-summary` is
10 words or fewer, one sentence, no caveats or effect sizes.

No pooling step runs against these 83 recordings in this task; that is a
separate, later decision the registry's own `weak-signals pool` command can
answer on its own predeclared look.
