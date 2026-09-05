# Baseline-parity regression suite (ENG-17, extended by ENG-28)

ROADMAP Phase 13, ENG-17: "Freeze small canonical fixtures proving
market-only, simple-model, active-model, and overlay comparisons use
identical games, grading rules, cutoffs, and push handling."

ROADMAP Phase 13, ENG-28: "Extend `nfl_ats.parity` so the active-model and
overlay paths run through the real `weak_stack`/ridge opener-snapshot
machinery (`clv.opener_pick_evaluation`) instead of
`walk_forward_backtest(market_context)`, with a commit-safe miniature
opener store; the population, cutoff and push parity claims then cover the
path the card is graded on." Adds two more paths --
`"active_model_production"` and `"overlay_production"` -- documented in
their own section below. The original four paths (table immediately below)
are UNCHANGED by ENG-28: they still grade through the
`walk_forward_backtest` stand-in described in "How opener vs close is
graded" below, and their own tests still pass unmodified.

This is a **parity** suite, not a verdict. It never says one comparison path
beats another, it never computes or reports a confidence interval, and it
never touches `registry/`. Per `AGENTS.md`'s binding invariant, an interval
or accuracy gap crossing zero is never grounds to reject or close anything —
that rule does not even apply here, because nothing in this suite closes a
line of research. The only thing under test is whether four comparison paths
agree on **which games they graded, when they were allowed to grade them,
and how they handled a push** — the plumbing every accuracy number in this
project depends on being identical before it means anything to compare two
of them.

## Files

- `src/nfl_ats/parity.py` — the adaptor: `grade_games(frame, line, path,
  *, min_train_games=..., opener_store=None)` and `paired_delta(a, b)`.
- `tests/fixtures/parity/games.csv` — the frozen fixture (60 rows, 3 seasons,
  deterministic seed 20260904; regenerated, if ever needed, by the
  documented generator described below).
- `tests/fixtures/parity/opener_store/` (ENG-28) — the miniature, commit-safe
  market-snapshot store `active_model_production`/`overlay_production` read,
  same directory layout as `data/market/raw/` (one `manifest.json` +
  `quotes.parquet` per committed snapshot). Covers only the two weeks any
  path ever grades (2022 week 4 and 5 -- see "Why one path is inlined"
  below); regenerated, if ever needed, by
  `.agent_tmp/generate_parity_opener_store.py` (not part of the shipped
  suite, same convention as `games.csv`'s own generator).
- `tests/test_baseline_parity.py` — the suite.

## The grading contract, in one table

| | Population | Chronological cutoff | Push rule | Line |
|---|---|---|---|---|
| **market_only** | Games in weeks with ≥ `min_train_games` (default 50) prior non-push completed games | `nfl_ats.parity._eligible_weeks` (mirrors, does not call, the identical inlined pattern below — see "Why one path is inlined") | `nfl_ats.clv.pick_correct`: `ats_margin == 0` excluded, never a loss/half-win | `spread_line_open` or `spread_line_close`, no vig removed via `nfl_ats.odds.no_vig_probabilities` |
| **simple_model** | Same rule, computed internally by `nfl_ats.backtest.walk_forward_backtest` (`feature_set="market"`) | `walk_forward_backtest`'s own `cutoff = weekly_games["gameday"].min(); training = completed.loc[completed["gameday"].lt(cutoff) & completed["home_cover"].notna()]` | Same `pick_correct` call, applied to the harness's own output | Whichever line `spread_line`/`total_line`/`ats_margin`/`home_cover` were built against (opener or close — see "How opener vs close is graded" below) |
| **active_model** | Same harness, `feature_set="market_context"` (stands in for the production `weak_stack`/ridge model — see below) | Same `walk_forward_backtest` cutoff | Same `pick_correct` call | Same swap mechanism |
| **overlay** | Identical to `active_model` — `nfl_ats.coach_fade_overlay.apply_coach_fade_overlay` only ever changes `home_cover_probability` on flipped rows; it cannot add, remove, or reorder games | Identical to `active_model` | Same `pick_correct` call, unaffected by the flip (pushes are a property of `result`/line, not of the pick) | Same swap mechanism |

`min_train_games` defaults to `nfl_ats.constants.MIN_FITTABLE_TRAIN_GAMES`
(50) — not a suite-local choice. It is the literal floor `fit_cover_model`
(`"At least 50 non-push games are required to train a model"`) and
`fit_margin_model` (`MIN_FITTABLE_TRAIN_GAMES`) hard-code; passing anything
lower does not make a smaller fixture usable, it just makes the model fit
raise once a week's training count falls between the caller's floor and 50.
This is also why the fixture is 60 rows rather than ~40: 50 non-push
training games plus a handful of graded games is the smallest population
that can ever produce a walk-forward test row in this codebase, full stop.

## How opener vs close is grading is done without a market-snapshot store (the ORIGINAL four paths)

This section describes `market_only`/`simple_model`/`active_model`/`overlay`
only. ENG-28 below adds two more paths that DO read a real market-snapshot
store; this section is left exactly as ENG-17 wrote it because it is still
an accurate description of those four, unchanged paths.

The project's real opener-vs-close active-model comparison
(`nfl_ats.clv.opener_pick_evaluation`, `docs/opener_evaluation.md`) reads a
point-in-time Tuesday-opener market-snapshot store
(`data/market/raw/`) to get a paired `tue_open`/close spread per game.
Reproducing that store for a 60-row fixture was out of scope for ENG-17's
four-path suite. Instead this suite reuses the SAME declared approximation
`nfl_ats.clv.active_model_residual_at_opener` already uses in production:
only `spread_line` (and `total_line`/spread odds) are swapped to the
requested line; every other feature stays fixed. `nfl_ats.parity.
_features_for_line` does this once, then calls
`nfl_ats.features.add_ats_outcomes` (real function) to recompute
`ats_margin`/`home_cover` fresh against whichever line was requested — so a
game that is a push at the opener but not at the close (or vice versa) is
handled correctly and identically by every path, because every path reads
`ats_margin`/`home_cover` off the SAME just-recomputed columns.

This is also why `simple_model` and `active_model` are NOT
`nfl_ats.clv.opener_pick_evaluation` calls: they are
`nfl_ats.backtest.walk_forward_backtest` calls (the real weekly-refit
harness backtests use) at two different, real `FEATURE_SETS` entries
(`"market"` vs `"market_context"`). Nothing about population, cutoff, or
push parity depends on which feature set is fit or which harness produces
the probability — only on whether `result`, the chosen line, and the push
rule agree, which they do by construction here.

## ENG-28: the production paths, through the REAL opener-snapshot store

`active_model_production` and `overlay_production` close the gap the section
above describes: they call `nfl_ats.clv.opener_pick_evaluation` directly,
against a real (miniature, commit-safe) market-snapshot store, at the real
production configuration (`feature_profile="weak_stack"`, `regressor="ridge"`,
`ridge_alpha=10.0`, `target="market_residual"` — matches
`artifacts/active_ats_model.json` as read 2026-09-04; frozen into
`nfl_ats.parity.PRODUCTION_MODEL_CONFIG` rather than read live, so this
suite's claims do not drift if the active artifact is retrained). This is the
actual entry point `docs/opener_evaluation.md`'s production archive is built
from, not a stand-in.

**The miniature store.** `tests/fixtures/parity/opener_store/` uses the exact
directory layout `data/market/raw/` does: one directory per committed
snapshot, each with a `manifest.json` (`capture_kind`, `request.season`,
`request.week`, `request.decision_label`, `observed_at_utc`) and a
`quotes.parquet` (one row per game with `nflverse_game_id`, `market`,
`outcome_side`, `home_spread_line`, `bookmaker_key`, `observed_at_utc`,
`commence_time_utc` — the minimal columns `nfl_ats.clv.decision_market_
consensus`/`build_pairing_table` actually read, not the full ~20-column
real-provider schema, which carries several fields no reader here touches).
`home_spread_line` is set directly from the fixture's own
`spread_line_open`/`spread_line_close` (no sign transform), so the store's
opener/close spreads are numerically IDENTICAL to what the other four paths
already use — this is what makes population/push/cutoff comparable, not an
incidental convenience. Only `opener_pick_evaluation`'s own weekly-refit
loop only ever scores weeks present in the store's pairing table, and the
store only needs to cover the two weeks that clear
`MIN_FITTABLE_TRAIN_GAMES` in this fixture (2022 week 4 and 5 — see "Why one
path is inlined" below); every earlier week is still used as plain
walk-forward training history, read from `games.csv` directly, same as
every other path.

**The snapshot-pair requirement, exercised on purpose.** One game --
`2022_05_T7_T8` -- gets a close snapshot but deliberately NO `tue_open`
snapshot. `opener_pick_evaluation`'s pairing table is an INNER join on
`tue_open`/close, so this game is silently absent from both production
paths at BOTH lines, while every walk-forward path (which never touches the
store) still grades it normally. This is the real 1,537-vs-1,552 mechanism
(below), reproduced mechanically inside the fixture rather than only
asserted from a read of the production archive: `tests/test_baseline_parity.py::
test_production_paths_match_market_only_population_where_opener_snapshot_exists`
and `::test_missing_opener_snapshot_excludes_a_game_from_production_paths_only`
cover it.

**One features table, not one per line.** Unlike the walk-forward paths
above (which genuinely need to rebuild the fixture per requested line,
because `walk_forward_backtest` is called once per line),
`nfl_ats.parity._production_features` builds ONE close-era feature table
(matching `game_features_weak_stack.parquet`'s real single-close-era
convention) and passes it to `opener_pick_evaluation` once;
`opener_pick_evaluation` internally overrides `spread_line` per scoring row
for its own `_at_open`/`_at_close` evaluation passes. Requesting
`line="opener"` vs `line="close"` for `active_model_production` therefore
reads two columns off the SAME underlying weekly-refit fit, exactly
matching how one real production week is evaluated at both lines, not two
separately-fit approximations.

**The probability pick rule, not the sign rule.** `opener_pick_evaluation`
computes both a predeclared sign rule (`residual > 0`, its historical
record) and production's actual pick rule (`home_cover_probability >= 0.5`,
the same rule `pool.py`/`backtest.py` use). The production paths here grade
with the probability rule, because that is the rule the card is actually
graded on.

**The weak_stack feature contract.** `margin_feature_columns("market_residual",
"weak_stack")` needs 32 columns (injury sub-splits, QB, roster-continuity,
bias terms) that are not in `MODEL_FEATURE_COLUMNS`. `_production_features`
zero-fills them, exactly like `_features_for_line` already zero-fills
`MODEL_FEATURE_COLUMNS` for the walk-forward paths — a constant zero column
is a legitimate (if uninformative) ridge input; `StandardScaler` sets scale
1.0 rather than dividing by zero variance, and the frozen production ridge
pipeline never engages the group-wise penalty path that would otherwise
require every column to resolve to a known feature-family block.

**The overlay composition.** `overlay_production` calls
`nfl_ats.four_overlay_composition.apply_four_overlay_composition` — the real
joint-OR union of coach-fade, division-revenge-tilt, player-arrests-back-side,
and spread-gap-zone-fade — against `active_model_production`'s own
predictions. The player-arrests member needs a live incident feed and a
freshness-checked snapshot descriptor in production; this fixture supplies an
EMPTY, schema-valid incident table (no arrests in the miniature window) and a
synthetic in-memory `ArrestSnapshot` built directly (bypassing
`load_latest_complete_arrest_snapshot`'s filesystem/freshness checks, which
have nothing to load here). `apply_four_overlay_composition` only reads the
snapshot's identity fields for provenance, never for gating, so this changes
no grading behavior — only that the arrests member is deterministically
never eligible to flip anything in this fixture.

**Why these two paths are NOT folded into `PATHS`.** `tests/test_baseline_
parity.py` keeps `PRODUCTION_PATHS = ("active_model_production",
"overlay_production")` separate from `PATHS`. Unlike the original four
paths (whose populations match by construction), the production paths'
population is EXPECTED to differ from `market_only`'s by exactly the games
the store lacks a paired snapshot for — folding them into the
all-paths-identical parametrization would make that documented, intentional
divergence look like a bug. See "Adding a new comparison path" below for
what this means for a future addition.

**Real-data read-only check (measured 2026-09-04, no artifacts written,
`.agent_tmp/eng28_real_data_check.py`, output to `$TEMP/eng28`).** Ran
`active_model_production`'s real adaptor (`opener_pick_evaluation` at the
real `weak_stack`/ridge-alpha-10 configuration, `min_train_games=50`)
against the real `data/market/raw/` store and the real
`data/processed/game_features_weak_stack.parquet` (REG, season >= 2020,
1,887 rows), alongside a `market_only` population computed with this
suite's own `_eligible_weeks` + vig-free spread-odds availability at the
same `min_train_games=50` floor (the same recipe the section below already
used): **market_only 1,552 games** (reproduces the section below's figure
exactly), **active_model_production 1,479 games**, **symmetric difference 73
— all 73 only in `market_only`, zero only in `active_model_production`**.
Fully explained, not a bug: `active_model_production` requires a matched
Tuesday-opener AND close snapshot pair (the real store's actual coverage
gaps), `market_only` requires only a close-era spread and price, and
`active_model_production`'s population is therefore a strict SUBSET of
`market_only`'s by construction — the zero-only-in-production count is the
clean confirmation of that subset relationship, not a coincidence. (1,479 is
smaller than the previously-read archived 1,537 from
`artifacts/opener_evaluation/20260819T174244Z/`; that archive was generated
2026-08-19 under whatever `min_train_games`/profile were active at the time,
not necessarily today's `min_train_games=50` — a different, unverified
comparison this check does not attempt to reconcile.)

## Why one path is inlined

Every comparison path in this project that fits a model computes its own
eligible-game population as a side effect of its own weekly-refit loop
(`walk_forward_backtest`, `opener_pick_evaluation`). `market_only` fits no
model, so there is no existing "market-only, cutoff-aware" function in the
codebase to call. `nfl_ats.parity._eligible_weeks` mirrors the identical
inlined `cutoff = weekly_games["gameday"].min(); training = completed.loc[
completed["gameday"].lt(cutoff) & completed["home_cover"].notna()]` pattern
verbatim, and `tests/test_baseline_parity.py::
test_market_only_cutoff_agrees_with_walk_forward_backtest_own_cutoff` checks
that this independent expression of the rule agrees with
`walk_forward_backtest`'s own internal behavior. This is the one place in
the suite where a piece of logic (population membership, not grading) is
written twice on purpose, and it is cross-checked precisely because it is
written twice.

## The push-population divergence this suite is named for

`docs/opener_evaluation.md` and `HANDOFF.md` report the frozen active model
at "53.36% on 1,503 games" from "the 1,537-paired-game archive." Those two
numbers are not a typo: the real tracked artifact
(`artifacts/opener_evaluation/20260819T174244Z/per_game.parquet`, read
2026-09-04) has exactly **1,537** rows, of which **34** push at the opener
(`margin_vs_open == 0`) and **30** push at the close (`margin_vs_close ==
0`) — giving 1,503 opener-evaluated and 1,507 close-evaluated games from the
identical 1,537-game population. `tests/test_baseline_parity.py::
test_push_populations_differ_by_line_the_1537_vs_1503_pattern` reproduces
this exact mechanism on the fixture: `active_model`'s `scored_game_ids` are
identical at both lines (8 games), but `pushed_game_ids` are not (2 at the
opener, 1 at the close) — a population divergence fully explained by the
push rule being evaluated at two different lines, not a data bug.
`tests/test_baseline_parity.py::
test_suite_catches_a_deliberately_divergent_population` covers the OTHER
shape of divergence this pattern warns about — one path silently losing
games it should have graded — by dropping two graded rows from one path's
input and asserting the resulting symmetric difference is caught, rather
than being masked by comparing two accuracy percentages that happen to look
similar.

## Read-only real-data check (measured 2026-09-04, no artifacts written)

`active_model`'s real population was read directly from the existing tracked
artifact `artifacts/opener_evaluation/20260819T174244Z/per_game.parquet`
(no re-computation): **1,537 games**. `market_only` was graded through this
suite's own `_eligible_weeks` + `nfl_ats.odds.no_vig_probabilities` +
`nfl_ats.clv.pick_correct` against the real production feature table
(`data/processed/game_features_weak_stack.parquet`, 2020-2025 REG,
`min_train_games=50`, CLOSE line — that table carries one close-era spread
per game, not a paired opener/close): **1,552 games scored, 1,519 evaluated
(33 pushes)**. Symmetric difference between the two populations: **131**
games (58 only in the opener-paired active-model archive, 73 only in the
close-only market-only read, 1,479 in both) — expected and explained by the
active-model path's stricter requirement (a matched Tuesday-opener AND
close snapshot) versus market_only's looser one (any close-era spread and
price), not a bug in either.

## Adding a new comparison path

1. Write a `_grade_<name>(frame, line, min_train_games, ...) -> PathResult` in
   `nfl_ats/parity.py` that:
   - calls the REAL production function that already grades that path (do
     not recompute `ats_margin`/`home_cover`/correctness by hand — reuse
     `nfl_ats.features.add_ats_outcomes` and `nfl_ats.clv.pick_correct`, or
     delegate to an existing harness such as `walk_forward_backtest` or
     `opener_pick_evaluation`);
   - returns a `PathResult` with `scored_game_ids` (every game the path
     attempted, pushes included), `pushed_game_ids`, `evaluated_game_ids`,
     `skipped_weeks`, and `correct_by_game` (built via `pick_correct` so the
     push rule is provably the same one every other path uses).
2. Register it in `grade_games`'s dispatch and `_PATHS`.
3. Decide whether the new path's population is EXPECTED to match the
   existing paths exactly, or is allowed to legitimately differ (e.g.
   because it reads a real snapshot store with its own coverage gaps, the
   way `active_model_production`/`overlay_production` do):
   - If it must match exactly, add the new path name to `PATHS` in
     `tests/test_baseline_parity.py` — the existing population/cutoff/push
     tests are parametrized over `PATHS` and will immediately check the new
     path against the others with no further changes.
   - If it is allowed to differ, do NOT add it to `PATHS` (folding it in
     would make the all-paths-identical assertion fail on an intentional,
     documented divergence, or — worse — get weakened to stop failing and
     silently lose its power to catch a REAL divergence). Instead give it
     its own tuple (see `PRODUCTION_PATHS` above) and write comparison tests
     that state the expected relationship explicitly (e.g. "identical to
     `market_only` once you exclude games without a paired snapshot", "a
     game excluded from one production path is excluded from all of them"),
     the same shape `test_production_paths_match_market_only_population_
     where_opener_snapshot_exists` and
     `test_missing_opener_snapshot_excludes_a_game_from_production_paths_only`
     use.
4. If the new path needs fixture columns the current 60-row fixture does not
   have (a new market signal, a new overlay trigger), extend
   `tests/fixtures/parity/games.csv` and `nfl_ats.parity.FIXTURE_COLUMNS`
   together, and re-verify `test_populations_are_nontrivial_not_the_whole_fixture`
   still passes (the `MIN_FITTABLE_TRAIN_GAMES` floor must still bite). If it
   needs its own snapshot-style store (like `opener_store/`), regenerate it
   deterministically from `games.csv` rather than hand-editing committed
   parquet/JSON files.
