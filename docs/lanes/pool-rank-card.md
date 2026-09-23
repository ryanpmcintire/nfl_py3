# pool-rank-card

## Goal
POOL-01: choose the card that maximises expected pool finishing position given the served probabilities and a fitted field, replayed over past weeks.

## State
- 2026-09-16: queued by owner; ROADMAP row written.
- Unit 1 DONE 2026-09-16 (subagent-measured, parent-verified by rerun).
  Script `scripts/pool_rank_replay.py`; artifact
  `artifacts/pool_rank_replay/` (`results.json`, `weekly.csv`). No
  registry cell: no valid effect-unit exists for ranks/week (forcing one
  would poison unit-based pooling); outcome lives here and in the
  artifact.

- 2026-09-23: the pool's own field split per game is public after lock (Splash contest Statistics -> General, per week). Weeks 1-2 aggregate side counts and Best Pick counts saved to `data/splash/field/2026_week0{1,2}_field_distribution.tsv` (32 games, 250 entries). Per-entrant pick lists are also shown there, but compiling them was blocked as personal-data handling; any per-entrant panel needs an explicit owner decision. After each lock, capture the week's aggregate split the same way.

## Tried
- 107 season-weeks replayed, one greedy rank-optimal rule vs served
  (LOSO served probabilities, 4000-sample Poisson-binomial field,
  100 entrants). Mean gain +1.47 ranks/week [-0.00, +2.84], P+ 0.974.
  Decisive weeks 64/107: 25 better / 16 tied / 23 worse (+2.46 on
  decisive). Per-season: +3.37 / +0.73 / +3.33 / +2.95 / -1.23 / -0.21.
  Served 841 vs rank-optimal 848 correct of 1,503.
- Field model is the weak leg: public splits cover 321/1,503 games,
  rest fall back to the opener favorite; pool published picks
  unavailable. Interval touches zero: unresolved_below_power by default.

## Next
- Standing-aware variant (unit 2) only with a fitted field on real pool
  picks; otherwise remeasure when public splits cover more games.
- 2026-09-23 unit-2-prerequisite subagent HOLD (hit 50-tool cap mid
  investigation, no script/artifact written yet). Data sources located,
  a fresh agent should continue directly from step (A) below:
  - Real field share: `data/splash/field/2026_week01_field_distribution.tsv`
    and `..._week02_...` (comment line, then header
    `away home away_score home_score team line result picks best_picks`,
    two rows per game, one per side). home_share = home_picks /
    (home_picks + away_picks).
  - Public split proxy: the snapshot `pool_rank_replay.py` defaults to
    (`data/raw/public_betting/20260820T111148Z/index.parquet`) has ONLY
    placeholder zeros for season-2026 week-1 real games and no week-2 rows
    at all -- not usable. Use `data/raw/public_betting_live/<ts>/index.parquet`
    instead: real non-zero splits, full 16-game coverage per week. Best
    per-game pre-kickoff snapshots: week1 `20260913T162822Z` (16 rows,
    season 2026 week 1), week2 `20260919T160012Z` or `20260920T161833Z`
    (16 rows, season 2026 week 2); earlier snapshots exist back to 20260906
    for week1 if a "closest-to-kickoff" join across all snapshots is wanted.
  - GOTCHA (verified): home/away labeling is NOT always consistent between
    the field TSV and `public_betting_live` for the same physical game
    (e.g. week2 BUF@DET: field has away=DET home=BUF, betting_live has
    away=BUF home=DET; same swap on the GB/NYJ week2 game). Other games
    match unswapped (e.g. CAR/ATL). Match games by team-pair identity
    (normalize JAC->JAX, same alias dict as `src/nfl_ats/public_betting_live.py`
    line 18), trust the field TSV's home team as authoritative (it matches
    the real result), and pick `spread_home_bet_pct` or `spread_away_bet_pct`
    from the betting row depending on whether its own home_team equals or
    is swapped vs. the field TSV's home_team.
  - Served model probability for the LIVE 2026 season is NOT in
    `build_fit_population()` (`src/nfl_ats/pick_probability_fit.py`) --
    that population is the 2020-2025 LOSO backtest only (fold_coefficients
    keys are only seasons 2020..2025; confirmed empty result when filtered
    to season 2026). Note also: `build_fit_population` currently fails to
    import outright because of an in-flight, uncommitted src/ refactor
    (git status: M `displayed_confidence.py`/`board_content.py`/`card_view.py`)
    that removed `served_strength_bands` from `displayed_confidence.py`
    while `public_board.py` (imported transitively) still does
    `from nfl_ats.displayed_confidence import served_strength_bands` at
    line ~63 -- ImportError. Do NOT fix src/ (out of scope). Safe runtime
    workaround verified: before importing, run
    `import nfl_ats.displayed_confidence as dc; dc.served_strength_bands = lambda *a, **k: None`
    then `from nfl_ats.pick_probability_fit import build_fit_population`.
    This only patches the module object in-process, no file changes.
  - (A) NEXT STEP: the live-season served probability likely lives in
    `artifacts/margin_predictions/2026-week-01-<timestamp>/*.parquet` (and
    the week-02 equivalent). Confirmed real: `artifacts/prospective/challenger_decisions.parquet`
    has 1306 rows for season 2026 weeks 1-2 (every non-served challenger
    overlay's pick per game) whose `source_artifact` column names exactly
    these margin_predictions directories (verified
    `2026-week-01-20260905T141453Z` exists on disk with that literal
    timestamp). Was about to open that parquet's columns when the tool
    cap hit -- do this first. Look for a `home_cover_probability` (or
    `_at_open`) column; there are multiple margin_predictions snapshots
    per week as the card refreshed, so either use the one matching each
    game's actual `source_artifact` in challenger_decisions, or the
    latest snapshot before that game's kickoff. Cross-check any recovered
    probability against `artifacts/prospective/pick_revisions.parquet`
    (`pick_revision_ledger_path` = `artifacts/prospective/pick_revisions.parquet`,
    only 24 rows = revision EVENTS, not the full 32-game baseline, but has
    `previous_home_cover_probability`/`new_home_cover_probability` for
    sanity-checking whatever margin_predictions gives you).
  - (B) Build the three proxies vs. real home_share and report MAE (share
    points) + Pearson r with an interval (n=32 is small; bootstrap or
    normal-approx CI; flag `unresolved_below_power` per AGENTS.md if the
    interval crosses zero -- that is not grounds to reject anything):
    public-split proxy, market spread size (`decision_home_spread` is
    already available in challenger_decisions, one row per game is enough),
    served model probability (from step A).
  - (C) Fit the simplest field model (logistic share on spread + public
    split) with week1/week2 as the two folds (fit one, score the other);
    save as a reusable function in new script `scripts/pool_field_share_fit.py`
    (no comments/docstrings).
  - (D) Re-run a `pool_rank_replay.py`-style replay for these two weeks with
    the fitted per-game field share swapped in for the current constant-lean
    proxy, report decisive-game differences from the served card. Note
    `pool.py`'s `FieldModel`/`simulate_pool_finish` currently take one
    scalar `public_lean` for the whole field, not a per-game vector --
    reusing `greedy_rank_card()`/`simulate_pool_finish()` from
    `pool_rank_replay.py` as-is will need either a per-game FieldModel
    workaround written inside the new script (no src/ edits) or a
    simplified direct comparison; decide pragmatically, no src/ changes.
  - No files written yet this round: no `scripts/pool_field_share_fit.py`,
    no `artifacts/pool_field_share/` artifact. All of the above is
    investigation only.

## Open
- Whether the owner wants rank-scale units added to the registry
  (feeds ENG-46 unit 2 scope).
- Whether `artifacts/margin_predictions/2026-week-0{1,2}-*` actually
  contains the served home-cover probability, or another artifact does --
  unconfirmed, first thing to check next round.
