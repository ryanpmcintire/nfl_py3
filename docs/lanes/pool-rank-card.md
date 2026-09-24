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

- 2026-09-23: the pool's own field split per game is public after lock (Splash contest Statistics -> General, per week). Weeks 1-2 aggregate side counts and Best Pick counts saved to `data/splash/field/2026_week0{1,2}_field_distribution.tsv` (32 games, 250 entries). Per-entrant pick lists are also shown there. Owner ruled 2026-09-23 that pool handles are anonymous and a per-entrant panel is approved; the in-page pull is still blocked by the harness permission classifier until the owner adds an allow rule for the browser JavaScript tool. After each lock, capture the week's aggregate split the same way.

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
- 2026-09-23 steps A-D DONE. Script `scripts/pool_field_share_fit.py`
  (no comments/docstrings); artifact
  `artifacts/pool_field_share/<ts>/{results.json,games.csv}` (latest run
  20260923T193620Z-ish, re-run to regenerate). Served probability
  recovered from `artifacts/margin_predictions/2026-week-0{1,2}-*/
  recommendations.csv`, one row per game, taking each game's own latest
  pre-kickoff snapshot (no leakage) -- `home_cover_probability` column is
  the served model probability, `model_name`/`method` columns show which
  challenger the served card used per game (mostly `ridge`/
  `market_residual`). n=32 games (2026 weeks 1-2, all with public split,
  market odds, served probability, and real field share).
  - **measured** MAE (share points) vs real home_share, n=32: public
    split 0.174, market vig-free-odds home probability 0.158, served
    model home_cover_probability 0.170. All three proxies barely beat a
    flat 0.5 guess (real home_share std is small at this n).
  - **measured** Pearson r [95% bootstrap CI], n=32: public split 0.133
    [-0.30, 0.57]; market vig-free 0.138 wrong-signed [-0.53, 0.21];
    served probability -0.078 [-0.41, 0.27]. Every interval crosses
    zero -- `unresolved_below_power` per AGENTS.md, not a rejection of
    any of the three signals, just that n=32 games cannot resolve an
    effect this small. No registry cell recorded (same reasoning as
    unit 1: no valid rank/share effect-unit for pooling yet).
  - Simplest field model: OLS `home_share ~ public_split +
    market_vig_free_home`, weeks as folds (fit one week, score the
    other). **measured**: trained-on-week2/scored-on-week1 MAE 0.170,
    r -0.318; trained-on-week1/scored-on-week2 MAE 0.155, r 0.036. The
    market coefficient flips sign between folds (+3.79 vs -3.83) --
    the 2-fold, 16-games-per-fold fit is unstable, consistent with the
    zero-crossing correlations above. Do not treat these coefficients as
    a real field model; they are a first pass sized by n=32.
  - Rank replay with this fitted (out-of-fold) field share substituted
    for the constant-lean proxy, `ENTRANTS=249` (measured from the field
    TSV's `entries=250`, minus our own entry), custom per-game-share
    simulator/greedy search written inside the new script (no
    `pool.py`/`FieldModel` edits -- `FieldModel.public_lean` is one
    scalar for the whole field, confirmed at `src/nfl_ats/pool.py:230`,
    so it cannot take a per-game vector without a src/ change, which is
    out of scope):
    - Week 1: 1 decisive flip out of 16 games -- DEN@KC. Fitted field
      said the field leans away (DEN, 38.7% home share) while the served
      card correctly picked home (KC) to cover, and KC did cover. The
      rank-optimal card under the fitted field would have flipped to
      DEN and made the realised rank **worse** (served rank 28.6 vs
      flipped-to-optimal rank 62.9, a -34.3 rank loss) -- the flip was
      wrong because the fitted field share (38.7% home) undershot the
      real field share for that game (41.6% home) enough to flip the
      side, and the unstable 2-fold model above explains why. **Do not
      serve this flip; it would have hurt.**
    - Week 2: 0 decisive flips -- the fitted field agreed with the
      served card on every game; realised rank unchanged (7.50).
  - **Decision implication**: with only 32 games and an unstable 2-fold
    field-share fit, this round finds no case where the fitted-field
    rank-optimal card should have overridden the served card -- the one
    disagreement (week 1 DEN@KC) would have cost rank, not gained it.
    The served card was not changed. This does not close the
    standing-aware/fitted-field idea (interval-crosses-zero is not
    grounds for rejection); it says the field-share model needs more
    weeks (more n) before its flips are trustworthy enough to act on.
- 2026-09-23 (superseded) unit-2-prerequisite subagent HOLD notes below,
  kept for the data-source specifics they still document:
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
- CONFIRMED 2026-09-23: `artifacts/margin_predictions/2026-week-0{1,2}-*/
  recommendations.csv` does contain the served home-cover probability
  (`home_cover_probability`, one row per game, picked per the game's own
  latest pre-kickoff snapshot).
- Re-run `scripts/pool_field_share_fit.py` once more weeks of real field
  share TSVs accumulate (capture after each week's lock, same as unit 1);
  a 2-fold, 32-game field model is too unstable to trust its flips (see
  State/Tried above). Do not serve the DEN@KC-style flip pattern without
  a larger, stabler fit.

- 2026-09-24 **read**: `scripts/capture_scheduler.py` grep for
  splash/pool -- only `splash_board_tue` (tue 12:05, runs
  `check_splash_board.py`) exists as a scheduled pool job. It only
  validates that a board/LINES capture is present for `weekly_lock`
  (`data/splash/2026_week0N_*.json`); it captures nothing about field
  pick shares. The field-share TSVs
  (`data/splash/field/2026_week0{1,2}_field_distribution.tsv`, header
  `captured_at_et=... capture_method=browser_read_by_agent`) are a
  manual per-week browser pull, not scheduled anywhere -- confirms the
  2026-09-23 note. `data/splash/` also holds board captures for week 1-3
  (`2026_week03_20260922_2019.json` exists, post-Tuesday-lock), but
  **no `field/2026_week03_field_distribution.tsv` yet** -- the week-3
  pick-share pull has not been done. Real field-share coverage is still
  exactly 2 weeks / 32 games, unchanged since the 2026-09-23 fit; no
  re-run was warranted this round (same n, same `artifacts/pool_field_share/
  20260923T193620Z/{results.json,games.csv}` is current).
- Splash's own aggregate picks/best_picks counts are fixed at each
  week's Tuesday-noon pool lock and do not move afterward (owner,
  2026-09-08: entries lock noon Tue); reading them any time after that
  lock (even post-game, as both captures were) is not a leakage
  violation for offline field-model fitting -- the value itself was
  pregame-determined, only our read of it was late. This is distinct
  from using them as a live production input before a game's own lock,
  which remains banned.
- **Unit 2 predeclaration draft** (params to fix now, before seeing week
  3+ data, per LOSO/no-in-sample-gates): field-share model form = OLS
  `home_share ~ public_split + market_vig_free_home` (already used for
  the 2-fold check); evaluation = leave-one-week-out across all weeks
  once >=3 are captured (not the current 2-fold swap); promotion
  question = does the LOWO-fitted field's rank-optimal card ever
  disagree with the served card on a decisive game, and does that
  disagreement net positive realised rank across held-out weeks (not
  per-game MAE/r, which is already `unresolved_below_power` at n=32 and
  won't resolve with one more week either). Do not run
  `greedy_rank_card()` against the fitted field again until >=3 weeks
  of real shares exist (currently 2) -- next agent's first move each
  week is capturing `field_distribution.tsv`, not re-fitting on the
  same n.
