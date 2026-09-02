# Postseason coverage (FND-15)

Written 2026-08-17. The confirmed Splash pool format requires forced ATS
picks for **all 13 playoff games**, but every feature table in this project
was regular-season only (`game_type == "REG"` filtered at the schedule,
team-stat, play-by-play, injury, roster, and snap layers). This document
records how postseason rows were added without disturbing anything the
frozen model's grades depend on.

## The invariant: REG rows are bit-identical

The 52.50%-at-openers result (`docs/opener_evaluation.md`) was one
predeclared look at the frozen active model. If adding playoff rows changed
any regular-season feature value, the retrained model would be a *different*
model whose opener performance is unknown — the look would have been spent
on a model we no longer run. So the extension is built around one invariant:

> Given the same input snapshots, the REG rows of the feature table are
> byte-for-byte identical whether or not postseason rows are requested, and
> every training/evaluation path consumes REG rows only.

The feature-table SHA-256 still changes (new rows exist), so the active
manifest desynchronizes until `margin-backtest` and `margin-predict` are
re-run — by design. Because REG rows are unchanged, the re-run reproduces
the same evaluation numbers and the same model.

## Design: two-pass feature build

`build_game_features(..., include_postseason=True)` (the CLI default for
`build-features`) runs the existing REG-only build unchanged (pass 1), then
replays the entire build with WC/DIV/CON/SB games included in every rolling
state — Elo, graph/schedule ratings, and EWM team states (pass 2) — and
keeps **only the postseason rows** from pass 2:

- REG rows come from pass 1, so playoff results never leak into the Elo,
  graph, or team-state histories REG rows see (a playoff result would
  otherwise perturb the *following* season's early-week features).
- Postseason rows see everything strictly earlier, including earlier
  playoff rounds: a Super Bowl row's Elo and team states reflect both
  teams' Wild Card through Conference results. Leak-safety is inherited
  from the strictly-earlier-date lookups of the existing builders.
- The cyclic week encoding clamps at week 18 (`week_sin`/`week_cos` of
  `min(week, 18)`), so playoff weeks read as "late season" instead of
  aliasing back to September. A no-op for REG rows in both eras (playoffs
  are weeks 18–21 before 2021, 19–22 after).
- The Super Bowl's neutral site flows through the existing
  `location == "Neutral"` handling (Elo home-advantage zeroed,
  `neutral_site = 1`).

## Training and evaluation stay REG-only, explicitly

Previously REG-only-ness was an implicit upstream guarantee. It is now an
explicit guard, `nfl_ats.modeling.regular_season_rows`, applied at:

- `modeling.fit_cover_model`, `margin.fit_margin_model`,
  `margin.fit_market_baseline` (training frames),
- `backtest.walk_forward_backtest` and `outcomes.walk_forward_outcomes`
  (entire frame — scored evaluation rows remain REG-only),
- `outcomes._target_and_models_for_week` (training only — the **target**
  week may be a playoff week; that is the serving path),
- `clv.opener_pick_evaluation` (the predeclared opener metric's game set
  must never widen) and `clv.upcoming_week` (predict-close remains a
  regular-season pilot; in January it fails closed rather than silently
  switching to playoff games).

Frames without a `game_type` column pass through untouched, so research
code operating on synthetic frames is unaffected.

`prediction_safety` now validates `game_type` when present: values must be
in {REG, WC, DIV, CON, SB} and a weekly card must contain exactly one game
type (one postseason round per card).

## Serving playoff weeks

`margin-predict --season <S> --week <19..22>` (18–21 for pre-2021 seasons)
works once the feature table carries postseason rows: target selection is
by (season, week), training excludes postseason rows, and the safety
contract accepts the single-round card. Spreads for upcoming playoff games
arrive through the normal nflverse schedule feed; the command still fails
closed when any target spread is missing.

## Player/PBP-enriched profiles (the active `player` profile)

The enrichment loop passes postseason base rows through safely today:
every state update inside `enrich_with_player_features` is keyed by
`game_id` against the snapshot contents, so with the current REG-only
player snapshots, playoff rows emit features from carried end-of-season
state and update nothing (REG rows bit-identical by construction).

Current v1 semantics for playoff rows in the `player` profile:

- QB identity/form, lineup continuity, roster continuity: carried from the
  last regular-season observation (a Super Bowl row uses week-18-era
  player state).
- Injury unavailability: playoff report weeks are absent from REG-only
  snapshots, so visible injuries are empty → unavailability features are
  0.0 (the neutral "no directional information" convention, `diff_* = 0`).
- Base features (market, Elo, graph, team states): fully postseason-aware
  from the two-pass build.

The snapshot contracts (injuries, rosters, snaps, player stats, PBP, role
actions) accept an `include_postseason` opt-in at write time (the ingest
CLI flag of the same name), recorded in each snapshot manifest; the
verified nflverse codes are `game_type` ∈ {REG, WC, DIV, CON, SB} for
injuries/rosters/snaps and `season_type` ∈ {REG, POST} for player stats
and play-by-play, with unknown codes raising on postseason-inclusive
writes. Every snapshot **loader** re-filters to REG by default
(`load_pbp_snapshot`, `load_player_snapshot`, `load_player_value_snapshot`,
`load_role_actions_snapshot`) — this closed three real leak paths where a
postseason-inclusive file would otherwise have fed playoff plays or rows
straight into the pbp/QB/participation/role feature builds. Refreshed
snapshots that contain postseason rows therefore cannot change any feature
until a deliberate, separately-declared upgrade flips the read side
(candidate follow-up: postseason-aware player state between playoff
rounds).

## What was verified

- Real-data check (2026-08-17, snapshot `20260812T130036Z`): rebuilding
  with `include_postseason=True` reproduces the shipped
  `game_features.parquet`'s 4,703 REG rows **exactly**
  (`assert_frame_equal(check_exact=True)`), and adds 199 postseason rows
  for 2009–2025 (80 WC / 68 DIV / 34 CON / 17 SB — six Wild Card games
  per season from the 2020 field expansion). Every model feature is 100%
  populated on postseason rows except `temp`/`wind` (70.9%, the usual
  dome/missing-weather gap).
- Unit coverage: `tests/test_postseason.py` locks the bit-identity, the
  training-immunity of every fit/backtest path, playoff-week serving, and
  the between-rounds state updates; snapshot-contract tests lock the
  write-side opt-in and read-side REG defaults.
- The full-chain gate (base → pbp → player enrichment, REG rows compared
  exactly against the shipped `game_features_player.parquet`) additionally
  caught a **pre-existing** reproducibility break unrelated to postseason
  work: the 2026-08-13 availability refactor had silently changed
  `fixed_unavailability`'s practice-status fallback, shifting injury
  features on 18 games from 2010–2015. Restored to the original
  substring-matched heuristic and pinned by a regression test
  (`test_fixed_unavailability_is_bit_faithful_to_the_original_heuristic`);
  the postseason build's REG subset was byte-identical to a REG-only
  rebuild both before and after that fix.

### January-style operations rehearsal (2026-09-02)

**Measured** from `data/processed/game_features_player.parquet` with
`pd.read_parquet`: season 2025 contains exactly 13 postseason rows and 13
unique games -- week 19 `WC` has 6, week 20 `DIV` has 4, week 21 `CON` has 2,
and week 22 `SB` has 1 -- with zero missing `spread_line` values. The table has
4,902 rows total; its adjacent manifest is
`data/processed/game_features_player.manifest.json`, and each rehearsal
artifact recorded the feature-table SHA-256
`f38e0766b6f1ca6092fbe6b49f1ec63cabe67f701ca6920b094d9c0b512be12b`.

**Measured** by running the real CLI once for each round, with both writable
roots deliberately isolated from production artifacts and registries:

```powershell
$env:NFL_ATS_ARTIFACTS_DIR='artifacts/postseason_rehearsal_2025_round_metadata'
$env:NFL_ATS_REGISTRY_DIR='artifacts/postseason_rehearsal_2025_round_metadata/registry'
foreach ($rehearsalWeek in 19,20,21,22) {
  .\.tools\uv.exe run --no-sync nfl-ats margin-predict `
    --season 2025 --week $rehearsalWeek `
    --features data/processed/game_features_player.parquet `
    --feature-profile player
}
```

**Measured** from the four resulting `metadata.json`,
`prediction_safety.json`, and `predictions.csv` files: all four commands exited
zero; round metadata was `WC` / `DIV` / `CON` / `SB`; the commands served 6 /
4 / 2 / 1 unique games and 30 / 20 / 10 / 5 five-method prediction rows; all
four safety audits returned `PASS` with zero warnings and included both
`card_scope` and `training_cutoff`; and the default line sweeps contained 306 /
204 / 102 / 51 rows (663 total). This was a serving-contract rehearsal only:
no outcome-derived accuracy, return, or policy comparison was computed.

**Measured** in the first isolated run: the prediction CSV and safety contract
carried the correct round, but top-level forecast metadata exposed only numeric
season/week. That made an operator reopen the prediction table to distinguish a
round, and postseason week numbers differ across eras. The CLI now copies the
safety-validated single `game_type` into `margin-predict` metadata, and
`activate_matching_ats_model` carries it into the active manifest's
`weekly_forecast` pointer. The end-to-end regression
`test_margin_predict_cli_preserves_postseason_round_in_artifact_metadata`
serves a synthetic Wild Card card through the real CLI, verifies the artifact
and safety files, and verifies that an unlinked isolated rehearsal creates no
active manifest.

**Measured** after the final rehearsal: the isolated root contains no
`active_ats_model.json`; the production `artifacts/active_ats_model.json`
SHA-256 remained
`4c4884e3b6234433130b982309dc133ef79ba31d4f746b19cb5a394b9e325d02`,
the same value captured before the commands; and `git status --short --
CURRENT_PREDICTIONS.md artifacts/active_ats_model.json` printed no changes.
The commands did not invoke publishing, prospective recording, or CLV
recording.

## Deliberately not done here

- No playoff-game accuracy evaluation was run. Grading the frozen model on
  historical playoff games is a **new predeclared look** (~65 games with
  archived openers, 2020–2025) and should be declared through the
  experiment registry before anyone computes it.
- Playoff rows are excluded from training everywhere. Training on playoff
  games is a candidate experiment, not a default.
- `predict-close`/CLV pilots stay regular-season; extending them to
  playoff closes would widen a declared pilot mid-stream.
