# Public-betting fade/follow battery: frozen predeclaration

Written and frozen **before** `scripts/public_betting_battery_screen.py` is
executed or any effect is computed. Per this task's instruction and
`docs/public_betting_sourcing.md` section 7's "Predeclared next-step
experiment", this is a **mined battery on a sparse archive** -- coverage
tops out around 34% of REG games in the best-covered season (section 5 of
the sourcing doc) -- and every cell below is exploratory/backfill-quality by
construction, not a confirmation look. Nothing here calls
`nfl_ats.rotation.assign_window`/`record_look`; no rotation window is drawn.

## Binding closing-grounds taxonomy (verbatim, AGENTS.md)

> An interval or CI that contains zero is NEVER grounds to reject, fail, or
> close an experiment. At this evaluator's ~2-point resolution, "contains
> zero" is the EXPECTED outcome for a real small signal. Only two grounds
> ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
> (whole interval on the wrong side of zero) or zero split-half reliability;
> (2) bounded by a positive control proven able to detect an effect that
> size. Everything else is `unresolved_below_power`: record it with
> `nfl-ats weak-signals record`, report `probability_positive`, never the
> binary "contains zero". The registry hard-rejects inadmissible closures;
> if a record command errors, the verdict is wrong, not the validator.
> Never use 95%/0.90 as a decision bar -- decide on expected value.
> Within-week game correlation is ZERO by owner mandate.

Every cell below defaults to `unresolved_below_power` regardless of point
estimate sign. A cell is only proposed `refuted_mechanism` /
`wrong_sign_resolved` if BOTH the week-blocked and season-blocked bootstrap
95% intervals sit entirely below zero (a genuinely resolved wrong sign, not
merely a negative point estimate). No cell in this battery has a positive
control, so `bounded_by_control` is not available to any cell here.

## Population construction (frozen)

1. Source archive: `data/raw/public_betting/20260820T111148Z/actionnetwork/
   index.parquet` (1,658 rows, **read** this session).
2. Match each row to `data/processed/game_features.parquet`'s REG-season
   schedule on `(away_team, home_team)` with `|kickoff - start_time_utc| <=
   72h`, identical to `scripts/ingest_public_betting.py`'s
   `build_coverage_report` join (sort by delta, drop duplicates) -- not
   re-derived, the same method reused so the population matches what
   section 5 of the sourcing doc already measured.
3. Restrict to captures with `capture_ts < kickoff` (pregame only).
4. **One row per game**: the single capture with the MAX `capture_ts` among
   that game's pregame captures ("latest capture before kickoff", per this
   task's instruction). `staleness_hours = (kickoff - capture_ts) /
   3600`, its distribution reported for this population before any effect
   is computed.
5. Seasons 2018-2025 inclusive, `game_type == "REG"`, a resolvable close
   (`spread_line` present) and a completed result. 2026 excluded (archive
   only has preseason captures so far, section 4/5 of the sourcing doc).
6. Grading at close reuses `game_features.parquet`'s own already-computed
   `ats_margin` / `home_cover` columns (`ats_margin = result - spread_line`,
   home covers iff `ats_margin > 0`) -- the project's one existing
   convention (`src/nfl_ats/cfb_features.py` line ~858,
   `src/nfl_ats/clv.py`), not re-derived. Pushes (`ats_margin == 0`) are
   excluded from every accuracy cell, counted separately.
7. Opener-grade variant (cell A only, 2020-2025 subset): the archive's own
   Tuesday-opener quote store only starts 2020 (`docs/public_betting_
   sourcing.md`, `data/market/raw`). Pulled via `nfl_ats.clv.
   build_pairing_table(labels=("tue_open",), schedule=features)` -- the
   same read-only opener-graded path `scripts/novig_diagnostics_screen.py`
   / `scripts/odds_microstructure_battery.py` import, called here for its
   `tue_open` consensus line only, no model fit. `margin_vs_open = result -
   tue_open_home_spread`; graded the same way as the close (positive =
   home covers, zero excluded as a push).
8. Model source for cell C: `artifacts/margins/20260820T004951Z/
   predictions.parquet`, filtered to `method == "market_residual"` and
   `model_name == "ridge"` -- **measured**: its `provenance.configuration_
   sha256` (`d5259477727e0cdd84c5c3e17200c71002697f31f83a808401b75d2ddd29eb05`)
   matches `artifacts/active_ats_model.json`'s
   `evaluation_configuration_sha256` exactly, so this is the frozen
   production model's own close-grade weekly-refit forced pick for every
   2018-2025 REG game (2,127 games), reused read-only rather than refit
   inline. `artifacts/` is local/gitignored and may be absent in a fresh
   clone (AGENTS.md); regenerating it is `nfl-ats margin-backtest` with
   `feature_profile=weak_stack`, `regressor=ridge`, `ridge_alpha=10.0`,
   `start_season=2018`, `min_train_games=500`, `calibration_method=none`,
   `probability_method=gaussian` (read from that run's own
   `metadata.json`). Production's own pick rule (`home_cover_probability >=
   0.5`) is reused, matching `pool.py`/`nfl_ats.clv.opener_pick_evaluation`'s
   `*_probability_rule` convention.

## Cells (frozen, before any number is computed)

**Cell A -- fade-heavy-public.** Condition: both `spread_home_bet_pct` and
`spread_away_bet_pct` present and `max(home_pct, away_pct) >= 70.0`. The
"fade side" is whichever side has the LOWER bet%. Metric: forced-pick
accuracy of "fade side covers" minus 0.50, week-blocked (`season, week`) and
season-blocked bootstrap, 20,000 resamples, seed `20260818` (this project's
odds-battery seed, reused not re-chosen), `P+` = fraction of resamples with
the metric positive.
- A.1 `public_betting_battery_fade_heavy_public_close`: graded at close,
  2018-2025.
- A.2 `public_betting_battery_fade_heavy_public_opener`: same fade rule,
  graded at `tue_open` instead, restricted to games with a resolvable
  opener line (2020-2025 subset, item 7 above).

**Cell B -- follow-sharp-divergence** (era2 only, `era ==
"era2_scoreboard_response"`, since money% does not exist in era1). Per
side, `gap = money_pct - bet_pct`. If `gap_home >= 15.0`: money side = home.
Elif `gap_away >= 15.0`: money side = away. Else excluded. Metric: forced-
pick accuracy of "money side covers" minus 0.50, graded at close, same
bootstrap spec as cell A.
- B.1 `public_betting_battery_sharp_divergence_close`.

**Cell C -- public-vs-our-model interaction.** Population: cell A's
fade-heavy games (`max bet% >= 70`, item above) intersected with the model
source (item 8), non-push on both `ats_margin` and the model's own pick
(`home_cover_probability != 0.5`... i.e., production's forced pick is
always defined). `public_side_home = home_bet_pct > away_bet_pct`.
`model_pick_home = home_cover_probability >= 0.5` (production's actual
rule). `against = public_side_home != model_pick_home` (public is heavy on
the side the model does NOT hold).
- C.1 `public_betting_battery_model_interaction_against`: model's forced-
  pick accuracy minus 0.50, restricted to `against == True`, same bootstrap
  spec, graded at close.
- C.2 `public_betting_battery_model_interaction_diff`: on the FULL
  intersected population (both `against` and `not against` rows), the
  bootstrap of `mean(correct | against) - mean(correct | not against)`,
  same week/season block spec -- directly answers "do we win more when the
  public is heavy against us" as a paired comparison rather than a single
  cell's distance from 50%.

Five cells total (A.1, A.2, B.1, C.1, C.2). Every cell is recorded via
`nfl-ats weak-signals record` regardless of what it shows, with
`--probability-positive` from the WEEK-blocked bootstrap (this project's
primary block per prior batteries) and `--effect-units accuracy_points`
(the registry's units are PERCENTAGE POINTS, so a fraction like 0.02 is
recorded as `2.00`, per `src/nfl_ats/weak_signals.py`'s own comment).
`--season-start`/`--season-end` reflect each cell's actual population
(2018/2025 for A.1, B.1, C.1, C.2; 2020/2025 for A.2).

No cell here is a rotation-registry confirmation look (mined battery, per
this document's opening paragraph); none calls `nfl_ats.rotation.
assign_window`/`record_look`.
