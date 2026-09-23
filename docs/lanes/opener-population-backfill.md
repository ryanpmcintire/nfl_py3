# Opener population backfill

## Goal
Six added-term units all underpowered vs the opener fit population (1,503
games, 2020-2025) from `src/nfl_ats/pick_probability_fit.py
build_fit_population` (src/nfl_ats/pick_probability_fit.py:187). Unit 1: find
whether pre-2020 NFL opener lines exist locally with enough provenance to
grade at the opener, and what restricts opener-evaluation to 2020+.

## State
Unit 1 DONE (read-only; no new artifact written; nothing served changed).

Unit 2 DONE. Predeclared and built: `scripts/extended_fit_population.py` (new,
ruff format+check clean), run once for real,
`artifacts/extended_fit_population/20260923T205041Z/population.parquet` +
`summary.json`. **3,734 games 2011-2025** vs the served 1,503 (2020-2025) --
**+2,231 games gained**, opener_source = `sbr_proxy` (2011-2019, n=2,231) vs
`tue_open` (2020-2025, n=1,503, identical to `build_fit_population`'s own
current output, reused unchanged including its discrete
`home_cover_probability_at_open`). `composition_flag_sum` nonzero on 50.86%
of all rows (computed for every row via the same
`signed_composition_flags`/PBP `build_flag_table` call `build_fit_population`
uses, PBP covers 2009+). `market_move`/`market_move_available` are 0 for all
2,231 sbr_proxy rows (no upstream archive pre-2020, as Unit 1 found) and real
for the 1,503 tue_open rows. `home_covered` at the (proxy or true) opener,
pushes dropped both arms.

**model_logit provenance, stated plainly**: 2020-2025 rows use the served
discrete mass-preserving-lattice probability
(`base_probability_policy=discrete_conditional_non_push_v1`), unchanged.
2011-2019 rows reuse the PRESERVED ARTIFACT
`artifacts/sbr_era_opener_eval/20260819T233013Z/scored.parquet`'s
`home_cover_probability_at_open_proxy` -- the walk-forward weekly-refit
production-model probability (`weak_stack`/ridge/alpha=10.0/market_residual,
matches today's active model exactly) graded at SBR's proxy Open, computed by
`scripts/proxy_opener_replication.py`'s `sbr_proxy_pick_evaluation`
(deleted from the tree in the 2026-09-10 repository cut, artifact output
preserved). That function used `model.predict(...)["home_cover_probability"]`
directly -- the SMOOTH margin-model probability, predating
`BASE_PROBABILITY_POLICY`'s discrete lattice, NOT the served discrete policy.
This is a labeled caveat, not a violation: this population is a
challenger/sanity artifact, never served, and the task explicitly said reuse
`sbr_era_opener_eval` machinery for this term. Rebuilding a genuine discrete
(`DiscretePushReader`/`serve_discrete_three_way`) walk-forward grade at SBR's
Open was scoped out as its own unit of new engineering (mirrors
`opener_pick_evaluation`'s internal loop including `HOME_SIDE_OFFSET_SERVED`
home-side-offset fitting) -- not attempted here; flagged in Open below.

**Sanity look (reported, not recorded)**, both a 4-term ridge-logit LOSO by
season, `FIT_FEATURES=(model_logit, composition_flag_sum,
market_move_toward_home, market_move_available)`, reusing
`pick_probability_fit._fit_logit/_design/_standardisers/_predict` unmodified:
- Extended 2011-2025 LOSO, overall: 53.59% (n=3,734).
- Extended 2011-2025 LOSO, graded ONLY on the 2020-2025 held-out folds:
  **56.69%** (n=1,503).
- Served-equivalent: 2020-2025-only population, LOSO among just those 6
  seasons: **57.15%** (n=1,503, same 1,503 games).
Adding the 2,231 pre-2020 sbr_proxy-sourced games to training reads
**-0.46 pts** on the identical 2020-2025 held-out games vs training on
2020-2025 alone -- a wash within noise at this game count, not a resolved
gain or loss. No registry write; this is a look, not a closed signal.

**No fetchable gap for NFL market/raw**: `data/market/raw` (the purchased
Odds API `tue_open` archive `opener_pick_evaluation` reads) starts
20200825T115500Z-futures; earliest real game capture 20200831. This IS the
real 2020 floor: `MARKET_MOVE_ARTIFACT_ROOT="sharp_weighted_follow"` and
`FORECAST_TEMP_ARCHIVE="raw/forecast_archive/full_2020_2025/forecasts.parquet"`
(pick_probability_fit.py:50-51) show both `model_logit` (walk-forward
predictions) and `market_move` (sharp-book movement) are pre-built only for
2020-2025 — no code season filter exists in `opener_pick_evaluation`
(clv.py:1978) or `build_fit_population`; the bound is these two upstream
artifacts' own coverage. `docs/opener_evaluation.md:37` confirms: "2020-2025
historical snapshot archive."

**A pre-2020 NFL opener proxy DOES exist, already deeply characterized by
prior sessions (not new this unit)**: `data/processed/sbr_odds.parquet`
(sportsbookreviewsonline.com), 2007-2021, `open_home_spread`/
`close_home_spread`, 267 games/season 2007-2019 (269 in 2020, 285 in 2021),
4,025 rows, 3,491 matched to `game_features.parquet`. No timestamp, no book
ID (raw HTML columns are `Date,Rot,VH,Team,1st,2nd,3rd,4th,Final,Open,Close,
ML,2H` — `docs/sbr_opener_provenance.md:46-51`). Validated empirically against
two independently-timestamped sources: vs true `tue_open` 2020-2021 (r=0.949
reported, mean|diff| 1.36pts, only 17.7% exact); vs Wayback-captured
VegasInsider boards 2009-2016 magnitude-only (r=0.926-0.970 every season,
mean|diff| 0.68pts, 30.5% exact, 82.4% within 1pt) — `docs/sbr_opener_
provenance.md` sections 2-3. Verdict there: "a correlated proxy with a stated
error band," not interchangeable with a true timestamped opener; side
(favorite direction) is independently confirmed via SBR's own raw V/H row
structure (deterministic), not via VI (magnitude-only).

A full SBR-settled era-stratified re-grade of the ACTIVE model (production +
sign rule, NOT the four-term fitted probability) already ran and is recorded:
`docs/sbr_opener_evaluation.md`, artifact `artifacts/sbr_era_opener_eval/
20260819T233013Z/summary.json`. Scored population after the 500-game
warm-up floor: **2,832 games, 188 weeks, 2011-2021** (vs 1,503 for
2020-2025) — a ~1.9x power gain in game count. Four registry entries already
recorded `unresolved_below_power` (`sbr_opener_era_2011_2014`,
`_2015_2019`, `_2020_2021`, `sbr_opener_pooled_2011_2021`).

**Gap for THIS task's actual need (the four-term fitted-probability
population `build_fit_population` produces, which the six underpowered MOD-20
units score against)**: that population needs `model_logit` (2020-2025 only,
forecast-archive-bound), `composition_flag_sum` (from
`pbp08_matchup_flags.build_flag_table`, PBP-derived — PBP snapshot
`data/pbp/raw/20260817T184927Z` has `season=2009` partitions onward, so
computable back to 2009, not a blocker), and `market_move` (sharp-book
movement, needs `data/market/raw`, genuinely 2020+ only, no upstream fetch
would fill it — same kind of true archive gap as CFB's documented 2020
`cfbd_provider_sparse` gap in the sibling XLG-09 lane). So three of four
served terms could in principle extend to 2009/2011+; `market_move` cannot
without either dropping it as a term or building a new sharp-book proxy for
pre-2020 (not attempted, out of scope this unit).

**Did not run a new evaluation this unit.** Building a genuine extended-window
four-term population is not cheap: it needs (a) a new walk-forward
`model_logit` scored at SBR's Open as settlement (the existing
`sbr_era_opener_eval.py` machinery does this for the production/sign rule
only, not the four-term fit), (b) composition flags computed and joined for
2009-2019 (not yet built/joined anywhere in this repo), and (c) a decision
on `market_move` for pre-2020 (omit as a 3-term fit, or find/build a proxy).
That is real new engineering, not a "run it into a new artifact dir" rerun,
so it was not attempted under this unit's read-only/cheap-only brief.

## Tried
- Grepped `opener-evaluation` CLI (`src/nfl_ats/cli_commands/clv.py:362-425`)
  and `opener_pick_evaluation` (`src/nfl_ats/clv.py:1978`) for season bounds:
  none in code; bound is upstream artifact coverage (market archive,
  forecast archive).
- Inspected `data/market/raw` (2020-08-25 floor) and
  `data/market/historical/open_close/raw` (2025 season only, not helpful).
- Read `data/processed/sbr_odds.parquet`/`.manifest.json` directly (season
  counts above, matches manifest).
- Read `docs/sbr_opener_provenance.md` and `docs/sbr_opener_evaluation.md`
  in full (prior sessions' work, not redone).
- Confirmed PBP local snapshot covers `season=2009` onward
  (`data/pbp/raw/20260817T184927Z`).

## Next
Unit 2 shipped `artifacts/extended_fit_population/20260923T205041Z/`
(population.parquet + summary.json) via `scripts/extended_fit_population.py`.
If a future unit wants to close the smooth-vs-discrete gap: rebuild
2011-2019's `model_logit` with the actual `DiscretePushReader` +
`serve_discrete_three_way` lattice (mirroring `opener_pick_evaluation`'s
loop, keyed to `sbr_odds.parquet`'s `open_home_spread` instead of
`tue_open_home_spread`, including `fit_home_side_offsets`) rather than the
preserved smooth artifact reused this unit; re-run the same LOSO sanity look
after; decide whether the -0.46pt wash changes with the discrete read. This
was scoped out as its own unit (real new engineering, not a rerun), same
conclusion Unit 1 already reached for the harder version of this ask.

## Open
- SBR archive ceiling is season 2021-22 (0 rows beyond); 2007-2008 has zero
  independent cross-check (predates VI Wayback and the model's own warm-up
  floor). 2017-2019 also has no independent timestamped cross-check run
  anywhere in this repo (only 2009-2016 vs VI and 2020-2021 vs true opener
  were checked).
- Whether SBR's Open is a single book, a consensus, or a delayed
  transcription remains genuinely unknown (SBR publishes no provenance
  mechanism) — a residual risk any future extension inherits, not resolved
  here.
- Unit 2's 2011-2019 `model_logit` is the SMOOTH margin-model probability
  (pre-dates the discrete lattice), not `BASE_PROBABILITY_POLICY`; fine for
  this challenger/sanity artifact (never served), not fine to promote toward
  serving without rebuilding at the discrete read (see Next).
- `scripts/sbr_era_opener_eval.py` / `scripts/proxy_opener_replication.py`
  no longer exist in the tracked tree (removed in the 2026-09-10 repository
  cut, `b7ed31d`); only their artifact outputs survive. A future discrete
  rebuild writes fresh code, it cannot import the old scripts.
