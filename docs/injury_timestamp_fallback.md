# ENG-39: the 2025+ injury feature block is a silent zero, and how to fix it

Written 2026-09-05. Implements the coordinator-approved plan
`laneO_injury_fix_plan.md` (opus, read-only investigation). Ships three
things: (1) an opt-in, leakage-safe timestamp fallback so a season with no
real revision timestamps still gets an injury signal; (2) an always-on
prediction-safety check that fails a prospective card whose injury feature
block is entirely null/zero, so this failure mode can never again ship
silently; (3) a source-freshness policy row that watches the timestamp
health of the snapshot production actually reads, not the capture directory
it does not.

This document does **not** rebuild any production table, activate a new
model, or republish the card. Doing so is an owner decision (see "Rebuild
sequence, not run here" below).

## 1. The failure, in one sentence

nflverse's 2025 injuries release drops the `date_modified` column entirely,
and `nfl_ats.players.canonicalize_injuries`'s only historical behaviour --
drop every row without one -- turns that schema change into a
`home_/away_/diff_injury_*` feature block that is exactly zero for every
2025 and 2026 row, and no check in the pipeline notices.

## 2. Findings

Labelled per this repository's provenance convention: **measured** (run this
session, command given), **read** (file opened this session, path given),
**reported** (the coordinator's approved plan states it; not independently
re-run this session).

**M1. The 2025 upstream release has zero real revision timestamps.**
measured: `.tools/uv.exe run --no-sync python -c "..."` reading
`data/raw/nflverse_injuries/20260826T122850Z/injuries.parquet` --
2025 has 6,068/6,068 null `date_modified` (100%); 2011-2024 have 0 null in
every season from 2011 onward (2009 and 2010, outside that window, have
their own nulls -- 4,804/4,821 and 62/4,491 respectively -- not part of this
finding). No 2026 rows exist in this snapshot. reported (plan, unverified
further this session): the manifest for the 2025 request lists `season_type`
among its columns and omits `date_modified` -- a schema change, not a
transient gap.

**M2. The production model reads a snapshot that has already dropped every
2025 row.** measured: `artifacts/active_ats_model.json` `feature_table_sha256`
starts `13644e5c45345644d6b47733b62dee592a49a306123ce0f521c5d987dd52bbcf`,
matching the plan's citation of `data/processed/game_features_weak_stack.parquet`.
measured: the pinned player snapshot manifest
`data/players/raw/20260817T184901Z/manifest.json` lists `injury_seasons`
`[2009 .. 2024]` -- 2025 is not requested at all, so its rows were never
fetched, let alone dropped after the fact. `write_player_snapshot` calls
`canonicalize_injuries` at capture time (`src/nfl_ats/players.py`, now lines
~530-545 after this change; ~369-400 before it).

**M3. The production feature table's injury block is measurably dead for
2025 and 2026.** measured:
`.tools/uv.exe run --no-sync python -c "..."` reading
`data/processed/game_features_weak_stack.parquet` --

| season | rows | all-zero `diff_injury_*` rows | mean std across `diff_injury_*` |
| --- | --- | --- | --- |
| 2023 | 285 | 4.9% | 0.488 |
| 2024 | 285 | 5.3% | 0.546 |
| 2025 | 285 | **100.0%** | **0.000** |
| 2026 | 272 | **100.0%** | **0.000** |

Total table rows measured this session: 4,902 (2025+2026's 557 affected rows
= 11.4% of the current table). reported (plan, at the time it was measured,
before this session's 2026 rows existed in the table): 285/4,630 = 6.16% --
the plan's total excluded the in-progress 2026 season; both figures describe
the same underlying defect, just at different points as the table grew.
`diff_{metric} = 0.0` is written whenever either side's state is null
(`src/nfl_ats/players.py`, the per-game diff loop), which is also the
legitimate value for a genuinely clean two-sided report -- the tell is a
whole **season** at exactly 0.0 with zero variance, never a single game.

**M4. Nothing failed.** measured:
`artifacts/margin_predictions/2026-week-01-20260903T143253Z/prediction_safety.json`
reads `"status": "PASS"` with 14 checks passed and zero warnings, and
predates this change. `_feature_checks` (the pre-existing check family) only
tests non-numeric values, infinities, and a >50%-missing threshold --
present-and-exactly-zero satisfies all three.

**M5. In healthy (2011-2024) seasons, a revision timestamp is not close to
usable at the pool's real lock instant.** reported (plan, unverified further
this session -- a 73,492-row join across 14 seasons is out of scope for a
same-session re-verification, and the numbers do not change the fix's
design): joining every REG revision to its own game's kickoff:

- 99.62% of revisions land >= 24h before their own kickoff.
- 99.89% of 7,429 team-games have >= 1 revision visible at kickoff-24h (the
  model's actual, existing decision cutoff).
- Median of each team-game's newest visible revision: 55.1h before kickoff.
- Only **0.24%** of team-games have a visible revision at a Tuesday 09:00 ET
  lock -- the pool's real per-game lock instant is `min(kickoff, Sunday
  16:00 ET)` (see `nfl_ats.nfl_week.pool_decision_cutoff`), so this is a
  **pre-existing train/serve timing mismatch**: the feature table is built
  as of kickoff-24h, while a card frozen at Tuesday morning would see almost
  none of it. This document's fix does not change that; it is called out so
  it is not mistaken for something the week_proxy fallback fixes.
- By weekday, share of team-games with a revision visible >= 36h before
  kickoff: **Sun 99.5%, Mon 99.3%, Sat 75.8%, Thu 48.7%** (median 35.9h for
  Thursday games -- the short week leaves less room for a revision to land
  well before kickoff).

**M6. Whether 2026 restores the column is unknown locally.** inferred (this
session's own judgement, not a measurement): if nflverse ships 2026 with the
same `season_type`-not-`date_modified` schema, `fetch_player_snapshot`
including season 2026 either yields all-null `date_modified` (silently
repeating M1-M3) or raises `DataContractError` from `require_columns` if the
column is dropped from the response entirely, matching what 2025 already
does. Not verified against a live 2026 nflverse response this session (no
network access; this session made no new fetches).

**M7. The freshness policy was watching the wrong directory.** read:
`src/nfl_ats/source_freshness_policy.py`'s `injuries_nflverse` policy
watches `data/raw/nflverse_injuries`, a directory the feature build
(`nfl_ats.players`) does not read from; the actual consumed source is
`data/players/raw/<snapshot id>`. A perfectly fresh capture in the watched
directory says nothing about whether the snapshot production actually reads
has a usable timestamp for the season being served. Fixed here with a new,
separate policy (section 5).

## 3. The fix: an opt-in, leakage-safe timestamp fallback

### Visibility rule (verbatim, as implemented)

> `effective_observed_at = date_modified` if present, else
> `injury_proxy_at`, where `injury_proxy_at` = (kickoff of that
> `(season, week, team)`'s own game) minus
> `INJURY_PROXY_HOURS_BEFORE_KICKOFF` (= 24 hours), clamped to be
> `>= 00:00 America/New_York on the Tuesday that starts that game's own NFL
> week` and `< kickoff`. A row with no scheduled game to resolve a kickoff
> against becomes `NaT` and is dropped -- exactly as it would be under the
> default `"drop"` behaviour.

Downstream visibility keeps testing `effective_observed_at <= decision_at`
-- nothing about *how* a row becomes visible changes; only the value used to
decide *whether* it is.

### Where it lives

`src/nfl_ats/players.py`:

- `canonicalize_injuries(frame, *, include_postseason=False,
  timestamp_fallback: Literal["drop", "week_proxy"] = "drop", schedule=None)`.
  `"drop"` (the default) is unchanged and produces byte-identical output to
  the pre-ENG-39 function -- no new columns, no schedule dependency, proven
  by a hash pin (`tests/test_injury_timestamp_fallback.py::
  test_week_proxy_default_drop_mode_is_byte_identical_to_pre_eng39`).
  `"week_proxy"` additionally tolerates a `date_modified` column that is
  missing entirely (inserted as `NaT` before the column-presence check runs)
  and emits two new columns: `effective_observed_at` (the real
  `date_modified` where present, else the proxy) and `observed_at_basis`
  (`"date_modified"` or `"week_proxy"`, tagged on every proxied row). A real
  `date_modified` is never overwritten.
- `_injury_proxy_times` / `_injury_week_tuesday_floor_utc`: build the
  per-`(season, week, team)` proxy time from a schedule frame
  (`season, week, home_team, away_team, kickoff`).
- `write_player_snapshot` / `fetch_player_snapshot`: thread
  `injury_timestamp_fallback` and (for `write_player_snapshot`) an explicit
  `injury_schedule` frame through to `canonicalize_injuries`; `fetch_player_snapshot`
  fetches nflverse schedules itself only when `"week_proxy"` is requested --
  the one new network dependency this adds, and only on that opt-in path.
  The snapshot manifest now also records `injury_timestamp_fallback`,
  `injury_proxy_hours_before_kickoff`, and `n_proxy_rows_per_season`.
- `_injury_rows_asof`: filters on `effective_observed_at` when the column is
  present, else falls back to `date_modified` -- back-compatible with every
  existing snapshot on disk.
- `enrich_with_player_features`: gained `injury_timestamp_fallback` (same
  default, same guarantee), builds the schedule frame from its own `games`
  parameter (already required to carry `kickoff`), and now also emits
  `{side}_injury_observed_at_basis` alongside the existing
  `{side}_injury_observed_at` -- `"date_modified"`, `"week_proxy"`, or the
  pre-existing ENG-23 `"snapshot_captured_at"` fallback tag. Always
  populated (including in default `"drop"` mode), so it never diverges from
  the injuries frame's own basis.
- `injury_missing_coverage(enriched)`: a new, read-only diagnostic --
  per-season/game counts of a wholly-missing injury observation (neither
  side has any `{side}_injury_observed_at`). This is the mechanical version
  of the M3 table above, over any `enrich_with_player_features` output.
  Wiring this into the feature-table manifest is left to whatever assembles
  that table; this lane does not own that script.

`src/nfl_ats/quarterbacks.py` carries the identical threading for the
separate named-QB availability path (`_canonicalize_qb_availability`,
`enrich_with_qb_features`), which reads injuries independently for starter
availability. Its proxy helpers are duplicated, not imported, from
`players.py` (`QB_INJURY_PROXY_HOURS_BEFORE_KICKOFF`,
`_qb_injury_proxy_times`, `_qb_injury_week_tuesday_floor_utc`) -- the same
convention `nfl_ats.transaction_wire_features.kickoff_utc` already uses to
avoid a circular import (`players` already imports from `quarterbacks`).

### CLI

`player-ingest` gained one additive flag:

```
nfl-ats player-ingest --timestamp-fallback week_proxy
```

Default is `drop`, unchanged. `tests/fixtures/cli_contract.json` was
regenerated (`scripts/cli_contract_snapshot.py --normalize-years`); the diff
is exactly this one new argument.

### Measured: what week_proxy actually does to the 2025 season

measured, scratch script (no network; local files only):
`data/raw/nflverse_injuries/20260826T122850Z/injuries.parquet` (2025 rows)
joined to `data/raw/20260824T115346Z/schedules.parquet` (newest local
schedule snapshot), `canonicalize_injuries(..., timestamp_fallback="week_proxy")`:

- 6,068 raw 2025 rows; 5,783 are REG (the rest are postseason, filtered by
  the pre-existing `include_postseason=False` default, unrelated to this
  change); all 5,783 survive canonicalization and all 5,783 are tagged
  `observed_at_basis="week_proxy"` (0 have a real `date_modified` -- matches
  M1 exactly).
- Every one of the 18 REG weeks has a 100% proxy fraction (197-394 rows per
  week).
- The proxy is uniformly exactly 24.0 hours before kickoff for every one of
  the 5,783 rows (min = max = mean = 24.0, std = 0.0) -- the Tuesday clamp
  never engages for any real 2025 game, because no real game kicks off
  within 24h of that week's own Tuesday. (The clamp is exercised directly in
  `tests/test_injury_timestamp_fallback.py` with a synthetic Tuesday-kickoff
  fixture, since no real NFL schedule can trigger it.)
- 100% of proxied rows are visible at kickoff-24h (tautological: the proxy
  *is* kickoff-24h) -- reported here to make explicit that
  `decision_hours_before_kickoff=24` (production's existing default) is
  exactly compatible with this fallback's own default lag.
- **0% of proxied rows are visible at a Tuesday 09:00 ET lock** -- the
  fallback does not fix the M5 train/serve mismatch; if anything, a
  proxy deliberately placed near kickoff makes it slightly more pronounced
  than the median real revision (55.1h before kickoff, per M5) for a card
  frozen Tuesday morning.

### Measured: the live 2026 Week 1 card now fails, a 2024 slice passes

measured, scratch script (read-only; loaded
`artifacts/margin_predictions/2026-week-01-20260903T143253Z/predictions.csv`
and `data/processed/game_features_weak_stack.parquet`; wrote nothing):

```
Live 2026 Week 1 card: 80 rows, 16 games, 29 injury-pattern columns, abs-sum=0.0
LIVE CARD: prediction_safety FAILED as expected ->
  Prediction safety check 'injury_feature_presence' failed: every value
  across 27 injury feature column(s) is null or exactly 0.0 -- see
  docs/injury_timestamp_fallback.md

2024 season week 1 slice: 16 rows, injury abs-sum=62.5210
2024 SLICE (direct _injury_feature_checks call): checks=['injury_feature_presence'] warnings=[]
```

(27, not 29: the check's column pattern deliberately excludes
`{side}_injury_observed_at` and `{side}_injury_observed_at_basis` -- see
"A regex-matching bug this measurement caught" below.)

## 4. The fix: an always-on prediction-safety check

`src/nfl_ats/prediction_safety.py`: a new `_injury_feature_checks(frame,
feature_columns, *, allow_empty_injury_block=False)`, wired into both
`validate_prediction_card` and `validate_outcome_prediction_card` when
`prospective=True` and the card has at least one row. It scans the injury
magnitude sub-block -- `^(?:home_|away_|diff_)injury_`, excluding the
`observed_at`/`observed_at_basis` lineage columns -- restricted to
`feature_columns` when the caller supplies one (as
`validate_prediction_card`'s existing callers do), else discovered from the
card's own columns (as `validate_outcome_prediction_card` does, since it has
no `feature_columns` parameter). It **fails** `injury_feature_presence` if
every value across the block is null or exactly 0.0, and **warns** if more
than 50% are, unless the caller explicitly passes
`allow_empty_injury_block=True`. **No production caller sets it.**

`validate_outcome_prediction_card` gained `feature_columns`, `prospective`,
and `allow_empty_injury_block` as new, additive, default-preserving keyword
arguments -- it previously had none of the three. Its only real caller today
(`orchestrate_margin_predict` / `margin-predict`) does not pass
`prospective=True`, so **today's production publish path is unchanged**;
the capability exists, is proven against the real live card above, and
activating it in production (passing `prospective=True` at that call site)
is an owner decision this lane does not make.

### A regex-matching bug this measurement caught

The first version of `_INJURY_FEATURE_COLUMN_PATTERN` was
`^(?:home_|away_|diff_)injury_`, which also matches
`{side}_injury_observed_at` and `{side}_injury_observed_at_basis` --
timestamps and provenance labels, not model inputs. `pandas.to_numeric` on a
timezone-aware datetime column does not raise and does not return null; it
silently converts a real timestamp to a large nonzero int64 (nanosecond
epoch). Running the check against the real production feature table
(measured, `data/processed/game_features_weak_stack.parquet`, which has a
real `home_injury_observed_at` datetime column with a mix of null and
populated values) would have made a genuinely all-zero magnitude block look
nonzero to `null_or_zero.all()` whenever `home_injury_observed_at` happened
to carry any real timestamp -- defeating the check on exactly the case it
exists to catch. Fixed with a negative lookahead,
`^(?:home_|away_|diff_)injury_(?!observed_at)`, and reproduced against the
real feature table above (27 columns matched, not 29) before this document
was written.

## 5. The fix: a source-freshness policy for the consumed snapshot

`src/nfl_ats/source_freshness_policy.py`: a new policy,
`injuries_nflverse_timestamps`, alongside the pre-existing
`injuries_nflverse` (which is left unchanged and still watches the capture
directory -- a different, still-useful question: did a capture land at
all). The new row's `SourceLocation` documents where the feature build
actually reads injuries from (`data/players/raw`, not
`data/raw/nflverse_injuries`, closing the M7 gap for at least this one
row), `on_absent=DEGRADED` (never `BLOCKED` -- this is a data-quality signal
about an already-permitted publish path, not a new fail-closed gate), and a
fallback description naming this document and the
`injury_feature_presence` check as the actual release-blocking mechanism.

A new function, `player_snapshot_injury_timestamp_observation(
player_snapshot_root, *, season)`, reads the snapshot's own manifest and
`injuries.parquet` directly and reports `observed_at=None` (absent) when
`season` has zero rows with a real (non-proxy) `date_modified` -- a fully
proxied season still reports absent, so the fallback's own manufactured
timestamp can never borrow credibility from this policy. `report_for_publication`
gained two new, additive, default-`None` keyword arguments
(`player_snapshot_root`, `player_snapshot_season`) so a future caller can
wire this override in; **no existing caller passes them**, so the live
publish path's behaviour is unchanged today -- without them, the new policy
row falls back to the generic snapshot-directory scan on `data/players/raw`
(a source that simply did not exist as a watched location before this
change).

`tests/test_source_freshness_policy.py`'s
`test_every_budget_matches_the_capture_schedule_arithmetic` hardcodes the
full policy id/budget table and needed one additive row for the new policy
(same `weekly_lock` cadence as `injuries_nflverse`, 10,080 + 120 = 10,200
minutes) to keep passing -- the only edit made outside this lane's originally
named file list, made because the new policy row is itself explicitly named
in the approved plan and no other assertion in that file needed to change
(verified by reading every assertion touching the full policy dict, then by
running the file).

## 6. Rebuild sequence, not run here

Per `docs/feature_lineage.md` and the approved plan, actually fixing the
live 2026 Week 1 card requires (owner decision, not run by this lane):

```
nfl-ats player-ingest --injury-start-season 2009 --injury-end-season 2026 \
  --timestamp-fallback week_proxy   # network, requires nflverse GREEN
nfl-ats build-features
nfl-ats build-pbp-features --snapshot 20260817T184927Z
nfl-ats build-learned-availability-features ... \
  --player-snapshot <new> --player-value-snapshot 20260817T184911Z \
  --pbp-snapshot 20260817T184927Z
nfl-ats margin-predict --season 2026 --week 1 --feature-profile weak_stack
```

then `margin-backtest` and re-activation before republishing (the model id
changes when its feature table's sha256 changes). This lane made none of
these calls: no network fetches, no writes under `data/processed` or
`artifacts/forecasts`, no model activation, no republish.

## 7. Tests

`tests/test_injury_timestamp_fallback.py` (new): default-mode hash pin
(`assert_frame_equal` between implicit-default and explicit `"drop"`, plus a
SHA-256 over `pandas.util.hash_pandas_object` of a multi-season,
out-of-order, duplicate-row fixture spanning 2011 and 2024); the Tuesday
clamp (a Thursday game that does not need it, and a synthetic Tuesday
kickoff that does); a real `date_modified` never overwritten; a 2025-shaped
frame missing the `date_modified` column entirely (survives `"week_proxy"`,
still raises `DataContractError` under the default); the leakage pin (a
proxied row invisible at a 48h-before-kickoff cutoff, visible at a 1h
cutoff, and invisible at either cutoff under the default `"drop"` mode); and
`injury_feature_presence` failing an all-zero block / passing a healthy one
for both `validate_prediction_card` and `validate_outcome_prediction_card`,
including the `allow_empty_injury_block` escape and the non-prospective
no-op.

`tests/test_illness_battery_leakage.py::test_a_null_date_modified_row_never_becomes_visible`
still passes unmodified -- the default `"drop"` mode this test exercises is
untouched.

## 8. Follow-up (2026-09-05, lane AE): a separate re-derivation site one level down

Lane S2's rebuild report (`build_availability_outcomes`'s "separate,
unfixed gap" section) measured that even after the players.py idempotency
fix (section 3) and the production rebuild it enabled, the *learned*
availability rates table (`weak_stack_availability_rates.parquet`, built by
`build-learned-availability-features`) still never saw a single proxied
2025 row: `nfl_ats.availability.build_availability_outcomes` had its own,
independent `date_modified`-only visibility filter -- a third
re-derivation site, structurally identical to the bug fixed in section 3,
but one level down, outside `players.py`.

**Fix:** `build_availability_outcomes` now prefers
`injuries["effective_observed_at"]` when that column is present (the
output of `canonicalize_injuries(..., timestamp_fallback="week_proxy")`),
else falls back to `injuries["date_modified"]` -- the identical
column-presence rule `nfl_ats.players._injury_rows_asof` already uses. A
real `date_modified` is never overwritten (that column, and the decision
of which value to prefer, are both already settled upstream by
`canonicalize_injuries`; this function only chooses which already-computed
column gates visibility). A frame without `effective_observed_at` (every
pre-ENG-39 snapshot, and any frame canonicalized with the default `"drop"`
mode) keeps filtering on `date_modified` exactly as before -- hash-pinned
byte-identical in `tests/test_availability.py::
test_availability_outcomes_plain_frame_is_byte_identical_to_pre_eng39`.

**Measured, side rebuild only (not promoted):**
`nfl-ats build-learned-availability-features --features
data/processed/game_features_pbp.parquet --destination
data/processed/game_features_weak_stack_eng39b.parquet --rates-destination
data/processed/weak_stack_availability_rates_eng39b.parquet
--evaluation-destination
data/processed/weak_stack_availability_evaluation_eng39b.csv
--player-snapshot 20260905T123614Z --player-value-snapshot
20260817T184911Z --pbp-snapshot 20260817T184927Z
--injury-timestamp-fallback week_proxy` (same snapshots lane S2's rebuild
used):

- `build_availability_outcomes`'s own output (measured directly, bypassing
  the CLI): season-2025 outcome rows go from **0** (simulating the pre-fix
  `date_modified`-only filter on this snapshot's already-week_proxy'd
  injuries) to **5,783** (matching the exact count of 2025 REG rows that
  survive `canonicalize_injuries(..., timestamp_fallback="week_proxy")` per
  section 1/M1 of this document); every other season's row count (2013-2024,
  62,206 rows total) is unchanged.
- `weak_stack_availability_rates.parquet`: **1,429 -> 1,436 rows**.
  `target_season=2026`'s own training window now ends at
  `source_end_season=2025` (was 2024) -- 2025 now contributes as *source*
  data for the next target season's lagged rate, which is the whole point
  of the fix (the current live model does not yet serve a `target_season
  2027`, so no target season's rates are trained *on* 2025 itself yet; 2026
  is the first target season whose training window can include it).
  `rates_sha256` moved from `91b4d81ab723925bc8d2c8cf228a41493f66368514e39d91afac7ed779076f35`
  (unchanged from the pre-fix table -- confirmed still the current value in
  production's own `weak_stack_availability_rates.parquet` at measurement
  time, even though production's *feature table* had already been rebuilt
  with the players.py fix) to `7364bc9e7866e81c88af14c0ac8324e8610d8c7b4798bc52f7b64f9bef335b75`.
- `weak_stack_availability_evaluation.csv`: scored player-games
  **57,294 -> 63,077** (+5,783, exactly the newly visible 2025 outcomes).
  Fixed-prior Brier **0.09453747 -> 0.09307263**; learned-rate Brier
  **0.08958403 -> 0.08850843**; fixed classification accuracy
  **0.87159214 -> 0.87369406**; learned classification accuracy
  **0.87949873 -> 0.88032088**. The manifest's own
  `availability_brier_improvement` (learned minus fixed) moved from
  **0.004953441704819983** to **0.004564199446518244** -- the gap between
  the two methods narrowed slightly once 2025 entered the evaluation set,
  but both methods individually improved in absolute Brier terms. This
  evaluation summary has no log-loss column; only Brier score and
  classification accuracy are computed by `summarize_availability_scores`.
- The `diff_injury_*` feature block, and all 255 shared non-`injury`
  columns, are **completely unaffected**: `game_features_weak_stack_eng39b.parquet`
  is byte-identical (same sha256,
  `41a778f26a38e63bede7e7bf01f4a4a30254c09164cae3c5ee2cce87bc2547f6`) to
  both lane S2's `game_features_weak_stack_eng39.parquet` and the current
  production `game_features_weak_stack.parquet` (which, by the time this
  fix was measured, had already been promoted to that same sha256). This
  is the expected, not the tested-for, result: this fix's only consumers
  are the separate rates/evaluation outputs (`availability_rates` is used
  downstream purely as a report/practice/position-category *lookup table*,
  never as a visibility gate on the injury magnitude features themselves).

The side rebuild is not promoted; whether to point
`build-learned-availability-features` at `--injury-timestamp-fallback
week_proxy` in the production rebuild (recreating the currently-stale
`weak_stack_availability_rates.parquet` in place) is an owner/coordinator
decision, not made by this lane.

Tests: `tests/test_availability.py` gained
`test_availability_outcomes_plain_frame_is_byte_identical_to_pre_eng39`
(hash pin),
`test_availability_outcomes_prefers_effective_observed_at_for_a_proxied_2025_row`,
`test_availability_outcomes_leakage_proxied_row_invisible_before_its_proxy_time`,
and `test_availability_outcomes_never_overwrites_a_real_date_modified`.


## 2026-09-05 CX8: real-revision precedence and proxy lineage

Measured (`tests/test_injury_timestamp_fallback.py`, CX8 targeted pytest): a
re-canonicalized proxy row now takes a supplied real `date_modified` as its
effective timestamp and changes its basis to `date_modified`; a report revised
after the cutoff stays invisible in both player enrichment and learned
availability outcomes. The approved kickoff-minus-24-hours proxy remains in
place for undated reports; it is an assumption about visibility, not a measured
publication time.

Read (`src/nfl_ats/players.py`, `canonicalize_injuries` and
`enrich_with_player_features`): proxy-bearing snapshots carry the boolean
`observed_at_is_proxy`. Feature tables carry each side's
`injury_observed_at_is_proxy` (any contributing proxy, not merely the latest
revision's basis) and `injury_proxy_row_count`, plus per-season/week counts in
`attrs["injury_proxy_provenance"]`. Measured (CX8 regression fixtures): the
visible fixture contributes one proxy row; moving the cutoff before its proxy
time contributes zero. Read (`src/nfl_ats/availability.py`): proxy-bearing
availability outputs retain the boolean, while the default non-proxy output
schema and its existing hash fixture remain unchanged.

Read (CX8 lane file restrictions): publish-lineage rendering is owned by the
coordinator; `publishing.py` and board files were not edited. The new table
columns and parquet attributes supply that integration without conflating
assumed visibility with real observation.
