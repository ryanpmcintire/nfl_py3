# End-to-end feature lineage (ENG-16)

Every weekly forecast now ships a `lineage.json` next to `recommendations.csv`,
and the release-blocking prediction-safety contract refuses a card whose
decision-bearing fields cannot say where they came from.

Implementation: `src/nfl_ats/lineage.py` (new), validated through
`src/nfl_ats/prediction_safety.py` (`_lineage_checks`,
`validate_prediction_lineage`, and an additive `lineage=` keyword on both
existing card validators). Tests: `tests/test_lineage.py`.

## Why

`recommendations.csv` answers "what did the model say". It does not answer the
question that actually comes up months later: **what did that number see, and
when could it have seen it.** Provenance existed, but scattered — a feature
table sha256 in `metadata.json`, snapshot ids inside a nested manifest, builder
versions on some columns and not others, overlay inputs only in memory during
publish. Nothing tied a *card field* to a *source snapshot* and a *cutoff*, and
nothing failed when the tie was missing.

## Decision-bearing, defined

A card field is **decision-bearing** when changing it would change what gets
submitted to the pool. The definition is enumerated in code
(`lineage.is_decision_bearing`), not described:

| Field | Meaning |
| --- | --- |
| `pick` | the side actually played |
| `model_probability` | the probability the pick is read from, and the confidence ordering |
| `market_line` | the line the pick is expressed and graded against |
| `model_input:<family>` | one per feature family the fitted model consumed |
| `overlay:<member_id>` | one per overlay that **fired**; a member that flipped nothing changed nothing |
| `tiebreaker:<input>` | one per tiebreaker input — the tiebreak score is a submitted number too |

`pick`, `model_probability` and `market_line` are
`REQUIRED_DECISION_BEARING_FIELDS`: they must be present on every card. Overlay
and tiebreaker fields are conditional by nature, but when present they are
validated in full.

Everything else is display. A display field may carry `"lineage": null`
**provided it carries an explicit `reason`** — see `PUBLISHED_DISPLAY_FIELDS`
for the four columns the tracked Markdown card renders (`Date`, `Matchup`,
`ATS prediction`, `Decision score`). Silence is not permitted anywhere: a field
with neither a record nor a reason fails the check.

## Reading `lineage.json`

```json
{
  "schema_version": 1,
  "builder_version": "v1",
  "generated_at_utc": "2026-09-04T...",
  "prediction_timestamp": "2026-09-03T14:32:53.143515+00:00",
  "season": 2026,
  "week": 1,
  "forecast_artifact": "margin_predictions/2026-week-01-20260903T143253Z",
  "model_id": "123d60be8c80a35d",
  "decision_bearing_fields": ["pick", "model_probability", "market_line", "..."],
  "fields": [
    {
      "card_field": "model_input:player_injuries",
      "decision_bearing": true,
      "lineage": {
        "card_field": "model_input:player_injuries",
        "feature_family": "player_injuries",
        "source_snapshot": "20260817T184901Z",
        "source_captured_at": "2026-08-17T18:49:01+00:00",
        "effective_timestamp": "2026-08-17T18:49:01+00:00",
        "effective_timestamp_basis": "source_capture",
        "builder_version": "v3-availability-v1",
        "builder_module": "nfl_ats.players",
        "unknown_source_reason": null
      },
      "reason": null
    },
    {
      "card_field": "Matchup",
      "decision_bearing": false,
      "lineage": null,
      "reason": "formatted from home_team/away_team, already covered by model_input:market"
    }
  ]
}
```

Field by field:

- **`card_field`** — the card field this record justifies, using the prefixes
  above.
- **`feature_family`** — the `constants.FEATURE_FAMILIES` key, or
  `overlay/<member>` / `tiebreaker/<input>` / `model_decision` /
  `model_probability` for the non-family records. A model input claimed by no
  declared family lands under the synthetic family `unassigned` rather than
  being dropped.
- **`source_snapshot`** — snapshot id (`20260817T184901Z`), or the feature-table
  identity (`feature_table:sha256:<digest>`) when the upstream capture is not
  recorded. Never silently null: see `unknown_source_reason`.
- **`source_captured_at`** — the capture instant. Derived from the snapshot id,
  because that is the only capture field uniform across sources — the manifests
  themselves variously call it `fetched_at_utc`, `created_at_utc`,
  `captured_at_utc`, `observed_at_utc`, `generated_at_utc` or `retrieved_at_utc`.
- **`effective_timestamp`** — the as-of cutoff the feature used. Must be `<=`
  `prediction_timestamp`.
- **`effective_timestamp_basis`** — how strong that number is, and this matters:
  - `declared` — the builder recorded an explicit cutoff.
  - `training_cutoff` — the model's `train_max_gameday`.
  - `source_capture` — the snapshot's own capture instant.
  - `feature_table_build` — the feature table's `built_at_utc`. This is an
    **upper bound** on an as-of that nobody wrote down, not the cutoff itself.
    Reporting an upper bound as though it were the cutoff would be exactly the
    kind of unlabelled claim `AGENTS.md` bans, so the basis is carried
    alongside the number rather than hidden behind it.
- **`builder_version` / `builder_module`** — which code produced the family.
  Preferred from the feature-table manifest when it records one (the production
  weak-stack table says `player_feature_version: "v3-availability-v1"` while
  `players.PLAYER_FEATURE_VERSION` on disk is `"v2"`), else from the module
  constant.
- **`unknown_source_reason`** — required whenever `source_snapshot` is null.
  An unrecorded provenance has to be *declared*, not merely absent. This is what
  keeps the gap list below from quietly growing.

Round-trip helpers: `CardLineage.to_json()` / `CardLineage.from_json()`, and
`lineage.write_card_lineage(lineage, directory)` /
`lineage.read_card_lineage(directory)`.

## What the safety check enforces

`prediction_safety.validate_prediction_lineage(lineage)` — and the additive
`lineage=` keyword on `validate_prediction_card` and
`validate_outcome_prediction_card` — raise `PredictionSafetyError` naming every
offending field. Four checks, reported alongside the pre-existing ones:

| Check | Fails when |
| --- | --- |
| `lineage_schema` | `schema_version` is not the supported version |
| `lineage_required_fields` | `pick`, `model_probability` or `market_line` is missing, null, or not marked decision-bearing |
| `lineage_completeness` | a decision-bearing field has no record; a record has an empty family/version/module/timestamp; `source_snapshot` is null with no `unknown_source_reason`; a display field has neither record nor reason; the basis is not one of the four recognized values |
| `lineage_effective_timestamp` | any record's `effective_timestamp` is unparseable, or **later than `prediction_timestamp`** |

That last one is the project's pregame-information invariant ("features may
only use information available before the prediction timestamp") restated at
the lineage layer, where it can be audited from the artifact alone instead of
from the builder's intent.

Passing no `lineage` leaves the existing contract byte-for-byte unchanged, so
historical artifacts and every caller that predates this keep validating as
before.

## Where it is emitted

`nfl-ats margin-predict` (the production forecast writer, the one
`active_ats_model.json` links) and `nfl-ats predict` both build the lineage,
validate it — release-blocking, before anything is written — then write
`lineage.json` into the artifact directory and record a reference in
`metadata.json`:

```json
"lineage": {
  "path": "lineage.json",
  "schema_version": 1,
  "builder_version": "v1",
  "checks_passed": ["lineage_schema", "lineage_required_fields",
                    "lineage_completeness", "lineage_effective_timestamp"],
  "decision_bearing_fields": ["pick", "model_probability", "market_line", "..."]
}
```

Model input families come from the model's own contract
(`margin.margin_feature_columns(target, profile)`), not from whatever columns
happen to sit in the table — so the emitted families are the ones the fit
actually consumed.

**ENG-24, `nfl-ats publish-predictions`:** the forecast's `lineage.json` above
answers "what did the FIT see"; it cannot answer "what did the PLAYED card
see", because the four-member overlay policy and the tiebreaker guess both run
at publish time. `publish_active_predictions` builds a second `CardLineage` --
the forecast's own (read back, or built fresh when absent) extended via
`lineage.extend_card_lineage_for_publication` with the overlay records
`overlay_sources_from_composition` returns and the tiebreaker records
`tiebreaker.tiebreaker_lineage_sources` returns -- validates it the same
release-blocking way, and writes it as `lineage.json` **beside the published
card** (`destination.parent`), never overwriting the forecast's file. See gap
items 4-5 below for the measured detail.

## Where lineage is unknowable today

Measured 2026-09-04 by building lineage from the live production artifact
`artifacts/margin_predictions/2026-week-01-20260903T143253Z` (feature profile
`weak_stack`). Six records name a real snapshot; nine declare why they cannot.

**Names a real source snapshot:** `pick`, `model_probability` (feature-table
sha256 identity), `model_input:player_qb` (`source_depth_snapshot`),
`model_input:player_injuries`, `model_input:player_continuity`
(`source_player_snapshot`), `model_input:player_values`
(`source_player_value_snapshot`).

**Declared unknown, and why:**

1. **The base nflverse snapshot id is not propagated through derived feature
   tables.** Affects `market_line`, `model_input:market`, `context`, `elo`,
   `experience`, `offense`, `results`, `defense`, `bias` — i.e. everything
   `nfl_ats.features` builds, including the spread the pick is graded against.
   Only `data/processed/game_features.manifest.json` records `source_snapshot`;
   every enrichment step (`game_features_pbp`, `game_features_qb`,
   `game_features_player*`, `game_features_weak_stack`) records `source_features`
   — a *path* — and drops the id. These records fall back to
   `feature_table:sha256:<digest>`, which pins the exact bytes the model read
   but not the upstream capture. **Follow-up:** carry `source_snapshot` forward
   in each enrichment manifest (a one-line addition per builder in
   `cli._cmd_build_*_features`). **Done (ENG-22, see the section below):** the
   propagation code is written and tested; it takes effect on the next
   rebuild of each table, not retroactively on the manifests already on disk.
2. **No market observation timestamp reaches the card.** `recommendations.csv`
   had no `market_observed_at_utc` / `line_observed_at_utc` column; the
   `spread_line` arrives through the nflverse schedules table, so the freshest
   provable bound was the feature table's `built_at_utc`. The point-in-time odds
   captures under `data/market/raw/<stamp>/` carry `observed_at_utc` but were
   not joined into the forecast. **Done (ENG-23):** new module
   `nfl_ats.market_observation.attach_market_observed_at` joins the Tuesday-opener
   capture's `observed_at_utc` onto the forecast frame by `game_id` — the same
   quote `nfl_ats.source_freshness_policy` already calls "the grade the pool
   settles on" — without touching `spread_line` or which side is picked. Wired
   into both `orchestrate_margin_predict` and `orchestrate_predict`
   (`src/nfl_ats/cli_commands/prediction.py`) right after the frame is scored, so
   `_prospective_checks`'s `market_timing` check now sees a real instant instead
   of warning that it cannot. `lineage.build_card_lineage`'s `market_line`
   record prefers this column's value for `source_captured_at`/
   `effective_timestamp` when present (a legacy frame without the column
   validates exactly as before). Historical rows with no matching capture (most
   of the archive predates the-odds-api ingestion) are left null, never an
   error. **Takes effect on the next `margin-predict`/`predict` run**, not
   retroactively on artifacts already on disk: the live production artifact
   `artifacts/margin_predictions/2026-week-01-20260903T143253Z` (measured
   2026-09-04, built 2026-09-03, before this work) has no
   `market_observed_at_utc` column at all. An in-memory-only rebuild with
   identical raw snapshot inputs (read-only; nothing written under `data/` or
   `artifacts/`) shows it populated for real:
   `2026_01_ARI_LAC`/`2026_01_ATL_PIT` both read `2026-08-18T13:00:13.026397+00:00`
   (computed, simulated, 2026-09-04), the actual capture instant of the Tuesday
   opener snapshot under `data/market/raw/` that names their `game_id`.
3. **`*_injury_observed_at` is empty for the forecast week.** The columns exist
   on the card and were all null for the 2026 Week 1 rows (measured), because
   `enrich_with_player_features` only ever set them from a visible team-specific
   revision (`visible_injuries["date_modified"].max()`); a team with no
   revision on record — a clean report, or one not yet filed — left the column
   null forever even though the injury snapshot itself WAS captured at a known
   instant. **Done (ENG-23):** `enrich_with_player_features` takes an optional
   `injury_snapshot_captured_at` and, only when no revision is visible AND that
   instant is not after the game's own decision cutoff (so it can never
   introduce a leak the existing `date_modified` filter was not already
   enforcing), fills `{side}_injury_observed_at` with it instead of leaving it
   null. All three player-feature CLI builders in
   `src/nfl_ats/cli_commands/features.py` now pass
   `parse_snapshot_capture(player_snapshot.snapshot_id)` at the call site.
   `lineage.build_card_lineage`'s `model_input:player_injuries` record prefers
   the frame's own `{side}_injury_observed_at` columns the same way the
   `market_line` record prefers item 2's column (source_snapshot is unaffected
   either way). Omitting the new argument reproduces the previous null
   behaviour bit for bit. **Takes effect on the next player-feature rebuild**,
   not retroactively: the live production artifact (same one named above) still
   shows null for both sampled games (measured 2026-09-04). The same
   in-memory-only simulated rebuild that closed item 2 shows both games'
   `home_injury_observed_at`/`away_injury_observed_at` populated at
   `2026-08-17T18:49:01+00:00` — `source_player_snapshot`'s own capture
   instant, since neither sampled team had a real revision on record for week 1
   (computed, simulated, 2026-09-04); `tests/test_market_observed_at.py`
   exercises a team that DOES have a real, earlier revision to prove the two
   sources are distinguished, not just coincidentally equal.
4. **Overlays are not represented in the FORECAST artifact.** The played
   four-member policy runs at publish time, not forecast time, so a
   `lineage.json` written by `margin-predict` still contains no `overlay:`
   fields, by design -- that file is the model card's lineage, not the played
   card's. **Done (ENG-24):** `nfl_ats.lineage.extend_card_lineage_for_publication`
   takes the forecast's own lineage (read back via `read_card_lineage`, or built
   fresh via `build_card_lineage` when the forecast never got one -- an artifact
   predating the ENG-16 wiring, or a legacy path) and appends the
   `lineage.overlay_sources_from_composition(production_overlay, ...)` records
   for whichever members actually flipped a game this week.
   `publish_active_predictions` (`src/nfl_ats/publishing.py`) calls this right
   after resolving the card view, validates the result with
   `validate_card_lineage` (fail closed -- a malformed or incomplete
   overlay/tiebreaker record blocks the publish, same footing as the
   artifact-contract and source-freshness gates already enforced there), and
   writes it as `lineage.json` **beside the published card artifact**
   (`destination.parent`, e.g. next to `CURRENT_PREDICTIONS.md`) -- never
   overwriting the forecast's own file under `artifacts/margin_predictions/`.
   The path and check list are surfaced on the publish summary as
   `played_card_lineage_path` / `played_card_lineage_checks_passed` /
   `card_metadata`. Within the policy only `player_arrests_back_side_policy`
   reads a snapshot of its own; `coach_fade`, `division_revenge_tilt` and
   `spread_gap_zone_fade` read the schedules table and the incoming card,
   already covered by the `model_input` records, and say so in
   `unknown_source_reason` -- unchanged from before this work.

   **Measured 2026-09-04, read-only, live Week 1 artifact**
   (`artifacts/margin_predictions/2026-week-01-20260903T143253Z`, still no
   `lineage.json` on disk, so this exercised the fresh-build fallback path;
   nothing under `artifacts/` or `data/` was written -- output went only to
   `%TEMP%\eng24\lineage.json`): the live four-member composition fired on 2
   of its 4 members this week -- `coach_fade` (2 flips: `2026_01_BAL_IND`,
   `2026_01_DAL_NYG`) and `spread_gap_zone_fade` (1 flip: `2026_01_CLE_JAX`);
   `division_revenge_tilt` and `player_arrests_back_side_policy` applied but
   flipped nothing. `overlay_sources_from_composition` accordingly returned 2
   records (one per member that fired, not one per flipped game), taking the
   base lineage's 15 decision-bearing fields to 20 once the 3 tiebreaker
   records below are added too. `validate_card_lineage` passed all four
   checks (`lineage_schema`, `lineage_required_fields`, `lineage_completeness`,
   `lineage_effective_timestamp`) on the result.
5. **Tiebreaker inputs are not emitted by any command.** `TiebreakerSource`
   existed and was tested, but `nfl-ats tiebreaker` wrote formatted text only
   and was wired into neither `weekly-run` nor `publish-predictions`. **Done
   (ENG-24):** the pool's tiebreaker game IS identifiable from data alone --
   `nfl_ats.tiebreaker.last_game_of_week(schedules, season, week)`, the week's
   last REG kickoff, already used by `tiebreaker_report` whenever a season/week
   are given -- so no separate configuration step was needed. A new adapter,
   `nfl_ats.tiebreaker.tiebreaker_lineage_sources(report, ...)`, turns an
   already-built `TiebreakerReport` (the library function's own structured
   result -- `tiebreaker_report` returned `game`/model margin/market
   margin/blend weight/predicted score as a dataclass separately from
   `format_report`'s text well before this work; the missing piece was only
   the lineage adapter and the publish-time wiring) into up to three
   `TiebreakerSource` records: `market_consensus` (always), `model_margin_view`
   and `model_total_view` (only when the active model / totals model actually
   priced the game). A capture stamp embedded in `MarketConsensus.source` /
   `ModelView.source` (both follow `nfl_ats.io.run_id`'s
   `%Y%m%dT%H%M%SZ` convention) resolves to a real `source_snapshot`/
   `source_captured_at`; the totals view and the schedules-fallback consensus
   name no such stamp and declare `unknown_source_reason` instead, same
   discipline as every other unrecordable source in this file.
   `publish_active_predictions` calls `tiebreaker_report` for the current
   card's own season/week and adds the resulting sources to the played-card
   lineage; a tiebreaker failure (no schedules snapshot, no lined game,
   misaligned totals table) degrades to zero tiebreaker records rather than
   blocking the card, the same fail-open contract the coach-fade snapshot
   fallback already uses elsewhere on this path. The CLI surface
   (`nfl-ats tiebreaker`'s argparse registration and its printed text) is
   UNCHANGED -- `tests/test_cli_contract.py` stays green -- because
   `publish_active_predictions` calls the library function directly rather
   than going through the CLI.

   **Measured 2026-09-04, read-only, same live artifact as above:** the
   designated tiebreaker game resolved to `2026_01_DEN_KC` (DEN at KC),
   producing 3 `TiebreakerSource` records -- `market_consensus` (snapshot
   `20260903T220048Z`), `model_margin_view` (snapshot `20260903T143253Z`,
   the forecast's own build stamp), and `model_total_view` (no snapshot;
   declares that its walk-forward training window is not a capture instant).
   The guess itself: KC 23 - DEN 20, matching the earlier owner-observed
   figure from the same 0.2 model/market margin blend.
6. **Weather and archive-backed families have no snapshot id.**
   `forecast_weather` reads `data/raw/forecast_archive/<name>/`, whose manifest
   is keyed by archive name rather than a capture stamp. Not exercised by the
   current production profile; registered in `FAMILY_BUILDERS` with a reason so
   it degrades honestly if a weather profile is ever promoted.

Any feature family with no entry in `lineage.FAMILY_BUILDERS` falls back to
`DEFAULT_FAMILY_BUILDER`, whose `unknown_source_reason` names the registry it
should be added to. That is deliberate: a new family gets a truthful "nobody
registered me" record rather than a silent gap or a hard failure.

## Adding a family

1. Give the building module a `BUILDER_VERSION` (or reuse its existing
   `*_FEATURE_VERSION`). `nfl_ats.features.BUILDER_VERSION` was added by this
   work; `pbp`, `players` and `quarterbacks` already had theirs.
2. Add a `FamilyBuilder` entry to `lineage.FAMILY_BUILDERS` naming the module,
   the version, and the manifest key holding its source snapshot. If there is no
   such key, supply an `unknown_source_reason` and add it to the gap list above.
3. If the manifest records the builder version, add the key to
   `lineage.MANIFEST_VERSION_KEYS` so the record reports what actually built the
   table rather than what is currently importable.

## ENG-22: propagating the base snapshot through every enrichment manifest

Closes gap item 1 above. New module `src/nfl_ats/feature_manifest.py`:
`inherit_source_snapshots(parent_manifest_paths)` reads a derived table's
upstream `*.manifest.json` sibling(s) and returns a merged `source_snapshots`
block naming every snapshot-bearing key it finds — `source_snapshot`,
`source_pbp_snapshot`, `source_depth_snapshot`, `source_player_snapshot`,
`source_player_value_snapshot`, `source_participation_snapshot` — on that
parent manifest *or on anything the parent itself already inherited*. A
parent manifest that cannot be read (missing, invalid JSON) degrades to one
explicit `{"snapshot_id": null, "reason": "upstream manifest absent"}` entry
rather than raising or silently dropping the dependency.

All five enrichment writers in `nfl_ats.cli_commands.features`
(`_cmd_build_pbp_features`, `_cmd_build_qb_features`,
`_cmd_build_player_features`, `_cmd_build_participation_features`,
`_cmd_build_learned_availability_features`) call
`inherit_source_snapshots([manifest_path_for(args.features)])` and, when the
result is non-empty, add it to their own manifest under `source_snapshots`
— right beside the existing ENG-09 `stamp()` call. Because each writer's only
manifest input is the one it directly enriches, the base `source_snapshot`
self-propagates transitively without any writer needing to know the base
table exists: `game_features_pbp.manifest.json` inherits it from
`game_features.manifest.json`, and `game_features_qb` /
`game_features_player` / `game_features_player_participation` /
`game_features_weak_stack` each inherit it (plus `source_pbp_snapshot`) from
`game_features_pbp.manifest.json` in turn. The base `_cmd_build_features`
writer is unchanged — it already records `source_snapshot` directly and has
no parent manifest to inherit from.

`lineage._family_record` and the `market_line` record in
`build_card_lineage` now check a manifest's `source_snapshots` block when the
direct `manifest_snapshot_key` lookup misses, before falling back to the
`feature_table:sha256` digest. A legacy manifest with no `source_snapshots`
block (every manifest on disk as of this writing) behaves exactly as before:
the digest fallback and its `unknown_source_reason` are unchanged, and the
existing 20 `tests/test_lineage.py` tests stay green. New tests:
`tests/test_feature_manifest.py` (single-parent merge, two-level transitive
inheritance, a missing parent that never blocks or hides a present one,
later-parent-wins collisions) and three additions to `tests/test_lineage.py`
(`test_market_and_base_families_prefer_an_inherited_snapshot_over_the_digest`,
`test_a_null_inherited_entry_still_falls_back_to_the_digest`,
`test_legacy_manifests_without_a_source_snapshots_block_are_unaffected`).

### The gap does not close until the tables are rebuilt

No production table under `data/processed/` was rebuilt to write this
section — those are live prediction inputs, not something an agent session
touches for real. What follows is read-only: computed in-session by calling
`build_card_lineage` directly against the real, on-disk artifact
`artifacts/margin_predictions/2026-week-01-20260903T143253Z` (feature profile
`weak_stack`, the one gap item 1 measured), with the manifest's
`source_snapshots` block set to exactly what `inherit_source_snapshots` would
produce today from the real `data/processed/game_features.manifest.json` and
`game_features_pbp.manifest.json` — i.e. simulating one rebuild pass without
performing one:

| Card field | Before ENG-22 (measured 2026-09-04, live) | After a rebuild (computed, simulated) |
| --- | --- | --- |
| `pick` | `feature_table:sha256:…` (its designed identity, not a gap) | unchanged |
| `model_probability` | `feature_table:sha256:…` (its designed identity, not a gap) | unchanged |
| `market_line` | `feature_table:sha256:…` (digest fallback) | `20260824T115346Z` (real) |
| `model_input:market` | digest fallback | `20260824T115346Z` |
| `model_input:context` | digest fallback | `20260824T115346Z` |
| `model_input:elo` | digest fallback | `20260824T115346Z` |
| `model_input:experience` | digest fallback | `20260824T115346Z` |
| `model_input:offense` | digest fallback | `20260824T115346Z` |
| `model_input:results` | digest fallback | `20260824T115346Z` |
| `model_input:defense` | digest fallback | `20260824T115346Z` |
| `model_input:bias` | digest fallback | `20260824T115346Z` |
| `model_input:player_qb` | `20260903T142643Z` (already real) | unchanged |
| `model_input:player_injuries` | `20260817T184901Z` (already real) | unchanged |
| `model_input:player_continuity` | `20260817T184901Z` (already real) | unchanged |
| `model_input:player_values` | `20260817T184911Z` (already real) | unchanged |

All 9 fields gap item 1 named resolve to the real base snapshot; `validate_card_lineage`
still passes on the simulated result. The 4 already-real player families and
the 2 fields that intentionally carry the feature-table identity are
unaffected either way — this table is the full 15-field decision-bearing set
measured for this artifact.

### Rebuild command sequence

The `weak_stack` production chain does not touch `build-qb-features`,
`build-player-features`, or `build-participation-features` — its own
manifest's `source_features` points straight at `game_features_pbp.parquet`
(see `docs/ops_runbook.md`'s manual fallback, which this mirrors). To make
the live card's manifest chain carry `source_snapshots` for real:

```
python -m nfl_ats build-features
python -m nfl_ats build-pbp-features --snapshot <PBP_SNAPSHOT>
python -m nfl_ats build-learned-availability-features --features data\processed\game_features_pbp.parquet --destination data\processed\game_features_weak_stack.parquet --rates-destination data\processed\weak_stack_availability_rates.parquet --evaluation-destination data\processed\weak_stack_availability_evaluation.csv --player-snapshot <PLAYER_SNAPSHOT> --player-value-snapshot <VALUE_SNAPSHOT> --pbp-snapshot <PBP_SNAPSHOT>
python -m nfl_ats margin-predict --season <SEASON> --week <WEEK> --features data\processed\game_features_weak_stack.parquet --feature-profile weak_stack
```

Order matters: each step's manifest inherits only from the parent named in
its own `--features`/`--destination` chain, so running a step out of order,
or skipping one, leaves that step's `source_snapshots` block un-propagated
and its children fall back to the digest exactly as before ENG-22. Any other
profile that also depends on `build-qb-features` / `build-player-features` /
`build-participation-features` needs those steps rebuilt first for the same
reason; they were not exercised by this work.
