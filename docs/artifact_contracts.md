# Artifact schema/version contracts (ENG-09)

ROADMAP Phase 13's definition of done: "Give feature tables, forecasts,
cards, and ledgers explicit schema and builder versions plus compatibility
checks, and refuse incompatible combinations before fitting or publishing."

Implementation: `src/nfl_ats/artifact_contracts.py` (new). Wired into
`src/nfl_ats/features.py`-family manifest writers and the `predict`/
`margin-predict` handlers in `src/nfl_ats/cli.py`, `src/nfl_ats/publishing.py`,
`src/nfl_ats/active_model.py`, and `src/nfl_ats/prediction_safety.py`. Tests:
`tests/test_artifact_contracts.py`.

## Why a second version axis

Two version mechanisms already existed before this module:

* Feature-table families carry fine-grained version constants recorded
  directly in the manifest (`pbp_feature_version`, `player_feature_version`,
  `qb_feature_version` — see `src/nfl_ats/pbp.py`, `players.py`,
  `quarterbacks.py`).
* Card provenance has `src/nfl_ats/lineage.py` (ENG-16): `schema_version` and
  `builder_version` on `CardLineage`, answering "where did this
  decision-bearing *field* come from".

Neither answered the artifact-kind-level question this module exists for:
given a model, the feature table it would fit on, and (optionally) a
forecast, do their **stamped versions agree**? That question needs a small,
uniform contract every artifact kind carries, plus one function that reads
two of those stamps together and decides whether they may be combined.

## The registry

`ARTIFACT_KINDS` (a `dict[str, ArtifactKindSpec]`) declares six kinds:

| Kind | Schema v | Builder version source | Required |
| --- | --- | --- | --- |
| `feature_table` | 1 | `nfl_ats.features.BUILDER_VERSION` | `built_at_utc`, `rows`, `destination` |
| `forecast` | 1 | `artifact_contracts.FORECAST_BUILDER_VERSION` | `created_at_utc`, `season`, `week` |
| `card` | 1 | `artifact_contracts.CARD_BUILDER_VERSION` | `model_id`, `season`, `week` |
| `decision_ledger` | 1 | `artifact_contracts.DECISION_LEDGER_BUILDER_VERSION` | `nfl_ats.clv.PAPER_DECISION_COLUMNS` (32 columns) |
| `pick_revision_ledger` | 1 | `artifact_contracts.PICK_REVISION_LEDGER_BUILDER_VERSION` | `nfl_ats.pick_refresh.PICK_REVISION_COLUMNS` (33 columns) |
| `lockday_package` | 1 | `artifact_contracts.LOCKDAY_PACKAGE_BUILDER_VERSION` | `kind`, `schema_version`, `season`, `week`, `created_at_utc` |

`feature_table` reuses `nfl_ats.features.BUILDER_VERSION` because that module
owns the base feature-table build path every family enriches. The other five
kinds have no single pre-existing owner, so `artifact_contracts.py` is that
owner — bump the matching `*_BUILDER_VERSION` constant there (and only
there) when this contract layer's own construction rules change for that
kind.

`lockday_package` is registered (so `read_contract`/`check_ledger` can
describe it uniformly) but deliberately **not** wired to `stamp()`:
`lockday_package.build_manifest` already owns a top-level `schema_version`
key with narrower, pre-existing semantics (the package format's own
version), and stamping would either collide with it or require renaming a
key an existing reader (`lockday_package.py`'s own loader,
`docs/`/`scripts/` references) depends on.

### Ledger required-column lists are copied, not imported

`_DECISION_LEDGER_COLUMNS` and `_PICK_REVISION_LEDGER_COLUMNS` inside
`artifact_contracts.py` are literal copies of `nfl_ats.clv.PAPER_DECISION_COLUMNS`
and `nfl_ats.pick_refresh.PICK_REVISION_COLUMNS`, not imports. Reason: both
`clv.py` and `pick_refresh.py` import `nfl_ats.prediction_safety`, and
`prediction_safety.py` imports `artifact_contracts.py` for
`CompatibilityReport` — an eager top-level import of `clv`/`pick_refresh`
from `artifact_contracts.py` would close that into a real circular import
(`prediction_safety` → `artifact_contracts` → `pick_refresh` →
`prediction_safety`, mid-initialization). `tests/test_artifact_contracts.py`
imports both source tuples directly (tests sit outside the cycle) and
asserts they still equal the registry's copies, so drift is caught
mechanically.

## The contract block

`stamp(kind, metadata) -> dict` returns a **copy** of `metadata` with one new
top-level key added:

```json
{
  "...": "...(every existing key, untouched)",
  "artifact_contract": {
    "kind": "feature_table",
    "schema_version": 1,
    "builder_version": "v1",
    "builder_module": "nfl_ats.features"
  }
}
```

Nested under `artifact_contract`, never flattened to top-level keys —
several artifacts already use a top-level `schema_version` key for something
else (`lockday_package`'s manifest, `CardLineage.to_dict()`'s lineage schema
version), so a flat stamp would either collide or be ambiguous about which
"schema_version" a reader is looking at.

`read_contract(path_or_metadata) -> ArtifactContract` reads that block back —
from a `Path` (parses JSON) or an already-loaded mapping. An artifact with no
`artifact_contract` block returns `ArtifactContract(legacy=True, kind=None,
schema_version=None, ...)` rather than raising: this is the expected shape
for every artifact that predates ENG-09, not an error.

## `check_compatible`

```python
check_compatible(
    model_manifest: Mapping[str, Any] | None,
    feature_table_manifest: Mapping[str, Any] | None,
    forecast_metadata: Mapping[str, Any] | None = None,
) -> CompatibilityReport
```

Two independent comparisons, both fail-soft into a `legacy_unversioned`
**warning** except for the one thing this function exists to catch — two
version stamps that are both present and disagree:

1. **Feature table vs. the model's own record of what it was fit on.**
   `feature_table_manifest`'s stamped `schema_version`/`builder_version`
   against `model_manifest["feature_table_schema_version"]`/
   `["feature_table_builder_version"]` (populated additively by
   `active_model.activate_matching_ats_model` — see below). Absent on either
   side → `legacy_unversioned` warning. Present on both and different →
   `version_mismatch` **hard failure**. `model_manifest=None` (no active
   model yet) skips this comparison entirely rather than treating "nothing to
   compare" as a mismatch.
2. **Forecast schema version**, when `forecast_metadata` is supplied. No
   `artifact_contract` block → `legacy_unversioned` warning. A block present
   but with a `schema_version` this code does not recognize →
   `unknown_forecast_schema` hard failure.

`CompatibilityReport.refuse_if_incompatible(action=...)` raises
`ArtifactContractError` when any hard failure is present; it is a no-op
otherwise (including when only warnings are present).

## `check_ledger`

```python
check_ledger(kind: str, columns: Iterable[str]) -> CompatibilityReport
```

For `decision_ledger`/`pick_revision_ledger`: a missing required column is
**always** a hard failure — there is no legacy-warning case, because both
ledger loaders (`clv.load_paper_decisions`, `pick_refresh.load_pick_revisions`)
already backfill legacy defaults for columns older artifacts lack before this
check would ever see them.

## Wiring

* **Feature-table manifests** (`nfl_ats.cli._cmd_build_features`,
  `_cmd_build_pbp_features`, `_cmd_build_qb_features`,
  `_cmd_build_player_features`, `_cmd_build_participation_features`,
  `_cmd_build_learned_availability_features`): each calls
  `metadata = stamp(KIND_FEATURE_TABLE, metadata)` immediately before writing
  `*.manifest.json`.
* **Forecast metadata** (`_cmd_margin_predict`, `_cmd_predict`): each calls
  `metadata = stamp(KIND_FORECAST, metadata)` after building the forecast's
  `metadata` dict, before it is written or fed to lineage/publish.
* **Before fitting**: both handlers call `check_compatible(active_model,
  feature_table_manifest)` immediately after loading the feature table and
  *before* calling `score_outcome_week`/`score_week` (where the model is
  actually fit), then `.refuse_if_incompatible(action="fit a model on this
  feature table")`. A malformed (not merely absent) active-model manifest is
  swallowed to "no active model to compare against" here — that failure mode
  is already surfaced loudly, and release-blockingly, by
  `publishing._publication_context` at publish time; this pre-fit check is an
  additional safety net, not the primary guard, so it degrades gracefully
  rather than blocking every command on an unrelated pre-existing problem.
* **Before publishing** (`publishing.publish_active_predictions`): calls
  `check_compatible(active, feature_table_manifest(metadata),
  forecast_metadata=metadata)` after the existing arrest/source-freshness
  gates and refuses with a clear message on a hard failure. The publish
  summary dict is stamped `KIND_CARD` and carries
  `artifact_contract_compatibility` (the full report, including any
  warnings) for visibility even when publish succeeds.
* **Active-model manifest** (`active_model.activate_matching_ats_model`):
  additively records `feature_table_schema_version` /
  `feature_table_builder_version`, read from the forecast's own
  `provenance.feature_table.manifest` block, as siblings of — never inside —
  `model_identity`, so `model_id`'s hash is unchanged. Absent when the
  feature table predates stamping (nothing is added; `check_compatible` then
  reports `legacy_unversioned`, not `null` fields).
* **`prediction_safety`**: `_contract_checks(compatibility)` mirrors the
  existing `_lineage_checks` pattern exactly. An additive `compatibility:
  CompatibilityReport | None = None` keyword on both `validate_prediction_card`
  and `validate_outcome_prediction_card` fails closed
  (`PredictionSafetyError`) on a hard failure and folds warnings into the
  audit's `warnings` list; passing nothing leaves the pre-existing contract
  unchanged. `validate_prediction_compatibility(report)` validates a report
  on its own, matching `validate_prediction_lineage`.

## Live read-only check (2026-09-04)

Run against the artifacts actually on disk in this repo (all three files
pre-date ENG-09, so this exercises the legacy path, not the mismatch path):

```
active model:     artifacts/active_ats_model.json
feature table:    data/processed/game_features_weak_stack.manifest.json
forecast:         artifacts/margin_predictions/2026-week-01-20260903T143253Z/metadata.json
```

```json
{
  "compatible": true,
  "issues": [
    {
      "severity": "warning",
      "code": "legacy_unversioned",
      "message": "active model manifest or feature-table manifest carries no artifact_contract version to compare (legacy artifact)"
    },
    {
      "severity": "warning",
      "code": "legacy_unversioned",
      "message": "forecast metadata carries no artifact_contract block (legacy artifact)"
    }
  ]
}
```

Both warnings are expected: none of these three files were built by the
ENG-09-stamped writers (they predate this change). `compatible: true` means
nothing here would refuse a fit or a publish. The next `build-features` /
`margin-predict` / `predict` run will stamp its own artifacts going forward;
the next real model activation will start recording
`feature_table_schema_version`/`feature_table_builder_version` on
`active_ats_model.json`, at which point a genuine builder upgrade (not
exercised here — no builder version actually changed) would surface as a
`version_mismatch` hard failure instead of a warning.
