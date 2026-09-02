# Pool workbench: editable ATS entry and ownership sensitivity (UI-09)

## Scope

The workbench serves this project's actual ATS forced-pick pool: one side for
every game and one Best Pick per regular-season week (**read**:
`docs/pool_edge_plan.md:73-88`; `src/nfl_ats/pool_workbench.py`, `PoolRules`).
It does not add straight-up, confidence, survivor, or multi-entry behavior
(**read**: `ROADMAP.md`, POL-01/POL-06; `src/nfl_ats/pool_workbench.py`).

## Entry persistence

Each game now exposes both ATS sides and a Best Pick selector, with the active
model card as the initial state (**read**:
`src/nfl_ats/pool_workbench.py`, `_entry_list_section`). Saving writes schema
version 1, the selected `HOME`/`AWAY` side per current `game_id`, and the Best
Pick `game_id` to browser `localStorage` (**read**:
`src/nfl_ats/pool_workbench.py`, `_entry_persistence_script`).

The key is scoped as `nfl-ats:pool-entry:v1:<season>:<week>`, and saving is
disabled when season/week is absent, so one unscoped page cannot overwrite a
different week's entry (**read**: `src/nfl_ats/pool_workbench.py`,
`build_pool_workbench_body`). Restores accept only the current schema version,
known current-card game IDs, and literal `HOME`/`AWAY` values; unknown or stale
values are ignored and the model default remains in place (**read**:
`src/nfl_ats/pool_workbench.py`, `_entry_persistence_script`). Browser storage
failure leaves the editor usable and reports that the entry was not saved
(**read**: the same script).

The saved entry is local UI state only; it does not modify an artifact, publish
a forecast, or submit a pool card anywhere (**read**:
`src/nfl_ats/pool_workbench.py`, the entry note and local-only script).

## Ownership scenarios

No pre-deadline pool-popularity feed is integrated (**read**:
`docs/pool_format_levers.md`, section 4; `ROADMAP.md`, POL-04). The workbench
therefore labels every ownership row as a hypothetical sensitivity case, never
as observed ownership (**read**: `src/nfl_ats/pool_workbench.py`,
`OwnershipScenario` and `_ownership_section`).

The three cases assume 50%, 65%, or 85% of a hypothetical field takes each
game's favorite; these are the control and two crowding inputs already used by
the POL-05 field-model sensitivity analysis (**read**:
`docs/pool_format_levers.md`, sections 3 and 5; `src/nfl_ats/pool.py`,
`FieldModel`). For each case, the page reports the average hypothetical field
share on the entry's selected sides and its complement as disagreements per 100
games; a pick'em contributes 50% because no favorite exists (**read**:
`src/nfl_ats/pool_workbench.py`, `build_ownership_scenarios` and the matching
browser recomputation).

Changing an entry pick recomputes those rows in the browser, while the visible
warning says the favorite is only a public-side proxy and that a scenario is
not a reason to flip a higher-expected-value forced pick (**read**:
`src/nfl_ats/pool_workbench.py`, `_ownership_section` and
`_entry_persistence_script`).

## Verification contract

Focused tests cover scenario arithmetic and validation, empty-state behavior,
week-scoped browser keys, local save/restore/reset hooks, stale-value rejection,
storage failure messaging, and the public-page disclaimers (**read**:
`tests/test_pool_workbench.py`). The repository-wide required checks remain the
release gate (**read**: `AGENTS.md`, “Required verification”).
