# Pool rules: the composed `PoolRules` contract (POL-01)

## Real-pool default

`PoolRules.from_defaults()` still describes the pool this project actually
plays: ATS forced picks, no passes, 272 regular-season plus 13 playoff picks,
one Best Pick per regular-season week, one entry, and opener grading
(**read:** `src/nfl_ats/pool_workbench.py`, `PoolRules`; source rules in
`docs/pool_edge_plan.md:73-88`). `cards_per_season` remains the 285 picks in one
entry, while `submissions_per_season` multiplies that count by `entry_count`
(**read:** `src/nfl_ats/pool_workbench.py`, the two properties).

The deadline was not generalized or reimplemented. Every variant delegates to
the same `pick_refresh.sunday_pick_lock` and `pick_refresh.pick_deadline`
functions, producing `min(game kickoff, that week's Sunday 16:00 ET lock)`
(**read:** `src/nfl_ats/pool_workbench.py`, `deadline_for`; and
`src/nfl_ats/pick_refresh.py:132-157`). Focused tests compare standard,
straight-up, confidence, and survivor instances at the same late kickoff
(**measured:**
`tests/test_pool_rule_variants.py::test_all_variants_keep_the_real_per_game_deadline_function`).

## Orthogonal rule dimensions

The configuration now separates three concepts so a variant cannot silently
reuse the wrong target or scoring rule (**read:**
`src/nfl_ats/pool_workbench.py`, `PoolRules.__post_init__`):

| Dimension | Supported values | Meaning |
|---|---|---|
| `pick_type` | `ats`, `straight_up` | ATS reads cover probabilities and a grading line; straight-up grades only the game winner. |
| `pool_type` | `standard`, `confidence`, `survivor` | All-game picks, ranked confidence points, or one reusable-team-constrained weekly selection. |
| `scoring_method` | `correct_picks`, `confidence_points`, `survival` | Must match the selected `pool_type`; it is not inferred later by a consumer. |

Standard correct-pick scoring carries explicit correct, incorrect, push, Best
Pick bonus, and Best Pick penalty values (**read:**
`src/nfl_ats/pool_workbench.py`, `PoolRules` fields). Confidence configuration
uses `unique_1_to_game_count`, meaning each weekly value is assigned once
(**read:** `PoolRules.confidence`). Survivor configuration requires straight-up
picks, one use per team, no Best Pick award, and a positive number of lives
(**read:** `PoolRules.survivor` and `PoolRules.__post_init__`).

`entry_count` is a positive integer for every format (**read:**
`PoolRules.__post_init__`). `submissions_per_season` is deliberately `None` for
survivor because the stored game totals cannot determine the number of weekly
survivor selections (**read:** `PoolRules.submissions_per_season`). POL-06's
card-allocation implementation is separate from this rule declaration
(**read:** `src/nfl_ats/multi_entry.py`).

## Constructors and validation

Four entry points cover the supported configurations (**read:**
`src/nfl_ats/pool_workbench.py`, `PoolRules` class methods):

- `from_defaults()` returns the unchanged real ATS rules;
- `straight_up()` changes the pick target and grades by `result`, never by a
  spread;
- `confidence(pick_type="ats" | "straight_up")` configures unique weekly
  confidence points;
- `survivor()` configures straight-up survival with one team use.

`from_dict()` continues to accept partial overrides and ignore unknown keys,
while filling the required scoring/target defaults for named variants
(**read:** `PoolRules.from_dict`). Directly contradictory combinations fail:
forced picks plus passes, mismatched pool/scoring methods, spread grading for
straight-up picks, result grading for ATS, confidence assignment outside a
confidence pool, team reuse outside survivor, or invalid counts/nonfinite point
values (**measured:** `tests/test_pool_rule_variants.py`).

## Verification

The variant tests cover defaults, straight-up, ATS and straight-up confidence,
survivor lives/reuse, multiple entries, shared deadline semantics, partial-dict
loading, descriptions, and invalid combinations (**measured:**
`tests/test_pool_rule_variants.py`). The original workbench tests continue to
cover the real format, exact imported deadline function, card construction,
ownership scenarios, and rendering (**measured:**
`tests/test_pool_workbench.py`). This is configuration and validation only; no
historical outcomes, experiment scoring, model promotion, or wager execution
enters the contract (**read:** the inputs and imports of
`src/nfl_ats/pool_workbench.py`).
