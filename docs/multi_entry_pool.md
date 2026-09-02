# Multi-entry ATS pool allocation (POL-06)

## Scope

`build_multi_entry_plan` creates several paper ATS pool cards from one weekly
probability card and never places or sizes a wager (**read:**
`src/nfl_ats/multi_entry.py:126-263`). Entry 1 takes the higher-probability ATS
side in every game; later entries diversify under an explicit pairwise-overlap
ceiling (**read:** `src/nfl_ats/multi_entry.py:160-184`).

This is deliberately a weekly allocator, capped at 18 games, because enumerating
candidate card variations is exponential and NFL weekly slates fit beneath that
bound (**read:** `src/nfl_ats/multi_entry.py:151-152`). It is not a season-level
optimizer and does not use future outcomes (**read:** the function input contract
in `src/nfl_ats/multi_entry.py:126-133`).

## Allocation rule

Flipping one game away from the primary card reduces expected correct picks by
`2 * abs(home_cover_probability - 0.5)` (**read:**
`src/nfl_ats/multi_entry.py:160-164`). Candidate cards are ordered by total
expected loss, then flip count, then `game_id`; the allocator takes the first
card satisfying the overlap ceiling against every previously selected card
(**read:** `src/nfl_ats/multi_entry.py:79-123`). This is a deterministic greedy
expected-score rule, not a claim of a globally optimal contest portfolio
(**read:** the explicit `method` and `candidate_order` audit fields at
`src/nfl_ats/multi_entry.py:248-261`).

Two optional hard controls make the tradeoff reviewable: maximum flips from the
primary card and maximum expected-correct loss per entry (**read:**
`src/nfl_ats/multi_entry.py:131-133,153-158`). Infeasible combinations fail
instead of silently relaxing those controls (**measured:**
`tests/test_multi_entry.py::test_flip_and_expected_loss_caps_fail_closed_when_diversification_is_impossible`).

## Audit output

`MultiEntryPlan.entries` retains every pick, its probability, whether it differs
from the primary, its game-level expected-score cost, and entry-level totals
(**read:** `src/nfl_ats/multi_entry.py:185-218`). `MultiEntryPlan.overlap`
retains agreements, disagreements, overlap rate, and the limit verdict for every
entry pair (**read:** `src/nfl_ats/multi_entry.py:220-246`). The focused tests
check all pairwise limits, deterministic order, exact expected-score accounting,
input validation, and bounded weekly scope (**measured:** `tests/test_multi_entry.py`).
