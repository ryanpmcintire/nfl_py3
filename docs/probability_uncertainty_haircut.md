# Candidate-specific probability uncertainty haircut (BET-04)

## Scope

The paper-sizing functions retain the scalar `probability_haircut` fallback and
also accept a labeled, caller-supplied uncertainty input (**read:**
`src/nfl_ats/portfolio.py`; `src/nfl_ats/probability_uncertainty.py`).

This changes paper stake sizing only. It reads the already-selected `bet_side`,
never selects or flips a forced pick, and has no wager-placement path (**read:**
`src/nfl_ats/probability_uncertainty.py`, `conservative_probability_audit`). If a
conservative probability reaches 0.5, Kelly exposure is zero while the side
remains on the ledger (**measured:**
`tests/test_probability_uncertainty.py::test_uncertainty_that_removes_edge_sizes_zero_but_does_not_change_pick`).
This is sizing uncertainty, not a promotion or forced-pick decision threshold
(**read:** `src/nfl_ats/probability_uncertainty.py:3-4`).

## Input contract

`probability_uncertainty` is a DataFrame indexed by the active candidates'
`game_id`. Its index must match exactly, and every row carries the same
`bet_side` as the card plus exactly one uncertainty statement (**read:**
`src/nfl_ats/probability_uncertainty.py`, `conservative_probability_audit`):

- `probability_lower_bound`: a caller-computed lower bound for the selected
  side, constrained not to exceed its point probability;
- `posterior_sd`: a caller-supplied posterior standard deviation, converted to
  `point_probability - posterior_z * posterior_sd`.

The result is floored at 0.5 for sizing, and the scalar haircut cannot be
silently stacked with candidate-specific uncertainty (**read:**
`src/nfl_ats/probability_uncertainty.py`). The caller owns the meaning of its
bound or posterior and the choice of `posterior_z`; this code does not infer
either from historical results (**read:** the module has no outcome or evaluator
input).

## Audit and integrations

Every sized ledger/allocation retains the point and conservative probabilities,
method, supplied bound or posterior SD, multiplier, and effective haircut
(**read:** `src/nfl_ats/probability_uncertainty.py`, `AUDIT_COLUMNS`). The same
transformation feeds sequential paper sizing, conditional bankroll Monte Carlo,
and correlated paper sizing (**read:** `src/nfl_ats/portfolio.py`). Mixed
lower-bound/posterior inputs and all three integrations are covered without
empirical scoring (**measured:** `tests/test_probability_uncertainty.py`).
