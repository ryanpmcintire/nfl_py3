# Correlated paper-portfolio sizing

`nfl_ats.portfolio.size_correlated_paper_portfolio` implements BET-05 as a
paper-analysis API. It returns stake fractions and diagnostics only. It does
not settle picks, write a ledger, contact an external service, or provide an
automated wagering path.

The allocator maximizes a deterministic quadratic approximation to fractional
Kelly growth under three layers of risk control:

1. each candidate is bounded by its independent fractional-Kelly amount and a
   caller-supplied per-candidate cap;
2. the full simultaneous slate has an aggregate exposure cap; and
3. callers may cap the absolute exposure to any named factor.

HOME/AWAY candidates automatically receive signed `team:<TEAM>` loadings, so
two positions exposed to the same team are not treated as independent.
Additional exposures are opt-in and must use a `total:`, `weather:`, or
`market:` prefix. Their non-negative scenario strengths must be named
explicitly. These strengths are risk assumptions, not measured correlations.
Alternatively, callers may supply a covariance `DataFrame`; its row and column
labels must exactly equal the active `game_id` values, and it must be finite,
symmetric, positive-semidefinite, and have a strictly positive diagonal. The
function reorders a valid labeled matrix and refuses ambiguous positional
input.

The result preserves PASS rows at zero allocation and includes the covariance,
factor loadings, expected paper return, portfolio variance, binding caps, and
optimizer iteration count. Optional factor caps use the same labels as the
exposure matrix; for example, `{"weather:windy": 0.02}` bounds the absolute
net windy-weather fraction at two percent of the hypothetical bankroll.

This is scenario analysis conditional on model probabilities and caller-supplied
risk assumptions. It does not establish that those probabilities or covariance
assumptions are correct, and it is not betting advice.
