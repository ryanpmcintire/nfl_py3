# Paper sizing-policy comparison

`nfl_ats.policy_comparison.compare_paper_policies` implements BET-07 as a
deterministic replay over caller-supplied decisions and outcomes. It compares:

- a constant flat unit based on the initial hypothetical bankroll;
- caller-configurable confidence tiers;
- independent quarter-Kelly sizing; and
- quarter-Kelly sizing with BET-05's covariance and factor-risk constraints.

Every policy sees the same rows, prices, probabilities, and results. Stakes for
all games in a season/week are fixed from the bankroll at the start of that
week, then settled together. Per-position and weekly caps apply to every
policy. The risk-constrained policy additionally supports shared-team risk,
optional `total:`/`weather:`/`market:` exposures, absolute factor caps, and a
fully labeled covariance matrix for every active week.

The returned ledger contains one row per input prediction per policy. PASS,
push, zero-stake, and unresolved rows remain visible; unresolved paper stakes
have `profit = NaN` and do not fabricate a bankroll change. The metrics table
reports final bankroll, return, net profit, peak, trough, maximum drawdown,
turnover, and settled win/loss/push counts for each policy.

If decisions include both `prediction_timestamp` and `kickoff`, the replay
fails unless every prediction precedes its kickoff. Outcome IDs must match the
decision IDs exactly, and non-push `home_cover` values must agree with the sign
of `ats_margin`.

This API returns analysis objects only. It writes no ledger, performs no model
selection, names no winning policy, contacts no external service, and has no
automated wagering path. Its results are conditional on the supplied model
probabilities, prices, covariance assumptions, and sizing configuration; they
are not evidence that any policy is profitable and are not betting advice.
