# Injury scenario margin mixture

Status: **distribution kernel implemented 2026-09-02; PER-10 remains open
because the repository has no defensible joint-scenario producer**.

No ATS experiment was run, no registry or active-model decision changed, and
the played-card path does not import this module.

## Audit result

**Read from `src/nfl_ats/quarterbacks.py`:** the named-depth path supplies one
starter probability per team, stable QB1/QB2 identities, visible depth and
injury timestamps, and each named quarterback's strictly prior EPA/CPOE state.
Its expected fields are marginal two-state means, not complete game-level
lineup scenarios.

**Read from `src/nfl_ats/players.py`:** injury availability and value features
sum player-level expected burdens. The feature table does not retain a joint
probability over combinations such as “QB1 inactive, WR1 active, LT inactive.”
Multiplying the marginal probabilities would assert conditional independence,
which is not established by the source or existing learned-availability model.

**Read from `src/nfl_ats/margin.py`:** each fitted margin model produces a
scenario center plus an empirical sample of out-of-time residuals. This is
already the correct distribution primitive to reuse once a scenario-specific
center exists. Neither the QB nor player feature path defines how EPA/dropback,
CPOE, role share, or value-lost state becomes a margin-point adjustment for a
particular lineup. No existing feature profile includes the named
`quarterback_depth` family.

These are two separate blockers to a complete producer:

1. a joint, mutually exclusive probability distribution over all uncertain
   player identities; and
2. a frozen mapping or fitted scenario-aware margin model that supplies the
   predicted margin center for every joint state.

Choosing independence, a copula/correlation model, replacement-player rules,
or an EPA-to-points multiplier would be a new research/model decision. The
kernel therefore requires those inputs explicitly and fails closed rather
than choosing for the caller. PER-10 stays open until both inputs have a
point-in-time implementation and adequate historical coverage.

## Implemented contract

`src/nfl_ats/injury_scenarios.py` accepts immutable scenario revisions with:

- game, revision, scenario, and source identity;
- probability and scenario-specific predicted-margin center;
- complete active and inactive player-ID collections;
- observed and effective timestamps; and
- the game's decision timestamp, spread, and an already-fitted out-of-time
  residual sample.

Only revisions whose observed **and** effective timestamps are no later than
the decision cutoff are eligible. The latest eligible atomic revision is
selected by observation/effective chronology; tied revisions fail as
ambiguous. The selected revision must contain at least two positive-probability
scenarios whose probabilities sum to one within `1e-9`.

Every scenario must partition the same non-empty player-identity universe:
each player is exactly active or inactive. A player on both sides, a missing
identity, an incomplete universe, a duplicate scenario ID, or a duplicate
active/inactive signature fails closed. The selected revision must also carry
one non-empty immutable source identity.

For scenario s with probability p(s), predicted center m(s), and common
out-of-time residual R, the mixture is M = m(s) + R. Its exact discrete
moments are:

```text
E[M]   = sum_s p(s) * (m(s) + E[R])
Var[M] = sum_s p(s) * (Var[R] + (m(s) + E[R] - E[M])^2)
```

Win and cover probabilities are probability-weighted scenario CDF reads using
the same continuity-corrected empirical reader as `MarginModel`; the
cover/push/loss split reuses its integer-margin settlement logic. The three
settlement probabilities therefore continue to sum to one.

## Regressions

`tests/test_injury_scenarios.py` pins:

- a hand-computed two-component mixture with residuals `[-1, +1]`, centers
  `0/4`, and probabilities `0.25/0.75`: mean `3`, variance `4`, smoothed cover
  probability `5/12`, and cover/push/loss `0.375/0.375/0.25` at a line of `3`;
- a future revision with radically different probabilities and centers cannot
  change the decision-time mixture, but becomes selectable after its observed
  timestamp;
- malformed probability mass, incomplete identity partitions, within-scenario
  active/inactive overlap, and duplicate signatures all fail closed; and
- absent or provenance-ambiguous visible revisions fail closed.

The kernel has no automatic feature-profile or card integration. A future
producer must pass the same leakage and coverage tests before this roadmap row
can close, and any ATS comparison remains a separately predeclared decision.
