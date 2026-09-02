# Named pregame quarterback state

PER-02 now has one deterministic feature-construction path that combines the
repository's timestamped depth identity, season-lagged availability
probability, and strictly prior quarterback performance state. This is
feature infrastructure only: no ATS experiment was run, no result was scored,
and no active model profile was changed.

## Decision-time contract

For each game and team, `enrich_with_qb_features`:

1. defines the decision timestamp as kickoff minus the configured number of
   hours (24 by default);
2. selects the latest depth-chart observation at or before that timestamp and
   rejects it when it is older than the configured maximum (14 days by
   default);
3. deterministically identifies QB1 and QB2 by `pos_rank`, with `gsis_id` as
   the stable tie-breaker;
4. uses only each named player's PBP appearances before the target game date
   to construct EPA/dropback, CPOE, sack, interception, explosive-pass, and
   experience states; and
5. uses only the latest starter injury revision visible by the same decision
   timestamp to calculate start probability.

If season-lagged availability rates are supplied, their existing contract
requires every target-season rate to have been learned from earlier seasons.
Otherwise the established fixed status prior is used. An absent starter row
means healthy only when that season exists in the injury source. An uncovered
season produces a null probability and null expected value rather than a
silent healthy assumption.

The expected EPA/dropback and CPOE states are probability-weighted mixtures of
the named QB1 and named QB2 states. A missing QB2 history therefore remains
missing whenever its probability has positive weight; it is never replaced
with a generic, anonymously valued backup. The output also exposes the two
identities, depth and injury observation timestamps, probability source,
side-level player states, and QB2-minus-QB1 adjustments for auditing. Every
new field uses a `depth_qb_*` namespace so it cannot silently overwrite the
older player pipeline's differently sourced `qb_*` columns.

## Registered family and operational build

The existing `player_qb` family retains ownership of its starter probability
and starter/expected EPA fields. The new `quarterback_depth` family owns the
fully namespaced depth-derived probability, starter, QB2, expected, and
adjustment columns. Both use the same availability resolver, while their
identity semantics remain explicit: `player_qb` retains its prior-appearance
projection and `quarterback_depth` uses the timestamped depth observation.
This preserves the one-owner-per-feature registry contract and prevents the
two tables from silently disagreeing under one column name. No existing model
profile includes `quarterback_depth`.

The normal build uses the latest PBP, depth, and player snapshots and the fixed
status prior:

```powershell
.\.tools\uv.exe run nfl-ats build-qb-features
```

The learned probability path is explicit and records the rate file and its
SHA-256 in the output manifest:

```powershell
.\.tools\uv.exe run nfl-ats build-qb-features `
  --availability-rates data/processed/player_availability_rates.parquet
```

The manifest records all three immutable source snapshot IDs, the selected
availability source, construction parameters, feature version, and coverage
counts for identities, states, named backups, and start probabilities.

For the conservative 2009–2024 weekly archive, select the separate historical
root explicitly; `docs/historical_depth_charts.md` documents its effective-time
rule, immutable snapshot, measured coverage, and intentional Week-1 gap.

## Leakage regressions

`tests/test_quarterbacks.py` pins the following contracts:

- current-game PBP mutations cannot change that game's pregame QB state;
- depth observations after the decision timestamp cannot change QB1 or QB2;
- injury revisions after the decision timestamp cannot change probability or
  expected value;
- fixed and season-lagged probabilities select the documented path;
- named QB2 value, not a replacement constant, enters the expected state; and
- uncovered injury seasons remain null and explicitly labeled.
