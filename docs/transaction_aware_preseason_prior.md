# Transaction-aware preseason prior

## Scope

This is a point-in-time feature foundation, not an ATS experiment or a model
promotion. **Read:** `src/nfl_ats/preseason_prior.py:209` accepts decision rows,
caller-supplied adjustments, and caller-supplied configuration; it never accepts
results, covers, prices, or labels. **Read:** `src/nfl_ats/preseason_prior.py:40-69`
sets every component weight to `0.0` by default, so creating a non-neutral prior
requires an explicit caller choice.

The five supported components are `qb`, `roster`, `coaching`, `draft`, and
`free_agency`. **Read:** that closed vocabulary is declared at
`src/nfl_ats/preseason_prior.py:17`. Inputs must already express all five kinds
of adjustment in one common unit; the builder validates the caller's `units`
label and performs arithmetic, rather than inventing conversions between unlike
signals. **Read:** validation is at `src/nfl_ats/preseason_prior.py:142-177`.

## Input contract

`decisions` requires:

| Column | Meaning |
| --- | --- |
| `season` | Season associated with the prior |
| `team` | NFL team code; historical aliases are canonicalized |
| `decision_at_utc` | Exact as-of boundary for this prior |

Other decision columns, such as `game_id`, are preserved in `result.priors`.
**Read:** the required decision fields and preservation behavior are implemented
at `src/nfl_ats/preseason_prior.py:19` and `src/nfl_ats/preseason_prior.py:319-325`.

`adjustments` requires:

| Column | Meaning |
| --- | --- |
| `season`, `team`, `component` | Join key and one of the five named components |
| `adjustment` | Caller-supplied signed value in the configured common units |
| `uncertainty` | Caller-supplied non-negative uncertainty in those units |
| `units` | Must equal `PreseasonPriorConfig.units` |
| `source_id` | Stable, non-empty audit identifier |
| `source_observed_at_utc` | When the information became observable |
| `effective_at_utc` | When the described change became effective |
| `application` | `additive` or `override` |
| `override_priority` | Caller-declared numeric priority for deterministic overrides |

**Read:** these required fields are declared at
`src/nfl_ats/preseason_prior.py:22-38`. Duplicate
`(season, team, component, source_id)` rows are rejected, preventing accidental
double counting. **Read:** `src/nfl_ats/preseason_prior.py:173-176`.

## As-of, decay, and override rules

A source is visible only if both `source_observed_at_utc <= decision_at_utc`
and `effective_at_utc <= decision_at_utc`. **Read:** the two filters are at
`src/nfl_ats/preseason_prior.py:238-239`. Consequently, a backdated transaction
first learned after the decision is unavailable, and a known transaction that
has not taken effect is also unavailable.

Each component has an explicit `weight` and optional `half_life_days`. With no
half-life the decay factor is `1`; otherwise it is
`0.5 ** (age_days / half_life_days)`, where age begins at `effective_at_utc`.
**Read:** `PriorComponentRule` validates this configuration at
`src/nfl_ats/preseason_prior.py:40-52`, and the decay arithmetic is at
`src/nfl_ats/preseason_prior.py:176-180`.

Visible additive sources are summed unless any visible override exists for the
same decision/team/component. An override suppresses all additive sources and
the winner is selected by highest `override_priority`, newest
`source_observed_at_utc`, then lexical `source_id`. **Read:** the stable ordering
and selection are implemented at `src/nfl_ats/preseason_prior.py:270-284`.

## Outputs and audit

`PreseasonPriorResult.priors` preserves decision columns and adds:

- the configured baseline and units;
- one contribution and uncertainty column for every named component;
- total adjustment, resulting prior value, and quadrature uncertainty;
- visible-source count, component override count, and latest visible source
  timestamp.

**Read:** aggregation and output construction are at
`src/nfl_ats/preseason_prior.py:297-325`. Quadrature is only deterministic
propagation of caller-supplied uncertainty; it is not a fitted posterior or a
claim that sources are empirically independent.

`PreseasonPriorResult.source_audit` retains one row per visible source and
decision. It records source/effective timestamps, age, decay, configured
weight/half-life, whether the source was selected, the selection reason, and
its weighted adjustment and uncertainty. **Read:** the audit schema is declared
at `src/nfl_ats/preseason_prior.py:182-206`.

## Example

```python
config = PreseasonPriorConfig(
    qb=PriorComponentRule(weight=1.0, half_life_days=90.0),
    roster=PriorComponentRule(weight=0.5),
)
result = build_transaction_aware_preseason_prior(
    decisions,
    adjustments,
    config=config,
)
prior_rows = result.priors
source_audit = result.source_audit
```

Those numbers are policy inputs supplied by the caller, not estimates selected
from game outcomes. **Read:** no outcome or evaluator argument exists in the
public function signature at `src/nfl_ats/preseason_prior.py:209-214`.

## Leakage regression

**Measured:** `tests/test_preseason_prior.py::test_post_cutoff_mutations_cannot_change_prior_or_source_audit`
builds a prior, mutates a post-cutoff source by a large amount, adds another
post-cutoff source, and asserts byte-equivalent pandas frames for both the prior
and audit. The focused module suite passed 11 tests in the implementation
session with `uv run --no-sync pytest tests/test_preseason_prior.py -q`.
