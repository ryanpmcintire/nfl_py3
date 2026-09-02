# Weak-signal registry: CLI reference

`registry/weak_signals.json` is the ledger implemented in
`src/nfl_ats/weak_signals.py` (see that module's docstring and
`docs/pool_edge_plan.md` for the taxonomy and the reasoning behind it). This
page is the CLI reference: what `nfl-ats weak-signals <subcommand>` does, the
full `effect_units` vocabulary and its sign conventions, and the
`retag-units` repair path added 2026-09-01 (WP16).

## Subcommands

- **`status [--classification ...]`** — lists every recorded signal: effect,
  units, direction (`favours_candidate`), family, source. Read-only.
- **`record --name ... --effect ... --effect-units ... --classification ...
  --league ... --season-start ... --season-end ... [more]`** — adds one new
  signal. Refuses to overwrite an existing name unless `--replace` is passed.
  Enforces the AGENTS.md closing-grounds taxonomy at write time (a terminal
  classification needs an admissible `--closing-ground`).
- **`pool [--league ...] [--effect-units ...] [--method random|fixed]`** —
  sign test plus inverse-variance pooling across the unresolved
  (`unresolved_below_power`) pile, with per-family overlap warnings. Refuses
  to mix leagues or mix effect units within one pooled group (units must be
  commensurable — see AGENTS.md). Read-only; safe to run against an empty or
  single-entry bucket for any unit, including the new ones below.
- **`retag-units --name ... --effect-units ... --reason ...`** — corrects a
  mis-tagged `effect_units` on one already-recorded entry (see below).

Mass-recording waves go through the serialized queue in
`docs/batch_record.md`, not direct `record` calls; `retag-units` is a
single-entry repair and is expected to be run directly, wrapped in the
session's cross-process lock like any other registry write.

## `effect_units` vocabulary and sign semantics

Every unit in this registry stores its `effect` field so that **positive
always favours the candidate**, whatever the underlying metric's own
polarity (`WeakSignal.favours_candidate` is simply `effect > 0.0` — the same
rule for every unit, with no per-unit special-casing anywhere in the pool or
sign-test code). How a caller gets to that sign varies by unit:

| Unit | Native meaning | How to get "positive = candidate better" |
|---|---|---|
| `ats_points` | Points of margin/line error | Natural: better already reads positive. |
| `accuracy_points` | Percentage points of forced-pick accuracy (record `1.10`, not `0.011`) | Natural. |
| `brier`, `log_loss` | Raw Brier / log-loss difference | **Ambiguous on its own.** Brier and log-loss improve *downward*, so the convention requires the caller to negate the natural `(candidate − baseline)` difference before storing. Nothing in the unit name says this happened — a `+0.0015` here means "0.0015 lower (better)", but only the recording session's `notes` say so. Kept unchanged for existing entries recorded this way. |
| `mae` | Raw mean-absolute-error difference | Same ambiguity as `brier`/`log_loss`: improves downward, so the stored value must already be the negated, pre-flipped natural difference. |
| `correlation` *(new 2026-09-01)* | A Pearson (or equivalent) correlation coefficient, native range `[-1, +1]` | Positive = the **predeclared candidate-favouring direction** — same convention as every other unit, not "positively correlated" independent of what direction was predeclared. |
| `mae_improvement` *(new)* | Points of MAE, but **higher is better** | Store `(baseline_mae − candidate_mae)` directly — no extra negation. `+0.00082` means the candidate's MAE was `0.00082` lower. |
| `brier_improvement` *(new)* | Brier points, **higher is better** | Store `(baseline_brier − candidate_brier)` directly. |
| `log_loss_improvement` *(new)* | Log-loss points, **higher is better** | Store `(baseline_log_loss − candidate_log_loss)` directly. |

The three `*_improvement` units and `correlation` do not change any stored
number's meaning versus the pre-existing convention — `favours_candidate`,
`pooled_effect`, and `sign_test` treat them exactly like every other unit.
Their entire point is that the unit **name** now says which way is better, so
a pooler never has to open `notes` to find out. Prefer them over bare
`mae`/`brier`/`log_loss` for any new entry recording an explicit improvement,
and prefer `correlation` over stuffing a correlation coefficient into
`accuracy_points` as a bare numeric container.

`weak-signals pool --effect-units <unit>` works the same way for every unit,
including the four new ones, on any bucket size — zero eligible signals
returns `"pooled_by_unit": {}` with `"eligible": []` and no error; one
eligible signal pools against itself (`sharpening_vs_best_single: 1.0`) with
no error.

## `retag-units`: fixing a mis-tagged unit after the fact

Two entries were recorded before `correlation`/`*_improvement` existed and
had to be forced into the closest available unit, with the true sign
explained only in prose inside `notes` — exactly the kind of note a pooler
combining many entries will not read. `retag-units` is the narrow repair:

```
nfl-ats weak-signals retag-units --name <entry> --effect-units <unit> --reason <text>
```

- Changes **only** `effect_units` on the named entry.
- Appends one audit line to that entry's `notes`:
  `[<UTC timestamp>] effect_units retagged: '<old>' -> '<new>'. Reason: <text>`
  — the pre-existing `notes` text (including any original sign-convention
  explanation) is preserved above the audit line, never deleted.
- Refuses (raises, CLI exits non-zero) if the entry does not exist, or if
  `--effect-units` is not one of `EFFECT_UNITS`.
- Touches nothing else: `effect`, `interval`, `standard_error`,
  `probability_positive`, `sample_games`, `sample_blocks`, `classification`,
  `classification_evidence`, `closing_ground`, `reliability`, `family`,
  `league`, `seasons`, `source`, `description`, `recorded_at`,
  `plain_summary`, `category` are all carried over unchanged (AGENTS.md
  forbids silently rewriting a recorded measurement; a unit correction is not
  a new measurement). Enforced by
  `test_retag_effect_units_changes_only_the_unit_and_appends_an_audit_note`
  in `tests/test_weak_signals.py`.

Like every other registry write, wrap `retag-units` in the session's
cross-process lock when other agents may be writing concurrently.

### Worked example (2026-09-01, WP16)

```
nfl-ats weak-signals retag-units \
  --name totals_market_residual_blend \
  --effect-units mae_improvement \
  --reason "was recorded under mae with the sign convention explained only in notes; ..."

nfl-ats weak-signals retag-units \
  --name xlg06_rookie_prior_stage1_qb \
  --effect-units correlation \
  --reason "was recorded under accuracy_points as a numeric container only; ..."
```

`totals_market_residual_blend` (family `totals_market_residual`, league
`nfl`) moved from `mae` to `mae_improvement`; `xlg06_rookie_prior_stage1_qb`
(family `xlg06_rookie_prior_stage1`, league `cfb`) moved from
`accuracy_points` to `correlation`. In both cases `effect`, `interval`,
`probability_positive`, `classification`, `closing_ground`, `reliability`,
and every other field are byte-identical before and after — read
`registry/weak_signals.json` directly to confirm on any given day, since this
page is not regenerated when the registry changes.
