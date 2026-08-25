# RWB-18: per-family overlap warnings + weak-signal validator tightening

Task id: `blog-rwb18-weak-signal` · Branch: `swarm/blog-rwb18-weak-signal` · Date: 2026-08-24

Provenance labels used below per AGENTS.md binding rule 2: **measured** = run this
session (command cited); **read** = file opened this session (path cited);
**inferred** = my reasoning, labelled.

## What was asked

ROADMAP item RWB-18 (read, `ROADMAP.md` line 115): make `weak-signals
pool`/`status` carry per-family overlap warnings consistent with HANDOFF;
verify all registry entries hold admissible classification values; fix
validator gaps as code; **no reclassification of any entry**.

## Context read this session

- **read** `HANDOFF.md`, `ROADMAP.md` RWB-16/RWB-18 rows, and the RESOLVED
  section of `docs/revisit_list.md` (lines 14–67): Tier 1 shrank to D4 + two
  bare-verdict entries; `registry/weak_signals.json` holds zero
  `refuted_mechanism` entries after the 2026-08-18 re-runs — superseded here:
  **measured** this session, the live ledger now contains exactly one terminal
  entry (`offseason_retention_per_metric_cfb`, `refuted_mechanism` /
  `wrong_sign_resolved`, effect −0.739, interval [−1.296, −0.199], P+ 0.0037),
  whose whole interval sits below zero — an admissible closure under the
  validator, left untouched.
- **read** `docs/registry_correlation_audit_20260822.md`: documents the pool
  overlap explosion (55,824 pairwise warnings at that date; risk #3) and the
  §3 family correlation map this feature mirrors. Its recommended actions
  (supersede/annotate duplicate body-clock cells, fix `cfb` tags on four
  `best_pick_followup_*` entries) are owner decisions I did NOT execute — not
  authorized to modify recorded entries.

## Verification of existing entries (no changes made)

- **measured** `load_registry("registry/weak_signals.json")`: all **447**
  entries load clean under the full validator; classification census is 446 ×
  `unresolved_below_power` + 1 × `refuted_mechanism`; every closing ground in
  use is admissible (`wrong_sign_resolved`, with a fully-negative interval).
  Nothing reclassified.
- **measured** ad-hoc coherence scan over the JSON: one pre-existing entry,
  `recurrence_flags_player_brier_validation`, records effect 0.0433 outside
  its own interval [0.0176, 0.0303]. Per the no-reclassification constraint
  this row was NOT edited; it is now surfaced by tooling instead (below).

## Code changes

### 1. Per-family overlap warnings (`src/nfl_ats/weak_signals.py`)

- New `signal_family()`: explicit `family` field wins; otherwise inference
  from the name mirroring `findings_registry`'s duplication passes (**read**,
  `src/nfl_ats/findings_registry.py` lines 415–448) — strip `_opener`,
  `_era_YYYY_YYYY` / bare-year window splits, `_preYYYY`/`_postYYYY`; collapse
  names carrying `battery`/`microstructure` in their first three tokens to the
  battery prefix.
- New `family_overlap_warnings()`: groups signals into families per league and
  reports one structured row per family with internal window overlap (members,
  shared seasons, member names, warning string), plus cross-family
  shared-window pair counts and pairwise totals (within-family and overall).
  Reports; never blocks — same posture as the pairwise list it complements.
- `combination_report()` now emits `overlap_warnings` as the per-family
  structure plus `overlap_pairwise_count` (the old list's length) and
  `measurement_coherence_problems`.
- CLI (**measured**, `weak-signals pool --league nfl --effect-units
  accuracy_points`): the NFL accuracy-points pool shows **290 families, 17
  with internal overlap, 64,574 total pairwise pairs compressed into those
  rows**. `weak-signals status` gains `families`, `overlap_warnings`, and
  `measurement_coherence_problems` over its filtered signal set (**measured**:
  447 recorded, 344 families, 18 internally-overlapping, 68,966 pairwise).
  The pairwise `overlap_warnings()` function itself is unchanged for callers.

### 2. Validator gaps closed as code

- New `validate_coherence()` + `coherence_problems()`: a point estimate
  outside its own interval is a recording contradiction. Enforced **at record
  time only** (both `record_signal` paths), deliberately not at load time so
  the pre-existing entry above keeps loading unmodified (AGENTS.md forbids
  silently rewriting recorded measurements); report time surfaces it via
  `measurement_coherence_problems`.
- `validate_closure()` tightening (load AND record time; live ledger passes
  both, measured):
  - `positive_control_bound` must cite quantitative evidence (an interval or
    `probability_positive`) — a control bound IS a measurement.
  - `no_split_half_reliability` cannot cite reliability above
    `NO_SPLIT_HALF_RELIABILITY_MAX = 0.10` — a trait measured at e.g. 0.719
    (the CFB role-continuity traits) can never close on this ground.
- Optional `family` field added to the schema (`WeakSignal.family`,
  round-tripped through payload save/load; `weak-signals record --family`),
  so future batteries can declare the family before signs are seen, per the
  AGENTS.md commensurability rule.

## Tests

New regression tests in `tests/test_weak_signals.py` (32 → all passing):
family inference (grades/era/pre-post/battery/explicit), per-family warning
structure incl. disjoint-window and cross-league cases, combination-report
shape, record-time refusal of effect-outside-interval + load-time soft
reporting, bounded-by-control quantitative-evidence requirement, and the
reliability-ceiling rule. One existing test fixture (`bounded` entry in
`test_only_genuinely_unresolved_signals_are_poolable`) gained the
`probability_positive` the new validator requires — a fixture update required
by the tightened contract, not a weakened test.

## Quality gates (**measured** this session)

- `ruff format --check .` — pass (576 files)
- `ruff check .` — pass
- `mypy src` — pass ("no issues found in 102 source files")
- `pytest -q` (basetemp outside repo, per task instructions) — **1810 passed,
  5 skipped** (skips are environment-dependent data tests, pre-existing)

## Not done / deferred (with reasons)

- No reclassification, no entry edits, no re-measurement — per task
  constraints. The audit doc's supersede/annotate recommendations for the six
  duplicate-candidate body-clock rows remain owner decisions (**reported** by
  the audit doc; unverified by me).
- HANDOFF.md not refreshed on this branch: the pre-commit hook's documented
  escape hatch `NFL_ATS_SKIP_HANDOFF=1` was used because this commit is
  branch-local and never pushed to master directly; the merge agent owns the
  master handoff refresh.

## Files changed

- `src/nfl_ats/weak_signals.py` — per-family overlap machinery, coherence +
  closure validators, optional `family` field
- `src/nfl_ats/cli.py` — status/pool output fields, `record --family`
- `tests/test_weak_signals.py` — new regression tests + fixture update
- `docs/pool_edge_plan.md` — overlap_warnings description updated (2 spots)
- `ROADMAP.md` — RWB-18 row appended with this session's outcome
