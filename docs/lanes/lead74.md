# LEAD-74 — within-spread price revisions

## Goal
Measure same-book price pressure against the served four-term probability.

## State
**Measured:** unit 2 complete; 528 paired games, 361 OOS games (2024-2025),
36 week blocks, 6 fits, 173 reserved looks. Decisive record 8-5;
61.54% [33.33, 91.67], probability_positive=0.7967 versus 50%.
**Inferred:** unresolved_below_power; orchestrator registration pending.

## Unit-2 amendment (saved before outcomes)
The declaration is preserved verbatim in `docs/lead74_unit2.md` under Frozen
declaration and hashed in `tests/scratch/codex/lead74_unit2/results.json`.
Historical opener proxy; standard 2020-2025 population restricted to 2023-2025
quote coverage; provider observed_at_utc assumed available, so findings are conditional.
One median same-book unchanged-spread price-logit change jointly fitted with the
four base terms; earlier-season-only folds; no later-season training or side flip.
B reduced 2 to 1; F=3, E=0; 10,000 paired week-block draws; no protocol revision after outcomes.

## Tried
**Measured:** `UV_NO_CACHE=1 .tools/uv.exe run --no-sync python scripts/lead74_unit2.py`
ran once, exit 0; Ruff check/format passed; saved arithmetic, declaration and quote
cutoffs verified. Accuracy 53.19% [49.31, 56.76]; log loss 0.7023 [0.6884, 0.7164];
Brier 0.2540 [0.2473, 0.2608]. Base-relative gains: accuracy +0.8310
[-1.3124, 2.8490], probability_positive=0.7967; log loss -0.001725
[-0.006414, 0.002595], probability_positive=0.2294; Brier -0.000711
[-0.002926, 0.001312], probability_positive=0.2614. All intervals are 95%.
IS/OOS accuracy gap -2.7701 [-6.8966, 1.3736] points; pressure slopes
2024 +2.2869, 2025 -0.2371. Full coefficients and reliability: `docs/lead74_unit2.md`.

## Record commands
Orchestrator only; NOT run here. Batch contains nine correlated metric records.

```bash
UV_NO_CACHE=1 .tools/uv.exe run \
  --no-sync nfl-ats weak-signals record \
  --batch tests/scratch/codex/lead74_unit2/record_payload.json \
  --source docs/lead74_unit2.md \
  --plain-summary 'Books sometimes change prices without changing the spread. Keep the current picks while we check whether those price changes help.'
```

## Next
Orchestrator: review the conditional result and run the record batch serially.
A further source-coverage unit needs a separate declaration.

## Open
Availability remains assumed; 2023 lacks earlier pressure training; cached Elo
is absent. Two OOS seasons limit stability. No research closure or serving change.

**Recorded 2026-09-29 (root):** only the three pressure-vs-four_term cells (baseline contrasts are not signals), unresolved_below_power.
