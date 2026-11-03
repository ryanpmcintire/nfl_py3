# ENG-57

## Goal
Weak-signal registry base tag: mark which recorded cells were graded on the raw model probability (ROADMAP ENG-57).

## State
Added 2026-11-03 (backlog expansion 8), not started, nothing run. Mechanism: Cells in registry/weak_signals do not say which base they were graded on, so a pooled summary mixes raw-base and served-base results. The error is a missing field, not a number. Fitted term: none; the tag lets nfl-ats weak-signals pool separate bases before any effect is pooled.

## Tried
Novelty check only. Nearest: ENG-51 (forward guard on new harnesses), ENG-54, weak-signals pool command. Differs: labels the past recorded cells and splits the pooled summary; ENG-51 only guards new harnesses.

## Next
First unit for one agent: Read the scripts behind this week's recorded families, tag each family raw, served or market-only in a side table, and report the pooled accuracy-point summary split by base without changing any stored cell or validator.

## Open
Rules: Never weaken the record validator; no stored value edited; the tag is a side table. Look count and family declared before any sign is seen.

## Unit 1 result (2026-11-03T22:55Z)
Side table: registry/weak_signals_base_tags.csv (973 rows, one per signal recorded 2026-11-03 or later; columns signal, family, effect_units, base, evidence, confidence read/inferred). Built by scripts/eng57_base_tags.py (family rule table, evidence = script:line). Outside registry/weak_signals/ so shard loading and `weak-signals status` are unaffected (status exit 0). Tags: raw_model 385, served_card 88, cfb_base 190, not_applicable 310 (correlation/epa/rate/payout units or no pick-probability base). 592 read, 381 inferred (family-level, not cell-level; cell-level exceptions such as mod41_u2 first cell, mod23 later units on the harness refit, and cells with a market-only or intercept comparator are not split out).
Pooled summary (`pool --league nfl --full`, saved artifacts/eng57/pool_*.json): the pool emits ONE pooled cell per effect unit, and each mixes bases. log_loss_improvement: 877 eligible = 591 recorded before 11-03 (untagged), 201 raw, 59 served, 11 cfb, 15 n/a. accuracy_points: 4365 eligible = 4206 untagged, 115 raw, 9 served, 31 cfb, 4 n/a. So 2 of 2 pooled cells mix bases; raw and served cells are re-tests of the same ideas, so they are NOT independent and the pooled estimate double counts them (overlap).
Reading: treat the pooled log_loss and accuracy numbers as a mixed-base descriptive summary only; cfb_base cells are a different population (commensurability rule). Prefer the served_card subset for any decision about the card.

- 2026-11-03 17:49 ET (root decision): the pool --base filter and per-base subtotals are a change to the non-pinned weak-signals CLI (PINNED_NAMES in src/nfl_ats/independent_validation.py:40 does not include it); queued as ENG-57 unit 2, no approval step needed. Until then, card decisions read only served_card rows of registry/weak_signals_base_tags.csv.

## Unit 2 result (2026-11-03 17:53:48)
Added `weak-signals pool --base` (raw_model|served_card|cfb_base|not_applicable|untagged) and by_base subtotals (signals, effective_signals, pooled_effect, interval, probability_positive) in cli_commands/registry.py and weak_signals.py (load_base_tags, restrict_to_base). Default output diff vs before: only generated_at_utc and the added by_base block. tests/test_weak_signals.py 64 passed.
Next: none queued; subtotals are descriptive, overlapping re-tests still double count.
