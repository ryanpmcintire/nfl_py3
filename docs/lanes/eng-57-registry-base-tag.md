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

## Unit 3 refresh (2026-11-14 18:07, date)
Measured: scripts/eng57_base_tags.py extended (NEW_RULES family evidence from script greps of the two probability columns, NOT_APPLICABLE_UNITS, audit correction tables RAW_ONLY_SHARDS and RAW_ONLY_FAMILIES from artifacts/eng54/label_audit_unit2.json, note column, new value unknown for unclear evidence; shard mkt28_served_base_retest.json is excluded from correction because it is the served retest). Rerun regenerates registry/weak_signals_base_tags.csv deterministically (cutoff 2026-11-03). Rows 973 to 2164 (1191 had no row: all 11-14 cells plus 11-03/11-08/11-11 families the first build missed). Tags before to after: raw_model 385 to 693, served_card 88 to 334, cfb_base 190 to 250, not_applicable 310 to 557, unknown 0 to 330. Existing rows changed by evidence: mod25_distillation_walkforward 5 raw to served, mod39_playoff_leverage 4 raw to served (freeze scripts grade the served column), lead106 2 n/a to unknown, lead104_revision_path 1 n/a to raw. Audit correction served to raw: 0 cells (every cell whose family cites a verified or raw-only audit script was already raw in the table; the note column is empty). Mixed scripts reading both columns (lead71_unit4, lead88_unit2, mod47_unit2_global_calibration, mod39_freeze) left as tagged.
Unknown (read the reason in evidence): lead106 20 (forward_score names no probability column), lead99_forward_decision_read 9 (reads recommendations raw column but cells are vs card), mod36_metric_power 8, opener_error_transfer_v1-v5 17 and xlg09_unit6 4 (own fitted base), xlg_harness_power 18, and 11-03..11-11 families with no script read yet: mod57 (11 families, about 160), mod07_joint_stack, mod53, mod54_mirrored_rows, lead111, per16_team_designation_habit, pol13, fix_picks, mod55, mod56, sim25, sim28, sim29. Unknown rows drop out of every by_base subtotal (the CLI accepts only the four values); they stay in the unfiltered pool.
Consumer run: nfl-ats weak-signals pool --league nfl --effect-units accuracy_points --base raw_model loaded the new CSV (eligible 207). Next: read scripts for the unknown families above and add NEW_RULES rows.
