# LEAD-103

## Goal
Read-block ablation: rerun stored LEAD-99 packets leaving out one information block at a time to find which block moves the read (ROADMAP LEAD-103).

## State
Added 2026-11-03 (backlog expansion 4), unmeasured, nothing run. Mechanism: If the read moves with beat posts and injury lines but not with weather or rest, the moving block is the candidate for a structured, cheaper, backtestable term in the model. Blocks: beat posts, injury lines, weather, market move, rest and travel. Contamination: forward games only; no historical backtest is admissible.

## Tried
Novelty check only. Nearest: LEAD-99 (full packet only). Grep for ablation and leave-one-block-out hit only unrelated lanes (mod25e seed bootstrap, line-move regrade). Differs: attribution inside the read, which the scoring plan does not give.

## Next
First unit: On the Week 9 deadline packet run five leave-one-block-out reads under the frozen prompt; report per block the mean absolute logit change from the full read and whether the sign agrees with the block's own implied direction.

## Open
Changes: Turns a forward read into a structured term candidate. Data: Stored packets under artifacts/lead99_llm_read/runs/. Five leave-one-block-out runs per scored week; attribution table. Zero crossing closes nothing; record unresolved results with nfl-ats weak-signals record and a closing ground. Look count and family declared before any sign is seen.
