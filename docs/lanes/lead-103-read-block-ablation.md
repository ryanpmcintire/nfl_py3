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

## Unit 1 log (2026-11-03 4:10 PM ET)
Script scripts/lead103_ablation.py (run / analyze). Blocks present in packet: market, injuries, injury_headlines, beat_posts, weather, rest (inactives empty in all 15 games, skipped). Block removed = key deleted from every game; source stored W9 Tuesday packet (hash verified), frozen prompt hash verified. Launched 6 sequential reads, python PID 19392 (launcher 29408), outputs in artifacts/lead99_llm_read/ablations/. Analyze: scripts/lead103_ablation.py analyze.
Done 2026-11-03 4:14 PM ET (measured): 6 valid reads stored in ablations/. mean|d| logit vs mean of 4 full reads (ratio to 0.052; 95% bootstrap over 15 games): market 0.095 (1.82 [1.04,2.71]); beat_posts 0.072 (1.39 [0.98,1.81]); injuries 0.067 (1.29 [0.69,2.03]); injury_headlines 0.042 (0.80); weather 0.042 (0.80); rest 0.035 (0.67). Caveat: pure read noise alone gives expected |d| near 0.05 (ratio ~1), so ratio 1 is the null floor; only market clearly exceeds it. One read per block, so each shift includes one read of noise. Look count 6, single family. Not recorded in weak-signals. No process left running.
Next: repeat each block with K>=3 reads (or the market and beat_posts blocks) to separate shift from noise; ablate Thursday deadline packet.
