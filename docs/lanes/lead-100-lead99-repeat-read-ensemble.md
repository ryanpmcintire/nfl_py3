# LEAD-100

## Goal
LEAD-99 repeat-read ensemble: K independent frozen-prompt reads per game, mean log-odds as the term and between-read spread as the per-game weight (ROADMAP LEAD-100).

## State
Added 2026-11-03 (backlog expansion 4), unmeasured, nothing run. Mechanism: A single sampled read carries sampling noise (stored Week 7-9 reads sit in 0.44-0.56); averaging cuts that noise with no new data, and games where reads disagree are the ones where the narrative is unresolved, so the spread enters as a fitted interaction, not a rule. Contamination: forward games only; no historical backtest is admissible.

## Tried
Novelty check only. Nearest: LEAD-99 and lane lead99-llm-read.md (one read per view). Grep for self-consistency, ensemble reads and read variance found nothing in ROADMAP.md, docs/lanes incl. done, docs/*.md or registry/weak_signals. Differs: tests the reliability of the read itself and what its disagreement means.

## Next
First unit: Re-run the stored Week 9 Tuesday packet three more times under the frozen prompt hash; report per-game between-read SD, split-half reliability (read 1 v mean of reads 2-3) over the 15 games, and the SD across the existing Week 7-9 views for context. No outcomes read.

## Open
Changes: Information the opener cannot hold, used with less noise. Data: scripts/lead99_llm_read.py packets and scripts/lead99_prompt.md (frozen hash). Repeat-read ensemble over the LEAD-99 packet; same prompt hash, K reads per game. Zero crossing closes nothing; record unresolved results with nfl-ats weak-signals record and a closing ground. Look count and family declared before any sign is seen.

## Unit 1 log (2026-11-03 3:55 PM ET)
Prompt sha 83800017... equals frozen.json and the stored W9 run. Runner cmd_run rebuilds the packet at call time, so unit 1 uses scripts/lead100_repeat_reads.py (stored packet, packet sha verified, same call_claude). Reads land in artifacts/lead99_llm_read/repeats/ (no overwrite). Analysis: scripts/lead100_analyze.py. Launched 3 sequential reads, python PID 33464.
Done 2026-11-03 3:59 PM ET (measured): 3 valid repeats in artifacts/lead99_llm_read/repeats/ (all rows valid, no retries logged). 4 reads x 15 games, logit scale: mean between-read SD 0.052 (prob 0.013, max 0.059 prob; largest NYG_PHI 0.236, GB_NE 0.172, DAL_IND 0). Between-game SD of mean logit 0.093. Split-half r (read1 v mean reads2-4) 0.237, bootstrap 95% [-0.39, 0.96] (10000 resamples, seed 20261103), P(r>0)=0.77; Spearman-Brown 0.38. Across-view SD, 27 games with >1 stored view (W7-9 mixed tuesday/deadline), mean 0.078 logit. Unresolved below power (n=15 games); not recorded via weak-signals yet (declared single look). No process left running.
## Next
Unit 2: K reads on Thursday deadline packet or later weeks to grow n; decide record of reliability look via weak-signals with effect-units after n>=45 games.
- 2026-11-14 15:17 local, registry catch-up (docs/lanes/registry-catchup-2026-11-14.md): recorded 1 cell, family lead100_repeat_read_ensemble, correlation, unresolved_below_power (split-half r 0.237 [-0.39, 0.96], P+ 0.77, 15 games). Not recorded: between-read SD 0.052, between-game SD 0.093 and across-view SD 0.078 logit (dispersion statistics, no effect unit); Spearman-Brown 0.38 is carried in the cell description.
