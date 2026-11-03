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
