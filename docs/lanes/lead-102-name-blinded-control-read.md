# LEAD-102

## Goal
Name-blinded control read: the LEAD-99 packet with team and player names replaced by role labels, so the gap to the named read is the reputation component (ROADMAP LEAD-102).

## State
Added 2026-11-03 (backlog expansion 4), unmeasured, nothing run. Mechanism: A named read blends team reputation, which the opener prices, with packet facts, which may not be priced. A blinded read (teams as A and B, players as position and numeric rating, rest, travel, weather and injury facts kept) isolates the part driven by facts. Both enter the fitted probability as separate terms; the difference is a diagnostic, never a rule. Contamination: forward games only; no historical backtest is admissible.

## Tried
Novelty check only. Nearest: LEAD-99 (blind only to the served model's probability). Grep for anonymised or name-blinded reads found only model-blinding statements in lead99-llm-read.md and unrelated uses of blind in the lead90-97 lanes. Differs: removes team identity to measure reputation.

## Next
First unit: Build the anonymiser for the Week 9 packet, check by diff that no name or city survives, and run named and blinded reads for the Thursday 11-05 deadline view; log both before kickoff.

## Open
Changes: Control that locates where any LEAD-99 signal lives. Data: LEAD-99 packets; rating and injury fields already in the packet. Role-label anonymiser plus paired named/blinded runs. Zero crossing closes nothing; record unresolved results with nfl-ats weak-signals record and a closing ground. Look count and family declared before any sign is seen.
