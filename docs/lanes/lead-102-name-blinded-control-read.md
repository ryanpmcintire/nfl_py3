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

## Unit 1 log (2026-11-03)
Script scripts/lead102_blinded_read.py (subcommands blind / run --k / analyze --noise). Blinded packet + mapping: artifacts/lead99_llm_read/blinded/blinded_packet.json; reads: .../blinded/reads/. Anonymiser: Team A (away) / Team B (home), "Team X Player n", teams/cities/nicknames/abbrs/stadiums/other-team and other-player names replaced, urls/authors/sources redacted. Leak scan vs market team names, 2020+ roster names, stadiums, game-roster surnames: 0 leaks. Prompt hash 83800017... equals frozen. Caveat: partial first-name fragments in lowercase headlines (e.g. "vega") and unrostered coaches/staff are not scanned. 3 blinded reads launched by orchestrated background call (see task output); then run analyze --noise 0.052 (named = W9 run + 3 repeats).

## Unit 1 result (2026-11-03T21:15Z)
3 blinded reads stored (artifacts/lead99_llm_read/blinded/reads/); all valid. analyze --noise 0.052 (named 4 reads vs blinded 3, 15 games, logit): corr 0.856 boot95 [0.725, 0.960]; mean named-minus-blinded -0.009 boot95 [-0.034, +0.015], probability_positive 0.224; SD of game diffs 0.050 vs noise SE 0.040; 2 of 15 games |z|>2: NYG@PHI -0.118 (z -3.0), JAX@BAL +0.084 (z +2.1). Reputation component is small (inferred); look count 15 games, one family. Next: record via weak-signals record (unresolved_below_power), then fit as a separate term in the forward probability.
- 2026-11-14 15:17 local, registry catch-up (docs/lanes/registry-catchup-2026-11-14.md): recorded 1 cell, family lead102_name_blinded_control_read, correlation, unresolved_below_power (named v blinded r 0.856 [0.725, 0.960], 15 games; the analysis stored no probability_positive for it, so none is recorded). Not recorded: the named-minus-blinded mean gap -0.009 logit [-0.034, +0.015] P+ 0.224 and the SD of game gaps (a mean logit gap has no admissible effect unit; carried in the cell notes).
