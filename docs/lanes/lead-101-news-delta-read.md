# LEAD-101

## Goal
News-delta read: an LLM sees only items timestamped after the Tuesday freeze and forecasts the direction and size of the move to the deadline line (ROADMAP LEAD-101).

## State
Added 2026-11-03 (backlog expansion 4), unmeasured, nothing run. Mechanism: LEAD-99 asks for a cover probability from everything, which mostly restates the line. The edge, if any, is in what arrived after the number froze (designations, beat posts, inactives, forecast revisions). Target is line move toward the side, the yardstick ENG-47 shows resolves effects 7-18x smaller than accuracy; it enters the one probability as a fitted term. Contamination: forward games only; no historical backtest is admissible.

## Tried
Novelty check only. Nearest: LEAD-99 (all information, cover target), SKY-01 (news-to-availability, moonshot), MKT-08 (news-triggered refresh, a trigger not a forecast), MKT-20 (move as target, no news input). Grep of ROADMAP.md, docs/lanes incl. done, docs/*.md and registry/weak_signals for a news-only read forecasting the move found none. Differs: information class and target.

## Next
First unit: Write the delta-packet builder from the stored Week 9 packet timestamps and freeze its prompt by hash; run the Thursday 11-05 and Sunday 11-08 views and write them to a forward log before kickoff; score move agreement and size after the deadline lines are captured.

## Open
Changes: Post-opener information inside the one probability. Data: LEAD-99 packet filtered to post-freeze timestamps; deadline-time pool and market lines. Delta-only prompt; target is signed move in points toward the pool-line side. Zero crossing closes nothing; record unresolved results with nfl-ats weak-signals record and a closing ground. Look count and family declared before any sign is seen.
