# Lane: SKY-01 news availability

## Goal
Timestamped report-to-status table; lead time of each source over official designation and line moves; test earliest-source signal as one fitted term on the four-term probability.

## State
Done 2026-11-02: scripts/sky01_news_availability.py, docs/sky01_news_availability.md, artifacts/sky01/. 6,765 rows (2024-2026). Beat leads official by median 45.6 h; beat to adverse line drift 37.4 h. Term OOS Brier gain -0.000178, p+ 0.3345, 4 decisive (2-2). 9 cells recorded, family sky01_news_availability, unresolved_below_power.

## Tried
Beat pre-opener out-count differential, LOSO 2024/2025 only. Inactives captures: 2026 weeks 3-8 only, not joined to line moves.

## Next
Score the term on 2026 once four-term opener probabilities exist for 2026 (Week 9 onward). Control line drift (compare moves after signal vs matched games without signal). Weight players by prior snap share. Join inactives first-seen (2026) and snapshot first-flag to beat table.

## Open
ROADMAP SKY-01 row not updated (orchestrator owns). Market lead is unconditional drift, no control.
