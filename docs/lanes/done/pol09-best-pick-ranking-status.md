# POL-09 Best Pick ranking status

## Goal

Verify the remaining development acceptance gap for ranking Best Pick by served conviction.

## State

Complete as implementation work. Research efficacy remains unresolved.

## Tried

- **Read:** `src/nfl_ats/best_pick_nomination.py:158` derives ranking conviction from the served calibrated probability; the nomination path at line 278 applies that score with a frozen opener fallback when unavailable. Ranking does not independently change the served side.
- **Read:** `docs/best_pick_served_score_ranker.md:108` documents shipped A1 behavior and the distinction between consistency and efficacy; lines 185-192 preserve unresolved registry results.
- **Measured:** The locked offline check of `artifacts/best_pick_served_score_ranker/20260914T163425Z` found 107 weekly rows, A1 64/103, 101 paired weeks and 37 different nominations. Historical accuracy difference was +5.9406 points [-1.9802, +13.8614], `probability_positive=0.933575`.
- Corrected only the ROADMAP development status; no prediction code, serving choice, or research verdict changed.

## Next

Continue prospective measurement under the existing protocol.

## Open

The historical comparison is not held-out evidence of a stable gain. AGENTS.md requires an admissible closing ground; this status correction closes no signal.
