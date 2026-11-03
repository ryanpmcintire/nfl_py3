# UI-24 Luck band

## Goal
Luck band: how far the standings sit from what pure coin flips and the market favourite would give ROADMAP row UI-24.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: POOL-02 found entrant spread equals what luck gives, yet the Home page shows rank without a yardstick. A band showing the points range that coin flippers and market followers reach by the same week lets the reader see whether our place is skill or noise. Data: data/splash/field standings and picks, the field simulator, the active-model artifact.

## Tried
Novelty check only: Nearest: UI-21 (forward chance of first), POOL-02 (the spread finding, not shown anywhere), UI-20 History strips. Differs: backward yardstick against luck; grep for luck band returned only the POOL-02 lane.

## Next
First unit: Compute the band per week from the observed field for coin flippers and market followers, add a plain-English line and chart on Home, publish-board, review the rendered diff.
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Look count and family to be declared before any sign is seen. Nothing here closes on zero crossing; record unresolved results with nfl-ats weak-signals record.

## Update 2026-11-03 (unit 1 shipped locally, not published)
Luck band added to the Home pool panel: exact binomial 5th-95th percentile over the games played (121 games: 51 to 70 wins), field 5th-95th from the captured weekly totals (50 to 66), our 65 as a marker. Code: _binomial_quantile and _load_pool_outlook in board_content.py, _luck_band_html in board_terminal.py. Rendered to tests/scratch/ui-audit-2026-11-03/site5 and checked at 390 and 1440. Next: orchestrator publishes and pushes; no weak-signals record needed (display only).
