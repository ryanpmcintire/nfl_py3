# Language-read analysis, stored reads only (2026-11-21)

## Goal
Score the stored LLM reads for LEAD-101, LEAD-104 and LEAD-107 without any new model call; record each look; leave ROADMAP lines for the orchestrator.

## State (all measured 2026-11-21 09:53-10:00 ET; every command exit 0)
- LEAD-101: `scripts/lead101_delta_read.py score` (scorer already existed; rerun unchanged). 54 rows, 29 games, W9-W11 Thursday. The read left the shown quote on 49 of 54 rows; its 5 departures went 1-4. Pooled gain vs no-change -0.057 [-0.144, +0.000] P+ .024; Thursday -0.105 [-0.267, +0.000] P+ .023; Sunday dead heat 0.000. 15 cells re-recorded, family lead101_news_delta_read. Lane lead-101-news-delta-read.md.
- LEAD-104: `scripts/lead104_revision_path.py --weeks 7 8 9 10 11 --first-view earliest tuesday`. 57 games W7-10; changes 15, last view 6-9; last vs first -0.0093 [-0.0315, +0.0117] P+ .198 (was -0.0200 P+ .058); Week 10 last 8-6. 58 looks; 18 cells re-recorded. Lane lead-104-read-revision-path.md.
- LEAD-107 unit 1: new scripts/lead107_dispersion_error.py. 15 W9 games; named rho -0.084 [-0.602, +0.461] P+ .377 (4 reads), blinded +0.034 [-0.560, +0.568] P+ .557 (3 reads); 4 looks; 2 cells recorded, family lead107_read_dispersion_weight. Lane lead-107-read-dispersion-weight.md.
- Every cell unresolved_below_power. `weak-signals pool --league nfl --effect-units accuracy_points` exit 0; shards lead101_news_delta_read.json, lead104_revision_path.json changed, lead107_read_dispersion_weight.json new.

## ROADMAP updates (orchestrator applies)
- ROADMAP LEAD-101 update: 2026-11-21 rescore W9-W11 Thu (54 rows, 29 games): read beside its shown quote, gain vs no-change -0.057 [-0.144, +0.000] P+ .024, departures 1-4, Sunday dead heat; 15 cells re-recorded, unresolved_below_power. Next: market-excluded prompt arm.
- ROADMAP LEAD-104 update: 2026-11-21 Weeks 7-10 (57 games): last vs first -0.0093 [-0.0315, +0.0117] P+ .198; first forward week (W10) last view 8-6; residual net of move a dead heat; 18 cells re-recorded, unresolved_below_power.
- ROADMAP LEAD-107 update: ⬜ to 🚧. 2026-11-21 unit 1 on 15 W9 games with 3-4 stored reads: dispersion vs error against the later line, named rho -0.084 [-0.602, +0.461] P+ .377, blinded +0.034 [-0.560, +0.568] P+ .557; 4 looks, 2 cells, unresolved_below_power. Powering it needs repeat reads of a later packet.

## Tried
No model call, no edit to src, no scheduler, no commit. Processes started: three background runs (all exited) and one grep stopped by TaskStop.

## Next
1. Tuesday 11-24: rerun LEAD-101 score and LEAD-104 with Week 11 settled; record with `--replace`.
2. Root decides on K repeat reads of a later packet for LEAD-107 (API spend), then rerun the script unchanged.

## Open
- Uncommitted: the three lanes, this lane, scripts/lead107_dispersion_error.py, artifacts and registry shards; the orchestrator commits.
