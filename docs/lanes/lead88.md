# LEAD-88 unit 3 - total response with identity at zero

## Goal
Isolate total-news response from lattice reprojection; research only.

## State
**Measured:** one declared replay complete, 2 candidate-versus-served looks; 1,333/1,343 games and 101 last games scored.
Zero-move changes: 0; zero-coefficient identity: 1,333/1,333 team-score pairs.
last: MAE improvement 0.099010 [-0.019802, 0.217822]; probability_positive=0.952550.
all: MAE improvement 0.053263 [0.009901, 0.097524]; probability_positive=0.992300.
Proposed unresolved_below_power; no registry writes. Report: docs/lead88_unit3.md.

## Protocol (frozen before outcomes)
Original declaration saved before scoring in tests/scratch/codex/lead88_unit3/protocol.md and copied verbatim into the report.
Protocol SHA256: 001b5dc1868c7639aa7d4c84624e10a769e5cc112ec2fb14f865f302b7cf47c0.
Same 2020-2025 population, opener proxy, four-term side, LAD LOSO response; adjust continuous centre only across a team-score lattice cell.
Two looks: last/all MAE; week-block intervals, exact closer null, fold slopes and IS/OOS gap. No outcome-driven revisions.

## Tried
**Measured:** .tools/uv.exe run --no-sync python scripts/lead88_unit3.py; local UV_CACHE_DIR; one job, at most two compute threads.
Replay guards passed; source hashes, clocks, zero identity, retained rows, same response slopes and fixed side checked.
**Measured:** .tools/uv.exe run --no-sync ruff check scripts/lead88_unit3.py and ruff format --check scripts/lead88_unit3.py both passed (exit 0).

## Record commands
Orchestrator only; serial bash commands, candidate versus served.
```bash
.tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record --name lead88_unit3_last_mae --family lead88_total_response --description 'Changing totals versus the current score guess on the final game of the week; 2 looks' --source docs/lead88_unit3.md --league nfl --season-start 2020 --season-end 2025 --effect 0.0990099009901 --effect-units mae_improvement --interval-low -0.019801980198 --interval-high 0.217821782178 --standard-error 0.0600535346539 --probability-positive 0.95255 --sample-games 101 --sample-blocks 101 --classification unresolved_below_power --classification-evidence 'Two declared looks; no refuted mechanism or powered control; pending review' --plain-summary 'For the final game of the week, the score guess follows changes in sportsbook totals only far enough to change a whole score cell. No change in the total leaves the current guess alone. These results do not settle whether this helps; keep the current guess.'
.tools/uv.exe run --no-sync --no-cache nfl-ats weak-signals record --name lead88_unit3_all_mae --family lead88_total_response --description 'Changing totals versus the current score guess on all scored games; 2 looks' --source docs/lead88_unit3.md --league nfl --season-start 2020 --season-end 2025 --effect 0.053263315829 --effect-units mae_improvement --interval-low 0.00990080172401 --interval-high 0.0975243810953 --standard-error 0.0222866473083 --probability-positive 0.9923 --sample-games 1333 --sample-blocks 107 --classification unresolved_below_power --classification-evidence 'Two declared looks; no refuted mechanism or powered control; pending review' --plain-summary 'For all scored games, the score guess follows changes in sportsbook totals only far enough to change a whole score cell. No change in the total leaves the current guess alone. These results do not settle whether this helps; keep the current guess.'
```

## Next
Orchestrator reviews the report and runs the two record commands serially; any further study needs a new declaration.

## Open
Retrospective feature-vintage and power limits remain. Zero crossing closes nothing; one fitted probability selects the side.
No serving change, publication, tests, commit or push. Scratch rows and summary: tests/scratch/codex/lead88_unit3/.
