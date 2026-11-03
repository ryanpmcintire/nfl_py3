# UI-21 Pool outlook panel

## Goal
Show readers the pool standing and chance of finishing first in plain words. ROADMAP row UI-21.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: POOL-01 built a field model and standings simulation but readers never see it. A plain panel shows rank, points behind the leader, weeks left and a range for the chance of finishing first, from the forward sim. Data: data/splash/field standings, POOL-01 simulation artifacts.

## Tried
Novelty check only (grep of ROADMAP, docs/lanes incl. done, docs/*.md, registry/weak_signals). Nearest prior: POOL-01 and POL-05 (analysis only), UI-20 queue items a-h (record strips). No page shows pool standing or title odds.

## Next
Locate the POOL-01 simulation artifact and render the panel in the current design system. First unit: First unit: add the panel to the Home page from the existing simulation artifact keyed to the active model, publish-board, review the rendered diff.
Research rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Chance is a model estimate; show a range, never a point as a promise.

## Update 2026-11-03
Built: PoolOutlook dataclass and _load_pool_outlook in board_content.py, BoardContent.pool_outlook, _pool_outlook_section in board_terminal.py (Home page, after headline). Sources: artifacts/pool_rank_replay/forward_sim_real_rivals_results.json (no model key; panel states the week it reflects and warns when stale), weekly_points_field.json, data/splash/field/2026_weekly_points_w01-08_raw.txt (leader 75 of 241 captured). Board tests 338 passed. Next: root runs publish-board and reviews the rendered diff.
