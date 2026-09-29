# Site auto-publish and pool board order (2026-09-29)

## Goal
The public site updates on its own after results settle and after the Tuesday
lock, and lists games in the pool board's order.

## State
Done. `scripts/publish_site.py` runs publish-board, stages only the site files,
commits through the normal hooks (HANDOFF refresh lands in the commit) and
pushes; it skips if unrelated paths are staged. Jobs `publish_site_fri_0030`,
`publish_site_mon_0030`, `publish_site_tue_0030`, `publish_site_tue_1430`.
`--run-job publish_site_tue_1430` MANUAL-RUN OK, pushed c44370a. Daemon
restarted, pid 45024. The lock's weekly-run timeout is now 3300 s (was 1800).
The board orders games by the Splash capture's order (`_pool_board_order` in
`board_content.py`), falling back to kickoff order.

## Tried
- Splash API (`api.splashsports.com/contests-service-v2`) returns CloudFront
  403 to non-browser clients; the board cannot be fetched headlessly without
  the owner's session. The Tuesday board read stays an agent Chrome step.

## Next
- None. Confirm Friday 00:30 run in `data/scheduler_log.txt` next session.

## Open
- None.
