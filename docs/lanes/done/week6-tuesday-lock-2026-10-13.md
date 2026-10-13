# Week 6 Tuesday lock (2026-10-13)

## Goal
Capture the Week 6 pool board, run the weekly lock, publish card and site with one visible dashboard improvement, push.

## State
- Done. Pool board read in Chrome 18:06 ET: `data/splash/2026_week06_20261013_1806.json`, 14 games, picks lock Sun Oct 18 4:00 PM ET.
- `weekly_lock` manual run: recorded_and_verified, 14 games, model `403b07a3c70aeef1`; Best Pick HOU +2.5 (57.3%).
- Dashboard: Books now read "No quote" on every Tuesday board because the first mid-week odds capture was Wednesday 18:00. Added `odds_private_tue` (Tue 14:25) to `scripts/capture_scheduler.py`, ran it through the daemon argv, restarted the daemon (233 jobs), republished; Books now shows a line for all 14 games.

## Tried
- The scheduled 12:20 lock MISSED because the headless Splash check 403s; the agent read the board in Chrome as usual.

## Next
- Wednesday 12:15 refresh runs on its own.

## Open
- CHI at ATL: book line ATL +3 already sits at the card's flip line (pool line +3.5 still grades the pick).
