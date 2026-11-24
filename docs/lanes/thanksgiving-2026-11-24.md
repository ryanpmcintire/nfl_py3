# thanksgiving-2026-11-24

## Goal
Week 12 lock plus the owner's Thanksgiving side pot ($500: all three Thanksgiving picks right AND closest combined final score of CHI at DET, PHI at DAL, KC at BUF). Done = Week 12 card locked on the noon pool lines, the three Thanksgiving sides audited game by game, a combined-score guess built from the freshest totals, and a final Thursday-morning number in front of the owner before the 1:00 PM ET kickoff.

## State (Tue 2026-11-24)
- Splash board read 08:35 ET (view only): 16 games, Wed GB at LA 8:00 PM, Thu CHI at DET 1:00 (CHI -2.5), PHI at DAL 4:30 (DAL -8.5), KC at BUF 8:20 (BUF -2.5), Fri DEN at PIT 3:00. Spreads lock Tue 12:00 PM ET, so capture after noon. Splash's own tiebreaker is CAR at TB (Mon); the Thanksgiving total is not on the Splash sheet. Entry: rank 72 of 250, 83 points after Week 11.
- Measured (2009-25, 4,427 REG games, scratchpad tg_hist.py): total shift fav-covers minus dog-covers +0.05 pts [-0.55, +0.60] P+ 0.59; Thanksgiving games n=54 mean total minus line 0.0, median +1.0. No Thanksgiving term.
- Measured (scratchpad tg_sum.py, kernel spread +-3 / total +-6 residuals, conditioned on our side covering, 200k draws): with Monday Bovada totals 55.5 / 47.5 / 50.0 (served 55.35 / 47.58 / 49.61) the combined-score median is 150.5-155 across the 8 side combinations, unconditional 153; 50% range about +-15.
- Scheduler slots for Thanksgiving exist: inactives 11:35 / 15:05 / 18:50, refreshes 11:55 / 15:25 / 19:15, odds_private 11:40 / 14:50 / 19:00.
- Built (uncommitted): scripts/thanksgiving_total.py (production tiebreaker neighborhoods, residuals around the served total, conditioned on the card's sides, exact convolution; --only-on date guard; --out json). 08:30 run: guess 152, middle half 137-168 (sides unknown, Week 11 card). Scheduler jobs thanksgiving_total_wed_1840 and thanksgiving_total_thu_1210 (write artifacts/thanksgiving_total/2026_week12.json; notify_after_job pushes "Thanksgiving score guess"); --run-job thu_1210 exits skipped today. Board: board_content.py `_load_side_pot` + fields side_pot_text/side_pot_guess; board_terminal.py line under the tiebreaker panel and in Copy picks. ruff and mypy clean. Live only after a scheduler restart.
- Pool standing After Week 11 (measured): rank 72 of 250, 83 points, leader 96, 70 above, 18 on 83; all 11 page column sums match. Files 2026_weekly_points_w01-11_raw.txt, 2026_week03-11_pick_distribution_raw.txt; pool01_field_best_pick and pool01_weekly_points --points 83 ran; sims (--weeks 7 --points 83 --rank 72), realised-cover sim, pool04 backtest all exit 0; check_pool_standing exit 0 (files on Week 11). First place 2.0% at the stated chance (55.0%), 0.1% at the realised rate (82 of 163). Panel goes live at the next publish-board.
- Field shares (Weeks 3-11, 131 games, measured): 58% of entries on the favourite; the larger side averages 63%. Contrarian side-pot value only matters near a coin flip.
- News (thanksgiving-news-2026-11-24.md): PHI on Joe Milton III (Hurts out for season), DAL opened -1.5 now -8.5; CHI -2.5 flipped from DET -2.5 in May; Buffalo Thursday night rain 0.5-0.75 in, gusts 26 (NWS, reported).
- Goal-55 Tuesday steps landed in the working tree (agent, verified by --status count 16 with the thanksgiving jobs): inactives_capture.py U1b patch, inactives_sun_early_second grace 40, inactives_thu_primetime_second Thu 19:35, 13 g55_forward_* jobs (detail in g55-u7 lane). Scheduler pytest 57 passed (test_capture_scheduler.py). Uncommitted; daemon not restarted. Restart once after the weekly_lock finishes (check the process list first, then confirm the job count in the log).

- 12:01 ET: board re-read after the spread lock, all 16 lines unchanged; captured data/splash/2026_week12_20261124_noon.json (16 games), check_splash_board exit 0. splash_board_tue 12:05 and weekly_lock 12:20 left to the daemon.

- 13:07 ET weekly_lock OK (16 games, recorded_and_verified, artifacts/scheduled_locks/2026-week-12). Thanksgiving card (card_terms 20261124T180844925368Z): CHI -2.5 54.5% (model alone CHI 62.6% raw, no flags); BUF -2.5 59.3% (model 56.9% BUF, +1 flag toward BUF, hook +0.077 at 2.5 v market 3); PHI +8.5 54.6% (model alone DAL 55.6%, margin DAL 12.3; division-revenge flag carries it to PHI). Product of the three 17.6%.
- thanksgiving_total with Week 12 sides (noon Bovada totals 55.5 / 48 / 47): guess 149, middle half 134-167; artifacts/thanksgiving_total/2026_week12.json; board shows it.
- Side-pot contrarian check (measured field shares): flipping CHI to DET leaves side-pot EV about equal ($2.75 v $2.90) and costs 0.09 main-pool points; keep the card.
- Scheduler restarted 13:29 (one daemon, 466 enabled jobs). thanksgiving_total_wed_1840 and _thu_1210 MANUAL-RUN OK (skipped by date).
- Fixed reader text: card_explanation.py said "taken together its inputs land on PHI" when the model leaned DAL and served terms carried the pick; now "the adjustments that follow carry the pick to ...". publish-predictions + publish-board rerun in background to rewrite explanations.json.

## Tried
- Neighborhood median from tiebreaker.py drifts low for high totals (CHI at DET 55.5 fell back to all history, median 44); residuals around the served total used instead.

## Next
1. Wed 18:40 and Thu 12:10 ET: thanksgiving_total_* jobs push "Thanksgiving score guess" to the owner; check data/scheduler_log.txt for OK + NOTIFY lines. If a job failed, run `scripts/thanksgiving_total.py --games 2026_12_CHI_DET,2026_12_PHI_DAL,2026_12_KC_BUF --out artifacts/thanksgiving_total/2026_week12.json` and tell the owner the number before 1:00 PM ET Thursday.
2. Watch PICK-CHANGE lines for the three games from refresh_wed (18:15), refresh_thu_inactives_early (11:55), _late (15:25), _primetime (19:15).
3. After Thursday: move this lane to done/.

## Open
- None.
