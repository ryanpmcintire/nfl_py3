# mod27-u2-move-term-2026

## Goal
Unit 2 of mod27-better-picks: clean 2026 move term vs served, and card vs pool-relative move tally, Weeks 1-8.

## State
- Script and per-game table: scratchpad agentU2/run.py, games.csv (121 games, 120 graded; PHI@WAS Mon unplayed).
- Served M reconstructed per last pre-deadline feed: old code (no baseline filter) + live-only quotes before 10-31 08:07 ET; new code + live-only to 16:49 ET; new + reparse after. Clean = new code + all quotes, now = pick deadline. Z_clean = logit(feed final_probability) + 0.221*dM + 0.0168*dAvail (active coefs 20261101T171318Z). Grading at Splash pool line (matches decision_home_spread 121/121).
- A (measured): move differs on 44/121 games (mean |M| 0.35 served, 0.51 clean) but changes the four-term side in 3: TB@CIN W1 (HOME->AWAY, M 0 -> -0.5), ATL@TB W8 (HOME->AWAY), LV@NYJ W8 (AWAY->HOME). Served right in all 3 (home covered; ATL won by 16 as dog covers... see games.csv). Four-term served 68-52, clean 65-55, card 64-56. Recorded move_term_clean_vs_served_2026_w1_8 (-2.5 pts, P+ 0.06, unresolved_below_power).
- B (measured): pool-relative move = leader median at deadline minus pool home_spread. Moved 86; card with 67 / against 19. Graded with 38-28, against 6-13, unmoved 20-15. W8 alone: moved 9, all with (4-4). Recorded card_against_pool_move_tally_2026_w1_8.
- No Week 8 forward log of the pool-relative challenger exists (no file; romana lane Unit 2 only planned). Inferred stand-in (challenger move coef 0.145 from romana lane, replaces served M): W8 6-7 vs card 5-8 (1 differs); W1-8 71-49 vs card 64-56 (13 differ). Not a served-path result.

## Tried
- Card vs feed side differ in 8 games (late revisions after the last feed), so feed-based sides are not the card.

## Next
- Build the real forward log (pool line vs leader median per refresh) if the owner wants it scored after Week 18.
- Tuesday 11-03 odds_private_tue is first live check of the parser fix.

## Open
- Feed final_probability used then-active coefficients, not the 20261101 set; backing out is approximate.
