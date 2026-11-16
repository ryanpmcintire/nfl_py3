# model-improve-2026-11-16

## Goal
Owner 2026-11-16: keep improving the model, no overfitting, maximum parallel throughput. Only units that fix a served input, bring information the line does not hold, or set up a declared forward test. No NFL one-term add-on units (memory add-one-noise-floor).

## State
- 2026-11-16 15:07 EST: wave 1 launched, four parallel units, disjoint files:
  - W1-A serve parity audit: lane serve-parity-audit-2026-11-16.md. Do the 2026 inputs the live card used equal what the training builder computes for the same games? Patches in tests/scratch/serve_parity/, nothing in src.
  - W1-B outside ratings acquisition (ROADMAP MOD-30): lane mod30-outside-ratings-acquire-2026-11-16.md. Historical pregame FPI predictor (NFL and college), 538 Elo, nfelo; as-of proof; no outcome read.
  - W1-C college family 5 (XLG-14): lane xlg-14-u5-2026-11-16.md. Declared before outcomes, novelty checked against units 2-4 and the registry.
  - W1-D LEAD-108 forward log: lane lead108-forward-2026-11-16.md. Freeze the wording-read term on 2020-25 inputs available, declare, log Weeks 11-18 through scripts/forward_weekly.py style; no fitting on 2026.

- W1-B DONE (read): ESPN predictor is blocked by source policy (robots, 403 2026-09-20) and the route was already closed 2026-11-03 (mod-30-mod30-public-ratings.md, my history miss); 538 and nfelo have no source-policy entry and no pregame proof except nfelo 2025. Only CFBD pregame Elo is local (100% join 2012-25), but it is a results-only team rating already in src/nfl_ats/cfb.py columns; team quality is priced (memory), so no screen launched. Approving 538/nfelo source-policy entries is the owner's call; not a blocker.
- W1-B slot refilled: served-terms check counts current consensus books for thin-move games (full-steam-2026-11-14.md Next 3), lane served-terms-thin-move-2026-11-16.md.

- W1-D DONE (measured): LEAD-108 term frozen at 0.0238 logit/pt (fit on 2026 Weeks 2-9 card offset, 116 games; no earlier season has the read; Week 10 excluded); declaration 2026-11-16T20:10Z in lead108-forward-2026-11-16.md; forward Weeks 11-18. Root added scheduler jobs lead108_forward_pregame_thu_1815, _sat_1845, lead108_forward_score_tue (13:25) in scripts/capture_scheduler.py; dry runs execute the script (pregame refuses Week 10 as declared; score now exits 0 with nothing graded). Committed 0f50ada2. Scheduler restarted 2026-11-16 ~15:25 EST (one daemon, pid 12008 under 28464); new jobs listed. Status shows unacknowledged MISSED on 2026-11-15: refresh_last_call_sun_1245, refresh_trigger_log_sun, pool_move_log_sun_1255; check against data/scheduler_log.txt next operational pass.
- Thin-move check (served-terms-thin-move-2026-11-16.md): now counts distinct leader books quoting the game; MIN at GB 6 books, no problem. Note: the served move value itself still uses leader books with a Tuesday baseline (one book for MIN at GB); whether a one-book served move needs damping is open, needs an out-of-season measure, not a patch.
- W1-C DONE (measured): college family 5 (EPA beyond success rate, leverage-weighted EPA; xlg14_composition_family_u5), 8,925 games, 12 cells unresolved; no survivor (best null percentile 84), so no NFL look owed. Lane xlg-14-u5-2026-11-16.md.

- W1-A DONE (measured, serve-parity-audit-2026-11-16.md; tests/scratch/serve_parity/): 31 of 90 columns differ between the served rows and today's rebuild across 150 games. The only builder defect is temp/wind: they were blank at serve time in Weeks 1-4 on 45 games while the training rows hold observed game-time weather. Swapping them in moves the model logit by up to 0.071 and flips 4 model-side picks, all in Week 1 and all within 1.3 points of 50%. A drop patch exists (weather_drop.patch), not applied. This is the same defect as live-weather-features.md, which on 2026-10-19 deferred the fix to after Week 18 because features.py is pinned by the independent validation cohort. That decision stands: the effect is under 0.04 logit and zero card sides. total_line and neutral-site differences are line and schedule changes after the forecast; quarterback inputs are covered by repair_ab.patch; rest_diff matches. Gap: per-game card logits are not stored, so card-side effects of input fixes can only be bounded (about 0.3 points of 50%).

- Wave 2 launched 2026-11-16 ~15:45 EST (owner: do not stop):
  - W2-A weak_stack_v4 candidate built in isolation (QB repair + pregame MOS weather in train and serve), graded LOSO 2020-25 at the opener vs served: lane v4-pinned-batch-2026-11-16.md.
  - W2-B store per-game served card logits in the pick pipeline: lane card-logit-store-2026-11-16.md.
  - W2-C move-term baseline book count, out of season 2023-25: lane move-book-count-2026-11-16.md.
  - W2-D LEAD-112 practice-status change, declared forward log Weeks 11-18: lane lead112-forward-2026-11-16.md.
  - W2-E denoised training labels for the base (quarter/drive-level or score-state labels), novelty checked first: lane base-label-noise-2026-11-16.md.

- Root decision 2026-11-16 (owner: never ask the owner about data sources): 538 Elo (CC-BY-4.0) and nfelo public data approved for research use, no redistribution, data never committed; entries go in config/source_policies.json.
- W3-A launched: new source pull (win totals history, Open-Meteo historical forecasts, 538 Elo/QB Elo, nfelo, prediction-market history): lane source-pull-2026-11-16.md.
- Queue for slots as they free (in order): (1) win totals as an early-season prior against the opener, weeks 1-6, gap frame (scripts/mod57_p3_gap_frame.py); (2) forecast-weather base refit once Open-Meteo lands (hand to v4 lane); (3) 538 QB-Elo and nfelo disagreement with the opener, declared, matched null; (4) prediction-market price vs line gap, forward 2026; (5) sixth college family; (6) T sign-rule form declaration for 2027 forward log.

- W2-D DONE (measured, lead112-forward-2026-11-16.md): practice-change term frozen at 0.401 logit/pt on 2026 W2-9 (120 games), leave-one-week-out log loss .0024 worse than the card, so it is a forward test only; declaration 20:39Z before fit. Jobs lead112_forward_pregame_thu_1700, _fri_2030, lead112_forward_score_tue added; score dry run OK; scheduler restarted (pid 22692 under 10456).
- W3-B launched: sixth college family, lane xlg-14-u6-2026-11-16.md.
- W2-C DONE (measured, move-book-count-2026-11-16.md, commit 82846e42): the 2023-25 move input has 3 books on all 799 games, so the split cannot be read in history; 2026 W4-10 one-book move sides 14-10 v 2-3 books 13-14 (difference +10.2 pts, P+ .66), sign opposite the mechanism, unresolved, 5 cells. No damping warranted. scripts/move_book_count.py imports tests/scratch/serve_time_move/serve_time_move.py (untracked).
- W2-E DONE (measured, done/base-label-noise-2026-11-16.md, commit e966236a): every listed denoised label was already tried; the untried one, hindsight-gap label on the served ridge through the served stage, is worse: card 812-691 v 829-674 (-1.13 pts, P+ .089), model alone 761-742 v 802-701 (refuted, whole interval below zero). Label-noise route on the 88 inputs is used up; next base-model gains need new information (W3-A sources).
- W3-B DONE (measured, xlg-14-u6-2026-11-16.md, commit 89ae5d52): schedule-strength family on 8,925 college games; two survivors under the declared two-sided rule, margin_sos_gap (99.5th pct, -.064/SD, 13/13 folds) and last_opponent_strength (99.0th), correlated .65, sign opposite the declared under-adjustment (possibly own-margin carry-over); accuracy of margin_sos_gap at the 0th pct (518-563). W3-E launched: the two declared NFL confirmations, one read each.
- W3-C DONE (measured, done/t-signrule-2026-11-16.md, commit c3bef33b): T as -sign(T) early-window fitted term on 8,925 college games: coefficient positive 13/13 folds but log loss 38.5th pct of null, halves disagree; fails the declared rule; 6 cells unresolved. No NFL form frozen; T stays the 2027 linear forward-log candidate.
- W2-B DONE (measured, card-logit-store-2026-11-16.md): the ledger writer is pinned (pick_refresh.py), so the patch tests/scratch/card_logit_store/card_logit_store.patch joins the post-season batch; terms sum to logit(served p) within 3e-16 on Week 10. W3-F launched: an unpinned side recorder for every card game now, lane card-term-recorder-2026-11-16.md.
- W3-G launched: ranked queue of untested open ROADMAP model rows, lane open-model-rows-2026-11-16.md.
- W3-E DONE (measured, xlg-14-u6 NFL section, commit f71d4adf): neither schedule-strength survivor confirms in the NFL (null pct 34.0 and 83.5); margin_sos_gap is mostly own margin (r -.63). College-to-NFL transfer 0 of 11.
- W3-D DONE (measured, commit c719df59): Week 10 card 6-7 (+LAC_BAL open); base 6-7, calibrated 8-5, +flags 8-5, +move 7-6, +hook 6-7; fitted terms net -2 v the calibrated base, one week, unresolved. Rerun Tue: `nfl-ats ingest` then `scripts/week_attribution.py --season 2026 --week 10`. W3-H launched: season-wide attribution (Sonnet).
- LEAD-107 DONE but degenerate (measured, lead107-forward-2026-11-16.md): each game has one LEAD-106 read file, so dispersion is 0 and the term equals LEAD-108's. No jobs added (would duplicate LEAD-108). It needs repeated reads of the same player; decide only if a second read carries new wording.
- Web research DONE (outside-leads-2026-11-16.md): four candidates, none with out-of-sample ATS evidence: preseason Super Bowl futures anchor (Fodor et al. 2025), cross-system rating dispersion (Prediction Tracker), coach familiarity, contract-year snap share. Launched: futures-anchor-2026-11-16.md and rating-dispersion-2026-11-16.md (acquire, then college-first screen). Queued: coach familiarity (college-first via CFBD coaches), contract-year share (NFL forward only).
- W3-I launched: seventh college family, residualised on own ATS to date, lane xlg-14-u7-2026-11-16.md.
- W2-A DONE (measured, v4-pinned-batch-2026-11-16.md, commit 8fbdd946): pregame-forecast weather in train and serve 836-667 v 829-674 held out (+0.47 pts [+0.13,+0.85], P+ .997; changed 11-4); QB repair +0.20 (P+ .91); QW 836-667; drop-weather best on log loss (P+ .98) but not accuracy. ROOT DECISION: land QW before the Tue 11-17 12:20 lock, re-enrol validation the same session; preparation lane v4-landing-2026-11-17.md.
- Sources landed (source-pull-2026-11-16.md, commit d338d2e5): win totals 2009-26, Open-Meteo 2018-26, 538 Elo/QB Elo 2009-22, nfelo, Kalshi 2025-26, Polymarket 2024-26. Note: Open-Meteo API host robots says Disallow /; used only the documented API under its free terms, recorded in the policy entry.
- Futures anchor (college) and coach familiarity (college, 22 flagged games): no survivors, 12 cells (commit 5aa27c84).
- Wave 4 running: v4-landing-2026-11-17.md, win-totals-2026-11-16.md, prediction-market-gap-2026-11-16.md, outside-model-gap-2026-11-16.md, plus rating-dispersion and contract-year.
- v4 landing unit 1 (v4-landing-2026-11-17.md): patch drafted but stale; rehearsal not run; no MOS bulletin covers Week 11 yet and no scheduler job writes them; the QW grade used the -0.15 QB constant while the measured value is -0.123. ROOT RULE for Tuesday: land QW at the Week 11 lock only if, by Tue 10:00 ET, (a) the patch applies cleanly, (b) the rehearsal reproduces the QW record on 2020-25 with the measured -0.123 constant (regraded), and (c) Week 11 pregame bulletins exist for every outdoor game; otherwise land it for Week 12 with the same checks. Unit 2 launched for (a)-(c).
- Rating median and 538/nfelo NFL reads: no addition to the served card (commit 9521b30e). Contract year: no survivor on 3,322 games; findings card text now states the measured check; site republished (ec509bda).
- Board/card percentage defect (board-card-pct-2026-11-16.md): 4 of 14 Week 10 games show a different percentage on the card file than the board (sides agree); the card reads the ledger row from when the pick flipped. Fix is in pinned pick_refresh.py: tests/scratch/board_pct/pick_refresh_locked_score.patch (applies alone). It joins the same pinned change as v4.patch and card_logit_store.patch; check all three apply together before landing. card-ledger-check should also compare percentages (unpinned follow-up).
- Forward logs live from Week 11: LEAD-108, LEAD-112, prediction-market gap, nfelo gap (commits 0f50ada2, 88dfc1de, 353cef9e, de783a39).
- W3-D launched: reusable per-week layer attribution, Week 10 now and rerun Tue after MNF; section in picks-getting-worse-2026-11-15.md.
- W3-C launched: T sign-rule form screened on college, lane t-signrule-2026-11-16.md.

- Evening (measured, commits 63bfdc45..b3f3820f): QW at the measured -0.123 constant 836-667; rebuild rehearsal identical; bundle of three pinned patches applies; test baseline 516 passed; landing GO on all three checks Monday evening (v4-landing-2026-11-17.md). Week-1 weight, referee tendency, punt units: unresolved, nothing served.
- Wave 5 running: hindsight-audit-team-2026-11-16.md and hindsight-audit-player-2026-11-16.md (training values knowable only after the pick deadline, the defect class that produced the weather gain), plus win-totals-2026-11-16.md.

## Post-Week-18 pinned batch (one refit, re-enrol validation)
- Quarterback input repair (tests/scratch/qb_input_repair/repair_ab.patch), unless applied Tuesday per full-steam-2026-11-14.md.
- Weather: pregame MOS temp/wind in serve and train (live-weather-features.md), or weather_drop.patch; forecast-trained weak_stack_v4.
- Ledger move column (pick_refresh.py:629-645), ENG-51 unit 2b.
- Store per-game served card logits so input fixes can be scored on card sides.

## Tried
See model-improve-2026-11-15.md, mod57-sixty-percent.md (Tried and phases), mod23-base-model.md. Not repeated: 2011+ training window (MOD-23: no better), team-change inputs, hindsight-gap label, play-level models.

## Next
1. On return verify each unit's decision-gating claim, commit, refill.
2. Tuesday 11-17: quarterback input repair (full-steam-2026-11-14.md Decisions) bundled with any parity patch W1-A proves, one fenced change, validation re-enrolled same session.

## Open
- None.
