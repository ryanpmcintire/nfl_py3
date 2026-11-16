# move-timing-parity-2026-11-16

## Goal
Train/serve parity of the served market-move term (goal-65). Does the move the stage was fitted on (2023-25) cover the same window, baseline, books and aggregation as the move computed at each live refresh? If not, refit the served stage on a training move rebuilt the serving way and grade it.

## State
Timing table (read + measured, 2026-11-16 17:58 EST):
| | Training (active leader_median_through_sunday_prekick_v1) | Serving (refresh-picks) |
|---|---|---|
| Code | scripts/sunday_market_probability_eval.py:22-74, 187-220; frozen file read at src/nfl_ats/pick_probability_fit.py:63,154 | src/nfl_ats/card_view.py:517-526 -> pick_probability.py:730-788 -> sharp_book_movement_features.py:43-169 include_sunday=True |
| Quotes | artifacts/sharp_book_weighted_movement/spread_quotes.parquet, intraday_hourly only (eval.py:189-191) | data/market/raw live captures (pick_probability.py:748) |
| Baseline | last quote in [Mon 00:00, Wed 00:00 ET); hourly -> Tue 23:55 ET; no pre-Wed quote required (eval.py:67-68) | same bounds; book needs a pre-Wed quote (sbmf.py:137-139); live -> Tue 14:00-18:00 ET (measured 2026) |
| End | nominal min(kickoff, Sun 12:45 ET) (eval.py:33-45); effective Sun 10:55 ET for every Sun/Mon game (last hourly snapshot, measured) | min(kickoff, refresh time) (pick_probability.py:762-763; sbmf.py:59-64), last calls Sun 12:45 and 15:35 (capture_scheduler.py:2118-2124); latest live leader quote median ~12:00 ET Sun for 1 PM and late games, up to 15:20 (measured 2026) |
| Books | LEADER_BOOKS bovada, williamhill_us, mybookieag (eval.py:47) | same |
| Aggregation | per-book sum of diffs, median across books | same |
| Available | legacy (Sunday-excluded) exposure; parity-mismatch games keep the legacy value (eval.py:207-220) | leader_books > 0 (pick_probability.py:771-777) |

Verdict: windows differ (end time ~10:55 vs ~12:00+ Sunday, baseline rule, availability rule).

Declared 2026-11-16 17:58 EST before grading: arm M = serving function sharp_book_movement_features(include_sunday=True) on the 2023-25 archive, all decision labels before cutoff = min(kickoff, Sun 15:35 ET), serving baseline and availability rules; effective Sunday end = 12:25 ET snapshot (archive has nothing between 12:25 and 16:10, so the late-game 12:25-15:35 stretch cannot be rebuilt). Replaces the move and available columns for 2023-25 games; served four-term stage refit LOSO 2020-25 on the 1,503-game opener population with the QW model probability (artifacts/v4_candidate/opener_QW.parquet), compared with the QW reference 836-667, paired season- and week-block bootstrap. Subsets: after-cap games (kickoff after Sun 16:00 ET) and Sunday 1 PM games. Looks: 1 arm x 3 metrics + 2 subsets. Family move_timing_parity_2026.

Result (measured 2026-11-16 18:01 EST; artifacts/move_timing_parity/20261116T230003Z/results.json, script scripts/move_timing_parity.py):
- Rebuild: 159 of 799 2023-25 moves changed, mean abs 0.12 pt; availability 799 both. Every after-cap (269) and Sunday 1 PM (372) game: training end Sun 10:55 -> arm end Sun 12:25.
- All 1,503: reference QW 836-667 reproduced; arm M 832-671, changed picks 36 (arm 16-20). Accuracy -0.27 pts [-0.85, +0.34] P+ 0.167 (season block); log loss gain -0.000065 P+ 0.459; Brier gain -0.000035 P+ 0.447. Move coefficient per fold 0.186-0.223 (reference 0.187-0.244); OOS-minus-in-sample log loss gap 0.00090 (reference 0.00103). Reliability tables in results.json.
- After-cap 269 (week block): 143-126 vs 142-127, changed 13 (arm 7-6); +0.37 pts [-2.48, +3.08] P+ 0.55; log loss gain +0.0014 P+ 0.80.
- Sunday 1 PM 372: 209-163 vs 212-160, changed 19 (arm 8-11); -0.81 pts [-3.19, +1.62] P+ 0.22; log loss gain -0.00064 P+ 0.33.
- Recorded 7 cells, family move_timing_parity_2026, all unresolved_below_power (artifacts/move_timing_parity/cells.json); pool run once, clean. Nothing served.

## Tried
- Read both code paths; measured archive and live capture times; arm M graded once (first run crashed on a column clash, fixed, rerun).

## Next
- The late-game 12:25-15:35 stretch has no 2023-25 source in this archive; the only way to test it is forward: log the served move at the 12:45 and 15:35 calls per game from 2026 Week 11 and compare with the move at Sun 12:25.
- Decision implication: matching the end time does not beat the served stage; no serving change.

## Open
- Live Thursday games: latest live leader quote median ~26 h before TNF kickoff (measured), training has hourly quotes to kickoff.

## Root note 2026-11-16
- The forward log of the served move per game at the 12:45 and 15:35 calls already exists: scripts/card_term_recorder.py runs at Sun 12:55 and 15:45 (card_terms_* jobs) and stores the move contribution per game, so the 12:25-15:35 stretch is measured from Week 11 without a new job.
