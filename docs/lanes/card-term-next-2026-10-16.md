# Card fifth term: totals-move scaled margin shift

## Goal
Find the next fitted term for the four-term served card (LOSO 2020-2025 full 863-640, Tuesday 836-667) that is untested, has pre-deadline free history, and has a named mechanism.

## State
Inventory (read, 2026-10-16). Every ideation candidate is already graded or source-gated:
- LEAD-66..74, 77..79, 82..88, 90..98: graded, unresolved or refuted (ROADMAP.md:853-885). LEAD-98 price skew is the latest fifth-term try (Brier worse).
- LEAD-75 wind innovation, 76 pre-2009 prior, 80 tiebreak entrants, 81 travel, 89 wait option: zero data, source-gated.
- acc-u7 survivors 7a availability history, 7b luck-stripped, 7c acquisitions: all graded (docs/lanes/done, acc-u7b lane, registry families mod24_availability_history, mod24_luck_stripped_inputs); Wikipedia pageviews discarded as already screened.
Untested as a card term (grep ROADMAP/docs/registry: total move appears only in LEAD-88 tiebreaker totals, tuesday_card_clv CLV, midweek_steam which is spread only): the Wednesday-to-deadline move of the TOTAL.

## Declaration (before any outcome is read)
Term `total_move_margin_shift` = leader-book median net total move (same window, books and timestamp guards as `sharp_book_movement_features`: Monday to min(kickoff, Sunday 4 PM ET), Wednesday on, strictly before cutoff) times `-tue_open_home_spread / tue_open_total`. Mechanism: expected margin is proportional to the total at a fixed team-strength ratio, so a total that rises (falls) moves the home margin by -spread/total per point; the spread did not move in response. No fitted constant in the feature; the coefficient is fitted LOSO. Games with no total move observed: term 0, no indicator. Tuesday view zeroes the move as the four-term harness does.
Data: `data/market/raw` totals quotes, tue_open + intraday labels, 2020-2025.
Leakage guard: quote observed_at_utc < cutoff; open total from tue_open only.
Arms (2): full view and Tuesday view; Tuesday zeroes the term as it zeroes the move, so it equals the four-term by construction.
Looks: 2 views x (5 metric cells) = 10, plus 6 fold coefficients per view.

## Tried
2026-10-16 `scripts/card_term_2026_10_16.py` -> `tests/scratch/card_term_2026_10_16/results.json`. Coverage 1475/1503 games with an observed total move (nonzero 1125); term SD 0.151; correlation with spread move -0.034.
Full view LOSO OOS: five-term 861-642 (57.29%) vs four-term 863-640; Brier .244878 vs .244838, log loss .682870 vs .682774; model-only 800-703, market 746-757. Accuracy -0.13 pts [-0.99, +0.72] P+ 0.359; Brier improvement -0.00004 [-0.00070, +0.00057] P+ 0.466. Changed picks 50: five-term 24, four-term 26. Coefficient positive 6/6 folds (mean 0.328 per natural unit, range 0.081-0.471), sign stable but no out-of-season gain. Tuesday view identical to four-term (836-667).
Recorded 2 cells family card_term_total_move, unresolved_below_power. 10 looks declared; 2 recorded.

## Next
Not served. Remaining untested candidates are source-gated (LEAD-75/80/81/89); next subtask is a fetch of a historical wind-forecast pair archive or entrant observations, only if a free dated source exists.

## Open
None.
