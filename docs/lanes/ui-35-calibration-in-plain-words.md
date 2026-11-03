# UI-35

## Goal
Calibration in plain words on the History page: when the card said X percent, how often did that side cover.

## State
Unit 1 done 2026-11-03 ~17:50 local. No such pick-level table existed (Model page only has home-cover reliability from the backtest). Added a fold "When the card says X%, how often did it cover?" in the season-so-far section: board_site_content.py `_calibration_rows` / `SeasonSoFar.calibration_rows`, board_terminal.py `_calibration_html`. Three equal-count bands of displayed_score (frozen ledger) over settled 2026 picks. Rendered: 50-53% n41 avg 51 covered 44; 53-56% n40 avg 55 covered 62; 56-70% n40 avg 59 covered 55. Not published or committed.

## Tried
No out-of-season sentence: the MOD-47 Unit 2 numbers (slope .92) live only in the lane, no stored artifact.

## Next
Orchestrator: review, publish-board, commit. Optional unit 2: store the out-of-season reliability bins as an artifact, then add the one-line sentence.

## Open
Look count: 3 bands, one family.
