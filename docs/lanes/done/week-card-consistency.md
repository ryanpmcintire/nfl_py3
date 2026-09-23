# Week card consistency

## Goal

Keep archived weekly cards inside the live board and inspector visual contracts while showing only stored historical fields.

## State

Complete. Archive rows use the live seven-column table structure and cell classes. Unavailable historical fields use em dashes. The selected archive detail uses the live inspector container and heading. No CSS change was needed.

## Tried

Compared `board_week_navigation.py` and `board_week_navigation.js` with the live table and inspector markup in `board_terminal.py`.

**Measured:** `node --check src/nfl_ats/board_week_navigation.js` and Python compilation both exited 0. `pytest -q -n 0 --basetemp .tmp/week-card-consistency-pytest-20260922 tests/test_board_site.py` passed 10 tests.

**Measured:** After Week 3 publication, cache-busted browser checks selected Week 1 and Week 2 as archives and Week 3 as the live card. All three cards had 16 game rows, the same seven headers and cell classes, one best row, an inspector, and no page or board overflow at 1440 by 1100. Week 1 and Week 2 retained one selected archive row; click and ArrowDown updated the Week 2 detail and selection. At 390 by 844, all three cards collapsed to one column with the inspector below the board and no page or board overflow.

**Measured:** Screenshots are in `.tmp/week-navigation/week-card-week1-desktop.png`, `.tmp/week-navigation/week-card-week1-mobile.png`, `.tmp/week-navigation/week-card-week2-desktop.png`, `.tmp/week-navigation/week-card-week2-mobile.png`, `.tmp/week-navigation/week-card-week3-desktop.png`, and `.tmp/week-navigation/week-card-week3-mobile.png`.

## Next

None.

## Open

Historical artifacts do not contain live kickoff, current book, flip-line, or confidence evidence. Archived cards therefore show em dashes for those fields instead of reconstructing unsupported values.
