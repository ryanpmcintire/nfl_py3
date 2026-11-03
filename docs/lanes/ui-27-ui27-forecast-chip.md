# UI-27 Kickoff forecast chip on each game row

## Goal
Kickoff forecast chip on each game row: wind, temperature, rain and roof, with no effect claimed that is not measured (ROADMAP UI-27).

## State
Added 2026-11-03, unmeasured. Weather is something pool players look up before kickoff; the data is already captured in the forecast archive. The chip shows it with no claimed effect.

## Tried
Nothing yet. Novelty: ENV-02 (stadium history), MOD-31 and MOD-35 (research terms). Grep of src/nfl_ats/board_terminal.py for 'wind' found no weather display.

## Next
First unit: Add a chip to the game row and phone drawer, render at 1440 and 390 widths, and compare before and after.

## Open
Data: data/raw/forecast_archive and roof tables. Zero crossing closes nothing; record unresolved results with a closing ground.
