# Dashboard formation selection state

## Goal

Expose the selected projected-offense player to assistive technology.

## State

Complete and integrated. Formation player buttons initialize with `aria-pressed="false"`, and activating a player sets only that player to pressed while preserving the visible selection and detail update.

## Tried

Integrated release checks passed: pytest reported 4529 passed and 9 skipped; Ruff format, Ruff check, mypy, and publish-board passed. Browser verification confirmed exactly one selected formation player and no JavaScript exceptions.

## Next

None.

## Open

None.
