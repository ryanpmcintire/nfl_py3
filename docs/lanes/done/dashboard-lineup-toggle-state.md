# Dashboard lineup toggle state

## Goal

Expose the initial projected-lineup unit selection to assistive technology.

## State

Complete and integrated. The three initially active lineup toggle buttons render with `aria-pressed="true"`, matching their visible active state and the existing click behavior.

## Tried

Integrated release checks passed: pytest reported 4529 passed and 9 skipped; Ruff format, Ruff check, mypy, and publish-board passed. Browser verification confirmed the initial pressed states and no JavaScript exceptions.

## Next

None.

## Open

None.
