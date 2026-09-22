# Dashboard ticker keyboard

## Goal

Keep the continuous game ticker while exposing each game only once in the keyboard tab order.

## State

Complete and integrated. The first ticker sequence remains linked; its visual repeat uses non-focusable, assistive-technology-hidden spans.

## Tried

Integrated release checks passed: pytest reported 4529 passed and 9 skipped; Ruff format, Ruff check, mypy, and publish-board passed. Browser verification confirmed 16 unique ticker links, 16 `aria-hidden` duplicate spans, and no JavaScript exceptions.

## Next

None.

## Open

None.
