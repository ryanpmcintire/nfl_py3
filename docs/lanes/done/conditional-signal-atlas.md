# Conditional signal atlas (MOD-19)

## Goal

Provide a findings explorer backed by paired comparisons of fitted calibrated probabilities.
Every served pick remains the result of one calibrated probability; no atlas member flips a side.

## State

- Complete for the current six-family declaration as of 2026-09-26.
- The declaration crosses three fitted terms (`composition_flag_sum`,
  `market_move_toward_home`, and `market_move_available`) with two splits
  (`week_in_season` and `spread_band`).
- `src/nfl_ats/signal_atlas.py::_split_masks` implements those two splits and rejects an
  unregistered split. The builder checks every declared cell against `registry/split_library.json`.
- `scripts/signal_atlas.py` is a compatibility entry point for
  `scripts/conditional_signal_atlas.py`; its production help exposes the paired atlas command.
- The declaration is retrospective and has `serving: false`. The research classification remains
  `unresolved_below_power`; no research-closing ground is established.

## Tried

- Built and loaded the six-family artifact through the production path.
- Added the findings rail, split tabs, detail view, and responsive layout to the board.
- Verified the declared family shape, source/declaration freshness guards, and compatibility CLI.
- Reconciled `spread_band` metadata after it had remained labeled unscored despite inclusion in
  the saved six-family declaration. The split rule and library version did not change.

## Next

- Treat further families and splits as a separate expansion with a new declaration and chronology
  review before scoring.
- Supply the missing source frame before attempting `ol_rush_continuity`.
- Certify historical per-game feature availability and reserve an untouched outer selection test
  before making any serving claim.

## Open

- Individual situational flags are present in the source artifact but were not separately fitted;
  they cannot become standalone side-changing rules.
- `ol_rush_continuity` still requires snap-count and week-1 roster inputs absent from the current
  atlas source frame.
- The seven queued split directions remain undeclared and unscored.
- An interval crossing zero does not close a signal.
