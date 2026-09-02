# Pool rules: the composed `PoolRules` (POL-01)

This documents the composed pool rule set now exposed by
`nfl_ats.pool_workbench.PoolRules` (`src/nfl_ats/pool_workbench.py`), what
was added under ROADMAP item POL-01, and the ROADMAP status this work
package proposes for POL-01 and BET-09. Every fact below is either **read**
from a cited file/line or **measured** by running the cited command/test;
none is guessed.

## The composed rules

| Rule | Value | Source |
|---|---|---|
| Format | ATS, forced picks, no passes | `docs/pool_edge_plan.md:73-77` ("The pool's actual format (confirmed by the user, 2026-08-17 evening)") |
| Regular-season games | 272 | `docs/pool_edge_plan.md:75` |
| Playoff games | 13 | `docs/pool_edge_plan.md:76` |
| Cards per season (`PoolRules.cards_per_season`, alias of `total_games`) | 285 | `docs/pool_edge_plan.md:75-77`; `AGENTS.md` "A promotion bar is not a decision bar": "The pool is FORCED PICKS: 285 cards must be submitted either way." Derived from `regular_season_games + playoff_games`, not a second hardcoded literal. |
| Best Pick | 1 per regular-season week | `docs/pool_edge_plan.md:78-79`; playoff weeks award none (`src/nfl_ats/pool_workbench.py` `_rules_section` bullet) |
| Grading line (`PoolRules.grading_line`) | `"opener"` | `docs/pool_edge_plan.md:5` ("beat the OPENING line the user's Splash Sports pool grades against"); `AGENTS.md` "Grade the decision at the OPENER" |
| Line lock | Tuesday (revised once Wednesday, then frozen for grading) | `docs/pool_edge_plan.md:80-88`, owner-corrected 2026-08-20 |
| Pick deadline (`PoolRules.deadline_for`, `PoolRules.deadline_rule`) | `min(game's own kickoff, that week's Sunday 16:00 ET lock)` | `src/nfl_ats/pick_refresh.py:151-157` (`pick_deadline`) and `:132-148` (`sunday_pick_lock`), read-only, imported not reimplemented; owner rule 2026-08-20, re-confirmed 2026-09-01 (`C:/Users/Ryan/.claude/projects/F--Repos-nfl-py3/memory/picks-lock-at-kickoff.md`); design record `docs/late_week_refresh.md` |
| Tiebreak (`PoolRules.tiebreak`) | `"final_score_last_game"` | `src/nfl_ats/tiebreaker.py` module docstring: "The pool breaks ties on the final score of the week's LAST game (owner, 2026-09-01...)" -- read-only reference; `nfl_ats.tiebreaker` implements the guess, not duplicated here |

`PoolRules.describe()` renders these as plain-English sentences for the
board/report; see `src/nfl_ats/pool_workbench.py`.

## What changed in `pool_workbench.py`

Additive only -- no existing field, method, or consumer behaviour changed.
Added to `PoolRules`:

- `grading_line: str = "opener"` and `tiebreak: str = "final_score_last_game"`
  -- new dataclass fields (both accepted by the existing `from_dict`
  override mechanism automatically, no code change needed there).
- `cards_per_season` -- a `@property` alias of the existing `total_games`
  (not a duplicate literal, so it cannot drift out of sync with
  `regular_season_games`/`playoff_games`).
- `deadline_rule` -- a `ClassVar` (not a dataclass field) holding
  `nfl_ats.pick_refresh.pick_deadline` directly, wrapped in `staticmethod`
  so instance access does not auto-bind `self` as its first argument.
  `PoolRules.deadline_rule is nfl_ats.pick_refresh.pick_deadline` is `True`
  (measured, `tests/test_pool_workbench.py::test_pool_rules_composed_fields_match_cited_sources`).
- `deadline_for(kickoff, week_kickoffs)` -- the ergonomic per-game call:
  composes `nfl_ats.pick_refresh.sunday_pick_lock(week_kickoffs)` and
  `pick_deadline(kickoff, sunday_lock)`, both imported verbatim from
  `nfl_ats.pick_refresh`; nothing about the deadline rule itself is
  reimplemented in `pool_workbench.py`.
- `describe()` -- plain-English rule list for the board/report.

`nfl_ats.pick_refresh` was confirmed **measured**, via
`python -c "import ast; ..."` walking its imports, and by successfully
importing both modules together, to have no circular-import path back to
`nfl_ats.pool_workbench` (`pick_refresh` imports `active_model`,
`calibration`, `clv`, `constants`, `data`, `io`, `lines`, `margin`,
`market_data`, `outcomes`, `prediction_safety`, `provenance`, `weekly`; none
of those import `pool_workbench` or `pool`), so `deadline_rule`/`pick_deadline`
are imported at module top level rather than lazily.

Tests added, all additive, in `tests/test_pool_workbench.py`:
`test_pool_rules_composed_fields_match_cited_sources`,
`test_pool_rules_deadline_for_agrees_with_pick_refresh_on_every_slot`
(agreement with `pick_refresh.pick_deadline`/`sunday_pick_lock` on a
Thursday game, a Sunday 13:00 ET game, a Sunday 16:25 ET
late-afternoon-window game, SNF, and MNF -- the last three all resolve to
the week's Sunday 16:00 ET lock, not their own kickoff), and
`test_pool_rules_describe_is_plain_english_and_cites_the_rules`. All 14
tests in the file pass (measured,
`.\.tools\uv.exe run --no-sync pytest tests/test_pool_workbench.py`), as do
the pool-related rows of `tests/test_board_terminal.py` that also import
`pool_workbench`.

## ROADMAP status proposed

**POL-01** ("Pool rule configuration: ATS, straight-up, confidence,
survivor, scoring, entry count", `ROADMAP.md:344`): propose 🚧, not ✅. The
`PoolRules` dataclass now composes the confirmed ATS/forced-pick format
with the per-game deadline rule (via `pick_refresh.pick_deadline`, imported
not reimplemented) and the tiebreak rule (named, referencing
`tiebreaker.py`, logic not duplicated), closing the specific composition
gap this work package was asked to close. It remains 🚧 because the
original roadmap line names straight-up, confidence, survivor, and
entry-count pool VARIANTS this project's actual Splash pool is not (the
pool is ATS forced-picks only, per `docs/pool_edge_plan.md`) -- building
configuration for pool types that do not apply to the one pool this project
plays would be speculative scope, not composition of what already exists.

**BET-09** ("Responsible-use controls: paper mode default; prominent
limitations and no auto-wager path", `ROADMAP.md:338`): propose ✅.
`tests/test_no_wager_path.py` (measured, 4 tests, 0.83s) now enforces in
code, not just prose: (a) no `pyproject.toml` dependency (any group)
matches a denylist of sportsbook/exchange wagering-client package names;
(b) no wager-placement verb (`place_bet`, `place_wager`, `submit_bet`, a
POST to a `/bets`-shaped endpoint) exists anywhere in `src/` or `scripts/`
outside the three modules confirmed read-only-quotes
(`nfl_ats.odds`, `nfl_ats.odds_backfill`, `nfl_ats.market_data` -- verified
**measured** this session that `odds_backfill.py:224-226` and
`market_data.py:249-251` both call `urllib.request.urlopen` with no `data=`
argument, i.e. a GET, never a POST); (c) a paper-only/limitations statement
exists (`docs/responsible_use.md`, new file this work package, since
`README.md:548-553`'s existing "Responsible use" section addresses a reader
who might wager real money rather than stating the codebase's own
paper-only architecture). Today's tree has zero wager-placement code and
zero wagering-client dependencies (measured, same test run) -- the guard
exists to keep that true, not because a violation was found.
