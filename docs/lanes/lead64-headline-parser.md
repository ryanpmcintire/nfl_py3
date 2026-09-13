# lead64-headline-parser

## Goal

LEAD-64 fallback: parse PFT injury headlines (`data/raw/injury_news`) into a
timestamped (player, team, designation, filed) table as an artifact. Data
product only; nothing reads it yet. Done when `nfl-ats injury-headlines`
writes `artifacts/injury_headline_designations/<ts>/` with clean names and
teams for the newest capture's headlines.

## State

First pass done, uncommitted (opencode lane C, `mimo-v2.5-free`):
`src/nfl_ats/injury_headlines.py` (419 lines), CLI `injury-headlines`
(`src/nfl_ats/cli_commands/clv.py:723`, registered :929-958),
`docs/injury_headlines.md`. Coordinator measured 2026-09-12 20:05 ET:
`nfl-ats injury-headlines --dry` -> 415 rows over 54 captures, 213 unparsed,
202 designations (80 out, 81 questionable, 14 doubtful, 21 ir, 6 active).
Ruff and mypy clean (lane report, then re-run by coordinator for the CLI).
Second pass (lane F, `mimo-v2.5-free`) verified 20:25 ET: roster index
from the newest player-snapshot season resolves team-less names, two-player
headlines split, `team_source` column, `roster_snapshot_id` in the summary.
Measured `--dry`: 458 rows, 55 captures, 221 unparsed. Third pass (lane H,
`big-pickle`, after mimo quit silently) verified 21:10 ET: opponent words
after vs/at/against are never the team, roster wins when it disagrees with
a single headline team, "rule out" scopes forward to the next designation,
every row with a team carries team_source (measured: 273 teamed rows, 0
without a source; 137 roster, 136 headline). The CLI contract fixture
(`tests/fixtures/cli_contract.json`) gained the `injury-headlines` entry.

## Tried

- Lane C's regex-only parse: 24/25 newest-capture headlines carry a
  designation keyword, but names spill ("dezhaun stribling carted off",
  "second round pick kayden mcdonald"), two-player headlines collapse into
  one bogus name, and headlines without a team word get team None.

## Next

Compare the parsed designations against the official nflverse feed on the
next late morning (feed regenerates ~07:35-08:15 ET): for each (player,
designation) the feed carries, was the headline earlier, and by how much.
Then the owner call in Open. Later:
compare the parsed designations against the official nflverse feed on the
next late morning (the feed regenerates ~07:35-08:15 ET).

## Open

Whether to feed the parsed designations into `report_status` when the
league feed is older than the newest parsed headline (owner call: it changes
the served injury features).
