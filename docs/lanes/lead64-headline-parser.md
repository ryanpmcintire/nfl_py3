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

- Morning comparison 2026-09-13 (subagent, reported, unverified by the
  coordinator; report `tests/scratch/lanes/lead64_morning_compare_20260913.md`,
  script `lead64_compare.py`, artifact `artifacts/injury_headline_designations/20260913T113000Z/`):
  `date_modified` is null on all 182 official 2026 rows, so lead time used
  first-capture-seen as the proxy; 51 of 59 target statuses first appeared in
  one capture (2026-09-12T13:41Z), so the proxy mostly measures time before
  that batch. 59 official week-1 rows (Out 27 / Q 26 / D 6); 9 matched a
  headline (15%), all 9 with the headline first, lead 16.8-21.9 h (median
  18.5). 12 of 21 distinct headline events had no official row (6 were IR,
  which the feed never emits). Parser defect: `designations.parquet` holds
  518 raw rows but 30 unique (url, player, team, designation), the same
  headline re-emitted per capture, inflating the CLI summary counts.

## Next

Dedupe `designations.parquet` on (url, player, team, designation) keeping the
earliest filed time, then re-run the comparison script against the next
official capture. Then the owner call in Open.

## Open

Whether to feed the parsed designations into `report_status` when the
league feed is older than the newest parsed headline (owner call: it changes
the served injury features).
