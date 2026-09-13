# Injury Headlines Parser

Fallback injury-designation source from PFT (ProFootballTalk) captured headlines.
Used on mornings when the official league injury-report feed is late.

## Source

Captures live under `data/raw/injury_news/<UTC ts>/current.parquet`.
Each capture contains PFT article headlines with `headline_guess` (lowercase words),
`matched_keywords`, and `injury_relevant` flag.

## Patterns

Designation detection uses regex over `headline_guess`:

| Designation | Patterns matched |
|---|---|
| `out` | "ruled out", "rule out", "is out", "out for season/game/week/day", "will not play", "wont play", "will miss", "downgraded to out" |
| `doubtful` | "doubtful" |
| `questionable` | "questionable" |
| `ir` | "placed on ir/injured reserve", "to ir/injured reserve", "on ir/injured reserve" |
| `active` | "activated", "cleared to play", "will play", "expected to play", "good to go", "removed from the injury report", "off injury report", "set to play" |

Player name extraction: strip leading team nickname/city, optional position token (QB, RB, WR, TE, OL, DL, LB, CB, S, K, P), and action verbs (place, put, add, list, rule, ruled, call, set, etc.) from the pre-designation text. Names are validated against the player roster when available; trailing stop-words (carted, off, listed, as, is, was, etc.) and position/ordinal words (second, round, pick, etc.) are stripped from unmatched names.

Team resolution: first checks nickname/city matching in the headline text. When no team is found, falls back to roster lookup via the latest player snapshot (full_name -> team). `team_source` records the origin: `'headline'` or `'roster'`.

## Output Schema

### `designations.parquet`

| Column | Type | Description |
|---|---|---|
| `url` | str | article URL |
| `filed_at_utc` | datetime | publisher's lastmod stamp |
| `first_seen_utc` | datetime | earliest capture that saw this URL |
| `player_name` | str | extracted player name |
| `position` | str or null | position token if found |
| `team` | str or null | team abbreviation |
| `team_source` | str or null | `'headline'` if team came from the headline text, `'roster'` if resolved from the player roster, or null |
| `designation` | str or null | one of: out, doubtful, questionable, ir, active |
| `parsed` | bool | whether designation was identified |
| `headline` | str | raw headline_guess |

### `summary.json`

```json
{
  "total_rows": 25,
  "designation_counts": {"out": 3, "doubtful": 4, ...},
  "unparsed_count": 3,
  "source_capture_count": 55,
  "newest_lastmod": "2026-09-12T19:52:43.364000+00:00",
  "roster_snapshot_id": "20260912T204606Z"
}
```

### `manifest.json`

```json
{
  "source_captures": ["20260819T191639Z", ...],
  "git_head_short_sha": "abc1234",
  "roster_snapshot_id": "20260912T204606Z"
}
```

### `latest.json`

Points at the most recent run directory. Same shape as `artifacts/best_pick_ranking/latest.json`.

## Limits

- Multi-player headlines (e.g. "rome odunze xavier woods questionable for bears vs panthers") now emit one row per rostered player, each taking the designation nearest after the player's own name. A `rule out`/`rules out` phrase marks `out` for the names that follow it until the next designation or the word `list`/`lists`/`and` (e.g. "jets rule out joseph ossai list ... dangelo ponds as doubtful" -> ossai out, ponds doubtful). Players not on the roster are included with trailing stop-words removed.
- Headlines without a recognized designation keyword are kept with `parsed=False`.
- A team word that follows `vs`, `v`, `at`, `against`, `versus`, `host`, `hosts`, `visit`, `visits`, `face`, or `faces` names the opponent and is never used as the player's team. Team resolution uses the roster when the headline yields only opponent words; when the roster gives exactly one team that disagrees with the headline's team word the roster wins with `team_source` `'roster'`. `team_source` is `'headline'` for every team read from the headline, `'roster'` for a roster-resolved team, and null only when no team is known.
- Roster-based name matching uses normalized forms (punctuation stripped, case-insensitive) and handles both dotted initials (T.J. Sanders) and apostrophes (D'Angelo Ponds).
- When no player snapshot is available, the parser operates without a roster (team may be null for headlines lacking a team name).
- Point-in-time discipline: `first_seen_utc` equals a real capture directory timestamp; no interpolation.
