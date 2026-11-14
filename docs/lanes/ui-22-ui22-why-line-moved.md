# UI-22 Why the line moved

## Goal
One plain sentence per unplayed game whose line moved since Tuesday, plus one sentence naming dated injury news about a regular on either team when it exists. The news is reported, never offered as the cause. ROADMAP row UI-22.

## State (2026-11-14)
News-found sentence built and rendered; not committed (orchestrator reviews and publishes).
- `src/nfl_ats/board_move_reason.py`: new `DesignationChange`, `_snapshot_time`, `_week_statuses`, `designation_changes`, `news_sentence`; `build_move_reasons` rewritten from the quote step down. Move definition unchanged (market consensus now v first market capture at or after Tuesday noon ET, at least one quote step).
- News source: official injury designations in `data/raw/nflverse_injuries/<UTC stamp>/injuries.parquet` (several captures a day; `date_modified` is empty, so the item time is the stamp of the first capture that shows the status). A change is a status of Out, Doubtful or Questionable that differs from the previous capture, after the Tuesday freeze.
- Selection rule `largest_snap_share_then_latest_designation_change` (given by the orchestrator, lane had none): among players on either team whose mean season snap share is above `MAJORITY_SNAP_SHARE` (more than half the snaps; `data/players/forward_snaps`), take the largest snap share, then his most recent change.
- Sentence: "The line moved X toward T. Injury news since Tuesday: TEAM position Name was listed as status as of Weekday." Weekday is the Eastern day of the first capture. No item: the old text stays ("No captured news explains this move.").
- Evidence (player id, previous and new status, capture time, capture path, snap share, rule, count of candidates) goes to `docs/move_reasons.json` only.
- Removed: the beat-post and `data/processed/injury_revisions_2026.parquet` path that only counted news at or before the move onset. The revisions file is stale (last row Week 8, 2026-11-01), which with the onset limit is why nothing ever rendered.
- Verified 2026-11-14: ruff check and format clean, mypy clean on the module, `tests/test_public_board.py` 134 passed.
- `publish-board` run once 2026-11-14 15:39 ET, exit 0; `docs/index.html` now carries six news sentences, each after "The line moved ...": "Injury news since Tuesday: CAR linebacker Bobby Okereke was listed as out as of Saturday." (CAR at NO); "MIN defensive back Josh Metellus ... questionable" (MIN at GB); "IND linebacker Akeem Davis-Gaither ... out" (MIA at IND); "KC defensive back Mansoor Delane ... questionable" (KC at ATL); "NYJ receiver Garrett Wilson ... questionable" (BUF at NYJ); "SEA quarterback Sam Darnold ... out" (SEA at LV). LAC at BAL has no item and keeps the old text. Darnold and Okereke checked by hand against the page and the raw captures.

## Tried
- 2026-11-03: first build compared pool line with market (all nine "moves" were the pool's own half-point hook); fixed to market v market.
- 2026-11-14 hand check from raw captures: Sam Darnold SEA QB and Bobby Okereke CAR LB have no status through the Friday 20:00 ET capture and Out in the Saturday 06:00 ET capture (`20261114T110007Z`). All Week 10 designations first appear in that capture.

## Next
1. Orchestrator: review the diff, look at the 390 px render of the Books now cell (wrap not looked at), commit, push.
2. Decide the ranking rule (see Open) and, if changed, rerun `publish-board`.

## Open
- Every Week 10 item was first captured Saturday 06:00 ET, after each move began (onsets Tuesday to Thursday in `docs/move_reasons.json`). The sentence reports news and claims no cause, but it sits beside the move; decide whether an item first seen after the move onset should still be shown there.
- The rule ranks by snap share only. MIN at GB shows a questionable safety (0.991) while MIN quarterback Kyler Murray (0.862) is listed out. Ranking Out above Questionable before snap share would show the quarterback; not changed here because the rule was given.
- Beat-reporter posts are dated and fresh (`data/processed/beat_reporter_posts.parquet`) but their status is a text classification, so they are not rendered as a named player's status.
- "No captured news explains this move." still implies cause for games with no item; wording left as it was.
- `MAJORITY_SNAP_SHARE` is a definition (more than half), not a fitted value.

- Root review 2026-11-14 15:47 ET: as first built, four of six items were about the team the line moved toward (for example the line moved 3 points toward SEA beside SEA's quarterback listed out), which reads as a contradiction. The sentence now shows an item only for the team the line moved away from (board_move_reason.py, build_move_reasons, moved_away_from); otherwise the row keeps the no-captured-news sentence. Rendered after the change: CAR linebacker out (toward NO), MIA receiver doubtful (toward IND), GB linebacker doubtful (toward MIN), KC defensive back questionable (toward ATL), BUF defensive back questionable (toward NYJ); SEA and LAC rows say no captured news. The SEA move's actual cause (reported by the stored language read: LV traded its starting quarterback on deadline day) is not an injury designation and is not shown. Board tests 134 passed on this version.
