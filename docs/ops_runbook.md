# Tuesday ops runbook

The pool locks picks **Tuesday at 12:00 ET**. Everything below exists to
get a published, synchronized card in front of that deadline without
improvisation. One command does the whole sequence; the manual fallback
is here for the week the command breaks.

Written 2026-08-17 from the SPEC-3 Deliverable B rehearsal (the real run,
timings measured, not estimated).

## The one command

```
python -m nfl_ats weekly-run --season <SEASON> --week <WEEK>
```

Add `--dry-run` to print the plan without running anything, `--skip-ingest`
to reuse the snapshot already on disk, and `--refresh-player-data` to pick
up newer player/PBP snapshots instead of the ones the current production
manifests name.

It runs seven steps in order, fails closed at the first error, and prints
one JSON summary. **No publish happens unless the synchronization
assertion passes** — the run aborts with the failing step's name rather
than putting a half-built card on the public site.

## The Tuesday timeline

| Time (ET) | What happens |
|---|---|
| ~06:00–09:00 | The scheduled live odds captures land; Splash posts its lines. |
| any time after | Run `weekly-run`. Budget **15 minutes**; it measured 4m21s. |
| by 11:30 | Card published, Pages redeployed, picks entered. |
| **12:00** | **Pool locks.** Nothing after this counts. |

The compute is small enough that the deadline is never the binding
constraint — the risk is forgetting, not running out of time. Wednesday's
single line revision is irrelevant to us: our grade is the Tuesday opener.

## Measured wall-clock (rehearsal, 2026-08-17, `--skip-ingest`)

| Step | Command | Seconds |
|---|---|---|
| 1 | `ingest` | *skipped — see below* |
| 2 | `build-features` | 28.7 |
| 3 | `build-pbp-features` | 66.7 |
| 3 | `build-player-features` | 93.3 |
| 4 | `margin-backtest` | 71.1 |
| 5 | `margin-predict` | 1.0 |
| 6 | assert-synchronized (in-process) | 0.0 |
| 7 | `publish-predictions --with-board` | 0.1 |
| | **total** | **261.0 (4m21s)** |

Measured on the owner's Windows machine. Step 1 was skipped in the
rehearsal, so it is the one unmeasured step; it is a network-bound
nflverse download of schedules plus team stats, and the first live
Tuesday run should append its time here. Even a generous ingest leaves
the 15-minute budget intact.

The two feature builds and the backtest are the whole cost. If a future
change makes any step materially slower, `docs/performance.md` holds the
evaluator's budgets and regression rules.

## What the rehearsal proved

Run at 2026-08-17T20:01:44Z, `weekly-run --season 2026 --week 1
--skip-ingest`, exit 0, `published: true`:

- The walk-forward evaluation reproduced the frozen baseline **exactly**:
  `0.5204819277108433`, 1,080 of 2,075 — with the new `bias_*` columns
  present in the table. This was the open verification from
  `docs/opus_session_blockers.md` § 5 and it closed clean, confirming the
  frozen `player` profile genuinely ignores additive columns.
- Every pick, probability, and line on the published card is unchanged.
- The active model id moved `80e458040e48b926` → `f92d446d0ccb50dd`. This
  is expected and benign: the id is a hash that includes
  `feature_table_sha256` (`active_model.py`), and the table gained
  columns. **A new model id does not by itself mean the model changed** —
  compare `historical_evaluation` before concluding anything.

## Manual fallback

If `weekly-run` fails, run the same seven steps by hand. These are the
exact commands the rehearsal executed; the snapshot ids come from the
current production manifests and change only when you deliberately
refresh player data.

```
python -m nfl_ats ingest --start-season 2009 --end-season 2026 --stats-end-season 2025
python -m nfl_ats build-features
python -m nfl_ats build-pbp-features --snapshot <PBP_SNAPSHOT>
python -m nfl_ats build-player-features --player-snapshot <PLAYER_SNAPSHOT> --player-value-snapshot <VALUE_SNAPSHOT> --pbp-snapshot <PBP_SNAPSHOT>
python -m nfl_ats margin-backtest --features data\processed\game_features_player.parquet --feature-profile player
python -m nfl_ats margin-predict --season <SEASON> --week <WEEK> --features data\processed\game_features_player.parquet --feature-profile player
python -m nfl_ats publish-predictions --with-board
```

`weekly-run --dry-run` prints this list with the snapshot ids already
resolved — use it to fill in the placeholders rather than reading
manifests by hand.

Between steps 6 and 7, check the sync by hand: the active manifest must
read `SYNCHRONIZED` **and** its `weekly_forecast` season/week must equal
the ones you asked for. `margin-predict` leaves the previous week's
manifest in place when activation does not match, so `SYNCHRONIZED` alone
is not sufficient evidence — that is exactly why step 6 checks all three
conditions. **Do not run step 7 if that check fails.**

## Traps

- **Use `python -m nfl_ats`, never `nfl-ats.exe`.** A running dashboard
  holds a lock on the executable. Never plain `uv run`; `uv run --no-sync`
  is acceptable.
- **Playoff weeks are weeks 19–22** (`--week 19` wild card through
  `--week 22` Super Bowl). The pool needs 13 playoff picks; serving them
  is already supported (`docs/postseason_support.md`).
- **The push guard blocks master** when the tracked publication does not
  match the active model. The fix is `publish-predictions`, then commit,
  then push — `weekly-run` does this for you as step 7.
- **Republishing never rewrites a pick's ledger anchor.** The CLV ledger
  records each pick at the line it was *first* published at, so a re-run
  the same week is safe and does not launder a worse entry price.
