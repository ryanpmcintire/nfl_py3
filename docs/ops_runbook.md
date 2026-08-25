# Tuesday ops runbook

The pool locks picks **Tuesday at 12:00 ET**. Everything below exists to
get a published, synchronized card in front of that deadline without
improvisation. One command does the whole sequence; the manual fallback
is here for the week the command breaks.

Written 2026-08-17 from the SPEC-3 Deliverable B rehearsal (the real run,
timings measured, not estimated).

## The one command

```
python -m nfl_ats weekly-run --season <SEASON> --week <WEEK> --record-decisions
```

Add `--dry-run` to print the plan without running anything, `--skip-ingest`
to reuse the snapshot already on disk, `--refresh-player-data` to pick
up newer player/PBP snapshots instead of the ones the current production
manifests name, and `--skip-prospective` to leave out the research-evidence
tail (steps 8-11).

**`--record-decisions` is required for the real Tuesday lock and is not the
default.** Without it, `weekly-run` still publishes the card (step 7) and
still builds and scores the challenger's own card (steps 8, 9, 11) — it just
does not append anything to either the paper-decision ledger or the
challenger ledger. This is deliberate, since 2026-08-18: an ordinary
`weekly-run` used for testing or a rehearsal must never be able to reach a
real ledger by accident (`docs/prospective_evidence.md`, "Known
divergence" — the exact incident this closes). Even with the flag passed,
both recorders separately refuse to write if the week being recorded is not
close to its own kickoff (`nfl_ats.clv.RECORDING_LOCK_WINDOW`, 7 days), so
running this on the real Tuesday, in the real lock week, is the only way it
actually records.

It runs **eleven** steps in order and prints one JSON summary. Steps 1-7 are
the card path and fail closed at the first error: **no publish happens unless
the synchronization assertion passes** — the run aborts with the failing
step's name rather than putting a half-built card on the public site.

Steps 8-11 collect prospective 2026 evidence (POL-10) and are **optional**:
they run after the publish, a failure is reported in `optional_failures` and on
stderr, and it does not abort the run. The card is the deliverable; losing a
week of research evidence must not take it down. They still have to run every
Tuesday before the lock, because a pick recorded after kickoff is refused
outright — see `docs/prospective_evidence.md`.

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
| | **total (card path)** | **261.0 (4m21s)** |
| 8 | `build-learned-availability-features` (weak stack) | ~98 |
| 9 | `margin-predict` (challenger) | ~1 |
| 10 | `prospective-record` | <1 |
| 11 | `prospective-score` | ~2 |
| | **total with the evidence tail** | **~6m** |

Steps 8-11 were added 2026-08-17 and are estimated from the weak-stack table's
own build manifest (97.9s) plus measured runs of steps 9-11; the first live
Tuesday should replace these with real numbers. Even so the 15-minute budget
holds with room, and the tail runs after the card is already published.

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
  `docs/archive/opus_session_blockers.md` § 5 and it closed clean, confirming the
  frozen `player` profile genuinely ignores additive columns.
- Every pick, probability, and line on the published card is unchanged.
- The active model id moved `80e458040e48b926` → `f92d446d0ccb50dd`. This
  is expected and benign: the id is a hash that includes
  `feature_table_sha256` (`active_model.py`), and the table gained
  columns. **A new model id does not by itself mean the model changed** —
  compare `historical_evaluation` before concluding anything.

## Manual fallback

If `weekly-run` fails, run the same steps by hand. **Corrected 2026-08-18:**
this section previously hardcoded the `player` feature profile for the card
path (steps 4-5). Production has run **`weak_stack`** since the 2026-08-17
promotion (`68b4dc0`), and `weekly-run`'s card path now builds and scores
whatever profile `artifacts/active_ats_model.json` names
(`active_card_profile`/`CARD_PATH_TABLES` in `src/nfl_ats/weekly.py`) instead
of a hardcoded one — so the manual fallback must match. The commands below
are the `weak_stack` sequence `weekly-run --dry-run` prints today; if a
future promotion moves the active profile again, `--dry-run` is the source
of truth, not this file. The snapshot ids come from the current production
manifests and change only when you deliberately refresh player data.

```
python -m nfl_ats ingest --start-season 2009 --end-season 2026 --stats-end-season 2025
python -m nfl_ats build-features
python -m nfl_ats build-pbp-features --snapshot <PBP_SNAPSHOT>
python -m nfl_ats build-player-features --player-snapshot <PLAYER_SNAPSHOT> --player-value-snapshot <VALUE_SNAPSHOT> --pbp-snapshot <PBP_SNAPSHOT>
python -m nfl_ats build-learned-availability-features --features data\processed\game_features_pbp.parquet --destination data\processed\game_features_weak_stack.parquet --rates-destination data\processed\weak_stack_availability_rates.parquet --evaluation-destination data\processed\weak_stack_availability_evaluation.csv --player-snapshot <PLAYER_SNAPSHOT> --player-value-snapshot <VALUE_SNAPSHOT> --pbp-snapshot <PBP_SNAPSHOT>
python -m nfl_ats margin-backtest --features data\processed\game_features_weak_stack.parquet --feature-profile weak_stack
python -m nfl_ats margin-predict --season <SEASON> --week <WEEK> --features data\processed\game_features_weak_stack.parquet --feature-profile weak_stack
# Check by hand before publishing: active manifest reads SYNCHRONIZED, its
# weekly_forecast season/week match <SEASON>/<WEEK>, AND feature_profile = weak_stack.
python -m nfl_ats publish-predictions --with-board --record-decisions
# steps 8-11, the prospective-evidence tail (safe to run late, never before 7).
# This still rebuilds and scores the weak_stack table as the registered
# "challenger" (mod07_weak_signal_stack) even though it is now also the
# active card-path profile above -- a known, documented quirk (see
# docs/week1_readiness.md, "owner decision #1"), not a mistake in this file.
python -m nfl_ats build-learned-availability-features --features data\processed\game_features_pbp.parquet --destination data\processed\game_features_weak_stack.parquet --rates-destination data\processed\weak_stack_availability_rates.parquet --evaluation-destination data\processed\weak_stack_availability_evaluation.csv --player-snapshot <PLAYER_SNAPSHOT> --player-value-snapshot <VALUE_SNAPSHOT> --pbp-snapshot <PBP_SNAPSHOT>
python -m nfl_ats margin-predict --season <SEASON> --week <WEEK> --features data\processed\game_features_weak_stack.parquet --feature-profile weak_stack
python -m nfl_ats prospective-record --challenger mod07_weak_signal_stack --season <SEASON> --week <WEEK>
python -m nfl_ats prospective-score
```

`weekly-run --dry-run` prints this list with the snapshot ids already
resolved — use it to fill in the placeholders rather than reading
manifests by hand.

`publish-predictions --record-decisions` is what actually appends this
card's picks to the paper-decision ledger; without the flag it publishes the
card but records nothing (default, since 2026-08-18). `prospective-record`
(step 10) has no such flag — it always attempts to record when invoked — but
it and `publish-predictions --record-decisions` both refuse outright if this
week's earliest kickoff is more than `RECORDING_LOCK_WINDOW` (7 days) away
from the moment you run them, so running either one outside the real lock
week does not reach the ledger even if you remember the flag.

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
- **Never pipe a native command's stderr with `2>&1` in a scheduled
  PowerShell script.** This already cost one capture. Under Windows
  PowerShell 5.1 each stderr line from an `.exe` becomes a
  `NativeCommandError` record, and with `$ErrorActionPreference = 'Stop'`
  that is a *terminating* error — so the script aborts on output that was
  never an error. `scripts/odds_capture.ps1` died exactly this way on
  2026-08-16 (logged in `data/market/capture_log.txt`) when plain `uv run`
  printed `Building nfl-ats @ file:///...` to stderr after a `src/` change.
  A capture that had exited 0 was thrown away. The fix, verified against a
  command that writes to stderr and exits 0: pass `--no-sync`, redirect
  stderr to its own file rather than into the pipeline, and relax
  `$ErrorActionPreference` around the call. `$LASTEXITCODE` is the only
  trustworthy success signal for a native executable.
- **Playoff weeks are weeks 19–22** (`--week 19` wild card through
  `--week 22` Super Bowl). The pool needs 13 playoff picks; serving them
  is already supported (`docs/postseason_support.md`).
- **The push guard blocks master** when the tracked publication does not
  match the active model. The fix is `publish-predictions`, then commit,
  then push — `weekly-run` does this for you as step 7.
- **Republishing never rewrites a pick's ledger anchor.** The CLV ledger
  records each pick at the line it was *first* published at, so a re-run
  the same week is safe and does not launder a worse entry price. The flip
  side: if you publish a week EARLY, the ledger scores that early card, not
  the one you enter at the lock. Publish on Tuesday. (2026 Week 1 already
  carries rehearsal rows from 2026-08-17 — see the "known divergence" section
  of `docs/prospective_evidence.md` for the two clean options.)
- **The week's Best Pick is only nominated while every game is still ahead.**
  Once any game of the week has kicked off, that week gets no Best Pick at all
  and the flag stays False forever. Another reason to run on Tuesday, not
  Thursday.
