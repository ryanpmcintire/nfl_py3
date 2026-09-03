# Timing-policy audit (MKT-08)

## Status

MKT-08 remains open. **[read: `ROADMAP.md`, MKT-08]** Its definition asks for
a comparison of fixed weekly timestamps and news-triggered updates. The fixed
side exists, but the source-triggered side does not yet have structured,
observed decision rows.

- **[read: `registry/experiments/`]** Five retained experiment artifacts cover
  fixed checkpoint families (`observed-movement-channel`,
  `movement-attribution`, and `reliability-movement`).
- **[read: `scripts/capture_scheduler.py`]** Every current decision job is
  dispatched by weekday and Eastern time. The Tuesday lock is dependency-gated
  on the opener capture; refresh jobs remain clock-driven, including the
  post-inactives passes.
- **[read: `src/nfl_ats/pick_refresh.py`]** The played revision ledger records
  a run ID and revision time, and since 2026-09-03 also structured
  `trigger_type` (`clock_dispatch` default, `news_event` reserved),
  `trigger_source` (scheduler job id, via `--trigger-source`), and
  `trigger_observed_at_utc` (defaults to plan time), with legacy-row
  backfill. Its free-text `reason` remains narration alongside the
  machine-checkable fields, not instead of them.
- **[measured: `python scripts/timing_policy_audit.py`, 2026-09-02]** The local
  paper-decision and five refresh ledgers contain zero rows before the 2026
  Week 1 lock. There is therefore no prospective fixed-versus-triggered
  comparison to score or reconcile yet.

This is an evidence-availability statement, not an ATS verdict. No experiment
was run and no model or registry decision changed.

## Read-only audit command

```powershell
.\.tools\uv.exe run --no-sync python scripts\timing_policy_audit.py
```

The command emits JSON to stdout and performs no capture, forecast, ledger
append, artifact write, or network request. It reads:

- the decision schedule from `scripts/capture_scheduler.py`;
- immutable capture directories for market, injury news/reports, weather,
  referee assignments, and inactives;
- `artifacts/clv_ledger/decisions.parquet`;
- the played, injury-signal, injury-report, inactives, and referee refresh
  ledgers under `artifacts/prospective/`; and
- the three existing fixed-timestamp registry experiment families.

For each non-empty refresh ledger it fails closed on missing required columns
or invalid timestamps, then reports these decision-time violations:

1. the refresh predates the frozen Tuesday decision;
2. the refresh is at or after the row's explicit deadline (or kickoff when no
   narrower deadline is recorded);
3. the source capture/fetch occurred after the refresh; or
4. no matching original decision is available.

The process exits nonzero when any violation is present. Missing prospective
rows are reported as an evidence gap, not treated as a process failure.

## Measured inventory on 2026-09-02

**[measured: `python scripts/timing_policy_audit.py`]** The current policy has
11 clock-dispatched decision jobs: one Tuesday lock, three general late-week
passes, and seven capture-followup inactives passes. Exactly one is explicitly
dependency-gated (`weekly_lock` requires `odds_tue_open`); none is an
event-triggered dispatcher.

**[measured: the same command]** Existing immutable source evidence includes
937 timestamped market directories, three paused legacy NFL.com injury-report
directories, one injury-news directory, two referee-assignment directories,
and eight weather-archive manifests. The licensed injury-report and inactives
roots have no captured snapshot yet. These counts describe retained inputs;
they do not imply that an update was triggered by any one input.

## Exact completion boundary

MKT-08 can close only after the audit observes valid rows on both sides of
the comparison. The schema half is done on the played path (all three trigger
fields land on every appended pick-revision row); what is still missing is
any real row at all — the ledgers are empty until the 2026-09-08 lock — and,
on the other side, a news-driven update path to compare against. The trigger
time must be no later than the refresh, and the refresh must remain before
the game's decision deadline. Until then, the fixed-checkpoint evidence is
real, but calling it a fixed-versus-news-triggered comparison would overstate
what the repository has measured.
