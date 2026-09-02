# Sportradar live NFL injury-report capture

Status: **implemented and credential-gated; no live capture yet** (2026-09-02).
ROADMAP item: PER-03.

## Source and policy

**Read** from Sportradar's current
[NFL Weekly Injuries reference](https://developer.sportradar.com/football/reference/nfl-weekly-injuries):
the v7 endpoint returns a requested season/week's team and player blocks,
including injury, practice status, game status, status date, and stable provider
IDs; its published cache TTL is four hours. **Read** from Sportradar's
[account-maintenance guide](https://developer.sportradar.com/football/docs/football-ig-account-maintenance):
trial access uses real-world data, requires an API key, and lasts 30 days.

**Read** from the provider's terms dated 2026-08-05: a free trial is for
non-commercial internal testing and evaluation; the customer's order form and
provider terms control any production use. The resulting tracked policy is
`sportradar_nfl_injuries` in `config/source_policies.json`: YELLOW, private raw
retention only, no raw redistribution, aggregates-only publication, and a
required credential. No key is stored in Git or written to a manifest.

This replaces neither the historical nflverse snapshot nor already-retained
NFL.com files. **Read** from `config/source_policies.json`: NFL.com remains RED
and new acquisition remains prohibited. **Read** from
`scripts/capture_scheduler.py`: the four old `injuries_*` jobs remain paused.

## Capture contract

`scripts/capture_sportradar_injuries.py`:

1. checks the tracked policy, private destination, and `SPORTRADAR_API_KEY`
   before creating output or issuing a request;
2. resolves the next regular-season week and required teams from the newest
   local schedule snapshot;
3. requests the documented v7 endpoint with the key in the `x-api-key` header;
4. rejects malformed, empty, stale/future, wrong-season/week, duplicate, or
   incomplete-slate responses;
5. writes `source.json` verbatim and canonical `injuries.parquet` under
   `data/raw/sportradar_injuries/<UTC>/`, then writes `manifest.json` last with
   the source URL, capture time, provider generation time, coverage, byte
   counts, and SHA-256 hashes.

The canonical availability timestamp is the immutable capture time, not the
provider's `status_date`. `load_for_decision()` considers only a complete
snapshot captured by the supplied decision time, verifies every manifest hash,
and rejects rows whose availability differs from that capture. A later revision
therefore cannot appear in an earlier information set. This boundary is not
wired to the active model.

Run manually after supplying a provider-authorized key:

```powershell
$env:SPORTRADAR_API_KEY = '<provider key>'
.\.tools\uv.exe run --no-sync python scripts/capture_sportradar_injuries.py
```

## Scheduler cadence

**Read** from the provider reference: the feed has a four-hour cache TTL and
updates according to the day of a game. **Read** from the existing weekly
capture plan: Wednesday/Thursday/Friday 17:30 ET plus Saturday 10:00 ET are the
revision points already reserved for practice/final reports. Four parallel
`sportradar_injuries_*` jobs now use those windows and a five-hour dedupe guard.
They enable only when the scheduler process has `SPORTRADAR_API_KEY`; without a
credential they are DISABLED, not falsely MISSED.

## Verification and remaining blocker

**Measured 2026-09-02** with
`pytest -q tests/test_sportradar_injury_capture.py tests/test_source_policy.py`:
14 tests passed. The fixtures cover exact season/week/schema parsing, required
team coverage, stale/malformed failures, pre-I/O credential failure, immutable
hashes, exclusion of a later revision at an earlier decision timestamp, hash
tampering, scheduler credential gating, and continued NFL.com pause.

**Measured 2026-09-02** by enumerating environment-variable names: no
Sportradar credential is configured. Consequently no provider request was made
and no real current response exists to prove provider-shape and slate-coverage
compatibility. PER-03 must remain open until an authorized trial/production key
is supplied and one current capture completes. The honest closure evidence is
`data/raw/sportradar_injuries/<UTC>/manifest.json` with `status: complete`, not
fixture success.

## Historical cache reconciliation

**Measured 2026-09-02** with `scripts/pfr_bulk_date_fetch.py --max-fetches 0`:
the 2022-2025 scope is 4,361/4,361 cached and the separately requested
2014-2021 scope is 9,080/9,080 cached; both report zero rows needing fetch.
Those completed news-wire caches do not substitute for the missing live
practice/game-status revision stream.
