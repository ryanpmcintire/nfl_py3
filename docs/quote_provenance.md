# Quote availability provenance for LEAD-74 and LEAD-77

**Inferred rule:** a historical as-of timestamp and a local receipt timestamp answer different questions. Use the provider's returned snapshot clock for historical as-of ordering, and a verified post-response capture clock for locally observed availability. A bookmaker update clock is quote age, never independent proof that the quote had reached the archive or this project. The local archive establishes historical snapshot identity; it does not establish a provider guarantee against retrospective revisions. No later revision was demonstrated by this audit.

The frozen timing requirements remain `docs/lanes/lead74.md:19` and `docs/lanes/lead77.md:24`. This audit does not amend either protocol, close either signal, or authorize fitting. An absent ingestion field is not itself proof of leakage; calling a retrospectively retrieved snapshot locally ingested at its historical timestamp would be incorrect.

## Evidence and the three historical clocks

| Field | Meaning and evidence | Permitted use |
| --- | --- | --- |
| `requested_at_utc` | **Read:** the target time is sent as the historical API `date` parameter (`src/nfl_ats/odds_backfill.py:180`); the writer stores the target separately (`src/nfl_ats/odds_backfill.py:260`). | A requested historical as-of boundary, not a download/receipt timestamp or automatically an authentic pool deadline. |
| Response `timestamp`; manifest `snapshot_timestamp_utc`; row/manifest `observed_at_utc` | **Read:** the historical parser takes the response `timestamp` and passes it as `observed_at` into the quote parser (`src/nfl_ats/odds_backfill.py:212`). The writer uses that returned clock for the directory and manifest (`src/nfl_ats/odds_backfill.py:246`, `src/nfl_ats/market_data.py:293`, `src/nfl_ats/market_data.py:312`). | The provider-asserted snapshot as-of time. Require equality of these clocks, snapshot <= request, and snapshot <= actual prediction cutoff. It is not evidence of project ingestion at that historical instant. |
| Response `previous_timestamp` / `next_timestamp` | **Read:** copied into the capture and manifest (`src/nfl_ats/odds_backfill.py:225`, `src/nfl_ats/odds_backfill.py:262`). | Check that the selected archive entry brackets the request: previous < snapshot <= request < next where adjacent entries exist. These pointers are not quote issuance or receipt times. |
| `bookmaker_last_update_utc` / `market_last_update_utc` | **Read:** copied from the bookmaker/market `last_update` fields (`src/nfl_ats/market_data.py:181`, `src/nfl_ats/market_data.py:183`). | Provider-supplied update clocks for age calculations. Require nonmissing valid clocks when a feature uses age and reject clocks later than the selected snapshot. Never substitute them for observation/ingestion. |

**Read:** the endpoint is explicitly `/v4/historical/sports/.../odds` (`src/nfl_ats/odds_backfill.py:29`). This is a historical snapshot API, not a reconstruction from current prices. The request carries a historical date; the response supplies a separate archived snapshot clock. **Inferred:** this supports *provider-asserted point-in-time data*, conditional on the provider's archive semantics. It does not prove that the exact returned values were immutable or publicly observable then. A corrected archive entry could retain the same timestamp and pass every clock check here.

**Read:** raw response and quote hashes are recorded when the snapshot is written (`src/nfl_ats/market_data.py:305`). The historical parser also binds each row to the response digest (`src/nfl_ats/odds_backfill.py:221`). **Inferred:** matching hashes establish consistency of the locally retained files, not the date they first existed at the provider. Filesystem creation/modification times, quota counters, and historical directory names must not be used to fill that gap. The local source audit describes snapshot cadence but provides no revision guarantee (`docs/mkt09_licensing_audit.md:96`). No provider documentation was fetched in this task.

## Availability rule by source

| Source | Availability clock under the current strict protocol | Validator treatment |
| --- | --- | --- |
| The Odds API historical backfill, NFL or CFB | No independently proven availability clock is present. `snapshot_timestamp_utc` is the appropriate *conditional historical as-of* clock. Project retrieval occurred later and is not recorded by this writer. | Report internally consistent as-of pairs separately; `availability_proven` remains false. A provider contract establishing contemporaneous archive capture and treatment of corrections, or contemporaneous retained evidence, is needed to certify exact-value historical availability. |
| The Odds API live full-game capture | Manifest/row `observed_at_utc`: **Read:** `datetime.now(UTC)` occurs after `fetch_odds_api_from_environment` returns (`src/nfl_ats/cli_commands/market.py:42`). | Accept the known live request shape, intact response and quote hashes, matching clocks, and observation <= cutoff and < kickoff. |
| Bovada public live capture | Manifest/row `observed_at_utc`: **Read:** recorded immediately after `response.read()` (`scripts/capture_bovada_private.py:173`). Update clocks are deliberately missing (`scripts/capture_bovada_private.py:149`); timestamp semantics are recorded (`scripts/capture_bovada_private.py:213`). | Accept this post-response capture as a conservative bound on local receipt. Missing bookmaker time does not invalidate capture availability, but prevents a bookmaker-age feature. |
| ESPN pickcenter self-capture | No acceptable completion clock from its current manifest. **Read:** `observed` is assigned before the scoreboard and per-event summary requests (`scripts/capture_espn_pickcenter.py:158`); that same clock reaches the writer (`scripts/capture_espn_pickcenter.py:199`). | Do not certify predeadline receipt using this start clock. This is a source-specific limitation; no writer was changed. |
| Other captures / unknown schemas | Require a separately traced receipt/completion timestamp and source rule. | Fail closed; a timestamp-shaped directory alone is insufficient. |

**Read:** the named Bovada example records `2026-09-29T20:37:30.945893+00:00` (`data/market/raw/20260929T203730Z/manifest.json:14`), while its directory stops at whole seconds. **Inferred:** use the full manifest timestamp; treating the directory as `20:37:30.000000` would admit that quote almost a second early. Self-capture clocks are conservative receipt bounds, not necessarily the earliest time a bookmaker published its price; they assume the capture host clock is correct.

## Validator and pair definition

`scripts/quote_provenance_check.py` reads no game outcomes, feature artifacts, predictions, or fitted models. It verifies response/Parquet hashes and byte/row counts, source and snapshot identity, returned/manifest/row clocks, adjacent historical snapshot order, bookmaker/market updates, and pre-kickoff ordering. It reads one snapshot at a time, uses at most two Arrow compute threads and one I/O thread, and writes only aggregate JSON to stdout. `validate_frame` applies the row rule; use `check_snapshot` to include file-integrity verification.

A counted pair is exactly one HOME and one AWAY spread price at one snapshot/event/book and opposite spread lines, with finite American prices of magnitude at least 100. Duplicates do not qualify. All markets remain in the quote-row counts. These are source-observation pairs, not independent games, REG-only games, Tuesday/deadline pairs, or fitted research samples. Seasons come from manifest requests, matching the 8,104-manifest census; they are not inferred from eventual game outcomes. NFL and CFB are reported separately.

The default cutoff is each historical request time or each live receipt time. This establishes source consistency, not pool-deadline eligibility. `--deadline` supplies an explicit inclusive timezone-aware cutoff for a selected snapshot; downstream callers must use each game's actual prediction timestamp and handle strict-before conventions if their protocol requires them. No timestamp tolerance is added. LEAD-74 must still join both dated observations from the same book at the same spread to authentic frozen openers. LEAD-77 must still keep every response-training label and quote before that game's deadline, exclude unsupported ages, and certify its folds.

```bash
UV_NO_CACHE=1 .tools/uv.exe run --no-sync python scripts/quote_provenance_check.py --include-snapshot data/market/raw/20260929T203730Z
```

The process exits nonzero for corrupt/unsupported snapshots, missing required columns, or an empty selection. Exit zero means the audit ran without structural errors; inspect `availability_proven_pairs` for actual availability acceptance. Historical revision uncertainty is reported explicitly, rather than silently accepting the archive or reporting it as corrupt.

## Measured audit

**Measured:** the command above completed with exit 0. All **8,104 historical snapshots** passed identity, raw-response/quote-file hash, byte/row-count, and timestamp consistency checks; all **5,031,980 quote rows** had internally consistent source clocks. There were zero structural errors. Of those rows, **33,730** were at/after kickoff and therefore excluded from pregame acceptance.

| Request season | NFL snapshots | NFL complete spread pairs | NFL as-of-consistent pregame pairs | CFB snapshots | CFB as-of-consistent pregame pairs | Historical availability-proven pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2023 | 2,548 | 848,407 | 846,625 | 80 | 47,916 | 0 |
| 2024 | 2,548 | 554,214 | 552,995 | 245 | 66,482 | 0 |
| 2025 | 2,548 | 658,813 | 657,254 | 132 | 49,150 | 0 |
| Total | 7,644 | 2,061,434 | 2,056,874 | 457 | 163,548 | 0 |

**Measured:** the remaining **3** manifests are Super Bowl winner markets, containing **800** rows and no spread pairs. They explain why NFL plus CFB snapshot counts alone do not equal 8,104. CFB has **168,132** complete pairs before the pregame filter. Across both leagues, **2,229,566** complete spread pairs become **2,220,422** as-of-consistent pregame pairs; **9,144** pairs fail pregame timing. Every as-of-consistent historical pair has a bookmaker clock. The zero in the last column reflects the source-evidence rule, not a discovered wrong timestamp or demonstrated revision.

**Measured:** the named Bovada snapshot passes all integrity checks: **64** quote rows, **16/16** complete spread pairs as-of consistent and availability proven at its receipt time, and **0/16** pairs with a bookmaker clock. This is not a claim that it precedes a particular game's pool deadline.

These are exact archive census counts, with no sampling intervals. Outcome looks consumed: **0**. Effect intervals, `probability_positive`, IS/OOS metrics, and their gaps are not estimated. The prior LEAD-77 event-season inventory uses a different season basis, so its CFB per-season file counts need not equal the request-season counts here.

**Inferred handoff:** LEAD-74/77 can cite the clock definitions and measured as-of counts immediately. Under their unchanged strict issuance/ingestion requirement, this audit certifies **0 historical Tuesday/deadline pairs** because neither endpoint has independently proven historical availability. It has not attempted that temporal join or counted eligible REG games. Resolving the provider's contemporaneous capture/correction contract is the remaining source question; do not represent missing local evidence as proof that revisions occurred, and do not silently treat the as-of count as final protocol eligibility. No weak-signal or rotation record is appropriate for this metadata audit.

