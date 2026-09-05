# Source outage and degraded-mode policy (ENG-14)

Written 2026-09-04. Every claim is labeled per `AGENTS.md`: **measured** (a
command run this session, command given), **read** (a file opened this session,
path given), or **inferred** (reasoning, labeled as such).

Implementation: `src/nfl_ats/source_freshness_policy.py`.
Tests: `tests/test_source_freshness_policy.py` (**measured**, 59 passed).

## Why the module is not called `source_policy`

**Read**, `src/nfl_ats/source_policy.py`: that module already exists and answers
a *legal* question — may we acquire, retain, and republish this source at all
(risk colour, terms review date, quota), backed by `config/source_policies.json`
(MKT-09). This module answers the *operational* question that sits on top of it:
given that a source is allowed, is the snapshot we hold fresh enough to publish
a card from, and if not, does the card fall back or refuse? Two registries, two
failure modes, deliberately not merged. `docs/source_freshness_policy.md` and
`tests/test_source_freshness_policy.py` follow the module's name for the same
reason (`tests/test_source_policy.py` was also already taken).

## Card states and neutral source states

| State | Meaning | Effect on the card |
| --- | --- | --- |
| `complete` | Every observed source is inside its freshness budget. | Publish normally. |
| `degraded` | At least one source is absent or stale and the card fell back onto that source's documented behaviour. | Publish, and say so on the card. |
| `blocked` | At least one **fail-closed** source breached. | Refuse to publish, naming the source and the rule. |
| `not_due` | The week's first inactive report is not due until 90 minutes before its first kickoff. | Neutral grey; excluded from degraded/blocked roll-up. |
| `not_configured` | The optional Sportradar capture has no `SPORTRADAR_API_KEY`. | Neutral grey; the league injury report stands in. |

**Read**, `src/nfl_ats/source_freshness_policy.py::_first_week_kickoff`: the
inactives window uses the scheduler's America/New_York clock and local schedule
table, retaining the whole current NFL week (including games already played).
Before the season it uses the next scheduled slate. Missing/unreadable schedules
do not grant an exemption, and future schedule snapshots are ignored. Missing
inactives become degraded at the first kickoff minus 90 minutes; the existing
scheduled capture cadence still determines the stale-copy budget.

**Read**, `src/nfl_ats/source_freshness_policy.py::_evaluate_one`: neutral states
never suppress future-dated observations or fail-closed breaches. Existing stale
copies retain their budget checks, except for the optional unconfigured feed.
Setting the credential restores normal Sportradar budget checks without a code
change. A nonempty report whose only gaps are neutral rolls up to complete.

**Read**, `src/nfl_ats/source_freshness_policy.py::SourceState.to_dict` and
`src/nfl_ats/board_content.py::_load_source_policy_view`: per-source states and
the inactive report's due instant are persisted in the existing source-policy
block. Readers derive the weekday/part-of-day wording from that saved instant,
never today's environment; older blocks without it remain readable without
inventing a weekday. The public legend explains grey as not due yet or not set up.

The roll-up is worst-wins. A source with **no observation at all** is
*unobserved*, not absent: it never appears in the per-source rows and never
contributes to the roll-up, because "we did not look" is not evidence about a
source. An empty report is `degraded`, never `complete` — nothing was looked
at, so nothing can be claimed.

## Where the budgets come from

Nothing here is a chosen number. Each source declares the
`scripts/capture_scheduler.py` `SCHEDULE` jobs that feed it as
`(day, "HH:MM", grace_minutes)` triples **read** from that file, and
`_derive_budget` computes

```
budget = (longest gap between consecutive scheduled captures, over one weekly
          cycle, including the Sunday -> Monday wrap)
       + (the grace of the job that CLOSES that longest gap)
```

That is exactly the oldest a HEALTHY source can be at an arbitrary evaluation
instant (**inferred**, from the scheduler's own window semantics: a job is not
late until its grace window has also closed). Anything older means a scheduled
capture did not land. A source with a single weekly job therefore gets a 7-day
budget plus its own grace — correct, and loose on purpose: a tighter number
would false-alarm at the end of every cycle, and a false `degraded` on the card
is as corrosive as a missed real one.

`tests/test_source_freshness_policy.py::test_every_budget_matches_the_capture_schedule_arithmetic`
pins each number against hand arithmetic, so a cadence edit in
`scripts/capture_scheduler.py` fails the suite instead of silently moving a
budget.

### One budget is tightened, and only downward

`player_arrests`: the cadence would allow 10,080 + 90 = 10,170 minutes, but
**read**, `src/nfl_ats/player_arrests_back_side_overlay.py:44`,
`MAX_SNAPSHOT_AGE = pd.Timedelta(hours=36)` is *already enforced fail-closed at
publish*. The policy layer must never be looser than a gate production already
applies, so the budget is that constant — imported, never re-declared, so the
two cannot drift.

## The table

Budgets in minutes. `states` reads *absent / stale / future-dated*.

| Source | `SCHEDULE` jobs | recurrence + grace = **budget** | states |
| --- | --- | --- | --- |
| `odds_opener` | `odds_tue_open` | 10080 + 180 = **10260** (7d 3h) | degraded / degraded / blocked |
| `odds_refresh` | `odds_tue_open`, `odds_thu_tnf`, `odds_sat`, `odds_sun_close`, `odds_sun_late`, `odds_mon_mnf` | 3420 + 90 = **3510** (2d 10h 30m) | degraded / degraded / blocked |
| `injuries_nflverse` | `weekly_lock` (runs `weekly-run` step 1 `ingest`) | 10080 + 120 = **10200** (7d 2h) | degraded / degraded / degraded |
| `injuries_sportradar` | `sportradar_injuries_wed`/`_thu`/`_fri`/`_sat` | 6210 + 240 = **6450** (4d 11h 30m) | degraded / degraded / degraded |
| `inactives` | `inactives_thu_afternoon_early`/`_late`, `inactives_thu_primetime`, `inactives_sat_early`/`_late`, `inactives_sun_early`/`_late` | 5575 + 15 = **5590** (3d 21h 10m) | degraded / degraded / degraded |
| `projected_lineups` | `lineups_mon` … `lineups_sun` | 1440 + 180 = **1620** (1d 3h) | degraded / degraded / degraded |
| `referee_assignments` | `referee_assignments_wed` | 10080 + 240 = **10320** (7d 4h) | degraded / degraded / degraded |
| `player_arrests` | `player_arrests_tue` | 10080 + 90, tightened to **2160** (1d 12h) | **blocked / blocked / blocked** |
| `pfr_transactions` | `pfr_transactions_wed`, `pfr_transactions_sat` | 5760 + 120 = **5880** (4d 2h) | degraded / degraded / degraded |
| `airnow_weather` | `airnow_tue_checkpoint` | 10080 + 15 = **10095** (7d 0h 15m) | degraded / degraded / degraded |

### The allowed fallbacks

| Source | Fallback when degraded | Behaviour already lives in |
| --- | --- | --- |
| `odds_opener` | Publish on the newest opener snapshot on disk and disclose that line freshness is unverified. | `nfl_ats.prediction_safety._prospective_checks` (warning path) |
| `odds_refresh` | The frozen Tuesday opener quote stands; no late-week line refresh, no CLV for the week. | `nfl_ats.clv`, `nfl_ats.pick_refresh` |
| `injuries_nflverse` | Reuse the previous weekly snapshot; availability features carry last week's report rather than an invented neutral value. | `nfl_ats.weekly` step 1, `nfl_ats.players` |
| `injuries_sportradar` | Dormant without `SPORTRADAR_API_KEY`; the nflverse report stands, no late-week revision. | `scripts/capture_sportradar_injuries.py` |
| `inactives` | `SOURCE_NO_SNAPSHOT`: the Tuesday pick stands, the row is tagged, and a zero-row snapshot counts as "no report yet", never "nobody is out". | `nfl_ats.inactives_refresh_overlay` |
| `projected_lineups` | The This Week lineup panel is omitted; a missing feed is displayed as unavailable, never estimated. | `nfl_ats.lineup_view` (`docs/projected_lineups.md`) |
| `referee_assignments` | Documented no-op: zero crew tilt, incumbent Tuesday pick stands, row tagged. | `nfl_ats.crew_tilt_refresh_overlay` |
| `player_arrests` | **None.** Fail-closed; there is no public fail-open switch. | `load_latest_complete_arrest_snapshot`, `card_view.resolve_player_arrests_overlay(require_fresh=True)`, `weekly-run` step 7 |
| `pfr_transactions` | Transaction-wire features fall back to their neutral (no-news) value. | `nfl_ats.transaction_wire_features` |
| `airnow_weather` | Environmental features fall back to their neutral value. | `nfl_ats.forecast_weather_features` |

## Which publish paths are blockable, and why the rest are only degraded

This was the deliberate design constraint, not an omission. **The policy layer
invents no new fail-closed behaviour.** Turning a currently-permitted publish
into a blocked one is an owner decision, not a side effect of writing a policy
table, so `degraded` is the strongest state a source can reach unless its
existing consumer *already* refuses.

**Newly nameable as `blocked` (all pre-existing behaviour):**

* `player_arrests`, on absence / staleness / future-dating. **Read**,
  `src/nfl_ats/publishing.py` → `_publication_context(require_fresh_arrest_overlay=True)`
  → `card_view.resolve_player_arrests_overlay(require_fresh=True)`, which
  re-raises; and **read**, `src/nfl_ats/weekly.py:524-529`, `weekly-run` step 7
  `ingest-player-arrests` is fatal ("publication is refused when this source
  refresh fails"). The policy layer only *names* that rule in one place.
* `odds_opener` / `odds_refresh`, on **future-dating only**. **Read**,
  `src/nfl_ats/prediction_safety.py:314-319`: an absent market timestamp is a
  WARNING, but a quote observed after freeze time or kickoff `_fail`s the
  release-blocking `market_timing` check. That asymmetry is reproduced exactly.

**Degraded-only, deliberately:** every other source. Each already has a
documented absent/stale fallback in the module that consumes it (table above),
and each of those modules currently publishes a card without the source. Making
any of them blocking would take the card down for a research feed that the pool
does not require, on a pool where 285 cards must be submitted either way.

`tests/test_source_freshness_policy.py::test_no_currently_permitted_publish_path_becomes_newly_blockable`
pins the two sets, so widening them requires editing an explicit test.

## Where the state is surfaced

1. **The published Markdown card** (`CURRENT_PREDICTIONS.md`, written by
   `nfl_ats.publishing.publish_active_predictions`) carries one line under the
   table:

   ```
   **Source freshness: DEGRADED.** Complete: player_arrests, odds_opener.
   Degraded (allowed fallback): inactives, referee_assignments. Blocked: none.
   Budgets, fallbacks and the three states: `docs/source_freshness_policy.md`.
   ```

2. **The publish metadata** returned by `publish-predictions` gains a
   `source_policy` block: overall `state`, `evaluated_at_utc`, the three
   bucket lists, `unobserved`, `blocking_reasons`, and a per-source row with
   `state`, `reason`, `age_minutes`, `budget_minutes`, `fallback`, `detail`.

3. **`weekly-run`** lifts that block to the top of its run summary
   (`summary["source_policy"]`). It copies what the publish step reported and
   never re-evaluates; when the publish step reported nothing, the key is
   absent rather than fabricated.

4. **A blocked publish raises** `SourceFreshnessError`, naming the source, the
   breach, the budget, and the module that enforces it:

   ```
   publication refused by source policy -- player_arrests: no snapshot present
   (budget 2160 min) (rule: budget 2160 min, fail-closed, enforced by
   nfl_ats.player_arrests_back_side_overlay.load_latest_complete_arrest_snapshot; ...)
   ```

5. **The public site** (ENG-34, 2026-09-04): a "SOURCES" panel on the This
   Week page, directly beneath the board's own card header, one dot-leader
   line per source (`source_id`, state, `as-of <instant>`), the worst-wins
   card state in the panel's header line, and a one-line plain-English
   legend. **Read**, `src/nfl_ats/board_content.py`'s
   `_load_source_policy_view`: it reads `metadata["source_policy"]` off the
   synchronized forecast's own `metadata.json` (`artifacts/active_ats_model.json`
   → `weekly_forecast` → `metadata.json`), the exact shape item 2 above
   describes. **Measured this session**, no forecast's `metadata.json` in
   this repo carries that key yet — only `publish_active_predictions`'s
   in-memory result dict does (item 2), and nothing persists it to disk — so
   the panel currently renders an explicit `NOT RECORDED` card state for
   every published forecast, including the live Week 1 artifact, rather than
   inventing a real one. Wiring `publishing.py` to write the block into the
   forecast's `metadata.json` (or into the `explanations.json` ENG-12 already
   writes unconditionally beside it) would make the panel show real states
   with no further site-side change; see `nfl_ats.board_content.SourcePolicyView`
   and `tests/test_board_content.py`'s `_load_source_policy_view` tests. The
   board assistant (`nfl_ats.board_assistant`) answers "were the sources
   complete this week" and siblings from the same view, in both the Python
   reference engine and the inline-JS port (`tests/test_assistant_js_parity.py`).

## How freshness is observed

`observe_from_disk` reads the newest UTC-stamped snapshot directory **name**,
not filesystem mtime — the same rule as
`scripts/capture_scheduler.py.newest_snapshot_age_minutes` (**read**): the name
is the capture instant the project treats as authoritative, and mtime moves for
reasons unrelated to capture time (a backup restore, a file copy, an antivirus
touch). `projected_lineups` is the one source read from a JSON field
(`generated_at` in `artifacts/lineups/current/lineups.json`), because that
artifact is replaced in place rather than accumulated
(**read**, `docs/projected_lineups.md`).

Both locators are **imported from `nfl_ats.capture_freshness`**
(`newest_snapshot_instant`, `newest_json_field_instant`) rather than
reimplemented here — see the ENG-03 section below.

`player_arrests` is never read by directory scan. Its observation is the
instant `load_latest_complete_arrest_snapshot` accepted — a manifest that is
complete, hash-verified, and neither future-dated nor over `MAX_SNAPSHOT_AGE` —
threaded through `report_for_publication(arrest_snapshot_at=...)`. A newer but
unverified directory therefore cannot make the fail-closed source look fresher
than the gate that already ran.

A source whose root is unavailable (`data_root=None`) is left **unobserved**,
not absent. Conflating "we could not look" with "there is nothing there" is
exactly how a fail-closed source would start blocking a rendering path that
never required it.

## Join with ENG-03 (`nfl_ats.capture_freshness`)

ENG-03 landed `src/nfl_ats/capture_freshness.py` during the same session and
deliberately made its two on-disk locators public "for that module's own future
join point" (**read**, its `newest_snapshot_instant` docstring). This module
imports both, so there is exactly one implementation of "read the capture
instant from the stamped name / payload field".

The **policy table stays here**, and that is not duplication:

* `capture_freshness` groups by each `SCHEDULE` job's `dedupe_dir`, so it has
  one source per directory. `odds_opener` and `odds_refresh` share
  `data/market/raw` but carry different budgets (10260 vs 3510) because the
  card treats the Tuesday opener and the late-week refresh as different
  obligations.
* `injuries_nflverse` has no capture job at all — it is refreshed by
  `weekly-run` step 1 `ingest`, driven by the `weekly_lock` job.
* `capture_freshness` answers "is each capture source producing data on
  schedule". This module answers "may this card publish", which additionally
  needs the per-source fallback, the fail-closed flag, and the roll-up.

Both derive a budget as *largest weekly gap + grace* (**read**, its
`derive_budget_minutes`), independently, and agree. A future consolidation
would move `SourceFreshnessPolicy.location` onto that module's locator
registry; the policy table, the state machine, `report_for_publication`, and
every caller stay unchanged either way.

## 2026-09-04: `pfr_transactions` coverage gap (ENG-32)

**Read**, ROADMAP Phase 13 ENG-32: `pfr_transactions` held a single snapshot
dated 2026-08-20 against the 5880-minute (4d 2h) budget derived above from
`pfr_transactions_wed`/`pfr_transactions_sat`, so `capture_scheduler.py
--health` reported it `stale`. **Read**, `data/scheduler_log.txt` (47 lines,
tail inspected): it contains zero `pfr_transactions_*` entries of any kind —
both jobs carry `added_on="2026-09-03"` and the schedule's own
`predates_job`/`snapshot_in_window` guard correctly suppressed every window
that closed before that date; the single snapshot on disk was a manual
research run (`docs/pfr_transactions_sourcing.md`), never a scheduler
capture, so this was the job's first opportunity to run at all, not a missed
run.

**Root cause found and fixed, not just a stale-clock symptom.** **Read**,
`scripts/ingest_transaction_news.py` before this session's edit: with no
flags (the exact argv the two `SCHEDULE` jobs passed), the script *resumes*
the most recent existing snapshot directory and skips any year whose
`<year>.parquet` chunk is already on disk — including the current year. Since
`nfl_ats.capture_freshness` / `nfl_ats.source_freshness_policy` read the
snapshot **directory name**, never file contents or mtime, as the capture
instant (both modules' own docstrings), this combination meant the job could
run to completion successfully every single week and the source would still
report `stale` forever: no new directory would ever be created, and the one
directory that existed would never gain a fresh timestamp. This was a defect
in this repository's code, not a remote-site or policy limitation.

**Fix (measured this session):** `scripts/ingest_transaction_news.py` gained
`create_fresh_snapshot_dir` and a `--fresh-snapshot` flag: each scheduled run
now writes a brand-new UTC-timestamped directory, copies forward every
already-cached past year with **zero network requests**, and force-refetches
only the current year's chunk (one sitemap-index fetch + one yearly-chunk
fetch, ~2 requests at the policy's 1-second minimum delay). Both
`pfr_transactions_wed` and `pfr_transactions_sat` in
`scripts/capture_scheduler.py` now pass `--fresh-snapshot`. A read-only
`--dry-run` flag was added alongside it (resolves the static sitemap-index
URL, checks `config/source_policies.json` via
`nfl_ats.source_policy.require_acquisition`, and reports which years would be
fetched vs. copied forward — zero network calls), and
`tests/test_capture_scheduler.py::test_pfr_transactions_argv_resolves_and_dry_run_exits_0_with_no_network`
pins that the exact scheduled argv plus `--dry-run` exits 0 as a subprocess
smoke test.

**Measured, the real sanctioned run** (`.\.tools\uv.exe run --no-sync python
scripts\ingest_transaction_news.py --fresh-snapshot`, the same argv the
scheduler now uses): a new snapshot `data/raw/pfr_transactions/20260904T215655Z/`
was created; the 12 years 2014–2025 were copied forward with 0 network
requests; the 2026 chunk was force-refetched (4,245 URLs, up from 3,926 in
the 2026-08-20 snapshot, +319 posts) and now includes URLs dated through
2026-09-04 (536 August rows, 53 September rows present in the new chunk —
directly checked, not inferred). `cumulative_index_rows` rose from 72,368 to
72,687 and `cumulative_transaction_relevant_rows` from 29,414 to 29,593.
**Measured**, `capture_scheduler.py --health` immediately after: `pfr_transactions`
reports `ok ... age 0.4m ... [fresh]` (`(offseason)` tag reflects the
source's own season-guarded flag, not a data problem). The daemon was **not**
restarted per this task's explicit instruction; `--health` separately flags
`code`/`schedule` as `STALE` relative to the running daemon's in-memory copy
— expected and unrelated to this fix, since editing `capture_scheduler.py`'s
`SCHEDULE` while the daemon stays alive necessarily changes the on-disk hash;
it resolves at the daemon's next routine restart.

**Backfill decision: URL-level coverage gap is closed; per-article precise
dating is intentionally out of scope here.** **Read**, PFR's sitemap chunks
one URL per YEAR (`docs/pfr_transactions_sourcing.md` section 1) rather than
per-day, and each URL is dated by its own true, immutable `datePublished` —
refetching the current year's chunk (as `--fresh-snapshot` now does every
scheduled run) always recovers the *complete* list of that year's posts to
date, so there is no separate "missed day" that additional backfill logic
could lose permanently: the 2026-08-20→2026-09-04 gap closed automatically
and completely the moment the chunk was refetched, **measured** above. This
differs from a point-in-time market quote or an official injury report,
which cannot be reconstructed after the fact — this source can be, by
design, on every refetch.

That is also why no `backfilled_at`/`covers_through` labelling was added.
**Read**, `src/nfl_ats/transaction_wire_features.py`'s leakage boundary
(`attach_transaction_counts`, `_window_counts`) is built entirely from each
transaction's own `precise_ts` (JSON-LD `datePublished`) against the game's
`kickoff_utc`/`freeze_utc` — it never reads a capture or retrieval timestamp
at all, and its existing leakage regression tests
(`tests/test_transaction_wire_features.py::test_leakage_transaction_published_after_kickoff_is_never_counted`
and two siblings) already pin exactly that boundary. A "we only learned this
at the backfill instant" distinction has no place to attach in a consumer
that never consults capture time in the first place, so labelling would add
a field nothing reads. **Not backfilled in this session, and out of this
ticket's scope**: per-article JSON-LD precise-date fetches for the ~589 new
August/September 2026 URLs (needed only for day/hour-precision features, not
for the URL-level freshness this ticket covers) — that is the separate,
already-established `scripts/pfr_bulk_date_fetch.py` /
`docs/transaction_wire_battery.md` targeted-dating effort, unaffected by and
independent of this fix; `transaction_wire_features` is not currently wired
into any production publish path (only into
`scripts/transaction_wire_battery_screen.py`, a research screen), so no
prediction-safety exposure follows from leaving those dates unfetched today.
