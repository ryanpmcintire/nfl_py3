# Half/quarter-game markets: per-event endpoint probe (LEAD-61 step 1b)

## Closing-grounds taxonomy (verbatim, as required for any experiment-adjudicating text)

An interval or CI that contains zero is NEVER grounds to reject, fail, or close an
experiment. At this evaluator's ~2-point resolution, "contains zero" is the EXPECTED
outcome for a real small signal. Only two grounds ever close a line of work: (1) refuted
mechanism -- a RESOLVED wrong sign (whole interval on the wrong side of zero) or zero
split-half reliability; (2) bounded by a positive control proven able to detect an effect
that size. Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the binary "contains
zero". The registry code hard-rejects inadmissible closures; if a record command errors,
the verdict is wrong, not the validator.

This document is not an experiment write-up (no accuracy-points comparison was scored
here) -- it is a source-availability/cost probe, included per the fleet brief's binding
instruction to paste the taxonomy verbatim in anything touching experiment grounds, since
LEAD-61 references a scored lead (`half_line_script_2h_underdog`, still
`unresolved_below_power`, docs/lead02_half_line_script.md, untouched by this probe).

## What this answers

Lane AM's step-1 probe (`scratchpad/reports/laneAM_half_market_probe.md`,
`docs/...` referenced from the LEAD-61 ROADMAP row) established that the four half-game
market keys (`spreads_h1`, `spreads_h2`, `totals_h1`, `totals_h2`) are rejected with HTTP
422 `INVALID_MARKET` on the **bulk board** endpoint
(`/v4/sports/americanfootball_nfl/odds/`, what `nfl_ats.market_data.fetch_odds_api` and
every scheduled capture job call), with an error message reading "not supported by this
**endpoint**" -- and inferred, unconfirmed, that The Odds API's period/alternate markets
are likely served only from the **per-event** endpoint
(`/v4/sports/{sport}/events/{eventId}/odds`).

This probe (lane AO, LEAD-61 step 1b) confirms that inference and measures its cost.

## Requests made (exact accounting)

The task granted at most 3 requests to The Odds API. **Only 1 was used.**

**Request that was skipped (saved):** the task's own step 1 calls for an events-list
request to obtain a current event id. **Read**, `data/market/raw/20260905T160024Z/response.json`
(the most recent same-day bulk-board capture, written before this probe by the routine
`odds-ingest --markets spreads,h2h,totals` job) already contains full event objects with
`id` fields for the entire remaining 2026 schedule (272 events, **measured** via a local
JSON parse, no network call). Event ids are not a scarce resource here -- the bulk board
capture already running every session hands them out for free. No events-list call was
made.

**Request 1 of 3 (measured):** the per-event odds endpoint for one upcoming event.

- Event: `8c94552d022acec4a0458d70c19d3da9` -- Seattle Seahawks (home) vs New England
  Patriots (away), `commence_time` 2026-09-10T00:20:00Z (this week's Thursday-night
  opener), chosen because it was the earliest-kickoff, best-covered event on the bulk
  board (11 bookmakers quoting `h2h`/`spreads`/`totals` there).
- URL: `https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/8c94552d022acec4a0458d70c19d3da9/odds`
- Params: `regions=us`, `markets=spreads_h1,spreads_h2,totals_h1,totals_h2,spreads_q1`,
  `oddsFormat=american`, `dateFormat=iso` (`spreads_q1` added per the per-event/period
  market documentation -- **reported**, background knowledge not re-verified by a web
  fetch this session: The Odds API's public market-list docs describe H1/H2/Q1-Q4 period
  markets for American football as available only through this per-event endpoint, which
  is exactly consistent with lane AM's 422 "not supported by this endpoint" wording on the
  bulk board).
- Result: **HTTP 200.** All five requested markets were returned by at least one book.
- Quota headers (**measured**): `x-requests-last: 5`, `x-requests-remaining: 99980`
  (down from the pre-probe 99985 -- **read**,
  `data/market/raw/20260905T160024Z/manifest.json`), `x-requests-used: 20`.
- Manifest + raw body written to
  `data/market/raw/20260905T180313Z-event-halves-probe/` (private, untracked --
  `data/market/**` is gitignored -- following the same directory/manifest-field
  convention `nfl_ats.market_data.write_market_snapshot` uses). Probe script (scratch
  only, not in the repo): `probe_event_odds.py` in this session's scratchpad.

**Request 2 and Request 3 of 3:** not used. One measurement was sufficient to answer both
"does the market exist here" (yes) and "what does it cost" (linear, 1 credit per
market-region -- see below), so the reserve retry was not needed.

## Measured findings

**The half-game markets are priced on the per-event endpoint, not the bulk board.**
`spreads_h1`, `spreads_h2`, `totals_h1`, `totals_h2` (plus `spreads_q1`, tested
alongside) all returned live quotes for this event:

| market | books quoting (of 6 total on this event) |
|---|---|
| `spreads_q1` | 6 |
| `totals_h1` | 6 |
| `spreads_h1` | 5 |
| `spreads_h2` | 3 |
| `totals_h2` | 3 |

(**measured**, parsed from `data/market/raw/20260905T180313Z-event-halves-probe/response.raw`:
draftkings, fanduel, betrivers quote all five; williamhill_us and bovada quote
`spreads_h1`/`spreads_q1`/`totals_h1` only; betmgm quotes `spreads_q1`/`totals_h1` only.)
Second-half markets (`spreads_h2`, `totals_h2`) have visibly thinner book coverage than
first-half/first-quarter markets at 5 days out from kickoff. **Inferred, not measured
this session:** coverage plausibly thickens closer to kickoff as more books post period
lines; that would need a same-week, closer-to-kickoff capture to confirm and is exactly
the kind of thing a live capture job would observe naturally once running.

**Cost model (measured + inferred):** this one call, 5 markets x 1 region, cost exactly
5 credits (`x-requests-last: 5`). The existing bulk-board captures already show the same
linear relationship -- **read**, `data/market/raw/20260905T160024Z/manifest.json`:
3 markets (`spreads,h2h,totals`) x 1 region cost exactly 3 credits. Two independent data
points (3->3 on the bulk endpoint, 5->5 on the per-event endpoint) are consistent with
**1 credit per market x region, charged per call regardless of endpoint** -- the per-event
endpoint is not more expensive per market, it is simply scoped to one game instead of the
whole board. By subtraction under that same linear model (not separately measured, to
avoid spending a second request purely to confirm arithmetic): the four *half* markets
alone (`spreads_h1`, `spreads_h2`, `totals_h1`, `totals_h2`, no `spreads_q1`) would cost
**4 credits per event**.

## Weekly cost projection

Per the task's own formula (16 events x cost x 2 captures/week, Tuesday open + Saturday):

- **4 half markets only:** 16 events x 4 credits x 2 captures = **128 credits/week**.
- **5 markets incl. `spreads_q1`:** 16 x 5 x 2 = **160 credits/week**.

For comparison (**measured/read**, this repo's current state):

- The live scheduled odds jobs today (`odds_tue_open`, `odds_thu_tnf`, `odds_sat`,
  `odds_sun_close`, `odds_sun_late`, `odds_mon_mnf` -- `scripts/capture_scheduler.py:249-360`)
  each call the bulk board with 3 markets, 1 region = 3 credits/run, 6 runs/week = **18
  credits/week** today, confirmed by the manifest's `requests_last: "3"`.
- Remaining balance this session (**measured**, this probe's own quota header):
  **99,980** requests remaining.
- The only quota *floor* defined anywhere in this repo -- **read**,
  `config/source_policies.json:117-120` (`"the_odds_api".quota.historical_minimum_remaining:
  600`) and `src/nfl_ats/odds_backfill.py:45` (`DEFAULT_QUOTA_FLOOR = 600`) -- is textually
  scoped to the `/v4/historical/...` endpoint's `execute_backfill`/`plan_backfill` path,
  which costs `10 x markets x regions` per call (`HISTORICAL_CREDITS_PER_MARKET_REGION = 10`,
  `odds_backfill.py:42`). There is **no existing floor constant for the live `/odds` or
  per-event `/events/{id}/odds` endpoints** this probe used -- lane AM already noted this
  gap and self-limited to the coordinator's request cap instead. The build plan below
  recommends adopting the same 600-credit floor as a project convention for the new job,
  not because a rule requires it today.

**Bottom line on affordability:** 128-160 credits/week is trivial against 99,980
remaining and against the existing 18 credits/week baseline -- adding the halves capture
at both Tuesday and Saturday would raise total weekly odds spend from 18 to roughly
146-178 credits/week, i.e. still under 0.2% of the remaining balance consumed per week.
At that rate the remaining balance would support this capture cadence for hundreds of
weeks (99,980 / 146 ~ 684 weeks), far beyond any planning horizon this project needs. Cost
is not a blocker.

## Build plan (design only -- NOT implemented in this lane)

**Do not implement in this lane.** This section is a design for a future build, per the
task's explicit instruction.

### Which existing job it rides on

Ride on the existing `odds_tue_open` and `odds_sat` jobs
(`scripts/capture_scheduler.py:250-262`, `:314-325`), the same two captures the task's own
cost formula uses. Add a new `Job` entry, e.g. `odds_halves_tue` / `odds_halves_sat`,
using the `requires=(...)` mechanism already proven by `weekly_lock` (`requires=
("odds_tue_open",)`, `capture_scheduler.py:280`) so the halves capture only runs after its
paired bulk-board capture has *succeeded* and written that week's event ids -- it must
never run standalone, since it depends on the bulk snapshot for which event ids exist.
`odds_thu_tnf`/`odds_sun_close`/`odds_sun_late`/`odds_mon_mnf` are close-side captures
outside the task's own two-capture cost formula and are not included in this plan (adding
them later is a simple extension of the same design, at extra weekly cost -- not
recommended without a specific reason, since the opener/pool-relevant read the project
cares about is graded at the Tuesday open per AGENTS.md's "grade at the opener" rule).

### Scope: current week only, not the whole 272-event board

The bulk-board snapshot this probe read from event ids off of contains **272 events**
(the entire remaining 2026 schedule, not just next week's slate) -- **measured**, local
parse of `data/market/raw/20260905T160024Z/response.json`. A halves job must filter to
the *current* week's ~16 events before looping (e.g. via `nfl_ats.nfl_week.week_cycle_sunday`,
already used by `odds_backfill.py` to anchor weekly decision timestamps), not iterate the
whole board. Getting this filter wrong would multiply the measured per-event cost by up
to ~17x (272/16) for no benefit -- worth stating explicitly since it is an easy
implementation mistake and the single biggest cost risk in this design.

### Manifest shape

Reuse the existing normalization/storage plumbing rather than inventing a new format:

1. For each of the current week's event ids (from the just-written bulk snapshot), call
   `GET /v4/sports/americanfootball_nfl/events/{eventId}/odds` with
   `markets=spreads_h1,spreads_h2,totals_h1,totals_h2` (no `spreads_q1` by default --
   out of scope for LEAD-61, add only if a future quarter-market lead wants it, at its own
   measured cost), `regions=us`.
2. Accumulate the raw per-event JSON objects (each already shaped like one element of the
   bulk board's array -- `id`, `sport_key`, `commence_time`, `home_team`, `away_team`,
   `bookmakers`) into one Python list, exactly matching the bulk endpoint's top-level array
   shape.
3. Serialize that list once with `json.dumps(...).encode()` and pass it to the existing
   `nfl_ats.market_data.parse_odds_api_response` (unmodified -- it already accepts an
   arbitrary JSON array of event objects) to get one combined `quotes` DataFrame, then
   `attach_nflverse_game_ids` as `_cmd_odds_ingest` already does.
4. Write with the existing `nfl_ats.market_data.write_market_snapshot` **once** per
   capture run (not once per event) -- one `response.json` (the combined array), one
   `quotes.parquet`, one `manifest.json` under a new stamped directory, e.g.
   `data/market/raw/<stamp>/` with `extra_manifest={"capture_kind": "event_halves",
   "events_requested": N, "events_returned": M}` (the `extra_manifest` hook already exists
   in `write_market_snapshot` and is used by `HISTORICAL_CAPTURE_KIND` in
   `odds_backfill.py` for the same purpose -- marking a snapshot's provenance so it is
   never confused with a different capture shape).
5. `quota` in that manifest should be the **last** call's headers (matches the existing
   convention -- `x-requests-remaining` is always the running balance, so the last call's
   value is authoritative for "what's left"), plus a summed `total_credits_this_run` field
   in `extra_manifest` since no single header reports the multi-call total directly.

### Quota check

Before starting the per-event loop, read `x-requests-remaining` from the paired bulk-board
capture's own manifest (already just written in the same job run) and refuse to start if
`remaining - (planned_events * 4)` would fall under a floor -- recommend adopting the same
`DEFAULT_QUOTA_FLOOR = 600` convention `odds_backfill.py` already uses for the historical
endpoint (`src/nfl_ats/odds_backfill.py:45`), even though no rule currently requires a
floor on this endpoint (see above). This mirrors `execute_backfill`'s existing
`quota_floor` parameter and `--quota-floor` CLI flag
(`src/nfl_ats/cli_commands/market.py:222-227`) rather than inventing a new pattern.
Given the measured 128-160 credits/week cost against 99,980 remaining, this floor check
would not block the capture in practice today -- it exists as a circuit breaker for a
future session, not because affordability is in question now.

### What is explicitly out of scope for this lane

- No new CLI subcommand, `Job` entry, or PowerShell wrapper was added.
- No change to `scripts/capture_scheduler.py`, `scripts/odds_capture.ps1`,
  `src/nfl_ats/market_data.py`, or `src/nfl_ats/cli_commands/market.py`.
- No confirmation yet that a season of half-market pairs exists to test the predeclared
  `half_line_script_2h_underdog` direction on production (LEAD-61's own step 3) -- that
  requires the capture job to actually run for a season first.

## Files touched by this probe

- `docs/half_game_markets.md` (this file, new).
- `data/market/raw/20260905T180313Z-event-halves-probe/` (new, private, untracked --
  `data/market/**` is gitignored).
- `ROADMAP.md` -- appended a dated 2026-09-05 note to the `LEAD-61` row only.
- Scratch-only, not in the repo: `probe_event_odds.py` (the probe script) and this
  report's counterpart at `scratchpad/reports/laneAO_half_market_event_probe.md`.
