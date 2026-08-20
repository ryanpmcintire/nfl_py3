# Injury-news bulk sourcing: verdict, evidence, and the experiment it unlocks

Written 2026-08-19, settling the open follow-up "injury-news aggregator
ingestion -- bulk path unverified." Distinct question from the official
injury/practice REPORT already ingested (`data/players/raw/*/injuries.parquet`
via `nflreadpy`, consumed by `players.py`'s `date_modified`-gated as-of
filtering, `docs/injury_value_lost.md`): this document is about injury NEWS
timing -- when information about a player's availability first became public,
relative to the pool's Tuesday-noon pick lock, not relative to kickoff.

**Verdict: a workable bulk path exists and was built.** NBC Sports'
public sitemap index (`https://www.nbcsports.com/sitemap.xml`) is a
chronological, per-article-dated archive of every ProFootballTalk (PFT) NFL
article back to at least September 2009. Ingested in full this session:
**299,739 dated ProFootballTalk NFL article urls, September 2009 - June
2026, 202/202 months, 0 failures**, of which 20,655 (6.9%) match an
injury/availability keyword (section 4 has the complete coverage table).
PFT is the outlet that
essentially every fantasy/injury news aggregator (RotoWire, ESPN, Yahoo, most
beat-writer round-ups) cites as the original wire source for roster-move and
practice-report news. Every candidate that looked like an "API" (ESPN news,
ESPN injuries, Sleeper, FantasyPros, PFR transactions) turned out on direct
test to be a **live-snapshot-only** product with no historical query
capability -- confirmed by fetching each one this session, not inferred from
documentation. The sitemap route is not an API in that sense; it is a
generic, dated content index that happens to expose real per-article
timestamps at the granularity the point-in-time discipline needs.

---

## 1. Why the already-ingested injury REPORT cannot answer this

`docs/pool_edge_plan.md` line 80 (read): **"Picks lock Tuesday at 12"** --
essentially at the opener. The same document's gap accounting (lines 195-201,
read) already asserts, without measurement, that pre-Tuesday-lock injury
information is "a thin slice, largely already inside our features" -- an
unverified claim this document's ingestion can now actually test.

Measured this session (`data/players/raw/20260817T184901Z/injuries.parquet`,
79,818 rows with a non-null `date_modified`, 2009-2024):

| weekday of `date_modified` | rows | share |
|---|---:|---:|
| Monday | 134 | 0.17% |
| Tuesday | 192 | 0.24% |
| Wednesday | 6,166 | 7.73% |
| Thursday | 3,171 | 3.97% |
| **Friday** | **64,972** | **81.40%** |
| Saturday | 5,119 | 6.41% |
| Sunday | 64 | 0.08% |

Every one of 2009-2024's 16 seasons independently shows Mon+Tue at well under
1% of that season's rows (range 0.06%-1.45%, computed the same way). This is
structural, not a data-quality artifact: the NFL's own injury-report rule
requires each team's first practice-status report of the game week on
**Wednesday**, so the official report literally cannot exist yet at a Tuesday
noon pick lock except for a designation carried over verbatim from the
previous week. **The official report source the project already ingests is
therefore the wrong instrument for this question by construction** -- it can
tell you what was known by kickoff, not what was known by Tuesday noon. This
motivates a genuinely different source: news that breaks on its own schedule
(a player exiting Sunday's game, an IR move, a beat-reporter update), not the
league's mandated Wednesday-Friday report cadence.

Also measured (`src/nfl_ats/players.py:957,987`, read): `decision_hours_before_kickoff`
defaults to **24** everywhere it is wired (`players.py`, `availability.py`,
`quarterbacks.py`, and every corresponding `--decision-hours` CLI flag in
`cli.py`, all default `24`). For a Sunday 1pm kickoff that is a **Saturday**
cutoff, five days later than the pool's Tuesday-noon lock. The
`docs/injury_value_lost.md` backtests (+1.316 pts, `probability_positive`
0.8875, the family currently `unresolved_below_power`) were run at this
Saturday-cutoff default against the historical injuries table, which by the
weekday table above is overwhelmingly populated by Wed-Fri report filings --
i.e. that backtest's injury-value-lost feature legitimately sees information
the live Tuesday-locked pool pick cannot. This is not a leakage bug in live
`weekly-run` (the Tuesday build simply cannot see reports that have not been
filed yet, since nflverse only has what's actually happened), but it IS a
mismatch between what the historical ablation measured and what the pool can
actually play -- exactly the gap this document's data exists to quantify. Not
re-litigated further here; flagged as the direct motivation for section 4's
proposed experiment.

---

## 2. Candidates investigated

Every fetch below is **measured** (run this session, timestamps 2026-08-19).

| Candidate | Result | Evidence |
|---|---|---|
| ESPN news API (`site.api.espn.com/.../news`) | **Dead end.** `limit=1000` is silently capped at 50; `dates=20200901-20200930` is silently ignored and returns today's news regardless. No historical query capability. | Fetched `?limit=1000` -> 50 articles, oldest `2026-08-19T01:21Z`. Fetched `?limit=20&dates=20200901-20200930` -> 20 articles, all dated `2026-08-19`. |
| ESPN injuries API (`site.api.espn.com/.../injuries`) | **Dead end for history.** Live snapshot only; `?season=2020` is silently ignored (`season` field in the response still reports `2026`). Per-player `date` field exists but only for the current live board. | Fetched with and without `?season=2020`; both returned `"season": {"year": 2026, ...}`. |
| ESPN core API team injuries (`sports.core.api.espn.com/.../seasons/{Y}/teams/{id}/injuries`) | **Dead end.** 404 for both a historical season (2024) and the current one (2026) -- the endpoint path itself does not exist at this shape. | Both requests returned `{"error":{"code":404}}`. |
| Sleeper API (`api.sleeper.app/v1/players/nfl`) | **Dead end for history.** Returns one row per player with `injury_status`/`injury_notes`/`practice_participation`/`news_updated` (a single unix-ms "last touched" field) -- a live snapshot, no per-event history. Only useful prospectively (poll going forward, like the project's own live-odds archive). | Fetched full 12,221-player table; inspected structure directly. |
| FantasyPros news API | **Reported, partially measured.** Unauthenticated `GET /v2/json/nfl/news` returns `403 Forbidden` (measured). Search results (unverified) say the free tier serves recent news only and historical/bulk access needs a commercial license. Not pursued further given a working free path exists. | curl to the endpoint without a key; web search for pricing/terms. |
| Pro-Football-Reference transactions (`pro-football-reference.com/years/{Y}/transactions.htm`) | **Dead end.** Blocked by a Cloudflare bot challenge ("Just a moment...") regardless of season -- would need a real browser session to bypass, disproportionate for this task and likely against the site's active anti-scraping posture. | Fetched `years/2020/transactions.htm`; got HTTP 403 with a Cloudflare challenge page body. |
| Wayback Machine (`web.archive.org/cdx`) on `espn.com/nfl/injuries` | **Works, but redundant.** CDX confirms real crawl snapshots from 2016 onward (~130+/year by 2017), each with a provable crawl timestamp. But the page it captures is the same official status-board content already in nflverse (Wed-Fri filings) -- a slower, less complete path to data already on hand, not new information. | CDX query returned real timestamped snapshot rows; first `200` capture 2016-08-06. |
| Wayback Machine on `rotoworld.com/football/nfl/player-news` | **Works but sparse.** Only 9 captures found across all of 2019 sampled, clustered in the offseason -- not dense enough to reconstruct a specific week reliably, and each site redesign (roughly 3 URL-scheme eras since 2007) would need its own scraper. Not pursued further given a denser source exists. | CDX query, 15 rows returned, dated Mar/Jul/Aug/Oct/Dec 2019 and Mar/Apr 2020. |
| RotoWire (`rotowire.com`, live site) | **Not pursued -- flagged as a live alternative.** `robots.txt` (read) exposes `rotowire.com/sitemap.xml` and a dedicated `rotowire.com/news-sitemap.php`, and only disallows a list of known bad-bot user agents (no blanket disallow), suggesting a similarly viable sitemap-based bulk path. Not built because the NBC Sports/PFT path below was already confirmed dense and precisely dated; a future session could extend coverage here if PFT alone proves insufficient. | Fetched `rotowire.com/robots.txt`; inspected the sitemap declarations. |
| Twitter/X API (beat-reporter archives) | **Confirmed infeasible, as anticipated.** Reported (web search, unverified but consistent across multiple pricing pages): as of Feb 2026 X moved to pay-per-use; full-archive historical search is gated to the Pro tier (~$5,000/mo, legacy, closed to new signups) or Enterprise (~$42-50k/mo). No affordable path for a private research project. | Web search on 2026 X API pricing; not independently fetched (would require a paid account). |
| Reddit (r/fantasyfootball, Pushshift) | **Confirmed infeasible for this task, as anticipated.** Reported (web search, unverified): Pushshift's public access was revoked in May 2023; the official Reddit API has no date-range historical search; the surviving option (Project Arctic Shift / academic torrent dumps) is a heavyweight, unstructured bulk-download-and-mine project, not a targeted injury-news source. Not pursued given a superior, already-structured source exists. | Web search on 2026 Reddit/Pushshift access status; not independently fetched. |
| **NBC Sports sitemap -> ProFootballTalk NFL** (`nbcsports.com/sitemap.xml` -> `sitemap-YYYYMM.xml` -> per-article `<lastmod>`) | **Works. Built the ingestion on this.** See section 3. | See section 3 for full fetch evidence. |

---

## 3. Verification: the winning source, fetched directly

`https://www.nbcsports.com/robots.txt` (fetched, read): `Crawl-delay: 10` for
`User-agent: *`, and explicitly lists `Sitemap: https://www.nbcsports.com/sitemap.xml`.
The ingestion script honors this delay on every request (`RateLimiter` in
`scripts/ingest_injury_news.py`).

`https://www.nbcsports.com/sitemap.xml` (fetched): a sitemap **index** of 275
monthly chunk files, `sitemap-200309.xml` through `sitemap-202606.xml` (plus a
rolling `sitemap-latest.xml`) -- September 2003 through June 2026, confirming
the number-string is a `YYYYMM` bucket, not an arbitrary id (275 files over
~22.75 years is almost exactly one per month).

Early buckets (`sitemap-200309.xml`) are golf/other-sport content only --
PFT/NFL urls start appearing by `sitemap-200909.xml` (measured: 1,471
ProFootballTalk NFL urls in that single month), consistent with PFT's known
2009 move under the NBC Sports/Comcast umbrella. The ingestion script
therefore defaults `--start 200909`.

**Density, measured directly from fetched monthly sitemaps** (PFT NFL urls /
month, injury-keyword-matched subset in parentheses):

| Month | PFT/NFL urls | Injury-keyword matches |
|---|---:|---:|
| Sept 2009 | 1,471 | -- |
| Sept 2012 | 1,717-1,735 | -- |
| Sept 2016 | 1,727-1,750 | -- |
| Oct 2020 | 1,939 | 255 (13.1%) |
| Sept 2022 | ~1,900-1,982 | 261 (13.2-13.7%) |

**Per-article timestamp verification** ("fetch a sample, one historical
week" -- the task's explicit ask): fetched five real article pages published
the week of 2020-09-28 through 2020-10-01 (all in the already-spent
`[2020, 2021]` opener window `injury_value_lost` was measured on):

| Article | Sitemap `<lastmod>` | JSON-LD `datePublished` |
|---|---|---|
| Byron Jones returns to practice, but Tua Tagovailoa remains out | 2020-10-01 15:12:52 UTC | `2020-10-01T15:12:52Z` |
| Browns add Odell Beckham Jr. to injury report | 2020-10-01 16:17:31 UTC | `2020-10-01T16:17:31Z` (secondary human-readable field on the page: `September 28, 2020 04:05 PM`) |
| Texans add Will Fuller to injury report | 2020-10-01 17:42:20 UTC | `2020-10-01T17:42:20Z` |
| Julio Jones limited practice; Calvin Ridley sits with ankle injury | 2020-10-01 17:49:16 UTC | `2020-10-01T17:48:55Z` |
| Joey Bosa returns to practice as limited participant | 2020-10-01 18:49:41 UTC | `2020-10-01T18:48:45Z` |

Two things confirmed directly by this fetch: (1) the sitemap's `<lastmod>` and
the article's own JSON-LD `datePublished` agree to the minute in 4 of 5 cases
and to within 21 seconds in the fifth -- `<lastmod>` is a real per-article
timestamp, not a platform artifact; (2) one article carries a second,
earlier, human-readable date string (`September 28`) alongside the ISO
`datePublished` (`October 1`) -- likely the true original-publish date versus
a later edit/re-index touch. **Recommendation for any downstream feature
build: treat `<lastmod>`/`datePublished` as the conservative "known no later
than" bound, never "known no earlier than"** -- consistent with this
project's pregame/point-in-time discipline, which requires proof of public
availability, not a best guess at it.

---

## 4. Ingestion built

`scripts/ingest_injury_news.py` (new, does not touch `experiment_runner.py`,
`margin.py`, `public_board.py`, or `cli.py`). Two modes:

- **Bulk mode** (default): fetches the sitemap index once, then each monthly
  chunk in the requested `[--start, --end]` range, filters to
  `/profootballtalk/` urls, tags each with a 41-term injury/availability
  keyword match against the URL slug (deliberately over-inclusive -- every
  PFT/NFL url is kept regardless of match, with an `injury_relevant` boolean,
  so the keyword line can be redrawn later without re-fetching). Writes one
  parquet per month plus a concatenated `index.parquet` and a `manifest.json`.
  Idempotent (skips months already on disk) and resumable.
- **`--verify-sample YYYYMM`**: fetches a handful of real article pages from
  one month and extracts JSON-LD `datePublished`/`headline`, for exactly the
  spot-check in section 3.

Rate-limited to the site's own declared `Crawl-delay: 10` (one request per 10
seconds to `www.nbcsports.com`), which is why this script does bulk headline+
timestamp collection at the sitemap level rather than bulk full-article-body
fetching -- fetching body text for thousands of individual articles at 10s
each would take days; sitemap-level collection over the full 2009-2026 span
takes well under an hour.

**Snapshot layout, corrected mid-session.** The first run wrote directly to
`data/raw/injury_news/manifest.json`. `nfl_ats.snapshots.latest_snapshot()`
treats any directory directly under `data/raw/` that contains a
`manifest.json` as a candidate *schedules* snapshot, so this collided with
the real schedule-snapshot selection and broke another agent's public-board
regeneration (`FileNotFoundError`) -- caught and reported mid-session by the
coordinator. Fixed two ways: (1) `scripts/ingest_injury_news.py` now writes
under a timestamped snapshot subdirectory, `--out/<UTC timestamp>/...`
(`resolve_snapshot_dir`), matching this repo's existing convention
(`data/raw/<timestamp>/`, `data/players/raw/<timestamp>/`) -- no
`manifest.json` ever sits directly at `data/raw/injury_news/`; (2) verified
directly (`nfl_ats.snapshots.latest_snapshot(Path("data/raw"))`) that it now
resolves to the real schedule snapshot, not `injury_news`. The already-fetched
months were moved into the new layout rather than re-fetched, and the crawl
resumed from where it left off.

**Coverage of this ingestion run, measured**
(`data/raw/injury_news/20260819T191639Z/manifest.json`, gitignored per the
existing `data/raw/**` rule, not committed -- full run completed
2026-08-19T19:52:59Z):

| | |
|---|---:|
| Months requested | 200909-202606 (202 months) |
| Months fetched, this run + resumed | 202 / 202 |
| Months failed | 0 |
| Cumulative ProFootballTalk NFL urls | **299,739** |
| Cumulative injury-keyword-relevant urls | **20,655** (6.9%) |
| Span | September 2009 - June 2026 (16.75 seasons; extends 1-2 years past nflverse's own 2009-2024 injury-report coverage) |

Re-run `python scripts/ingest_injury_news.py --out data/raw/injury_news` (no
`--snapshot` needed -- it resumes the most recent existing snapshot
automatically) to extend past June 2026 as new months publish, without
re-fetching anything already on disk. Per-article body-text samples live at
`data/raw/injury_news/20260819T191639Z/sample_articles/*.json` (5 files,
October 2020, section 3).

**Usage note carried in the manifest itself:** private research caching only,
matching this project's own CFBD/cfbfastR precedent
(`docs/data_feasibility.md`, License item 6: "private caching/retention"
permitted, raw tables "must never be republished"). NBC Sports' terms of use
were not independently reviewed this session -- this is a policy stance
inferred by analogy, not a verified legal fact, and should be revisited
before any publication or redistribution of the raw archive.

---

## 5. The specific experiment this unlocks

Not run in this session (task scope: ingestion + coverage report only). Two
directly testable questions, both requiring nothing more than what is now on
disk:

1. **Does the Tuesday-noon opener already price the injury information a
   Saturday-cutoff feature sees?** Rebuild the `injury_value_lost_narrowed`
   isolation (`docs/injury_value_lost.md` sec 4) with a true Tuesday-noon
   decision cutoff instead of `decision_hours_before_kickoff=24` (Saturday),
   using this ingestion to identify which official-report designations were
   already foreshadowed by PFT news dated before that Tuesday-noon cutoff (a
   designation reported Wednesday but foreshadowed by a Sunday/Monday PFT
   "exited game with X, X-ray on the way" post counts as pre-Tuesday-known;
   one filed cold on Wednesday with no prior PFT mention does not). Compare
   accuracy/`probability_positive` at the Tuesday-cutoff construction against
   the already-measured Saturday-cutoff +1.316 pts, P+ 0.8875. If the effect
   survives near-intact, the signal is real and playable in the actual pool;
   if it collapses, the backtest's edge was an artifact of information the
   live Tuesday-locked pick can never use.
2. **Does the "thin slice, largely already inside our features" claim in
   `docs/pool_edge_plan.md` (line 199) hold up?** For each historical Tuesday
   noon, diff the set of PFT injury-relevant headlines dated Sunday-night
   through Tuesday-morning against what the official Friday-final report
   later confirmed. A high overlap supports the existing claim (already
   priced); a material gap identifies specific games where the opener-graded
   pool pick could have used information the model's current Saturday-cutoff
   features do not distinguish from later-arriving news.

Per AGENTS.md: whatever `probability_positive` either analysis returns, an
interval crossing zero is not grounds to close the question -- record any
resulting measurement through `nfl-ats weak-signals record` with the
appropriate `unresolved_below_power` classification unless it clears one of
the two admissible closing grounds (refuted mechanism / positive-control
bound), and report `probability_positive`, never "contains zero."

### 5.1 Both experiments run, 2026-08-19

**Binding closing-grounds taxonomy, restated per AGENTS.md:** an interval or
CI that contains zero is NEVER grounds to reject, fail, or close an
experiment -- at this evaluator's ~2-point resolution "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line
of work: (1) refuted mechanism -- a RESOLVED wrong sign (whole interval on
the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else
is `unresolved_below_power`, reported by `probability_positive`, never
"contains zero." Neither ground is cleared below -- both new arms stay
`unresolved_below_power`, recorded, not closed.

**Method.** `scripts/injury_tuesday_cutoff_experiment.py` (new). Mechanism:
`decision_hours_before_kickoff` only gates which injury rows are "visible"
for the game currently being scored (`players.py:_injury_rows_asof`); every
other piece of state (snap shares, production, QB ratings, roster
continuity) is built from strictly prior completed games and does not depend
on it. So instead of a day-of-week-dependent hours offset, each Tuesday arm
pre-filters the injuries dataframe to rows that pass their OWN game's
own-week Tuesday-noon-ET test (computed exactly per game, the same
"most-recent-Tuesday" convention `nfl_ats.clv.live_tuesday_openers` already
uses, anchored at noon instead of UTC midnight), then calls the unmodified
`enrich_with_player_features` with `decision_hours_before_kickoff=0`
(decision_at = kickoff, always after Tuesday noon, so the function's own
filter never removes anything further). `players.py` itself was never
edited. Three arms, all built from the exact snapshots the frozen
`game_features_player_value.parquet` manifest recorded (`player_snapshot
20260812T200527Z`, `pbp_snapshot 20260812T142851Z`, `player_value_snapshot
20260813T121050Z`), on the already-spent `mod07_weak_signal_stack`
`[2020, 2021]` opener window (free re-read precedent, same as
`scripts/availability_ablation.py`; no `rotation.assign`/`record_look`, the
frozen `[2022, 2023]` window was never touched). Seed 20260819, 20,000
week-blocked bootstrap samples.

**Reproduction check (measured):** a fresh from-scratch rebuild at
`decision_hours_before_kickoff=24` reproduced the recorded D-A contrast at
**+1.3158 pts** (recorded +1.316) with `probability_positive` 0.8893
(recorded 0.8875, small difference explained by 20,000/seed-20260819
bootstrap draws here vs. the original 2,000/seed-20260817) -- the underlying
per-arm accuracies matched to machine precision (0.513158 / 0.526316 both
runs). The rebuild pipeline is confirmed faithful before trusting its two new
arms.

**Experiment 1 result -- collapses, does not survive intact:**

| arm | baseline acc | candidate acc | delta (pts) | 95% week-blocked | P+ | disagreements |
|---|---:|---:|---:|---|---:|---:|
| Saturday (`decision_hours=24`, recorded) | 51.32% | 52.63% | **+1.316** | [-0.661, +3.261] | 0.8893 | 26 |
| Tuesday, official report only | 51.32% | 51.32% | **+0.000** | [-0.877, +0.875] | 0.3965 | 4 |
| Tuesday, official + PFT-foreshadowed | 51.75% | 51.54% | **-0.219** | [-1.129, +0.668] | 0.24805 | 7 |

Coverage behind the collapse (measured, `coverage` block of
`artifacts/injury_tuesday_cutoff/result.json`): only **328 of 76,784**
official injury-report rows (0.43%) have their own `date_modified` at or
before their game's own Tuesday noon ET -- confirming section 1's point
structurally, now on the exact rows the D-A contrast uses. Matching this
week's injury-relevant PFT headlines (full-name substring match, deliberately
conservative/precision-favoring -- undercounts last-name-only headlines) adds
5,918 more rows, raising visibility to **8.13%** -- still leaving 91.9% of
designations unseen by Tuesday noon.

**The Tuesday-to-Saturday channel, isolated directly (paired, same 456
games):**

| contrast | channel delta (pts) | 95% week-blocked | P+ |
|---|---:|---|---:|
| Saturday D-A minus Tuesday-official D-A | **+1.3158** | [-0.454, +3.185] | 0.9003 |
| Saturday D-A minus Tuesday+PFT D-A | **+1.5351** | [-0.445, +3.672] | 0.91725 |

Read plainly: the official-only channel delta (+1.3158) is numerically
**the entire Saturday effect** (+1.3158 of +1.3158) -- under a strict
Tuesday-noon cutoff with no PFT credit at all, 100% of the measured edge
traces to information that does not exist yet at the pool's lock. Crediting
every PFT-foreshadowed designation as Tuesday-known (the more generous arm)
still leaves the channel delta at +1.5351, slightly LARGER than the whole
Saturday effect, because that arm's own D-A point estimate went slightly
negative (-0.219) rather than merely flat.

**Decision, stated before the caveat, forced-pick frame:** ~~the pool grades
against a Tuesday-noon lock. `injury_value_lost_narrowed`'s previously
recorded +1.316 pt / P+ 0.8875 edge is not currently playable by a forced
Tuesday pick -- under a true Tuesday cutoff the same contrast on the same
games is flat-to-slightly-negative (0.000 to -0.219 pts), and the paired
channel-delta test says with P+ 0.90-0.92 that essentially all of the
originally measured edge is attributable to information arriving strictly
after Tuesday noon.~~ **Owner-corrected 2026-08-20:** the pool's LINE grades
against a Tuesday-noon lock, but our PICKS are not forced to Tuesday --
picks are editable up to each game's real deadline (**refined 2026-08-20:
min(kickoff, Sunday 16:00 ET) -- SNF/MNF lock early at Sunday 4pm, not at
kickoff**). The measurements above
still stand exactly as reported (0.000 to -0.219 pts under a true
Tuesday-cutoff *construction*, +1.32 to +1.54 pt channel delta between that
construction and the Saturday-cutoff one, P+ 0.90-0.92) -- they describe
what is knowable at Tuesday PUBLISH time versus at a Saturday-ish decision
time, which remains a real and useful distinction. What changes is the
verdict drawn from it: the Saturday-cutoff construction is not a
theoretical ceiling the pool can't reach, it is what a late-week pick
refresh actually plays, so `injury_value_lost_narrowed`'s +1.316 pt / P+
0.8875 edge IS the playable figure for this pool, provided picks are
refreshed close to kickoff rather than left at their Tuesday-publish state.
This does not refute the mechanism (value-weighted
injury magnitude is still real information, reliability 0.87-0.93,
`docs/injury_value_lost.md` sec 3.1) and it is not bounded by a positive
control -- both intervals above cross zero, exactly the AGENTS.md-expected
shape at this resolution, so the family stays `unresolved_below_power`, not
closed. ~~The correction is narrower and more useful than a closure: the
existing `injury_value_lost_narrowed` predeclaration (`docs/
injury_value_lost.md` sec 7) should not be read as pool-playable evidence
without re-deriving its cutoff -- if the `[2022, 2023]` window is ever spent
on this family, the predeclared construction must specify a Tuesday-noon (or
later-arriving-information-excluded) decision cutoff, not the Saturday
default, or it will re-measure a channel the pool cannot use.~~ **Owner-
corrected 2026-08-20:** if the `[2022, 2023]` window is ever spent on this
family, the predeclared construction should use the Saturday (or another
genuinely pre-kickoff) decision cutoff -- that is what a late-week refresh
pass sees and is therefore the pool-playable construction, not a channel the
pool cannot use. Four results
recorded to `registry/weak_signals.json`:
`injury_value_lost_tuesday_cutoff_official`,
`injury_value_lost_tuesday_cutoff_pft_augmented`,
`injury_value_lost_tuesday_saturday_channel_official_only`,
`injury_value_lost_tuesday_saturday_channel_pft_augmented`.

**Experiment 2 result -- the "thin slice, largely already inside our
features" claim (`docs/pool_edge_plan.md` line 199) does not hold up, and
is consistent with experiment 1's collapse.** For every historical
(season, week, team), took the LATEST-filed official designation per player
(the "Friday-final" report the model actually sees) and checked for a
matching PFT injury-relevant headline dated Sunday ~18:00 ET through Tuesday
noon ET of that same week (conservative full-name match, same lower-bound
caveat as above):

| report status | designations | already headline-visible by Tuesday noon |
|---|---:|---:|
| Out | 13,807 | 5.96% |
| Doubtful | 2,909 | 3.85% |
| Questionable | 20,312 | 2.10% |
| Probable | 14,323 | 0.68% |
| **All statuses** | **76,782** | **2.39%** (1,838 rows) |

This is a lower bound (the matcher undercounts last-name-only headlines), so
the true fraction is higher than 2.39% -- but even doubling or tripling it
leaves the great majority of Friday-final designations NOT foreshadowed in
the Sunday-night-to-Tuesday-morning PFT window. The "thin slice, largely
already priced" framing materially overstated how much of the eventual
official report is knowable by Tuesday; measured directly, it is a thin
slice that is mostly NOT already inside our features, which is exactly why
experiment 1's Saturday-cutoff edge evaporates under a true Tuesday lock.
This measurement is diagnostic, not an ATS effect estimate, so it was not
recorded to `registry/weak_signals.json` (no `accuracy_points`/`ats_points`
unit applies); the numbers live here and in
`artifacts/injury_tuesday_cutoff/result.json`
(`experiment_2_thin_slice_check`).

---

## 6. Provenance summary

- **Measured this session:** every entry in section 2's table, the weekday
  distribution in section 1, the `decision_hours_before_kickoff` default in
  `players.py`/`availability.py`/`quarterbacks.py`/`cli.py`, every sitemap
  fetch and article-page fetch in section 3, the ingestion script's own
  output.
- **Read this session:** `docs/injury_value_lost.md`,
  `docs/injury_value_lost_tilt_overlay.md`, `docs/pool_edge_plan.md`,
  `docs/data_feasibility.md`, `registry/weak_signals.json`
  (`injury_value_lost_gradient` confirms the reported split-half reliability
  0.9325, matching the task brief's "0.933"), `.gitignore`,
  `www.nbcsports.com/robots.txt`.
- **Reported, unverified:** FantasyPros' free-vs-commercial-tier terms
  (search results only), X/Twitter 2026 pricing, Reddit/Pushshift 2026
  access status. None of these gate this document's verdict -- they explain
  why those three candidates were not built, not the reason a path exists.
- **Inferred:** the "private research caching, never republish" policy
  stance for the NBC Sports archive (by analogy to the project's existing
  CFBD precedent, not independent legal review); the recommendation to treat
  `<lastmod>`/`datePublished` as a conservative upper bound on "known by."

**Section 5.1 addendum (later the same day, 2026-08-19).** Measured this
pass: every number in 5.1's two tables, the reproduction check, the coverage
percentages, and the experiment-2 by-status table, all from
`scripts/injury_tuesday_cutoff_experiment.py` and
`artifacts/injury_tuesday_cutoff/result.json`. Read this pass:
`docs/injury_value_lost.md` sec 4 and 7 (the D-A construction and frozen
predeclaration), `docs/pool_edge_plan.md` lines 193-201 (the "thin slice"
claim), `scripts/availability_ablation.py` (the free-re-read window-split
machinery this script re-derives), `src/nfl_ats/clv.py`
(`live_tuesday_openers`'s own-week-Tuesday convention, reused here),
`registry/rotation_registry.json` via `nfl_ats.rotation.load_registry`
(confirmed `mod07_weak_signal_stack`'s only spent window is `[2020, 2021]`;
`[2022, 2023]` never referenced or touched). Four registry entries recorded
via `nfl-ats weak-signals record`, all `unresolved_below_power`, no
`closing_ground` (neither admissible ground was cleared).
