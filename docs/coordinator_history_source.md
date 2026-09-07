# Wikipedia coordinator revision source (PER-07)

**Measured** (`data/raw/coordinators/20260907T144330576130Z/manifest.json`):
the bounded KC/GB 2019–2025 pull returned all 42 sampled HC/OC/DC assignments,
including 28 OC/DC assignments, from 14 historical staff-template revisions.
It used 13 new requests and one cached probe response; every request returned
HTTP 200. **Measured**
(`data/raw/coordinators/20260907T144526855711Z/coverage_audit.json`): 42/42
assignments had dated observations before the corresponding team's first
regular-season game, with lead times 11.80–166.96 days. This is a census of
the selected sample, not an estimated league coverage rate or an ATS effect;
there is no sampling interval to report.

**Inferred:** the source-access blocker is resolved for this sample. Full
2009–2025 league coverage and in-season change completeness remain unmeasured.

## Source assessment

**Read** ([MediaWiki revisions API](https://www.mediawiki.org/wiki/API:Revisions),
timestamp/content and rvstart sections): a request can retrieve revision content
as of a historical cutoff. **Read**
([Wikimedia terms, section 7](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use)):
Wikipedia text is reusable under CC BY-SA 4.0 with attribution and applicable
share-alike requirements. **Read**
([API etiquette](https://www.mediawiki.org/wiki/API:Etiquette)): serial requests,
contact User-Agent, caching and maxlag are the relevant acquisition practices.

**Measured** (`data/raw/coordinators/20260907T143754Z/probe_0.response`):
the 2019 Chiefs season article's August 30 revision contains a HC infobox but
transcludes `Kansas City Chiefs staff` for its current staff. **Inferred:**
expanding that historical article with today's template would introduce a
historical-content error. **Measured**
(`data/raw/coordinators/20260907T143853Z/probe_0.response`): querying the
template's own pre-September revision instead supplies all three roles,
observed March 25, 2019. **Read**
([Chiefs head-coach list](https://en.wikipedia.org/wiki/List_of_Kansas_City_Chiefs_head_coaches)):
this list concerns head coaches, not a complete OC/DC history.

**Read** ([Wikidata licensing](https://www.wikidata.org/wiki/Wikidata:Licensing)):
structured claims use CC0. **Read**
([P580](https://www.wikidata.org/wiki/Property:P580)): start-time qualifiers
describe when something starts; **inferred:** that is not by itself evidence
of when the claim became publicly observable. **Measured**
(`data/raw/coordinators/20260907T143940Z/probe_0.response`): the bounded
Eric Bieniemy entity probe has no P39 position claims. **Inferred:** this
single probe does not establish absence across Wikidata; Wikipedia revisions
already supply a more directly usable sample.

**Measured** (`data/raw/coordinators/20260907T144526855711Z/coverage_audit.json`):
516 existing raw parquet schemas, excluding the newly created coordinator
snapshots, contained no column name including `coordinator`; no schema reads
failed. **Reported, unverified** (`docs/coaching_leads.md:5–9`): the prior
broader inventory also found no usable dated playcaller history. The schema
scan alone does not rule out names embedded in free text or CSV files.

## Collection and input contract

**Read** (`scripts/ingest_coordinator_history.py:47–291`): run with explicit
team-to-page mappings and a bounded request budget:

```powershell
.\.tools\uv.exe run --no-sync python scripts/ingest_coordinator_history.py --team 'KC=Template:Kansas City Chiefs staff' --team 'GB=Template:Green Bay Packers staff' --start-season 2019 --end-season 2025 --max-requests 14 --delay 1
```

**Read** (`scripts/ingest_coordinator_history.py:167`): supported years are
2009–2025; default cutoff is September 1 at 00:00 UTC. `--cutoff` accepts
another month/day/time within each season. Each call returns the latest
revision at or before that cutoff, not its first introduction and not all
revision history. Supplying other cutoffs can sample additional observations;
this collector does not yet enumerate every transition. Explicit template
titles accommodate renamed franchises without following current redirects.

**Read** (`scripts/ingest_coordinator_history.py:153–267`): acquisition passes
`require_acquisition`, uses a contact project URL in the User-Agent, maxlag=5,
serial requests and at least one second after each request. A failed HTTP,
transport or API response stops the batch; there is no automatic retry loop.
Successful cached requests are reused in fresh timestamped directories.
Raw bytes, request status, URL, timestamp, digest and code provenance are
preserved alongside `manifest.json` and `coordinator_history.parquet`.
Completed snapshots are not overwritten. Retrying the same command resumes
through the cache, including after reaching a request budget.

**Measured** (`data/raw/coordinators/20260907T144601351192Z/manifest.json`):
the repeat command made zero network requests and emitted the same 42 names
with a typed UTC `effective_observed_at` column.

**Read** (`scripts/ingest_coordinator_history.py:84–150`): canonical rows carry
season, team, role, person, real revision timestamp, observation basis,
revision permalink/id, source title, retrieval timestamp and requested cutoff.
Names use wikilink targets to retain disambiguation. Unsupported, conflicting,
multi-person, unlinked or transcluded role values are omitted. A missing or
post-cutoff revision timestamp is rejected, never replaced with season start.
No season-undated assignments are manufactured from current pages.

**Read** (`src/nfl_ats/coordinator_changes.py:225`):
`build_coordinator_history_features(games, history)` accepts these rows, excludes
HC and `season_undated` observations, and passes the revision timestamp as both
effective and observed boundary to the unchanged fail-closed OC/DC builder.
It partitions seasons so an old season's sparse sample cannot silently fill
a new season. The revision timestamp means the assignment was present in that
revision, not that the employment contract started then.

**Inferred:** a September sample does not prove staff remained unchanged in
October, identify an exact firing date, or establish playcalling responsibility.
Do not use this sparse sample to assert absence of in-season changes. A complete
change study should enumerate historical revisions or collect decision-specific
cutoffs and preserve unknown/vacant role state. Renames, inline plain names,
vacancies and co-coordinator structures warrant coverage review before a larger pull.

**Read** (`tests/test_ingest_coordinator_history.py:1`): offline synthetic fixtures
cover extraction, timestamps, ambiguity, late-revision leakage, undated exclusions,
cross-season isolation, immutable cache reuse, failure stops and request budgets.
**Read** (`config/source_policies.json:132–169`): two additive Wikimedia entries
record source terms and crawl conditions. **Measured** (this lane's commands):
no ATS experiment, effect measurement, registry write, scheduler change,
publication, commit or push ran.
