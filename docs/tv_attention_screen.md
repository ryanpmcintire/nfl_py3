# TV-attention screen: lagged Sports Media Watch viewership as a public-tax fade (MKT-14)

Written 2026-09-11 (lane MKT-14). Script: `scripts/tv_attention_screen.py`.
Rotation family: `tv_attention_fade_on_production` (opener grade, window
[2020, 2021], the same window other lanes used tonight). Registry entries:
`nfl-ats rotation record` (below) and `nfl-ats weak-signals record --name
tv_attention_fade_on_production`.

## Binding closing-grounds taxonomy (verbatim, restated per CLAUDE.md/AGENTS.md)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds
ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
(whole interval on the wrong side of zero) or zero split-half reliability;
(2) bounded by a positive control proven able to detect an effect that
size. Everything else is `unresolved_below_power`: record it with
`nfl-ats weak-signals record`, report `probability_positive`, never the
binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator.

## Data source and what has/has not been done before this lane

`docs/sports_media_watch_ingest.md` (read) already built the ingestion and
the publication-timestamp backfill: `data/raw/sports_media_watch/20260820T164046Z/`
holds the raw 2014-2023 seasonal-page snapshot (1,065 structured rows,
2014-2021 HTML tables plus 2022-2023 ratings images), and
`data/raw/sports_media_watch_publications/20260902T235500Z/` (status
`COMPLETE`, read) holds the versioned backfill that matches each row to a
dated Sports Media Watch article and freezes `source_published_at`
separately from `source_modified_at`. **Read**: as of that backfill, no ATS
experiment, registry change, or model wiring had run against this source --
this lane is the first.

**Measured** (`scripts/tv_attention_screen.py`, this session): filtering
`data/raw/sports_media_watch_publications/20260902T235500Z/ratings_rows.parquet`
to `point_in_time_usable == True` with both teams identified leaves 417
point-in-time-safe rows, 2014-2021:

| season | usable rows |
|---|---|
| 2014 | 11 |
| 2015 | 82 |
| 2016 | 76 |
| 2017 | 3 |
| 2018 | 28 |
| 2019 | 69 |
| 2020 | 67 |
| 2021 | 81 |

2017 is nearly unusable (3 rows) and 2014 is thin (11); the rest carry
real weekly coverage. Every row is a single nationally-featured broadcast
(a doubleheader/primetime/special-window game); regional/single-game rows
without an identified matchup are dropped, so per-team coverage tracks how
often a team was a national/doubleheader draw, not every game it played.

**Read**: `nfl-ats opener-evaluation`'s per-game artifact
(`artifacts/opener_evaluation/20260910T211255Z/per_game.parquet`) -- the
frozen forced-pick opener grade documented in `docs/opener_evaluation.md`
and `docs/pool_edge_plan.md` -- only exists for 2020-2025 (Tuesday-opener
line archives do not reach further back). The intersection of that
population with usable Sports Media Watch coverage is exactly **2020 and
2021**, which is also the window `docs/pool_edge_plan.md` and other lanes
this session used. This lane declared and was assigned that window through
`nfl-ats rotation declare/assign` (family `tv_attention_fade_on_production`,
grade `opener`, `--acknowledge-mined`); the registry independently landed on
`[2020, 2021]`.

## (1) Feature: trailing team attention, slot-normalised

**Predeclared**, before any cover-rate or accuracy number was computed:

- **N = 3.** Trailing mean of a team's last 3 usable (point-in-time-safe,
  both-teams-identified) national broadcasts, strictly before the target
  game (ordered by `(season, week)`; a team's own trailing value at game
  *g* only ever averages appearances with `(season, week) < g`'s
  `(season, week)`). N=3 was chosen because usable national appearances
  are sparse (a team appears roughly every 3-6 weeks in this table, not
  every week), so 3 balances recency against single-game noise without
  requiring a full season of history.
- **Slot normalisation.** Sports Media Watch labels each row's broadcast
  window with year-specific strings (`SNF`, `MNF`, `Late DH`, `Early DH`,
  `London`, `Special`, ... -- 24 distinct labels across the 417 rows, plus
  a blank label for all of 2015). These are bucketed into three groups
  before z-scoring, matching the task's own framing:
  - **primetime**: `SNF`, `MNF`, `TNF` (and asterisked/early-late variants),
    `Kickoff`.
  - **sunday_late** (Sunday afternoon): `Late DH`, `National`, `Late Sat`.
  - **sunday_early** (early window): `Early DH`, `Regional`, `Single`,
    `Early/Sat` variants, `London`/`London*`, `Special`/`Special**`.
  - **unknown**: the blank label (all 82 of 2015's rows carry no slot
    label in the source page).
  Each row's raw audience is z-scored against its own bucket's mean/std
  (`viewers` pooled over the full 417-row sample, one constant per
  bucket -- a scale-only nuisance transform, not an outcome-dependent
  one, but **not walk-forward**; this is disclosed as a simplification,
  not hidden). London-game placement in `sunday_early` (mid-morning ET
  kickoff) and `Special`/holiday games are judgment calls, documented here
  rather than re-litigated case by case; none of the affected labels
  exceeds 23 rows.
- **Coverage in the graded population.** All 466 games in the 2020-2021
  opener-graded population have both a home and an away trailing value
  defined (every team had accumulated >= 3 qualifying national broadcasts
  by 2020) -- **measured**, `graded_games_both_trailing_defined ==
  graded_games_total == 466` in
  `artifacts/tv_attention_screen/20260911T031523Z/results.json`.

## (2) Predeclared mechanism and cover rate by decile

**Predeclared mechanism** (task-level, stated before scoring): public
betting money follows the teams people watch, so the market line on a
high-attention team carries a "public tax" -- i.e. backing a team the
public has been watching heavily should show a **lower** ATS cover rate
than backing a low-attention team, because the line is shaded to account
for lopsided public action on the popular side. Direction predeclared:
cover rate should fall as a team's own trailing-attention decile rises.

**Measured** (932 team-game observations = 466 graded games x 2 sides,
20 exact pushes excluded, pooled home+away, deciles of the pooled trailing
distribution, Wilson 95% intervals):

| decile | n | cover rate | 95% interval |
|---|---|---|---|
| 1 (lowest attention) | 93 | 0.473 | [0.375, 0.574] |
| 2 | 93 | 0.538 | [0.437, 0.636] |
| 3 | 89 | 0.404 | [0.309, 0.508] |
| 4 | 90 | 0.533 | [0.431, 0.633] |
| 5 | 91 | 0.451 | [0.352, 0.553] |
| 6 | 94 | 0.532 | [0.432, 0.630] |
| 7 | 90 | 0.633 | [0.530, 0.726] |
| 8 | 92 | 0.435 | [0.338, 0.537] |
| 9 | 103 | 0.485 | [0.391, 0.581] |
| 10 (highest attention) | 77 | 0.519 | [0.410, 0.627] |

**Measured**: Spearman correlation between decile index and cover rate is
0.018 -- no detectable monotone gradient at this sample size. Top-tercile
(deciles 8-10, n=272) cover rate 0.478 vs bottom-tercile (deciles 1-3,
n=275) 0.473 -- indistinguishable. This is the expected shape of a real
small effect swamped by ~90-per-bin noise, per the binding rule above: it
is **not** evidence the mechanism is false (no decile's interval sits
entirely below the others in a resolved pattern), it is
`unresolved_below_power`.

## (3) Production screen: stacked on production's opener pick

**Predeclared rule**, frozen before scoring: flip production's `weak_stack`
opener pick (`pick_home_at_open_probability_rule` from
`artifacts/opener_evaluation/20260910T211255Z/per_game.parquet`) to the
other side whenever the team production picked has its own trailing
attention in the pooled top tercile (decile >= 8, using the same bin edges
as the table above). Comparator: `correct_at_open_probability_rule`
(production's served forced-pick correctness) vs. the same value with
flipped games inverted; exact pushes (34 of 1,537 in the full artifact, 10
of 466 in 2020-2021) excluded from both, matching how `opener_accuracy_
probability_rule` itself is computed. Interval: week-blocked bootstrap,
20,000 resamples, seed 20260910 (`nfl_ats`-style block-by-`(season,week)`
resampling, implemented directly in `scripts/tv_attention_screen.py`
because the repo's `clv.week_blocked_bootstrap` is specialised to
`clv_points`). `probability_positive` uses `P(>0) + 0.5*P(==0)` on the
bootstrap draws, matching `nfl_ats.evidence_conventions
.probability_positive_from_draws`.

**Measured** (`artifacts/tv_attention_screen/20260911T031523Z/results.json`,
`production_screen`):

| population | n games | picks changed | base accuracy | flipped accuracy | delta (accuracy points) | 95% interval | P(effect > 0) |
|---|---|---|---|---|---|---|---|
| pooled 2020-2021 | 456 | 141 | 53.29% | 55.26% | **+1.97** | [-4.39, +8.64] | **0.717** |
| 2020 only | 220 | 80 | 52.27% | 56.82% | **+4.55** | [-5.26, +14.82] | **0.813** |
| 2021 only | 236 | 61 | 54.24% | 53.81% | **-0.42** | [-8.79, +7.93] | **0.459** |

Every interval crosses zero. Per the binding rule, that is the expected
outcome at this sample size and is **not** grounds to reject the rule --
the 2020 leg alone would already clear a naive "P+ > 0.8" bar, and per
AGENTS.md's promotion-bar-is-not-a-decision-bar rule, `probability_positive`
above 0.5 (0.717 pooled) favours playing the fade on expected value, not a
0.90 threshold. This is reported as a single below-power confirmation
look, not a promotion decision.

## (4) Split-half reliability of the attention trait

**Measured**: using the full 2014-2021 usable sample (not just the graded
window, to maximise team-season coverage), each team's mean slot-normalised
z-score was computed separately for odd seasons (2015, 2017, 2019, 2021)
and even seasons (2014, 2016, 2018, 2020). 31 of 32 teams have data in both
halves (Washington and the Rams are thin -- 1-2 usable appearances in one
half each -- reflecting the underlying scarcity noted in section 1, not a
processing bug).

- Pearson r (raw half-to-half) = **0.648**, 95% Fisher-z interval
  [0.381, 0.815], n = 31 teams.
- Spearman-Brown corrected (full-length) reliability = **0.786**.

The underlying team-popularity trait is clearly reliable -- large-market,
frequently-televised teams (e.g. Dallas, Green Bay) score consistently
high across both halves. This matters for classification: a reliability
this far from zero means `no_split_half_reliability` is **inadmissible**
as a closing ground for the production-screen result above. The trait is
real and reproducible; the measured *effect on ATS cover rate* is simply
below this sample's power to resolve, which is exactly the
`unresolved_below_power` category, not a refutation.

## Recording

```
nfl-ats rotation declare --name tv_attention_fade_on_production --grade opener --acknowledge-mined ...
nfl-ats rotation assign --name tv_attention_fade_on_production --size 2
  -> assigned [2020, 2021]
nfl-ats rotation record --name tv_attention_fade_on_production \
  --artifact artifacts/tv_attention_screen/20260911T031523Z/results.json \
  --verdict unresolved --probability-positive 0.7171 \
  --effect 1.9737 --effect-units accuracy_points \
  --interval-low -4.3862 --interval-high 8.6396 --sample-blocks 35
nfl-ats weak-signals record --name tv_attention_fade_on_production \
  --family tv_attention_fade_on_production --league nfl \
  --season-start 2020 --season-end 2021 \
  --effect 1.9737 --effect-units accuracy_points \
  --classification unresolved_below_power \
  --interval-low -4.3862 --interval-high 8.6396 \
  --probability-positive 0.7171 --sample-games 456 --sample-blocks 35 \
  --reliability 0.6481 --category attention ...
```

Both commands succeeded on the first attempt (no file-conflict retry
needed); the recorded entries are in `registry/rotation_registry.json`
(family `tv_attention_fade_on_production`) and
`registry/weak_signals.json` (same family name).

## What this does and does not establish

- The trailing-attention trait is real and reliable (r=0.648 split-half).
- Its relationship to ATS cover rate at this sample size (466 games, one
  feasible 2020-2021 window) is a coin flip leaning toward the predeclared
  fade direction (P+ 0.717 pooled, 0.813 in 2020, 0.459 in 2021) with every
  interval crossing zero -- `unresolved_below_power`, recorded as such.
- Nothing here changes the played card. This is a single confirmation
  look on a rotation-assigned window; per the decision-bar rule, a
  probability above 0.5 favours the fade on expected value, but promotion
  would need either a second independent window or pooling alongside other
  attention-family signals (`nfl-ats weak-signals pool`), not a
  re-litigation of this one look.
- 2014, 2017 (and to a lesser extent 2018) are too thin in the raw Sports
  Media Watch ingestion to grade against openers even if opener-line
  archives existed further back; if the odds archive is ever extended
  before 2020, 2015/2016/2018/2019 have enough usable rows (69-82 each) to
  be worth a second look, but 2017's 3 rows do not.

## 2026-09-11 follow-up: the live 2026 path, and a prospective challenger

A second lane picked this screen up the same night to (1) establish a live
weekly ingestion path for the 2026 season and (2) wire the production
screen above as a no-window-cost `ACTIVE_PROSPECTIVE` challenger,
following the `veteran_rest_back_overlay` / `post_bye_new_playcaller_back_overlay`
precedent (`docs/veteran_rest_back_overlay.md`).

### The live season-table path is measured dead for 2022+

**Read**, `docs/sports_media_watch_ingest.md` and `scripts/ingest_sports_media_watch.py`
scrape Sports Media Watch's per-season NFL TV-ratings HTML table page
(`nfl-tv-ratings-viewership/<season>-season/`, with the CURRENT season
special-cased to the bare `nfl-tv-ratings-viewership/` URL). **Measured,
2026-09-11**, live against the primary site:

- The dedicated URL 404s for every season from 2023 onward
  (`.../2024-season/`, `.../2025-season/`, `.../2026-season/`); `.../2025-season/`
  in fact 301-redirects to `college-football-tv-ratings/2025-season/`, a
  different sport's page entirely.
- The site's own WordPress `pages` REST API
  (`wp-json/wp/v2/pages?search=nfl tv ratings`) lists a dedicated NFL-ratings
  page object for every season 2014 through **2022** (id 109324, "2022 NFL
  television ratings") and no 2023/2024/2025/2026 page object at all.
- The one surviving "current" page (bare `nfl-tv-ratings-viewership/`, the
  URL this ingester's `season == 2023` branch fetches) carries `<title>NFL
  TV ratings page, 2023 edition</title>` and `article:modified_time
  2025-08-28T03:33:08+00:00` -- untouched for over a year relative to the
  in-universe date -- and **zero rows match the ingester's own
  weekday-comma-date row-boundary regex** (`_parse_event_date`), i.e. even a
  forced re-parse of that page yields no usable rows for any season.
- A WordPress-posts search (`wp-json/wp/v2/posts?search=...`) for a
  dedicated NFL weekly-ratings recap article published since 2026-08-01
  returns none for the 2026 season as of this check, while the equivalent
  college-football Week 1 recap (`.../college-football-week-1-ratings-recap-.../`)
  had already posted the same day -- so there is currently no live NFL
  viewership content published in EITHER the old season-table format or a
  narrative weekly-recap-post format.

**Conclusion, measured, not inferred**: this is not "Week 1 is thin," it is
Sports Media Watch having discontinued the machine-readable NFL
season-table page this ingester depends on. Re-running the same command
next week or next month will keep returning zero rows until Sports Media
Watch either resumes that page or starts a parseable weekly-recap format;
nothing about the code was broken.

### Both source scripts had been deleted by the same day's repository cut

**Read**, `scripts/ingest_sports_media_watch.py` and
`scripts/backfill_sports_media_watch_timestamps.py` (and their test file)
do not exist in the working tree at the start of this lane, and `git log`
shows exactly one commit touching them since they were written --
`b7ed31d` ("Repository cut: 469,660 -> 216,083 Python lines"), which
deleted all three files outright, apparently classifying an
as-yet-unused-by-any-experiment ingester as one-off research code. Both
scripts were restored verbatim from `9f84d09` (the commit immediately
before the cut, already comment/docstring-stripped by that day's earlier
comment ban) via `git show 9f84d09:<path>`; the test file was NOT restored
(the test moratorium bans new/restored test files). `ingest_sports_media_watch.py`
received one additive fix beyond the restore: a season page returning
HTTP 404 is now caught and recorded in the manifest as
`pages_unavailable_404` instead of raising after three useless retries --
without this, the command below would crash every week it is scheduled,
which is exactly the "job that was never exercised and dies on its first
in-season run" failure mode AGENTS.md's scheduler section names.

**Measured, live run, 2026-09-11:**

```
.\.tools\uv.exe run --no-sync python scripts\ingest_sports_media_watch.py `
  --output data\raw\sports_media_watch\20260911T034948Z --seasons 2026
```

produced a clean, non-crashing manifest: `pages_unavailable_404: [2026]`,
`structured_rows: 0`, `structured_team_identified_rows: 0`, no
`ratings_rows.parquet`/`source_index.parquet` written (there is nothing to
write). Attempting the paired backfill command against that same
zero-row snapshot,

```
.\.tools\uv.exe run --no-sync python scripts\backfill_sports_media_watch_timestamps.py `
  --source data\raw\sports_media_watch\20260911T034948Z `
  --output data\raw\sports_media_watch_publications\<snapshot> `
  --schedules data\raw\20260824T115346Z\schedules.parquet
```

raises `FileNotFoundError` on `ratings_rows.parquet` -- **by design, not a
bug**: `backfill()` enriches an existing archive's rows with publication
timestamps, and there were zero rows to enrich. Deliberately NOT patched
to succeed on an empty source: doing so would write a new, later-dated
"COMPLETE" publications snapshot with zero rows, and every downstream
reader (`latest_smw_publications_snapshot`, in this screen and in the new
overlay below) picks the **newest** COMPLETE snapshot by timestamp --
publishing an empty one would silently blind every future run to the real
2014-2021 data still sitting in `20260902T235500Z/`. **Operational rule for
scheduling these two commands weekly (e.g. Tuesday morning): run the
ingest command unconditionally; run the backfill command only when the
ingest manifest reports `structured_rows > 0`.**

### `tv_attention_fade_overlay`: the production screen, wired live

`src/nfl_ats/tv_attention_fade_overlay.py` reimplements this screen's
feature construction (slot-bucket z-scoring, N=3 trailing mean, same
window buckets) as a self-contained module (following the
`veteran_rest_back_overlay` precedent of reimplementing rather than
importing from `scripts/`), registered as `ACTIVE_PROSPECTIVE` in
`artifacts/prospective/challengers.json` (display name "Fade the
most-watched team" in `CHALLENGER_DISPLAY_NAMES`,
`src/nfl_ats/dashboard/findings_content.py`).

**One deliberate, disclosed difference from the screen for live safety.**
The screen's own historical study pools ALL 2014-2021 usable rows into one
team-long trailing series with no era boundary, because it is a fixed
retrospective study of a fixed window. A live weekly challenger cannot do
that safely: with the season-table source dead (see above), a naive
carry-forward trailing lookup would keep reporting each team's LAST
2020/2021-vintage z-score indefinitely, as if a team's national-broadcast
popularity in 2020 still describes it in 2026 and beyond. Instead,
`tv_attention_fade_overlay` computes each team's own CURRENT trailing
value only from observations at `season >= LIVE_ERA_START_SEASON` (2022,
one season past the spent `[2020, 2021]` rotation window) using the same
newest point-in-time-usable snapshot; the top-tercile THRESHOLD it
compares that value against is still fit on the full pooled distribution
of all seasons strictly before the one being scored (2014-2021, for a 2026
game), which is a broader population than the screen's own
2020-2021-only decile fit and is disclosed as a second difference (a live
challenger has no fixed graded population to draw edges from).

With zero rows on file at `season >= 2022` (see above), every team's live
trailing value is undefined, so the overlay never has enough history to
flip any pick -- this was proven both ways: a synthetic snapshot with
season-2022 rows and a decisively top-tercile team DOES flip correctly
(verified directly, not via a committed test, per the moratorium), and the
real current snapshot correctly produces zero flips.

**Confirmed live, 2026-09-11**, against the active 2026 Week 1 card
(`artifacts/margin_predictions/2026-week-01-20260910T210852Z`, model
`d49194e04945a5e5`): `record_tv_attention_fade_overlay_decisions` recorded
14 of 16 games (2 had already kicked off by the time this ran) into
`artifacts/prospective/challenger_decisions.parquet`, `flip_count: 0`, and
every one of the 14 recorded `pick_side` values matches production's own
raw pick exactly -- confirming the "fewer than three prior audiences ->
no flip, record production's pick 1:1" contract holds on the actual
current week, for the actual reason (a dead source), not a stub.

**Not wired anywhere else.** No switch applies this overlay to the
published card; `recommendations.csv`, `CURRENT_PREDICTIONS.md`,
`docs/*.html`, `pick_refresh.py`, `tiebreaker.py`,
`best_pick_renomination.py`, `board_content.py` and `board_terminal.py`
were not touched by this lane. The day a usable `season >= 2022` row is
ever backfilled -- by Sports Media Watch resuming publication, or by a
future lane building a narrative-post parser against their weekly-recap
articles -- this same code starts scoring on it without further changes.

## 2026-09-11 candidate sourcing for a live 2026 weekly feed (MKT-14)

A third lane the same night picked up the still-open half of the problem
above: the season-table page is confirmed dead (measured, previous
section), and no 2026 NFL weekly recap post exists yet on Sports Media
Watch's posts API either -- so a *different* live source is needed for
`tv_attention_fade_overlay` ever to score on a current game. This section
probes candidates with single polite requests (one browser-style user
agent, no loops, no bulk pulls) and records what each is worth.

### Candidate table

| candidate | 2025 weekly data exists? | fields | publication timestamp | license/terms | robots.txt stance |
|---|---|---|---|---|---|
| Sports Media Watch weekly posts (posts API) | not probed further this lane -- see below | would be network/window/viewers per prior seasonal-page schema | `article:published_time`/`article:modified_time` meta on dated posts (established 2026-08-20/09-02, prior sections) | no dedicated terms page found; site copyright, no explicit automated-access clause | **measured, direct fetch of `sportsmediawatch.com/robots.txt`**: a `User-agent: anthropic-ai` block with `Disallow: /` names Anthropic's crawler specifically and blocks the entire site, separate from generic bot blocks (`GPTBot`, `ChatGPT-User`, `cohere-ai`, `scrapy`, `curl`, `Python-urllib`, ...) |
| Sports Business Journal / Nielsen trade coverage | not fetched (see robots finding) | would be network/window/viewers via SBJ's own ratings-roundup coverage | not checked | subscription/paywalled trade press (inferred, not measured this lane) | **measured, direct fetch of `sportsbusinessjournal.com/robots.txt`**: an explicit, dated (`Last-Updated: July 28, 2026`) 83-agent blocklist names **both** `User-agent: anthropic-ai` ("Anthropic AI Crawler") **and** `User-agent: ClaudeBot` ("Anthropic Claude Bot"), each with `Disallow: /` -- the most explicit exclusion found of any candidate |
| NFL's own press site (`nflcommunications.com`) | not reachable | unknown | unknown | nfl.com's own terms are already `red`/`acquisition_allowed: false` in `config/source_policies.json` (`nfl_com_injuries` row, terms_url `nfl.com/legal/terms/`, a site-wide ToS, not injury-specific) | **measured**: `nflcommunications.com` redirects to `mediaarchive.nfl.net`, which returned **HTTP 403 Forbidden** on a plain `robots.txt` request -- a credential-gated media portal, not a public page |
| Wikipedia per-season "NFL on television" ratings tables | **measured, no**: the flagship article's table of contents (18 top-level sections, fetched via `action=parse&prop=sections`) covers rights/scheduling/blackout/flex/history and has no ratings-by-week section; a full-text search for `NFL week 1 2025 viewers million Nielsen` (`action=query&list=search`) surfaces only isolated notable-game mentions in unrelated articles (a playoff game, a Super Bowl, an unrelated TV-drama page), not a per-season structured table | n/a | Wikipedia revisions carry timestamps and the repo already has a reviewed `wikipedia_coordinator_revisions` policy row | clean (`en.wikipedia.org/robots.txt`, no AI-agent block; existing repo policy already covers acquisition terms) |
| **Awful Announcing weekly `nfl-ratings`-tagged recap posts** (found under "any other page") | **measured, yes**: the `/tag/nfl-ratings` archive lists weekly recap posts back through multiple seasons (URL pattern `awfulannouncing.com/nfl/<slug>-week-<n>-ratings-....html`); fetched and confirmed content for the 2024 Week 4 post (published 2024-10-02) and the 2025 Week 3 post (published 2025-09-25); **no 2025 Week 18 or 2026 Week 1 recap exists yet** on this site as of this lane (newest tagged post is a 2026-08-20 preseason article; a targeted web search for a 2026 Week 1 ratings recap found only announcing-schedule articles, no ratings numbers) | network, broadcast window (SNF/MNF/TNF/Sunday windows), viewers, and (when the prose names them) the two teams -- embedded in narrative sentences, not a table | `article:published_time` / `article:modified_time` meta tags on every post, directly on the dated article itself (no separate backfill step needed, unlike Sports Media Watch's living seasonal pages) | no dedicated terms-of-use page found (the linked `legal-disclaimer.html` 404-redirects to the homepage); standard "Copyright (c) 2026 www.AwfulAnnouncing.com - All Rights Reserved" footer, same posture as the existing `pfr_transactions` policy row | **measured**, clean: only old (2006-2012) archive-index paths are disallowed for `User-agent: *`; no AI-agent-specific block of any kind |

### Recommendation: Awful Announcing, not Sports Media Watch or SBJ

**Measured**, both Sports Media Watch's and Sports Business Journal's
`robots.txt` name Anthropic's crawler by identifier and disallow it from
the entire site (SBJ names `ClaudeBot` explicitly). This agent is exactly
that crawler. Per the repository's instruction to respect `robots.txt`
and terms, no further request was made to either site this lane beyond
the single `robots.txt` fetch itself (which is the mechanism a site uses
to state its wishes, not content access). This is a stricter reading than
the letter of robots.txt matching (a non-self-identifying user agent
string does not literally match the `anthropic-ai` token), but using a
different identifier specifically to route around a rule that names this
agent's own operator would defeat the purpose of the rule; that judgment
call is disclosed here for the owner to override if a broader interpretation is
intended. The existing Sports Media Watch ingestion built by prior
sessions already used a non-self-identifying research user agent before
this finding was made; this lane did not re-fetch page content there and
flags the question rather than deciding it unilaterally.

Of the remaining candidates, Wikipedia does not have the needed data in
structured form and the NFL's own press site is not publicly reachable.
**Awful Announcing is recommended**: clean `robots.txt`, a confirmed
weekly cadence with real publication timestamps embedded directly in each
article (no separate publication-backfill step required, unlike Sports
Media Watch), and content that is free to read. Its policy row is added
to `config/source_policies.json` as `awful_announcing_tv_ratings`
(`risk: yellow`, modelled on the existing `pfr_transactions` row: no
explicit terms-of-use document exists, so acquisition is conservatively
scoped to `research_use_only`, raw redistribution `prohibited`, derived
publication `aggregates_only`, plus a `prose_extraction_manual_audit_required`
condition specific to this source, disclosed below).

### `scripts/ingest_tv_audiences.py`: built, run once, real defects found and fixed in-session

The new ingester (`scripts/ingest_tv_audiences.py`) fetches one Awful
Announcing recap article (`--url`, or `--discover` to poll
`/tag/nfl-ratings` for the newest untaken post), extracts
`article:published_time`/`article:modified_time` from the page's own meta
tags (fails closed if absent), and regex-extracts `(network, window,
team_a, team_b, viewers)` rows from sentences containing "X million
viewers", each carrying the verbatim matched sentence
(`source_sentence`) for audit. Output mirrors the Sports Media Watch
ingest's shape: `pages/<slug>.html` (raw capture), `rows.parquet`
(structured rows), `manifest.json` (coverage, hashes, the same
point-in-time contract dict as `ingest_sports_media_watch.py`), and
`provenance.json` (the publication-timestamp evidence: article URL,
title, both meta timestamps, page hash, extraction-rule version).
Guarded by `nfl_ats.source_policy.require_acquisition` /
`require_private_raw_destination` against the new policy row, matching
`scripts/ingest_transaction_news.py`'s convention.

**Measured, live run, 2026-09-11** (single fetch, 2-second delay, against
`https://awfulannouncing.com/nfl/strong-viewership-continues-fox-week-3.html`,
the newest verified real recap available -- 2025 Week 3, published
2025-09-25T01:30:53Z):

```
.\.tools\uv.exe run --no-sync python scripts\ingest_tv_audiences.py `
  --output data\raw\tv_audiences\<snapshot> `
  --url https://awfulannouncing.com/nfl/strong-viewership-continues-fox-week-3.html `
  --season 2025 --week 3
```

produced 6 rows:

| network | window | team_a | team_b | viewers | source sentence (verbatim) |
|---|---|---|---|---|---|
| (none) | SUNDAY_4:25P.M.ET | SF | CHI | 25,470,000 | "The network notched the most-watched game of the week for the second consecutive week, averaging 25.47 million viewers for the 4:25 p.m. ET window that featured the Chicago Bears-Dallas Cowboys game in 85% of markets, and the Arizona Cardinals-San Francisco 49ers game in the rest of the country." |
| (none) | UNSPECIFIED | (none) | (none) | 25,660,000 | "The window was nearly bang-on last year's season-to-date average of 25.66 million viewers." |
| NBC | SNF | KC | NYG | 25,300,000 | "Transitioning to Sunday night, NBC and Peacock scored 25.3 million viewers for Sunday Night Football, which featured the Kansas City Chiefs earning their first win of the season against the New York Giants." |
| ESPN | MNF | DET | BAL | 22,800,000 | "ESPN's Monday Night Football captured a similarly impressive audience, earning 22.8 million viewers for the Detroit Lions' win over the Baltimore Ravens." |
| Prime Video | TNF | BUF | MIA | 16,450,000 | "Rewinding to last Thursday, Prime Video scored its third most-watched Thursday Night Football game ever as the Buffalo Bills' win over the Miami Dolphins drew 16.45 million viewers, up 23% versus last year's comparable game (Patriots-Jets, 13.37 million)." |
| CBS | UNSPECIFIED | (none) | (none) | 16,150,000 | "CBS averaged 16.15 million viewers for its single-game window on Sunday afternoon." |

**Two real extraction defects were found and fixed while building this,
disclosed rather than hidden:** (1) a naive sentence-boundary search on
`". "` broke inside "p.m." (splitting "4:25 p.m." mid-abbreviation),
corrupting the SNF/MNF window's team and network capture -- fixed with a
negative-lookbehind sentence-boundary regex (`(?<![pa]\.m)\.\s+(?=[A-Z0-9])`);
(2) embedded Twitter/X post captions in the article HTML (four
`<blockquote class="twitter-tweet">... -- NBC Sports PR (@NBCSportsPR)...</blockquote>`
attribution lines) were bleeding into the extraction context window and
mislabelling the Thursday Night Football row's network as `ESPN` instead
of the correct `Prime Video` -- fixed by stripping `<blockquote>` content
before extraction. **Remaining, disclosed limitations, not fixed:** row 1's
network is blank because that sentence refers to "the network" instead of
repeating "Fox" by name (an anaphora-resolution gap); row 2 (25.66M) is
almost certainly the same Fox window's season-to-date comparison figure
restated, not a second independent game, and would need to be
recognised as a duplicate before feeding a feature; row 6 (CBS,
16.15M) has no team names because the source sentence itself does not
name the matchup. This is exactly why the policy row's
`prose_extraction_manual_audit_required` condition exists: every row's
`source_sentence` must be read before the row is trusted, unlike a clean
HTML table.

**Exact weekly command to schedule** (auto-discovers the newest
not-yet-ingested post; a no-op, zero-row manifest when nothing new has
posted, matching `ingest_sports_media_watch.py`'s dead-source behaviour):

```
.\.tools\uv.exe run --no-sync python scripts\ingest_tv_audiences.py `
  --output data\raw\tv_audiences\<snapshot> --discover
```

**Not wired to any challenger.** This lane only proves the source and the
parser on one real week; `tv_attention_fade_overlay` still reads the
Sports Media Watch snapshot (which has no `season >= 2022` rows, so it
never flips live -- see above) and was not repointed at this new source.
Wiring it would first require: deduplicating same-window repeat mentions,
resolving anaphoric network references, and accumulating several weeks of
`--discover` runs before any team has three trailing observations under
the overlay's own `N=3` rule -- none of that was attempted this lane.

### 2026-09-11, later the same night: the first scheduled `--discover` run mis-ingested a preseason post

**Measured**: the first live scheduler run of `--discover` picked
`network-preseason-ratings-down-rams-chiefs-decade-high.html` (a
2026-08-20 NFL Network preseason-ratings recap) and produced a row
labelled `season=2026, week=1` with `team_a=ARI, team_b=DAL, viewers=7,000,000`
from the sentence "Arizona Cardinals-Carolina Panthers averaged 7.0
million viewers, up 1% from Steelers-Cowboys last year (6.9 million)" --
two compounding defects: (1) nothing in the ingester recognised the post
itself as preseason content, so its games were labelled as if they were
2026 regular-season Week 1; (2) `_find_teams` collected every team
nickname anywhere in the sentence in `TEAM_NAMES` **dict order**, not
text order, so "Cowboys" (from the trailing "compared to last year"
clause) sorted ahead of "Panthers" (the real, primary-clause opponent)
purely because `Cowboys` happens to sit earlier in that dict than
`Panthers` -- the same class of bug could have misattributed a real
in-season game's teams too, not only a preseason one.

Three independent fixes were made, not just the one needed to clear this
specific article, because each guards a different failure mode:

1. `EXCLUDED_LINK_TOKENS` gained `"preseason"`, so `--discover` never
   selects a preseason-titled post's URL in the first place; `ingest_article`
   also checks the fetched title/URL directly (`_is_preseason`) and writes
   a `status: skipped_preseason_post` manifest instead of processing it,
   so an explicit `--url` pointed at a preseason post is refused the same
   way.
2. `_find_teams` now sorts by each match's **character position** in the
   text, not dict-insertion order, and team resolution runs only against
   `_primary_clause(window_text)` -- the text up to (not including) the
   first `up|down ... from` or `compared` comparison marker -- so a
   "compared to last year" or "up/down N% from last year's X-Y game"
   clause can no longer supply the recorded teams.
3. Every row with both teams identified is now checked against the
   newest `data/raw/*/schedules.parquet` (`validate_against_schedule`):
   an exact-week match for a row with a known `week` (`REG` games only),
   or an any-week match within the season when `week` is `None`. A row
   whose team pair matches no scheduled game is dropped from
   `rows.parquet` into a sibling `dropped_rows.parquet` and counted in
   the manifest's new `rows_dropped_schedule_mismatch` field -- nothing
   is silently discarded. This is deliberate defense in depth: even if a
   future post's preseason/regular-season status is misjudged, or a
   different clause pattern reintroduces a comparison-clause team, an
   invented pair can no longer reach `rows.parquet` unnoticed.

**Verified directly** (no test file added, per the moratorium):
`_primary_clause` on the exact offending sentence now yields
`"Arizona Cardinals-Carolina Panthers averaged 7.0 million viewers, "`
and `_find_teams` on that clause returns `["ARI", "CAR"]` (Panthers, not
Cowboys); `_is_preseason` returns `True` for the offending article's
title and URL.

**Cleanup and re-run**: deleted the bad snapshot directory
(`data/raw/tv_audiences/20260911T043740Z`) and removed the preseason
URL from `data/raw/tv_audiences/ingested_urls.json`, then re-ran:

```
.\.tools\uv.exe run --no-sync python scripts\ingest_tv_audiences.py --discover
```

`--discover` this time correctly skipped the preseason post (per fix 1)
and advanced to the next tagged post,
`nfl-scores-second-best-regular-season-viewership-average-ever.html`
(the 2026-01-08 season-wrap article, `season=2025` inferred, no single
`week`), producing 2 rows, 0 dropped:

| season | week | network | window | team_a | team_b | viewers | schedule_validated | source sentence |
|---|---|---|---|---|---|---|---|---|
| 2025 | (none) | (none) | UNSPECIFIED | (none) | (none) | 18,700,000 | n/a (no teams) | "Highlights include a 10% viewership increase over the 2024 regular season and an average of 18.7 million viewers per game." |
| 2025 | (none) | (none) | UNSPECIFIED | KC | DAL | 57,300,000 | True (season-level match, no single `week` on this article) | "The average viewership was boosted by the all-time record Thanksgiving audience that saw more people watch a regular-season game than ever before, when 57.3 million people watched the Chiefs-Cowboys game." |

`--output` stayed optional: the scheduler job `tv_audiences_tue` can keep
calling `--discover` with a static argv and a fresh stamped directory is
created under `data/raw/tv_audiences/` each run.
