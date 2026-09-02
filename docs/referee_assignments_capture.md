# Weekly officiating-crew assignment capture (WP22)

Written 2026-09-01. Every claim below is labeled per `AGENTS.md`'s binding
rule: **measured** (a command run this session, command/output given),
**read** (a file opened this session, path[:line] given), **reported** (a web
fact, URL given, not independently re-verified beyond the fetch itself), or
**inferred** (reasoning, labeled as such).

## Binding closing-grounds taxonomy (restated verbatim, applies to any future
verdict this doc's design informs)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds ever
close a line of work: (1) refuted mechanism — a RESOLVED wrong sign (whole
interval on the wrong side of zero) or zero split-half reliability; (2)
bounded by a positive control proven able to detect an effect that size.
Everything else is `unresolved_below_power`: record it with `nfl-ats
weak-signals record`, report `probability_positive`, never the binary
"contains zero". The registry code hard-rejects inadmissible closures; if a
record command errors, the verdict is wrong, not the validator.

Nothing in this document runs an experiment or records a verdict — it
predeclares/builds a CAPTURE only, per the WP22 task scope — but the taxonomy
is restated here because Section 5 below discusses what it would take to
score the two existing `penalty_crew_tendencies` cells prospectively, and any
future session that does so must record through the two admissible commands,
never through prose.

## 1. Why

`docs/referee_battery.md` and `docs/penalty_crew_tendencies.md` (**read**, in
full) found a reliable crew-level penalty-rate trait (`mean_total`, MEASURED
split-half year-over-year Pearson r **+0.370**, 158 referee-season pairs) and,
building on it, two cells at `probability_positive` ≈ 0.90-0.92:

- **Cell C, `penalty_crew_holding_tilt_run_heavy`** — home team in the BOTTOM
  quartile of prior-rolling pregame pass rate (run-heavy) AND the game's
  referee's PRIOR-season Offensive Holding rate in the TOP quartile.
  **Read**, `docs/penalty_crew_tendencies.md` cell C: week-blocked effect
  **+0.1390** accuracy points, 95% **[-0.0595, +0.3415]**, **P+ = 0.9015**,
  built on Offensive Holding's own real split-half reliability (**+0.3226**,
  158 pairs), consistent direction across both measured eras. Registry key
  `penalty_crew_holding_tilt_run_heavy`; flag builder
  `_flag_referee_holding_tilt_run_heavy` / registry name
  `referee_holding_tilt_run_heavy` (`src/nfl_ats/experiment_runner.py:1853`).
- **Cell A, `penalty_crew_high_flag_heavy_underdog_opener`** — home team
  getting >= 7 points at the OPENER (heavy underdog) AND the game's
  referee's PRIOR-season `mean_total` (overall penalty rate) in the TOP
  quartile, graded at the opener per `AGENTS.md`'s binding "grade the
  decision at the opener" rule. **Read**, same doc, cell A: week-blocked
  effect **+0.1056** accuracy points, 95% **[-0.0351, +0.2320]**, **P+ =
  0.9204**, stable magnitude across both measured opener-era halves despite
  a thin n_flag=9/era. Registry key
  `penalty_crew_high_flag_heavy_underdog_opener`; flag builder
  `_flag_referee_high_flag_heavy_underdog` / registry name
  `referee_high_flag_heavy_underdog` (`src/nfl_ats/experiment_runner.py:1749`).

Both are correctly `unresolved_below_power` (P+ is a promotion bar, never a
decision bar, and `AGENTS.md` is explicit that a P+ this high on a forced-pick
pool is "the other side of an 87/13 bet", not caution) — but both are
**post-hoc, PBP-joined constructs**: `_build_referee_trait_data`/
`_build_referee_type_trait_data` (`src/nfl_ats/experiment_runner.py:1281`,
`:1652`) key entirely off `data/raw/officials/*/officials.parquet`
(`nflreadpy.load_officials()`), which only carries a referee assignment for a
game **after nflverse has ingested it** — i.e., after the game exists in the
historical record. **Read**, ROADMAP.md line 193 (PBP-07, "Penalty state"):
its own text does not use the word "post-hoc" — that framing was this
package's own paraphrase, corrected here rather than left as a false direct
quote. The row that actually says this in so many words, **read**, is line
199, **PBP-10 ("Referee effects")**: `⬜` status, definition of done "**Only
if assignments are point-in-time and effects survive shrinkage**" — i.e. the
project's own roadmap already named point-in-time assignment capture as the
explicit precondition for this whole line of work, before this package
existed to satisfy it. Nothing in this repository, before this work package,
captured the UPCOMING week's officiating assignment, so neither
`penalty_crew_tendencies` cell could ever be played or tracked prospectively
— only re-measured on an ever-larger historical slate. This package builds
that capture (the first half of PBP-10's stated precondition); Section 5
below states plainly what ELSE would still be needed to turn it into a
scored prospective challenger (not built here — that is future work,
correctly out of this package's scope), and the second half of PBP-10's
precondition ("effects survive shrinkage") is entirely unaddressed by this
capture and remains open.

## 2. Source survey (measured this session, 2026-09-01)

### Primary: Football Zebras (`https://www.footballzebras.com/`)

- **robots.txt** (measured, `curl`): `Disallow: /wp-admin/` only, with
  `Allow: /wp-admin/admin-ajax.php` explicitly re-permitted, no `Crawl-delay`.
  Nothing under `/category/` or `/{season}/...` is blocked.
- **URL pattern**: one post per REG week, `https://www.footballzebras.com/
  {season}/{month}/week-{N}-referee-assignments-{season}/`. **MEASURED
  unreliable to construct directly**: fetched 10 real 2025-season posts
  (weeks 1, 5, 8, 9, 10, 14, 15, 16, 17, 18) — nine matched the pattern
  exactly, but Week 5's real slug was `week-5-referee-assignments-5` (a
  numeric disambiguator, not `-2025`). A direct-guess-only strategy would
  have silently missed that week.
- **Discovery mechanism (primary strategy)**: `https://www.footballzebras.com
  /category/assignments/`, a reverse-chronological index of every
  "assignments"-tagged post. **Measured**: its current first page (fetched
  2026-09-01) lists 2025-season weeks 8 through 18 plus playoff posts;
  `find_week_url` (`src/nfl_ats/referee_assignments_capture.py`) regex-matches
  `href="...week-{N}-referee-assignments..."`, preferring a URL whose path
  year equals the target season (every observed REG post embeds the season,
  not the calendar publish year — Week 18's Jan-kickoff games were posted
  under `/2025/12/...`). A direct URL guess (`_guess_urls`, built from the
  target week's own schedule kickoff month) is tried only as a defensive
  fallback when the index does not (yet) list the week at all.
- **Structure** (measured against real fetches, both saved verbatim under
  `tests/fixtures/`): server-rendered plain HTML (WordPress). The post body
  contains an `assignment_list` block of repeated `b_post` divs:
  `b_post-game` ("Team at Team", or "Team vs. Team" for a designated-home
  international game — both forms list the away team first) and
  `b_post-referee` (a plain "First Last" name). **MEASURED**: the site emits
  BOTH single- and double-quoted class attributes across different posts
  (Week 10, 2025 is double-quoted; Week 18, 2025 is single-quoted — both
  fixtures preserved verbatim) and decorates team names with a
  `<sup>seed</sup>` playoff-seed annotation and/or a trailing `*` footnote
  marker in the season's final week. The parser strips both before nickname
  lookup and never trusts the "at"/"vs." reading order as authoritative —
  see Section 3.
- **Publication timing** (measured, `article:published_time` meta tag, 10
  sampled 2025-season posts, converted to America/New_York):

  | Week | Published (ET) | Day |
  |---|---|---|
  | 1 | Tue 16:53 | Tuesday |
  | 5 | Tue 19:57 | Tuesday |
  | 8 | Wed 12:42 | **Wednesday** |
  | 9 | Wed 12:39 | **Wednesday** |
  | 10 | Tue 20:33 | Tuesday |
  | 14 | Tue 21:27 | Tuesday |
  | 15 | Tue 19:21 | Tuesday |
  | 16 | Tue 18:51 | Tuesday |
  | 17 | Tue 17:43 | Tuesday |
  | 18 | Mon 12:44 | Monday (compressed regular-season finale) |

  **Never measured before Tuesday afternoon** in a normal week (Week 18's
  Monday post is the season-finale's compressed schedule, not a counter-
  example to the general pattern), and twice landed Wednesday around
  midday. This is a load-bearing, measured constraint, not a scheduling
  convenience: it means a captured assignment can essentially **never** feed
  the Tuesday-lock/opener card (`odds_tue_open`'s own grade, "the grade the
  pool settles on") — only a later-week refresh, up to each game's own
  `min(kickoff, Sunday 16:00 ET)` deadline
  (`C:/Users/Ryan/.claude/projects/F--Repos-nfl-py3/memory/
  picks-lock-at-kickoff.md`, owner rule). Both `penalty_crew_tendencies`
  cells above were predeclared and measured at the OPENER grade (Cell A
  explicitly; Cell C's population is opener-graded too, per that doc's
  Section 4 "Blocking: week primary, season secondary. Grade: mostly
  `close`, one cell `opener`" line and Cell A's own `grade="opener"`) — so a
  prospective score built from this capture would necessarily be graded at
  whatever LINE is current at the time of the late-week refresh, not the
  opener the historical measurement used. That is a real, honest mismatch a
  future prospective build must state plainly, not smooth over.
- **2026 season state** (measured, 2026-09-01): direct fetch of the guessed
  `https://www.footballzebras.com/2026/08/week-1-referee-assignments-2026/`
  returned **HTTP 404**; the category index's real current listing does
  **not** contain a `week-1-referee-assignments-2026` link either — only
  `referee-assignments-for-preseason-week-1-2026` and `-week-2-2026` exist so
  far (a different slug shape for preseason posts). This is the exact,
  genuine `not_yet_published` state a live run of this capture hits today —
  confirmed end-to-end (Section 6).

### Considered and ruled out: NFL Football Operations (`operations.nfl.com`)

- **Measured**: `operations.nfl.com/robots.txt` itself returns a Next.js
  `__next_error__` 404 page (`<html id="__next_error__">`), i.e. the site is
  entirely client-rendered — there is no server-delivered content this
  fetch model (plain HTTP GET, no JS execution) can read at all, on ANY
  page, not just this one.
- **Measured**: `operations.nfl.com/officiating/` returns HTTP 404. Web
  searches for a weekly officiating-assignments page on this domain
  (`operations.nfl.com officiating assignments week referee`, `"operations.
  nfl.com" "officiating" weekly assignments schedule referee crew page`)
  surfaced only rules/development content
  (`operations.nfl.com/officiating/the-officials/officiating-development`)
  and third-party (Football Zebras) results — **no evidence operations.nfl.com
  publishes a weekly per-game officiating assignment at all**, not merely
  that it is hard to reach. Not recommended as primary or fallback.

### Considered and not found: an independent third-party mirror

Web searches for a structured, server-rendered, per-game officiating listing
on ProFootballTalk, VSiN, and SportsbookWire (`"referee assignments" week NFL
2025 site:profootballtalk.nbcsports.com OR site:sportsbookwire.com OR
site:vsin.com`) returned only prose betting-preview mentions of a game's
referee, never a structured per-game table. **No genuinely independent
second SOURCE was found** — this is reported plainly rather than papered
over with a nominal fallback that would not actually survive Football
Zebras' own markup changing. The module's resilience instead comes from
having TWO strategies against the same site (category-index discovery, plus
a direct-URL-guess fallback for when the index has not yet caught up) —
weaker than a truly independent second source, but real: the two failure
modes (index markup breaks vs. index simply hasn't listed the week yet) are
different enough that this still catches the case Section 2's own Week 5
slug irregularity demonstrates matters in practice.

## 3. Referee-name join to the historical crew traits (measured)

`docs/referee_battery.md`'s and `docs/penalty_crew_tendencies.md`'s flag
builders key on `data/raw/officials/*/officials.parquet`'s own
`official_name` field (`nflreadpy.load_officials()`, filtered
`position == "Referee"`, `season_type == "REG"`;
`src/nfl_ats/experiment_runner.py:1299-1302`, **read**), a plain "First Last"
string. Football Zebras' `b_post-referee` cell uses the same convention.

**Measured** this session: fetched Football Zebras' own 2026-season crew
roster (`https://www.footballzebras.com/2026/08/officiating-crews-for-the-
2026-season/`, 200), extracted the head referee ("R" row) of each of its 17
crews, and joined against the 29 distinct `official_name` values in
`data/raw/officials/20260819T190537Z/officials.parquet`
(`position == "Referee"`, 2015-2025):

- **16 of 17 (94.1%) match exactly.**
- **The lone miss: "Ron Torbert"** (Football Zebras) vs. **"Ronald Torbert"**
  (nflverse `official_name`) — a real mismatch, also confirmed in-context in
  the live "Bills at Dolphins" row of the Week 10, 2025 fixture
  (`tests/fixtures/footballzebras_week10_2025_referee_assignments.html`) and
  reproduced against a genuine live capture this session (`data/players/
  referee_assignments/20260901T193057Z/`, Week 18 2025, real network fetch —
  the `NO`/`ATL` row's `referee_source_name` is "Ron Torbert",
  `referee` resolves to "Ronald Torbert").
- `REFEREE_NAME_ALIASES` (`src/nfl_ats/referee_assignments_capture.py`) maps
  this one known case explicitly — not fuzzy-matched, so the join stays
  exact and auditable — bringing the measured match rate to **17/17
  (100%)**. `tests/test_referee_assignments_capture.py::
  test_current_crew_names_join_the_historical_officials_snapshot` pins this
  measurement so a future officials.parquet refresh or roster change cannot
  silently break the join without a test failing.

Two other measured data-quality notes, neither new to this session but worth
restating since they bear on the same join: `official_name` (not
`official_id`) is the correct join key — **read**, `docs/referee_battery.md`:
"`official_id` is NOT stable for the same person across nflverse-data's own
history (16 of 29 referees carry two different `official_id` values across
their careers)" — and `officials.parquet` itself already contains a
within-dataset spelling variant unrelated to this capture ("Brad Rogers" vs.
"Bradley Rogers" for the same person across different seasons, **measured**
this session by inspecting the raw snapshot) — out of scope for this package
to fix, noted here only so a future session does not mistake it for a bug in
this capture's own alias table.

## 4. Build

- **`src/nfl_ats/referee_assignments_capture.py`** — the fetch/parse/write
  module. `run_capture(season, week, out_root, ...)` resolves the live
  (season, REG week) via `resolve_current_reg_week` (imported verbatim from
  `scripts/ingest_nflcom_injuries.py`, the same reuse
  `src/nfl_ats/inactives_capture.py` already makes for the identical reason)
  when not given explicitly, fetches the category index, resolves the post
  URL (Section 2), parses it (`parse_assignment_page`), joins each row's
  unordered team-code pair against the newest `data/raw/*/schedules.parquet`
  snapshot for `(game_id, home_team, away_team)` (never trusting the source
  page's "at"/"vs." order — confirmed correct against the real 2025 Week 10
  Falcons-at-Berlin game: the local schedule snapshot shows `home_team=IND,
  away_team=ATL, location=Neutral` for the page's own "Falcons vs. Colts"
  listing, i.e. the source's away-first convention holds even for the
  designated-home international form, but the join does not depend on that
  holding), and writes an immutable snapshot: `assignments.parquet`,
  `manifest.json`, `category_index.html`, and `post.html` (when a post was
  actually fetched) under `data/players/referee_assignments/<UTC ts>/`.
  Team-name mapping (`NICKNAME_TO_CODE`) and HTML-stripping (`strip_html`)
  are imported verbatim from `scripts/ingest_nflcom_injuries.py`, same reuse
  rationale as above.
- **`scripts/capture_referee_assignments.py`** — thin CLI wrapper, mirrors
  `scripts/capture_inactives.py`'s own precedent exactly (a plain script the
  scheduler invokes as a subprocess, since `src/nfl_ats/cli.py` is out of
  scope for this work package).
- **`assignments.parquet` schema**: `captured_at_utc, season, week, game_id,
  home_team, away_team, referee, referee_source_name, crew_number,
  game_day_label, source_url`. `referee` is alias-resolved to match
  `official_name`'s convention (Section 3); `referee_source_name` preserves
  the page's own text for audit. `crew_number` is always null — Football
  Zebras' weekly post does not publish a numbered crew designation (only the
  season crew-roster page assigns a personnel/jersey number to each
  official, a different concept), so the column exists to match the task's
  requested schema shape but carries no data from this source.
- **`empty_reason` values** (zero-row snapshot): `no_schedule_snapshot` /
  `no_upcoming_reg_kickoff` (schedule cannot resolve a live week at all,
  `ok=True`), `not_yet_published` (index loaded fine, week simply not listed
  yet — the genuine, measured, current 2026-Week-1 state, `ok=True`),
  `unrecognized_page_structure` (index says the post exists but it could not
  be read/parsed — a bug to fix, `ok=False`),
  `primary_and_category_fetch_failed` (the index itself could not be
  fetched, `ok=False`).

## 5. What the two P+ ≈ 0.9 cells would need to become prospective challengers

**Not built in this package** — this is a design note for a future session,
written so it does not have to be re-derived. The taxonomy in this doc's
header applies to any verdict such a session eventually records.

This capture supplies exactly one missing input: **who is officiating an
upcoming game** (`referee`, joined to `home_team`/`away_team`/`game_id`).
Both cells' flag builders (`_flag_referee_high_flag_heavy_underdog`,
`_flag_referee_holding_tilt_run_heavy`, `src/nfl_ats/experiment_runner.py`)
otherwise already read only PREGAME-safe, already-computed inputs for a
FUTURE game:

- Cell A additionally needs the opener spread (`spread_line <= -threshold`)
  — already captured by `odds_tue_open`.
- Cell C additionally needs the home team's prior-rolling pregame pass rate
  quartile (`_merge_home_pass_rate_quartile`, reading
  `game_features_pbp.parquet`'s `home_pbp_off_pass_rate`, an EWMA of games
  strictly before the one being scored) — already pregame-safe and already
  computed for every scheduled game once its home team has a game history.

**What is genuinely missing, beyond this capture, to score either cell on a
real upcoming game:**

1. **A referee-level prior-season trait lookup that does not require the
   CURRENT game to already be in the PBP/officials.parquet historical join.**
   `_build_referee_trait_data`/`_build_referee_type_trait_data`
   (`src/nfl_ats/experiment_runner.py:1281`, `:1652`) compute a referee's
   `mean_total`/Offensive-Holding-rate quartile by joining
   `officials.parquet` to COMPLETED games via the schedules `old_game_id`
   crosswalk — that join is definitionally empty for a game that has not
   been played. The needed adapter is narrow: given this capture's
   `referee` name for an upcoming game, look up that SAME referee's most
   recent COMPLETED season's aggregate from the existing historical
   officials/game-penalty-type snapshots. This is the identical "PRIOR
   season" lag both cells already use — extended one hop forward to a
   not-yet-played game — so it introduces no new leakage surface, only a
   new lookup path into data that already exists.
2. **Frozen quartile cutpoints, not a freshly recomputed `pd.qcut`.** Both
   builders currently call `pd.qcut(..., 4)` over the WHOLE historical
   population inside the same function that also does the game join; a
   prospective single-game score cannot re-run `qcut` over a population that
   does not include the future game. The boundary values from the last
   completed-season population would need to be persisted (e.g. alongside
   the officials snapshot, or as a small derived artifact) and reused to
   bucket one referee's value for scoring, rather than recomputed inline.
3. **A registry-family decision for a prospective variant.** Per `AGENTS.md`,
   the family must be declared BEFORE any sign is seen. Neither existing
   spec (`registry/experiment_specs/penalty_crew_high_flag_heavy_underdog_
   opener.json`, `registry/experiment_specs/penalty_crew_holding_tilt_run_
   heavy.json`) is a prospective design; a new family name (analogous to how
   `docs/inactives_channel.md` proposes `inactives_channel_*` for its own
   not-yet-built prospective arm) would need to be predeclared before this
   capture's first real referee assignment is scored against either cell,
   and any verdict would flow through `nfl-ats weak-signals record` /
   `nfl-ats rotation record-look` — never through prose in a doc.
4. **The opener-grade mismatch from Section 2 must be decided, not ignored.**
   Both cells were measured at the OPENER grade; this capture can only ever
   supply a mid-week-or-later referee assignment (Section 2). A prospective
   score is therefore necessarily a DIFFERENT construct from the measured
   historical cell — either (a) graded at whatever line is current when the
   late-week refresh runs (a `close`-flavoured grade for what was measured
   as `opener`, the exact substitution `AGENTS.md`'s "grade at the opener"
   section warns inverts the project's stated priority if used to reject a
   candidate), or (b) explicitly framed as a genuinely new, separately-named
   family that measures the SAME mechanism under a LATE-WEEK grade rather
   than claiming to reproduce the opener-graded historical number. Point (3)
   above is the place that framing decision belongs.

None of this is a criticism of the two cells' existing historical
measurement — `unresolved_below_power` at P+ 0.90-0.92 stands as measured,
and per `AGENTS.md` a promotion bar is not a decision bar. It is the honest
gap between "this capture exists" and "these two cells are prospectively
playable", stated so the next session does not have to re-discover it.

## 6. Tests and gates (measured, this session)

`tests/test_referee_assignments_capture.py` (23 tests): parse correctness
against two real, trimmed fixtures (`tests/fixtures/
footballzebras_week10_2025_referee_assignments.html`, double-quoted, both
"at" and "vs." matchup forms, three day headers, the real Ron/Ronald Torbert
mismatch; `tests/fixtures/
footballzebras_week18_2025_referee_assignments_excerpt.html`, single-quoted,
`<sup>` seed tags, `*` footnote marker), the measured historical join rate
(Section 3), `find_week_url`'s category-index discovery (including the real,
current absence of a 2026 Week 1 link), every `empty_reason` branch and which
ones must exit non-zero, schedule-derived `game_id`/`home_team`/`away_team`
resolution by unordered team pair (including an intentionally-unmatched pair
that must resolve to `None`, not crash), manifest field presence, and the
scheduler dedupe/naming contract `referee_assignments_wed` depends on
(matching `player_arrests_tue`'s and the `inactives_*` rows' own precedent).
Plus 22 additive pins in `tests/test_capture_scheduler.py` from the pre-WP22
suite (all passing).

**Measured**, this session:

```
.\.tools\uv.exe run --no-sync pytest tests/test_referee_assignments_capture.py tests/test_capture_scheduler.py \
    -p no:cacheprovider --basetemp=<scratch> -q
# 45 passed
.\.tools\uv.exe run --no-sync ruff format --check <files>   # 5 files already formatted
.\.tools\uv.exe run --no-sync ruff check <files>             # All checks passed!
.\.tools\uv.exe run --no-sync mypy src                       # 0 errors in referee_assignments_capture.py
                                                               # (6 pre-existing errors elsewhere, unrelated,
                                                               # in untracked files from other in-flight work)
```

**End-to-end live validation** (measured, real network, this session, not a
fixture): `.\.tools\uv.exe run --no-sync python scripts\capture_referee_assignments.py --current`
resolved season=2026 week=1, fetched the real category index (200) and the
real guessed post URL (404, confirming Section 2's "not yet published"
finding), and wrote a correct zero-row `not_yet_published` snapshot
(`ok=True`). A second real run,
`--season 2025 --week 18`, found the real listed URL via the category index,
fetched and parsed it (200), and produced 16/16 correctly schedule-joined
rows including the real "Ron Torbert" → "Ronald Torbert" alias resolution —
both runs' output live locally under `data/players/referee_assignments/`
(gitignored, not committed).

## 7. Files

- `src/nfl_ats/referee_assignments_capture.py` — the capture module.
- `scripts/capture_referee_assignments.py` — thin CLI wrapper.
- `tests/test_referee_assignments_capture.py` — tests (23).
- `tests/fixtures/footballzebras_week10_2025_referee_assignments.html`,
  `tests/fixtures/footballzebras_week18_2025_referee_assignments_excerpt.html`,
  `tests/fixtures/footballzebras_category_assignments_index.html` — real,
  trimmed, verbatim fixtures.
- `scripts/capture_scheduler.py` — one additive `Job` row,
  `referee_assignments_wed`, appended at the end of `SCHEDULE`.
- `tests/test_capture_scheduler.py` — two additive pins for that row.
- `docs/capture_scheduling.md` — additive table row, job count, and a new
  "The weekly referee-assignments capture (WP22)" section.
- This document.
## Late-week crew-tilt challenger predeclaration (2026-09-01, WP47)

**Written BEFORE any line of `src/nfl_ats/crew_tilt_refresh_overlay.py`
existed and before any 2026 or back-test number was computed.** This section
turns Section 5's "what would still be needed" list into a frozen rule. It
predeclares one prospective challenger, `crew_tilt_refresh_v1`, and nothing
else: no production pick changes, no promotion, no rotation-registry window
spent. Provenance tags per `AGENTS.md` are used throughout — **measured**,
**read** (path:line), **reported** (unverified), **inferred**.

### The closing-grounds taxonomy still binds here

Restated verbatim because this section's back-test produces a number that a
future reader will be tempted to treat as a gate. An interval or CI that
contains zero is NEVER grounds to reject, fail, or close an experiment. At
this evaluator's ~2-point resolution, "contains zero" is the EXPECTED outcome
for a real small signal. Only two grounds ever close a line of work: (1)
refuted mechanism — a RESOLVED wrong sign (whole interval on the wrong side
of zero) or zero split-half reliability; (2) bounded by a positive control
proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`,
report `probability_positive`, never the binary "contains zero".

### 1. Challenger identity

- **`challenger_id`**: `crew_tilt_refresh_v1`.
- **Comparator / incumbent**: the frozen **Tuesday card** — the played
  production-chain pick for each game, as it stands at the refresh pass
  (`nfl_ats.pick_refresh.RefreshedGame.new_pick_side`, i.e. after the
  four-overlay union policy
  `overlay_union_coach_division_revenge_player_arrests_spread_gap_v1` and
  after the observed-movement policy has had its say). Paired per game; this
  is a two-arm comparison, never a standalone accuracy claim.
- **Scoring**: `nfl-ats prospective-score`, **both grades** —
  `decision_line` primary (the frozen Tuesday opener line the pool grades
  against, POL-11) and `close_line` secondary.
- **Status at registration**: `ACTIVE_PROSPECTIVE`. Paper evidence at zero
  window cost.

### 2. Overlay semantics (frozen)

At each `nfl-ats refresh-picks` pass, for every game in the pass:

1. **Find the crew.** Take the NEWEST `data/players/referee_assignments/
   <stamp>/assignments.parquet` snapshot that (a) covers the pass's
   `(season, week)`, and (b) whose `captured_at_utc` is strictly BEFORE that
   game's own pick deadline `min(own kickoff, Sunday 16:00 ET)` — the
   deadline `nfl_ats.pick_refresh.pick_deadline` already computes and
   `RefreshedGame.deadline` already carries (**read**,
   `src/nfl_ats/pick_refresh.py:151-156`). Anything else is out of window.
2. **No in-window row → keep the Tuesday pick.** A game with no snapshot, no
   row in the snapshot, a snapshot captured at or after that game's own
   deadline, or a referee name that does not resolve to the historical crew
   traits is a documented NO-OP: zero tilt, the challenger's pick IS the
   incumbent's pick, and the row is tagged with the reason. Never an
   exception, never a fallback to a different information set.
3. **Compute the two cells' flags from the SEASON-LAGGED crew traits,
   exactly as the screen did.** The screen's own builders are imported, not
   reimplemented: `_build_referee_type_trait_data(repo_root, "Offensive
   Holding")` and `_build_referee_trait_data(repo_root)`
   (**read**, `src/nfl_ats/experiment_runner.py:1281`, `:1652`), plus
   `_merge_home_pass_rate_quartile` (**read**, same file, `:1711`) for the
   home team's prior-rolling pregame pass-rate quartile and
   `_HEAVY_UNDERDOG_THRESHOLD_DEFAULT = 7.0` (**read**, same file, `:1614`)
   for the heavy-underdog cut. The one thing those builders structurally
   cannot do — Section 5 item 1's forward hop — is supplied by a small,
   pinned adapter in this module: the referee's PRIOR completed season's
   mean rate, bucketed against the **frozen** quartile cutpoints of the
   builder's own lagged population (Section 5 item 2). A test measures that
   this adapter reproduces the builder's `lag_type_quartile` /
   `lag_penalty_rate_quartile` exactly on every historical (referee, season)
   pair; it is a pinned second path, not a second definition.
   - **Cell C flag** (`penalty_crew_holding_tilt_run_heavy`): home team's
     `home_pbp_off_pass_rate` quartile == 1 (BOTTOM = run-heavy) AND the
     game's referee's PRIOR-season Offensive Holding rate quartile == 4.
   - **Cell A flag** (`penalty_crew_high_flag_heavy_underdog_opener`): the
     game's referee's PRIOR-season `mean_total` penalty-rate quartile == 4
     AND the home team is a heavy underdog at the **frozen Tuesday line**
     (`decision_home_spread <= -7.0`; nflverse sign convention, positive =
     home favored, the same convention `prospective_scoring` settles with —
     **read**, `src/nfl_ats/prospective_scoring.py:206`).
4. **Tilt the production home-cover probability by each cell's own measured
   per-game gap, in the cell's own direction.** The base is
   `RefreshedGame.new_home_cover_probability`, the production probability at
   the refresh pass. The tilt sizes are NOT chosen here; they are each
   cell's own measured raw per-game cover-rate gap:

   | cell | registry entry | field | value | sign | tilt applied to P(home cover) |
   |---|---|---|---|---|---|
   | C | `penalty_crew_holding_tilt_run_heavy` | `result.raw_gap_pct`, `artifacts/experiment_runner/20260820T113432Z/metadata.json` | `+5.994893289010933` pts | `-1` (`classification_evidence`: "sign=-1") | **-0.05994893289010933** |
   | A | `penalty_crew_high_flag_heavy_underdog_opener` | `result.raw_gap_pct`, `artifacts/experiment_runner/20260820T113443Z/metadata.json` | `+16.772226131832042` pts | `+1` (`classification_evidence`: "sign=+1") | **+0.16772226131832042** |

   The registry stores `raw_gap_pct` already multiplied by the construct's
   `sign` (**read**, `src/nfl_ats/experiment_runner.py:3629-3630`:
   `raw_gap_pct = construct.sign * (subset_cover - complement_cover) * 100`),
   so the signed home-cover gap is `sign * raw_gap_pct`, which is what the
   tilt column above carries. Both numbers are the cells' **unscaled**
   per-game gaps, not the full-slate-scaled `effect` (`+0.1390` / `+0.1056`
   accuracy points) — the scaled figure answers "what does this do to the
   whole card", the raw gap answers "how much does one flagged game's home
   cover rate move", and only the second is a per-game probability tilt. A
   test asserts the module's constants equal these artifact/registry values
   to the last digit; an underived constant here would be a defect
   (`C:/Users/Ryan/.claude/projects/F--Repos-nfl-py3/memory/underived-constants-are-wrong.md`).
   - **Disclosure, stated before any number was seen:** the complement in
     both cells is the screen's own "everyone else" team-game population
     (both sides of unflagged games), not a home-only baseline. The gap is
     therefore the cell's measured contrast as recorded, used as a
     probability tilt by construction, not a re-derived home-vs-home effect.
5. **Both cells compose ADDITIVELY.** `tilt_points = (cell C tilt if C
   flagged else 0) + (cell A tilt if A flagged else 0)`, and
   `tilted_home_cover_probability = clip(production_probability +
   tilt_points, 0.0, 1.0)`. **The composition was NEVER measured.** No
   experiment in this repository has scored the two cells jointly; additivity
   is an assumption frozen here for auditability, and a game flagged by both
   cells is the one population where this challenger's rule is least
   supported by evidence. Stated plainly rather than smoothed over, per
   `AGENTS.md`'s "composition is not the signal".
6. **Derive the would-be pick.** `tilted_side = HOME if
   tilted_home_cover_probability >= 0.5 else AWAY`. The tilt FLIPS the
   played pick when — and only when — it moves the probability across 0.5
   relative to the production probability's own side
   (`RefreshedGame.model_only_pick_side`). So
   `crew_would_be_pick_side = opposite(new_pick_side)` if the tilt crossed
   0.5, else `new_pick_side` unchanged. This composition-on-the-played-side
   form is the same one `nfl_ats.nflcom_refresh_overlay` already uses
   (**read**, `src/nfl_ats/nflcom_refresh_overlay.py:67-70`), and it
   guarantees the no-flag case equals the incumbent exactly, including on
   games where the observed-movement policy moved the played pick away from
   the model's own side.
7. **The played card is never touched.** The would-be pick lives only in a
   separate append-only ledger,
   `artifacts/prospective/crew_tilt_refresh_decisions.parquet`. Nothing is
   written to `pick_revisions.parquet`, `decisions.parquet`, or the
   published card, and the `RefreshResult` handed in is consumed read-only.

### 3. Timing: why this is playable, and what it is not

- **Wednesday capture, Sunday-16:00 lock.** The `referee_assignments_wed`
  capture runs Wed 15:00 ET (Section 2). Every game's own deadline is
  `min(own kickoff, Sunday 16:00 ET)`, so a Wednesday capture is before the
  deadline of every game in a normal week **including SNF and MNF** — those
  lock EARLY at Sunday 16:00 ET, which is still after Wednesday. SNF/MNF are
  therefore playable for this channel, and the module verifies it
  per-game against `pick_deadline` rather than assuming it. A Thursday-night
  game is also in window (Wednesday precedes its kickoff), but only barely,
  and a capture that slips to Thursday afternoon (Section 2 measured two
  Wednesday-midday publications in ten sampled weeks; **inferred**: a later
  slip is possible) drops TNF from the channel that week, tagged, not
  silently.
- **Lines freeze Tuesday (POL-11).** The refreshed pick is graded at the
  FROZEN Tuesday line: `decision_home_spread` is the pairing anchor and is
  never re-picked. Cell A's heavy-underdog leg is evaluated against that
  same frozen Tuesday number, which is exactly the opener line cell A was
  measured at.
- **The grade mismatch from Section 2, resolved rather than ignored.**
  Section 5 item 4 left an open choice. It is decided here as option (b):
  `crew_tilt_refresh_v1` is a **separately-named, genuinely new construct**
  that measures the same mechanism under a LATE-WEEK decision timestamp. It
  does not claim to reproduce cell A's or cell C's historical numbers. Its
  primary grade is nonetheless `decision_line` — the frozen Tuesday opener —
  because the pick is graded at the line the pool settles on, not the line
  in the market when the refresh ran. Cell C was measured at the `close`
  grade and cell A at the `opener` grade; pooling them into one rule is a
  composition across grades and is disclosed as such.

### 4. Anti-backdating

- A snapshot whose `captured_at_utc` is at or after a game's own kickoff can
  NEVER apply to that game, and neither can one at or after its Sunday-16:00
  lock; the deadline check covers both because the deadline is the minimum
  of the two. Pinned by test.
- The recorder is opt-in (`--record-decisions`), refuses outside
  `nfl_ats.clv.refuse_if_outside_recording_lock_window`, and is append-only.
  **No paper-decision or ledger row is written by this work package.** The
  first real Week 1 write is 2026-09-08.

### 5. Stacked back-test: context, declared NOT a gate

Before it was run: the overlay is applied to the frozen paired opener
archive `artifacts/opener_evaluation/20260819T174244Z/per_game.parquet`
(1,537 REG games, 2020-2025, graded at the Tuesday opener under the
production probability rule), on top of the reconstructed four-member
production chain, with the HISTORICAL crew from the officials/PBP join
standing in for the capture. That stand-in is point-in-time-EQUIVALENT for
the crew identity specifically — assignments are known by Wednesday of the
game week (Section 2), so knowing who refereed a 2021 game is not knowing
anything a Wednesday-2021 forecaster could not have known — but it is NOT
point-in-time-equivalent in any other respect, and the seasons are MINED:
several members of the incumbent chain were themselves screened on windows
this archive re-touches. It spends no rotation-registry window.

**Verdict handling, frozen before the number existed:** the result is
recorded under the new family name `crew_tilt_stacked_on_production` as
`unresolved_below_power`, reported with `probability_positive`, UNLESS the
whole week-blocked interval sits on the wrong side of zero — in which case
it is `refuted_mechanism` with `--closing-ground wrong_sign_resolved` and
the challenger is NOT registered. No other outcome closes anything, and an
interval crossing zero is never one of them.

### 6. What this package deliberately does not do

- **No `src/nfl_ats/cli.py` hook.** The sibling `nflcom_refresh_overlay` is
  called from `refresh-picks` inside `cli.py`; that file is owned by another
  in-flight work package this session, so `record_crew_tilt_refresh_overlay`
  ships fully built and tested but is not yet called by the CLI. This is a
  declared `known_gap` on the registration, not a silent omission: the
  wiring is one call at the same place the NFL.com overlay's is made.
- **No change to `MOVEMENT_POLICY_THRESHOLD`, the four-overlay policy, the
  published card, or any registry entry other than the two additive records
  this section names.**

### Wiring status addendum (2026-09-02)

The planned `refresh-picks` hook is now present. It calls
`record_crew_tilt_refresh_overlay(..., record_decisions=args.record_decisions)`
after the sibling refresh overlays, reports its separate-ledger result, and
contains failures so they cannot interrupt the refreshed card. This remains a
late-refresh-only arm; it is not called by Tuesday publishing.
