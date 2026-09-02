# Kickoff-minus-90 inactives channel: feasibility study and predeclaration (WP5)

Written 2026-09-01. This is a feasibility study and a frozen predeclaration
only. **No experiment was run, no data was recorded, no registry or rotation
window was spent, and no scheduler job was added.** Everything below is
either `measured` (a command run this session, command/output given),
`read` (a file opened this session, path[:line] given), `reported` (a web
fact, `(web: URL)`, not independently verified), or `inferred` (reasoning,
labeled as such) — per `AGENTS.md`'s labeling rule.

## Binding closing-grounds taxonomy (restated verbatim, applies to every verdict below)

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

Nothing in this document reaches a verdict on the channel's effect — it has
not been measured yet — so this taxonomy is restated for the record and for
whichever session runs the experiment predeclared in Section 5, not because
it is applied to a result here.

## 1. Ledger row

**Read**, `docs/archive/idea_ledger.md:105` (row 85 of the consolidated
2026-08-22 idea ledger, `docs/archive/idea_ledger.md:1-18` for the ledger's
own provenance/ranking methodology). Quoted in full:

> **Lead:** Kickoff-minus-90 inactives channel: surprise absences (players
> NOT ruled out by the Fri/Sat report who record zero snaps) parsed from
> official inactives, valued by the existing injury-value model, applied
> only to games whose inactives instant precedes `pick_refresh.pick_deadline`
> (owner-backlogged 2026-09-01)
>
> **Source-doc section:** this session's registry/ledger sweep: "inactiv"
> scores ZERO hits in this ledger, pool_edge_plan, late_week_refresh, and all
> 615 registry signal names/notes (measured); scout v4 §6 spotted the T-90
> window for ROOF status only
>
> **Mechanism one-liner:** Official inactives at T-90 are the last unclaimed
> slice of the injury timeline (Tue/Fri/Sat all tested); mechanically
> unpriced in the Tuesday opener the pool grades against; injury news is the
> measured top driver of movement value (+17.07 P+ 0.976, correlated
> decomposition)
>
> **Filter class:** PRICING-GAP
>
> **Data-ready?** Y for the backtest — participation parquet + Fri/Sat
> report features reconstruct historical surprise absences; prospective
> needs new scheduler passes (Sun ~11:45, ~15:15, Thu ~19:00)
>
> **Effort:** M
>
> **Expected-value note:** Eligible games ~13 of 16/week (early slate
> 11:30→13:00 kickoff lock; late slate ~14:50→16:00 cap; TNF/Sat to
> kickoff); SNF/MNF structurally EXCLUDED — their inactives arrive after the
> Sunday 16:00 lock (owner-corrected 2026-09-01). Rare events × large
> per-game effect ⇒ expect a wide interval; record probability_positive,
> never "contains zero" (inferred design)
>
> **Status:** untested

A repo-wide grep this session (`measured`, Section 2 below documents the
search) confirms the ledger's own claim that "inactiv" scored zero hits
everywhere else at the time it was written still holds today: `inactives`
(case-insensitive) now appears in exactly `docs/archive/idea_ledger.md` (this
row), `docs/qb_news_channel.md` (twice — its QB-news keyword list includes
the literal tokens `inactive, inactives` at line 83, and a mention of "a
late-morning/early-afternoon inactives-list story" at line 180, discussing a
*different*, already-built channel: news-foreshadowing of backup QB starts,
not the official T-90 inactives list), and
`scripts/qb_backup_news_visibility.py:152` (the same keyword list). `T-90`
(and variants) appears in `docs/roof_decision_screen.md`,
`registry/weak_signals.json`, `scripts/roof_decision_screen.py`, and
`ROADMAP.md` — all about **roof status** (dome/open/closed), matching the
ledger row's own citation ("scout v4 §6 spotted the T-90 window for ROOF
status only"). **No other ledger, registry row, or built script addresses
game-day inactives.** Row 85 is the only place this idea lives in the
repository, and its status remains `untested`.

## 2. Deadline arithmetic (measured from data)

**Method (measured).** `data/raw/20260824T115346Z/schedules.parquet` is the
newest local schedule snapshot (`measured`: `ls data/raw | grep -E
'^[0-9]{8}T[0-9]{6}Z$' | sort | tail` → newest is `20260824T115346Z`; older
candidates checked: `20260817T235649Z`, `20260812T*`). For every 2026 REG
game (272 rows) — and, as a fallback shape check on a fully realized season,
every 2025 REG game (272 rows) — kickoff was computed with
`nfl_ats.features._kickoff_utc` (unchanged, imported verbatim: combines
`gameday` + `gametime`, localizes `America/New_York`, converts to UTC,
`src/nfl_ats/features.py:78-88`). The per-game deadline was computed with
`nfl_ats.pick_refresh.sunday_pick_lock` + `pick_deadline`
(`src/nfl_ats/pick_refresh.py:132-157`, unchanged, imported verbatim — the
exact functions `refresh-picks` uses in production, not a reimplementation).
`T-90 = kickoff - 90 minutes`. A game is **playable** for this channel iff
`T-90 < deadline`, i.e. the official inactives instant falls strictly before
the game's own pick deadline. The script used
(`wp5_deadline_arithmetic.py`) ran only in the scratchpad directory and
touched no repository file; its parquet outputs were deleted after the run
(this work package creates exactly one new file, this document).

**Result, 2026 REG, by slot** (measured):

| Slot | n games | playable | slack at T-90 (min) |
|---|---:|---:|---|
| Thu | 19 | 19 | +90 (all) |
| Fri (Black Friday) | 4 | 4 | +90 (all) |
| Sat | 2 | 2 | +90 (all) |
| Wed (opener) | 2 | 2 | +90 (all) |
| Sun intl early (09:30 ET) | 6 | 6 | +90 (all) |
| Sun early (13:00 ET) | 147 | 147 | +90 (all) |
| Sun late-afternoon (16:05–17:00 ET) | 58 | 58 | +65 to +85 |
| **SNF** (20:00–20:35 ET Sunday) | 17 | **0** | **−170 (all)** |
| **MNF** (20:15 ET Monday) | 17 | **0** | **−1605 (all)** |

"Slack" is `deadline − T-90` in minutes; positive means T-90 falls before
the deadline (playable), negative means after (excluded). For every
non-Sunday-evening/non-Monday slot, T-90 is exactly 90 minutes before the
deadline because the deadline equals the game's own kickoff for that slot
(`pick_deadline` returns `min(kickoff, sunday_lock)` and kickoff is the
binding term). For the Sunday late-afternoon slot, the deadline is the
week's Sunday 16:00 ET lock (binding, since 16:05/16:25/16:30/17:00 ET
kickoffs are all at or after it), so slack shrinks to 65–85 minutes — still
positive, confirming the ledger row's and the task prompt's claim that this
slot is playable, **computed, not assumed**. SNF (kickoff 20:00–20:35 ET)
and MNF (kickoff 20:15 ET Monday) are excluded exactly as the ledger row and
`docs/pick_refresh.py`'s per-game deadline design predict: their own
kickoff is after the week's 16:00 ET lock, so `pick_deadline` binds to
16:00 ET and T-90 (≈18:30–19:05 ET Sunday, ≈18:45 ET Monday) falls **after**
it by 170 minutes (SNF) to 1,605 minutes / 26.75 hours (MNF).

**Per-week playable counts, 2026 REG** (measured):

| Week | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Total games | 16 | 16 | 16 | 16 | 15 | 14 | 14 | 14 | 15 | 14 | 13 | 16 | 14 | 15 | 16 | 16 | 16 | 16 |
| Playable | 14 | 14 | 14 | 14 | 13 | 12 | 12 | 12 | 13 | 12 | 11 | 14 | 12 | 13 | 14 | 14 | 14 | 16 |

**Season total 2026: 238 / 272 playable (87.5%), average 13.2 of 16 games/week**
— close to the ledger row's own inferred "~13 of 16/week" figure, now
measured rather than estimated. **2025 fallback-shape check: 233 / 272
(85.7%)**, same slot pattern (SNF/MNF always excluded, all other slots
always playable), confirming the pattern is not an artifact of the 2026
schedule specifically. Week 18 shows 16/16 playable in 2026 because that
week (measured) carries no SNF/MNF game in this snapshot — every game is a
Sunday-or-earlier slot.

**Caveat, inferred:** this is the deadline-eligibility ceiling, not a
forecast of how many games would actually see a *changed* pick. Most
"surprise absences" (players not on the Friday/Saturday report who then
don't play) will not be surprising enough, or valuable enough by the
existing injury-value construct, to move a pick — this section only answers
"could the channel legally act on this game," per the task's Section 2
scope.

## 3. Live-source survey

**Method.** WebSearch + one WebFetch test per candidate this session
(2026-09-01), no login, no credentials. Week 1 2026 kicks off 2026-09-09/10,
so no live game window existed to fetch during this session — every fetch
below tests reachability and structure, not live T-90 content, and is
labeled accordingly.

### Primary recommendation: `nfl.com/inactives/`

- **URL pattern.** `https://www.nfl.com/inactives/` (all games) and
  `https://www.nfl.com/inactives/?team=<abbr>` (per team) — `reported (web:
  https://www.nfl.com/inactives/)`, found via WebSearch.
- **Fetch test (measured, WebFetch, this session).** `GET
  https://www.nfl.com/inactives/` returned 200, server-rendered plain HTML
  (no JS required for the visible message), currently showing a placeholder
  — `"Please check back soon for NFL Inactive Reports for this Season"` —
  because the 2026 season has not started. This is the **expected** state
  given the date, not a failure of the source.
- **robots.txt (measured, WebFetch, this session).** `GET
  https://www.nfl.com/robots.txt` lists `Disallow:` only for
  `/_ctv/`, `/_fantasy-app/`, `/_libraries/`, `/_mobile-app/`,
  `/_mobileview/`, `/_phs/`, `/account/`, `/nfl-films-beta/`, `/search/` —
  **nothing under `/inactives/`**, and no `Crawl-delay`. This matches the
  precedent already in this repo:
  `scripts/ingest_nflcom_injuries.py:8-10` (`read`) recorded the identical
  finding for `/injuries/` on 2026-08-21 and that scraper is already
  production code (`docs/nflcom_friday_refresh.md`'s integration contract,
  cited in that script's docstring, not independently re-read here).
- **Structure, inferred from the sibling scraper (not yet confirmed for
  `/inactives/` specifically, since no live data exists to inspect).**
  `ingest_nflcom_injuries.py` parses `/injuries/league/{season}/reg{week}`
  as plain HTML tables per team
  (`SECTION_SPLIT`/`TEAM_ABBR`/`TABLE`/`ROW`/`CELL` regexes,
  `scripts/ingest_nflcom_injuries.py:69-78`, `read`); `/inactives/` is the
  first-party, official source for the specific artifact this channel
  needs (the T-90 inactive list, not the Wed–Fri practice/game-status
  report `/injuries/` already provides), and is reachable by the same
  polite-crawl pattern (fetch robots.txt first, fail closed, ≥2s delay,
  immutable UTC-stamped snapshot directory) already proven in this repo.
- **Latency, reported (web, RotoWire FAQ, see fallback below, corroborating,
  not this source's own page):** "NFL inactives are released 90 minutes
  before kickoff for every game" — matches the T-90 convention already
  used throughout this repo's owner-corrected deadline rule; not
  independently timed against a live game this session (none was in
  progress).

### Fallback recommendation: RotoWire's inactives page

- **URL.** `https://www.rotowire.com/football/inactives.php` — `reported
  (web)`, found via WebSearch.
- **Fetch test (measured, WebFetch, this session).** 200, server-rendered
  plain HTML (no JS-dependent placeholder divs observed), currently
  reading `"No teams have announced their inactives for this week yet"` —
  same expected preseason gap as the primary source. Its own FAQ text
  states explicitly: **"NFL inactives are released 90 minutes before
  kickoff for every game"** — an independent, third-party confirmation of
  the T-90 convention this whole channel and the owner's per-game deadline
  rule already assume.
- **Why fallback, not primary.** Third-party aggregator, not the league's
  own first-party page; structure/markup could change without notice and
  is not covered by this repo's existing robots.txt-checked scraper
  pattern. Useful as (a) a cross-check against NFL.com's own listing and
  (b) a resilience fallback if NFL.com's markup or availability changes.

### Considered and not recommended: ESPN's public JSON API

- **Reachability (measured, WebFetch, this session).**
  `https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates=20260104`
  returned **HTTP 403 Forbidden**. The sibling host
  `https://site.web.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard`
  (no `dates` param, and with `?dates=20260104`) returned **200, JSON, no
  auth**, both measured this session. The two hostnames behave
  differently; only `site.web.api.espn.com` is usable.
- **Inactives structure — measured absent, on both a future and a
  completed game.** `GET
  .../summary?event=401872656` (NE @ SEA, 2026 Week 1, not yet played):
  200, JSON, top-level keys include `boxscore`, `injuries` (the Wed–Fri
  practice/game-status report, same category `/injuries/` already
  provides — sample rows measured: `"Amari Kight", "Injured Reserve",
  "2026-09-01T04:00Z"`), but **no `inactives` key and no
  inactive/DNP-status field inside `boxscore.players`**. `GET
  .../summary?event=401772966` (a completed 2025 Week 18 game, chosen so a
  final inactives list should exist if ESPN's API ever surfaces one): 200,
  JSON, and the `boxscore.players` arrays contain **only players who
  actually appeared in the box score** — no `didNotPlay`, `reason`, or
  `inactive` field, and no separate inactives section, even post-game.
  **Verdict: ESPN's public `summary` JSON endpoint does not carry a
  structured inactives list at all, pre- or post-game**, contradicting a
  web claim surfaced by an initial WebSearch (that a `didNotPlay` boxscore
  field exists) — that claim did not reproduce against this session's own
  fetch and is treated as unverified/wrong for this endpoint. ESPN remains
  useful for lines/odds/injury-report data already covered elsewhere in
  this repo, but is **not recommended as a source for T-90 inactives
  specifically**.

### Considered and ruled out: nflverse / `nflreadpy`

- **Measured, this session** (`./.tools/uv.exe run --no-sync python -c
  "import nflreadpy as nfl; print([f for f in dir(nfl) if
  f.startswith('load_')])"`): 22 `load_*` functions available, including
  `load_injuries` (already ingested locally,
  `data/players/raw/*/injuries.parquet`: `report_status` values are only
  `Questionable/Probable/Out/Doubtful/Note` — the Wed–Fri practice report,
  measured via `.value_counts()` — **no `Inactive` status and no
  T-90-relative timestamp**) and `load_participation`/`load_snap_counts`
  (post-game, play-level or box-score participation — not a pregame,
  timestamped inactives feed; their own docstrings, `read` via `help()`,
  describe "player involvement on specific plays" and snap counts, both
  outcome data). **No nflverse dataset carries a timestamped, pregame T-90
  inactives list.** A WebSearch for a nflverse/nflreadpy inactives dataset
  turned up nothing beyond the same `load_injuries`/`load_participation`
  pair already checked locally.

## 4. Historical proxy for backtesting

**What exists locally, measured this session:**

- `data/players/raw/20260817T184901Z/injuries.parquet` — nflverse weekly
  injury report, 79,818 rows, columns `season, game_type, team, week,
  gsis_id, position, report_status, practice_status, date_modified`. This
  is the Wed–Fri practice-report side (already the input to
  `injury_value_lost`/`injury_value_lost_tilt_overlay`, per
  `docs/prospective_evidence.md`'s Tuesday-visibility audit, `read`
  earlier this session).
- `data/players/raw/20260817T184901Z/snap_counts.parquet` — nflverse snap
  counts, 324,611 rows, columns `game_id, season, game_type, week, player,
  pfr_player_id, position, team, offense_snaps, offense_pct,
  defense_snaps, defense_pct, st_snaps, st_pct`. **This is the "actually
  played" label**: a player-game row with `offense_snaps == 0 AND
  defense_snaps == 0 AND st_snaps == 0` is, to the resolution of this
  dataset, a player who did not play that game — the closest available
  local proxy for "was inactive," reconstructed after the fact.
- `data/processed/game_features_player_learned_availability.parquet` —
  4,703 rows × 265 columns, includes the same
  `home/away/diff_injury_{offense,defense,special_teams,offensive_line,
  skill,front,secondary}_unavailability` and
  `home/away/diff_injury_{skill_epa,defense_disruption}_value_lost`
  families the injury-value overlay already uses (confirmed by column
  match against `injury_value_tilt_overlay.py`'s two named columns, `read`
  earlier this session), i.e. this table already encodes the *valuation*
  half of the construct the ledger row wants to apply to surprise
  absences; it does not itself carry a raw inactive/participation flag.

**Verdict, precisely stated:**

- **Leak-free for the label side of a backtest.** Reconstructing "which
  team-games actually saw a surprise absence" from `snap_counts.parquet`
  (post-game truth) crossed against `injuries.parquet` (Fri/Sat-dated
  report rows, point-in-time-safe by their own `date_modified`) is exactly
  analogous to how this project already uses `result` — a fully post-game
  fact — solely to **grade** a pregame pick, never as a pregame feature
  (`docs/prospective_evidence.md`'s "Two grades" section, `read`). Used
  only to build the historical evaluation population (which team-games
  were "surprise absences," which the market/report did not flag by the
  Friday/Saturday cutoff) and never fed into the pregame feature the model
  sees, this construction does not leak: it is a **grading construct**,
  not a **feature**.
- **Not leak-free if used as a pregame feature or substituted for a real
  T-90 fetch.** `snap_counts.parquet` cannot ever be part of the pregame
  input to a live pick — it is definitionally only known after the game.
  A backtest that mistakenly fed `snap_counts`-derived "inactive" status
  as of the game's own week into a pregame feature table (rather than
  using it purely to label historical rows for evaluation) would be a
  leakage bug requiring exactly the regression test class AGENTS.md
  requires for every new feature family (Section 5 below names it).
- **What is still missing for a full historical backtest, honestly
  flagged.** `snap_counts.parquet` tells you a player did not play; it
  does not by itself tell you the player was on the OFFICIAL T-90
  inactive list specifically (vs. e.g. a healthy scratch never on any
  report, a game-day injury during warmups, or a DNP for
  in-game-not-pregame reasons such as an ejection or in-game injury). The
  historical proxy is therefore a **reasonable but imperfect stand-in**
  for the true T-90 list — it will over-count zero-snap players who
  became unavailable *during* the game (impossible to separate from a
  pregame inactive using `snap_counts` alone) and cannot recover the exact
  historical T-90 instant itself (no historical NFL.com/RotoWire T-90
  snapshot archive exists locally or, per Section 3, in nflverse). This
  matters for Section 5's design: the historical arm is a **proxy
  experiment bounding the mechanism**, not a literal replay of what the
  live channel would have captured.

## 5. Predeclared design (frozen wording, no numbers)

Predeclared before any cover rate, effect, or interval is computed, per
this project's standing discipline (`docs/qb_news_channel.md`'s own
predeclaration is the template followed here).

- **Candidate.** The late-week refreshed pick (`nfl_ats.pick_refresh`,
  `docs/late_week_refresh.md`) at the frozen Tuesday line, with
  `injury_value_lost` (or the `learned_availability` construct feeding
  `weak_stack`, whichever the active model at run time actually uses per
  `artifacts/active_ats_model.json`) **recomputed after inactives are
  known** — i.e. a feature-table rebuild that incorporates the surprise-
  absence signal from Section 4's construct (historical arm) or a live
  T-90 fetch (prospective arm, Section 7), followed by a `refresh-picks`
  pass at the frozen Tuesday spread.
- **Comparator.** The Tuesday-published production pick — the standard
  `refresh-picks`-vs.-Tuesday-card comparison already used by every other
  arm in `docs/late_week_refresh.md`'s "Observed-movement pick policy"
  section.
- **Population.** Playable games only, from Section 2: every slot except
  SNF and MNF, i.e. Thu/Fri/Sat/Wed/Sun-early/Sun-late-afternoon/Sun-intl,
  measured at 238/272 (87.5%) of 2026 REG games and 233/272 (85.7%) of
  2025 REG games. SNF and MNF are excluded from this population **by
  construction**, not by later filtering — their T-90 instant is always
  after that week's Sunday 16:00 ET pick lock (Section 2), so no
  inactives-aware refresh is ever eligible to touch them.
- **Grade.** The frozen Tuesday line — opener-graded semantics, per
  `docs/late_week_refresh.md`'s "Grading vs. deciding" table (grading line
  frozen Tuesday, decided at each game's own deadline) and per
  AGENTS.md's "grade the decision at the opener" rule.
- **Metric.** Paired forced-pick accuracy delta in `accuracy_points`
  (candidate minus comparator on the identical game set), primary interval
  via `nfl_ats.clv.week_blocked_bootstrap`
  (`src/nfl_ats/clv.py:708`, week-blocked, matching every other
  registered arm in this project's registry).
- **Positive control.** The comparator's construct replaced by realised
  margin — i.e. an oracle variant of the same pipeline that substitutes
  the actual game outcome/margin for the injury-value construct's
  estimate, to establish whether this measurement instrument (this
  population size, this grading line, this bootstrap) is even capable of
  detecting an effect of the size this channel could plausibly produce.
  Mirrors the oracle-arm precedent already used for the observed-movement
  channel (`observed_movement_oracle_full_slate`,
  `observed_movement_oracle_sunday_am_realism`,
  `docs/late_week_refresh.md`'s "Observed-movement pick policy" table,
  `read`).
- **Null.** Within-week permutation null, alongside the week-blocked
  bootstrap — matching this project's standing rule
  (`within-week correlation is zero`, and the home-tilt-null precedent
  that a naive zero-centred null can mis-state an offset arm's true null
  distribution).
- **Recording.** `nfl-ats weak-signals record`, under a **new family
  name** not yet used by any existing registry entry (proposed:
  `inactives_channel_*`, to be finalized when the experiment actually
  runs, since AGENTS.md requires the family be declared before signs are
  seen, not chosen retroactively to fit a result).
- **Script to extend.** `nfl_ats.pick_refresh` (`src/nfl_ats/pick_refresh.py`)
  is the natural extension point: it already owns the "recompute at
  current features, grade at frozen Tuesday line" machinery this design
  needs; the new work is (a) a feature-table rebuild step that folds in
  the surprise-absence signal before `refresh-picks` runs, analogous to
  how `docs/prospective_evidence.md`'s step 8
  (`build-learned-availability-features`) already precedes step 9's
  margin predict, and (b) for the prospective arm only, a new capture
  script (Section 6) `refresh-picks` would need to have run *after*.
  `refresh-picks` itself needs no change to its deadline logic — Section 2
  already shows `pick_deadline`/`sunday_pick_lock` correctly exclude
  SNF/MNF without modification.

### Section 5 execution protocol (frozen before scoring, 2026-09-02)

The original predeclaration fixes the candidate, comparator, playable
population, opener grade, metric, control concept, null family, and recording
path, but it leaves the mechanically required rotation assignment and random
seeds unstated. Before any result was computed, those details are completed as
follows. The family is `inactives_channel_historical_proxy_v1`, grade
`opener`, with the default two-season confirmation block assigned by
`nfl-ats rotation assign`; this deterministically selects the earliest eligible
block, `[2020, 2021]`, and acknowledges the registry's mined 2018–2025
population. The feature table, player snapshot, and market archive are read
only; the historical label uses zero total snaps in `snap_counts.parquet` and
the latest report visible 24 hours before kickoff, while the pregame candidate
table changes only the seven existing unavailability columns through
`players._injury_features`. SNF/MNF are excluded with the unchanged
`pick_refresh.pick_deadline` test.

The frozen estimator is `market_residual`, `weak_stack`, ridge alpha 10,
`min_train_games=500`, with opener probabilities graded using the production
`home_cover_probability >= 0.5` rule and paired against the unchanged Tuesday
opener card. Uncertainty is the existing week-blocked bootstrap with 20,000
resamples and seed `20260819` (the established late-refresh injury protocol).
The within-week permutation null uses 200 draws and seed `20260902`; it
permutes candidate/comparator correctness labels within each `(season, week)`
block and is reported as a null distribution, never as a second estimate. The
positive control is the predeclared realised-margin oracle: on the identical
playable rows, the comparator construct is replaced by the realised ATS
margin, and the same paired bootstrap is run. The execution order is binding:
(1) oracle positive control, (2) within-week permutation null, (3) historical
candidate-versus-Tuesday screen. All three retain per-game rows and their
configuration/provenance in one immutable artifact directory. Any result that
is not a resolved wrong sign, a zero split-half reliability, or bounded by the
positive control is recorded as `unresolved_below_power` through both required
record commands before this document is updated with evidence.

## 6. Capture-job proposal (text only, not implemented)

Per `docs/capture_scheduling.md`'s `Job` dataclass
(`scripts/capture_scheduler.py:64-99`, `read`: fields `name, day, at,
grace_minutes, command, enabled, why, season_guarded, dedupe_dir,
dedupe_minutes, added_on, catch_up`). Proposed source script (not written):
`scripts/ingest_nflcom_inactives.py`, mirroring
`scripts/ingest_nflcom_injuries.py`'s pattern verbatim — robots.txt checked
before every fetch (already confirmed clear for `/inactives/`, Section 3),
polite delay, immutable UTC-stamped snapshot directory under
`data/raw/nflcom_inactives/<ts>/`.

**Sunday early window** (covers the 147 Sun-13:00-ET games measured in
Section 2; T-90 ≈ 11:30 ET):

```
name:            inactives_sun_early
weekday/time:    sun 11:35 ET
grace:           15 min   # tight on purpose, same logic as odds_sun_close:
                           # a capture much past ~11:50 leaves refresh-picks
                           # too little runway before 13:00 kickoffs.
source:          ingest_nflcom_inactives.py --current --slot sun_early
idempotence key: dedupe_dir="data/raw/nflcom_inactives", dedupe_minutes=60
offseason:       season_guarded=True (no REG game in the window off-season)
```

**Sunday late window** (covers the 58 Sun-16:05–17:00-ET games; T-90 ranges
14:35–15:30 ET depending on the exact slate that week, deadline is the fixed
16:00 ET lock in every case per Section 2):

```
name:            inactives_sun_late
weekday/time:    sun 14:40 ET
grace:           15 min   # deadline for this slot is the fixed 16:00 ET
                           # lock (Section 2); a capture must land well
                           # before it to leave time for a refresh pass.
source:          ingest_nflcom_inactives.py --current --slot sun_late
idempotence key: dedupe_dir="data/raw/nflcom_inactives", dedupe_minutes=60
offseason:       season_guarded=True
```

**Thu variant** — flagged as a genuine design gap, not a simple fixed slot.
Section 2 measured Thu 2026 kickoff times as `13:00, 16:30, 20:15, 20:20,
20:35` (Thanksgiving and season-opener weeks kick much earlier than the
usual TNF 20:15/20:20), so a single fixed weekday/time job would miss the
T-90 window for the early Thanksgiving games and fire needlessly early for
the primetime ones. Two options, neither implemented here:

```
# Option A (conservative, matches the Job model as it exists today):
#   one job per historically observed Thu kickoff cluster, each short-grace.
name:            inactives_thu_afternoon   (T-90 for a 13:00/16:30 kickoff)
weekday/time:    thu 11:35 ET / thu 15:05 ET   (two separate jobs)
grace:           15 min each
name:            inactives_thu_primetime
weekday/time:    thu 18:50 ET   (covers 20:15-20:35 kickoffs)
grace:           20 min

# Option B (recommended, mirrors the existing Sunday-anchor precedent):
#   a week-relative job, computed from that week's OWN schedule the way
#   nfl_ats.pick_refresh.sunday_pick_lock already derives its Sunday anchor
#   from the week's own kickoffs rather than a hardcoded calendar time
#   (src/nfl_ats/pick_refresh.py:132-148). This is a scheduler capability
#   the current SCHEDULE table does not have (every Job is a fixed
#   weekday+time), so Option B is a small design extension, not a
#   same-shape SCHEDULE row -- reported here as the more correct fix, not
#   proposed as something to build in this work package.
```

**Sat variant** — same gap as Thu, smaller in scope (Section 2 measured only
2 Sat games in the 2026 snapshot, kickoff times `17:00`/`20:20`, but
`docs/late_week_refresh.md`'s late-season Saturday slate can carry more
games at more varied times in a real December). Same Option A/B choice as
Thu; Option A concretely:

```
name:            inactives_sat
weekday/time:    sat 15:30 ET   (covers a 17:00 kickoff; a 20:20 kickoff
                                  needs a second job at sat 18:50 ET)
grace:           15-20 min
source:          ingest_nflcom_inactives.py --current --slot sat
idempotence key: dedupe_dir="data/raw/nflcom_inactives", dedupe_minutes=60
offseason:       season_guarded=True
```

**A finding that changes the operating plan, not just the schedule table
(measured, `docs/capture_scheduling.md:24-25`, `read`).** The existing
`refresh_sun` job targets **Sunday 10:00 ET**, grace 300 minutes. Under
normal (non-sleeping-machine) operation the scheduler fires it at ~10:00-
10:01 ET — **before both proposed inactives windows (11:35 and 14:40 ET)**.
So today's single Sunday refresh pass, exactly as scheduled, **cannot** see
either Sunday inactives window even once the capture jobs above exist; it
would need either an additional Sunday `refresh-picks` pass timed after
each capture, or `refresh_sun` retimed later at the cost of the margin it
currently has before the 13:00 slate. This is a real scheduling
consequence of building this channel, not merely a hypothetical — flagged
here so Section 5's design and Section 6's schedule rows are read together,
not separately.

## 7. Prospective 2026 path

`docs/prospective_evidence.md` (`read` in full earlier this session)
establishes that prospective scoring "needs no registry window at all
... it costs nothing but patience" and is the intended way to settle a
result without spending one of this project's scarce opener-confirmation
windows. Given Section 4's verdict that the historical proxy is a
reasonable-but-imperfect bound on the true mechanism (not a literal replay
of the live channel), the admissible, low-cost path is:

1. Build the capture jobs proposed in Section 6 (or, at minimum, a manual
   `ingest_nflcom_inactives.py --current` run during each week's playable
   windows) starting Week 1 (2026-09-08 lock, kickoffs 2026-09-09/10
   onward per Section 2's schedule read).
2. Each week, for playable games only (Section 2's population), run the
   Section 5 candidate pipeline as an additional `refresh-picks` pass timed
   after the relevant inactives capture, and record it through
   `nfl_ats.prospective_scoring`'s challenger-recording path
   (`prospective-record`, the same mechanism `docs/prospective_evidence.md`
   already uses for `mod07_weak_signal_stack`) under a new challenger id —
   **not** the production pick path, so nothing here touches what the pool
   actually plays until/unless promoted.
3. Because prospective recording is append-only and anti-backdating-
   guarded (`docs/prospective_evidence.md`'s "anti-backdating guarantee"
   section: refused at write past kickoff, re-checked at read against
   `recorded_at_utc`), every week that runs this way accumulates admissible
   evidence automatically — no rotation window, no opener-confirmation
   spend, and no need to wait for the historical backtest (Section 4) to
   be built first. The historical arm (Section 4/5) and the prospective
   arm (this section) are complementary, not sequential: the prospective
   arm alone, run patiently from Week 1, is sufficient to eventually
   settle Section 5's predeclared metric on real, unambiguous T-90 data,
   exactly the kind of evidence `docs/prospective_evidence.md` was built
   to make cheap.

## Provenance

- **Measured this session:** the full deadline-arithmetic table (Section
  2, run against `nfl_ats.features._kickoff_utc` and
  `nfl_ats.pick_refresh.sunday_pick_lock`/`pick_deadline`, unmodified,
  imported from the locked env); every WebFetch/WebSearch result cited in
  Section 3 (NFL.com `/inactives/`, NFL.com `/robots.txt`, RotoWire
  `/football/inactives.php`, ESPN `site.api.espn.com` 403,
  `site.web.api.espn.com` 200 + structure on both a future and a
  completed game); the `nflreadpy` `load_*` function inventory and
  `load_injuries`/`load_participation` docstrings; the local
  `injuries.parquet`/`snap_counts.parquet`/
  `game_features_player_learned_availability.parquet` shapes and columns
  (Section 4); the repo-wide greps for `inactives` and `T-90` (Section 1).
- **Read this session:** `docs/archive/idea_ledger.md` (row 85 in full,
  plus header/provenance), `docs/late_week_refresh.md` (in full),
  `docs/capture_scheduling.md` (in full), `docs/qb_news_channel.md` (in
  full), `docs/prospective_evidence.md` (in full),
  `scripts/capture_scheduler.py:1-260` (`Job` dataclass and the odds/
  injuries `SCHEDULE` rows), `scripts/ingest_nflcom_injuries.py:1-80`
  (docstring, robots.txt precedent, regex parser),
  `src/nfl_ats/pick_refresh.py:100-170` (`sunday_pick_lock`,
  `pick_deadline`), `src/nfl_ats/features.py:70-90` (`_kickoff_utc`),
  `src/nfl_ats/clv.py:708` (`week_blocked_bootstrap` exists at this
  location), `src/nfl_ats/weak_signals.py` (registry function names),
  `C:/Users/Ryan/.claude/projects/F--Repos-nfl-py3/memory/
  picks-lock-at-kickoff.md` (the owner's per-game deadline rule).
- **Reported (web, unverified beyond the fetch itself):** the T-90 timing
  convention as stated in RotoWire's own FAQ text; the general claim (not
  reproduced against this repo's own ESPN fetch) that ESPN boxscore JSON
  carries a `didNotPlay` field.
- **Inferred:** the recommendation to prefer NFL.com over ESPN/RotoWire as
  primary; the Section 6 Option B scheduler-capability suggestion; the
  Section 4 judgment that the historical proxy "bounds the mechanism"
  rather than replaying it exactly; the Section 2 caveat that deadline
  eligibility overstates how many picks would actually change.
- **Not done:** no experiment was run, no `weak-signals record` or
  `rotation record-look` call was made, no scheduler job was added, no
  script was written beyond the scratchpad-only, deleted-after-use deadline
  calculator. This document creates exactly the one file it is itself
  written to (`docs/inactives_channel.md`); no other repository file was
  modified.

## Implementation status (2026-09-01)

Superseded by a follow-up work package (WP17) the same day, run by a
different session with no memory of this document's authoring session
beyond reading it fresh: the capture PLUMBING described above as
not-yet-built now exists. This section is appended, not a rewrite of the
"Not done" line above — that line was accurate for the session that wrote
it, and the labeling/provenance discipline this repo runs on
(`AGENTS.md`, "Label how you know it") is better served by a dated addendum
than by silently editing history out from under it.

**What is built (measured/read this session, 2026-09-01):**

- `src/nfl_ats/inactives_capture.py` — fetch/parse/write logic. Team-name ->
  team-code mapping, the HTML-stripping helper, and the current-week resolver
  are IMPORTED VERBATIM from `scripts/ingest_nflcom_injuries.py` (not
  duplicated), per this repo's existing reuse convention (the same one
  `nfl_ats.fluview_production_feature`/`nfl_ats.fluview_cfb_feature` already
  use for `scripts/fluview_battery_screen.py`).
- `scripts/capture_inactives.py` — thin CLI wrapper, mirroring how
  `INJURY_CAPTURE` in `scripts/capture_scheduler.py` already calls
  `scripts/ingest_nflcom_injuries.py --current` as a subprocess rather than
  through a `nfl-ats` subcommand. `src/nfl_ats/cli.py` was off-limits for
  this work package, so this is not a choice between two equally-open paths.
- Seven `inactives_*` `SCHEDULE` rows in `scripts/capture_scheduler.py` (see
  `docs/capture_scheduling.md`'s "The official inactives capture (WP17)"
  section for the full per-row rationale and how the built rows differ from
  this document's Section 6 proposal in naming and `dedupe_dir`).
- `tests/test_inactives_capture.py` (18 tests) and seven pins appended to
  `tests/test_capture_scheduler.py`, all passing; `ruff format`, `ruff check`,
  and `mypy src` all clean for every file this session owns (one scoped
  `[[tool.mypy.overrides]] module = ["scripts.ingest_nflcom_injuries"]`
  override was added to `pyproject.toml`, exact precedent already set by the
  existing `scripts.fluview_battery_screen`/`scripts.fluview_battery_ingest`
  override immediately above it in that file, for the identical reason: this
  session's import surfaces three pre-existing typing gaps in a script this
  work package must not edit).

**What Section 3's live-source survey could not settle, and how the build
handled it.** A fresh fetch of `https://www.nfl.com/inactives/` this session
(2026-09-01, still preseason) returned the same placeholder this document
already recorded — 200, 372,655 bytes, zero occurrences of
`nfl-c-matchup-strip__team-abbreviation`/`d3-o-table`/any "inactive"- or
"report"-named class anywhere in the DOM. `web.archive.org` (Wayback
Machine, tried as a way to inspect a real in-season snapshot) was
unreachable from this session's network sandbox (connection timeout on both
HTTP and HTTPS). So the primary parser's populated-page structure is still
**inferred by analogy** to `/injuries/`'s confirmed-real markup, not
measured — this was true when this document was written and remains true
now; the build does not resolve it, it only makes the uncertainty
operationally safe:

- The manifest records `empty_reason="unrecognized_page_structure"` (and the
  script exits non-zero, unlike the genuine off-season case) whenever a page
  does NOT show the known placeholder text but the guessed markup still
  parses zero rows — so the first real Week 1 run will either work or fail
  LOUDLY and visibly in the scheduler log, never silently.
- `tests/fixtures/nflcom_inactives_populated.html` is a CONSTRUCTED fixture
  (fictional player names, real class-name conventions borrowed from
  `/injuries/`'s working parser), not a captured real page — its own header
  comment says so. It proves the parsing/mapping/manifest CODE is correct
  against a page that matches the assumed shape; it cannot and does not
  prove NFL.com's real `/inactives/` page matches that shape.
- **First concrete action for whichever session is live when Week 1 games
  post real inactive lists** (2026-09-08 lock, kickoffs 2026-09-09/10 per
  Section 2): read the manifest of the first `inactives_sun_early`/
  `inactives_thu_*` snapshot. If `source_used="none"` and
  `empty_reason="unrecognized_page_structure"`, open the saved `primary.html`
  in that snapshot directory (written regardless of parse success), find the
  real wrapper/table class names, and fix `_SECTION_SPLIT_CANDIDATES` (and,
  if needed, the column-label lists) in `src/nfl_ats/inactives_capture.py`
  against real data for the first time.

**What the next stage still needs — wiring into `refresh-picks` — is
everything Section 5 and Section 7 of this document already predeclare and
this work package deliberately did NOT touch:**

1. The feature-table rebuild step Section 5 describes ("recomputed after
   inactives are known ... a feature-table rebuild that incorporates the
   surprise-absence signal ... analogous to how `docs/prospective_evidence.md`'s
   step 8 already precedes step 9's margin predict") does not exist yet. The
   capture only writes `data/players/inactives/<ts>/inactives.parquet`; no
   code reads it back into a feature, and `src/nfl_ats/pick_refresh.py` was
   explicitly off-limits for this work package.
2. The Section 7 prospective-recording path (a new `refresh-picks` pass timed
   after each capture, recorded through `prospective-record` under a new
   challenger id, never touching the production pick) is unbuilt. Section 6's
   own "finding that changes the operating plan" still applies unchanged: the
   existing `refresh_sun` job (Sun 10:00 ET) fires BEFORE both new Sunday
   inactives windows (11:35/14:40 ET), so an additional Sunday `refresh-picks`
   pass timed after each inactives capture — or a retiming of `refresh_sun`
   — is still needed before the captured data can reach any pick, prospective
   or production. Not added here: scope was the capture channel itself, and
   `refresh_sun`/`pick_refresh.py` were both off-limits.
3. The Section 5 predeclared experiment (candidate vs. comparator, positive
   control, permutation null, `weak_stack`/`injury_value_lost` recomputation)
   has not run and cannot run yet — it needs real captured data, which needs
   (1) and (2) above, which needs Week 1 to actually happen. No registry
   window, no rotation window, and no `weak-signals record`/
   `rotation record-look` call has been spent on this channel by this work
   package, matching this document's original "frozen predeclaration only"
   framing.
4. A new family name for `nfl-ats weak-signals record` (Section 5 proposes
   `inactives_channel_*`, still to be finalized when the experiment actually
   runs, per AGENTS.md's "declare the family before signs are seen" rule) is
   still just a proposal, not registered anywhere.

## Prospective wiring predeclaration (2026-09-01, WP41)

Written BEFORE the overlay code it describes existed, and before any 2026 game
has posted a real inactive list, so the rule text is frozen ahead of any sign
being seen — the same discipline Section 5 applies to the experiment itself.
WP17 (the "Implementation status" section above) built the capture; this
section predeclares how a captured snapshot reaches a pick, and registers the
arm that carries the evidence.

### Binding closing-grounds taxonomy (restated verbatim, applies to every verdict below)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains zero"
is the EXPECTED outcome for a real small signal. Only two grounds ever close a
line of work: (1) refuted mechanism — a RESOLVED wrong sign (whole interval on
the wrong side of zero) or zero split-half reliability; (2) bounded by a
positive control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". The registry code
hard-rejects inadmissible closures; if a record command errors, the verdict is
wrong, not the validator.

Nothing in this section reaches a verdict. It declares a rule and registers a
prospective arm, which is paper evidence at zero window cost, never a
promotion.

### 1. Overlay semantics, per game

For each game on the week's card, at every `nfl-ats refresh-picks` pass:

1. **Structurally excluded (SNF/MNF).** If the game's official inactives
   instant, `kickoff − 90 minutes`, falls at or after its own pick deadline
   (`nfl_ats.pick_refresh.pick_deadline` = `min(own kickoff, that week's
   Sunday 16:00 ET lock)`), the game can never be acted on by this channel.
   It keeps the Tuesday pick, tagged `tuesday_card (SNF/MNF excluded)`.
   Section 2 above measured this as exactly the SNF (slack −170 min) and MNF
   (−1,605 min) slots in both 2026 and 2025, and as *no other slot* — the
   Sunday 16:05–17:00 ET slot stays playable at +65 to +85 minutes of slack,
   so a naive "deadline earlier than kickoff" test would wrongly exclude it.
   The 90-minute lead is the league's published inactives convention, reported
   in Section 3 (RotoWire's own FAQ: "NFL inactives are released 90 minutes
   before kickoff for every game") and used unchanged throughout Section 2's
   arithmetic; it is not a tuned parameter.
2. **No in-window snapshot.** Otherwise, look for the newest inactives
   snapshot under `data/players/inactives/<UTC stamp>/` whose manifest's
   `captured_at_utc` is strictly before that game's deadline, whose manifest
   `season`/`week` match the pass, and which actually reported inactives
   (`row_count > 0`). If there is none — including the off-season/placeholder
   case, where the manifest carries `row_count: 0` and an `empty_reason` — the
   game keeps the Tuesday pick, tagged
   `tuesday_card (no in-window snapshot)`. A zero-row snapshot is treated as
   "no report yet", not as "nobody is inactive": the same fail-open convention
   `nfl_ats.nflcom_refresh_overlay` already uses for an absent or stale
   NFL.com page. **Strictly before the deadline** is also the anti-backdating
   guard: a snapshot captured at or after kickoff can never apply to that
   game, because the deadline is at most the kickoff.
3. **In-window snapshot.** Otherwise, recompute the injury-value construct
   with every player named on that snapshot's inactive list treated as
   **P(plays) = 0**, and everyone else exactly as the existing injury report
   already has them; then re-run the production pick at the frozen Tuesday
   line exactly as `refresh-picks` does. The game is tagged
   `inactives_snapshot <stamp>`.

### 2. What "P(plays) = 0" means in this codebase, and why it invents no constant

Production's injury features are built from an *unavailability* weight per
player: `nfl_ats.players._injury_unavailability` returns the learned
availability when one is attached, else
`nfl_ats.availability.fixed_unavailability(report_status, practice_status)`.
That function maps `"out" -> 1.0` (read,
`src/nfl_ats/availability.py:96-103`). So "P(plays) = 0" is unavailability
`1.0`, which is a **definition**, and it is simultaneously **bit-identical to
the weight production already assigns a player ruled Out**. Nothing is tuned
and no new number enters the pipeline.

Because that weight is already credited for players the Friday/Saturday report
ruled Out, the overlay applies the *increment*, not the level:

```
increment(player) = 1.0 − fixed_unavailability(that player's newest visible report row)
```

which is `0.0` for a player already Out (no double-count), `0.65` for a
Questionable player who turns out to be inactive, and `1.0` for a genuine
surprise absence never on the report — the exact population Section 1's ledger
row named. `fixed_unavailability` is imported and called, never re-typed.

The increment is then folded through **production's own aggregation function**,
`nfl_ats.players._injury_features`, imported verbatim rather than
reimplemented — the same function `enrich_with_player_features` calls — with
the increment supplied as the row's `_unavailability` (the learned-availability
hook that function already reads) and the player's role shares supplied in the
same `roles` mapping shape. Its `severity × share / 11` and per-group `/5`,
`/6`, `/7` normalizers are therefore production's, not a copy of them.

**Role shares** come from the player's most recent strictly-earlier snap-count
row in `data/players/raw/*/snap_counts.parquet` (`offense_pct`, `defense_pct`,
`st_pct`), the same table and the same prior-game-share proxy
`nfl_ats.nflcom_refresh_overlay` already uses for its starter proxy. A player
with no prior snap row scores share 0 and therefore contributes nothing —
identical to production's own `roles.get(gsis_id, {})` default. Name → GSIS
identity is resolved through `nfl_ats.players.canonicalize_rosters` and
`_normalized_player_name`, production's own crosswalk.

### 3. Scope boundary, stated up front rather than discovered later

The active model (`artifacts/active_ats_model.json`, `feature_profile:
weak_stack`) consumes **nine** injury columns, measured this session from
`nfl_ats.constants.FEATURE_SETS["full_weak_stack"]`: the seven
`diff_injury_*_unavailability` columns and the two
`diff_injury_*_value_lost` columns.

This overlay adjusts **the seven unavailability columns only.** The two
`*_value_lost` columns are produced by `players._injury_value_features`, which
multiplies severity by a per-player *value rate* drawn from a span-16 EWMA
state that `enrich_with_player_features` builds transiently and **never
persists to disk** — reconstructing it inside a refresh pass would be a
reimplementation of production's aggregation, which is exactly what this
design refuses to do. The consequence is directional and disclosed: the
candidate arm moves **less** than a full feature-table rebuild would, so a
null or small reading from it bounds the channel from below, not above. Closing
the gap needs a `build-player-features` pass over an inactives-augmented
injuries frame, which is a separate work package.

### 4. Re-running the pick: the same machinery, not a parallel one

The adjusted columns are written into a **copy** of the production feature
table in which only the target week's rows differ, and
`nfl_ats.pick_refresh.plan_refresh` is called on that copy at the same `now`
as the production pass. Two consequences, both deliberate:

* Training rows are byte-identical, so `fit_margin_models_for_week` fits the
  **same** model; only the target week's prediction responds to the inactives.
* Everything downstream is production's, unchanged: the frozen Tuesday line
  override (`apply_external_lines` against `decision_home_spread`), the frozen
  composed-overlay flip, the `>= 0.5` forced-pick rule, and the observed-
  movement policy at `pick_refresh.MOVEMENT_POLICY_THRESHOLD = 1.0` — cited,
  reused, and **not re-tuned here** (that constant is frozen by
  `docs/observed_movement_channel.md`'s predeclared 0.5/1.0 grid, see the
  comment at `src/nfl_ats/pick_refresh.py:164-180`).

A pick therefore flips **only when the recomputed home-cover probability
crosses 0.5** (or when the unchanged movement policy would have overridden it
anyway) — never because a player's name appeared on a list.

### 5. The challenger

* **Name.** `inactives_refresh_v1`, status `ACTIVE_PROSPECTIVE`.
* **Candidate arm.** The refreshed pick described above.
* **Paired comparator.** The **incumbent Tuesday card** — the `pick_side`
  already frozen in `artifacts/clv_ledger/decisions.parquet` for that game,
  exactly the comparator Section 5's predeclaration named. The same-pass
  played refresh pick (`RefreshedGame.new_pick_side`) is recorded alongside it
  as context, so the arm can also be read against the played card without a
  second pass.
* **Grade.** Both grades, like every other challenger: `nfl-ats
  prospective-score` reads `decision_line` (the frozen Tuesday line —
  opener-graded, the pool's own grade and this project's declared primary
  goal) as primary and `close_line` as secondary. Per AGENTS.md's "grade the
  decision at the OPENER", a close-graded number may never veto this arm.
* **Metric.** Paired forced-pick accuracy delta in `accuracy_points` on the
  identical game set, `probability_positive` reported, never "contains zero".
* **Population.** Section 2's playable set: 238/272 (87.5%) of 2026 REG games,
  minus games with no in-window snapshot in the week they are played.
* **Window cost.** None. A prospective registration spends no rotation
  registry window and changes no published pick.
* **This is not a promotion.** Nothing in the overlay module is wired into
  `publishing.py` or the played pick path, and it cannot be: it consumes the
  `RefreshResult` strictly read-only and writes only its own append-only
  ledger, `artifacts/prospective/inactives_refresh_decisions.parquet`.

### 6. Wiring gap this work package could not close

`src/nfl_ats/inactives_refresh_overlay.py` mirrors
`src/nfl_ats/nflcom_refresh_overlay.py` exactly, including its
`record_*(artifacts_root, data_root, plan, *, record_decisions)` signature. The
NFL.com overlay is invoked from a hook inside `nfl_ats.cli._cmd_refresh_picks`
(read, `src/nfl_ats/cli.py:1098-1106`). **`src/nfl_ats/cli.py` was off-limits
to this work package**, so the identical hook was NOT added and the overlay
does not yet run automatically on a `refresh-picks` pass. The challenger's
`weekly_recording_command` therefore names the direct function call until
someone who owns `cli.py` applies it — the same constraint and the same
resolution `docs/challenger_expansion_20260901.md` section 1.5 records for six
other challengers registered the same day. The patch is three lines beside the
existing NFL.com block:

```python
from nfl_ats.inactives_refresh_overlay import record_inactives_refresh_overlay
...
    try:
        result["inactives_refresh_overlay"] = record_inactives_refresh_overlay(
            _artifacts_root(), _data_root(), plan, record_decisions=args.record_decisions
        )
    except (ValueError, FileNotFoundError, DataContractError) as error:
        result["inactives_refresh_overlay"] = {"recorded": 0, "error": str(error)}
```

Registering under `nfl-ats refresh-picks --record-decisions` (rather than
`publish-predictions --record-decisions`) also keeps
`tests/test_cli.py::test_publish_challenger_result_map_covers_live_active_registry`
green, which is the correct registration for a refresh-time arm regardless —
the same command `injury_signal_refresh_tilt` already registers under.

### 7. What the scheduler needs before this can work on a Sunday

WP17's finding stands and is the operative blocker: `refresh_sun` fires at
**Sunday 10:00 ET** (read, `scripts/capture_scheduler.py:328-342`), which is
before **both** Sunday inactives captures — `inactives_sun_early` at 11:35 ET
and `inactives_sun_late` at 14:40 ET (read,
`scripts/capture_scheduler.py:404-451`). Today's three refresh passes
(`refresh_thu` 15:00, `refresh_sat` 10:30, `refresh_sun` 10:00) can never see
a single inactives snapshot, so the channel would honestly record
`no in-window snapshot` for all 18 weeks.

`scripts/capture_scheduler.py` was off-limits to this work package, so the
rows below are **proposed, not added**. Each time is derived, not chosen: a
pass must start after its capture's window closes (`at + grace_minutes`, plus
five minutes for the capture to finish writing) and must still be able to fire
before the earliest binding deadline of the games that capture covers, less a
ten-minute safety margin — so `grace_minutes = (deadline − 10 min) − at`.

| proposed job | day | at (ET) | grace | derived from | covers |
|---|---|---|---|---|---|
| `refresh_thu_inactives_early` | thu | 11:55 | 55 | `inactives_thu_afternoon_early` 11:35+15 -> 11:50, +5; cap 13:00−10 = 12:50 | 13:00 ET Thu kickoffs |
| `refresh_thu_inactives_late` | thu | 15:25 | 55 | `inactives_thu_afternoon_late` 15:05+15 -> 15:20, +5; cap 16:30−10 = 16:20 | 16:30 ET Thu kickoffs |
| `refresh_thu_inactives_primetime` | thu | 19:15 | 50 | `inactives_thu_primetime` 18:50+20 -> 19:10, +5; cap 20:15−10 = 20:05 | 20:15–20:35 ET TNF |
| `refresh_sat_inactives_early` | sat | 15:50 | 60 | `inactives_sat_early` 15:30+15 -> 15:45, +5; cap 17:00−10 = 16:50 | 17:00 ET Sat kickoffs |
| `refresh_sat_inactives_late` | sat | 19:15 | 55 | `inactives_sat_late` 18:50+20 -> 19:10, +5; cap 20:20−10 = 20:10 | 20:20 ET Sat kickoffs |
| `refresh_sun_inactives_early` | sun | 11:55 | 55 | `inactives_sun_early` 11:35+15 -> 11:50, +5; cap 13:00−10 = 12:50 | the 147 Sun-13:00 ET games |
| `refresh_sun_inactives_late` | sun | 15:00 | 50 | `inactives_sun_late` 14:40+15 -> 14:55, +5; cap = the fixed Sunday 16:00 ET lock, −10 = 15:50 | the 58 Sun-16:05–17:00 ET games |

Each row is `_cli("refresh-picks", "--record-decisions", "--note",
"<slot>_inactives")`, `season_guarded=True`, `catch_up=False` (a refresh pass
whose deadline has passed is a no-op by construction — `plan_refresh` marks
those games ineligible — so catching one up later writes nothing and would only
muddy the run log). **None of them carries `--publish-card`**: `refresh_sun` at
10:00 remains "the only one that touches the card", and whether the published
card should instead follow the last pre-deadline pass is a played-pick policy
decision that belongs to whoever owns `pick_refresh`, not to a prospective arm.

## Section 5 historical screen result (2026-09-02)

This addendum supersedes the earlier “not run” status only for the historical
proxy arm; the live 2026 prospective arm remains separate and active. Before
scoring, the completed protocol above was recorded by declaring and assigning
`inactives_channel_historical_proxy_v1` through `nfl-ats rotation`, which
selected `[2020, 2021]`. The screen then ran the frozen oracle → permutation
null → candidate order with `scripts/inactives_channel_historical_screen.py`.

**Measured** from
`artifacts/inactives_channel_historical_proxy/20260902T155950Z/results.json`
and its paired parquet: 554 target games, 507 deadline-playable games, 6,675
proxy labels affecting 528 games, and 429 paired non-push opener-graded games
across 33 weeks. The realised-margin oracle control measured **+45.9207
accuracy points**, week-blocked 95% **[+40.0911, +51.9139]**,
`probability_positive=1.000`. The candidate versus the unchanged Tuesday card
measured **−1.3986 accuracy points**, week-blocked 95% **[−3.1963,
+0.2427]**, `probability_positive=0.0418`; the two-season secondary
season-blocked interval was [−1.9139, −0.9091]. The 200-draw within-week
paired-label null had mean −0.0186 points and 95% [−1.3986, +1.3986].

The result is recorded as `unresolved_below_power` in
`registry/weak_signals.json` and `unresolved` in the spent rotation window;
the primary interval does not establish a resolved wrong sign, and the large
oracle control does not bound a roughly 1.4-point effect. This is therefore
not a closure or a promotion. The artifact retains prediction-level pairs and
the post-game proxy labels for audit. The historical proxy uses weekly-roster
minus positive snaps because `snap_counts.parquet` omits most zero-snap
players; it is a reasonable but imperfect stand-in for an official T-90 list,
and the candidate changes only the seven unavailability columns (the two
value-lost columns remain unchanged). No played card, active model, or
published forecast was altered.

**Provenance repair (2026-09-02):** the existing scored artifact was not
rerun. The exact helper invocation was `uv run --no-sync python -c "..."`
(loading `results.json`, adding `artifact_provenance(...)`, and calling
`write_experiment_artifact(...)` for command
`inactives-channel-historical-screen`). It preserved the original
`created_at_utc` and all screen metrics, added the provenance block, and wrote
`registry/experiments/inactives-channel-historical-screen/20260902T155950Z.json`.
Future runs now use the same helper directly in
`scripts/inactives_channel_historical_screen.py`.

## POL-11 integration status (2026-09-02)

**Measured, this session:** the `refresh-picks` CLI now calls
`record_inactives_refresh_overlay` beside the existing NFL.com refresh
challenger (`src/nfl_ats/cli.py`, `_cmd_refresh_picks`). The recorder consumes
the already-computed `RefreshResult` read-only and appends only
`artifacts/prospective/inactives_refresh_decisions.parquet`; it does not alter
the played refresh plan, `pick_revisions.parquet`, or the published card.

**Measured, this session:** the seven refresh rows proposed in this section are
now present in `scripts/capture_scheduler.py`. Each starts five minutes after
its corresponding inactives capture window closes, has a grace window ending
ten minutes before its relevant kickoff/deadline, is `season_guarded=True`,
and is `catch_up=False`; none publishes a card. The scheduler coverage test
pins these timing relationships (`tests/test_capture_scheduler.py`).

**Measured, this session:** `inactives_refresh_overlay` now fails closed to
`tuesday_card (no in-window snapshot)` when a manifest is absent, malformed,
unrecognized, failed, zero-row, wrong-season/week, future-dated at the refresh
instant, stale from a different local game day, or its rows do not explicitly
match the target game's id, teams, season, week, and capture timestamp. The
targeted safety tests are in `tests/test_inactives_refresh_overlay.py`.

**Not done:** the Section 5 historical/prospective ATS experiment, its positive
control and permutation null have not run; no weak-signal or rotation registry
entry was written. The separate `build-player-features` reconstruction needed
to extend this challenger from the seven unavailability columns to the two
value-lost columns remains outside this plumbing slice. The international
Sunday capture/refresh gap below remains unbuilt.

### Refresh-hook status addendum (2026-09-02)

The intended `nfl-ats refresh-picks --record-decisions` hook is now present in
`nfl_ats.cli._cmd_refresh_picks`, beside the NFL.com refresh overlay. It
reports `inactives_refresh_overlay` separately and remains fail-open, so an
inactives-recorder problem cannot break the production refresh or card append.
It is deliberately not a Tuesday `publish-predictions` recorder.

**One capture gap this section flags but does not fix.** Section 2 measured 6
Sunday international games at 09:30 ET in 2026 (T-90 = 08:00 ET). There is no
`inactives_sun_intl` capture row, so those six games have no reachable
snapshot even after the refresh rows above are added; they will honestly record
`tuesday_card (no in-window snapshot)`. Closing that needs a capture at
`sun 08:05 ET` (grace 15) plus a refresh pass at `sun 08:25 ET` (grace 55, cap
09:30−10 = 09:20) — proposed here for the same reason, and equally not added.
