# Arctic Shift subreddit-volume ATS battery: predeclaration

Written 2026-08-26, **before any cover-rate sign in this battery has been
examined and before the widened fetch's coverage numbers were read**, per
the separate decision `docs/arctic_shift_gate.md` explicitly left open:
"Correlation here measures construct overlap, not predictive value; an ATS
battery remains a separate decision." The gate (`artifacts/arctic_shift_gate/
results.json`) passed its reliability leg (YoY log-volume reliability
0.8902) and failed its shared-variance leg against Wikipedia pageviews
(pooled log-scale r=0.7319 >= 0.70) -- high overlap on RAW VOLUME, but per
AGENTS.md a shared-variance failure is a construct-overlap finding, not an
admissible closing ground (not a resolved wrong sign, not zero reliability,
not a positive-control bound). This document freezes cells, thresholds, and
the reliability method before scoring, exactly matching the precedent in
`docs/fluview_battery.md`.

## Binding taxonomy (owned verbatim, per AGENTS.md / CLAUDE.md)

An interval or CI that contains zero is NEVER grounds to reject, fail, or
close an experiment. At this evaluator's ~2-point resolution, "contains
zero" is the EXPECTED outcome for a real small signal. Only two grounds
ever close a line of work: (1) refuted mechanism -- a RESOLVED wrong sign
(the whole interval on the wrong side of zero) or zero split-half
reliability; (2) bounded by a positive control proven able to detect an
effect that size. Everything else is `unresolved_below_power`: record it
with `nfl-ats weak-signals record`, report `probability_positive`, never
the binary "contains zero". The registry code hard-rejects inadmissible
closures; if a record command errors, the verdict is wrong, not the
validator. Every cell below is recorded regardless of sign or interval
shape.

## 1. Why volume alone was the wrong test, and what this battery tests instead

The gate measured **weekly volume level** against **Wikipedia pageviews**
and found high overlap (r=0.73 log-scale, borderline above the 0.70 bar).
That is expected: both series are, at bottom, "how much attention is this
team getting this week," and a game that generates a lot of press coverage
also generates a lot of subreddit chatter. Re-testing raw volume against
the spread would just re-test the same construct GDELT/Wikipedia batteries
already screened (`attention_battery_both_cold`, `attention_battery_both_
cold_gdelt_replication` -- the latter's headline: -0.2742 accuracy points,
probability_positive 0.23225, **reported** from `registry/weak_signals.json`,
not independently re-verified by this document beyond the read described in
section 5), and a Reddit cell that merely reproduces that construct is
expected to be similarly weak.

This battery instead builds constructs Wikipedia/GDELT structurally cannot
measure, because both of those instruments are consumption/coverage proxies
(a pageview or a news article, produced by casual lookups and journalists),
while a team subreddit's post/comment stream is produced almost entirely by
that team's own invested fanbase actively talking to each other:

- **Within-week SPIKES relative to a team's own baseline** (news breaking,
  R1/R2/R3 below) -- the same "hot vs not" shape `attention_battery_screen.
  py` uses, but on an instrument that measures active fan reaction, not
  passive readership.
- **Sentiment-free volume asymmetry between the two fanbases** -- captured
  by R3's DIFFERENTIAL design (both sides' spike z-scores, each normalized
  against that team's own history, so a market-size gap between e.g.
  Dallas and Jacksonville cancels out structurally; what remains is which
  fanbase is reacting harder than ITS OWN normal, this week, relative to
  the other side's own normal). This is not a quantity GDELT's media-side
  instrument or Wikipedia's readership instrument can express at all.
- **Comment-to-post ratio as an argument/anxiety proxy** (R4/R5) -- a single
  announcement post can drive huge comment counts without reflecting the
  volume of independent voices; a rise in comments-PER-post beyond a team's
  own normal ratio is closer to "how much is the fanbase arguing/worrying
  about this," a forum-structure quantity neither a pageview count nor a
  news-article count can express, since neither instrument has a
  "reactions per announcement" concept.

## 2. Widened data (measured this session)

The frozen gate sampled 6 subreddits (DAL/cowboys, NE/Patriots, GB/
GreenBayPackers, PIT/steelers, JAX/Jaguars, TEN/Tennesseetitans) x REG
2019-2021. The Arctic Shift `/api/time_series` endpoint costs the SAME one
request per (subreddit, kind) regardless of the requested date range, so
widening SEASONS is free in request count; widening TEAMS costs two
requests each. `scripts/arctic_shift_battery_fetch.py` re-fetches all 32
NFL team subreddits (the original 6 re-fetched too, for one consistent
full-history file per team) over `2010-08-01` to `2026-08-27`, saving every
raw response under `data/raw/arctic_shift/` with a sha256 manifest
(`manifest.json`) and a per-team fetch summary (`fetch_summary.json`) --
n_days, total_count, and first/last date per (team, kind), which IS the
verification for the best-effort subreddit-name mapping in that script (a
team whose fetch returns zero days or an explicit API error has its
subreddit name wrong or the subreddit does not exist under that name, and
is reported as excluded, not silently guessed around).

`scripts/arctic_shift_battery_screen.py` reports actual per-team, per-season
data coverage (fraction of team-games with a computable trailing baseline)
in its results JSON, mirroring `docs/fluview_battery.md` section 3's
"two season floors" transparency -- this is a PREDICTOR-side coverage
measurement (whether Reddit itself has enough history for a given team by a
given season), computed and disclosed before any cover-rate sign is
examined, exactly the same admissible pre-scoring exception
`docs/team_style.md`'s reliability gate and `docs/fluview_battery.md`
section 4's peak-week window both use. No season or team is hand-picked in
advance to be excluded; the coverage/eligibility flags built in section 3
below do that mechanically per row.

## 3. Point-in-time-safe construction (frozen algorithm)

Per team, from the fetched daily `posts` and `comments` count series:

- `daily_volume = posts + comments` (per calendar day, UTC-midnight epochs
  from `/api/time_series` -- the search/aggregate endpoint's ~T22:00Z bucket
  labels are NOT used, per `docs/arctic_shift_gate.md`'s own recorded
  caveat, carried forward unchanged here).

Per (team, game) -- one row per team-game side, built from the newest
`schedules.parquet` snapshot, in the team's own chronological game order
(so bye weeks are skipped naturally, identical to
`scripts/attention_battery_screen.py`'s `build_team_game_long`):

- **Decision cutoff / window**: `window_end` = the Tuesday of the game's own
  week (`gameday - ((weekday - 1) % 7)` days, Monday=0), `window_start` =
  `window_end - 6 days` -- byte-identical convention to
  `scripts/attention_battery_screen.py` / `scripts/fluview_battery_screen.
  py` / `scripts/arctic_shift_gate.py`. Every day summed into a window is
  therefore on or before the Tuesday strictly preceding the game, for every
  scheduling slot (Thu/Sun/Mon) in that NFL week.
- `window_volume` = sum of `daily_volume` inside the window.
- `window_posts`, `window_comments` = same-window sums of the two raw
  series separately (not summed together).
- `comment_post_ratio = window_comments / window_posts` (missing when
  `window_posts == 0`).
- **Trailing baseline** (own team's STRICTLY PRIOR games only, in schedule
  order -- never including the current window): trailing mean/std of
  `window_volume` and, separately, of `comment_post_ratio`, each over the
  team's own previous `TRAILING_WINDOW_GAMES` games (min
  `TRAILING_MIN_GAMES`), reusing
  `scripts/attention_battery_screen.py`'s own frozen constants
  (`TRAILING_WINDOW_GAMES=8`, `TRAILING_MIN_GAMES=2`) by import, not a
  fresh unexamined pick.
- `volume_z = (window_volume - trailing_mean_volume) / trailing_std_volume`;
  `ratio_z` defined the same way from `comment_post_ratio`. Both are
  `NaN` (and the row excluded from any cell needing that side) when the
  team has no computable trailing baseline yet (start of a subreddit's
  fetched history) or trailing_std is zero.

This mirrors `scripts/attention_battery_screen.py`'s own leak-safety
argument exactly: the current window's own value never appears in its own
baseline (`shift(1)` before the rolling window), and the window itself only
sums days at or before the Tuesday decision cutoff, so no information from
game week itself (let alone the game) reaches the feature. A leakage
regression test (`tests/test_arctic_shift_battery.py`, mirroring
`tests/test_gdelt_attention_screen.py`'s
`test_tuesday_features_ignore_news_after_tuesday_cutoff` pattern: perturb a
day strictly after the cutoff and assert the computed window/z-score is
unchanged) is required before this battery is scored.

## 4. Frozen thresholds

Both `volume_z` and `ratio_z` use the SAME `>= 2.0` "spike" threshold
`scripts/attention_battery_screen.py` already froze for its
`hot_team_fade`/`away_hot` cells -- a reused precedent constant, not a
second freshly-picked one for the ratio instrument. Picking a different,
untested threshold for `ratio_z` would be exactly the kind of underived
constant this project's own standing rule treats as a defect; reusing the
one number that already has a prior use in this codebase is the more
defensible choice even though it was not independently derived for a ratio
specifically.

## 5. Predeclared cells (5, mirroring `docs/fluview_battery.md`'s 5-cell shape)

Population for all cells: NFL REG games, close-graded via
`schedules.spread_line` (`nfl_ats.features.add_ats_outcomes`, pushes
dropped) -- same convention as `docs/fluview_battery.md` /
`scripts/attention_battery_screen.py`. Method: joint week-blocked bootstrap
(block = `season*100+week`) PRIMARY, season-blocked bootstrap SECONDARY,
both algorithm-identical to `scripts/fluview_battery_screen.py`'s
`block_bootstrap_two_group`. Full-slate effect scaling via
`nfl_ats.experiment_runner.scale_subset_effect` (imported, not
reimplemented). 20,000 bootstrap samples, seed 20260826 (repo convention:
today's date). Within-week correlation is zero by owner mandate -- no ICC
term. No season or team is excluded up front; rows without a computable
baseline on the side(s) a cell needs are excluded from BOTH the subset and
complement of that cell (reported as `n_excluded_missing`), never defaulted.

**R1. `reddit_home_spike_fade`** -- home team `volume_z >= 2.0` (eligible =
home has a computable volume baseline) vs not, response `home_cover`.
**Predicted sign: NEGATIVE.** A fan-volume spike in the home team's own
subreddit signals a hype/news/distraction event in the days before the
game; the same overhype-fade mechanism `attention_battery_screen.py`'s
`hot_team_fade` cell tests, but on active fan engagement rather than
passive pageview lookups -- a different population producing the signal
even where the two magnitudes correlate.

**R2. `reddit_away_spike_value`** -- away team `volume_z >= 2.0` (eligible =
away has a computable volume baseline) vs not, response `home_cover`.
**Predicted sign: POSITIVE.** Mirror mechanism: the away team's own spike
is a distraction for them, favoring the home side.

**R3. `reddit_spike_gap_home_worse`** -- restricted to games where exactly
one side spikes (home XOR away on `volume_z >= 2.0`, BOTH sides' baselines
required non-missing); subset = home spikes & away does not, complement =
away spikes & home does not, response `home_cover`. **Predicted sign:
NEGATIVE.** The differential/asymmetry cell (section 1): both z-scores are
normalized against each team's OWN history, so a structural subscriber-base
gap between the two franchises cancels out by construction -- what remains
is genuine relative over-reaction this week, a quantity no media-coverage
or readership instrument expresses.

**R4. `reddit_home_comment_ratio_elevated`** -- home team `ratio_z >= 2.0`
(eligible = home has a computable ratio baseline) vs not, response
`home_cover`. **Predicted sign: NEGATIVE.** A rise in comments-per-post
beyond the team's own normal ratio is read as elevated
argument/anxiety-driven discussion (bad news, injury worry, QB
controversy) rather than mere announcement-driven volume; the same
"something's wrong here" mechanism as R1 but on a structurally distinct
quantity neither Wikipedia nor GDELT can express.

**R5. `reddit_away_comment_ratio_elevated`** -- mirror, away team
`ratio_z >= 2.0`, response `home_cover`. **Predicted sign: POSITIVE.**

## 6. Reliability check (measured, run before cover-rate scoring)

Two separate split-half reliability figures via
`nfl_ats.cfb_qb_dependence.split_half_reliability` (reused directly, the
same function `docs/fluview_battery.md` section 6 and PBP-05/PBP-08/
injury-value-lost were built on), each on a team-season long panel
(`team_id` <- team code, `season`, `week` <- NFL week for the odd/even
split):

- **Volume reliability**: metric = raw per-team-game `window_volume`
  (RAW, not z-scored -- testing whether "this team's subreddit is
  chronically loud this season-half" is a persistent trait, not an
  artifact of the trailing-window smoothing). Recorded against cells
  R1/R2/R3.
- **Ratio reliability**: metric = raw per-team-game `comment_post_ratio`.
  Recorded against cells R4/R5, since this is a structurally distinct
  construct from volume and the task instruction is explicit that a newly
  constructed feature "deserves its own number" rather than inheriting the
  gate's volume-only YoY figure (0.8902, `artifacts/arctic_shift_gate/
  results.json`, a DIFFERENT reliability design -- year-over-year of
  team-season MEANS, not this section's within-season odd/even split-half
  -- so the two numbers are not directly comparable and neither substitutes
  for the other).

Per AGENTS.md, a `no_split_half_reliability` closing ground requires this
figure's CI to sit AT (not near) zero -- an interval crossing zero here is,
as everywhere else, not grounds to close.

## 7. Files

- `scripts/arctic_shift_battery_fetch.py` -- widened raw fetch (32 team
  subreddits, full available history), writes `data/raw/arctic_shift/`
  (gitignored) + `manifest.json` + `fetch_summary.json`.
- `scripts/arctic_shift_battery_screen.py` -- as-of construction, cell
  scoring, writes `artifacts/arctic_shift_battery/<UTC>/results.json`
  (measure-only, no registry writes).
- `scripts/arctic_shift_battery_record.py` -- records all 5 cells to
  `registry/weak_signals.json` via `nfl-ats weak-signals record`, verifies
  after writing.
- `tests/test_arctic_shift_battery.py` -- leakage regression test(s).
