# Arctic Shift subreddit-volume shared-variance gate (scout v5 Section C #3)

Status: **gate frozen before any gate number was computed.** This section was
written after API shape probing only (endpoint existence, bucket alignment,
limit behavior); no correlation or reliability number had been computed when
the rule below was fixed.

## Question

Does team-subreddit post/comment VOLUME carry variance that the
already-built Wikipedia-pageview attention feature does not? If subreddit
volume were nearly a duplicate of the Wikipedia attention series, an ATS
battery on it would re-test the same construct and burn a season window for
nothing.

## Existing comparison feature (located, read this session)

- The Wikipedia-pageview attention construct is implemented in
  `scripts/attention_battery_screen.py` (`TEAM_ARTICLES`, `load_team_daily_views`,
  Tuesday-ending 7-day window in `build_team_game_long`). Its raw inputs are
  daily pageview JSONs under the scratchpad dir recorded there
  (`agent_attention/raw/<Article>.json`, 2015-07-01 through 2026-01-01).
- There is **no processed parquet** of the Wikipedia feature;
  `data/processed/gdelt_weekly_attention.parquet` is the separate GDELT
  news-volume cross-validation companion (`scripts/build_gdelt_weekly_features.py`),
  not the Wikipedia table. Comparison here uses the same raw JSON source the
  battery script uses.

## Frozen gate rule

Proceed to an ATS battery on team-subreddit volume **only if both**:

1. **Shared-variance:** Pearson correlation between weekly subreddit volume
   and weekly Wikipedia pageviews over the pooled sampled team-weeks is
   **r < 0.7**; and
2. **Own reliability:** year-over-year reliability of subreddit volume
   (Pearson correlation of team-season mean weekly log1p volume between
   consecutive seasons, averaged over adjacent-season pairs) is **>= 0.2**.

If either leg fails, record the outcome as a category-3 unresolved result via
the weak-signals recorder flow described by AGENTS.md (this task performs no
registry writes itself; recording is a separate explicit command).

## Sampling plan (frozen)

- **Teams (6, fixed now, spread of market sizes):** DAL→r/cowboys,
  NE→r/Patriots, GB→r/GreenBayPackers, PIT→r/steelers (large),
  JAX→r/Jaguars, TEN→r/Tennesseetitans (small).
- **Seasons:** NFL REG 2019, 2020, 2021.
- **Source:** Arctic Shift API `https://arctic-shift.photon-reddit.com`
  (live, no-auth, verified this session), endpoints
  `/api/posts/search/aggregate` and `/api/comments/search/aggregate` with
  `aggregate=created_utc&frequency=day`. Polite delay >= 2 s between requests,
  retry with backoff on server-side timeout ("Timeout. Maybe slow down a bit").
  Every raw response body saved under `data/raw/arctic_shift/` with a sha256
  manifest.
- Known API quirks probed this session (documented, not gate-relevant):
  search `limit` maxes at 100; daily aggregate buckets are labeled at
  `T22:00:00Z` boundaries (server-side offset from UTC midnight); comments
  aggregates on active subreddits can time out and need retries.

## Weekly construct (frozen)

Per team-week (game weeks only, REG 2019-2021, from the newest
`schedules.parquet` snapshot): window ends the Tuesday of game week
(gameday minus `((weekday - 1) % 7)` days, Monday=0), window start = end - 6
days — identical to the Wikipedia attention window in
`scripts/attention_battery_screen.py`. Subreddit weekly volume = sum of daily
post counts + daily comment counts inside the window. Wiki weekly value =
same-window sum of daily pageviews. Weeks are identified by ISO year-week of
the window end for reporting.

## Analysis (frozen)

- Primary shared-variance statistic: Pearson r on log1p-transformed weekly
  values across all pooled team-weeks. Secondary: Spearman rho, per-team
  Pearson r, and raw-scale r, reported but not gating.
- Reliability statistic: per adjacent pair (2019 vs 2020, 2020 vs 2021)
  Pearson r of team mean log1p weekly volume (6 teams per pair); headline
  reliability = mean of the two pair correlations.
- Verdict against the frozen thresholds, with effective-n caveats stated.

## Results

Computed 2026-08-22 by `scripts/arctic_shift_gate.py` (all numbers **measured**
this session; outputs in `artifacts/arctic_shift_gate/results.json` and
`team_weeks.csv`, raw responses + sha256 manifest under `data/raw/arctic_shift/`).

### API adaptation (documented deviation from scout description)

The scout's `/api/{posts,comments}/search/aggregate` endpoint shape was as
described (`{"data":[{created_utc, count}]}`, no auth), but full-season and
even monthly aggregate ranges returned HTTP 422 persistently, and the server
enforces an apparent hourly quota (`X-RateLimit-Reset` / `-Reset-At` headers;
observed recovery only at the top of the hour after a burst). Adaptation: the
equivalent documented `/api/time_series` endpoint with
`key=r/<subreddit>/{posts,comments}/count&precision=day` was used instead —
12 requests total. Cross-check: for the overlapping probe window the two
endpoints return identical counts (r/cowboys posts Sep 1-7, 2019: 57/46/73/
122/68/47/72 on both, measured). The time-series dates are clean UTC-midnight
epochs, so the earlier `T22:00Z` bucket-offset caveat does not apply to the
final data.

### Gate numbers

Shared-variance leg (weekly subreddit volume vs weekly Wikipedia pageviews,
same Tuesday-ending windows, pooled team-weeks):

| statistic | value | n |
| --- | --- | --- |
| **Pearson r, log scale (predeclared primary)** | **0.7319**, 95% CI [0.6739, 0.7809] | 294 |
| Spearman rho | 0.7624 | 294 |
| Pearson r, raw scale | 0.6205 | 294 |
| per-team r (DAL/GB/JAX/NE/PIT/TEN) | 0.744 / 0.675 / 0.637 / 0.571 / 0.618 / 0.640 | 49 each |

Year-over-year reliability of subreddit volume (mean log1p weekly volume,
adjacent seasons, n=6 teams per pair):

| pair | r | Fisher 95% CI |
| --- | --- | --- |
| 2019 vs 2020 | 0.8325 | [0.065, 0.981] |
| 2020 vs 2021 | 0.9479 | [0.591, 0.994] |
| headline (mean of pairs) | **0.8902** | ~[0.65, 0.97] (n=12) |

### Verdict against the frozen gate

**GATE FAIL** — reliability leg passes easily (0.890 >= 0.2), but the
shared-variance leg fails: the predeclared primary statistic is
r = 0.7319 >= 0.7. Per the frozen rule, this does NOT proceed to an ATS
battery on subreddit volume as-is.

Honest caveats around that verdict:

- The failure is borderline, not decisive: the point estimate sits at 0.73
  and its CI extends below 0.7; the raw-scale r (0.62) would pass. The
  primary metric was frozen before computing and is not switched post hoc;
  the honest read is "high shared variance, close to but above the bar",
  not "redundant construct proven".
- Effective n: 294 team-weeks but only 6 teams and 3 seasons; all rows of a
  team share its fanbase size, and the YoY reliability legs rest on 6 data
  points per pair (CIs correspondingly enormous). Nothing here is resolved
  at high precision.
- High correlation measures construct overlap, not predictive value either
  way. Under AGENTS.md this outcome is category-3 information about
  redundancy, not a refutation of any signal; if the owner wants the
  residual (~47% of log-scale variance unshared) pursued, that is a fresh
  decision, ideally with more teams to sharpen both estimates.
- No shared-registry writes were performed by this task: the script stamps
  its own provenance via `write_experiment_artifact` with the stamp rooted at
  `artifacts/arctic_shift_gate/experiment_registry/` (the shared
  `registry/experiments/` tree is untouched). Recording any of this as a
  weak signal via `nfl-ats weak-signals record` remains a separate explicit
  step. Numbers above reproduced identically on a second full run (measured).

