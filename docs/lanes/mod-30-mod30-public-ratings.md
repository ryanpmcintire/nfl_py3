# MOD-30 Public rating disagreement

## Goal
Grade an independent published model (nfelo) against the pool opener on top of the served card probability, using only numbers provably published pregame. ROADMAP row MOD-30.

## State (2026-11-21 16:13 UTC)
- Source settled, no new entry (read): config/source_policies.json:434 `github_nfelo_public_ratings` (added 2026-11-16, acquisition allowed, output_data CSV only, commit time is the as-of proof, no outcome columns); :418 `fivethirtyeight_nfl_elo`. ESPN FPI closed (red entry, 403 block); Massey excluded by robots.
- Forward capture already covers MOD-30 (read): scripts/nfelo_forward.py, lane docs/lanes/nfelo-forward-2026-11-16.md, scheduler jobs nfelo_forward_{capture,pregame,score} (scripts/capture_scheduler.py:996). Stored: 2 Week 11 captures (11-18, 11-21), 12 games admissible; no 2026 outcome graded yet. scripts/capture_public_ratings.py is superseded (would duplicate the source): not run, not scheduled; no scheduler diff needed.
- 2025 commit-stamped look DONE (declared 16:07 UTC below before fetch): scripts/mod30_nfelo_stamped_2025.py fetch|score; 549 commits, 36 snapshots (artifacts/mod30/commits_2025.json, artifacts/mod30/stamped/<sha>.json, game_id + nfelo_dif_base only), result artifacts/mod30/stamped_2025.json. Stamped coverage 272/272 schedule games, 267 served games (opener pushes absent).
- Measured: stamped vs current-file nfelo margin r 0.985, mean abs diff 0.72 pts (2025 values were revised a little after the fact). Coefficient fitted 2020-24: +0.01663 logit/pt (+0.0346 per SD, 1,236 games); frozen forward coefficient +0.01484.
- Stamped look (267 games, 18 week blocks): changed picks 10, record 4-6. Log-loss gain +0.00025 [-0.00177, +0.00226], P+ .60, null pct 76.0 of 200 within-week shuffles, detectable size 0.00151. Brier +0.00013, P+ .58, null pct 81. Accuracy -0.75 pts [-3.0, +1.5], P+ .22, null pct 77.5 (null median -1.5), detectable 1.87 pts. In-sample 2020-24 gain 0.000146; in-sample minus held-out -0.00011.
- Control (current file, same games; LV rows unjoined as in the 2020-25 study, 233 nonzero): changed 8, 4-4; log-loss -0.00007 [-0.0018, +0.0017], P+ .49, null pct 61.
- Reliability (served-probability quintiles, served / combined / observed): .430/.429/.426, .491/.491/.434, .520/.521/.528, .552/.557/.604, .616/.617/.574. The term barely moves the probability.
- Recorded 3 cells, family mod30_nfelo_stamped_2026_11_21 (registry/weak_signals/mod30_nfelo_stamped_2026_11_21.json, new), all unresolved_below_power; pool --league nfl --effect-units accuracy_points exit 0. Looks: 2 (stamped, control), 3 cells.

## Declaration (stamped 2026-11-21 16:07 UTC, before any 2025 commit-stamped value was fetched or joined)
2025 REG; emulated captures Wed and Sat 09:00 America/New_York, latest commit before each; per game the latest capture before min(kickoff, pick deadline). Term = 0.036507 x nfelo_dif_base minus tue_open_home_spread. One offset logistic (no intercept) on out_of_season_home_probability fitted 2020-24, applied to 2025. Week-block bootstrap 2000, 200 within-week shuffles, detectable = null p95 minus null median. Control = current-file values.

## Tried
- 2026-11-03 ESPN probe (policy breach, route closed); 2026-11-16 entries added and 2020-25 served look unresolved (log loss -0.00058, P+ .12, 28-27).
- First score run computed pick deadlines over the whole season at once (every game got the Week 1 Sunday), so only 16 games qualified; fixed by computing deadlines one week at a time. The reported numbers are from the second run.

## Next
- Weeks 11-18 2026 forward grade runs in the nfelo-forward lane (score Tue); after Week 18 pool its cell with this family's stamped look. Then move this lane to done/.

ROADMAP MOD-30 update: Source and capture settled: nfelo via existing policy entry and scripts/nfelo_forward.py (no new source, capture_public_ratings.py superseded). 2025 commit-stamped look (267 games, coefficient fitted 2020-24): changed 10, 4-6; log-loss +0.00025 [-0.0018,+0.0023] P+ .60 null pct 76, detectable 0.0015; accuracy -0.75 pts [-3.0,+1.5] P+ .22; current-file control log-loss -0.00007 P+ .49. Stamped vs current r .985. unresolved_below_power, 3 cells family mod30_nfelo_stamped_2026_11_21. Nothing served. Next: 2026 Weeks 11-18 forward score (nfelo-forward lane).

## Open
- One pregame-provable season is far below the 1.5-pt add-one floor; the 2026 forward weeks add about 120 games.
