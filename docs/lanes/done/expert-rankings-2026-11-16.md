# expert-rankings-2026-11-16

## Goal
Outside-leads-2 candidate 2: FantasyPros weekly expert consensus rankings (ECR) summed to a team offensive expectation, compared with the market. Part 1 source and timing; Part 2 screen at the close, one served look on QW only for a survivor. Owned files: scripts/expert_rankings.py, data/external/fp_ecr/ (gitignored), artifacts/expert_rankings/.

## State (screen done 18:16 EST; frozen 2026 + forward done 2026-11-16 ~18:30 EST, see Results; forward log awaits root scheduler jobs)
- Source (measured): github.com/dynastyprocess/data files/db_fpecr.parquet (nflreadr load_ff_rankings type all), 1,885,069 rows. Weekly pages (QB, PPR RB/WR/TE): 2020-2026 only, one scrape per week, mostly Friday (2019 has one Dec 27 scrape). No 2017-19 weekly history. Policy entry dynastyprocess_fpecr added to config/source_policies.json. Joins: db_playerids.csv (fantasypros_id to gsis_id), nflverse stats_player_week 2020-25 for actual PPR points.

## Declaration (written 2026-11-16 18:12 EST, before any ATS outcome was joined)
- Population: REG 2020-25, weeks 2+, games with a pregame scrape for that week: latest scrape whose date is strictly before the ET date of the pick deadline min(kickoff, Sun 4 PM ET) and whose next Sunday is the game's Sunday. Thursday games after a Friday scrape drop out.
- Rank to points: per season, week and position, isotonic (decreasing) fit of actual PPR points on ECR position rank over that season's earlier weeks (ranked, joined players; did-not-play = 0). No hand constant.
- Team expectation: sum of mapped points over the team's ranked QB/RB/WR/TE in the scrape; converted to real points by OLS of actual team points on that sum over the season's earlier weeks.
- Implied team totals: home minus away = the spread (total cancels). As-of: Tuesday quote (hindsight_audit_team.tuesday_quotes) where available, else the close; coverage reported.
- Cells (2 looks): gap = E_diff_points - spread_asof; gap_orth = E_diff_points minus its OLS fit on spread_asof over the season's earlier weeks. Declared sign POSITIVE on home cover.
- Screen vs the close, pushes dropped: OLS slope per SD in accuracy points, 200 within-season shuffles, halves 2020-22 / 2023-25, season bootstrap. Survivor = above null p95 and positive in both halves; a survivor gets ONE served look at the opener 2020-25, pushes kept, on QW.

## Tried
- Part 1 (measured, artifacts/expert_rankings/part1.json): weekly scrapes 2020 12, 2021 19, 2022 16, 2023 16, 2024 15, 2025 16, 2026 10; weekdays Fri 90, Thu 10, Sat 2, Tue 1, Wed 1. Scrape to deadline date: 2 days for 1,048 games, 3 for 132, 1 for 63. Pregame-covered REG games 152/237/221/221/193/219 (2020-25); missing mostly Week 1 (scrapes start Weeks 2-4 in 2022-24) and Thursday games. gsis join 99.2-99.96% per year; all team codes mapped. Spread as-of Tuesday/SBR open 1,117 games, close 14 (corr to close .967). E_diff vs spread corr .779; k median .279 real points per fantasy point.
- Screen vs close, REG 2020-25 weeks 2+, 1,131 games, 2 looks (measured, screen.json; rerun after lint byte-identical):
  - gap: +1.73 acc pts/SD [-0.36, 3.48], P+ .941, null pct 87.5 (p95 +2.82), halves +1.64/+1.83, 4/6 seasons, gap side 575-556. Not a survivor.
  - gap_orth: +1.32 [-0.07, 2.53], P+ .965, null pct 81.5 (p95 +2.45), halves +0.02/+2.53, 5/6 seasons, 567-564. Not a survivor.
- Recorded: registry/weak_signals/expert_rankings_2026_11_16.json (2 cells, unresolved_below_power, accuracy_points); pool ran clean. No served look (protocol).

## Next
Root: add the scheduler jobs (argv/slots below); run `score` each Tuesday through Week 18; verify registry/weak_signals/expert_rankings_served_2026_11_16.json shows cell 1 refuted_mechanism + wrong_sign_resolved (the readback hit the subagent cap); commit. Then move this lane to done/.

## Open
- Bootstrap over 6 seasons is narrower than the shuffle null; read the null percentile, not the interval.

## Frozen 2026 and forward
### Declaration (written 2026-11-16 18:20 EST, before the opener served stage was run and before any 2026 outcome was joined for this term)
- Term: gap exactly as built in scripts/expert_rankings.py, declared sign POSITIVE on home cover; standardised by the fit population's non-missing mean and SD; no pregame scrape (Week 1, Thursday after a Friday scrape) = 0. Owned: scripts/expert_rankings_forward.py, artifacts/expert_rankings_forward/.
- Step 1 (the family's one served look): build_fit_population keep_pushes 2020-25, pushes labelled 0.5; offset = QW served stage (Fit(FOUR) on decided games, model_logit from v4_candidate opener_QW), held season leave-one-season-out, training offsets nested (held and own season excluded). One offset-logistic coefficient, no intercept. Report record v QW (expect 836-667), changed picks and their record, accuracy points, log-loss gain with season- and week-block bootstrap and probability_positive, fold coefficients, halves, in-sample minus held-out, intercept-only control, reliability. Frozen coefficient = full 2020-25 fit on the LOSO offsets.
- Step 2: 2026 Weeks 2-10 scored once: frozen coefficient on the played card (last publish before deadline) offset at the card's pool line (Tuesday-frozen line, also the as-of spread for the term); rank map and k from 2026 earlier weeks (nflverse 2026 weekly stats); ECR scrape dated before the deadline date. Record on decided games, changed picks, log-loss and Brier gain (pushes 0.5) with week-block bootstrap.
- Step 3: Weeks 11-18 forward log (pregame/score). Recording only; nothing served.
### Results (2026-11-16, artifacts stamped 20261116T2323-2325Z; scripts/expert_rankings_forward.py fit|live|pregame|score|argv|batch)
- Step 1 (measured, fit.json): 1,537 games (34 pushes), term on 1,102; rebuilt gap identical to screen (max diff 0.0); QW reproduced 836-667. Full coefficient +0.0649/SD (mean -1.29, SD 3.75 pts), folds 6/6 positive (0.016-0.096), halves +0.075/+0.052. LOSO term 800-703; changed 115 (3 pushes), term 38-74 v QW 74-38. Accuracy -2.40 pts, week-block [-3.63, -1.07] P+ .0003 (season-block [-3.79, -0.99] P+ 0). Log-loss gain -0.00033 [-0.00185, +0.00124] P+ .36; Brier P+ .36; in-sample +0.00037, gap +0.0007; intercept-only -0.0000003. Diagnostic: gap side at the opener 530-548 (close screen 575-556); corr(x, y-QW) +0.033 drives the positive coefficient while the flips (72 of 112 toward home, home covered 44.6%) lose; QW near 0.5 is 552-467, so not a stage bug.
- Frozen: artifacts/expert_rankings_forward/frozen.json (created 20261116T232359Z, sha 73298c8b...).
- Step 2 (measured, live.json; scored once): 133 settled card games Weeks 2-10 (Week 10 MNF pending), 124 with term, 0 pushes. Term 69-64 v card 67-66; changed 6, 4-2 v 2-4 (wk2 1, wk3 1, wk4 3, wk6 1). Accuracy +1.50 [-2.92, +6.98] P+ .70; log-loss -0.0036 [-0.0090, +0.0017] P+ .096; Brier P+ .095 (9 week blocks).
- Step 3: Week 10 pregame dry run exit 0, newest ECR 2026-11-13, 13 games, 0 changed, matches live term on 12 games (max diff 0.0); `score` with no rows prints "nothing graded", exit 0. Scheduler argv `.tools/uv.exe run --no-sync python scripts/expert_rankings_forward.py pregame` Fri 20:45, Sat 12:05, Sun 09:20 ET; `... score` Tue 09:20 ET. Inputs fetched via nflreadpy to artifacts/expert_rankings_forward/inputs/ (gitignored).
- Recorded (exit 0; pool exit 0): family expert_rankings_served_2026_11_16, 4 cells. Opener accuracy = refuted_mechanism, wrong_sign_resolved (whole interval below zero, applied by rule in `batch`); opener log loss and both 2026 cells = unresolved_below_power.

## Root decision 2026-11-16
- Served look 2020-25 at the opener on QW: 800-703 v 836-667, changed picks 38-74; opener accuracy cell refuted_mechanism (wrong_sign_resolved, interval [-3.63,-1.07] wholly below zero, checked in registry). 2026 W2-10 4-2 on 6 changed picks does not outweigh it. Forward jobs NOT scheduled (owner stop on new lanes, refuted at the opener); scripts/expert_rankings_forward.py kept.
