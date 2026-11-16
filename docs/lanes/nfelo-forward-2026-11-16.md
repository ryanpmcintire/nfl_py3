# nfelo forward 2026-11-16

## Goal
Clean forward test of nfelo's pregame model margin against the pool opener, 2026 Weeks 11-18, on top of the served card probability. The 2020-25 served look (outside-model-gap lane) was 28-27 on changed picks and unresolved_below_power; early-history strength is consistent with retroactive tuning, so only pregame numbers stamped by the GitHub commit time count.

## Declaration (stamped 2026-11-16 21:08 UTC, before freezing and before any Week 11 nfelo number or outcome was seen)
- Source: github greerreNFL/nfelo output_data/nfelo_games.csv, read at a pinned commit SHA; the commit time comes from the GitHub commits API. Upcoming-week rows (nfelo_dif_base) appear pregame (Week 10 rows present at commit ec46b415, 2026-11-10 13:41 UTC; the Week 10 tracker first landed between 2026-11-09 13:40 and 2026-11-10 13:41 UTC, i.e. Monday night to Tuesday morning; the workflow commits several times a day).
- Term: nfelo_gap = nfelo model home margin minus the pool opener (Tuesday-open home line from mod25_wf_forward.tuesday_open, home-margin convention), points, home minus away. Model margin = points_per_elo x nfelo_dif_base, points_per_elo the value measured in outside_model_gap (least squares of nfelo's own home_line_pre_regression on nfelo_dif_base, no outcomes). Market-blended nfelo columns are not used.
- Frozen coefficient: the mean of the six 2020-25 LOSO fold coefficients per SD in artifacts/outside_model_gap/served.json, divided by the SD of nfelo_gap on that same served population (1,503 games). No new fit.
- Combined: logit(p) = logit(served card home probability at the last publication before the game's cutoff) + coefficient x nfelo_gap. Base = served card probability.
- Admissible capture: commit time and capture time both before min(kickoff, pick deadline); latest admissible capture per game is used.
- Metric: log-loss gain vs the card (week-block bootstrap interval and probability_positive), Brier, changed-pick record; ends Week 18. A zero-crossing interval closes nothing; verdict via weak-signals record.

## State (2026-11-16 21:10 UTC)
- Frozen artifacts/nfelo_forward/frozen.json (20261116T210946Z, sha 4bcac3e6): coefficient 0.014843 logit per point = LOSO mean 0.030886 per SD / term SD 2.0809 pts (1,503 games, matched study); points_per_elo 0.036507 (1,564 rows).
- scripts/nfelo_forward.py freeze|capture|pregame|score; ruff clean. Captures go to artifacts/nfelo_forward/captures/, pregame files to artifacts/nfelo_forward/.
- Week 11 capture 21:10 UTC: commit 5dfa6152 (2026-11-16 20:42 UTC) has 0 Week 11 rows; nothing written. Pregame Week 11: all 13 games no_admissible_capture; nothing written. Score: nothing graded, exit 0.
- Week 10 dry run (live file): 14/14 games, nfelo margin vs opener r 0.875, 1 changed side (WAS at NYG); nothing written.
- Timing: nfelo upcoming-week rows land Mon night to Tue morning UTC; cards first published Tue (Wk9 11-03 17:36 UTC, Wk10 11-11 01:21 UTC), so Wed 09:00 pregame sees both.

## Tried
- Team codes mapped OAK->LV etc. forward; the 2020-25 study joined raw nfelo ids (LV games got term 0). Named deviation; coefficient unchanged.

## Next
- Root adds scheduler jobs (argv run from repo root): .\.tools\uv.exe run --no-sync python scripts/nfelo_forward.py capture | pregame | score. Slots: capture Wed 09:00 and Sat 09:00 local, pregame right after each capture, score Tue 13:40.
- After Week 18 score: record via weak-signals record (family outside_model_gap_2026_11_16), move lane to done/.

## Open
- Wed capture uses auto week = first week with an unplayed game; on Wed that is the coming week.
