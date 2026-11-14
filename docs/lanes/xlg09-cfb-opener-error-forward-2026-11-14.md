# XLG-09 college opener-error term: frozen forward log

## Goal
Log, on games not yet played and at the pool's Tuesday line, the term that learns where the opening line is wrong from college football only and applies that rule to NFL openers (`_v3`, `docs/lanes/done/opener-error-transfer.md`). History cannot settle it: 2020-25 reads +0.67 accuracy points [-0.32, +1.50], p+ .92, under the roughly 1.5 points the NFL harness can see. Research only; nothing is served and no pick changes.

## Declaration (stamped 2026-11-14 18:19:49 ET (-05:00) by Get-Date; frozen file created 2026-11-14T23:14:02Z)
- Term `xlg09_cfb_opener_served`. Frozen: `artifacts/forward_terms/xlg09_cfb_opener_served/frozen_coefficients.json`, sha256 3043e17f4a9cdd85825480899392636a3bd3129516d4c3e724509261c1e8f4ed, coefficient hash 02ff2d5b...415e. Fit script `scripts/xlg09_freeze_cfb_opener_served.py` (sha256 881c93c3...c956; refuses to overwrite).
- Rule: college ridge logistic of home cover at the college opener, 4,301 games, seasons 2012-19 and 2021-25; NFL inputs scaled by the college means and SDs. Combined logit = card logit - 0.0237 - 0.4600 x rule logit. No in-season refit; adding college 2026 or changing any input is a new term with a new name.
- Inputs, each known at the Tuesday job (read `week_terms` in the fit script): the game's published card line (favourite flag, spread size, distance to the nearest of 3, 7, 10, 14, 17); each team's previous game this season: its schedule closing line minus its card line (prior move) and the days since it (rest). No same-game closing move is used. Fails closed when a previous game has no result in the newest schedule snapshot.
- Tuesday versions that differ from the history, a new look: the previous game is the schedule's (the fit used the previous game in the fitted population; 149 of 1,503 fit rows differ, measured); the close is the schedule close (the history used a captured consensus where one existed; equal on 1,047 of 1,537, measured).
- Population: 2026 Weeks 11-18, 122 scheduled games (measured), every game with a record written before kickoff; Week 10 is excluded. 2027 continues on the same file only after a dated addendum extends the runner's season and week range before 2027 Week 1.
- Primary metric (one): paired log loss against the published card on matched graded games, read with the matched-null percentile. Secondary (four): the record on games where the side differs from the card (stated first), Brier, accuracy, and the forward slope of cover on the stored rule logit with the card logit as offset (sign and interval).
- Looks: 5, read once on Tue 2027-01-12 after Week 18 settles, then `weak-signals record`. Weekly ledger lines before then are monitoring. A probability_positive short of any bar closes nothing.

## State (measured this session unless labelled)
- Stage 2 on the served probability (`out_of_season_home_probability`, `artifacts/pick_probability/20261114T184614Z`, offset coefficient 1), 2020-25 true Tuesday opener, 1,503 games, leave-one-season-out. Held-out-fold coefficient: 2020 -0.508, 2021 -0.601, 2022 -0.404, 2023 -0.421, 2024 -0.295, 2025 -0.616, negative in all six. Frozen all-season b -0.4600, intercept -0.0237 (control intercept +0.0028).
- Held out against intercept-only: sides differ on 180 games, term 86, control 94; log loss gain +0.00011 [-0.00167, +0.00196] p+ 0.550; Brier +0.00007 [-0.00080, +0.00097] p+ 0.566; accuracy -0.53 points [-2.33, +1.20] p+ 0.273. In-sample log loss gain +0.00045, gap 0.00034. Game-level bootstrap. Recorded unresolved_below_power as `xlg09_cfb_opener_served_loso_{logloss,brier,accuracy}` (batch in `tests/scratch/xlg09_forward/`); `weak-signals pool` exit 0, 4,642 signals.
- Sign: the fitted NFL coefficient on the college rule is negative, so the frozen term fades the college rule. The corrected v3 run agrees: its fold coefficient on the rule is negative in all 13 folds, -0.003 to -0.049 per SD (read `artifacts/opener_error_transfer_unit3/20261114T225530Z/summary.json`); the done lane's line saying positive in all 13 folds predates the sign repair.
- Registered in `scripts/forward_terms_week.py` (TERMS, served base, skips weeks outside 11-18 unless `--dry`) and `scripts/forward_ledger.py` TERM_DIRS. No scheduler edit (read `capture_scheduler.py`:945-954 `forward_terms_week.py --week auto`, default all terms; :770-781 `forward_weekly.py`).
- Dry run on Week 10, nothing written: 14 games, 14 with a value, rule logit mean +0.021, SD 0.068, range -0.072 (WAS@NYG) to +0.143 (LA@ARI); largest probability move 0.022 (LA@ARI 0.429 to 0.407); 0 side changes against the card. Scratch output deleted.
- No existing record changed: `artifacts/forward_terms` hashed before and after, 39 files identical, one added (the frozen file). `forward_weekly.py` exit 0; Weeks 1-9 ledgers (136 games, 87 columns, all-weeks and core) identical to the 18:02 ET copies; the manifest now lists the term as having no record yet. ruff check and format pass on the three scripts.

## Tried
- Diagnostic, 2 looks, not recorded: slope of cover on the rule logit with no offset is -0.09 on 2013-19 archive openers and -0.20 on 2020-25 true openers; by season 2020-25: -0.39, -0.31, -2.61, -0.75, -3.00, +2.01 (rule SD 0.13, unstable).

## Next
- Tue 2026-11-17: the existing forward job writes the Week 11 record once the card is published; `forward_weekly.py` scores it as `term_xlg09_cfb_opener_served` after Week 11 settles. Score the declared looks on Tue 2027-01-12.

## Open
- The motivating +0.67 was measured against v3's own refit base over 13 seasons; on the served base the same rule reads -0.53 accuracy points. Root decides whether the done lane needs the negative-coefficient note and whether a second coefficient deserves its own declared term.
- 122 games cannot resolve an effect this size alone; the log is accretive and is meant to continue in 2027.
- If the schedule snapshot at job time lacks Monday's result, the term fails closed that week; rerun before kickoff.
