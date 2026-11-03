# XLG-14 College-first discovery

## Goal
College-first discovery: screen a declared candidate family on CFB, carry only the survivors to a pre-declared NFL confirmation ROADMAP row XLG-14.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: CFB supplies about 4-5x the graded games and the pooled ranking holds in 2023-25, so a family of candidate features has far more power to show a real sign in CFB. Discovering there and confirming once in NFL cuts the NFL look count from K to the survivors, which is the multiplicity that currently buries small real signals. Data: CFB espn_book_era 2023-25 opener population and the shared 14-input harness (mod24-more-games lane); candidate family declared before any sign is seen; NFL opener grade 2020-25 as the one-shot confirmation.

## Tried
Novelty check only: Nearest: cfb_lead_screens_wave1.md and wave2.md (NFL leads replicated in CFB, the opposite direction), XLG-10 (NFL candidates screened on the pooled harness), cfb_replication_program_20260901.md. Differs: CFB is the discovery arm and NFL the single confirmation; grep for college-first and CFB-first discovery returned no hit.

## Next
First unit: Declare the family (K candidates with college analogs, survivor count k fixed by the declared look budget, not by sign), run the CFB leave-one-season-out screen, freeze the survivor list and its expected sign in a registry declaration, and only then compute the NFL confirmation; count every candidate as a look in both leagues.
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Look count and family to be declared before any sign is seen. Nothing here closes on zero crossing; record unresolved results with nfl-ats weak-signals record.

## Unit 1 DECLARATION: out-of-league replication of MOD-41 (written 2026-11-03 16:50 local BEFORE any outcome computed)
Spec (analog of scripts/mod41_cumulative_workload.py + mod41_unit2_components.py), script scripts/xlg14_mod41_cfb.py.
- Plays: CFB pbp rush or pass plays, regular season, not kneel_down, pos_team_id present (no EPA or competitive filter; NFL used all pass/run). run = plays by team offense in the game, defended = plays by the opponent offense, total = run + defended.
- Per team-game: load_total = mean of total over the prior 3 games (same season) minus the season-to-date mean of total over all prior games. own_td = season-to-date mean of run (prior games); opp3 = mean of defended over prior 3 games. Usable = 4+ prior games for both teams, else term 0. Game term s_* = home minus away.
- s_resid = s_load_total minus OLS fit on (1, s_own, s_opp) over usable rows (pooled, as NFL). Primary cell: s_resid.
- Population: FBS vs FBS regular season (table filter in cfb_game_features), seasons 2012-2019 and 2021-2025 with spread_open present, 2020 excluded, pushes (result == spread_open) dropped. 2026 excluded (no results).
- Base: LOSO ridge (alpha 10, median impute, standardized, same as xlg12 base) on ats-margin-vs-opener (result - spread_open) using the pooled shared columns with spread_open as the market input and NO closing line column (spread_line, spread_dispersion, total_line dropped); p = Phi(pred/resid_sd) with resid_sd from the training seasons; logit_p = logit(p). Then LOSO logistic [logit_p] is the base, [logit_p, term] the test, as in NFL. Intercept-only LOSO control reported. Grade: home cover vs opener (y = result > spread_open).
- Metric: out-of-sample log loss improvement vs base, LOSO by held-out season (13 folds); blocks = season-week bootstrap 1000 draws, interval and probability_positive; accuracy points secondary; decisive (pick changed) record first; in-sample vs out-of-sample gap; per-fold coefficient on the term with NFL sign reference (NFL residual coef -0.0395..-0.0501, same sign = negative = more recent residual workload for home hurts home cover).
- Looks declared (6): total, resid (primary), own, opp, joint own+opp+resid, resid placebo (window shifted one game earlier). Recorded as log_loss_improvement, unresolved_below_power, league cfb, family xlg14_mod41_cfb_u1.
- Commensurability caveat: CFB plays/game, tempo, season length (12 games vs 17) and population differ; no pooling with NFL unless declared before signs (not declared). Refutation only if whole interval on the wrong (positive coef, ll loss) side with resolved sign.

## Unit 1 status 2026-11-03 16:54
Declaration above written first. scripts/xlg14_mod41_cfb.py written (ruff clean), launched once in background (task bkcui0tkq; log in scratchpad xlg14.log). Output to artifacts/xlg14/mod41_cfb.json when done. Not yet read; no cells recorded. Next: read the artifact, report per-fold coef vs NFL negative sign, ll gain with P+, decisive record, then record 6 cells (log_loss_improvement, unresolved_below_power, league cfb, family xlg14_mod41_cfb_u1, 2012-2025 ex 2020).

## Unit 1 RESULT 2026-11-03 16:57
artifacts/xlg14/mod41_cfb.json; 6 cells recorded (xlg14_mod41_cfb_u1, unresolved_below_power, league cfb). 8925 games, 6145 usable, 13 seasons. Base acc 50.3% (opener input), intercept ll .69311, base .69304.
- resid (primary): coef +.0013 (NFL -.044), positive in 11/13 folds; ll gain -.00013 [-.00025,-.00001] p+ .014; decisive 202-191 of 393; gap .00014. Opposite sign, near-zero magnitude. Total: coef -.0018, ll -.00011 p+ .09, 258-256. Own coef -.0072 ll +.00015 p+ .74; opp ll +.00024 p+ .79; joint +.00021 p+ .72; placebo +.00008 p+ .61.
- Not recorded as wrong_sign_resolved: CFB-only, not commensurable (tempo, 12-game seasons, base ~coin flip), magnitude 1/30 of NFL. NFL signal neither strengthened nor refuted.
Next: CFB EPA-based load or per-play fatigue analog; then XLG-14 family declaration.

## Unit 2 DECLARATION: family screen (written 2026-11-03 16:59 local BEFORE any outcome computed)
Script scripts/xlg14_family_screen.py; CFB arm uses the unit 1 harness (base logit_p from LOSO ridge on opener, close columns dropped; LOSO logistic [logit_p] vs [logit_p, term]; season-week block bootstrap 1000; 13 folds). All features come from the game table alone (gameday, home/away team, neutral flag, result, spread_open), so they exist in both leagues. Prior = the team's previous game in the same season; missing prior or a first game gives 0. Game term s = home value minus away value. Neutral-site games count as not-away. Rest = days since own previous game. WINDOW=3 inherited from unit 1 (no new constant).
Candidates (5 looks, sign hypothesis = expected coef sign of s on home cover):
1. short_week: 1[rest < 7] home minus away (7 = one calendar week, the schedule modal). Hypothesis negative (less recovery hurts the short side). Base carries rest_diff linearly; this is the threshold nonlinearity.
2. bye: 1[rest >= 13] home minus away (13 = a skipped week). Hypothesis positive (extra prep and recovery).
3. prior_ats: sign(prior-game margin minus that game's opener, team perspective) home minus away; prior game needs its own opener. Hypothesis negative (market over-weights last result; reversion). Sign used, so no cap constant.
4. early_carryover: (home prior-season mean margin minus away prior-season mean margin) times 1[both teams have played fewer than WINDOW games this season]. Hypothesis positive (early lines under-use prior-season strength).
5. road_trip: 1[away team previous game was away] minus 1[home team previous game was away]. Hypothesis positive (consecutive road games fatigue/travel on the visitor).
Not carried (no free data in both leagues): kickoff temperature (CFB table has none), turnover-luck and special-teams luck (would need NFL pbp rebuild; deferred to a later unit, not dropped for sign).
Promotion screen (never a closure): carry to NFL if coef has hypothesised sign in >= 10 of 13 LOSO folds (75%, ceil) AND out-of-sample log-loss gain P+ >= 0.75, and the CFB coefficient sign equals the hypothesis. If more than 2 pass, carry the 2 with highest P+. Decisive record (pick changed) reported first. Failing the screen does not close a candidate; it just is not spent on NFL.
NFL confirmation (survivors only, one cell each): 2020-25 opener grade, per_game.parquet from artifacts/opener_evaluation/20260929T192743Z, base = served logit of home_cover_probability_at_open, LOSO logistic add-one, intercept-only control, same bootstrap. Prior ATS uses per_game openers within season.
Recording: log_loss_improvement, family xlg14_family_u2, league cfb or nfl, unresolved_below_power unless a closing ground is met. Total looks: 5 CFB + k NFL.

## Unit 2 status 2026-11-03 17:05
Declaration above written first. scripts/xlg14_family_screen.py written (ruff clean). CFB screen launched once in background (task bz9x5zdx7, output C:\Users\Ryan\AppData\Local\Temp\claude\F--Repos-nfl-py3\352ecbdc-246f-44c0-9f81-c287a6ef4436\tasks\bz9x5zdx7.output); writes artifacts/xlg14/family_cfb.json. Not read; nothing recorded; no NFL run. Next: read artifact, apply declared screen (>=10/13 folds hypothesis sign and P+ >=0.75, max 2), freeze survivors here, run `scripts/xlg14_family_screen.py nfl <survivors>`, record cells (family xlg14_family_u2).

## Unit 2 RESULT 2026-11-03 17:15
artifacts/xlg14/family_cfb.json, 8925 games. Per candidate: decisive (term-base correct of changed), ll gain [CI] P+, coef, folds with hypothesis sign.
- short_week 256-253 of 509; -.00007 [-.00031,.00018] P+ .29; coef -.060; 13/13 sign
- bye 162-174 of 336; -.00010 [-.00021,.00001] P+ .04; +.017; 13/13
- prior_ats 116-115 of 231; -.00007 [-.00011,-.00002] P+ .005; 8/13
- early_carryover 70-65 of 135; -.00008 [-.00015,-.00002] P+ .001; 4/13
- road_trip 287-290 of 577; -.00015 [-.00027,-.00003] P+ .007; 2/13
Screen (P+ >= .75): zero survivors, so no NFL confirmation run (5 looks total). Not a closure. Fold sign for short_week and bye is 13/13 but ll gain is negative (sign without out-of-sample gain). Not yet recorded via weak-signals record (5 CFB cells, unresolved_below_power, family xlg14_family_u2).
Next: record the 5 cells; then a family with data needing a rebuild (turnover/ST luck from pbp in both leagues).
- 2026-11-03 17:04 ET (root): 5 CFB cells recorded (family xlg14_family_u2, unresolved_below_power) from artifacts/xlg14/family_cfb_batch.json.
