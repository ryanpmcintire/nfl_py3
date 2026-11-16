# XLG-14 College-first discovery

## Goal
Screen declared candidate families on college games first, carry only survivors to one pre-declared NFL confirmation each (ROADMAP row XLG-14). Zero crossing closes nothing; one fitted probability selects the side; every parameter leave-one-season-out; every cell a look; opener input and opener grade.

## State (2026-11-16)
Unit 4, scoreboard-over-play-quality family `xlg14_scoreboard_family_u4`, is screened and recorded. One of four candidates passed the declared two-sided rule: `season_epa_gap`, with the sign OPPOSITE to the hypothesis (positive). Its NFL confirmation ran once: positive sign, 47.5th percentile, not confirmed, unresolved (3 cells). Nothing is served.

## Unit 4 declaration (2026-11-16 12:38, before the script was written or any outcome joined)
- Mechanism: the opener over-reads final scores; points carry noise that play-level efficiency strips out, so a team whose recent scoreboard outran its play quality is over-priced next week. Category market. Hypothesis sign negative for all; rule two-sided.
- Novelty (**read**): luck families spent (unit 3; NFL close-game, turnover, return-TD, field-goal entries); workload/rest spent (unit 1 MOD-41, sim13, unit 2 schedule incl. prior ATS); `xlg_early_overreaction_cfb` is surprise against earlier closing lines; MOD-28 is an NFL late-game compression trait. No registry or ROADMAP row tests points margin against EPA or success margin, or fourth-quarter margin, as an opener term.
- Inputs: college regular-season play-by-play (unit 3 partitions). Final and end-of-third score = maximum of post-play score columns; EPA margin = own offensive EPA sum minus opponent's; success margin = own mean `EPA_success` minus opponent's. Points margin regressed on each play margin by least squares on team-games in strictly earlier seasons.
- Candidates (team value from games in strictly earlier weeks of the same season, 0 with none, home minus away): (1) `last_game_epa_gap` previous game's margin minus EPA-predicted margin; (2) `last_game_success_gap` same with success margin; (3) `season_epa_gap` per-game mean of (1) over all earlier weeks; (4) `last_game_q4_margin` previous game's margin from the fourth period on.
- Harness: unit 3 `read_candidate` (offset add-one per SD, no intercept, LOSO), 200 within-season shuffles, seed 36 stream 4. Survivor: log-loss gain above the 95th percentile of its own null AND equal coefficient sign in both halves (2012-18 / 2019, 2021-25). Looks: 4 gating, 12 recorded cells, 8 half coefficients, 2 controls. Detectable (**read**, `docs/lanes/xlg-harness-power-2026-11-14.md`): .08-.11 per SD (+0.7 to +1.1 points) four times in five.

## Unit 4 result (**measured**; `scripts/xlg14_family_u4.py build` then `screen`; `artifacts/xlg14_family_u4/` build.log, build.json, screen.log, summary.json, weak_signals_batch.json, team_values.parquet)
8,925 college games, 13 seasons, all with both teams. As-of check: 5 cutoffs, 486 team-games, max diff 1.8e-15. Earlier-season fits: EPA slope 1.04-1.08 points per EPA point, success 163-182, intercept 0. Null log-loss 5th-95th about [-.0002, +.0001].

| Candidate (nonzero) | Changed; record | Accuracy pts [95%] P+ (null pct) | Log-loss gain [95%] P+ (null pct) | In-sample; gap | Coef/SD; folds; halves |
|---|---|---|---|---|---|
| last_game_epa_gap (8,399) | 485; 244-241 | +0.03 [-0.45,+0.50] .55 (74.0) | -.00018 [-.00041,+.00004] .05 (7.5) | +.00006; +.00024 | +.021; 12/13; +.046 / -.004 |
| last_game_success_gap (8,399) | 232; 118-114 | +0.05 [-0.29,+0.38] .61 (71.2) | -.00017 [-.00032,-.00004] .004 (7.5) | +.00001; +.00019 | -.010; 13/13; +.008 / -.032 |
| season_epa_gap (8,399) | 1,532; 788-744 | +0.49 [-0.39,+1.36] .87 (98.0) | +.00055 [-.00026,+.00136] .91 (100.0) | +.00078; +.00023 | +.079; 13/13; +.119 / +.039 |
| last_game_q4_margin (7,808) | 924; 480-444 | +0.40 [-0.26,+1.05] .88 (93.0) | +.00009 [-.00030,+.00048] .66 (93.0) | +.00018; +.00008 | +.038; 13/13; +.066 / +.007 |

Controls: base 50.48% / .69400; constant 50.72% / .69311; base plus intercept 50.40% / .69403.
- Verdict: `season_epa_gap` survives (100th percentile, both halves positive, 13/13 folds); size .079 per SD sits at the bottom of the detectable band. Its sign is opposite to the declared over-reaction hypothesis: teams that outscored their EPA so far keep covering, so the opener UNDER-reads it. **Inferred:** with EPA slope near 1, the gap is mostly points the offensive EPA sum does not carry (special teams, defence, scoring-play conversion), a persistent skill rather than noise. Four gating reads at a 95th bar: about 19% of no-information families pass one (1 - .95^4). Survival buys one NFL look only.
- The other three: no visible effect; none closed. The two last-game gaps carry whole or near-whole negative log-loss intervals but sit at the 7.5th percentile of their nulls: one fitted term's cost, not a resolved wrong sign. q4 reached 93rd on both, below the bar.
- Recorded: 12 cells `xlg14_scoreboard_cfb_<candidate>_<units>` (league cfb, category market, unresolved_below_power). `weak-signals pool --league nfl --effect-units accuracy_points` exit 0.

## NFL confirmation of `season_epa_gap` (declared 2026-11-16, run once 2026-11-16)
- Build: NFL play-by-play (`data/pbp/raw` latest snapshot per season), regular season; per team-game points margin from final scores and EPA margin = own `epa` sum on plays where `posteam` is the team minus the opponent's; slope and intercept from team-games in strictly earlier seasons; value = per-game mean over strictly earlier weeks of the same season of margin minus predicted margin, 0 in week 1, home minus away; same as-of check (five season-weeks, tolerance 1.5e-8).
- Frame: MOD-36 `games()` (NFL 2020-25, opener grade, 1,503 games), base served `out_of_season_home_probability`; offset add-one, no intercept, LOSO; intercept-only control; 200 within-season shuffles on a new stream (seed 36, template index 5).
- The single read: held-out log-loss percentile in its matched null. Expected sign POSITIVE (the college sign; the original negative hypothesis is already contradicted in college). Confirmed = positive full-sample coefficient and percentile above 95; positive and at or below 95 = not confirmed; negative = opposite sign. Refuted only if the 95% game-bootstrap interval of the coefficient is wholly negative. NFL sees about .26 per SD four times in five, so a college-sized .08 will usually not show: expect a sign and a percentile. Record league nfl in family `xlg14_scoreboard_family_u4`. Passing is not serving.
- Result (**measured**; `scripts/xlg14_scoreboard_nfl_confirmation.py build` then `confirm`; `artifacts/xlg14_family_u4/nfl_confirmation/` build.log, build.json, confirm.log, summary.json, weak_signals_batch.json, pool.log). Final scores from `game_features` home/away_score (NFL play-by-play carries only `posteam_score_post`); EPA from all regular-season plays with `posteam` and `epa`, 2009-25. As-of check passed: 5 cutoffs, 156 team-games, max diff 1.1e-16. Earlier-season slope .869-.886 points per EPA point, intercept 0. Base `artifacts/pick_probability/20261116T171555Z/per_game.parquet`; 1,503 games, 1,415 nonzero, SD 1.21 points.
- Declared read: log-loss gain -.00037 [-.00136,+.00063] P+ .23, null percentile 47.5 (null 5-95 [-.00137,+.00055]). Coefficient +.034 per SD [-.062,+.137] P+ .76 (null pct 75.0; null SD .049); folds positive 5/6 (2022 -.006). Changed picks 66, record 30-36. Accuracy -0.40 [-1.40,+0.67] P+ .22 (pct 36.5); Brier -.00019 [-.00068,+.00030] P+ .22 (pct 47.0). In-sample ll +.00015, gap +.00052. Controls: base 55.16% / .68671; constant 49.57% / .69328; base plus intercept 55.29% / .68675.
- Outcome: **not confirmed** (positive sign, 47.5th percentile at or below 95). Not refuted (interval not wholly negative). Three cells `xlg14_scoreboard_nfl_season_epa_gap_<units>` recorded unresolved_below_power; `weak-signals pool --league nfl --effect-units accuracy_points` exit 0. Do not re-cut.
- Overlap (**measured**, Pearson, 1,503 games): with served logit +.010; with `diff_off_epa_per_play` -.224, `diff_def_epa_per_play` +.192, off pass -.231, off rush -.084, def pass +.187, def rush +.079. The market/model probability does not carry it; it leans against the model's EPA-per-play inputs (**inferred**: a team scoring above its EPA tends to have worse per-play EPA, so part of the gap is the per-play inputs' complement).

## Tried
- Unit 1 (2026-11-03): MOD-41 workload on college, 6 cells `xlg14_mod41_cfb_u1`; nothing refuted.
- Unit 2 (2026-11-03): schedule family (short week, bye, prior ATS, early carryover, road trip), 5 cells `xlg14_family_u2`; zero survivors.
- Unit 3 (2026-11-14): luck family `xlg14_luck_family_u3`, 15 cells; survivor `fumble_recovery_luck` (99.5th pct, -.058/SD); its declared NFL confirmation ran once: opposite sign (+.013/SD, 16th pct), not confirmed, unresolved, 3 cells. Do not re-cut it. Full tables in git history of this file and `artifacts/xlg14_luck_family/`.
- `xlg_harness_power.py` and `mod36_addone_noise_floor.py` end in bare `main()`; both are loaded from source without that call.

## Next
- A fifth college family, or split-half reliability of the unit 3/4 traits for the registry `reliability` field.

## Open
- `reliability` is unrecorded on all 33 XLG-14 cells.
- What the college EPA gap is made of (special teams vs scoring conversion) is inferred, not measured.
- ROADMAP row XLG-14 and the handoff are the orchestrator's to update; nothing committed.
