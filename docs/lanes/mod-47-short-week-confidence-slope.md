# MOD-47

## Goal
Short-week confidence slope: calibration slope of the served logit inside short-week games, as a fitted interaction on the one calibrated probability.

## State
Current (2026-11-14 15:03 local): units 1-3 done; unit 3 (side term, bottom of this file) is the latest. Nothing served, no src change; every cell unresolved_below_power.
Unit 1 done 2020-25 / 2011-19 (2026-11-03 ~17:40). Script scripts/mod47_short_week_slope.py (log in scratchpad m47.log). Short = min(home,away) team rest days < median team rest (7.0, derived); week-1 rest unknown excluded; pushes excluded. Looks: 4 slope fits, 2 LOSO interaction arms, ~8 dLL contrasts (family mod47_short_week_slope).
Slopes (served logit): 2020-25 short 0.54 [-0.22,1.33] n=309, normal 0.31 [-0.10,0.73] n=1106; 2011-19 SBR short 0.10 [-0.26,0.46] n=444, normal 0.11 [-0.06,0.30] n=1646. Served prob is overconfident in BOTH groups (slope far below 1); short vs normal not separable.
LOSO z x short interaction vs served: 2020-25 dLL -0.00149 [-0.0036,+0.0006] P+ .08; decisive 256 games, interaction 117 / served 139; zs coef per fold -0.19..+0.72 unstable. 2011-19: dLL -0.00103 [-0.0019,-0.0002] P+ .008; decisive 217, 105/112; zs coef ~0 (-0.19..+0.07). Intercept-only control also not beaten.
Recorded 2 cells (log_loss_improvement, unresolved_below_power).

## Tried
Unit 1: interaction (z, short, z*short) LOSO; intercept-only and served-only controls. Unit 2: global slope and intercept shrink on the served probability. Unit 3: home-short minus away-short as one fitted term on the served offset and inside a four-input refit, with intercept-only and within-season shuffled controls (arm A's training offsets were fitted with the held-out season inside them, a second-order leak; arm B has none and agrees).

## Next
1. The Thursday question is a different term: when both teams are short the only side asymmetry is home versus road (travel on short rest). Declare and grade a both-short home term on the served offset (97 games 2020-25). First check registry/weak_signals/public_claim_battery.json and the other short-week rows so nothing is rerun.
2. More power for the differential needs the 2011-19 proxy era on a served-equivalent base (unit 1 had 2,090 games there); a graded rest-day differential only as a newly declared family.

## Open
Unit 3 answered the earlier question: XLG-10's +0.60 on the raw ridge did not carry to the served base (-0.13). The 2020-25 lean (shorter side covered 93-85) and 2026 (shorter side 4-11) point opposite ways; neither is resolved.

## Unit 2: global calibration of the served probability (2026-11-03 17:26 local)
Script scripts/mod47_unit2_global_calibration.py (log in scratchpad m47u2.log). Unit 1's z = raw model_probability (= home_cover_probability_at_open, max diff 0), NOT the served probability. Served = four_term_pick_probability_v1 (pick_probability.py:795 calibrated_pick_probability; active artifacts/pick_probability/20261103T195411Z, model_logit coef 0.308 already the global shrink). Served OOS = per_game.parquet out_of_season_home_probability, n=1503 2020-25.
Slope [95% boot]: raw 0.347 [0.009,0.671]; served OOS 0.922 [0.507,1.363] P(slope<1) .64, intercept 0.001 [-0.099,0.105]; in-sample 0.999. LL: coin 0.69315, raw 0.69692, served 0.68670. Served vs coin dLL +0.00644 P+ .986; vs raw +0.01021 P+ .995. Market-neutral baseline is 0.5 at the opener (same as coin).
LOSO slope+intercept shrink on served: LL 0.68692, dLL vs served -0.00021 [-0.00077,+0.00035] P+ .206 (worse, 2 of 6 seasons >0; fold slopes .86-1.01). 10 side changes only because of the fitted intercept; slope-only shrink is monotone and never flips.
Live 2026 Wk1-8 n=121: slope 1.42 [-0.89,4.51], LL 0.68586 vs coin 0.69315, mean p .551 obs .537 (small sample, descriptive).
Looks: 3 slope fits + 1 live fit, 1 LOSO arm, 3 dLL contrasts, 2 reliability tables (family mod47_global_calibration). Recorded 1 cell.
Conclusion: served is NOT overconfident; unit 1's overconfidence was the raw model, already corrected inside the served fit. No global shrink warranted; no v8 enrolment. Next: short-week as side-specific term (home short minus away short) inside the four-term fit.

## Unit 3: side-specific short-week term (declared 2026-11-14 14:58 local, before any outcome join)
Prior work, not rerun: unit 1 and mod51_mod47_short_week_interaction_served measured a confidence-slope interaction (z x either-team-short), not a side term. XLG-10 `short_week_diff` (home short minus away short, NFL +0.60 [-0.14,+1.39] P+ .92, flips 23-14) was an input added to the raw 14-input ridge margin model, graded against that raw base. mnf_road_short_week_on_production was a Monday-road-only flag on the retired weak_stack over two seasons. None put the differential beside the served probability; that is this unit.
Family `mod47_short_week_side_term`. Term `short_diff = home_short - away_short` in {-1,0,+1}; a team is short when its previous game that season was fewer than 7 days earlier (7 = the calendar week, XLG-10's NATURAL_WEEK_DAYS; unit 1 measured median team rest 7.0); a season opener is not short. Built only from published game dates in data/processed/game_features.parquet, known long before the Tuesday freeze. Mechanism: recovery and preparation deficit relative to the opponent, under-priced by the opener; expected sign NEGATIVE on the home logit (the shorter side covers less). When both teams are short (most Thursday games) the term is zero by construction.
Arms (4): A offset, logit p = logit(served out-of-season) + c*short_diff, one unpenalised coefficient, LOSO 2020-25; B joint, LOSO refit of the served inputs in per_game (model_logit, division+protection flag sum, move available, move toward home) with versus without short_diff, graded against its own four-input refit; C control, intercept only on the served offset; D control, short_diff permuted within season (1000 draws) through arm A.
Metrics (3): accuracy points, log-loss improvement, Brier improvement, paired game bootstrap (2000), P+ with ties counted half. Looks: 6 graded contrasts (2 arms x 3 metrics), 2 controls, in-sample versus out-of-sample per arm, 1 frozen-coefficient read on 2026 Weeks 1-9 (descriptive). Opener grade, pushes excluded (per_game drops 34), served column `out_of_season_home_probability` from artifacts/pick_probability/20261114T184614Z; no line is a model input.

### Unit 3 result (2026-11-14 15:03 local; all measured)
Script scripts/mod47_short_week_side_term.py, artifacts/mod47/report.json, n=1503.
- Counts: 178 games with a nonzero differential (home only 95, away only 83; by season 22/19/24/36/40/37); 161 Sunday, 16 Saturday, 1 Monday; 141 are a 6-versus-7-day pair (Monday-night team the next Sunday). Thursday: 97 games with both teams short, 0 with a nonzero differential, so the term never fires on a Thursday.
- Raw: the shorter side covered 93-85; the served pick was already on the shorter side in 103 of 178.
- Arm A offset: decisive 11-13 of 24; accuracy -0.13 [-0.73,+0.53] P+ .37; log loss -0.00046 [-0.00126,+0.00037] P+ .14; Brier -0.00023 [-0.00063,+0.00017] P+ .13. In-sample -0.13 / +0.00007 / +0.00003, gap 0.00 / +0.00053 / +0.00026. Fold coefficients +.05 -.04 +.13 +.08 +.09 +.12 (5 of 6 positive, opposite the declared sign); full +0.068 [-0.235,+0.359], P(c>0) .67. Accuracy by season 0.00 +0.42 -2.02 +0.38 +0.75 -0.37.
- Arm B joint: decisive 14-16 of 30; accuracy -0.13 [-0.86,+0.60] P+ .34; log loss -0.00047 [-0.00134,+0.00033] P+ .13; Brier -0.00024 [-0.00067,+0.00016] P+ .12. In-sample +0.13 / +0.00007 / +0.00003, gap +0.27 / +0.00054 / +0.00027. Fold coefficients +.05 -.04 +.14 +.08 +.10 +.12. The four-input refit reproduces the served column (correlation 1.0000), so A and B are near-duplicates, not two votes.
- Control C intercept only: decisive 7-5 of 12; accuracy +0.13 [-0.33,+0.60] P+ .72; log loss -0.00005 [-0.00020,+0.00012] P+ .30.
- Control D shuffled within season: log loss mean -0.00038, band [-0.00174,+0.00122]; share of shuffles at or above the real arm .52 accuracy, .55 log loss, .57 Brier. The real term scores like a shuffled one.
- 2026 Weeks 1-9 (n=136, coefficient frozen at +0.068): 15 nonzero games (home only 11, away only 4), 8 Thursday games all both-short; decisive 1-0 of 1 (TEN at NYG); accuracy +0.74 [0.00,+2.21] P+ .82; log loss -0.00181 [-0.00372,+0.00001] P+ .025; Brier -0.00090 [-0.00184,-0.00001] P+ .024. Raw shorter side 4-11, the declared direction and against the frozen sign.
Card implication (inferred): no Thursday pick would move. About 30 Sunday or Saturday games a season would shift by roughly 1.7 probability points and about 4 picks a season would change side, 11-13 so far. Not served. Research stays open: the coefficient interval spans zero on both sides, so there is no resolved wrong sign, and no positive control exists at this size.
Recorded 9 cells in registry/weak_signals/mod47_short_week_side_term.json with `record --batch` (6 arm, 2 control, 1 frozen 2026), all unresolved_below_power; `pool --league nfl --effect-units accuracy_points` exit 0. Looks: 6 graded contrasts, 2 controls, 2 in-sample gaps, 1 frozen read, plus 1 undeclared raw short-side count.
