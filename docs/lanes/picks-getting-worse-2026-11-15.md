# picks-getting-worse-2026-11-15

## Goal
Owner (2026-11-15): the card (what is played) gets worse every week; name the cause and fix what the evidence supports.

## State (measured, 2026 W1-9, 135 non-push games at the pool line)
- Card by week: 9-7, 14-2, 6-10, 9-7, 8-6, 6-8, 7-7, 6-8, 5-10 = 70-65.
- Served structure history (artifacts/pick_probability, 124 fits since 09-08, 4 structures): 09-14 to 10-25 seven flags (arrests, bye, coach, cold_visitor, division, protection, tank_zone); 10-25 protection only; 11-02 + pool hook; 11-03 division + protection.
- Current formula (artifacts/pick_probability/20261115T171352Z, true served flags from signed_composition_flags_fail_open, summed over the artifact's counted columns as pick_probability.py:842-849 does) replayed W1-9: 90-45, ll .6668; without flags 84-51, ll .6792; 14 differing sides, flags 10-4. Script tests/scratch/fixpicks/w11_flags.py.
- Flag records 2026: protection nonzero 33, 24-9; division nonzero 4, 3-1.
- Card minus current replay: W1-7 59-47 v 74-32 (seven in-sample flags, no move input W5); W8-9 11-18 v 16-13 (no hook before W9, single-book hook anchor W9, conversion; fixed 11-10).
- CORRECTION: the U5 claim (flag-free 83-53 beats flagged) used flags inverted from served probability and is wrong; the two remaining flags help on 2026.
- Card against the line move, by week: 1-0, 2-1, 0-3, 0-1, 1-3, 1-2, 1-2, none, 0-1 (6-13, mostly the seven-flag era). Week 10 card goes with the move on 10 of 11 moved games.
- Caveat: the current formula was chosen after seeing W1-9 losses, so 90-45 is optimistic; Week 10 is the first forward week on it. Owner reports Week 10 tracking below .500. Latest schedules snapshot is 20261111T010519Z, no W10 scores yet.

- Week 10 graded 2026-11-15 ~21:50 ET (snapshot data/raw/20261115T214249Z, feed artifacts/waterfall_feed/20261115T171430Z, scripts tests/scratch/fixpicks/w10_flipcheck.py): card 3-6 with 5 open (LA_ARI, SEA_LV, SF_DAL, PIT_CIN, LAC_BAL). Base model side alone 4-5. Only settled game flipped by fitted terms: WAS_NYG (base WAS .476, served NYG .506), lost. Five of six losses had model and card agreeing; four were underdogs blown out (NYJ+7.5, TEN+10.5, ATL+3.5, NE+2.5).
- Underdog picks by week: W1-7 31-20, W8 3-5, W9 1-7, W10 3-4 so far (W8-10 7-16). Favourite picks season 35-33; big-line (|line|>=6.5) favourite picks 6-16.
- History 2015-25, close-graded (schedules spread_line): favourites cover 46.7% W1-7 (n 1148), 49.2% W8-12 (754), 50.4% W13-18 (916). Candidate mechanism: the early-season underdog value fades as lines sharpen, while the model is fit on all weeks pooled.
- Phase x underdog term (2026-11-15, measured, tests/scratch/fixpicks/u9_main.py, u9_replay.py, outputs .out/.json beside them; family season_phase_x_underdog_pick_probability, 2 looks): adding dog + dog*week (continuous) to the served stage 1, hook refit, LOSO 2020-25 opener grade 1503 rows: log loss gain -0.0035 (season-block [-0.0068,-0.0013], week-block [-0.0054,-0.0019]), P(+) 0.00, worse in all 6 seasons; acc 808-695 v 829-674; 131 sides changed, term 55-76; in-sample minus OOS 0.0037. Pool-line two-stage -0.0040 [-0.0073,-0.0012]. Fold dog_week +0.0071,+0.0098,+0.0064,+0.0092,+0.0097,-0.0075 (sign opposite the hypothesis 5/6); bootstrap [-0.013,+0.026], P(+) 0.71, unresolved. Look 2 (dog alone) -0.0023. 2020-25 underdogs cover the opener 50.3% W1-7, 50.4% W8-12, 51.4% W13+: no early-season dog edge in the served population. Model leans dog early (+0.053 logit), favourite late (-0.032).
- 2026 replay W1-10 (fold-free 2020-25 coefs, served flags): served 94-51 (W1-9 91-45, one game off the 90-45 above), with term 90-55; only 6 sides change (W1,1,2,3,7,9), term 1-5. Week 10 3-6 both. Recorded: weak-signals pick_prob_underdog_x_week_term, unresolved_below_power. The term does not explain the decline.
- Line size (2026-11-15, measured, tests/scratch/fixpicks/u10_linesize.py/.out; nflverse sign: positive home spread = home favoured). 2026 W1-10 settled 145: current formula 94-51 v card 73-72 (120 same side). Big-line (|pool|>=6.5) favourite picks: current 10-13, card 6-16; big-line dog picks current 14-7, card 11-11. Current formula 2026 reliability slope (pick covered on |line|, offset logit q) fav -0.155/pt, week-block [-0.380,+0.054], P(+) 0.066; dog -0.015 [-0.186,+0.218]. 2020-25 LOSO served, opener, 1491 non-pick'em: fav picks 386-320 slope -0.019 [-0.063,+0.023] P(+) 0.20 (season-block [-0.061,+0.022]); dog picks 436-349 slope -0.000 [-0.050,+0.047]; fav-minus-dog -0.019 [-0.084,+0.046]. Five looks (+1 descriptive 6.5 cut). Within interval, no term fitted, zero sides changed. Recorded family served_probability_line_size_reliability (2 cells, unresolved_below_power). Verdict: 6-16 is mostly the older structures; current formula is 10-13 there, an unresolved lean not visible out of season.

## Week 10 attribution (2026-11-16 15:50 -05:00, measured)
Script scripts/week_attribution.py (card = clv_ledger/published_picks latest per game, cross-checked against the pick-revision ledger, 0 mismatches; pool line = latest data/splash file; graded with clv.pick_correct on result minus pool home spread; served terms replayed through PickProbabilityModel.home_probability with the pick-probability fit active at each game's last pre-deadline feed). Outputs artifacts/week_attribution/2026_week10.{csv,md}. Schedules snapshot 20261116T204645Z (nfl-ats ingest today). W10 replay matches the served feed exactly (max error 0.0000; fits 20261112T221522Z, 20261114T184614Z, 20261115T171352Z).

| layer (cumulative) | W10 record | sides changed v previous layer |
|---|---|---|
| discrete base model | 6-7 | - |
| + intercept/slope calibration | 8-5 | 3: SF_DAL AWAY W, MIA_IND AWAY W, LAC_BAL AWAY open |
| + counted flags (division; protection none) | 8-5 | 2: PIT_CIN HOME W, MIN_GB HOME L |
| + line-move term | 7-6 | 3: SF_DAL HOME L, MIA_IND HOME L, MIN_GB AWAY W |
| + pool hook = served = card | 6-7 | 1: WAS_NYG HOME L |
| current formula (20261116T171555Z) | 6-7 | v base: WAS_NYG L, PIT_CIN W, LAC_BAL open |

Nonzero-term games (term-direction record, settled only): flags 3 (2-1, all division); move 10 (6-3); hook 9 (3-5). Open: LAC_BAL (card LAC +5.5, MNF). Reading: the fitted terms net -2 games against the calibrated base (flags 0, move -1, hook -1); one week, 9 changed sides, unresolved_below_power, no verdict recorded.
Week 9 check: card 5-10 reproduced (matches State). Base 7-8, flags +1 (JAX_BAL W), move 0-2 on its flips, hook 9-4 on nonzero games. W9 replay differs from the served feed by up to 0.043 because the served W9 hook used a single-book anchor and today's quote store has more books (State line on W8-9); DET_MIA and LV_SF sides differ replay v feed.
Tuesday rerun after MNF: `.\.tools\uv.exe run --no-sync nfl-ats ingest` then `.\.tools\uv.exe run --no-sync python scripts/week_attribution.py --season 2026 --week 10`.

## Next
0. Done (above). Season-phase and line-size both checked; neither is a learnable defect of the current formula on 2020-25.
1. Monday: refresh schedules, grade Week 10 card per game with the production grader (never hand-roll spread signs), split losses by term (model, move, hook, flags) as in w11_flags.py.
2. Keep the current structure for Week 11; any structural change needs an out-of-season test plus the forward weeks, not another in-season patch.

## Open
- Whether Week 10 losses trace to a term or to the base model; base model went 10-19 in W8-9 with no stale input (U6).

## Season attribution (2026-11-16 15:56 -05:00, measured)
Command: `.\.tools\uv.exe run --no-sync python scripts/week_attribution.py --season 2026 --weeks 1-10` (new --weeks option; single-week mode unchanged). Outputs artifacts/week_attribution/2026_season.{csv,md}. Probabilities P+ = Beta(w+1,l+1) posterior mass above 0.5 (uniform prior), not a verdict. Card = latest published pick per game, 0 card v ledger mismatches in all 150 games. Weeks 1-2 have no usable feed (feed lacks the discrete-base field), so card record only, no layers. Replay differs from the served feed in Weeks 3-9 (quote store has since changed; W9 single-book hook), so those weeks' layer rows are replay, the card column is real.

| week | base | calibrated | flags | move | hook | card (real) |
|---|---|---|---|---|---|---|
| 1-2 | - | - | - | - | - | 9-7, 14-2 |
| 3 | 10-6 | 7-9 | 5-11 | 5-11 | 5-11 | 6-10 |
| 4 | 8-8 | 10-6 | 10-6 | 8-8 | 8-8 | 9-7 |
| 5 | 7-8 | 8-7 | 11-4 | 11-4 | 11-4 | 8-7 |
| 6 | 7-7 | 9-5 | 6-8 | 7-7 | 7-7 | 6-8 |
| 7 | 11-3 | 12-2 | 10-4 | 9-5 | 9-5 | 7-7 |
| 8 | 3-11 | 3-11 | 5-9 | 8-6 | 8-6 | 6-8 |
| 9 | 7-8 | 7-8 | 8-7 | 6-9 | 7-8 | 5-10 |
| 10 | 6-7 | 8-5 | 8-5 | 7-6 | 6-7 | 6-7 (1 open) |
| W3-10 replay | 59-58 (P+ 0.54) | 64-53 (0.84) | 63-54 (0.80) | 61-56 (0.68) | 61-56 (0.68) | 76-73 over all 150 (0.60) |

Flips v previous layer, record of the new side (W3-10): calibration 14-9 of 23 (P+ 0.85); flags 15-16 of 31 (0.43); move 10-12 of 22 (0.34); hook 2-2 of 4 (0.50); card v hook replay 4-12 of 16 (0.02; this is replay-v-served drift, not a model layer).
Terms, nonzero games, pushed-side record: flags 50 games 26-24 (0.61); move 60 games 31-28 (0.65); hook 22 games (W9-10 only) 12-9 (0.74). Counted flags (overlap with the flags layer): coach 32 games 11-21 (0.04); protection 27 games 19-8 (0.98); bye 6 games 2-4; division 4 games 2-2; arrests, cold_visitor, tank_zone 0 nonzero games. Per-week cells are in the artifact.

Reading: of the model layers, the worst cumulative flips are the move term (10-12 on 22 settled flips) and flags (15-16 on 31); neither is below 0.34 P+, so both are unresolved_below_power. Base to calibration to flags to move all sit within 4 games of each other (59-58 to 64-53); the card sits at 76-73 only because Weeks 1-2 (9-7 and 14-2, 23-9 combined, no layers available) lifted it; Weeks 3-10 card alone is 53-64 versus replay hook 61-56 on the same games. The sharpest single cell is the coach flag, 11-21 on 32 games (P+ 0.04), one of eight counted flags and a look among about 50 cells; not recorded as a finding and not a cause (it is one term in a layer that nets 15-16). Nothing recorded in weak-signals.
