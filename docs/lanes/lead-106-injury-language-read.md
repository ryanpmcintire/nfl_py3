# LEAD-106

## Goal
Injury-report language read: a frozen language-model prompt maps each listed player's designation, practice note and headlines to P(plays) and expected snap share (ROADMAP LEAD-106). Availability input only; one fitted probability selects every side, so nothing here changes a pick.

## State (2026-11-14 15:04 ET, forward Week 9 scored; all numbers measured this session)
Command: `scripts/lead106_forward_score.py --week 9` -> `artifacts/lead106_injury_language/forward_score/2026_week09.json` and `2026_week09_scored.parquet`. Same id-join scorer as Unit 2 (gsis -> pfr, verified rows only), expanding prior-week ladders with add-one smoothing, team-week bootstrap 2000 reps. Reads scored as frozen, no model calls. Positive = read better than the ladder.
- Timing: 360 Week 9 reads (28 Thursday, 332 Sunday/Monday), every one recorded at least 164 minutes before its kickoff, file modified time equals recorded time, enrolled prompt hash and model match. 0 excluded. 309 verified (Out 100%, Questionable 86%, none 82%, Doubtful 4 of 6), 30 team-week clusters, 15 games.

| metric, baseline | Weeks 2-8 (294 rows, 143 clusters) | Forward Week 9 (309, 30) | Pooled 2-9 (603, 173) |
|---|---|---|---|
| log loss, designation ladder | .065 [.014, .130] P+ .9985 | .083 [.019, .161] P+ .9975 | .067 [.022, .126] P+ .999 |
| log loss, designation x practice | .036 [-.003, .083] P+ .959 | .036 [-.002, .080] P+ .967 | .036 [.002, .077] P+ .980 |
| Brier, designation | .020 [.005, .041] P+ 1.0 | .021 [.004, .043] P+ .993 | .020 [.007, .036] P+ 1.0 |
| Brier, designation x practice | .012 [-.001, .027] P+ .964 | .012 [-.002, .028] P+ .963 | .012 [.001, .025] P+ .987 |
| snap-share MAE, designation | .035 [.014, .055] P+ .999 | .026 [.011, .040] P+ .9995 | .034 [.016, .053] P+ .9995 |
| snap-share MAE, designation x practice | .036 [.017, .056] P+ 1.0 | .025 [.011, .038] P+ .9995 | .035 [.018, .053] P+ .9995 |

- Forward by designation, log loss vs designation / vs practice ladder: Out 63 rows -.011 [-.013, -.008] P+ 0 / -.008 [-.010, -.005] P+ 0 (nobody Out played; read floor .014 vs ladder .004); Doubtful 4 rows in 2 clusters -.079 / -.050 (0 of 4 played, read .10); Questionable 62 rows -.036 [-.125, .052] P+ .24 / -.061 [-.120, -.003] P+ .019; none 180 rows +.161 [.057, .282] P+ .9995 / +.087 [.030, .148] P+ .9995.
- Pooled by designation: Questionable +.055 [-.006, .116] P+ .96 / -.001 [-.057, .057] P+ .50; none +.105 [.025, .206] P+ .999 / +.069 [.011, .138] P+ .994; Doubtful -.033 [-.055, -.008] P+ .006 / +.024 [-.008, .062] P+ .93; Out -.001 P+ .29 / -.007 P+ .36.
- Concentration: three games carry 93% of the summed forward gain vs the designation ladder (BUF@MIN 12.6, JAX@BAL 7.1, HOU@LAC 4.4 of 25.8 nats); the other 12 games sum to +1.7 (vs the practice ladder -2.2). The gain sits in no-tag players who did not practise (26 rows, 46% played; read .69, ladders .95 and .85). BUF@MIN had no game designation in the capture: Monday teams file after the Saturday capture. Without that game (284 rows, 28 clusters): .047 [.002, .104] P+ .981 vs designation, .016 [-.013, .050] P+ .84 vs practice.
- Reading: one forward week is a small sample, neither a confirmation nor a negative. Headline cells match Weeks 2-8 in sign and size; inside them the read's edge is in no-tag non-practising players, and in Week 9 it trailed the ladders on Questionable, Doubtful and Out rows. Implication (inferred): enter the read as a fitted term beside the ladder, not as a replacement.
- Looks: 66, family lead106 (forward 36 = 3 metrics x 2 ladders x 6 subsets; pooled 30). 18 recorded as unresolved_below_power in `registry/weak_signals/lead106.json`: 16 `lead106_forward_wk9_*` and `lead106_pooled_wk2_9_ll_{des,desprac}_all`. `weak-signals pool --league nfl --effect-units accuracy_points` loads (exit 0, no coherence problems).

## Tried
- Unit 1 (11-03): `scripts/lead106_injury_language.py` (freeze/run/score) and `scripts/lead106_prompt.md` (sha256 08170f6d...3d2c). Population 2459 player-weeks, weeks 1-8; sample 331 (all 228 headline rows plus up to 100 per designation). Name-join score .075 was not trusted (33.5% unmatched rows labelled not played).
- Unit 2: `ids` command, gsis -> pfr join, 87.4% verified, 294 rows: the Weeks 2-8 column above; halves even/odd .030/.108 and .032/.041; Questionable +.068 P+ .979 (vs practice +.009 P+ .62), none +.095, Doubtful -.026 [-.051, -.002].
- Unit 4: gsis-keyed outcome (`artifacts/per03/played_by_gsis.parquet`, 282 rows): .066 [.013, .133] P+ .997 and .036 [-.003, .085] P+ .955; these are registry cells `lead106_injury_language_p_plays*`. Not rerun for Week 9 (that parquet ends at Week 8 and belongs to PER-03).
- Unit 3: `forward --week N --view thursday|sunday`; scheduler jobs `lead106_forward_*` write `artifacts/lead106_injury_language/forward/` (on disk: Week 9 Thursday and Sunday, Week 10 Thursday).
- Unit 5 (11-14, this): new `scripts/lead106_forward_score.py`; imports the lane script unchanged; its outcome rule reproduces `id_outcomes` exactly on weeks 1-8 (parity true); recomputed Weeks 2-8 match the stored `score_ids.json` within .0002; ruff check and format clean. Practice status for forward rows comes from the capture-time injuries snapshot (designation equals the frozen row in 360 of 360). Week 10 Thursday (20 reads, WAS@NYG) not scored: `--week 10` exits on missing `data/players/forward_snaps/snap_counts_2026_before_week_11.parquet`, which the forward-log jobs write at their first Week 11 run; no capture was missed.

## Next
1. Score Week 10 once `snap_counts_2026_before_week_11.parquet` exists: `scripts/lead106_forward_score.py --week 10`, then record the same cell set (all x 3 metrics x 2 ladders, log loss by designation and status-filed games, pooled log loss) by `weak-signals record --batch`.
2. LEAD-108 can start now: Week 9 is the first full-population week (`forward_score/2026_week09_scored.parquet`, 309 verified of 360; raw reads `forward/2026_week09_*.json`), weeks 2-8 are the sampled 294 (`sample.parquet`, `raw/batch_*.json`). No `scripts/lead108*` yet; PER-10 kernel and PER-09 rating paths not checked here. Use the read as a fitted weight beside the ladder.
3. LEAD-109 can start its rerun now: `scripts/lead109_position_weights.py` on the Week 9 scored parquet (has position, read and ladder probabilities, outcome), which gives every QB and OL row without sampling; one week, so group coefficients stay below power.
4. LEAD-110 can start its rerun now for Week 9 only: `scripts/lead110_absorption.py` needs full team sums, and Week 9 is the first census (15 games; BUF@MIN was read before designations); weeks 2-8 sums stay partial.
5. Root (scheduler-owned): add a Monday view to the forward capture so Monday teams are read after their designations are filed.

## Open
- Sub-cells with the whole interval below zero, not closed here, for the orchestrator to adjudicate (one week, 66 looks): forward Out (both ladders), forward Doubtful (4 rows), forward Questionable vs the practice ladder, pooled Doubtful vs the designation ladder (31 rows; +.024 P+ .93 vs the practice ladder).
- Forward rows carry no headline flag, so there is no headline subset; no split-half on one forward week (reliability left empty).
- 51 of 360 forward rows are unverified (no pfr id in the roster crosswalk); the Unit 4 gsis-keyed outcome would recover some.
- Weeks 2-8 reads saw final designations; the forward Saturday capture precedes Monday teams' designations, so forward "none" is a wider population for those teams.
- The registry batch builder was a scratch file; rebuild it with the week as a parameter.
- Rules: zero crossing closes nothing; one fitted probability selects the side; parameters leave-one-season-out; every cell a look; opener grade with opener input once this reaches a game-level term.
