# MOD-10 graph model: player-state reliability

## Goal
Row MOD-10: matchup graph only after player state is reliable (no numeric gate; PER-09 .436 used). Served-probability test is one fitted term on the fair opener grade.

## State
- Unit 1-3 (artifacts/mod10_graph_reliability/*.json): RAPM odd v even SB .27-.44; the later .55-.59 was team-season quality; net of it player state ~0-.1 (S2 P+ .87-.92). Team-level gate not cleared; no closing ground.
- Unit 4 (mod10_ftn_traits/result.json, family mod10_ftn_player_traits): FTN has no player ids (joined to pbp). Reliable: contested catch .60, created reception .45, QB catchable .68; drop rate null.
- Units 5-7 (family mod10_graph_term, scripts/mod10_graph_term*.py, mod10_graph_edge.py): receiver graph term 2023-25 acc -0.50 P+ .21; QB-receiver edge term acc +0.88 [-1.01,2.75] P+ .80, LL P+ .31. Not served.
- History check (unit 8): man/zone coverage matchup already ran (mod23 unit5, family mod23_unit5_coverage_matchup, participation data 2018+, acc +0.47 P+ .77, six cells); FTN charting has NO coverage shell column. Defender-box x personnel ran (sim07_ftn_*, simulator labels, not the served grade). Blitz x QB matchup on the served grade was untested.

## Tried (unit 8, measured: artifacts/mod10_graph_term/report_blitz.json, scripts/mod10_graph_blitz.py, batch weak_signals_batch_blitz.json; family mod10_graph_blitz, 6 cells unresolved_below_power)
Declared before outcomes: defense blitz tendency (n_blitzers>=1 share of dropbacks, prior-weeks, shrunk, k from SB) x QB trait deviation, home minus away, 2022 warm-up, expanding past-only served harness (not LOSO), scored 2023-25.
Reliability 2022-25 odd v even SB: defense blitz rate .855 [.78,.91] (128 team-seasons); QB catchable .686 [.58,.77]; QB EPA gap blitz-minus-no-blitz -0.02 [-0.58,0.35] (127 QB-seasons) so the vs-blitz skill is not a reliable trait; the first spec (that gap) had a zero-signal input and was replaced before any outcome look by QB catchable rate. k_def 103, k_qb 217.
Outcome: base 416-383, arm 423-376, 799 decisive: acc +0.88 [-2.74,4.49] P+ .67; LL gain +0.0003 [-0.0073,0.0076] P+ .53; Brier gain +0.0003 P+ .57. Mean LL base .6973 arm .6970 (market 0.6931). Per season acc +4.5/-1.9/0.0. Fold coef 2023 0 (warm-up zeros), 2024 .43, 2025 .48; coef bootstrap mean .65 [-0.21,1.51] P+ .91. 163 flips of 799. Looks: 3 outcome + 3 reliability = 6 (plus 1 replaced spec, reliability only).
Reading (inferred): positive-leaning but flat in probability scores; the offense side of a coverage/blitz matchup is not a reliable trait, the defense side is.

## Next
Not served. Candidate (not run): defense blitz/box tendency x receiver created/contested skill; or a lineup-level target. Coverage shell needs another source (participation, already tried).

## Open
Row has no numeric gate. Unit 8 deviates from LOSO (expanding past-only served harness). Reliability P+ values in the batch are 1.0/0.5/0.0 by interval side, not bootstrap P+.

## 2026 forward read (2026-11-03, measured; scripts/forward2026_ftn_terms.py, artifacts/forward_2026_ftn_terms/report.json)
Frozen blitz x QB catchable term (coef .299 scaled, fit seasons <2026; frozen_blitz.json created 17:05:12Z before scoring), reliability k_def 102.6, k_qb 217.4 from <=2025. 2026 values via team_values on prior plays, QB ids from the locked recommendations. Base and mapping as in the unit 12 lane (locked line sweep shifted by term delta, Tuesday opener).
Result n 102: served 60-42, arm 58-44. Accuracy -1.96 pts [-9.80, +5.88] P+ .26; log loss gain +0.0020 [-0.0137, +0.0174] P+ .60; Brier gain +0.0009 [-0.0068, +0.0085] P+ .59. Decisive (16 flips) arm 7-9 v served 9-7. Mean abs delta 0.50 pts. Recorded 3 cells *_forward2026_w01_07 in family mod10_graph_blitz, unresolved_below_power; not served.

## Unit 9 coverage matchups (2026-11-03, measured; scripts/mod10_graph_coverage.py, artifacts/mod10_graph_coverage/report.json, family mod10_graph_coverage, 3 cells unresolved_below_power)
Premise check: FTN charting 2022-2026 has no coverage column (read 2024 parquet columns; n_blitzers, n_pass_rushers, n_defense_box only), and 2026 participation is absent (acc-u4 lane). Coverage labels come from participation (97% of 2022-25 dropbacks man/zone labelled, 88% cover-type). Declared before outcomes: 3 reliability looks (odd v even week, SB, team-season, >=30 plays per half), no outcome looks.
Result: defence man share SB .94 [.92,.96] (128 team-seasons); defence Cover-3 share .92 [.88,.94]; offence EPA/dropback vs man minus vs zone SB -0.17 [-0.69, 0.17] (128). Same pattern as blitz: defence side reliable, offence-by-coverage side not a trait.
LOSO and 2026 forward not run: the served-grade coverage matchup (man/zone and six cover types, 16-game window, shrunk) is already recorded in mod23_unit5_coverage_matchup (man/zone +0.47 P+ .77, cover type +0.20 P+ .60); re-running it would repeat a tested idea, and a 2026 W1-7 forward has no coverage input. No frozen file exists.
Next: SKY-02 tracking, or a coverage term whose offence side is a receiver/QB trait that clears reliability (none identified).
