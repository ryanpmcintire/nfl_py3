# LEAD-89 retain-the-option policy

Question: does a Best Pick that waits only when later games are expected to offer a stronger pick beat locking on Tuesday noon, Thursday capture or Sunday 12:30.

Policy (declared before scoring in tests/scratch/codex/lead89_option_policy/protocol_before_run.md): at Tuesday and Thursday, lock the current best still-playable game, or retain the option when the expected best still-playable calibrated confidence at the next stage, learned from other seasons, is higher; Sunday locks. One joint weekly law: each training week's (Tuesday, Thursday, Sunday) best confidences are one resampling unit; conditional expectations are least-squares lines, Thursday to Sunday, then Tuesday to max(Thursday, expected Sunday). Leave-one-season-out. Only calibrated probabilities enter; no outcome or later quote enters a decision. The stage-fitted probability picks the side.

Law fit, all seasons: E[Sunday best | Thursday best] = 0.231 + 0.659 x; E[Thursday value | Tuesday best] = 0.338 + 0.523 x (slopes below one: regression of the weekly maximum toward the mean). Lock stage out of season: Sunday 81 weeks, Thursday 26, Tuesday 0.

Results, 107 common weeks, weekly Best Pick graded at the opener. Decisive-week record (policy vs fixed), then probability_positive and interval in accuracy points:
- vs Tuesday: 69-36-2 vs 62-42-3; +6.07 [-4.21, 16.82], probability_positive 0.869. Outer 2023-2025: 32-21-1 vs 29-25-0, +6.48, 0.807.
- vs Thursday: 69-36-2 vs 70-35-2; -0.93 [-9.35, 7.48], 0.410. Outer: 32-21-1 vs 32-20-2, -0.93, 0.444.
- vs Sunday: 69-36-2 vs 67-38-2; +1.87 [0.00, 4.67], 0.932. Outer: identical records, 0.00.
In-sample versus out-of-season policy-minus-fixed (points): Tuesday 5.14 vs 6.07, Thursday -1.87 vs -0.93, Sunday 0.93 vs 1.87; gaps about -0.93 (one week). Per-season cells are in tests/scratch/codex/lead89_option_policy/results.json (2020 is the weak season vs Thursday, -29.4 [-52.9, -11.8]; 2022 the strong vs Tuesday, +19.4).

Looks: 48 new (3 comparisons x 2 metrics x 8 populations), 96 with the stage-timed family. Classification unresolved_below_power for all; no refuted mechanism, no positive control. The policy is nearly Sunday-lock (names a different game from Sunday in 8 weeks of 107), so it adds almost nothing to the Sunday contrast. Limit: weeks where a retained option finds no later nominee are unscored.
