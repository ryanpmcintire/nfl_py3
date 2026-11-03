# MOD-23 unit 12 FTN team scheme (2026-11-03)

## Goal
Test FTN team-level scheme/process aggregates as base-model inputs on the fair opener test (served 802-701, 2020-25).

## State
History check (read): FTN used only in sim07_ftn_state (box/personnel), mod10_ftn_player_traits (player traits), mod24_ftn_turnover_luck (INT/drop luck), mod10_graph_blitz (QB vs blitz graph term). Team play-action, motion, RPO, screen, no-huddle, blitz, rushers, created/contested/catchable aggregates as base inputs: not run. Not in unit 11's 27 entries.
Declared before look: 15 aggregates (off: motion, no_huddle, play_action, rpo, screen, out_of_pocket, catchable, created, contested; def: blitz, rushers, play_action_faced, motion_faced, run_box, catchable_allowed), EWMA span 8 min 3 prior games (mod24 convention), home minus away, ridge alpha 10 (served), expanding out-of-season evaluator, 2022 warm-up, scored 2023-25. One fit, 15 columns, 3 metrics.
Script scripts/mod23_unit12_ftn_scheme.py run; output artifacts/mod23_unit12_ftn_scheme/report.json; pregame truncation check passed.
Result (measured, n 799): arm 413-386 v base 416-383, acc -0.38 pts [-3.67, +2.77] P+ .40; log loss gain -0.0105 [-0.0218, +0.00005] P+ .026; Brier gain -0.0046 [-0.0099, +0.0003] P+ .035. Flips 189, arm won 93 lost 96. Per season acc +1.13, -1.13, -1.12.
Split-half Spearman-Brown (128 team-seasons): motion .91, no_huddle .97, oop .89, blitz .85, rpo .85, play_action .78, rushers .67, catchable .65, screen .69, run_box .55, catchable_allowed .54, contested .44, def_pa_faced .43, motion_faced .08, created -.12. None dropped.
Fold coefficients (scaled units) unstable in sign for most terms (e.g. blitz -.41, -.53, +.21; play_action -.37, -.10, +.15); stable sign: off_created negative, def_motion_faced/def_run_box positive.
Recorded 3 cells, family mod23_unit12_ftn_scheme, unresolved_below_power. Nothing served.

## Tried
The full 15-feature ridge term only. No subsets (would add looks).

## Next
Subset arms (offence-only, defence-only) are new looks; 2026 W1-7 forward score would need a freeze first.

## Open
Interval leans wrong on log loss (P+ .03) but not a refutation; 15 terms on ~3 seasons of nonzero history is thin, ridge shared with served alpha.
- Root review 2026-11-03: EWMA span 8 and the 3-prior-game minimum are not derived in this unit (AGENTS no-magic-numbers); before any subset arm, cite span 8 to the served builder or choose it out of season, derive the minimum, and re-run unit 12.
- v2 2026-11-03 (supersedes the run above; its span 8/min 3 were underived): span 8 is the served value (read: features.py:346,648, cli_common.py:195, game_features.manifest.json:178, purged_cv.py:16), imported as TEAM_STATE_SPAN. Min-3 replaced by shrinkage to the prior league mean, weight k/(k+n0), n0 = within/between variance from 2022 team-games only (measured; off_created n0 1315, def_motion_faced inf, others 1.4-52). Output artifacts/mod23_unit12_ftn_scheme_v2/report.json; cells re-recorded with --replace.
- v2 result (measured, n 799): arm 428-371 v 416-383, acc +1.50 [-2.41, +5.36] P+ .76; log loss gain -0.0094 [-0.0243, +0.0047] P+ .10; Brier -0.0038 [-0.0106, +0.0027] P+ .135; flips 240 won 126 lost 114; per season +4.89, -1.13, +0.75. Fold signs still unstable for most terms; stable: off_created neg, def_run_box and def_catchable_allowed pos. Looks 1 fit/15 columns/3 metrics. Unresolved_below_power; nothing served.

## 2026 forward read (2026-11-03, measured; scripts/forward2026_ftn_terms.py, artifacts/forward_2026_ftn_terms/report.json)
Frozen v2 ridge term (fit seasons <2026, 4,431 rows, alpha 10; frozen_unit12_v2.json created 17:05:07Z before scoring). Inputs from the same builders, prior plays only (truncation check passed, only 2 team-games on the Week 7 cutoff day). Served base = locked recommendations line sweep (market_residual method, excluding-push probability); term delta (points) added to the locked margin = served prob read at opener minus delta. Mapping is a line-shift of the discrete conditional read, an approximation. Graded at the Tuesday opener, game-level bootstrap.
Result n 98 (9 of 107 dropped: no push, opener exists, shifted line inside sweep): served 58-40, arm 51-47. Accuracy -7.14 pts [-17.35, +3.06] P+ .07; log loss gain -0.0121 [-0.0434, +0.0184] P+ .22; Brier gain -0.0062 [-0.0213, +0.0086] P+ .21. Decisive (27 flips) arm 10-17 v served 17-10. Mean abs delta 1.42 pts. Looks: 3 metrics, 1 term. Recorded 3 cells *_forward2026_w01_07, unresolved_below_power; not served. Leans wrong on 2026 but below power.
