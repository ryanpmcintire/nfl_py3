# LEAD-108

## Goal
Wording-read availability inside the injury scenario mixture: the LEAD-106 per-player play probability replaces the ordinal ladder as the mixture weight (ROADMAP LEAD-108).

## State
Unit 1 (2026-11-15, measured; `scripts/lead108_wording_mixture.py` -> `artifacts/lead108_wording_mixture/`): linear expected-lineup term (LEAD-110 spread points per Out x (1 - P_play), home minus away), one LOWO slope on logit(served), 116 games weeks 2-9. Read vs ladder log loss +.0003 [-.0018, .0024] P+ .62, decisive 2-2; both arms vs served negative (P+ .14, .08). 4 cells unresolved_below_power. Forward test of the read term from Week 11 frozen in `docs/lanes/lead108-forward-2026-11-16.md` (not touched here).

Unit 2 (2026-11-21 10:31 -0500, measured; `scripts/lead108_full_mixture.py`, ruff clean -> `artifacts/lead108_full_mixture/{report.json,games.parquet,players.parquet}`): full PER-10 lattice mixture, same values, weeks 2-10, 130 opener games, pushes dropped. Both arms centred on the ladder expected shift; centre bisected to reproduce the served card probability at the opener (max error 8e-9, 0 failures); each scenario moves the tilt centre, band conditioned on the opener; independent sits by exact convolution (PER-10 pairwise coupling dropped, disclosed). Week 10 built in memory from LEAD-106 forward reads (345 read, 298 verified, `snap_counts_2026_before_week_11`). No LLM calls. 28 declared looks (4 contrasts x log loss, accuracy x wk2_8, wk9_10, wk2_10 + 2 controls x 2).
Coverage per game (listed verified / with read / listed starters / starters with read): weeks 2-8 13.6-23.6 / 0.9-4.2 / 6.1-10.8 / 0.19-2.08 (value mass read 8-40%); weeks 9-10 20.6-21.3 all read, 9.3-9.8 starters all read. Starter = majority of unit snaps in latest prior 2026 game.
Results (log loss gain, + = first arm better): read vs ladder, no fitted parameter: decisive 1-1 (2 side changes); wk2_10 +.0007 [-.0011, .0030] P+ .72, detectable .0029, matched-null pct .51; wk9_10 (full reads, 29 games) +.0048 [-.0016, .0150] P+ .87; wk2_8 -.0005 P+ .20. Accuracy wk2_10 0.0 [-2.3, 2.3]. Ladder curvature-only vs served +.00003 P+ .80, sd .00017, 0 side changes (PER-10 null replicates). Fitted slope arms (LOWO, LOSO impossible): ladder slope 165-340 (week 9 -168) on a near-zero term, read 4.6-13.1 (week 9 -13.0); read fit vs ladder fit -.0137 [-.0593, .0184] P+ .29, decisive 10-8; both fitted arms worse than served OOS (read -.031, ladder -.017; in-sample +.002/.003). Controls vs served: intercept -.0093 P+ .32, line move -.0174 P+ .21. Served log loss .6864, read raw .6857.
Reading (inferred): unfitted read mixture is a free, tiny probability shift (sd .006) that leans positive where reads cover every listed player; fitting a slope on 130 games overfits.
Registry: 5 cells unresolved_below_power, family lead108_wording_mixture (`lead108_fullmix_read_vs_ladder_{ll,acc}_wk2_10`, `lead108_fullmix_read_vs_ladder_ll_wk9_10`, `lead108_fullmix_read_fit_vs_ladder_fit_ll_wk2_10`, `lead108_fullmix_ladder_curvature_vs_served_ll_wk2_10`); pool --league nfl --effect-units accuracy_points loaded; shard modified.

ROADMAP LEAD-108 update: Measured 2026-11-21 (unit 2, `scripts/lead108_full_mixture.py`): full PER-10 lattice mixture, read vs ladder weights, 130 opener games weeks 2-10, no fitted parameter: log loss +.0007 [-.0011, .0030] P+ .72, decisive 1-1, detectable .0029; full-read weeks 9-10 +.0048 P+ .87; fitted-slope arms overfit (worse than served OOS). 5 cells unresolved_below_power; nothing served. Next: score the unfitted read mixture forward on Weeks 11-18 beside the frozen linear term.

Unit 3 (2026-11-21 10:51 -0500, read + measured): the unfitted read mixture is declared as forward arm "declaration 2" in `docs/lanes/lead108-forward-2026-11-16.md` (Weeks 12-18, 109 scheduled games, graded at the pool line against the ladder arm and the played card, decisive record first, 8 looks, scoring after Week 18 only). Logger `scripts/lead108_mixture_forward.py log` calls unit 2's `mixtures()` unchanged; smoke files for Weeks 11 and 10 written under `artifacts/lead108_mixture_forward/` (never scored). Nothing recorded (no result yet); nothing served.

ROADMAP LEAD-108 update: 2026-11-21: unfitted read-weighted lattice mixture declared as a forward arm for Weeks 12-18 before the Week 12 openers (`docs/lanes/lead108-forward-2026-11-16.md`, declaration 2); per-game read-arm, ladder-arm and card probabilities logged before kickoff by `scripts/lead108_mixture_forward.py log` (auto week, SKIP outside 12-18); scheduler jobs proposed (Thu 18:30, Sat 19:00, Sun 12:20), not yet added; detectable log loss about .0066 at about 105 games; scored after Week 18.

## Tried
Novelty check: nearest PER-10 (ordinal mixture, 2020-2024, unresolved), LEAD-106, LEAD-110, LEAD-109. Unit 1 linear term; unit 2 full lattice mixture with tilt-centre shift.

## Next
1. Root: add the three `lead108_mixture_forward_*` scheduler jobs (diff in the forward lane) before Thanksgiving Week 12 (first Week 12 game Wed 2026-11-25 20:00 ET). Before Week 18: add the `score` subcommand per declaration 2; after Week 18: score and record `lead108_fullmix_read_vs_ladder_{ll,acc}_fwd_wk12_18`.
2. Optional: restore PER-10 pairwise unit coupling in the convolution (same arms) to test the independence deviation.

## Open
Rules: zero crossing closes nothing; one fitted probability selects the side; weights chosen out of sample (leave-one-week-out here, one season of reads); every cell counted as a look; unresolved results go through weak-signals record; graded at the opener with intercept-only and line-move controls.
