# LEAD-89 stage-timed Best Pick rebuild

Question: does waiting from Tuesday to Thursday to Sunday to lock the Best Pick improve it, using only information that existed at each stage.

Method (declared before the run in tests/scratch/codex/lead89_stage_timed/protocol_before_run.md): nested leave-one-season-out replay per stage on the active model; Tuesday and Thursday rebuild the 11 injury and quarterback columns as known at the stage cutoff (carry rule), Sunday uses production features for games passing the injury clock. Control arm at the production cutoff reproduces the active table exactly (4,902 rows, max abs diff 0.0 on all 11 columns). 48 looks, 107 weeks, 2020-2025.

Case counts (cells of 4,902 games): Tuesday carried prior-week absences 2,734, nothing known 191, carried known no absence 103, current-week report 46. Thursday carried 1,842, current-week report 979, nothing known 153, carried known no absence 70.

Arms (all games): Tuesday accuracy 55.16%, log loss 0.6865, Brier 0.2467; Thursday 56.32%, 0.6832, 0.2450; Sunday 57.33%, 0.6794, 0.2432. Model-only accuracy 52.2 to 52.6%.

Contrasts, weekly Best Pick, paired by week, graded at the opener. Decisive-week record (later W-L-P vs earlier), then probability_positive and interval, accuracy points:
- Sunday vs Tuesday: 67-38-2 vs 62-42-3; probability_positive 0.775; +4.21 [-6.54, 15.42]. Outer 2023-2025: 32-21-1 vs 29-25-0; 0.807; +6.48 [-8.33, 21.30]. Exact discordant p 0.736.
- Thursday vs Tuesday: 70-35-2 vs 62-42-3; probability_positive 0.912; +7.01 [-3.27, 17.29]. Outer: 32-20-2 vs 29-25-0; 0.864; +7.41 [-5.56, 20.37]. Exact p 0.362.
- Sunday vs Thursday: 67-38-2 vs 70-35-2; probability_positive 0.267; -2.80 [-11.68, 5.61]. Outer: 32-21-1 vs 32-20-2; 0.445; -0.93 [-12.96, 11.11]. Exact p 0.523.

Per-season cells are in tests/scratch/codex/lead89_stage_timed/results.json (weeks_by_season). Classification: unresolved_below_power for all three; no refuted mechanism and no positive control. Nothing here serves or flips a pick.

Caveat: the LOSO baseline uses the 7 protocol flags, while the served card now counts only flag_protection.
