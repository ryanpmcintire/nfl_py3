# coach-flag-parity-2026-11-16

## Goal
Train/serve parity of `flag_coach`, which carries most of the pooled stack's 2020-25 gain (pooled-stack lane, Ablation). Compare the history flag table with the live path, explain the 46 games where live disagrees with mod27 g26, and refit only if history uses information that was not knowable before the deadline.

## Declaration (stamped 2026-11-16T23:13:33Z, before any interval was computed)
- Decision rule: refit the pooled stack only if the as-of coach record differs from the history value in any 2020-25 game.
- Record the 2026 Weeks 1-8 coach-flag reading at the pool line in two cells, family `coach_flag_parity_2026_11_16`, effect_units `accuracy_points` (100 x (win rate - 0.5) on decided nonzero games). Cell A uses the latest-snapshot values, which follow the history definition. Cell B uses the deadline as-of snapshot values, which the live feed served. The records were already seen in explain.json (PARTLY SEEN). Interval: exact Clopper-Pearson 95%. probability_positive: Beta(1+W, 1+L) P(rate > .5). Classification: `wrong_sign_resolved` only if the whole exact interval is below zero, otherwise `unresolved_below_power`. Looks: 2.

## State
- Code (read): history comes from pick_probability_fit.py:308, the latest schedules snapshot, through :343 and signed_composition_flags (pick_probability.py:618-627). Live comes from card_view through the same function, using the schedules snapshot at refresh. Both call coach_fade_overlay.py:74-124 `year_one_by_game`. Its inputs are the game's `home_coach`/`away_coach` and the prior season's modal REG coach (:40-71). The flag is +1 when only the away coach is in year one and -1 when only the home coach is; it applies only in weeks 1-8 (COACH_WEEK_MAX = OVERLAY_WEEK_MAX = 8, coach_fade_overlay.py:31).
- 46 mismatches (measured, artifacts/coach_flag_parity/audit.json): g26 (artifacts/mod27_replay_inputs/g26.parquet, Weeks 2-8, 104 games) holds 0 for every composition flag (all 9 flags have 0 nonzero values). The 46 are every live nonzero coach flag in Weeks 2-8: 25 are +1 and 21 are -1. The protection figure of 80/104 is also a comparison against zeros. g26 is not a flag table from either path.
- History (measured, explain.json): the stored fit flag equals the rebuild from the latest snapshot and the rebuild from the first on-disk snapshot (2026-08-12) on 1,537/1,537 games. 2020-25 coach columns did not change across the 2026 snapshots. No schedule snapshot from 2020-25 is on disk. 236 games are nonzero (136-96-4 at the opener). 7 of them are midseason interims (Crennel 2020, Morris 2020, Bisaccia 2021, Wilks 2022), and those went 3-4. Each interim was named before his first game (reported). The inputs are knowable before the deadline, so the decision rule does not trigger a refit.
- Live 2026 (measured): before the 11-11 snapshot, nflverse listed stale coaches for ATL (Morris, not Stefanski), ARI (Gannon, not M. LaFleur) and BUF (McDermott, not Brady), plus an LV spelling variant. The deadline as-of path therefore under-fired: 40 nonzero in W1-8, against 53 with the latest snapshot, and 23 games differ. The defect is in the serving feed, not in history.
- Pool line W1-8 (measured): latest 20-33, as-of 14-26. The coach flag is zero from Week 9 on, so the pooled stack's forward W11-18 log cannot test it.

- Recorded (measured; `coach_flag_parity.py record` exit 0, `weak-signals record --batch` exit 0, 2 cells; `pool --league nfl --effect-units accuracy_points` exit 0; files artifacts/coach_flag_parity/batch.json, record.log and pool.log). A latest snapshot: 20-33, -12.26 pts, exact 95% CI [-25.21, +2.11], P+ .038. B deadline as-of: 14-26, -15.00 pts, CI [-29.37, +1.68], P+ .030. Both are unresolved_below_power. The 2026 reading points the wrong way against history's 136-96-4, and nothing is closed.

## Tried
- audit, explain and record in scripts/coach_flag_parity.py (all exit 0). No refit: the declared rule did not trigger.

## Next (done for this lane; for the orchestrator)
- Serving fix: take the coach identity from a dated coach-hire record, not the nflverse future-game coach field. The as-of feed missed 13 W1-8 flags in 2026.
- Pooled stack: its coach weight is untestable in the forward W11-18 log, because the flag is zero after Week 8. The card does not count coach (counted flags are division and protection only), so the card offset does not double-count it.

## Open
- None.
