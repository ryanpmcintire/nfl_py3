# Confidence and Best Pick unification

## Goal

Assess whether the single served probability ranks games within a week and calibrates the top pick. The owner requires an overfit assessment, actual within-week ordering, and top-pick calibration; merely making Best Pick equal the highest displayed probability does not complete this work.

## State

- Measured final verification: 4,529 tests passed, nine skipped; Ruff format/lint and mypy passed. Local `publish-board` passed; rendered diff reflects the existing Denver lock and new Findings summaries. Saved pre-kickoff inputs render the unlocked nominee's 0.4-point gap; the locked card omits it. Exact checks and review sweep: `docs/confidence_top_calibration.md` execution verification.
- Measured 2026-09-20: the declared three-season calibration repair is complete; protocol/results `docs/confidence_top_calibration.md`, reproducible script `scripts/confidence_top_calibration.py`, outputs `artifacts/confidence_top_calibration/20260920_fixed/`. The served probability is unchanged.
- Chronological nominee Brier improvement +0.001040 [-0.027412,+0.030943], probability_positive 0.5227; all-game change -0.002490 [-0.006082,+0.001022], P+ 0.0853. Nominee inverse temperatures 1.083538, 0.000001, 0.242607 vary by fold. All 28 cells are recorded unresolved; no closing ground demonstrated.
- Dashboard gap text now compares an unlocked Best Pick with its eligible runner-up only when deadlines and probabilities are complete. Locked nominees omit the comparison because the lock-time candidate distribution is unavailable.
- Probability/side/Best Pick integration is published in release `5fb88a2`. The live nominee can change at each refresh; its argmax status remains provisional pending reliability work. No ranking change follows automatically from this replay.
- The fixed matched replay and full plan/results are in `docs/confidence_best_pick_sunday_matched.md`; prediction-level outputs are `artifacts/confidence_best_pick_sunday_matched/20260920_fixed/`.
- Eight corrected full-pool unresolved inferential cells were recorded in `registry/weak_signals.json` under `confidence_best_pick_sunday_matched_full_pool_v2`. The eight earlier v1 cells were invalidated with `weak-signals invalidate` and linked to v2, so pooling excludes the outcome-conditioned nonpush diagnostic while preserving its history.

## Tried

- Predeclared one comparison before scoring: Sunday-inclusive frozen heldout probability argmax versus archived alpha=2000 chance of the **same current served side**, on the same regular-season games with kickoff after each week's Sunday 12:45 p.m. ET cutoff; grade both nominees at opener. No fitting, thresholds, side changes, or old eligibility screens.
- The chronological alpha archive supplies full-game prior-week scores, but no alpha LOSO scores. Its parameter was previously selected and the NFL years reused; this replay is diagnostic rather than untouched validation.
- Corrected chronological pool: all 1,537 opener games can enter selection; 930 eligible in 72 weeks, 141 started games excluded. Nominees differ in 58 weeks; 27 decisive W/P/L grades split 14 better for current, 13 for alpha (exact two-sided sign null p=1). Current W/L/P 40/32/0, alpha 38/31/3; wins per nominated week +2.78 points [-11.11,+16.67], `probability_positive=0.6504`. Pushes were not replaced or treated as losses; accuracy and probability loss use the 72/69 nonpush nominees respectively.
- Current nonpush top predicted 65.44%, realized 55.56%: actual-minus-predicted -9.88 points [-21.25,+1.47], `probability_positive=0.0443`. Current top versus rest -1.91 points [-13.64,+9.98], P+ 0.3805; within-week slope +2.95 points per 10 confidence points [-3.52,+9.36], P+ 0.8082. Current LOSO top is 63/43/1 W/L/P in 107 weeks; alpha has no matched LOSO evidence.
- Measured by root: `.\.tools\uv.exe run --no-sync python scripts/confidence_best_pick_sunday_matched.py` reproduced the corrected results; log `data/environment_recovery/matched_ranking_final_verified.txt`. It rewrites research outputs and does not alter the registry or live state. Ruff format/check passed.

## Next

- Continue the Sunday-readiness lane, preserving locked games. Calibration research next needs additional timestamped weekly nominees and a declared mechanism; monotone temperature fitting cannot establish better ranking. Do not repeat these 28 looks or promote a best cell.
- Implementation and research results are verified and ready for commit/push.
- Owner standing instruction (2026-09-20): commit and push at every verified clear stopping point, without asking again; persisted in `AGENTS.md`.

## Open

- Current top-pick calibration is concerning and ordering evidence is weak, but neither mechanism is closed by an interval crossing zero. No untouched outer period or matched alpha LOSO exists. Sunday's private odds cannot be shown as public Books-now prices. Backups are in `data/environment_recovery/before_probability_unification`.
