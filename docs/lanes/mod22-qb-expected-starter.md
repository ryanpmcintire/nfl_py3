# mod22-qb-expected-starter (ROADMAP MOD-22, unit 4)

## Goal
New construct, not a transform of diff_lineup_total: the team's expected starting QB
(the QB with most offense snaps in the team's previous game, strictly earlier kickoff)
listed Out or Doubtful on an injury report visible by the decision time (kickoff - 24h).
Mechanism: the frozen Tuesday line prices the usual QB; a report visible later that
removes him is information the Tuesday line could not price.

## Declaration (written before any outcome was looked at, 2026-09-29)
- Population: 2020-2025 REG games in `build_fit_population` with tue_open_home_spread; snap counts and injuries from data/players/raw/20260910T205112Z.
- Term `qb_out_diff` = away_qb_out - home_qb_out (positive favours home). qb_out is 1 when the expected starter has a report_status of Out or Doubtful with effective_observed_at <= decision_at. Runtime asserts: prior game kickoff < decision_at, every used report timestamp <= decision_at.
- One look: qb_out_diff added as the 5th fitted term to the four-term served base, LOSO by season, via evaluate_candidate (accuracy points, Brier, log loss, decisive-game record, IS-OOS gap, per-fold coefficients). Look count 1, family qb_expected_starter_v1.
- Also reported, not looks: number of games with a nonzero term, name-to-gsis match rate.
- No revision after results.

## State
Script scripts/mod22_unit4.py (see Tried/Open for run result).

## Tried
Ran `.tools/uv.exe run --no-sync python scripts/mod22_unit4.py` once (exit 0) -> artifacts/mod22_unit4/20260929T204114Z/summary.json.
1,503 games; 72 team-games with expected QB Out/Doubtful visible by decision (home 38, away 34); 70 games nonzero term.
Added term, LOSO: accuracy -0.665 pts, 95% CI [-1.410, -0.065], P+ ~0.01 (interval fully negative); Brier and log loss intervals fully negative (P+ 0.022, 0.021); decisive 16 games, full model won 3 (exact-null p 0.021); IS-OOS gap 0.0080 vs base 0.0020; per-fold coefficient +0.032, -0.086, -0.178, +0.033, -0.284, +0.054 (in-sample -0.074, wrong sign vs mechanism, sign flips 3 of 6 folds).
Read: fitted term on 70 sparse games adds noise, does not help; coefficient sign unstable so this is not a resolved mechanism reversal, only a term that does not earn its place. Not served.

## Next
Root: adjudicate classification. Candidate: unresolved_below_power (only 70 nonzero games, unstable fold signs, split-half not measured); wrong_sign_resolved needs a stable wrong-sign coefficient and does not hold.
Record commands (root runs; not run here):
- nfl-ats weak-signals record --name qb_expected_starter_out_pick_probability_term --family qb_expected_starter_v1 --league nfl --season-start 2020 --season-end 2025 --effect -0.6653359946773121 --effect-units accuracy_points --interval-low -1.410458562718959 --interval-high -0.06518798652901232 --probability-positive 0.01 --sample-games 1503 --sample-blocks 6 --classification unresolved_below_power --reliability 0.5 --category onfield --source artifacts/mod22_unit4/20260929T204114Z/summary.json --classification-evidence "70 nonzero games, 16 decisive; per-fold coefficient flips sign 3 of 6 folds" --description "expected starting QB Out/Doubtful visible at decision time, home-away diff, 5th fitted term LOSO, MOD-22 unit 4"
  (adjust probability-positive and reliability to the exact summary values before running; reliability is a placeholder, not measured)

## Open
Name-to-gsis match rate in summary.json; QB quality not weighted (binary flag).

**Recorded 2026-09-29 (root):** registry cell `qb_expected_starter_out_pick_probability_term` as refuted_mechanism / wrong_sign_resolved on the Brier interval (whole interval below zero, P+ 0.0215), scoped to this binary construct.

**Corrected 2026-09-29 (root, after verify lane):** the refuted closure was wrong. 18/72 flagged team-games used guessed report timestamps and 39 name matches missed Gardner Minshew II, so the report-visible construct is unverified. Record replaced as unresolved_below_power; remeasure after the timing/name repair.
