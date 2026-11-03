# PER-13 Week 17-18 resting hazard

## Goal
Pregame hazard that a seed-locked or eliminated team rests starters in Weeks 17-18, priced or not at the opener (ROADMAP PER-13).

## State
Unit 1 done 2026-11-03, family per13_locked_seed_rest, 10 looks (2 flags x 5 metrics), all recorded unresolved_below_power. Script scripts/per13_locked_seed_rest.py, cells in artifacts/per13_locked_seed_rest/cells.json. Status from artifacts/sim09/stakes_teamweek.csv (as-of simulation; locked = one seed with prob >= 0.99; elim = p_make < 0.01), Weeks 17-18, 2009-2025 (Week 16 has only 2 locked rows, not used). Decisive record first: 19 locked team-games (12 snap-matched, 2016+), none vs a locked opponent.
- Starter snap drop (top 11 by prior-week mean, pp): locked offense +34.1 [21.9, 45.8] P+ 1.00; defense +34.2 [20.9, 46.4]. Elim +2.8 [0.5, 4.8] / +1.8 [-0.6, 4.2].
- Locked ATS at opener (Tuesday 2020-25, SBR else): -2.5 vs unlocked +0.76; diff -3.26 [-9.5, +3.3]; fade P+ 0.847. At close diff -0.33 [-6.2, 5.9].
- Line move open to close against locked team: 2.91 [1.26, 4.71], P+ 1.00. The market prices the rest by close, not at the opener.
- Elim: ATS diff -1.55 [-3.8, +0.7] at opener and close; no line move (+0.03).

## Tried
Intercept-only control = unflagged team mean ATS. No fitted model, no LOSO parameter yet.

## Next
Pregame lock probability feeding a fitted term (logit with opener as input, LOSO) is not buildable on 19 games; pool with Week 18 final-week elim and 2026 forward capture. Record forward locked games prospectively.

## Open
2026 heads-up (inferred, records only, no simulation): leaders BUF, MIN, SF 6-1; PIT, DEN, CHI 6-2 are seed-lock candidates; LAC 0-7, MIA 1-6, TEN, HOU, NO, IND, PHI, NYJ are elimination candidates.

## Unit 2 (2026-11-03) term design
Script scripts/per13_term_design.py; cells artifacts/per13_locked_seed_rest/term_design_cells.json; frozen term artifacts/forward_terms/per13_locked_rest/ (frozen_coefficients.json with created_at, procedure.md). 5 looks, 4 recorded unresolved_below_power (family per13_term_design).
- Label: move toward team (close minus open), Weeks 16-18 2009-25, 1212 team rows, 606 games; regressors own-minus-opponent lock probability (p_seed_modal if playoff seed) and (1-p_make), no intercept, LOSO.
- Lock coef -1.18 [-2.19, -0.22] P+ 0.008; elim coef -0.63 [-1.31, +0.03] P+ 0.032. LOSO R2 vs zero 0.0014 (in-sample 0.0116); 10 of 17 seasons negative.
- Decisive 19 locked games: opener ATS -2.5 [-8.2, +3.8], LOSO predicted move -0.51; realized minus predicted -1.99 [-7.7, +4.3] P+ 0.26. Cover slope on predicted move -1.45 [-5.6, +2.9] P+ 0.25 (expected +1 if opener unpriced).
- Pregame unit: P(rest) x starter value is not separately measured; value is absorbed into the coefficient (34 pp snap drop is the unit 1 measure).
- 2026 as-of-now lock probs (inferred, sim from rating model, 600x300): artifacts/per13_locked_seed_rest/lock_probabilities_2026.csv; P(locked by Wk17 Tue) SF .07, BUF .03, MIN .03; Wk18 SF .18, NYG .17, MIN .13. Elim by Wk17: TEN 1.0, LAC .995, MIA .99, NYJ .94. Refresh weekly: rerun the script.
## Next (unit 3)
Forward-log challenger Wk17/18 2026 per procedure.md; the slope sign vs +1 is unresolved. Serving needs pinned pick_probability edit.
