# Over/under classifier (predeclaration, queued — not run)

**Status:** predeclared 2026-09-03 as POL-12 program session 1. Sections
0–8 are frozen. NOTHING below has been run: no family assigned, no window
spent, no fit scored. A scoring session cites this document unchanged and
appends section 9.

**Parent program:** `docs/totals_program.md` (feasibility: full-game total
data present, 4,742/4,902 rows).

---

## 0. Binding closing-grounds taxonomy (AGENTS.md, pasted verbatim)

An interval or CI that contains zero is NEVER grounds to reject, fail, or close
an experiment. At this evaluator's ~2-point resolution, "contains zero" is the
EXPECTED outcome for a real small signal. Only two grounds ever close a line of
work: (1) refuted mechanism — a RESOLVED wrong sign (whole interval on the
wrong side of zero) or zero split-half reliability; (2) bounded by a positive
control proven able to detect an effect that size. Everything else is
`unresolved_below_power`: record it with `nfl-ats weak-signals record`, report
`probability_positive`, never the binary "contains zero". The registry code
hard-rejects inadmissible closures; if a record command errors, the verdict is
wrong, not the validator.

---

## 1. Target and population (frozen)

Binary over/under: `over` iff `(home_score + away_score) > total_line`
(pushes dropped, counted). Population: schedule rows with non-null scores,
`total_line`, and over/under odds, joined to the canonical feature table on
`game_id`; REG primary, playoffs separate (FND-15 lineage).

## 2. Arms (frozen)

- **Baseline:** the market over/under implied by over/under prices (no-vig
  normalization mirroring the spread path).
- **Candidate:** production pipeline verbatim (median impute + indicators →
  StandardScaler → Ridge(10)) on the totals feature allowlist
  (`docs/totals_model.md` §Frozen contract — same 40 columns, no additions),
  trained to predict `total_residual`, converted to an over/under pick by
  sign of the predicted residual. N0-style tuning: none; alpha and blend
  fixed by the parent regime.

## 3. Protocol and grade (frozen)

Expanding-window walk-forward, train strictly before the target week,
min-500 floor. Paired non-push games, probability-rule over/under picks.
Grade: OPENER where paired opener totals exist (archive-dependent; close
secondary). Metric: paired accuracy-points delta with week-blocked 95%
interval and `probability_positive`; prediction-level pairs retained.

## 4. Controls (frozen)

Frozen-pick null (200 within-week permutations); realized-total positive
control (leak actuals into the residual — must clear decisively or the run
is void).

## 5. Decision rule (frozen)

Record through both registries under `totals_over_under_on_production`:
`unresolved_below_power` / `unresolved` unless an admissible ground fires.
No tiebreaker change under any outcome; a positive lean proposes a
prospective challenger at most.

## 6. What this screen may therefore claim

At most: whether the production totals machinery picks over/under better
than market prices on one assigned window. Nothing about team totals,
halves, or calibration (program sessions 2–4 own those).

## 7. Family declaration (run 2026-09-03 — declaration only, no window)

`totals_over_under_on_production` declared open; assign/record NOT run
(pools exhausted). The declaration is the queued vehicle, not a look.

## 8. Results (not run — scoring queued on fresh blocks or 2026 prospective)
