# Why accuracy is stuck, and how to get past it (MOD-24 parent)

## Goal
Raise the model's own record against the opener (802-701, 53.36%, 2020-2025)
by fixing the grading instrument and adding information the opener lacks.

## State
2026-10-01 theory: (1) inputs are public team data the opener prices; the
team-quality ceiling is 0.013 pts; (2) 1,503 games give an SE of about
1.3 pts, so win-loss can't see gains of 0.5-1 pt; (3) noise of 13 pts vs an
edge of 1 pt; (4) about 7,700 tests reuse 2020-2025.
Correction: the extended population is not new. docs/proxy_opener_replication.md
(2026-08-19) graded the served model on SBR opens: 2011-2019 50.38%, against
53.36% for 2020-2025. The SBR open differs from the Tuesday line by 1.36 pts
on average, so it was kept out of the headline. XLG-09 already used a
2011-2025 population. New here: paired candidate-vs-base grading on it.
Owner approved all four units 2026-10-01. ROADMAP MOD-24.

## Results 2026-10-01 (measured, all unresolved_below_power, 58 cells)
- U1 extended 2011-2025 (3,734 games): base 51.19% proxy era, 53.36% true
  era, 52.06% pooled. Week-blocked intervals 25-50% narrower than with
  2020-2025 alone. Pooled accuracy diffs: unit 1 nested -0.24 (P+ .35),
  alpha-only -1.02 [-1.73,-0.29], compact net -0.27 (.37), man/zone 0.00 (.47).
- U2/U2b: base raw probabilities are overconfident (temperature about 0.34);
  recalibrated base log loss 0.6922 vs 0.6931 for a coin. Raw proper-score
  gains of units 1, 3, 4 are calibration. Survivors: compact net +0.0005
  (P+ .79), man/zone +0.0001 (P+ .90).
- U3/U3b: Next Gen Stats ingested (2016+, 24 leak-free inputs); all three
  arms are slightly worse than base on proper scores; |r| with margin vs the
  opener is 0.046 or less.
- U4: man/zone forward log running Thu 19:00 / Sun 11:00 through Week 18.
  The 2026 participation file is 404, so the 2026 coverage inputs are 2025
  tendencies.
Verdict: theory point 1 (information) is binding. Better grading confirmed
no hidden winner. Alpha-only tuning lost in both eras; it is left unresolved
because recalibrated scores disagree and it is one of about 50 looks.

## Next
Only information the opener cannot have at posting time can move the model
alone. Inventory candidates against registry families before proposing any
(check-history rule). Keep U4 running and score it after Week 18.

## Hazard
Editing any pinned source (src/nfl_ats: names in PINNED_NAMES, *features.py,
*_overlay.py, names containing margin or model, cli_commands/prediction.py)
stops the v2 validation capture. All MOD-24 code goes in scripts/.

## Open
Why does the 2011-2019 proxy grade sit 3 pts below 2020-2025: line noise,
era, or 2020-2025 selection reuse? U1 reports it per era.
