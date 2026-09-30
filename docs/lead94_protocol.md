# LEAD-94 execution details

The verbatim ROADMAP.md:877 and Protocol C/D declaration was saved first in
docs/lanes/lead94.md, before inventory or outcomes. This supplement fixes
implementation details before the first fit or score; it adds no variants.

- Population: every source-complete 2020–2025 REG same-book Tuesday/Sunday total
  pair with an archived opener. Median totals use the same paired bookmaker set.
  Weekly last-game membership comes from the full schedule, before source filtering.
- Target: final combined score; primary last-game absolute error, secondary
  all-game absolute error. ATS sides remain the chronological four-term base.
- Equation: Tuesday total + b times (Sunday 12:30 total minus Tuesday total).
  Fit b by weighted absolute loss through Y−3. Weights are density ratios for
  last-game membership, estimated from Tuesday total, absolute opener and
  categorical local weekday/time kickoff slot, using balanced logistic loss with
  ridge .001. Standardization and slot vocabulary use training games only.
  Undo balancing using the training class prior; weight = P(last|x)/P(last),
  capped at 10. Scores never enter the membership model. Report ESS and coefficients.
- Folds: outer 2023/2024/2025; train through Y−3, reserve Y−2, calibrate Y−1.
  Fixed ridge has no tuning grid or selection choice. The discrete total law uses
  calibration-season integer residual atoms, median-centered at the equation's
  centre. Standalone integer-total rounding is the existing round(median_total)
  rule in src/nfl_ats/tiebreaker.py:466; no new score-pair or field-rank claim.
- Total controls: Tuesday total, deadline total, training-median-intercept-adjusted
  deadline total and training unconditional median. Apply identical integer rounding.
  No sixth scored total arm. The unweighted response coefficient is reported only
  as a coefficient diagnostic to identify the effect of weighting.
- ATS diagnostics reuse LEAD-83's certified fixed-cutoff refits on the source-complete
  intersection, with their source hashes checked: four-term, model-only, market,
  Elo and an identical candidate. Preserve pushes for total fitting and scoring;
  exclude pushes only from conditional-cover diagnostics. No cached weekly model
  is treated as a chronological refit.
- Report each fold and pooled IS (training, optimistically using later calibration),
  OOS and OOS-minus-IS gap for every arm and contrast. Six endpoints: accuracy,
  Brier, log loss, RPS, all-game total MAE and last-game total MAE. Show decisive
  records first, coefficient stability and five equal-width ATS reliability bands.
  Use 10,000 paired season/week-block bootstrap draws, seed 20260994; half-credit
  exact bootstrap ties in probability_positive. Repeated IS games share blocks.
- Fixed declaration: F=3, B=7, K=6; (27*6+7+4)*(3+1)+25 = 717 looks.
  Zero crossing never closes a signal; unresolved_below_power unless an admissible
  closing ground is established. Registry commands are saved for serial execution.
