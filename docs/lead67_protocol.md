# LEAD-67 protocol

Read from ROADMAP.md row LEAD-67 before reading outcomes; line wrapping only.

| LEAD-67 | ⬜ | Price the late move in key-number probability, not raw points (rank 2 of 8) | **Added 2026-09-29 (unmeasured).** Mechanism: the served four-term probability enters the
leader-median late move as one linear coefficient in points (MKT-18/19), but half a point across 3 or 7 moves cover probability by roughly 4 to 7 points while half a point between 4.5 and 5
moves it by about 1 to 2 (row 693 measured push rate 10.1% at line 3), so the linear term mixes informative and uninformative moves and under-weights crossings. Replace `c*move` with `c*dP`,
where dP is the change in the discrete-lattice cover probability of the frozen-line side implied by moving the line from the frozen opener to the pre-deadline leader-median line; an additive
arm keeps both. Predeclared: 799 opener-graded games 2023-2025 (the move-available population), LOSO by season, the served four-term base as comparator, 2 looks, metrics log loss, Brier,
accuracy and line-move-toward-pick with `probability_positive`, coefficients per fold with stability, decisive-game record first. The move remains a fitted term in one probability; no flip
rule. Checked, not duplicated: MKT-16/17/18/19 (raw points), MOD-20 unit 5 (flag-sum and availability interactions), `sunday-market-probability` lane (which cutoff, not the move's units),
LEAD-66 (grades the PMF; this row changes a served-model input). About 15 tool calls. |

Implementation declaration, saved before this study computes outcomes:

- Freeze the active four-term input artifact `pick_probability/20260929T192747Z`.
  Use its 799 move-available 2023-2025 rows, including its Sunday move input.
  Fit all arms on that same population, holding out each entire season.
- Target is home cover at the frozen Tuesday opener; pushes are already excluded.
  Home is the canonical orientation: reversing to the frozen side and back leaves
  the signed probability-valued covariate unchanged.
- Use the same three leader books as production. Their last dated quote before
  min(kickoff, Sunday 12:45 Eastern) defines the median late line. No later quote
  or result enters a feature. Quotes must be observed within the game's week and
  bookmaker updates must precede observation. Never substitute a closing line.
- Price both the frozen opener and late line on one discrete PMF conditioned on
  the late line with its mean at that line. `dP = P(home covers frozen opener |
  no push at opener) - P(home covers late line | no push at late line)`. This is
  positive for a favorable home move and directly prices crossed margin atoms.
  Keep the production band widths, expansion rule, and exponential tilting.
- Lattice priors use completed earlier seasons in the production trailing window,
  excluding the outer held season from every training and scoring row. All
  logistic means, scales, and coefficients fit only the other two seasons.
  Frozen model probabilities retain their original walk-forward construction.
- Base terms: intercept, model logit, composition flag sum, raw move, availability.
  Replacement swaps raw move for dP; additive keeps both. Availability is constant
  on this population. Use production ridge 0.001, without parameter search.
- Two experimental looks: replacement and additive; mandatory all-row in-sample
  diagnostics and three LOSO folds are estimates of those same fixed arms.
  No subgroup search, threshold tuning, or best-arm selection.
- Report decisive disagreement wins-losses first, then log loss, Brier, accuracy,
  close-minus-opener move toward pick, their in-sample/LOSO gaps, season stability,
  natural coefficients, and fixed-width home-probability reliability bins.
  Include the frozen model and a neutral 0.5 market benchmark on identical rows.
- Primary uncertainty: paired season-block bootstrap, 10,000 draws, seed 6701;
  secondary season-week block sensitivity uses the same draw count and seed.
  Report percentile 95% intervals and `probability_positive` with half credit
  for exactly zero draws. Only three season blocks limit resolution.
- Save prediction-level output only in the authorized local Markdown file;
  registry commands are prepared in the lane for serial execution by the owner.
