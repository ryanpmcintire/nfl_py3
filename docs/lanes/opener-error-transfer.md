# Opener error transfer (XLG-09)

Full unit-by-unit predeclarations, results, and drafted `weak-signals record`
commands live in `docs/opener_error_transfer.md`. This lane stays a
one-page pointer.

## Goal

Test whether an opener-error model learned on CFB data transfers to the NFL
opener (accuracy at the opener, line movement toward the pick), graded as
one added fitted term inside the served four-term pick probability — never
as a flip rule. Separate transfer value from feature value. Unit 5 (this
pass) additionally tests a pooled two-league joint fit with ridge-shrunk
league interactions, as an alternative to CFB-only training.

## State

Units 1-5 complete, run once each, all real out-of-sample LOSO fits. No cell
in this family supports adding a CFB-transfer or pooled-transfer term to
`src/`. Units 1-4 established the CFB-only-trained term stays
`unresolved_below_power` on every accuracy and line-move cell except one
2020-2025 line-move variant root closed `bounded_by_control` against the
positive-control MDE. Unit 5's pooled joint fit is a stronger negative: its
own LOSO coefficient is negative in all 13 folds (both arms), and 4 of its 6
cells resolve `refuted_mechanism/wrong_sign_resolved` (whole interval below
zero) — pooling with ridge shrinkage toward CFB coefficients does not help,
and on line movement it is resolved worse than even the simple CFB-only
term.

## Tried (one line per unit, measured number + P+; full detail in the doc)

- Unit 1 (`opener_error_transfer_v1`): added-term-vs-base -0.399 acc pts,
  P+=0.326, unresolved. CFB-direct-vs-base -6.92 pts, P+=0.0, refuted
  wrong-sign. NFL-only-twin-vs-base -7.385 pts, P+=0.0, refuted wrong-sign.
  NFL-only-vs-CFB-direct -0.466 pts, P+=0.459, unresolved. Move-vs-zero
  -0.0181 MAE-improvement, P+=0.0, refuted wrong-sign.
- Unit 2 (`_v2`, 13-season widened CFB pool): added-term-vs-base +0.133
  pts, P+=0.7599, unresolved. NFL-only-vs-CFB-direct +0.399 pts, P+=0.5819,
  unresolved.
- Unit 3 (`_v3`, point-in-time CFB rule, 2011-2025 extended pop, 2013-2025
  graded): added-term-vs-base all-graded +0.1237 pts, P+=0.6694, unresolved.
  2020-2025 subset +0.2661 pts, P+=0.7735, unresolved. Per-fold betas
  positive all 13 folds (0.019-0.063).
- Unit 4 (`_v4`, line-move yardstick, real 2013-2019 SBR opens): all-graded
  -0.000309 ats pts, P+=0.509, unresolved (near-exact null). 2020-2025
  subset -0.005988 ats pts, P+=0.2915, unresolved on paper — root later
  reclassified this subset cell `bounded_by_control` against the
  positive-control MDE (0.0654 ats pts), closing that variant only.
- Unit 5 (`_v5`, pooled two-league joint ridge fit, this pass):
  accuracy-all-graded +0.4329 pts, P+=0.7215, unresolved. **Accuracy
  2020-2025 subset -1.3972 pts, interval [-2.302,-0.513], P+=0.0005,
  refuted wrong-sign.** Accuracy-vs-CFB-only-term (context) +0.3092 pts,
  P+=0.6431, unresolved. **Line-move all-graded -0.0785 ats pts, interval
  [-0.1112,-0.0496], P+=0.0, refuted wrong-sign.** **Line-move 2020-2025
  subset -0.0692 ats pts, interval [-0.1187,-0.0250], P+=0.0, refuted
  wrong-sign.** **Line-move-vs-CFB-only-term (context) -0.0782 ats pts,
  interval [-0.1246,-0.0334], P+=0.0, refuted wrong-sign** — pooling is
  resolved worse than CFB-only training, not an improvement. Pooled-term
  LOSO beta negative in all 13 folds, both arms (-0.087 to -0.134).

## Next

- 2026-09-25 root: Unit 5 recorded (registry 7,255-7,260, family
  `opener_error_transfer_v5`). Line-move cells (vs base in both windows and vs
  the CFB-only term) recorded `refuted_mechanism` / `wrong_sign_resolved`.
  Accuracy cells are all `unresolved_below_power`; the 2020-2025 accuracy cell
  (-1.40 [-2.30,-0.51]) was recorded unresolved rather than closed because the
  enclosing 2013-2025 window reads +0.43 P+ 0.72. The pooling mechanism is closed
  on line movement. The CFB-only added term (Units 1-4) stays
  unresolved_below_power; reopen only with a new mechanism or more NFL seasons.

## Open

- Unit 5's two all-graded accuracy cells (vs base, vs CFB-only term) remain
  `unresolved_below_power` — the pooling mechanism is refuted on line-move
  (both cells) and on the 2020-2025 accuracy subset, but not yet on
  all-graded accuracy; do not claim full closure of the pooling idea, only
  of the cells that resolved.
- Only 6 NFL seasons for any 2020-2025-subset outer scoring; season-block
  bootstrap intervals stay wide until more NFL seasons accumulate — expected,
  not a rejection reason on its own (it is what let the wide-population
  cells resolve where the narrow ones could not).
- CFB 2020 stays excluded from training (upstream `cfbd_provider_sparse`
  gap, confirmed via manifest, not fetchable). Split-half reliability of the
  CFB-to-NFL transfer predictor class: 0.141 (13-season pool), reused for
  Unit 5's `--reliability` field since the pooled predictor's own
  split-half was not separately measured (no single fixed training set to
  split under the point-in-time design, same reasoning as Unit 3).
- Remaining levers, not run: (a) a positive-control MDE comparison sized for
  the pooled-fit mechanism specifically; (b) more NFL seasons or a CFB
  source before 2012 to thicken early folds; (c) why the pooled model's own
  coefficient is uniformly negative while the CFB-only term's is uniformly
  positive — worth a mechanism note if this family reopens.
