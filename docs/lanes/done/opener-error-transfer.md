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

Closed 2026-09-26: Unit 5 recorded; the pooling mechanism is a resolved
wrong sign on line movement. The CFB-only term stays unresolved_below_power;
reopening needs a new mechanism or more NFL seasons (time/data-gated, not a
pending action).

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

## Unit 6 (2026-09-29, worker)

Nested-LOSO rerun of the pooled arm to test whether unit 5's negative beta was a stacking leak. Refuted: nested beta negative 13/13 folds; line move -0.088 [-0.127,-0.054] all-graded, -0.085 [-0.149,-0.031] 2020-2025 (P+ 0.0); accuracy +0.46 [-0.78,+1.78] P+ 0.75 and -1.00 [-2.38,+0.38] P+ 0.08. Declaration, results, and Record commands (not run) in `docs/xlg09_unit6.md`; script `scripts/xlg09_unit6.py`. Next: orchestrator records the four cells; pooling stays closed on line movement.

## Sign audit (2026-11-14, ENG-54 unit 3 worker)

Evidence: `tests/scratch/opener_transfer_signs/measure_signs.json`, `measure_signs2.json` (measured this session).
- College team-row lines are betting notation (home row negative when the home team is favoured): all 13 seasons corr(home margin, home_open) -0.62 to -0.84, home wins 69-86% when negative and 15-35% when positive, away row equals minus home row on 781 of 781 (2023). So `home_margin + home_open > 0` is the right home cover: eval:123, unit2:126, unit3:140 are correct (10 of 10 hand checks 2023 Bovada, 10 of 10 2015 5Dimes).
- NFL openers (tue_open_home_spread, SBR proxy) are positive when the home team is favoured (home wins 65.6% positive, 34.1% negative, corr +0.435; 2013-2019 66.4% and 36.6%); home_covered equals margin minus open above zero on 1503 of 1503 and 1731 of 1731.
- Inverted on NFL rows: every NFL loader set home_favorite = open < 0 (eval:136, unit2:139, unit3:181, unit4:91; units 5 and 6 import them). 0 of 10 sampled games flagged the true favourite; flag rate 0.404 against true 0.588. prior_move_diff was also opposite: the college move was close minus open in betting notation (positive is against the home team), NFL open_move is close minus open in repository notation (positive is toward the home team, corr 0.965 with spread_line minus the opener).
- Fix (scripts only): NFL home_favorite = open > 0 in the four loaders; college move = home_open - home_close in the three college loaders. ruff check and format pass.
- Rerun of eval, unit2, unit3, unit4, unit5, xlg09_unit6 launched single-threaded 2026-11-14 22:52Z (about 70 s each for the first two, both exit 0); logs in `tests/scratch/opener_transfer_signs/rerun_*.log`, exit codes in `rerun_status.txt` (six `exit=0` lines mean done). To finish from here: run `compare_cells.py` then `make_batches.py` in that folder with the locked uv python (the second prints old against new per registry cell and writes `batch_<family>.json` for changed cells only; a refuted cell stays refuted only if the corrected interval is wholly below zero, a bounded cell only if inside the 0.0654 ats control bound, nothing new is closed), then `nfl-ats weak-signals record --batch <file> --replace` per family and `weak-signals pool --league nfl --effect-units accuracy_points` once. DONE 2026-11-14: all six exit 0; 21 cells re-recorded with replace (`batch_<family>.json`, table `old_new_table.json`); `pool` loads (exit 0, 4640 signals). Measured, old to new, accuracy points unless noted:
- v1: added term -0.399 P+ 0.326 to -0.931 [-2.083, +0.215] P+ 0.062, unresolved. College-direct -6.92 to -5.26 [-8.68, -2.16]; NFL-only twin -7.39 to -5.39 [-8.39, -2.74]; move against zero -0.0181 to -0.0393 [-0.0561, -0.0239] MAE: all three still refuted, whole interval below zero. Twin against college-direct -0.47 to -0.13 P+ 0.405, unresolved.
- v2: added term +0.133 P+ 0.760 to -0.732 [-1.837, +0.210] P+ 0.081; twin against college-direct +0.399 to -1.663 [-3.413, +0.128] P+ 0.035; both unresolved.
- v3 (supersedes the Unit 3 Tried line): all graded +0.124 P+ 0.669 to +0.959 [+0.271, +1.651] P+ 0.9945; 2020-2025 +0.266 P+ 0.774 to +0.665 [-0.320, +1.497] P+ 0.919. Recorded unresolved (no closing ground applies to a positive read); the college-only term now leans clearly positive on accuracy. Not a serving decision; orchestrator adjudicates.
- v4 line move, ats points: all graded -0.0003 to -0.0023 [-0.0210, +0.0193] P+ 0.415; 2020-2025 -0.0060 to -0.0090 [-0.0266, +0.0070] P+ 0.145. Both stay bounded_by_control, inside the 0.0654 bound (read from the prior adjudication, not re-measured).
- v5 pooled line move: all graded -0.0785 to -0.0844 [-0.1242, -0.0477] P+ 0.0; 2020-2025 -0.0692 to -0.0832 [-0.1465, -0.0210] P+ 0.003; against the college-only term -0.0782 to -0.0821 [-0.1347, -0.0327] P+ 0.0005: all three still refuted. Accuracy: all graded +0.433 to +0.928 [-0.410, +2.417] P+ 0.903; 2020-2025 -1.397 to -0.931 [-1.807, +0.194] P+ 0.055; against college-only +0.309 to -0.031 P+ 0.472; unresolved.
- Unit 6 nested line move: all graded -0.0880 to -0.0586 [-0.0960, -0.0235] P+ 0.0; 2020-2025 -0.0855 to -0.0762 [-0.1359, -0.0237] P+ 0.001: still refuted. Accuracy +0.464 to +0.711 P+ 0.881 and -0.998 to -0.665 P+ 0.077, unresolved.
- Verdict: the pooling refutation on line movement stands on the corrected features; no closure was withdrawn. Caveat (inferred): old-to-new changes mix the sign repair with any change in the served base since September; the NFL-only twin, which the repair touches only on pick'em rows, moved 2.0 points.
- Open: an isolated before/after on today's base (revert the seven lines, rerun) if the repair's own effect is wanted; the orchestrator's read of the v3 all-graded cell; ROADMAP XLG-09 wording.

- Root adjudication 2026-11-14 18:04 ET: the v3 all-graded cell (+0.959 [+0.271, +1.651]) stays unresolved and is not the read to quote. Its 2013-19 games use the archive opening number, which sits 1.37 points from a true Tuesday opener and fades its own error (lane mkt-25-opener-revision-size.md, unit 3), exactly what an opener-error term would pick up. The clean read is 2020-25: +0.665 [-0.320, +1.497], P+ .919, below the size the NFL harness can see (MOD-36). It earns a forward log on true openers, declared before Week 11; nothing is served. The pooling refutation on line movement stands on the corrected features.
