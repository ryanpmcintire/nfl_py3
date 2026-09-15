# four-term-probability

## Goal

Owner directive (paraphrased): `docs/joint_probability_model.md` (MKT-17)
fit all nine composition flags with separate weights and the penalty
over-shrank them; the owner's read is the served flip chain behaves like
one large shared weight, not nine small votes. Build that directly: M5 =
intercept + a*logit(model) + b*S (signed sum of the nine flags) + c*move,
no penalty, LOSO 2020-2025, scored against M0/M1/M3, plus a Best Pick
comparison against the served weekly nominator. Done when the doc, script,
registry cells, ROADMAP row (MKT-18) and required checks are in.

## State

Done, 2026-09-14. Shipped: `docs/four_term_probability.md`,
`scripts/four_term_probability_eval.py`,
`artifacts/four_term_probability/20260914T222345Z/` (summary.json,
per_game.csv, coefficients.csv, best_pick_weekly.csv). Registry family
`four_term_probability_v1`, 39 cells, all `unresolved_below_power` (zero
resolve to a wrong sign). ROADMAP row `MKT-18` added next to `MKT-17`.
Nothing in `src/` touched; the served card is unchanged.

## Tried

- Reused `joint_probability_model_eval.py`'s data loaders directly
  (`load_opener_population`, `build_base`, `add_flags`, `build_m1`,
  `loso_arm`, bootstrap helpers) by importing the module via `sys.path`,
  rather than re-deriving the population -- confirmed identical population
  (1,503 graded games, 34 pushes dropped, 799 with the market move) and
  identical M3/M1 reproduction.
- M5 fit on the full 1,503-game population with `move` imputed to 0 for the
  704 games with no market row (2020-2022) plus a `move_available` control
  indicator, rather than restricting to the 799-game market subset -- lets
  every variant share one LOSO population and one set of six folds.
- **Headline finding is a calibration ordering, not the horse race**:
  binned by each model's own out-of-season confidence (five bands,
  50-52/52-55/55-58/58-62/62+), M5's accuracy rises with its own stated
  confidence (52.0/55.9/55.4/61.9/59.7%) while M0's falls with its own
  (56.0/53.9/54.0/52.3/50.7%) -- inverted. This is the same overconfidence
  `docs/joint_probability_model.md`'s M4 found in log-loss/Brier terms,
  restated sharper in accuracy terms, and M5 does not inherit it.
- **M5 leans ahead of M1 on every metric** (log loss +0.0027 P+ 0.809,
  Brier +0.0014 P+ 0.814, accuracy +0.333 pts P+ 0.640), a materially more
  favourable read than the prior lane's M3 vs M1 (all three leaned
  *behind* M1, P+ 0.15-0.24). Neither resolves (all touch zero).
- **The model term `a` is not distinguishable from zero in any of six
  folds** (0.09 to 0.32, every interval contains zero), while the shared
  flag weight `b` clears zero in all six folds and the market-move weight
  `c` clears zero in five of six. Per rule (a) this is not grounds to drop
  `a` -- positive in six of six folds is itself evidence -- and a direct
  M5-vs-M5b (drop `a`) comparison reads +0.665 pts [-0.601, +1.926] P+
  0.848, i.e. the two are not distinguishable from each other either.
  **This document recommends keeping M5 (with the model term), not
  switching to M5b**, because preferring M5b purely for its nominally
  better record among four predeclared variants would be after-the-fact
  selection, not evidence the term is worthless.
- M5b and M5c both beat M0 and M3 on accuracy with intervals entirely above
  zero (M5b vs M0 +3.194 [+0.329,+6.053] P+ 0.985; M5b vs M3 +2.063
  [+0.133,+3.987] P+ 0.981; M5c vs M0 +2.728 [+0.131,+5.333] P+ 0.979).
  M5a (no market move) is the weakest of the four against M1 in every
  season from 2023 on -- the move is doing real work.
- Positive control (263 games where M5 disagrees with M1): +8.916 pts
  [+7.495,+10.347] P+ 1.0, the ceiling for any rule agreeing with that
  263-game set -- M5's own +0.333 sits well inside it.
- Best Pick: M5's own weekly star (unrestricted, highest M5 confidence)
  beats the old alpha=2000 nominee 61.76% vs 55.88% on 102 paired weeks,
  +5.882 pts [-8.824,+20.588] P+ 0.784. Against the ranker actually served
  since 2026-09-14 (the card's own decision-score ranker), the two are
  **indistinguishable**: both hit exactly 64/103 (62.14%), effect 0.000 pts
  P+ 0.505, and on the 95 of 103 weeks where the two name a different game,
  both still go 57-38.
- Look-count caveat stated plainly in the doc, not buried: the nine flags
  were already promoted and jointly fit on this identical 1,503-game
  population in the prior lane, and the market move on the prior
  market-move lane's population -- `S`'s apparent edge inherits whatever
  optimism those two documents already accumulated. 51 looks taken here,
  on top of 33 + 19 in the two prior lanes on largely the same games. Only
  a forward read settles whether `b`/`c` beating `a` is real.

## Next

- A forward (prospective) read of M5's coefficients against games this
  population has not yet seen is the only thing that can move the model
  term question past `unresolved_below_power`, per the look-reuse caveat
  above.
- If the owner wants to act on this without a forward read, the model-lean
  variant (M5, keeping `a`) is what this document recommends over M5b,
  per Result 4's reasoning; nothing here is switched onto the served card
  either way.

## Open

- Whether the owner wants any M5 variant considered for the served card is
  an owner decision; this lane only measures, per the research/decision
  split in AGENTS.md.
- The Best Pick read against the currently-served ranker is a coin flip on
  this population (P+ 0.505) -- not grounds to change the star, and not
  grounds to conclude the two rankers are secretly the same rule (they
  disagree on the actual game in 92% of weeks despite the identical
  record).
