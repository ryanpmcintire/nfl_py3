# joint-probability-model

## Goal

Owner directive (paraphrased): the served nine-member composition chain
(`four_overlay_composition.py`) can only leave the base model's opener
probability alone or replace it with `1 - p`; that architecture is wrong,
because a situational signal is evidence with a fitted weight inside one
probability, never a flip rule. `docs/market_updated_model.md` (MKT-16)
already did this for the late-week market move alone; this lane is the full
version -- base-model logit, all nine composition flags, and the market
move, fit together leave-one-season-out and scored against the card (M0)
and the served flip chain (M1). Done when the doc, script, registry cells,
ROADMAP row (MKT-17) and required checks are in.

## State

Done, 2026-09-14. Shipped: `docs/joint_probability_model.md`,
`scripts/joint_probability_model_eval.py`,
`artifacts/joint_probability_model/20260914T214742Z/` (summary.json,
per_game.csv, coefficients.csv). Registry family
`joint_probability_model_v1`, 19 cells, all `unresolved_below_power`
(zero resolve to a wrong sign). ROADMAP row `MKT-17` added. Nothing in
`src/` touched; the served card is unchanged.

## Tried

- Confirmed (read, `src/nfl_ats/clv.py:2086-2117`) that the opener-evaluation
  stream's `home_cover_probability_at_open` column already has a
  walk-forward home-side offset applied (401/1,537 games, 46 picks changed);
  used the rawer `_at_open_raw` column as M0 instead.
- Confirmed all nine composition flags are computable historically for
  2020-2025 from already-archived, point-in-time-safe sources (full arrest
  database back to 2000; point-in-time-walked historical MOS forecast
  archives for both weather members; trailing PBP for protection mismatch;
  schedule-only for the rest) -- none needed to be marked structurally
  unavailable.
- Confirmed (read, `four_overlay_composition.py:410-460`) the served chain's
  real semantics: each member votes independently against the model's *raw*
  probability, and the union of flagged games gets complemented once
  (`joint_or_against_raw_card_complement_once`), not a sequential chain
  where later members see earlier flips. M1 calls the seven non-held
  members' real `apply_*` functions directly and unions their flip sets;
  measured 0 mismatches against an independent signed-flag derivation of
  the same union (465/1503 games flipped either way).
- **M4 (pure recalibration): the raw model's logit needs a fitted slope of
  only 0.135-0.424 across the six LOSO folds** to match outcomes -- the
  model is overconfident by roughly 2.4x-7x, consistent with (slightly
  above) the sibling market_updated_model.md's 0.15-0.33 reading on a
  different population. Log loss/Brier improve (+0.0044 P+0.943, +0.0021
  P+0.936) with accuracy essentially unchanged (-0.067 pts, P+0.433) --
  overconfidence is a calibration defect, not mostly a sign-flip defect.
- **M3 (flags jointly fit) beats M0 on log loss and Brier with intervals
  entirely above zero** (+0.00814 [+0.00074,+0.01553] P+0.985; +0.00394
  [+0.00034,+0.00755] P+0.984), all nine flag coefficients positive in
  every fold (same sign as each member's own mechanism). **But M1 (the
  served flip chain on the same nine flags) reads an even cleaner
  improvement over M0** (log loss +0.01023 [+0.00216,+0.01844] P+0.993;
  Brier +0.00498 [+0.00104,+0.00897] P+0.993), and M3 vs M1 leans negative
  on every metric though every interval touches zero (accuracy -1.065
  [-3.098,+1.021] P+0.154). **The joint model does not beat the served
  chain on this measurement**; the clearly supported change is M4's
  recalibration, independent of whether the flags stay as flip rules.
- M2 (+ late market move, 2023-2025 only, 799 games): directionally the
  same as M3 but noisier (3 LOSO folds); market-move coefficient positive
  in all 3 folds, interval clears zero in one (2024).
- Positive control (298 games where M2 disagrees with M0): +18.90 pts
  [+16.13,+21.74] P+1.0, the ceiling for any rule agreeing with that
  disagreement set -- M2's own +0.50 sits well inside it, textbook
  `unresolved_below_power`.

## Next

- If the owner wants to act on this without waiting for the flags-vs-chain
  question to resolve further, M4's recalibration slope (~0.14-0.42x the
  raw logit) is the one change this document supports on its own, since it
  does not depend on the flags-vs-joint-model comparison coming out either
  way.
- If picked back up cold, read `docs/joint_probability_model.md` Results
  section 3 first (the decision-relevant M3/M2-vs-M1 comparison); that is
  the open question, not the calibration finding (settled direction,
  unresolved magnitude).

## Open

- Whether the owner wants M4's recalibration slope (or M2/M3's flag
  weights) considered for the served card is an owner decision; this lane
  only measures, per the research/decision split in AGENTS.md.
- Whether it is worth re-running M3 vs M1 with a recalibrated M1 (i.e. the
  served chain's own un-flipped probabilities also shrunk by M4's slope)
  to separate "the flags should be fitted, not flipped" from "the raw
  model is overconfident and M1 inherits that on every un-flipped game" --
  not built here, flagged as the natural next split.
