# Pool hook term

## Goal
Test whether a hook term (logit discrete cover prob at the pool line minus at the Tuesday line) added to the four-term served pick fit improves it when graded at a simulated half-point pool line.

## State
Script tests/scratch/pool_hook/hook.py, output tests/scratch/pool_hook/out.txt (measured). Whole-line Tuesday opens 2020-25: n 661, push 5.1 pct, hook-side cover 52.57 pct (= 0.5+push/2). LOSO +hook vs four-term: accuracy -0.03 pts (P+ 0.47), Brier gain 0.00026 (P+ 0.997), log loss gain 0.00051 (P+ 0.996); hook coef 0.77-0.88 per fold. 2026: 39 whole-line games, hook side 22-17, card took it 22 of 39 (13-9 when took, 8-9 when not). Recorded as family pool_hook_term_2026_10_27 (2 cells).

## Tried
Pooled prior-season push mass within +-1 spread point; hook = logit(0.5(1-/+p0)); 2020 has no prior seasons so hook 0 there.

## Rebuild (measured, 2026-10-27)
rebuild.py/hook2.py/week8.py in tests/scratch/pool_hook; out2.txt, out_week8.txt. Rebuilt 34 push rows with served code paths (max diff 0.0 vs served on 1503 rows). Push model_logit was already real; 5 of 34 have a nonzero protection flag; none are in the served market-move table (move 0/0 as served). Result unchanged: accuracy +0.10 pts (P+ 0.585), Brier +0.00026 (P+ 0.997), log loss +0.00051 (P+ 0.996), hook coef 0.77-0.88. In-sample 54.62 pct vs OOS 54.81. Week 8 2026: 5 whole-number Tuesday games, mean abs hook contribution 0.085 logit, max 0.155. Recorded accuracy cell, replaced logloss cell.

## Next
Hook coef near 1 matches the mechanism (partly mechanical). Serving needs per-game pool line capture and a push-inclusive fit population (market-move table lacks pushes).

## Open
Does serving need the pool line captured per game (forward log)? 2020 prior-season p0 gap.

## Serve attempt (2026-10-27) BLOCKED by source guard
src/nfl_ats/independent_validation.py:38-49 PINNED_NAMES includes pick_probability.py, card_view.py, pick_refresh.py (read); _sources() also pins any name with model/margin or ending features.py/_overlay.py. Study combined_vs_raw_2026_w07_w18_v4 (frozen coefficients, enrollment 2026-10-20) would fail its source guard on any edit. No file edited. pick_probability_fit.py is not pinned but its FIT_FEATURES must match pick_probability.py. Decision needed: re-enroll v5 same session as the edit (per memory pinned-source rule), then implement. Design notes: hook2.py builds p0 from prior seasons within +-1 spread point (not a full discrete distribution); 2020 hook 0.

Orchestrator decision 2026-10-27: serving needs edits to pinned files (independent_validation.py PINNED_NAMES: pick_probability.py, card_view.py, pick_refresh.py), which would void validation v4 (enrolled 2026-10-20, Weeks 7-18). The hook's value is calibration (accuracy +0.10 pts), not worth losing two prospective weeks alone. Bundle it with the next servable change that needs pinned edits (likely MOD-25 v3 distillation) and enroll v5 once, in the same session as the edit.
