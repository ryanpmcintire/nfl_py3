# MOD-34 Orthogonalised add-one

## Goal
Orthogonalised add-one: add only the part of a candidate feature that lies outside the span of the 88 served inputs ROADMAP row MOD-34.

## State
Added 2026-11-03 as an unmeasured backlog row; nothing run, no registry window spent. Mechanism: Add-one features mostly flip coin-flip games because most of a candidate is already in the served inputs and only reweights them. Residualising each candidate on the served inputs (fit leave-one-season-out, season by season) leaves only genuinely new information, which can be graded without the reweighting noise. Data: Served-model input matrix and the candidate columns from MOD-23 units 6-18; no new data.

## Tried
Novelty check only: Nearest: mod23-unit10r-robust-wide.md and mod23-unit10-candidates.md (wide candidate sets added raw), acc-u7-new-information.md (new-information screen by source, not by span). Differs: the added column is the span-residual and the diagnostic is the flip pattern.

## Next
First unit: For the finished MOD-23 candidates compute the share of variance outside the served-input span, and compare raw versus residualised add-one on decisive games and on flips of games inside the model indistinguishable band; report the record on decisive games before any headline.
Rules: Zero crossing closes nothing; one fitted probability selects the side; weights, cut points and alphas chosen leave-one-season-out; every cell counted as a look; unresolved results go through nfl-ats weak-signals record with a closing ground; graded at the opener on the same market and simple-model baselines.

## Open
Look count and family to be declared before any sign is seen. Nothing here closes on zero crossing; record unresolved results with nfl-ats weak-signals record.

## Unit 1 status (2026-11-03)
Family declared: mod34_residualised_addone, 5 looks (arms A_pass_rush, B_total, D_drive, P_playquality, H_hours), each residualised vs raw add-one on the served base.
Script scripts/mod34_residualised_addone.py (ruff clean, no comments); opener input read at src/nfl_ats/clv.py:2208 (at_open spread_line = tue_open_home_spread), served-base harness scripts/mod24_u1.py true_eval -> clv.opener_pick_evaluation (line 235). Base reproduces 802-701 (gate in script).
Method: per scored season s, OLS (standardised, median-imputed 88 inputs) fit on REG rows season<s, applied to all rows, then true_eval; keep season s rows.
RUNNING: launched in background, logging to artifacts/mod34_residualised_addone/run.log; outputs report_<arm>.json there. Check processes whose command line contains mod34_resid before relaunch; pass arm names as argv to run a subset.
Next: read reports, record 5 cells via nfl-ats weak-signals record (effect_units accuracy_points, unresolved_below_power), report coin-flip flip counts (band_raw vs band_resid).

## Unit 1 RESULT (done)
5 cells recorded (unresolved_below_power), batch artifacts/mod34_residualised_addone/weak_signals_batch.json. Resid vs base (base 802-701), d_acc pp (pp+): A -0 (0.451) 802-701 flips 76 (38-38; raw 75, 34-41); B -0.73 (0.087) flips 59; D -0.33 (0.307) flips 175 (raw 161); P -1.26 (0.000) flips 223 (raw 217); H -0.47 (0.043) flips 29 (raw 18). Residualising did not cut coin-flip flips (A 71/71, B 56/55, D 144/151, P 169/170, H 18/29). Log loss/Brier no better than base. Next: none from this mechanism; flips are not reweighting of existing inputs.
