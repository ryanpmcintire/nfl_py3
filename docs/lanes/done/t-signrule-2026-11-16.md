# T sign-rule form, college screen (2026-11-16)

## Goal
The college sign-rule fade of the early-surprise term T read +1.73 pts against the close (P+ .97) while the linear fitted term read nothing (lane xlg-early-overreaction-cfb-2026-11-14.md). Declare one fitted sign-rule form of T before looking, screen it on college games at the opener grade, and, if it survives, freeze the NFL form and its 2027 forward-log declaration. No NFL look (NFL 2020-25 already used for T). Serves nothing.

## Declaration (written 2026-11-16 15:43:32 local, before any outcome was joined to this form)
- Script `scripts/t_signrule.py`, artifacts `artifacts/t_signrule/`, family `t_signrule_cfb`, league cfb.
- Form (one): `F = -sign(T)` on early-window games (either team brings 1 to 5 earlier lined games, the window of the xlg lane, inherited unchanged), 0 elsewhere and where T = 0. T is `surprise_values` from MOD-57 P2-Q unchanged. Entered as ONE added fitted term on the opener-input cover logit of the XLG-14 college harness (`x14.cfb_frame`, opener the only line input, graded at the opener), coefficient fit leave-one-season-out over 13 seasons (2012-19, 2021-25). No clip, no threshold, no grid: the form has no free parameter beyond its coefficient. Positive coefficient = fade (the NFL direction). The fitted probability selects the side; no rule flips a pick.
- Matched null: 200 within-season shuffles of the term (the mod36/xlg-harness-power college add-one, `M36.permuted`, seed stream after the xlg lane's three windows), real log loss and accuracy gains read as percentiles of it.
- Halves: seasons split chronologically, first 6 (2012-17) and last 7 (2018-25); out-of-season accuracy and log loss gain of the LOSO predictions inside each half, and the coefficient fit inside each half.
- Positive control: planted-size ladder on this shape (k x null coefficient SD, k = 1..10, 200 draws), smallest size seen at P+ >= .90 in 80% of draws.
- Survivor rule (carry to NFL forward log): (a) full-sample coefficient positive (fade), (b) real log loss gain at or above the 90th percentile of its matched null (`M36.NOTABLE[0]`), (c) coefficient positive in both halves. Failing it closes nothing: unresolved_below_power unless a declared closing ground is met.
- Looks: 1 form x 2 metrics (accuracy points, log loss) = 2 primary cells, plus 4 half cells (2 halves x 2 metrics) = 6 looks, family `t_signrule_cfb`. Controls: intercept-only and base read from the same harness run.

## State
Done 2026-11-16 15:46 (all **measured**, `artifacts/t_signrule/summary.txt`, results.json, batch.json; 119 s, exit 0). 8,925 harness games, 3,583 nonzero.
- Added term (LOSO, 13 folds): changed picks 373-368; accuracy +0.06 [-0.56, +0.73] P+ .58; log loss -0.000073 [-0.00031, +0.00013] P+ .27; in-sample log loss +0.000046, gap +0.00012; coefficient +0.031 (fade), folds +0.015 to +0.048, positive in 13 of 13.
- Controls: intercept-only 50.7% / .69311; base 50.3% / .69304; term 50.4% / .69311.
- Halves: 2012-17 changed 197-196, acc +0.03 P+ .50, ll P+ .11, in-half coefficient -0.024; 2018-25 changed 176-172, acc +0.08 P+ .58, ll P+ .62, coefficient +0.072.
- Matched null (200 within-season shuffles): log loss 38.5th percentile, accuracy 12.5th, coefficient 81.0th.
- Positive control: seen in 80% of draws at +0.77 pts (accuracy) and +1.17 pts (log loss).
- Survivor rule: (a) pass, (b) fail (38.5 < 90), (c) fail (first half negative). Does not survive. Not closed: unresolved_below_power; no NFL frozen form or 2027 declaration written for this form. T's own 2027 forward log (mod57 lane) is unchanged.
- Recorded 6 cells, family `t_signrule_cfb`, league cfb, unresolved_below_power; `weak-signals pool --league nfl --effect-units accuracy_points` exit 0.
- Reading (**inferred**): the +1.73 sign-rule fade against the close does not carry to a fitted term at the opener grade; the fitted coefficient leans fade in every fold, but it is too small to beat its shuffle null. College opener fade was already only +0.80 (P+ .81).

## Tried
- One declared form only; no grid, so no clip looks.

## Next
None in this lane; move to done/ after the orchestrator verifies.

## Open
- None.
